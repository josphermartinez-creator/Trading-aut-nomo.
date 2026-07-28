"""Filtro de tendencia: la prohibicion de operar en contra.

Este modulo existe porque "nunca entrar contra la tendencia" no puede ser una
sugerencia dentro del genoma. Si fuese una condicion mas del arbol genetico, el
algoritmo la mutaria o la eliminaria en cuanto encontrase un tramo del historico
donde ir a contracorriente pagaba mejor -- y lo encontraria, porque en dos anos
de datos siempre hay un mes que premia lo contrario de lo sensato.

La direccion de tendencia se calcula **fuera** del genoma, se inyecta en la
matriz de features con un nombre reservado y se aplica como mascara final en
:meth:`StrategyGenome.generate_signals`. El genetico no puede verla en su
catalogo ni construir condiciones sobre ella: solo puede operar dentro de lo que
el filtro permite.

Metodos disponibles (``method`` en la configuracion):

* ``ema_stack``  -- las EMAs rapida/media/lenta ordenadas. La definicion mas
  comun y la mas facil de auditar a ojo en un grafico.
* ``ema_slope``  -- pendiente de la EMA lenta. Tolera solapamientos pero exige
  que la media larga apunte en la direccion correcta.
* ``structure``  -- estructura de mercado SMC (BOS/CHoCH). Reacciona antes que
  las medias pero es mas ruidosa.
* ``combined``   -- exige acuerdo entre los anteriores. Menos operaciones, pero
  la direccion es mucho mas fiable. Es el valor por defecto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goldbot.features import indicators as ta
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)

# Nombre reservado de la columna. El guion bajo inicial marca que es de uso
# interno: `FeatureBuilder` la excluye del catalogo para que el genetico no
# pueda construir condiciones sobre ella ni, por tanto, esquivarla.
TREND_COLUMN = "_trend_direction"


@dataclass
class TrendFilter:
    """Calcula la direccion de tendencia permitida en cada barra."""

    enabled: bool = True
    method: str = "combined"
    fast_ema: int = 50            # ~4 horas en M5
    mid_ema: int = 200            # ~17 horas
    slow_ema: int = 576           # 2 dias de negociacion
    min_slope: float = 5e-6       # pendiente minima para considerar que hay tendencia
    allow_flat: bool = False      # si True, opera tambien sin tendencia clara
    swing_left: int = 3
    swing_right: int = 3

    # ------------------------------------------------------------------ #
    def direction(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Serie con +1 (solo largos), -1 (solo cortos) o 0 (no operar).

        Devolver 0 es una decision deliberada: en un mercado lateral la mejor
        operacion suele ser ninguna, y el oro pasa buena parte de la sesion
        asiatica exactamente asi.
        """
        if not self.enabled:
            return pd.Series(0.0, index=ohlcv.index).replace(0.0, np.nan).fillna(0.0) + 0.0

        close = ohlcv["close"]

        if self.method == "ema_stack":
            trend = self._ema_stack(close)
        elif self.method == "ema_slope":
            trend = self._ema_slope(close)
        elif self.method == "structure":
            trend = self._structure(ohlcv)
        elif self.method == "combined":
            trend = self._combined(ohlcv, close)
        else:
            raise ValueError(f"Metodo de tendencia desconocido: {self.method!r}")

        if self.allow_flat:
            # Con allow_flat las barras sin tendencia dejan pasar ambos lados.
            # Se representa con NaN y el genoma lo interpreta como "sin veto".
            trend = trend.replace(0.0, np.nan)

        return trend.fillna(0.0) if not self.allow_flat else trend

    # ------------------------------------------------------------------ #
    def _ema_stack(self, close: pd.Series) -> pd.Series:
        """EMAs apiladas en orden: rapida > media > lenta = alcista."""
        fast = ta.ema(close, self.fast_ema)
        mid = ta.ema(close, self.mid_ema)
        slow = ta.ema(close, self.slow_ema)

        bullish = (fast > mid) & (mid > slow) & (close > fast)
        bearish = (fast < mid) & (mid < slow) & (close < fast)

        return _to_direction(bullish, bearish, close.index)

    def _ema_slope(self, close: pd.Series) -> pd.Series:
        """Pendiente de la EMA lenta, normalizada por precio."""
        slow = ta.ema(close, self.mid_ema)
        # Pendiente medida sobre una ventana, no vela a vela: menos ruido.
        slope = (slow - slow.shift(24)) / (24 * close)

        bullish = (slope > self.min_slope) & (close > slow)
        bearish = (slope < -self.min_slope) & (close < slow)

        return _to_direction(bullish, bearish, close.index)

    def _structure(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Estructura de mercado SMC: maximos y minimos crecientes o decrecientes."""
        from goldbot.features.smc import market_structure

        result = market_structure(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"],
            self.swing_left, self.swing_right,
        )
        return result["structure"].astype("float64")

    def _combined(self, ohlcv: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Acuerdo entre medias y estructura.

        Se exige que las EMAs y la estructura apunten al mismo lado. Reduce
        bastante el numero de oportunidades, pero las que quedan son las que un
        operador discrecional describiria como "tendencia clara", que es
        justamente lo que se pide.
        """
        stack = self._ema_stack(close)
        slope = self._ema_slope(close)
        structure = self._structure(ohlcv)

        votes = stack + slope + structure
        # 3 votos posibles; se exige mayoria estricta y sin votos en contra.
        bullish = (votes >= 2) & (stack >= 0) & (slope >= 0) & (structure >= 0)
        bearish = (votes <= -2) & (stack <= 0) & (slope <= 0) & (structure <= 0)

        return _to_direction(bullish, bearish, close.index)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        if not self.enabled:
            return "Filtro de tendencia DESACTIVADO (se permiten entradas en contra)"
        flat = "permite operar en lateral" if self.allow_flat else "no opera en lateral"
        return (
            f"Filtro de tendencia '{self.method}' "
            f"(EMAs {self.fast_ema}/{self.mid_ema}/{self.slow_ema}, {flat})"
        )

    def coverage(self, trend: pd.Series) -> dict[str, float]:
        """Reparto de barras por direccion permitida.

        Util para diagnosticar: si el filtro deja fuera el 90% del historico,
        casi ninguna estrategia reunira operaciones suficientes y conviene
        relajar el metodo antes de concluir que "no hay estrategias buenas".
        """
        total = len(trend)
        if total == 0:
            return {"alcista": 0.0, "bajista": 0.0, "sin_tendencia": 0.0}
        return {
            "alcista": float((trend > 0).sum() / total),
            "bajista": float((trend < 0).sum() / total),
            "sin_tendencia": float((trend == 0).sum() / total),
        }


def _to_direction(bullish: pd.Series, bearish: pd.Series, index: pd.Index) -> pd.Series:
    """Combina dos mascaras en una serie +1/-1/0."""
    direction = pd.Series(0.0, index=index)
    direction[bullish.fillna(False)] = 1.0
    direction[bearish.fillna(False)] = -1.0
    return direction


def apply_trend_veto(
    signals: pd.Series, trend: pd.Series | None, allow_flat: bool = False
) -> pd.Series:
    """Anula las señales que van contra la tendencia.

    Se aplica como ultimo paso de la generacion de señales, despues de todo lo
    que decida el genoma. Un ``NaN`` en ``trend`` significa "sin veto" (solo
    ocurre con ``allow_flat``); un 0 significa "lateral, no operar".
    """
    if trend is None or signals.empty:
        return signals

    aligned = trend.reindex(signals.index)
    out = signals.copy()

    # NaN = sin opinion sobre la tendencia -> se respeta la señal del genoma.
    has_opinion = aligned.notna()
    if not has_opinion.any():
        return out

    veto_long = has_opinion & (aligned <= 0) & (signals > 0)
    veto_short = has_opinion & (aligned >= 0) & (signals < 0)

    if allow_flat:
        # En lateral (0) no se veta nada: solo se prohibe ir en contra.
        veto_long = has_opinion & (aligned < 0) & (signals > 0)
        veto_short = has_opinion & (aligned > 0) & (signals < 0)

    out[veto_long | veto_short] = 0.0
    return out
