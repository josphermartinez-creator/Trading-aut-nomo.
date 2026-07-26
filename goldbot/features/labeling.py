"""Etiquetado por triple barrera (Lopez de Prado).

Etiquetar con "el retorno dentro de N barras" es el error clasico: ignora que
una operacion real muere cuando toca el stop, no cuando expira el reloj. El
metodo de triple barrera etiqueta segun *que barrera se toca primero*:

* barrera superior -> toma de beneficios,
* barrera inferior -> stop loss,
* barrera vertical -> se agota el tiempo maximo de permanencia.

Las barreras se dimensionan con el ATR, asi que se adaptan solas al regimen de
volatilidad del oro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goldbot.features import indicators as ta
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LabelResult:
    """Etiquetas y su contexto, todo alineado con el indice de entrada."""

    labels: pd.Series          # -1 / 0 / +1  (o 0/1 en meta-etiquetado)
    returns: pd.Series         # retorno realizado hasta el toque
    holding_bars: pd.Series    # barras hasta el toque
    touch_barrier: pd.Series   # 'pt' | 'sl' | 'vertical' | 'none'
    sample_weight: pd.Series   # peso por unicidad temporal

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "label": self.labels,
                "ret": self.returns,
                "bars": self.holding_bars,
                "barrier": self.touch_barrier,
                "weight": self.sample_weight,
            }
        )


def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    lookahead_bars: int = 24,
    profit_take_atr: float = 1.5,
    stop_loss_atr: float = 1.0,
    atr_period: int = 14,
    side: pd.Series | None = None,
    min_return: float = 0.0,
) -> LabelResult:
    """Etiqueta cada barra segun la primera barrera tocada.

    Parameters
    ----------
    ohlcv:
        Velas con columnas ``open/high/low/close``. La entrada se simula en la
        apertura de la barra siguiente, nunca en el cierre de la actual.
    lookahead_bars:
        Barrera vertical (maximo de barras en mercado).
    profit_take_atr, stop_loss_atr:
        Multiplos de ATR para las barreras horizontales.
    side:
        Si se pasa (+1 largo / -1 corto / 0 sin senal), se hace **meta-etiquetado**:
        la etiqueta es 1 si esa operacion concreta habria ganado y 0 si no. Es
        la forma correcta de que el ML filtre las senales del genoma en vez de
        intentar predecir el mercado por su cuenta.
    min_return:
        Retorno minimo (en tanto por uno) para considerar util una operacion;
        sirve para exigir que cubra costes.
    """
    if ohlcv.empty:
        empty = pd.Series(dtype="float64", index=ohlcv.index)
        return LabelResult(empty, empty, empty, pd.Series(dtype="object", index=ohlcv.index), empty)

    high = ohlcv["high"].to_numpy(dtype="float64")
    low = ohlcv["low"].to_numpy(dtype="float64")
    close = ohlcv["close"].to_numpy(dtype="float64")
    open_ = ohlcv["open"].to_numpy(dtype="float64")

    atr_values = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], atr_period).to_numpy(dtype="float64")

    n = len(close)
    sides = (
        np.ones(n)
        if side is None
        else side.reindex(ohlcv.index).fillna(0.0).to_numpy(dtype="float64")
    )

    labels = np.zeros(n)
    returns = np.zeros(n)
    holding = np.zeros(n, dtype="int64")
    barriers = np.full(n, "none", dtype=object)

    meta_mode = side is not None

    for i in range(n - 1):
        direction = sides[i]
        if meta_mode and direction == 0:
            continue

        atr_i = atr_values[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # Entrada realista: apertura de la barra i+1, ya conocido el cierre de i.
        entry = open_[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue

        if direction >= 0:
            pt_level = entry + profit_take_atr * atr_i
            sl_level = entry - stop_loss_atr * atr_i
        else:
            pt_level = entry - profit_take_atr * atr_i
            sl_level = entry + stop_loss_atr * atr_i

        end = min(i + 1 + lookahead_bars, n)
        touched = "vertical"
        exit_price = close[end - 1]
        bars_held = end - (i + 1)

        for j in range(i + 1, end):
            hit_pt = high[j] >= pt_level if direction >= 0 else low[j] <= pt_level
            hit_sl = low[j] <= sl_level if direction >= 0 else high[j] >= sl_level

            if hit_pt and hit_sl:
                # Ambas dentro de la misma vela: sin datos de tick no sabemos el
                # orden, asi que asumimos el stop. Preferimos infraestimar.
                touched, exit_price, bars_held = "sl", sl_level, j - i
                break
            if hit_sl:
                touched, exit_price, bars_held = "sl", sl_level, j - i
                break
            if hit_pt:
                touched, exit_price, bars_held = "pt", pt_level, j - i
                break

        raw_return = (exit_price - entry) / entry * (1.0 if direction >= 0 else -1.0)

        returns[i] = raw_return
        holding[i] = bars_held
        barriers[i] = touched

        if meta_mode:
            labels[i] = 1.0 if raw_return > min_return else 0.0
        elif touched == "pt":
            labels[i] = 1.0
        elif touched == "sl":
            labels[i] = -1.0
        else:
            labels[i] = 1.0 if raw_return > min_return else (-1.0 if raw_return < -min_return else 0.0)

    index = ohlcv.index
    holding_series = pd.Series(holding, index=index, name="bars")
    weights = _uniqueness_weights(holding_series, n)

    result = LabelResult(
        labels=pd.Series(labels, index=index, name="label"),
        returns=pd.Series(returns, index=index, name="ret"),
        holding_bars=holding_series,
        touch_barrier=pd.Series(barriers, index=index, name="barrier"),
        sample_weight=weights,
    )

    distribution = result.labels.value_counts(normalize=True).round(3).to_dict()
    logger.debug("Distribucion de etiquetas: %s", distribution)
    return result


def _uniqueness_weights(holding_bars: pd.Series, n: int) -> pd.Series:
    """Peso por unicidad: penaliza etiquetas que se solapan en el tiempo.

    Dos operaciones que comparten casi todo su periodo de vida no son dos
    observaciones independientes. Sin este peso, el modelo cree tener miles de
    muestras cuando en realidad tiene unos cientos.
    """
    bars = holding_bars.to_numpy(dtype="int64")
    concurrency = np.zeros(n, dtype="float64")

    for i in range(n):
        span = bars[i]
        if span <= 0:
            continue
        concurrency[i + 1 : min(i + 1 + span, n)] += 1.0

    concurrency = np.maximum(concurrency, 1.0)
    weights = np.zeros(n, dtype="float64")
    for i in range(n):
        span = bars[i]
        if span <= 0:
            continue
        window = concurrency[i + 1 : min(i + 1 + span, n)]
        if window.size:
            weights[i] = float((1.0 / window).mean())

    # Normalizamos a media 1 para no alterar la escala efectiva del dataset.
    positive = weights[weights > 0]
    if positive.size:
        weights = weights / positive.mean()
    return pd.Series(weights, index=holding_bars.index, name="weight")


def purged_train_test_split(
    index: pd.DatetimeIndex,
    train_ratio: float = 0.75,
    purge_bars: int = 24,
    embargo_bars: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Split temporal con purga y embargo.

    Entre train y test se elimina una franja (``purge_bars``) porque las
    etiquetas del final del train miran hacia adelante y pisarian el test.
    Ademas se aplica un ``embargo`` posterior para cortar la autocorrelacion
    residual. Sin esto, el AUC del modelo sale inflado y la estrategia parece
    mucho mejor de lo que es.
    """
    n = len(index)
    split = int(n * train_ratio)
    train_end = max(0, split - purge_bars)
    test_start = min(n, split + embargo_bars)

    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, n)
    return train_idx, test_idx


def purged_kfold_splits(
    n: int, n_splits: int = 5, purge_bars: int = 24, embargo_bars: int = 12
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-Fold temporal purgado (Combinatorial Purged CV simplificado)."""
    fold_size = n // n_splits
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for k in range(n_splits):
        test_start = k * fold_size
        test_end = n if k == n_splits - 1 else (k + 1) * fold_size
        test_idx = np.arange(test_start, test_end)

        left = np.arange(0, max(0, test_start - purge_bars))
        right = np.arange(min(n, test_end + embargo_bars), n)
        train_idx = np.concatenate([left, right])

        if train_idx.size and test_idx.size:
            splits.append((train_idx, test_idx))
    return splits
