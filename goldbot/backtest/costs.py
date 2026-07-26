"""Modelo de costes de transaccion.

La mayoria de estrategias de M5 que parecen rentables dejan de serlo al aplicar
costes realistas: en 5 minutos el movimiento esperado del oro es del orden del
spread. Por eso el modelo es deliberadamente pesimista.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goldbot.config import CostConfig


@dataclass
class CostModel:
    """Spread, comision, deslizamiento y swaps.

    Todos los precios se expresan en USD por onza; el tamano en lotes
    estandar (100 onzas por lote, por defecto).
    """

    spread_points: float = 0.25
    commission_per_lot: float = 7.0
    slippage_points: float = 0.10
    contract_size: float = 100.0
    swap_long: float = -0.5
    swap_short: float = 0.2
    # El spread se ensancha cuando la volatilidad se dispara (y en el rollover).
    volatility_spread_multiplier: float = 1.5

    @classmethod
    def from_config(cls, cfg: CostConfig) -> CostModel:
        return cls(
            spread_points=cfg.spread_points,
            commission_per_lot=cfg.commission_per_lot,
            slippage_points=cfg.slippage_points,
            contract_size=cfg.contract_size,
            swap_long=cfg.swap_long,
            swap_short=cfg.swap_short,
        )

    # ------------------------------------------------------------------ #
    def entry_price(self, mid_price: float, direction: int, volatility_factor: float = 1.0) -> float:
        """Precio de ejecucion de entrada, ya penalizado.

        Compramos en el ask y vendemos en el bid, y siempre en nuestra contra.
        """
        half_spread = 0.5 * self.spread_points * self._spread_factor(volatility_factor)
        slip = self.slippage_points * self._spread_factor(volatility_factor)
        return mid_price + direction * (half_spread + slip)

    def exit_price(self, mid_price: float, direction: int, volatility_factor: float = 1.0) -> float:
        """Precio de ejecucion de salida (cruza el spread en sentido contrario)."""
        half_spread = 0.5 * self.spread_points * self._spread_factor(volatility_factor)
        slip = self.slippage_points * self._spread_factor(volatility_factor)
        return mid_price - direction * (half_spread + slip)

    def _spread_factor(self, volatility_factor: float) -> float:
        """El spread crece con la volatilidad, pero de forma acotada."""
        if volatility_factor <= 1.0:
            return 1.0
        return float(min(self.volatility_spread_multiplier, 1.0 + 0.5 * (volatility_factor - 1.0)))

    def commission(self, lots: float) -> float:
        """Comision de ida y vuelta en USD."""
        return abs(lots) * self.commission_per_lot

    def swap(self, lots: float, direction: int, nights: int) -> float:
        """Coste de financiacion por mantener la posicion abierta de un dia a otro."""
        if nights <= 0:
            return 0.0
        rate = self.swap_long if direction > 0 else self.swap_short
        return abs(lots) * rate * nights

    def round_trip_cost_points(self, volatility_factor: float = 1.0) -> float:
        """Coste total de ida y vuelta expresado en USD/onza.

        Es la barrera que cualquier estrategia debe superar para ganar dinero;
        el motor evolutivo la usa para descartar genomas sin recorrido.
        """
        factor = self._spread_factor(volatility_factor)
        spread_cost = self.spread_points * factor
        slippage_cost = 2 * self.slippage_points * factor
        commission_points = self.commission_per_lot / self.contract_size
        return spread_cost + slippage_cost + commission_points

    def pnl(
        self,
        entry: float,
        exit_: float,
        lots: float,
        direction: int,
        nights: int = 0,
    ) -> float:
        """P&L neto en USD de una operacion cerrada."""
        gross = (exit_ - entry) * direction * lots * self.contract_size
        return gross - self.commission(lots) + self.swap(lots, direction, nights)


def estimate_volatility_factor(atr_pct: np.ndarray, baseline_window: int = 500) -> np.ndarray:
    """Factor de ensanchamiento del spread a partir del ATR relativo.

    Se compara el ATR actual con su mediana movil; si el mercado esta el doble
    de agitado que de costumbre, el spread se ensancha en consecuencia.
    """
    import pandas as pd

    series = pd.Series(atr_pct)
    baseline = series.rolling(baseline_window, min_periods=50).median()
    factor = (series / baseline.replace(0, np.nan)).fillna(1.0)
    return factor.clip(lower=0.8, upper=3.0).to_numpy(dtype="float64")
