"""Dimensionamiento de posiciones y limites de riesgo.

La estrategia decide *cuando* y *hacia donde*; este modulo decide *cuanto*, que
es lo que determina si una racha mala es un mal mes o el final de la cuenta.

Reglas que no se negocian:

* Riesgo fijo por operacion como fraccion del capital actual (no del inicial).
* Kelly fraccional acotado, nunca Kelly completo: el Kelly optimo teorico
  supone conocer la distribucion real de resultados, cosa que jamas ocurre, y
  su drawdown esperado es intolerable en la practica.
* El tamano se deriva SIEMPRE de la distancia al stop. Sin stop no hay operacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from goldbot.config import Config
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionPlan:
    """Plan completo de una operacion antes de mandarla al broker."""

    direction: int
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_pct: float
    reason: str = ""
    approved: bool = True

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_risk(self) -> float:
        risk = self.stop_distance
        if risk <= 0:
            return 0.0
        return abs(self.take_profit - self.entry_price) / risk

    def summary(self) -> str:
        side = "COMPRA" if self.direction > 0 else "VENTA"
        if not self.approved:
            return f"RECHAZADA ({self.reason})"
        return (
            f"{side} {self.lots:.2f} lotes @ {self.entry_price:.2f} | "
            f"SL {self.stop_loss:.2f} TP {self.take_profit:.2f} | "
            f"riesgo {self.risk_amount:.2f} USD ({self.risk_pct:.2%}) R:R {self.reward_risk:.2f}"
        )


class RiskManager:
    """Calcula tamanos y hace cumplir los limites diarios."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.risk = config.risk
        self.contract_size = config.costs.contract_size

        # Estado del dia en curso.
        self._current_day: date | None = None
        self._day_start_equity: float = config.risk.initial_balance
        self._trades_today: int = 0
        self._realized_pnl_today: float = 0.0

        # Historial reciente, para el ajuste por Kelly.
        self._recent_results: list[float] = []
        self._peak_equity: float = config.risk.initial_balance

    # ------------------------------------------------------------------ #
    def new_day(self, today: date, equity: float) -> None:
        """Reinicia los contadores diarios."""
        if self._current_day == today:
            return
        self._current_day = today
        self._day_start_equity = equity
        self._trades_today = 0
        self._realized_pnl_today = 0.0
        logger.info("Nuevo dia de trading %s | equity inicial %.2f", today, equity)

    def register_fill(self, pnl: float, equity: float) -> None:
        """Registra el resultado de una operacion cerrada."""
        self._realized_pnl_today += pnl
        self._recent_results.append(pnl)
        # Ventana movil: la ventaja de hace un ano no informa sobre el tamano de hoy.
        if len(self._recent_results) > 100:
            self._recent_results.pop(0)
        self._peak_equity = max(self._peak_equity, equity)

    def register_entry(self) -> None:
        self._trades_today += 1

    # ------------------------------------------------------------------ #
    def daily_loss_pct(self, equity: float) -> float:
        if self._day_start_equity <= 0:
            return 0.0
        return max(0.0, (self._day_start_equity - equity) / self._day_start_equity)

    def drawdown_pct(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity)

    def can_trade(self, equity: float) -> tuple[bool, str]:
        """Comprueba los limites antes de abrir. Devuelve (permitido, motivo)."""
        if equity <= 0:
            return False, "capital agotado"

        drawdown = self.drawdown_pct(equity)
        if drawdown >= self.risk.max_drawdown_pct:
            return False, f"drawdown maximo alcanzado ({drawdown:.1%})"

        daily_loss = self.daily_loss_pct(equity)
        if daily_loss >= self.risk.max_daily_loss_pct:
            return False, f"limite de perdida diaria alcanzado ({daily_loss:.1%})"

        if self._trades_today >= self.risk.max_trades_per_day:
            return False, f"limite de operaciones diarias ({self._trades_today})"

        return True, "ok"

    # ------------------------------------------------------------------ #
    def build_plan(
        self,
        direction: int,
        entry_price: float,
        atr: float,
        equity: float,
        stop_atr: float | None = None,
        target_atr: float | None = None,
        confidence: float = 1.0,
    ) -> PositionPlan:
        """Construye el plan de la operacion, o lo rechaza motivadamente."""
        stop_multiple = stop_atr if stop_atr is not None else self.risk.atr_stop_multiplier
        target_multiple = target_atr if target_atr is not None else self.risk.atr_target_multiplier

        rejected = PositionPlan(
            direction=direction, lots=0.0, entry_price=entry_price,
            stop_loss=0.0, take_profit=0.0, risk_amount=0.0, risk_pct=0.0, approved=False,
        )

        allowed, reason = self.can_trade(equity)
        if not allowed:
            rejected.reason = reason
            return rejected

        if direction == 0:
            rejected.reason = "sin direccion"
            return rejected

        if not np.isfinite(atr) or atr <= 0:
            rejected.reason = "ATR no valido"
            return rejected

        if not np.isfinite(entry_price) or entry_price <= 0:
            rejected.reason = "precio de entrada no valido"
            return rejected

        stop_distance = stop_multiple * atr
        stop_loss = entry_price - direction * stop_distance
        take_profit = entry_price + direction * target_multiple * atr

        # --- fraccion de riesgo --- #
        risk_fraction = self.risk.risk_per_trade
        if self.risk.use_kelly:
            risk_fraction *= self._kelly_scale()

        # La confianza del modelo de ML reduce el tamano, nunca lo aumenta.
        risk_fraction *= float(np.clip(confidence, 0.0, 1.0))
        risk_fraction = float(np.clip(risk_fraction, 0.0, self.risk.max_risk_per_trade))

        if risk_fraction <= 0:
            rejected.reason = "fraccion de riesgo nula"
            return rejected

        risk_amount = equity * risk_fraction
        raw_lots = risk_amount / (stop_distance * self.contract_size)

        # Redondeo hacia abajo al paso del broker: nunca arriesgar de mas.
        lots = np.floor(raw_lots / self.risk.lot_step) * self.risk.lot_step
        lots = float(np.clip(lots, 0.0, self.risk.max_lot_size))

        if lots < self.risk.min_lot_size:
            rejected.reason = (
                f"tamano calculado {lots:.4f} por debajo del minimo {self.risk.min_lot_size}"
            )
            return rejected

        # Riesgo efectivo tras el redondeo (puede diferir del objetivo).
        effective_risk = lots * stop_distance * self.contract_size

        return PositionPlan(
            direction=direction,
            lots=lots,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=effective_risk,
            risk_pct=effective_risk / equity if equity > 0 else 0.0,
            reason="ok",
            approved=True,
        )

    # ------------------------------------------------------------------ #
    def _kelly_scale(self) -> float:
        """Factor de escala segun el criterio de Kelly fraccional.

        Se calcula sobre las operaciones recientes y se acota a [0.25, 1.5]:
        el objetivo es reducir tamano en malas rachas mas que aumentarlo en las
        buenas, porque el Kelly estimado sobre pocas muestras es muy optimista.
        """
        results = self._recent_results
        if len(results) < 20:
            return 1.0  # sin muestra suficiente, tamano nominal

        wins = [r for r in results if r > 0]
        losses = [r for r in results if r < 0]

        if not wins or not losses:
            return 1.0

        win_rate = len(wins) / len(results)
        avg_win = float(np.mean(wins))
        avg_loss = abs(float(np.mean(losses)))

        if avg_loss <= 0:
            return 1.0

        payoff = avg_win / avg_loss
        # Kelly: f = p - (1-p)/b
        kelly = win_rate - (1 - win_rate) / payoff

        if kelly <= 0:
            # Ventaja negativa en la muestra reciente: al minimo permitido.
            return 0.25

        scaled = kelly * self.risk.kelly_fraction
        # Se normaliza contra el riesgo nominal para obtener un multiplicador.
        return float(np.clip(scaled / self.risk.risk_per_trade, 0.25, 1.5))

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Estado actual, para logs y panel de control."""
        return {
            "day": str(self._current_day) if self._current_day else None,
            "trades_today": self._trades_today,
            "realized_pnl_today": self._realized_pnl_today,
            "day_start_equity": self._day_start_equity,
            "peak_equity": self._peak_equity,
            "kelly_scale": self._kelly_scale() if self.risk.use_kelly else 1.0,
            "recent_samples": len(self._recent_results),
        }
