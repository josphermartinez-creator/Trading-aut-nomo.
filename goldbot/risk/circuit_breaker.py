"""Cortacircuitos: la ultima linea de defensa.

Un bot autonomo puede fallar de formas que ninguna funcion de fitness anticipa:
el broker devuelve precios corruptos, la conexion se cae a media operacion, la
estrategia entra en un bucle de entradas y salidas, o el mercado abre con un
hueco del 5% por una noticia geopolitica.

Este modulo no intenta ser listo. Cuenta, compara con umbrales y desconecta.
Deliberadamente aburrido: la logica que debe funcionar cuando todo lo demas
falla tiene que ser trivial de auditar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from goldbot.config import Config
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import now_utc

logger = get_logger(__name__)


class CircuitState(str, Enum):
    """Estados posibles del cortacircuitos."""

    CLOSED = "closed"      # operativa normal
    HALF_OPEN = "half_open"  # en prueba tras un enfriamiento
    OPEN = "open"          # bloqueado, no se opera


@dataclass
class TripEvent:
    """Registro de un disparo, para la auditoria posterior."""

    timestamp: datetime
    reason: str
    detail: str
    auto_reset_at: datetime | None = None


@dataclass
class CircuitBreaker:
    """Interruptor con enfriamiento y reentrada gradual."""

    config: Config
    state: CircuitState = CircuitState.CLOSED
    trips: list[TripEvent] = field(default_factory=list)

    # Contadores internos.
    _consecutive_losses: int = 0
    _consecutive_errors: int = 0
    _reset_at: datetime | None = None
    _last_reason: str = ""

    # Umbrales que no dependen de la configuracion de estrategia.
    max_consecutive_losses: int = 8
    max_consecutive_errors: int = 5
    cooldown_minutes: int = 60
    max_price_jump_pct: float = 0.05  # hueco de precio que se considera anomalo

    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        """``True`` si esta bloqueado ahora mismo (aplica el enfriamiento)."""
        if self.state is CircuitState.OPEN and self._reset_at and now_utc() >= self._reset_at:
            # Se pasa a medio abierto: se permite probar con cautela.
            self.state = CircuitState.HALF_OPEN
            logger.info("Cortacircuitos en modo prueba tras el enfriamiento")
        return self.state is CircuitState.OPEN

    @property
    def allows_trading(self) -> bool:
        return not self.is_open

    @property
    def size_multiplier(self) -> float:
        """En modo prueba se opera a tamano reducido hasta confirmar normalidad."""
        return 0.5 if self.state is CircuitState.HALF_OPEN else 1.0

    # ------------------------------------------------------------------ #
    def trip(self, reason: str, detail: str = "", cooldown_minutes: int | None = None) -> None:
        """Dispara el interruptor."""
        minutes = cooldown_minutes if cooldown_minutes is not None else self.cooldown_minutes
        reset_at = now_utc() + timedelta(minutes=minutes)

        self.state = CircuitState.OPEN
        self._reset_at = reset_at
        self._last_reason = reason
        self.trips.append(TripEvent(now_utc(), reason, detail, reset_at))

        logger.error(
            "CORTACIRCUITOS DISPARADO [%s] %s | reanudacion prevista: %s",
            reason,
            detail,
            reset_at.strftime("%Y-%m-%d %H:%M UTC"),
        )

    def reset(self, reason: str = "manual") -> None:
        """Vuelve a la operativa normal."""
        self.state = CircuitState.CLOSED
        self._reset_at = None
        self._consecutive_losses = 0
        self._consecutive_errors = 0
        logger.info("Cortacircuitos rearmado (%s)", reason)

    # ------------------------------------------------------------------ #
    # Comprobaciones
    # ------------------------------------------------------------------ #
    def check_equity(self, equity: float, peak_equity: float, day_start_equity: float) -> bool:
        """Limites de drawdown y de perdida diaria. Devuelve ``True`` si todo va bien."""
        risk = self.config.risk

        if equity <= 0:
            self.trip("capital_agotado", f"equity={equity:.2f}", cooldown_minutes=100_000)
            return False

        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown >= risk.max_drawdown_pct:
                # Un drawdown maximo no se reintenta en una hora: exige revision
                # humana o un ciclo completo de redescubrimiento.
                self.trip(
                    "drawdown_maximo",
                    f"{drawdown:.2%} >= {risk.max_drawdown_pct:.2%}",
                    cooldown_minutes=24 * 60,
                )
                return False

        if day_start_equity > 0:
            daily_loss = (day_start_equity - equity) / day_start_equity
            if daily_loss >= risk.max_daily_loss_pct:
                # Hasta el dia siguiente: el limite diario es diario.
                self.trip(
                    "perdida_diaria",
                    f"{daily_loss:.2%} >= {risk.max_daily_loss_pct:.2%}",
                    cooldown_minutes=self._minutes_until_next_session(),
                )
                return False

        return True

    def check_trade_result(self, pnl: float) -> bool:
        """Vigila las rachas de perdidas consecutivas."""
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self.trip(
                    "racha_perdedora",
                    f"{self._consecutive_losses} perdidas consecutivas",
                )
                return False
        else:
            self._consecutive_losses = 0
            if self.state is CircuitState.HALF_OPEN:
                # Una operacion ganadora en modo prueba confirma la normalidad.
                self.reset("operacion ganadora en modo prueba")
        return True

    def check_error(self, error: Exception | str) -> bool:
        """Vigila errores repetidos del broker o del feed."""
        self._consecutive_errors += 1
        logger.warning("Error %d/%d: %s", self._consecutive_errors, self.max_consecutive_errors, error)

        if self._consecutive_errors >= self.max_consecutive_errors:
            self.trip("errores_repetidos", str(error), cooldown_minutes=15)
            return False
        return True

    def clear_errors(self) -> None:
        """Una operacion correcta limpia el contador de errores."""
        self._consecutive_errors = 0

    def check_price_sanity(self, price: float, previous_price: float | None) -> bool:
        """Detecta precios corruptos o huecos extremos.

        Operar con un precio erroneo del feed es una de las pocas formas de
        perder mucho dinero muy deprisa, asi que ante la duda se para.
        """
        if price <= 0 or not _is_finite(price):
            self.trip("precio_invalido", f"precio={price}", cooldown_minutes=10)
            return False

        if previous_price and previous_price > 0:
            jump = abs(price - previous_price) / previous_price
            if jump > self.max_price_jump_pct:
                self.trip(
                    "hueco_de_precio",
                    f"salto del {jump:.2%} ({previous_price:.2f} -> {price:.2f})",
                    cooldown_minutes=30,
                )
                return False

        return True

    def check_data_freshness(self, last_bar_time: datetime, max_delay_minutes: int = 15) -> bool:
        """Comprueba que el feed no se haya congelado."""
        age = (now_utc() - last_bar_time).total_seconds() / 60
        if age > max_delay_minutes:
            self.trip(
                "datos_obsoletos",
                f"la ultima vela tiene {age:.0f} minutos",
                cooldown_minutes=10,
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    def _minutes_until_next_session(self) -> int:
        """Minutos hasta la medianoche UTC (reinicio del limite diario)."""
        now = now_utc()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, int((tomorrow - now).total_seconds() / 60))

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "allows_trading": self.allows_trading,
            "size_multiplier": self.size_multiplier,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_errors": self._consecutive_errors,
            "last_reason": self._last_reason,
            "reset_at": self._reset_at.isoformat() if self._reset_at else None,
            "total_trips": len(self.trips),
        }


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(value)
