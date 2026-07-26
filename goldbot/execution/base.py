"""Contrato comun de los brokers.

Un unico interfaz para papel, CCXT y MetaTrader 5. El runner en vivo no sabe
con cual esta hablando, lo que permite incubar una estrategia en papel y
promoverla a real sin tocar una linea del bucle de trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_direction(cls, direction: int) -> OrderSide:
        return cls.BUY if direction > 0 else cls.SELL

    @property
    def direction(self) -> int:
        return 1 if self is OrderSide.BUY else -1

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Orden enviada al broker."""

    side: OrderSide
    volume: float
    symbol: str
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str = ""
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_volume: float = 0.0
    created_at: datetime | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    def summary(self) -> str:
        if self.status is OrderStatus.REJECTED:
            return f"RECHAZADA {self.side.value} {self.volume} {self.symbol}: {self.error}"
        price = f"@{self.filled_price:.2f}" if self.filled_price else ""
        return f"{self.status.value} {self.side.value} {self.filled_volume or self.volume} {self.symbol} {price}"


@dataclass
class Position:
    """Posicion abierta."""

    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    opened_at: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    position_id: str | None = None
    unrealized_pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        return self.side.direction

    def pnl_at(self, price: float, contract_size: float = 100.0) -> float:
        return (price - self.entry_price) * self.direction * self.volume * contract_size


@dataclass
class AccountInfo:
    """Estado de la cuenta."""

    balance: float
    equity: float
    margin: float = 0.0
    free_margin: float = 0.0
    currency: str = "USD"
    leverage: int = 1

    @property
    def margin_level(self) -> float:
        return (self.equity / self.margin * 100) if self.margin > 0 else float("inf")


class Broker(ABC):
    """Interfaz de ejecucion."""

    name: str = "base"

    @abstractmethod
    def connect(self) -> bool:
        """Establece la conexion. ``True`` si tuvo exito."""

    @abstractmethod
    def disconnect(self) -> None:
        """Cierra la conexion limpiamente."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Estado actual de la cuenta."""

    @abstractmethod
    def get_price(self, symbol: str) -> tuple[float, float]:
        """Devuelve (bid, ask) actuales."""

    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Posiciones abiertas."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Envia una orden y devuelve la orden actualizada con el resultado."""

    @abstractmethod
    def close_position(self, position: Position, reason: str = "") -> Order:
        """Cierra una posicion abierta."""

    def close_all(self, symbol: str | None = None, reason: str = "cierre masivo") -> list[Order]:
        """Cierra todas las posiciones. Usado por los cortacircuitos."""
        return [self.close_position(p, reason) for p in self.get_positions(symbol)]

    def modify_position(
        self, position: Position, stop_loss: float | None = None, take_profit: float | None = None
    ) -> bool:
        """Modifica SL/TP. Por defecto no soportado."""
        return False

    def is_market_open(self, symbol: str | None = None) -> bool:
        from goldbot.utils.timeutils import market_is_open

        return market_is_open()

    def health_check(self) -> tuple[bool, str]:
        """Comprueba que el broker responde y devuelve precios coherentes."""
        try:
            account = self.get_account()
            if account.equity <= 0:
                return False, "equity no positiva"
            return True, "ok"
        except Exception as exc:
            return False, f"health check fallido: {exc}"

    def __enter__(self) -> Broker:
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()
