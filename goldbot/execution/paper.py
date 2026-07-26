"""Broker de papel: simula la ejecucion con precios reales.

Es la pieza clave de la incubadora. Una estrategia que supera el backtest pasa
aqui varios dias operando contra precios en vivo antes de tocar dinero real.
Eso detecta lo que ningun backtest ve: latencia del feed, señales que llegan
tarde, huecos de fin de semana, y sobre todo la diferencia entre el rendimiento
esperado y el observado en tiempo real.

Aplica los mismos costes que el motor de backtest, de modo que una divergencia
entre ambos apunta a un problema real y no a una diferencia de contabilidad.
"""

from __future__ import annotations

import uuid

from goldbot.backtest.costs import CostModel
from goldbot.config import Config
from goldbot.execution.base import (
    AccountInfo,
    Broker,
    Order,
    OrderStatus,
    Position,
)
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import now_utc

logger = get_logger(__name__)


class PaperBroker(Broker):
    """Simulador de ejecucion con contabilidad realista."""

    name = "paper"

    def __init__(self, config: Config, market_data=None) -> None:
        self.config = config
        self.market_data = market_data
        self.costs = CostModel.from_config(config.costs)

        self.balance = config.risk.initial_balance
        self._positions: dict[str, Position] = {}
        self._closed_orders: list[Order] = []
        self._last_price: float | None = None
        self._connected = False

    # ------------------------------------------------------------------ #
    def connect(self) -> bool:
        self._connected = True
        logger.info("Broker de papel conectado | saldo inicial %.2f USD", self.balance)
        return True

    def disconnect(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------ #
    def set_price(self, price: float) -> None:
        """Inyecta el precio actual.

        El runner en vivo llama a esto en cada vela cerrada; asi el broker de
        papel usa exactamente el mismo precio que vio la estrategia.
        """
        if price > 0:
            self._last_price = float(price)

    def get_price(self, symbol: str) -> tuple[float, float]:
        """(bid, ask) derivados del ultimo precio medio conocido."""
        mid = self._last_price
        if mid is None and self.market_data is not None:
            try:
                df = self.market_data.load()
                if not df.empty:
                    mid = float(df["close"].iloc[-1])
                    self._last_price = mid
            except Exception as exc:
                logger.warning("No se pudo obtener precio para papel: %s", exc)

        if mid is None:
            raise RuntimeError("El broker de papel no tiene precio; llama antes a set_price()")

        half_spread = 0.5 * self.costs.spread_points
        return mid - half_spread, mid + half_spread

    # ------------------------------------------------------------------ #
    def get_account(self) -> AccountInfo:
        equity = self.balance + sum(self._unrealized(p) for p in self._positions.values())
        return AccountInfo(
            balance=self.balance,
            equity=equity,
            margin=0.0,
            free_margin=equity,
            currency="USD",
        )

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        for position in positions:
            position.unrealized_pnl = self._unrealized(position)
        return positions

    def _unrealized(self, position: Position) -> float:
        if self._last_price is None:
            return 0.0
        return position.pnl_at(self._last_price, self.costs.contract_size)

    # ------------------------------------------------------------------ #
    def place_order(self, order: Order) -> Order:
        """Ejecuta la orden al instante, aplicando spread y deslizamiento."""
        order.created_at = now_utc()

        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error = "broker no conectado"
            return order

        if order.volume < self.config.risk.min_lot_size:
            order.status = OrderStatus.REJECTED
            order.error = f"volumen {order.volume} por debajo del minimo"
            return order

        try:
            bid, ask = self.get_price(order.symbol)
        except RuntimeError as exc:
            order.status = OrderStatus.REJECTED
            order.error = str(exc)
            return order

        mid = (bid + ask) / 2
        direction = order.side.direction
        fill_price = self.costs.entry_price(mid, direction)

        # Si ya hay posicion en el simbolo, la orden contraria la cierra en
        # lugar de abrir una nueva: el sistema opera con una sola posicion.
        existing = self._positions.get(order.symbol)
        if existing is not None:
            if existing.direction != direction:
                close_order = self.close_position(existing, reason="orden contraria")
                order.metadata["closed_position"] = close_order.order_id
            else:
                order.status = OrderStatus.REJECTED
                order.error = "ya existe una posicion en la misma direccion"
                return order

        order.order_id = uuid.uuid4().hex[:12]
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_volume = order.volume

        self._positions[order.symbol] = Position(
            symbol=order.symbol,
            side=order.side,
            volume=order.volume,
            entry_price=fill_price,
            opened_at=order.created_at,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            position_id=order.order_id,
            metadata={"comment": order.comment},
        )

        logger.info(
            "[PAPEL] %s %.2f %s @ %.2f | SL %.2f TP %.2f",
            order.side.value.upper(),
            order.volume,
            order.symbol,
            fill_price,
            order.stop_loss or 0.0,
            order.take_profit or 0.0,
        )
        return order

    def close_position(self, position: Position, reason: str = "") -> Order:
        """Cierra la posicion y liquida el P&L contra el saldo."""
        order = Order(
            side=position.side.opposite,
            volume=position.volume,
            symbol=position.symbol,
            comment=reason,
            created_at=now_utc(),
        )

        try:
            bid, ask = self.get_price(position.symbol)
        except RuntimeError as exc:
            order.status = OrderStatus.REJECTED
            order.error = str(exc)
            return order

        mid = (bid + ask) / 2
        fill_price = self.costs.exit_price(mid, position.direction)

        nights = max(0, (now_utc().date() - position.opened_at.date()).days)
        pnl = self.costs.pnl(
            entry=position.entry_price,
            exit_=fill_price,
            lots=position.volume,
            direction=position.direction,
            nights=nights,
        )

        self.balance += pnl
        self._positions.pop(position.symbol, None)

        order.order_id = uuid.uuid4().hex[:12]
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_volume = position.volume
        order.metadata = {
            "pnl": pnl,
            "entry_price": position.entry_price,
            "reason": reason,
            "bars_held_seconds": (now_utc() - position.opened_at).total_seconds(),
        }
        self._closed_orders.append(order)

        logger.info(
            "[PAPEL] CIERRE %s %.2f @ %.2f | P&L %+.2f USD (%s) | saldo %.2f",
            position.symbol,
            position.volume,
            fill_price,
            pnl,
            reason,
            self.balance,
        )
        return order

    def modify_position(
        self, position: Position, stop_loss: float | None = None, take_profit: float | None = None
    ) -> bool:
        current = self._positions.get(position.symbol)
        if current is None:
            return False
        if stop_loss is not None:
            current.stop_loss = stop_loss
        if take_profit is not None:
            current.take_profit = take_profit
        return True

    # ------------------------------------------------------------------ #
    def check_stops(self, high: float, low: float) -> list[Order]:
        """Comprueba SL/TP contra el rango de la vela recien cerrada.

        Como en el motor de backtest, si ambos niveles caen dentro de la misma
        vela se asume el stop: es la hipotesis pesimista y la unica prudente sin
        datos de tick.
        """
        closed: list[Order] = []
        for position in list(self._positions.values()):
            direction = position.direction

            hit_stop = position.stop_loss is not None and (
                low <= position.stop_loss if direction > 0 else high >= position.stop_loss
            )
            hit_target = position.take_profit is not None and (
                high >= position.take_profit if direction > 0 else low <= position.take_profit
            )

            if hit_stop:
                self.set_price(position.stop_loss)
                closed.append(self.close_position(position, "stop_loss"))
            elif hit_target:
                self.set_price(position.take_profit)
                closed.append(self.close_position(position, "take_profit"))

        return closed

    @property
    def closed_orders(self) -> list[Order]:
        return list(self._closed_orders)

    def reset(self) -> None:
        """Reinicia la cuenta simulada (util entre incubaciones)."""
        self.balance = self.config.risk.initial_balance
        self._positions.clear()
        self._closed_orders.clear()
        self._last_price = None
