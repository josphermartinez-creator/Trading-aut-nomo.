"""Ejecucion via MetaTrader 5: la via correcta para XAU/USD real.

MetaTrader 5 es donde de verdad se opera oro spot con apalancamiento, cortos y
un spread competitivo. Su limitacion es practica: la libreria oficial
``MetaTrader5`` solo existe para Windows, asi que en un VPS Linux hay que
recurrir a Wine o a un VPS Windows.

El modulo se importa sin problemas en cualquier plataforma; si la libreria no
esta disponible, ``connect()`` devuelve ``False`` con un mensaje claro en lugar
de reventar al importar.
"""

from __future__ import annotations

from datetime import datetime

from goldbot.config import Config
from goldbot.execution.base import (
    AccountInfo,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import UTC, now_utc

logger = get_logger(__name__)


class MT5Broker(Broker):
    """Adaptador de MetaTrader 5."""

    name = "mt5"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.symbol = config.execution.mt5_symbol
        self.magic = config.execution.mt5_magic
        self._mt5 = None
        self._connected = False
        self._symbol_info = None

    # ------------------------------------------------------------------ #
    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.error(
                "La libreria MetaTrader5 no esta disponible (solo Windows). "
                "Instala con 'pip install MetaTrader5' en Windows, o usa Wine, "
                "o cambia execution.mode a 'ccxt'/'paper'."
            )
            return False

        self._mt5 = mt5
        login = self.config.mt5_login
        password = self.config.mt5_password
        server = self.config.mt5_server

        if login and password and server:
            initialized = mt5.initialize(login=login, password=password, server=server)
        else:
            # Se conecta al terminal ya abierto y con sesion iniciada.
            initialized = mt5.initialize()

        if not initialized:
            logger.error("mt5.initialize() fallo: %s", mt5.last_error())
            return False

        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error("El simbolo %s no existe en este broker", self.symbol)
            mt5.shutdown()
            return False

        if not info.visible and not mt5.symbol_select(self.symbol, True):
            logger.error("No se pudo activar el simbolo %s en Market Watch", self.symbol)
            mt5.shutdown()
            return False

        self._symbol_info = mt5.symbol_info(self.symbol)
        self._connected = True

        account = mt5.account_info()
        logger.info(
            "Conectado a MT5 | cuenta %s | servidor %s | saldo %.2f %s",
            getattr(account, "login", "?"),
            getattr(account, "server", "?"),
            getattr(account, "balance", 0.0),
            getattr(account, "currency", "USD"),
        )
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
        self._connected = False

    # ------------------------------------------------------------------ #
    def get_account(self) -> AccountInfo:
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() fallo: {self._mt5.last_error()}")
        return AccountInfo(
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            free_margin=float(info.margin_free),
            currency=info.currency,
            leverage=int(info.leverage),
        )

    def get_price(self, symbol: str) -> tuple[float, float]:
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) fallo: {self._mt5.last_error()}")
        return float(tick.bid), float(tick.ask)

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        raw = self._mt5.positions_get(symbol=symbol or self.symbol)
        if raw is None:
            return []

        positions: list[Position] = []
        for item in raw:
            # Solo las posiciones abiertas por este bot (identificadas por magic).
            if item.magic != self.magic:
                continue
            positions.append(
                Position(
                    symbol=item.symbol,
                    side=OrderSide.BUY if item.type == self._mt5.POSITION_TYPE_BUY else OrderSide.SELL,
                    volume=float(item.volume),
                    entry_price=float(item.price_open),
                    opened_at=datetime.fromtimestamp(item.time, tz=UTC),
                    stop_loss=float(item.sl) or None,
                    take_profit=float(item.tp) or None,
                    position_id=str(item.ticket),
                    unrealized_pnl=float(item.profit),
                )
            )
        return positions

    # ------------------------------------------------------------------ #
    def place_order(self, order: Order) -> Order:
        order.created_at = now_utc()

        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error = "MT5 no conectado"
            return order

        if self.config.execution.dry_run:
            order.status = OrderStatus.REJECTED
            order.error = "dry_run activo: orden no enviada"
            logger.warning("[DRY RUN] Orden NO enviada: %s", order.summary())
            return order

        mt5 = self._mt5
        try:
            bid, ask = self.get_price(order.symbol)
        except RuntimeError as exc:
            order.status = OrderStatus.REJECTED
            order.error = str(exc)
            return order

        is_buy = order.side is OrderSide.BUY
        price = ask if is_buy else bid
        volume = self._normalize_volume(order.volume)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": max(1, int(self.config.execution.max_slippage_pct * price * 10)),
            "magic": self.magic,
            "comment": (order.comment or "goldbot")[:31],  # MT5 limita el comentario
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }
        if order.stop_loss:
            request["sl"] = self._normalize_price(order.stop_loss)
        if order.take_profit:
            request["tp"] = self._normalize_price(order.take_profit)

        result = mt5.order_send(request)

        if result is None:
            order.status = OrderStatus.REJECTED
            order.error = f"order_send devolvio None: {mt5.last_error()}"
            logger.error("Orden rechazada: %s", order.error)
            return order

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            order.status = OrderStatus.REJECTED
            order.error = f"retcode={result.retcode} ({result.comment})"
            logger.error("Orden rechazada: %s", order.error)
            return order

        order.order_id = str(result.order)
        order.status = OrderStatus.FILLED
        order.filled_price = float(result.price)
        order.filled_volume = float(result.volume)
        order.metadata["deal"] = result.deal

        logger.info("Orden ejecutada en MT5: %s", order.summary())
        return order

    def close_position(self, position: Position, reason: str = "") -> Order:
        order = Order(
            side=position.side.opposite,
            volume=position.volume,
            symbol=position.symbol,
            comment=reason or "cierre",
        )
        order.created_at = now_utc()

        if self.config.execution.dry_run:
            order.status = OrderStatus.REJECTED
            order.error = "dry_run activo"
            return order

        mt5 = self._mt5
        bid, ask = self.get_price(position.symbol)
        is_buy_to_close = position.side is OrderSide.SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy_to_close else mt5.ORDER_TYPE_SELL,
            "position": int(position.position_id),
            "price": ask if is_buy_to_close else bid,
            "deviation": 20,
            "magic": self.magic,
            "comment": (reason or "cierre")[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            order.status = OrderStatus.REJECTED
            order.error = f"cierre fallido: {getattr(result, 'comment', mt5.last_error())}"
            logger.error("%s", order.error)
            return order

        order.order_id = str(result.order)
        order.status = OrderStatus.FILLED
        order.filled_price = float(result.price)
        order.filled_volume = float(result.volume)
        logger.info("Posicion cerrada en MT5: %s (%s)", position.symbol, reason)
        return order

    def modify_position(
        self, position: Position, stop_loss: float | None = None, take_profit: float | None = None
    ) -> bool:
        """Mueve SL/TP: es como se implementa el trailing stop en vivo."""
        if self.config.execution.dry_run or not self._connected:
            return False

        mt5 = self._mt5
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": int(position.position_id),
            "sl": self._normalize_price(stop_loss) if stop_loss else (position.stop_loss or 0.0),
            "tp": self._normalize_price(take_profit) if take_profit else (position.take_profit or 0.0),
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            logger.warning("No se pudo modificar SL/TP: %s", getattr(result, "comment", "?"))
        return ok

    # ------------------------------------------------------------------ #
    def _normalize_volume(self, volume: float) -> float:
        """Ajusta el volumen al paso y a los limites del simbolo."""
        info = self._symbol_info
        if info is None:
            return round(volume, 2)
        step = info.volume_step or 0.01
        volume = max(info.volume_min, min(info.volume_max, volume))
        return round(round(volume / step) * step, 8)

    def _normalize_price(self, price: float) -> float:
        info = self._symbol_info
        digits = info.digits if info is not None else 2
        return round(price, digits)

    def _filling_mode(self):
        """Modo de ejecucion aceptado por el broker.

        No todos los brokers admiten los mismos modos; elegir uno no soportado
        provoca el rechazo de todas las ordenes con un error poco descriptivo.
        """
        mt5 = self._mt5
        info = self._symbol_info
        if info is None:
            return mt5.ORDER_FILLING_IOC

        mode = info.filling_mode
        if mode & 1:
            return mt5.ORDER_FILLING_FOK
        if mode & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def health_check(self) -> tuple[bool, str]:
        if not self._connected:
            return False, "no conectado"
        try:
            terminal = self._mt5.terminal_info()
            if terminal is None or not terminal.trade_allowed:
                return False, "el terminal no permite operar (revisa AutoTrading)"
            bid, ask = self.get_price(self.symbol)
            return True, f"ok (bid {bid:.2f} ask {ask:.2f})"
        except Exception as exc:
            return False, f"health check fallido: {exc}"
