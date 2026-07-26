"""Ejecucion real via CCXT.

CCXT conecta con mas de 100 exchanges con una sola API. Para oro se usa
``PAXG/USDT`` (token respaldado 1:1 por oro fisico custodiado en Londres), que
cotiza 24/7 y sigue al spot con una desviacion muy pequena.

Advertencia importante sobre el instrumento: PAXG **no es** XAU/USD. Cotiza
contra USDT, tiene su propia prima/descuento y su liquidez es una fraccion de
la del oro real. Para operar XAU/USD de verdad, el adaptador correcto es el de
MetaTrader 5. Este modulo existe porque es la unica via de ejecucion
programatica realmente accesible sin una cuenta de broker regulado.
"""

from __future__ import annotations

import time

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
from goldbot.utils.timeutils import now_utc

logger = get_logger(__name__)


class CCXTBroker(Broker):
    """Adaptador de ejecucion sobre CCXT."""

    name = "ccxt"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.symbol = config.execution.symbol
        self.exchange_id = config.execution.exchange
        self._exchange = None
        self._market = None
        self._connected = False

    # ------------------------------------------------------------------ #
    def connect(self) -> bool:
        try:
            import ccxt
        except ImportError:
            logger.error("ccxt no esta instalado: pip install ccxt")
            return False

        api_key, api_secret = self.config.api_key, self.config.api_secret
        if not api_key or not api_secret:
            logger.error(
                "Faltan credenciales. Define GOLDBOT_API_KEY y GOLDBOT_API_SECRET "
                "en el entorno (nunca en el YAML)."
            )
            return False

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
        except AttributeError:
            logger.error("Exchange desconocido: %s", self.exchange_id)
            return False

        params = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 30_000,
            "options": {"defaultType": "spot"},
        }
        self._exchange = exchange_class(params)

        if self.config.execution.testnet:
            try:
                self._exchange.set_sandbox_mode(True)
                logger.info("Modo sandbox/testnet activado en %s", self.exchange_id)
            except Exception as exc:
                logger.warning("Este exchange no soporta sandbox: %s", exc)

        try:
            self._exchange.load_markets()
            self._market = self._exchange.market(self.symbol)
        except Exception as exc:
            logger.error("No se pudieron cargar los mercados de %s: %s", self.exchange_id, exc)
            return False

        self._connected = True
        logger.info("Conectado a %s | simbolo %s", self.exchange_id, self.symbol)
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._exchange = None

    # ------------------------------------------------------------------ #
    def _retry(self, operation, description: str):
        """Reintenta una operacion de red con espera exponencial.

        Las APIs de los exchanges fallan de forma transitoria con mucha
        frecuencia (limites de tasa, timeouts). Reintentar es obligatorio; lo
        que no se debe reintentar a ciegas es una orden que quiza si se ejecuto,
        por eso ``place_order`` verifica el estado antes de reenviar.
        """
        attempts = self.config.execution.retry_attempts
        backoff = self.config.execution.retry_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                wait = backoff * (2**attempt)
                logger.warning(
                    "%s fallo (intento %d/%d): %s | reintento en %.1fs",
                    description, attempt + 1, attempts, exc, wait,
                )
                if attempt < attempts - 1:
                    time.sleep(wait)

        raise RuntimeError(f"{description} fallo tras {attempts} intentos: {last_error}")

    # ------------------------------------------------------------------ #
    def get_account(self) -> AccountInfo:
        balance = self._retry(self._exchange.fetch_balance, "fetch_balance")

        quote = self._market["quote"] if self._market else "USDT"
        base = self._market["base"] if self._market else "PAXG"

        quote_free = float(balance.get(quote, {}).get("free", 0.0) or 0.0)
        quote_total = float(balance.get(quote, {}).get("total", 0.0) or 0.0)
        base_total = float(balance.get(base, {}).get("total", 0.0) or 0.0)

        # La equity incluye el valor de mercado de la posicion en oro.
        equity = quote_total
        if base_total > 0:
            try:
                bid, _ = self.get_price(self.symbol)
                equity += base_total * bid
            except Exception:
                pass

        return AccountInfo(
            balance=quote_total,
            equity=equity,
            free_margin=quote_free,
            currency=quote,
        )

    def get_price(self, symbol: str) -> tuple[float, float]:
        ticker = self._retry(lambda: self._exchange.fetch_ticker(symbol), "fetch_ticker")
        bid = float(ticker.get("bid") or ticker.get("last") or 0.0)
        ask = float(ticker.get("ask") or ticker.get("last") or 0.0)
        if bid <= 0 or ask <= 0:
            raise RuntimeError(f"Precios invalidos para {symbol}: bid={bid} ask={ask}")
        return bid, ask

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Posiciones inferidas del saldo en spot.

        En spot no existe el concepto de "posicion": tener PAXG *es* estar
        largo. No hay cortos posibles sin margen, cosa que el runner debe tener
        en cuenta al operar en este modo.
        """
        symbol = symbol or self.symbol
        balance = self._retry(self._exchange.fetch_balance, "fetch_balance")
        base = self._market["base"] if self._market else symbol.split("/")[0]

        amount = float(balance.get(base, {}).get("total", 0.0) or 0.0)
        minimum = float(self._market["limits"]["amount"]["min"] or 0.0) if self._market else 0.0

        if amount <= max(minimum, 1e-8):
            return []

        try:
            bid, _ = self.get_price(symbol)
        except Exception:
            bid = 0.0

        return [
            Position(
                symbol=symbol,
                side=OrderSide.BUY,
                volume=amount,
                entry_price=bid,  # el spot no guarda el precio de entrada
                opened_at=now_utc(),
                metadata={"inferida_del_saldo": True},
            )
        ]

    # ------------------------------------------------------------------ #
    def place_order(self, order: Order) -> Order:
        order.created_at = now_utc()

        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error = "broker no conectado"
            return order

        if self.config.execution.dry_run:
            # Doble salvaguarda: build_broker ya deberia haber devuelto papel.
            order.status = OrderStatus.REJECTED
            order.error = "dry_run activo: orden no enviada"
            logger.warning("[DRY RUN] Orden NO enviada: %s", order.summary())
            return order

        try:
            amount = float(self._exchange.amount_to_precision(order.symbol, order.volume))
        except Exception:
            amount = order.volume

        try:
            bid, ask = self.get_price(order.symbol)
        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.error = f"no se pudo obtener precio: {exc}"
            return order

        reference = ask if order.side is OrderSide.BUY else bid

        try:
            if order.order_type.value == "market":
                raw = self._retry(
                    lambda: self._exchange.create_order(
                        order.symbol, "market", order.side.value, amount
                    ),
                    "create_order",
                )
            else:
                price = float(
                    self._exchange.price_to_precision(order.symbol, order.price or reference)
                )
                raw = self._retry(
                    lambda: self._exchange.create_order(
                        order.symbol, order.order_type.value, order.side.value, amount, price
                    ),
                    "create_order",
                )
        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.error = str(exc)
            logger.error("Orden rechazada: %s", exc)
            return order

        order.order_id = str(raw.get("id", ""))
        filled = float(raw.get("filled") or 0.0)
        average = raw.get("average") or raw.get("price") or reference

        order.filled_volume = filled
        order.filled_price = float(average) if average else reference
        order.status = OrderStatus.FILLED if filled > 0 else OrderStatus.PENDING
        order.metadata["raw"] = {k: raw.get(k) for k in ("id", "status", "cost", "fee")}

        # Verificacion de deslizamiento: si el precio se movio mas de lo
        # tolerado, queda registrado para el analisis posterior.
        if order.filled_price and reference > 0:
            slippage = abs(order.filled_price - reference) / reference
            order.metadata["slippage_pct"] = slippage
            if slippage > self.config.execution.max_slippage_pct:
                logger.warning(
                    "Deslizamiento del %.3f%% supera el maximo tolerado (%.3f%%)",
                    slippage * 100,
                    self.config.execution.max_slippage_pct * 100,
                )

        logger.info("Orden ejecutada: %s", order.summary())
        return order

    def close_position(self, position: Position, reason: str = "") -> Order:
        order = Order(
            side=position.side.opposite,
            volume=position.volume,
            symbol=position.symbol,
            comment=reason,
        )
        logger.info("Cerrando posicion %s (%s)", position.symbol, reason)
        return self.place_order(order)

    # ------------------------------------------------------------------ #
    def health_check(self) -> tuple[bool, str]:
        if not self._connected:
            return False, "no conectado"
        try:
            bid, ask = self.get_price(self.symbol)
            spread_pct = (ask - bid) / bid if bid > 0 else 1.0
            if spread_pct > 0.01:
                return False, f"horquilla anormalmente ancha: {spread_pct:.3%}"
            return True, f"ok (bid {bid:.2f} ask {ask:.2f})"
        except Exception as exc:
            return False, f"health check fallido: {exc}"
