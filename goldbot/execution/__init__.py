"""Ejecucion de ordenes: papel, CCXT y MetaTrader 5."""

from goldbot.execution.base import Broker, Order, OrderSide, OrderType, Position
from goldbot.execution.ccxt_broker import CCXTBroker
from goldbot.execution.mt5_broker import MT5Broker
from goldbot.execution.paper import PaperBroker

__all__ = [
    "Broker",
    "CCXTBroker",
    "MT5Broker",
    "Order",
    "OrderSide",
    "OrderType",
    "PaperBroker",
    "Position",
    "build_broker",
]


def build_broker(config, market_data=None):
    """Instancia el broker indicado en la configuracion.

    En modo ``dry_run`` se devuelve siempre el broker de papel, sea cual sea el
    modo configurado. Es una salvaguarda intencionada: la unica forma de mandar
    una orden real es desactivar ``dry_run`` de forma explicita.
    """
    from goldbot.utils.logging import get_logger

    logger = get_logger(__name__)
    mode = config.execution.mode

    if config.execution.dry_run and mode != "paper":
        logger.warning(
            "dry_run activo: se ignora el modo '%s' y se opera en papel. "
            "Desactiva execution.dry_run para operar de verdad.",
            mode,
        )
        return PaperBroker(config, market_data)

    if mode == "paper":
        return PaperBroker(config, market_data)
    if mode == "ccxt":
        return CCXTBroker(config)
    if mode == "mt5":
        return MT5Broker(config)
    raise ValueError(f"Modo de ejecucion desconocido: {mode}")
