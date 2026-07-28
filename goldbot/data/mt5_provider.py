"""Descarga de velas reales desde MetaTrader 5.

Este proveedor es el que cumple el requisito de "lo primero que haga al
conectarse es bajar 5.000 velas reales del broker". Frente a yfinance o CCXT
tiene dos ventajas decisivas: son las velas **del broker con el que se va a
operar** (mismo spread, mismo horario de servidor, mismo instrumento), y llegan
en un solo tiro sin el limite de 60 dias de Yahoo.

El problema practico es que cada broker bautiza los instrumentos a su manera:

* **XM** publica el oro como ``GOLD``
* **Vantage Markets** como ``XAUUSD+`` en cuentas STP y ``XAUUSD`` en las raw
* ambos anaden sufijos segun el tipo de cuenta: ``.a``, ``m``, ``_i``, ``+``
* lo mismo ocurre con ``EURUSD``, que aparece como ``EURUSD``, ``EURUSD+`` o
  ``EURUSD.a`` segun donde se abra la cuenta

Codificar un nombre fijo garantiza que el bot falle al cambiar de broker, asi
que el simbolo se resuelve automaticamente contra el catalogo del instrumento
(ver :mod:`goldbot.instruments`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from goldbot.data.providers import DataProvider
from goldbot.instruments import InstrumentSpec, get_instrument
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import UTC

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Correspondencia timeframe -> constante de MT5. Se resuelve en caliente porque
# el modulo MetaTrader5 solo existe en Windows.
_TIMEFRAME_NAMES = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}


@dataclass
class SymbolInfo:
    """Simbolo resuelto en el broker conectado."""

    name: str
    description: str = ""
    digits: int = 2
    point: float = 0.01
    contract_size: float = 100.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    spread_points: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.name} ({self.description}) | {self.digits} decimales | "
            f"contrato {self.contract_size:,.0f} | "
            f"lotes {self.volume_min}-{self.volume_max} paso {self.volume_step}"
        )


def resolve_symbol(
    mt5, instrument: InstrumentSpec, preferred: str | None = None
) -> SymbolInfo | None:
    """Encuentra el simbolo de ``instrument`` en el broker conectado.

    Cada broker bautiza los instrumentos a su manera: XM publica el oro como
    ``GOLD``, Vantage como ``XAUUSD+``, y ambos anaden sufijos segun el tipo de
    cuenta. Fijar un nombre en el codigo garantiza que el bot deje de funcionar
    al cambiar de broker, asi que se resuelve en tres pasos, de mas especifico a
    mas general:

    1. El simbolo configurado por el usuario, si existe.
    2. Los alias conocidos del instrumento.
    3. Barrido del catalogo del broker.

    El tercer paso es el que permite sobrevivir a un broker no previsto.
    """
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(alias for alias in instrument.aliases if alias != preferred)

    for name in candidates:
        info = _try_symbol(mt5, name, instrument)
        if info is not None:
            logger.info("Simbolo resuelto para %s: %s", instrument.name, info.summary())
            return info

    logger.info(
        "Ningun alias conocido de %s existe en este broker; se busca en el catalogo",
        instrument.name,
    )
    try:
        all_symbols = mt5.symbols_get()
    except Exception as exc:
        logger.error("No se pudo leer el catalogo de simbolos: %s", exc)
        return None

    if not all_symbols:
        return None

    # Se buscan los simbolos cuyo nombre empieza por el del instrumento: asi
    # "EURUSD.a" o "XAUUSDm" entran, pero "EURGBP" o "XAGUSD" no.
    matches = [
        s.name for s in all_symbols
        if s.name.upper().startswith(instrument.name) or _matches_gold(s.name, instrument)
    ]

    # Los nombres cortos suelen ser el instrumento principal; los largos,
    # variantes exoticas (micro, cent, swap-free).
    for name in sorted(matches, key=len):
        info = _try_symbol(mt5, name, instrument)
        if info is not None:
            logger.info("Simbolo encontrado por barrido: %s", info.summary())
            return info

    logger.error(
        "No se encontro ningun simbolo para %s. Candidatos vistos: %s",
        instrument.name, matches[:10],
    )
    return None


def _matches_gold(name: str, instrument: InstrumentSpec) -> bool:
    """XM llama al oro ``GOLD``, que no empieza por 'XAUUSD'."""
    if instrument.name != "XAUUSD":
        return False
    return name.upper().startswith("GOLD")


def _try_symbol(mt5, name: str, instrument: InstrumentSpec) -> SymbolInfo | None:
    """Comprueba que el simbolo existe, se puede activar y cotiza."""
    try:
        info = mt5.symbol_info(name)
    except Exception:
        return None

    if info is None:
        return None

    # Un simbolo puede existir en el servidor pero no estar en Market Watch;
    # sin activarlo, copy_rates devuelve vacio sin dar ningun error.
    if not info.visible and not mt5.symbol_select(name, True):
        logger.debug("El simbolo %s existe pero no se pudo activar", name)
        return None

    info = mt5.symbol_info(name)
    tick = mt5.symbol_info_tick(name)
    if tick is None or tick.bid <= 0:
        logger.debug("El simbolo %s no devuelve cotizacion", name)
        return None

    # Comprobacion de cordura: cada instrumento cotiza en su rango. El oro va
    # en miles y el euro cerca de 1; si el precio cae fuera, hemos resuelto mal
    # el simbolo y estamos mirando otro mercado.
    if not instrument.is_plausible_price(tick.bid):
        logger.debug(
            "El simbolo %s cotiza a %.5f, fuera del rango de %s (%g-%g)",
            name, tick.bid, instrument.name, instrument.price_min, instrument.price_max,
        )
        return None

    return SymbolInfo(
        name=name,
        description=getattr(info, "description", ""),
        digits=info.digits,
        point=info.point,
        contract_size=getattr(info, "trade_contract_size", 100.0),
        volume_min=info.volume_min,
        volume_max=info.volume_max,
        volume_step=info.volume_step,
        spread_points=getattr(info, "spread", 0) * info.point,
    )


class MT5Provider(DataProvider):
    """Velas reales del broker via MetaTrader 5.

    Se puede usar de dos formas: como proveedor mas dentro del pipeline de
    datos, o directamente al arrancar para el volcado inicial de 5.000 velas.
    """

    name = "mt5"

    def __init__(
        self,
        symbol: str | None = None,
        instrument: InstrumentSpec | str | None = None,
        bootstrap_bars: int = 5000,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
    ) -> None:
        self.configured_symbol = symbol
        # El instrumento se deduce del simbolo configurado si no se pasa: asi
        # basta poner symbol: EURUSD en el YAML para cambiar de mercado.
        if isinstance(instrument, InstrumentSpec):
            self.instrument = instrument
        else:
            self.instrument = get_instrument(instrument or symbol or "XAUUSD")
        self.bootstrap_bars = bootstrap_bars
        self.login = login
        self.password = password
        self.server = server
        self._mt5 = None
        self._symbol_info: SymbolInfo | None = None
        self._owns_connection = False

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        try:
            import MetaTrader5  # noqa: F401
        except ImportError:
            logger.debug("MetaTrader5 no disponible (solo Windows)")
            return False
        return True

    def connect(self, external_mt5=None) -> bool:
        """Inicializa MT5 y resuelve el simbolo del oro.

        ``external_mt5`` permite reutilizar la conexion que ya abrio el broker
        de ejecucion, en lugar de abrir una segunda: MT5 solo admite un
        terminal por proceso.
        """
        if external_mt5 is not None:
            self._mt5 = external_mt5
            self._owns_connection = False
        else:
            if not self.is_available():
                return False
            import MetaTrader5 as mt5

            if self.login and self.password and self.server:
                ok = mt5.initialize(login=self.login, password=self.password, server=self.server)
            else:
                ok = mt5.initialize()

            if not ok:
                logger.error("mt5.initialize() fallo: %s", mt5.last_error())
                return False

            self._mt5 = mt5
            self._owns_connection = True

        self._symbol_info = resolve_symbol(self._mt5, self.instrument, self.configured_symbol)
        if self._symbol_info is None:
            logger.error(
                "No se pudo resolver el simbolo de %s en este broker", self.instrument.name
            )
            return False

        account = self._mt5.account_info()
        if account is not None:
            logger.info(
                "MT5 conectado | broker=%s | cuenta=%s | divisa=%s",
                getattr(account, "company", "?"),
                getattr(account, "login", "?"),
                getattr(account, "currency", "?"),
            )
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None and self._owns_connection:
            self._mt5.shutdown()
        self._mt5 = None

    @property
    def symbol_info(self) -> SymbolInfo | None:
        return self._symbol_info

    # ------------------------------------------------------------------ #
    def bootstrap(self, timeframe: str = "5m", bars: int | None = None) -> pd.DataFrame:
        """Descarga las ultimas ``bars`` velas cerradas del broker.

        Es la primera llamada del arranque: da al motor evolutivo datos reales
        del instrumento que se va a operar, en lugar de un proxy.

        La vela en curso (indice 0) se descarta siempre: todavia se esta
        formando y su maximo, minimo y cierre cambiaran. Entrenar con ella
        introduce una vela de ruido en el punto exacto donde el modelo toma la
        decision, que es el peor sitio posible.
        """
        if self._mt5 is None or self._symbol_info is None:
            logger.error("MT5 no conectado: llama antes a connect()")
            return _empty()

        count = bars or self.bootstrap_bars
        tf = self._timeframe_constant(timeframe)
        if tf is None:
            return _empty()

        # Se pide una vela de mas para poder descartar la que sigue abierta.
        try:
            rates = self._mt5.copy_rates_from_pos(self._symbol_info.name, tf, 0, count + 1)
        except Exception as exc:
            logger.error("copy_rates_from_pos fallo: %s", exc)
            return _empty()

        if rates is None or len(rates) == 0:
            logger.error(
                "El broker no devolvio velas para %s. Revisa que el grafico este "
                "abierto en el terminal y que el historico este descargado.",
                self._symbol_info.name,
            )
            return _empty()

        df = _rates_to_frame(rates)
        if len(df) > 1:
            df = df.iloc[:-1]  # fuera la vela en formacion

        logger.info(
            "Descargadas %d velas %s de %s [%s -> %s]",
            len(df),
            timeframe,
            self._symbol_info.name,
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )

        if len(df) < count * 0.8:
            logger.warning(
                "Se pidieron %d velas y solo llegaron %d. En MT5 hay que abrir el "
                "grafico del simbolo y desplazarse hacia atras para que el terminal "
                "descargue el historico completo.",
                count,
                len(df),
            )
        return df

    def fetch(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """Velas en un rango de fechas (interfaz estandar de proveedor)."""
        if self._mt5 is None and not self.connect():
            return _empty()
        if self._symbol_info is None:
            return _empty()

        tf = self._timeframe_constant(timeframe)
        if tf is None:
            return _empty()

        try:
            rates = self._mt5.copy_rates_range(self._symbol_info.name, tf, start, end)
        except Exception as exc:
            logger.warning("copy_rates_range fallo: %s", exc)
            return _empty()

        if rates is None or len(rates) == 0:
            # Sin datos en el rango: se cae al volcado por posicion, que es mas
            # robusto cuando el historico del terminal esta incompleto.
            logger.info("Sin velas en el rango pedido; se usa el volcado por posicion")
            return self.bootstrap(timeframe)

        return _rates_to_frame(rates)

    # ------------------------------------------------------------------ #
    def _timeframe_constant(self, timeframe: str):
        attribute = _TIMEFRAME_NAMES.get(timeframe)
        if attribute is None:
            logger.error("Timeframe no soportado por MT5: %s", timeframe)
            return None
        return getattr(self._mt5, attribute)

    def broker_costs(self) -> dict[str, float]:
        """Costes reales leidos del broker, para calibrar el backtest.

        Usar el spread que de verdad cobra XM o Vantage en lugar de una
        estimacion es lo que hace que el backtest y la operativa converjan.
        """
        if self._symbol_info is None:
            return {}
        return {
            "spread_points": float(self._symbol_info.spread_points),
            "contract_size": float(self._symbol_info.contract_size),
            "min_lot_size": float(self._symbol_info.volume_min),
            "max_lot_size": float(self._symbol_info.volume_max),
            "lot_step": float(self._symbol_info.volume_step),
        }


# --------------------------------------------------------------------------- #
def _rates_to_frame(rates) -> pd.DataFrame:
    """Convierte el array estructurado de MT5 en un DataFrame OHLCV en UTC."""
    df = pd.DataFrame(rates)

    # El servidor MT5 marca los tiempos en su propia zona horaria, pero el
    # campo 'time' es un epoch UTC, asi que la conversion es directa.
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    # tick_volume es el numero de cambios de precio: en Forex/CFD es el unico
    # volumen fiable, porque real_volume suele venir a cero.
    volume = df["real_volume"] if "real_volume" in df and df["real_volume"].sum() > 0 else df.get("tick_volume", 0)

    out = pd.DataFrame(
        {
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": pd.Series(volume, index=df.index).astype("float64"),
        },
        index=df.index,
    )
    return out[~out.index.duplicated(keep="last")]


def _empty() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz=UTC, name="timestamp")
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=index, dtype="float64")
