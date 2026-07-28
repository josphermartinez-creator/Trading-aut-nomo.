"""Proveedores de velas de oro.

El oro spot (XAU/USD) no cotiza en exchanges cripto, asi que combinamos varias
fuentes segun disponibilidad:

* ``yfinance``  -> futuros COMEX ``GC=F`` (o el spot ``XAUUSD=X``). Gratis y
  fiable, pero solo devuelve ~60 dias de intradia de 5 minutos.
* ``ccxt``      -> ``PAXG/USDT`` en Binance: token respaldado 1:1 por oro
  fisico en camaras de Londres. Cotiza 24/7 y permite paginar anos de M5.
* ``csv``       -> exportacion propia (p.ej. de MetaTrader 5), el historico de
  mayor calidad para XAUUSD real.
* ``synthetic`` -> generador para tests y desarrollo sin red.

Ninguno es perfecto en solitario; la clase :class:`~goldbot.data.pipeline.MarketData`
los encadena y va acumulando en el cache local.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from goldbot.config import Config
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import UTC, ensure_utc_index, is_weekend_gap

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    """Interfaz comun a todas las fuentes."""

    name: str = "base"

    @abstractmethod
    def fetch(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """Devuelve OHLCV en [start, end] indexado en UTC. Vacio si no hay datos."""

    def is_available(self) -> bool:
        """``True`` si el proveedor puede usarse en este entorno."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - cosmetico
        return f"<{type(self).__name__} name={self.name!r}>"


# --------------------------------------------------------------------------- #
# yfinance
# --------------------------------------------------------------------------- #
class YFinanceProvider(DataProvider):
    """Futuros/spot de oro via Yahoo Finance.

    Limite conocido: el intervalo de 5m solo cubre los ultimos ~60 dias
    naturales. Por eso el cache incremental es imprescindible.
    """

    name = "yfinance"
    MAX_INTRADAY_DAYS = 59  # Yahoo rechaza peticiones de 5m mas largas

    def __init__(self, symbol: str = "GC=F", fallback_symbol: str | None = "XAUUSD=X") -> None:
        self.symbol = symbol
        self.fallback_symbol = fallback_symbol

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            logger.debug("yfinance no instalado")
            return False
        return True

    def fetch(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            return _empty()

        symbols = [self.symbol] + ([self.fallback_symbol] if self.fallback_symbol else [])
        for symbol in symbols:
            df = self._fetch_symbol(symbol, start, end, timeframe)
            if not df.empty:
                if symbol != self.symbol:
                    logger.info("yfinance: usando simbolo de respaldo %s", symbol)
                return df
        logger.warning("yfinance no devolvio datos para %s", symbols)
        return _empty()

    def _fetch_symbol(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        import yfinance as yf

        # Yahoo limita el intradia; recortamos la ventana a lo permitido.
        span_days = (end - start).days
        if timeframe.endswith("m") and span_days > self.MAX_INTRADAY_DAYS:
            start = end - timedelta(days=self.MAX_INTRADAY_DAYS)
            logger.debug(
                "yfinance: ventana recortada a %d dias por limite de intradia",
                self.MAX_INTRADAY_DAYS,
            )

        frames: list[pd.DataFrame] = []
        # Yahoo se atraganta con peticiones largas de 5m: troceamos en bloques de 7 dias.
        chunk = timedelta(days=7) if timeframe.endswith("m") else timedelta(days=365)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            try:
                raw = yf.download(
                    symbol,
                    start=cursor,
                    end=chunk_end,
                    interval=timeframe,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
            except Exception as exc:
                logger.warning("yfinance fallo en %s [%s]: %s", symbol, cursor.date(), exc)
                raw = None

            if raw is not None and not raw.empty:
                frames.append(_normalize_yfinance(raw))
            cursor = chunk_end
            time.sleep(0.35)  # cortesia con la API publica

        if not frames:
            return _empty()
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df


def _normalize_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    """Aplana el MultiIndex de columnas y normaliza nombres."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance devuelve (campo, ticker) cuando se pide un solo simbolo.
        df.columns = [str(col[0]) for col in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {"adj_close": "adj_close", "close": "close"}
    df = df.rename(columns=rename)
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col == "volume" else np.nan
    df = ensure_utc_index(df[OHLCV_COLUMNS])
    return df.astype("float64")


# --------------------------------------------------------------------------- #
# CCXT
# --------------------------------------------------------------------------- #
class CCXTProvider(DataProvider):
    """OHLCV via CCXT. Por defecto PAXG/USDT como proxy de oro fisico.

    A diferencia de Yahoo, aqui se puede paginar hacia atras varios anos de
    velas de 5 minutos, lo que da al motor evolutivo una muestra decente desde
    el primer arranque.
    """

    name = "ccxt"
    LIMIT = 1000  # velas por peticion en la mayoria de exchanges

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "PAXG/USDT",
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange = None

    def is_available(self) -> bool:
        try:
            import ccxt  # noqa: F401
        except ImportError:
            logger.debug("ccxt no instalado")
            return False
        return True

    def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange
        import ccxt

        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"Exchange desconocido para ccxt: {self.exchange_id}")
        params: dict = {"enableRateLimit": True, "timeout": 30_000}
        if self.api_key and self.api_secret:
            params.update({"apiKey": self.api_key, "secret": self.api_secret})
        self._exchange = getattr(ccxt, self.exchange_id)(params)
        return self._exchange

    def fetch(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            return _empty()
        try:
            exchange = self._get_exchange()
        except Exception as exc:
            logger.warning("No se pudo inicializar ccxt/%s: %s", self.exchange_id, exc)
            return _empty()

        if timeframe not in getattr(exchange, "timeframes", {timeframe: None}):
            logger.warning("%s no soporta el timeframe %s", self.exchange_id, timeframe)
            return _empty()

        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[float]] = []
        stall_guard = 0

        while since < end_ms:
            try:
                batch = exchange.fetch_ohlcv(self.symbol, timeframe, since=since, limit=self.LIMIT)
            except Exception as exc:
                logger.warning("ccxt fetch_ohlcv fallo: %s", exc)
                break
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= since:
                # El exchange no avanza: evitamos el bucle infinito.
                stall_guard += 1
                if stall_guard >= 3:
                    break
                since += self.LIMIT * _timeframe_ms(timeframe)
            else:
                stall_guard = 0
                since = last_ts + _timeframe_ms(timeframe)
            time.sleep(max(exchange.rateLimit, 200) / 1000.0)

        if not rows:
            return _empty()

        df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return df.astype("float64")


def _timeframe_ms(timeframe: str) -> int:
    """Convierte '5m', '1h', '1d' a milisegundos."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    unit = timeframe[-1].lower()
    if unit not in units:
        raise ValueError(f"Timeframe no soportado: {timeframe}")
    return int(timeframe[:-1]) * units[unit]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
class CSVProvider(DataProvider):
    """Historico propio en CSV (por ejemplo exportado de MetaTrader 5).

    Acepta los formatos habituales: cabecera con ``time/date/timestamp`` o el
    export tabulado de MT5 (``<DATE>\\t<TIME>\\t<OPEN>...``).
    """

    name = "csv"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def is_available(self) -> bool:
        return self.path.exists()

    def fetch(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            logger.debug("CSV no encontrado en %s", self.path)
            return _empty()
        try:
            df = self._read()
        except Exception as exc:
            logger.warning("No se pudo leer el CSV %s: %s", self.path, exc)
            return _empty()
        if df.empty:
            return df
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]

    def _read(self) -> pd.DataFrame:
        sep = "\t" if self.path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(self.path, sep=sep)
        df.columns = [str(c).strip().lower().strip("<>").replace(" ", "_") for c in df.columns]

        # Fecha y hora en columnas separadas (formato MT5).
        if "date" in df.columns and "time" in df.columns:
            stamps = df["date"].astype(str) + " " + df["time"].astype(str)
        else:
            time_col = next(
                (c for c in ("timestamp", "datetime", "time", "date") if c in df.columns), None
            )
            if time_col is None:
                raise ValueError(f"El CSV no tiene columna temporal. Columnas: {list(df.columns)}")
            stamps = df[time_col]

        df.index = pd.to_datetime(stamps, utc=True, format="mixed")
        df.index.name = "timestamp"

        rename = {"vol": "volume", "tickvol": "volume", "tick_volume": "volume", "real_volume": "volume"}
        df = df.rename(columns=rename)
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col == "volume" else np.nan
        df = df[OHLCV_COLUMNS].astype("float64")
        return ensure_utc_index(df)


# --------------------------------------------------------------------------- #
# Sintetico
# --------------------------------------------------------------------------- #
class SyntheticProvider(DataProvider):
    """Genera velas realistas sin red: tests, CI y desarrollo offline.

    No es ruido blanco: mezcla tendencia con reversion a la media, volatilidad
    agrupada (GARCH-like) y estacionalidad por sesion, de modo que una
    estrategia trivial no gane dinero por accidente.
    """

    name = "synthetic"

    def __init__(self, seed: int = 42, start_price: float = 2000.0) -> None:
        self.seed = seed
        self.start_price = start_price

    def fetch(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        minutes = _timeframe_ms(timeframe) // 60_000
        # Alineamos al reloj de velas para que los timestamps sean realistas
        # (00:05, 00:10, ...) y no arrastren los segundos de "ahora".
        start = start.replace(second=0, microsecond=0)
        start -= timedelta(minutes=start.minute % minutes)
        index = pd.date_range(start=start, end=end, freq=f"{minutes}min", tz=UTC)
        index = index[~is_weekend_gap(index)]
        n = len(index)
        if n == 0:
            return _empty()

        rng = np.random.default_rng(self.seed)

        # Volatilidad agrupada: la de hoy depende de la de ayer.
        vol = np.empty(n)
        vol[0] = 0.0006
        shocks = rng.normal(0, 1, n)
        for i in range(1, n):
            vol[i] = np.sqrt(1e-8 + 0.05 * (vol[i - 1] * shocks[i - 1]) ** 2 + 0.93 * vol[i - 1] ** 2)

        # Mas volatilidad en el solape Londres/NY.
        session_boost = np.where((index.hour >= 12) & (index.hour < 16), 1.6, 1.0)
        vol = vol * session_boost

        # Deriva lenta con reversion a la media (AR(1)), NO un paseo aleatorio:
        # integrar dos veces produciria una serie con tendencias absurdas en la
        # que cualquier estrategia de seguimiento ganaria, invalidando los tests.
        drift = np.zeros(n)
        drift_shocks = rng.normal(0, 3e-6, n)
        for i in range(1, n):
            drift[i] = 0.995 * drift[i - 1] + drift_shocks[i]

        noise = rng.normal(0, 1, n) * vol
        # Reversion a corto: parte del movimiento de la vela previa se deshace.
        reversion = np.concatenate([[0.0], -0.08 * noise[:-1]])

        log_returns = drift + noise + reversion
        close = self.start_price * np.exp(np.cumsum(log_returns))

        open_ = np.concatenate([[self.start_price], close[:-1]])
        wick = np.abs(rng.normal(0, 1, n)) * vol * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        volume = rng.gamma(shape=2.0, scale=500.0, size=n) * session_boost

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        ).astype("float64")


# --------------------------------------------------------------------------- #
# Fabrica
# --------------------------------------------------------------------------- #
def build_providers(config: Config) -> list[DataProvider]:
    """Instancia los proveedores declarados en la configuracion, en orden."""
    registry: dict[str, callable] = {
        "yfinance": lambda: YFinanceProvider(
            symbol=config.data.yfinance_symbol or config.instrument.yfinance_symbol,
            fallback_symbol=config.data.yfinance_fallback or config.instrument.yfinance_fallback,
        ),
        "ccxt": lambda: CCXTProvider(
            exchange_id=config.data.ccxt_exchange,
            symbol=config.data.ccxt_symbol,
            api_key=config.api_key,
            api_secret=config.api_secret,
        ),
        "mt5": lambda: _build_mt5(config),
        "csv": lambda: CSVProvider(path=config.path(config.data.csv_path)),
        "synthetic": lambda: SyntheticProvider(),
    }
    providers: list[DataProvider] = []
    for name in config.data.providers:
        factory = registry.get(name)
        if factory is None:
            logger.warning("Proveedor desconocido en la configuracion: %s", name)
            continue
        providers.append(factory())
    if not providers:
        logger.warning("Sin proveedores validos; se usara el sintetico")
        providers.append(SyntheticProvider())
    return providers


def _build_mt5(config: Config) -> DataProvider:
    """Proveedor MT5 configurado con el instrumento y el volcado inicial."""
    from goldbot.data.mt5_provider import MT5Provider

    return MT5Provider(
        symbol=config.data.mt5_symbol or None,
        instrument=config.instrument,
        bootstrap_bars=config.data.mt5_bootstrap_bars,
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
    )


def _empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=idx, dtype="float64")
