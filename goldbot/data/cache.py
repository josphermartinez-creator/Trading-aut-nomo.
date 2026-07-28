"""Cache incremental de velas en Parquet.

Es la pieza que hace que el historico crezca solo: yfinance unicamente sirve
~60 dias de M5, pero si cada ejecucion diaria vuelca lo nuevo al cache, al cabo
de unos meses el bot dispone de un historico M5 que ningun proveedor gratuito
entrega de una sola vez.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import ensure_utc_index

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class ParquetCache:
    """Almacen columnar de OHLCV con merge idempotente."""

    def __init__(self, cache_dir: str | Path, symbol: str, timeframe: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe

    @property
    def path(self) -> Path:
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_symbol}_{self.timeframe}.parquet"

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> pd.DataFrame:
        """Devuelve el cache completo, o un DataFrame vacio si no hay nada."""
        if not self.exists():
            return _empty_frame()
        try:
            df = pd.read_parquet(self.path)
        except Exception as exc:  # parquet corrupto: no es fatal, se reconstruye
            logger.warning("Cache ilegible en %s (%s). Se reconstruira.", self.path, exc)
            return _empty_frame()
        if df.empty:
            return _empty_frame()
        df = ensure_utc_index(df)
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Cache sin columnas %s; se descarta.", missing)
            return _empty_frame()
        return df[OHLCV_COLUMNS]

    def save(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = ensure_utc_index(df)[OHLCV_COLUMNS]
        tmp = self.path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, compression="snappy")
        tmp.replace(self.path)  # escritura atomica: nunca dejamos un cache a medias
        logger.debug("Cache guardado: %d barras en %s", len(df), self.path)

    def merge(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Funde ``new_data`` con lo ya cacheado y persiste el resultado.

        Ante timestamps duplicados gana el dato nuevo: los proveedores suelen
        revisar la ultima vela una vez consolidada.
        """
        if new_data is None or new_data.empty:
            return self.load()

        new_data = ensure_utc_index(new_data)
        keep = [c for c in OHLCV_COLUMNS if c in new_data.columns]
        new_data = new_data[keep]

        existing = self.load()
        if existing.empty:
            combined = new_data
        else:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()

        added = len(combined) - len(existing)
        if added:
            logger.info(
                "Cache %s: %d barras nuevas (total %d, %s -> %s)",
                self.symbol,
                added,
                len(combined),
                combined.index[0].date(),
                combined.index[-1].date(),
            )
        self.save(combined)
        return combined

    def last_timestamp(self) -> datetime | None:
        df = self.load()
        return None if df.empty else df.index[-1].to_pydatetime()

    def coverage(self) -> tuple[datetime | None, datetime | None, int]:
        """(primera barra, ultima barra, total) para diagnostico."""
        df = self.load()
        if df.empty:
            return None, None, 0
        return df.index[0].to_pydatetime(), df.index[-1].to_pydatetime(), len(df)


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=idx, dtype="float64")
