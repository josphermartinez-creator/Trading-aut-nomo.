"""Capa de datos: proveedores, cache incremental y limpieza."""

from goldbot.data.cache import ParquetCache
from goldbot.data.pipeline import MarketData
from goldbot.data.providers import (
    CCXTProvider,
    CSVProvider,
    DataProvider,
    SyntheticProvider,
    YFinanceProvider,
    build_providers,
)

__all__ = [
    "CCXTProvider",
    "CSVProvider",
    "DataProvider",
    "MarketData",
    "ParquetCache",
    "SyntheticProvider",
    "YFinanceProvider",
    "build_providers",
]
