"""Orquestacion de datos: descarga incremental, limpieza y validacion.

`MarketData` es el unico punto por el que el resto del sistema pide velas. Se
encarga de decidir que falta, a quien pedirselo, como limpiarlo y de dejarlo
persistido para la siguiente sesion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from goldbot.config import Config
from goldbot.data.cache import OHLCV_COLUMNS, ParquetCache
from goldbot.data.providers import DataProvider, SyntheticProvider, build_providers
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import UTC, ensure_utc_index, is_weekend_gap, now_utc

logger = get_logger(__name__)


@dataclass
class DataQuality:
    """Informe de calidad. Si ``is_usable`` es False, no se entrena."""

    total_bars: int
    start: datetime | None
    end: datetime | None
    duplicated: int
    gaps: int
    largest_gap_minutes: float
    nan_rows: int
    invalid_ohlc: int
    zero_volume_pct: float
    is_usable: bool
    notes: list[str]

    def summary(self) -> str:
        head = f"{self.total_bars} barras"
        if self.start and self.end:
            head += f" [{self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d}]"
        return (
            f"{head} | huecos={self.gaps} (max {self.largest_gap_minutes:.0f}min) "
            f"| duplicados={self.duplicated} | NaN={self.nan_rows} "
            f"| OHLC invalidos={self.invalid_ohlc} | vol0={self.zero_volume_pct:.1%}"
        )


class MarketData:
    """Fachada de datos de mercado con cache incremental."""

    def __init__(self, config: Config, providers: list[DataProvider] | None = None) -> None:
        self.config = config
        self.providers = providers if providers is not None else build_providers(config)
        self.cache = ParquetCache(
            cache_dir=config.path(config.data.cache_dir),
            symbol=config.data.symbol,
            timeframe=config.data.timeframe,
        )

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def update(self, lookback_days: int | None = None) -> pd.DataFrame:
        """Trae lo que falte desde el ultimo cierre cacheado y lo persiste.

        Es la llamada del ciclo diario: barata cuando el cache esta al dia y
        capaz de reconstruir el historico completo en el primer arranque.
        """
        end = now_utc()
        last = self.cache.last_timestamp()

        if last is None:
            start = end - timedelta(days=lookback_days or self.config.data.history_days)
            logger.info("Cache vacio: descarga inicial desde %s", start.date())
        else:
            # Solapamos una vela para capturar revisiones de la ultima barra.
            start = last - timedelta(minutes=self.config.data.timeframe_minutes)
            if (end - start) < timedelta(minutes=self.config.data.timeframe_minutes):
                logger.debug("Cache ya al dia (ultima barra %s)", last)
                return self.load()
            logger.info("Actualizacion incremental desde %s", start)

        fetched = self._fetch_from_providers(start, end)
        if fetched.empty:
            logger.warning("Ningun proveedor devolvio datos nuevos")
            return self.load()

        fetched = self._clean(fetched)
        merged = self.cache.merge(fetched)
        return self._clean(merged)

    def load(self, min_bars: int | None = None) -> pd.DataFrame:
        """Devuelve el historico cacheado ya limpio (sin tocar la red)."""
        df = self._clean(self.cache.load())
        required = min_bars if min_bars is not None else 0
        if required and len(df) < required:
            logger.warning("Solo hay %d barras en cache (se pedian %d)", len(df), required)
        return df

    def get(self, min_bars: int | None = None, refresh: bool = True) -> pd.DataFrame:
        """Atajo: actualiza si procede y devuelve el dataset listo para usar.

        Si tras actualizar no se alcanza el minimo de barras configurado, cae a
        datos sinteticos para que el sistema pueda arrancar y entrenarse en
        seco en lugar de morir. Se avisa por log de forma bien visible.
        """
        df = self.update() if refresh else self.load()
        required = min_bars if min_bars is not None else self.config.data.min_bars_required

        if len(df) < required:
            logger.warning(
                "Historico insuficiente (%d < %d). Se generan datos SINTETICOS: "
                "los resultados NO son validos para operar en real.",
                len(df),
                required,
            )
            df = self._synthetic_fallback(required)

        quality = self.assess(df)
        logger.info("Calidad de datos: %s", quality.summary())
        if not quality.is_usable:
            logger.error("Datos no utilizables: %s", "; ".join(quality.notes))
        return df

    def assess(self, df: pd.DataFrame) -> DataQuality:
        """Audita el dataset y decide si sirve para entrenar."""
        notes: list[str] = []
        if df.empty:
            return DataQuality(0, None, None, 0, 0, 0.0, 0, 0, 0.0, False, ["dataset vacio"])

        duplicated = int(df.index.duplicated().sum())
        nan_rows = int(df[OHLCV_COLUMNS].isna().any(axis=1).sum())

        invalid = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        )
        invalid_ohlc = int(invalid.sum())

        step = pd.Timedelta(minutes=self.config.data.timeframe_minutes)
        deltas = df.index.to_series().diff().dropna()
        # Los huecos de fin de semana son normales en el oro: no cuentan.
        weekend = pd.Series(is_weekend_gap(df.index), index=df.index).reindex(deltas.index).fillna(False)
        real_gaps = deltas[(deltas > step) & (~weekend.to_numpy())]
        gaps = int(len(real_gaps))
        largest_gap = float(real_gaps.max().total_seconds() / 60) if gaps else 0.0

        zero_volume_pct = float((df["volume"] <= 0).mean())

        if duplicated:
            notes.append(f"{duplicated} timestamps duplicados")
        if invalid_ohlc:
            notes.append(f"{invalid_ohlc} velas con OHLC incoherente")
        if nan_rows:
            notes.append(f"{nan_rows} filas con NaN")
        if largest_gap > self.config.data.max_gap_minutes:
            notes.append(f"hueco de {largest_gap:.0f} min")
        if zero_volume_pct > 0.5:
            notes.append(f"volumen nulo en el {zero_volume_pct:.0%} de las velas")

        is_usable = (
            len(df) >= self.config.data.min_bars_required
            and invalid_ohlc == 0
            and nan_rows == 0
            and duplicated == 0
        )
        if len(df) < self.config.data.min_bars_required:
            notes.append(f"solo {len(df)} barras (minimo {self.config.data.min_bars_required})")

        return DataQuality(
            total_bars=len(df),
            start=df.index[0].to_pydatetime(),
            end=df.index[-1].to_pydatetime(),
            duplicated=duplicated,
            gaps=gaps,
            largest_gap_minutes=largest_gap,
            nan_rows=nan_rows,
            invalid_ohlc=invalid_ohlc,
            zero_volume_pct=zero_volume_pct,
            is_usable=is_usable,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Interno
    # ------------------------------------------------------------------ #
    def _fetch_from_providers(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Recorre los proveedores en orden y devuelve el primero que sirva."""
        timeframe = self.config.data.timeframe
        for provider in self.providers:
            if not provider.is_available():
                logger.debug("Proveedor %s no disponible", provider.name)
                continue
            try:
                logger.info("Consultando %s (%s -> %s)", provider.name, start.date(), end.date())
                df = provider.fetch(start, end, timeframe)
            except Exception as exc:
                logger.warning("Proveedor %s fallo: %s", provider.name, exc)
                continue
            if not df.empty:
                logger.info("%s devolvio %d barras", provider.name, len(df))
                return df
            logger.info("%s no devolvio barras", provider.name)
        return pd.DataFrame()

    def _synthetic_fallback(self, required: int) -> pd.DataFrame:
        minutes = self.config.data.timeframe_minutes
        # Con margen: el filtro de fin de semana descarta ~28% de las barras.
        days = int(required * minutes / (60 * 24) * 1.6) + 30
        end = now_utc()
        df = SyntheticProvider().fetch(end - timedelta(days=days), end, self.config.data.timeframe)
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza, deduplica, corrige OHLC y elimina el ruido de fin de semana."""
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()

        df = ensure_utc_index(df)
        keep = [c for c in OHLCV_COLUMNS if c in df.columns]
        df = df[keep].copy()

        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col == "volume" else np.nan
        df = df[OHLCV_COLUMNS].astype("float64")

        df = df[~df.index.duplicated(keep="last")]

        # Precios no positivos o nulos: la vela no es recuperable.
        prices = df[["open", "high", "low", "close"]]
        df = df[prices.notna().all(axis=1) & (prices > 0).all(axis=1)]

        df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)

        # Coherencia de mechas: high/low deben envolver a open/close.
        body_max = df[["open", "close"]].max(axis=1)
        body_min = df[["open", "close"]].min(axis=1)
        df["high"] = df[["high"]].join(body_max.rename("b")).max(axis=1)
        df["low"] = df[["low"]].join(body_min.rename("b")).min(axis=1)

        # Velas planas repetidas (feed congelado) y fines de semana.
        df = df[~is_weekend_gap(df.index)]
        flat = (df["high"] == df["low"]) & (df["volume"] <= 0)
        if flat.any():
            logger.debug("Descartadas %d velas planas sin volumen", int(flat.sum()))
            df = df[~flat]

        # Saltos absurdos de precio (errores del feed): >8% en una vela de 5m.
        returns = df["close"].pct_change()
        spikes = returns.abs() > 0.08
        if spikes.any():
            logger.warning("Descartados %d saltos de precio anomalos", int(spikes.sum()))
            df = df[~spikes.fillna(False)]

        return df.sort_index()


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Reagrega velas a un timeframe superior (util para features multi-escala)."""
    df = ensure_utc_index(df)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(timeframe, label="right", closed="right").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
