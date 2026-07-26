"""Helpers de tiempo y sesiones de mercado.

Todo el sistema trabaja internamente en UTC. Las sesiones (Asia, Londres,
Nueva York) se derivan de la hora UTC porque el oro cotiza casi 24h y el
comportamiento del precio cambia mucho segun la sesion activa.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

UTC = timezone.utc

# Rangos de sesion en horas UTC (aproximados, sin ajuste de horario de verano).
SESSIONS: dict[str, tuple[int, int]] = {
    "sydney": (21, 6),
    "tokyo": (0, 9),
    "london": (7, 16),
    "newyork": (12, 21),
}

# La ventana de mayor volumen del oro: solape Londres/NY.
OVERLAP_LONDON_NY = (12, 16)


def now_utc() -> datetime:
    """Instante actual, siempre timezone-aware en UTC."""
    return datetime.now(tz=UTC)


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que el indice sea un DatetimeIndex tz-aware en UTC y ordenado."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(UTC)
    elif str(df.index.tz) != "UTC":
        df = df.copy()
        df.index = df.index.tz_convert(UTC)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def in_session(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    """Mascara booleana de las barras que caen dentro de ``session``."""
    if session not in SESSIONS:
        raise KeyError(f"Sesion desconocida: {session!r}. Validas: {sorted(SESSIONS)}")
    start, end = SESSIONS[session]
    hours = index.hour.to_numpy()
    if start <= end:
        return (hours >= start) & (hours < end)
    # Sesion que cruza medianoche (p.ej. Sydney 21:00 -> 06:00).
    return (hours >= start) | (hours < end)


def in_overlap(index: pd.DatetimeIndex) -> np.ndarray:
    """Mascara del solape Londres/Nueva York, la franja mas liquida del oro."""
    start, end = OVERLAP_LONDON_NY
    hours = index.hour.to_numpy()
    return (hours >= start) & (hours < end)


def is_weekend_gap(index: pd.DatetimeIndex) -> np.ndarray:
    """Barras del cierre de fin de semana (viernes 21:00 UTC -> domingo 22:00 UTC)."""
    dow = index.dayofweek.to_numpy()
    hours = index.hour.to_numpy()
    saturday = dow == 5
    friday_late = (dow == 4) & (hours >= 21)
    sunday_early = (dow == 6) & (hours < 22)
    return saturday | friday_late | sunday_early


def bars_per_day(timeframe_minutes: int, hours_per_day: float = 23.0) -> int:
    """Barras aproximadas por dia de negociacion para un timeframe dado."""
    return max(1, int(round(hours_per_day * 60 / timeframe_minutes)))


def annualization_factor(timeframe_minutes: int, trading_days: int = 252) -> float:
    """Factor para anualizar metricas calculadas sobre retornos por barra."""
    return float(np.sqrt(bars_per_day(timeframe_minutes) * trading_days))


def floor_to_timeframe(ts: datetime, timeframe_minutes: int) -> datetime:
    """Redondea ``ts`` hacia abajo al inicio de su vela."""
    ts = ts.astimezone(UTC)
    discard = timedelta(
        minutes=ts.minute % timeframe_minutes,
        seconds=ts.second,
        microseconds=ts.microsecond,
    )
    return ts - discard


def next_bar_close(ts: datetime, timeframe_minutes: int) -> datetime:
    """Instante en que cierra la vela que contiene a ``ts``."""
    return floor_to_timeframe(ts, timeframe_minutes) + timedelta(minutes=timeframe_minutes)


def seconds_until_next_close(timeframe_minutes: int, offset_seconds: float = 2.0) -> float:
    """Segundos hasta el proximo cierre de vela, con un pequeno margen.

    El margen evita leer una vela que el proveedor de datos todavia no ha
    consolidado.
    """
    now = now_utc()
    target = next_bar_close(now, timeframe_minutes) + timedelta(seconds=offset_seconds)
    return max(0.0, (target - now).total_seconds())


def market_is_open(ts: datetime | None = None) -> bool:
    """Mercado spot del oro abierto (dom 22:00 UTC -> vie 21:00 UTC)."""
    ts = ts or now_utc()
    ts = ts.astimezone(UTC)
    dow, clock = ts.weekday(), ts.time()
    if dow == 5:  # sabado
        return False
    if dow == 6:  # domingo: abre a las 22:00
        return clock >= time(22, 0)
    if dow == 4:  # viernes: cierra a las 21:00
        return clock < time(21, 0)
    return True
