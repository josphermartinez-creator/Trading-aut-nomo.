"""Indicadores tecnicos implementados sobre numpy/pandas.

Sin TA-Lib a proposito: compilar TA-Lib en un VPS es una fuente clasica de
problemas de despliegue y aqui no aporta nada que pandas no resuelva.

Regla de oro de todo el modulo: **ningun indicador mira al futuro**. Todas las
ventanas son causales (``rolling``/``ewm`` hacia atras). Cualquier cambio aqui
debe preservar esa propiedad o el backtest mentira.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Medias
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype="float64")
    return series.rolling(period, min_periods=period).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: mucho menos retardo que una SMA equivalente."""
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    return wma(2 * wma(series, half) - wma(series, period), sqrt_period)


def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive MA: se acelera en tendencia y se frena en rango."""
    change = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period, min_periods=period).sum()
    efficiency = (change / volatility.replace(0, np.nan)).fillna(0.0)
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    smoothing = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2

    values = series.to_numpy(dtype="float64")
    alpha = smoothing.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    if len(values) <= period:
        return pd.Series(out, index=series.index)
    out[period] = values[period]
    for i in range(period + 1, len(values)):
        out[i] = out[i - 1] + alpha[i] * (values[i] - out[i - 1])
    return pd.Series(out, index=series.index)


# --------------------------------------------------------------------------- #
# Osciladores
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Suavizado de Wilder = EMA con alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Sin perdidas en la ventana el RSI es 100 por definicion.
    return out.where(avg_loss != 0, 100.0)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    span = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / span
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    span = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - close) / span


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    mean = typical.rolling(period, min_periods=period).mean()
    mad = (typical - mean).abs().rolling(period, min_periods=period).mean()
    return (typical - mean) / (0.015 * mad.replace(0, np.nan))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period) * 100


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)


def tsi(series: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index: momento doblemente suavizado, poco ruidoso."""
    diff = series.diff()
    double = ema(ema(diff, long), short)
    double_abs = ema(ema(diff.abs(), long), short)
    return 100 * double / double_abs.replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Volatilidad
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(
    series: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return mid + std_mult * std, mid, mid - std_mult * std


def bollinger_pct(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Posicion dentro de las bandas: 0 = banda baja, 1 = banda alta."""
    upper, _, lower = bollinger(series, period, std_mult)
    span = (upper - lower).replace(0, np.nan)
    return (series - lower) / span


def bollinger_width(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Anchura normalizada: detecta compresiones previas a la expansion."""
    upper, mid, lower = bollinger(series, period, std_mult)
    return (upper - lower) / mid.replace(0, np.nan)


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(close, period)
    band = atr(high, low, close, period) * mult
    return mid + band, mid, mid - band


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    return upper, (upper + lower) / 2, lower


def realized_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    return np.log(series / series.shift(1)).rolling(period, min_periods=period).std(ddof=0)


def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Estimador de Parkinson: usa el rango, mas eficiente que el cierre a cierre."""
    factor = 1.0 / (4.0 * np.log(2.0))
    log_hl = np.log((high / low).replace(0, np.nan)) ** 2
    return np.sqrt(factor * log_hl.rolling(period, min_periods=period).mean())


# --------------------------------------------------------------------------- #
# Tendencia
# --------------------------------------------------------------------------- #
def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX con +DI y -DI. Mide fuerza de tendencia, no direccion."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_line, plus_di, minus_di


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """SuperTrend. Devuelve (linea, direccion) con direccion en {1, -1}."""
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = (hl2 + mult * atr_).to_numpy(dtype="float64")
    lower = (hl2 - mult * atr_).to_numpy(dtype="float64")
    close_arr = close.to_numpy(dtype="float64")

    n = len(close_arr)
    line = np.full(n, np.nan)
    direction = np.ones(n)

    start = int(np.argmax(~np.isnan(atr_.to_numpy())))
    if start == 0 and np.isnan(atr_.to_numpy()[0]):
        return pd.Series(line, index=close.index), pd.Series(direction, index=close.index)

    final_upper, final_lower = upper[start], lower[start]
    line[start], direction[start] = final_lower, 1.0

    for i in range(start + 1, n):
        # Las bandas solo se estrechan mientras la tendencia se mantiene.
        final_upper = upper[i] if (upper[i] < final_upper or close_arr[i - 1] > final_upper) else final_upper
        final_lower = lower[i] if (lower[i] > final_lower or close_arr[i - 1] < final_lower) else final_lower

        if direction[i - 1] == 1:
            direction[i] = -1.0 if close_arr[i] < final_lower else 1.0
        else:
            direction[i] = 1.0 if close_arr[i] > final_upper else -1.0
        line[i] = final_lower if direction[i] == 1 else final_upper

    return pd.Series(line, index=close.index), pd.Series(direction, index=close.index)


def linreg_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Pendiente de la regresion lineal, normalizada por precio.

    Se calcula en forma cerrada (no ``np.polyfit`` por ventana) porque el motor
    genetico evalua esto miles de veces.
    """
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(window: np.ndarray) -> float:
        return float(((x - x_mean) * (window - window.mean())).sum() / x_var)

    slope = series.rolling(period, min_periods=period).apply(_slope, raw=True)
    return slope / series.replace(0, np.nan)


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> tuple[pd.Series, pd.Series]:
    up = high.rolling(period + 1, min_periods=period + 1).apply(
        lambda x: float(np.argmax(x)) / period * 100, raw=True
    )
    down = low.rolling(period + 1, min_periods=period + 1).apply(
        lambda x: float(np.argmin(x)) / period * 100, raw=True
    )
    return up, down


def ichimoku(
    high: pd.Series, low: pd.Series, conversion: int = 9, base: int = 26, span_b: int = 52
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku sin desplazar hacia adelante: las nubes se dejan en su barra.

    Desplazar el span hacia el futuro es precisamente lo que introduce
    look-ahead en muchas implementaciones; aqui se evita.
    """
    def _mid(period: int) -> pd.Series:
        return (high.rolling(period, min_periods=period).max() + low.rolling(period, min_periods=period).min()) / 2

    tenkan = _mid(conversion)
    kijun = _mid(base)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = _mid(span_b)
    return tenkan, kijun, senkou_a, senkou_b


# --------------------------------------------------------------------------- #
# Volumen
# --------------------------------------------------------------------------- #
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    flow = typical * volume
    up = flow.where(typical > typical.shift(1), 0.0)
    down = flow.where(typical < typical.shift(1), 0.0)
    up_sum = up.rolling(period, min_periods=period).sum()
    down_sum = down.rolling(period, min_periods=period).sum()
    ratio = up_sum / down_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def vwap_session(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP que se reinicia cada dia (como hace cualquier mesa de trading)."""
    typical = (high + low + close) / 3
    day = close.index.normalize()
    pv = (typical * volume).groupby(day).cumsum()
    vol = volume.groupby(day).cumsum().replace(0, np.nan)
    return pv / vol


def volume_zscore(volume: pd.Series, period: int = 50) -> pd.Series:
    mean = volume.rolling(period, min_periods=period).mean()
    std = volume.rolling(period, min_periods=period).std(ddof=0).replace(0, np.nan)
    return (volume - mean) / std


# --------------------------------------------------------------------------- #
# Estructura y estadistica
# --------------------------------------------------------------------------- #
def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def percent_rank(series: pd.Series, period: int = 100) -> pd.Series:
    """Percentil del valor actual dentro de su ventana. Escala-invariante."""
    return series.rolling(period, min_periods=period).rank(pct=True)


def hurst_exponent(series: pd.Series, period: int = 100, max_lag: int = 20) -> pd.Series:
    """Exponente de Hurst por ventana.

    >0.5 sugiere persistencia (seguir tendencia); <0.5 antipersistencia
    (reversion). Es una senal ruidosa pero da contexto de regimen.
    """
    log_prices = np.log(series.replace(0, np.nan))
    lags = np.arange(2, max_lag + 1)
    log_lags = np.log(lags)
    log_lags_centered = log_lags - log_lags.mean()
    denom = (log_lags_centered**2).sum()

    def _hurst(window: np.ndarray) -> float:
        taus = np.empty(len(lags))
        for j, lag in enumerate(lags):
            diff = window[lag:] - window[:-lag]
            taus[j] = np.std(diff) if diff.size else np.nan
        if np.any(~np.isfinite(taus)) or np.any(taus <= 0):
            return np.nan
        log_taus = np.log(taus)
        return float((log_lags_centered * (log_taus - log_taus.mean())).sum() / denom)

    return log_prices.rolling(period, min_periods=period).apply(_hurst, raw=True)


def fractal_dimension(high: pd.Series, low: pd.Series, period: int = 30) -> pd.Series:
    """Indice de eficiencia de Kaufman sobre el rango: 0 = ruido, 1 = tendencia pura."""
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    net = (highest - lowest).abs()
    path = (high - low).rolling(period, min_periods=period).sum().replace(0, np.nan)
    return net / path


def candle_body_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Cuerpo / rango total. Cerca de 1 = vela decidida; cerca de 0 = doji."""
    rng = (high - low).replace(0, np.nan)
    return (close - open_).abs() / rng


def upper_wick_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    return (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng


def lower_wick_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    return (pd.concat([open_, close], axis=1).min(axis=1) - low) / rng


def rolling_sharpe(returns: pd.Series, period: int = 100) -> pd.Series:
    mean = returns.rolling(period, min_periods=period).mean()
    std = returns.rolling(period, min_periods=period).std(ddof=0).replace(0, np.nan)
    return mean / std


def distance_from_high(close: pd.Series, period: int = 100) -> pd.Series:
    highest = close.rolling(period, min_periods=period).max()
    return (close - highest) / highest.replace(0, np.nan)


def distance_from_low(close: pd.Series, period: int = 100) -> pd.Series:
    lowest = close.rolling(period, min_periods=period).min()
    return (close - lowest) / lowest.replace(0, np.nan)
