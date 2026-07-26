"""Metricas de rendimiento y riesgo.

Se calculan sobre la curva de equity y la lista de operaciones. Las metricas
que se usan para *seleccionar* estrategias estan pensadas para castigar la
suerte: Sharpe deflactado, consistencia mensual y estabilidad de la curva.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from goldbot.utils.timeutils import annualization_factor


@dataclass
class PerformanceMetrics:
    """Resumen completo de una simulacion."""

    # Rentabilidad
    total_return: float = 0.0
    annual_return: float = 0.0
    final_equity: float = 0.0

    # Riesgo
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    downside_deviation: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0

    # Ratios
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    deflated_sharpe: float = 0.0

    # Operativa
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    max_consecutive_losses: int = 0
    avg_bars_held: float = 0.0
    trades_per_day: float = 0.0

    # Estabilidad
    equity_r2: float = 0.0
    monthly_win_rate: float = 0.0
    return_consistency: float = 0.0
    ulcer_index: float = 0.0

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"ret={self.total_return:+.2%} sharpe={self.sharpe:.2f} "
            f"maxDD={self.max_drawdown:.2%} PF={self.profit_factor:.2f} "
            f"trades={self.total_trades} win={self.win_rate:.1%} R2={self.equity_r2:.2f}"
        )


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    initial_balance: float = 10_000.0,
    timeframe_minutes: int = 5,
    risk_free_rate: float = 0.0,
    n_strategies_tried: int = 1,
) -> PerformanceMetrics:
    """Calcula todas las metricas a partir de la curva de equity y las operaciones.

    ``n_strategies_tried`` alimenta el Sharpe deflactado: si has probado 5.000
    estrategias, la mejor tiene un Sharpe alto por puro azar, y hay que
    descontarlo.
    """
    m = PerformanceMetrics()

    if equity is None or len(equity) < 2:
        m.notes.append("curva de equity insuficiente")
        return m

    equity = equity.astype("float64")
    m.final_equity = float(equity.iloc[-1])
    m.total_return = float(equity.iloc[-1] / initial_balance - 1.0)

    returns = equity.pct_change().fillna(0.0)
    returns = returns.replace([np.inf, -np.inf], 0.0)

    ann_factor = annualization_factor(timeframe_minutes)
    bars_per_year = ann_factor**2

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))

    m.volatility = std_r * ann_factor
    n_bars = len(returns)
    years = max(n_bars / bars_per_year, 1e-9)

    # Retorno anualizado compuesto; si la cuenta se arruina, -100%.
    if m.final_equity > 0:
        m.annual_return = float((m.final_equity / initial_balance) ** (1 / years) - 1.0)
    else:
        m.annual_return = -1.0

    # --- ratios ---
    if std_r > 0:
        m.sharpe = float((mean_r - risk_free_rate / bars_per_year) / std_r * ann_factor)

    downside = returns[returns < 0]
    if len(downside) > 1:
        dd_std = float(downside.std(ddof=1))
        m.downside_deviation = dd_std * ann_factor
        if dd_std > 0:
            m.sortino = float(mean_r / dd_std * ann_factor)

    # --- drawdown ---
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    drawdown = drawdown.fillna(0.0)
    m.max_drawdown = float(abs(drawdown.min()))

    if m.max_drawdown > 1e-9:
        m.calmar = float(m.annual_return / m.max_drawdown)

    m.max_drawdown_duration_bars = _max_drawdown_duration(equity)
    # Ulcer Index: penaliza drawdowns profundos Y prolongados, no solo el peor.
    m.ulcer_index = float(np.sqrt(np.mean(np.square(drawdown * 100))))

    # --- cola de perdidas ---
    if len(returns) > 20:
        m.value_at_risk_95 = float(np.percentile(returns, 5))
        tail = returns[returns <= m.value_at_risk_95]
        if len(tail):
            m.conditional_var_95 = float(tail.mean())

    # --- estabilidad de la curva ---
    m.equity_r2 = _equity_linearity(equity)
    m.monthly_win_rate, m.return_consistency = _monthly_consistency(equity)

    # --- estadisticas de operaciones ---
    if trades is not None and len(trades) > 0:
        _fill_trade_metrics(m, trades, n_bars, timeframe_minutes)

    # --- Sharpe deflactado ---
    m.deflated_sharpe = _deflated_sharpe(returns, m.sharpe, n_strategies_tried, ann_factor)

    return m


def _fill_trade_metrics(
    m: PerformanceMetrics, trades: pd.DataFrame, n_bars: int, timeframe_minutes: int
) -> None:
    pnl = trades["pnl"].to_numpy(dtype="float64")
    m.total_trades = int(len(pnl))

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    m.win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0
    m.avg_win = float(wins.mean()) if len(wins) else 0.0
    m.avg_loss = float(losses.mean()) if len(losses) else 0.0
    m.expectancy = float(pnl.mean())

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    if gross_loss > 0:
        m.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        m.profit_factor = float("inf")

    if m.avg_loss != 0:
        m.payoff_ratio = float(abs(m.avg_win / m.avg_loss))

    m.max_consecutive_losses = _max_consecutive(pnl < 0)

    if "bars_held" in trades.columns:
        m.avg_bars_held = float(trades["bars_held"].mean())

    days = max(n_bars * timeframe_minutes / (60 * 24), 1e-9)
    m.trades_per_day = float(m.total_trades / days)


def _max_consecutive(mask: np.ndarray) -> int:
    """Racha maxima de ``True`` consecutivos."""
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)


def _max_drawdown_duration(equity: pd.Series) -> int:
    """Barras que tarda la equity en recuperar un maximo previo."""
    values = equity.to_numpy(dtype="float64")
    peak = values[0]
    peak_idx = 0
    longest = 0
    for i, value in enumerate(values):
        if value >= peak:
            peak, peak_idx = value, i
        else:
            longest = max(longest, i - peak_idx)
    return int(longest)


def _equity_linearity(equity: pd.Series) -> float:
    """R^2 de la equity contra el tiempo.

    Una estrategia estable sube en linea recta. Un R^2 alto con pendiente
    negativa no vale nada, asi que devolvemos 0 en ese caso.
    """
    if len(equity) < 10:
        return 0.0
    y = equity.to_numpy(dtype="float64")
    x = np.arange(len(y), dtype="float64")
    try:
        result = stats.linregress(x, y)
    except Exception:
        return 0.0
    if result.slope <= 0:
        return 0.0
    return float(result.rvalue**2)


def _monthly_consistency(equity: pd.Series) -> tuple[float, float]:
    """(% de meses en verde, consistencia). Detecta el 'un mes lo gano todo'."""
    if not isinstance(equity.index, pd.DatetimeIndex) or len(equity) < 30:
        return 0.0, 0.0

    monthly = equity.resample("ME").last().dropna()
    if len(monthly) < 2:
        return 0.0, 0.0

    monthly_returns = monthly.pct_change().dropna()
    if monthly_returns.empty:
        return 0.0, 0.0

    win_rate = float((monthly_returns > 0).mean())

    mean_r = float(monthly_returns.mean())
    std_r = float(monthly_returns.std(ddof=1)) if len(monthly_returns) > 1 else 0.0
    # Consistencia = media/desviacion, acotada. Alta => meses parecidos entre si.
    consistency = float(np.clip(mean_r / std_r, -3, 3)) if std_r > 1e-12 else (3.0 if mean_r > 0 else 0.0)
    return win_rate, consistency


def _deflated_sharpe(
    returns: pd.Series, sharpe: float, n_trials: int, ann_factor: float
) -> float:
    """Sharpe deflactado (Bailey & Lopez de Prado, simplificado).

    Corrige por (a) el numero de estrategias probadas y (b) la asimetria y
    curtosis de los retornos. Es la metrica honesta cuando un GA ha explorado
    miles de combinaciones.
    """
    n = len(returns)
    if n < 30 or not np.isfinite(sharpe) or n_trials < 1:
        return 0.0

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))

    sr_per_bar = sharpe / ann_factor

    # Umbral esperado del maximo Sharpe bajo la hipotesis nula tras N intentos.
    if n_trials > 1:
        euler = 0.5772156649
        expected_max = (1 - euler) * stats.norm.ppf(1 - 1 / n_trials) + euler * stats.norm.ppf(
            1 - 1 / (n_trials * np.e)
        )
    else:
        expected_max = 0.0

    denominator = 1 - skew * sr_per_bar + (kurt - 1) / 4 * sr_per_bar**2
    if denominator <= 0:
        return 0.0

    numerator = (sr_per_bar - expected_max / np.sqrt(n)) * np.sqrt(n - 1)
    z = numerator / np.sqrt(denominator)
    return float(stats.norm.cdf(z))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Serie de drawdown relativo, util para graficar."""
    running_max = equity.cummax()
    return (equity - running_max) / running_max.replace(0, np.nan)
