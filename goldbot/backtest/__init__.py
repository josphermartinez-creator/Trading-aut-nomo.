"""Motor de backtesting vectorizado, metricas y validacion walk-forward."""

from goldbot.backtest.costs import CostModel
from goldbot.backtest.engine import BacktestEngine, BacktestResult, Trade
from goldbot.backtest.metrics import PerformanceMetrics, compute_metrics
from goldbot.backtest.walkforward import WalkForwardAnalyzer, WalkForwardResult

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "PerformanceMetrics",
    "Trade",
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    "compute_metrics",
]
