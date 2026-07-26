"""Validacion walk-forward y Monte Carlo.

Un backtest sobre todo el historico no dice casi nada: la estrategia ya ha
"visto" esos datos durante la evolucion. Lo que importa es como se comporta en
tramos que no participaron en su seleccion, y con que consistencia lo hace.

Dos herramientas:

* **Walk-forward**: trocea el historico en ventanas sucesivas train/test y
  evalua siempre fuera de muestra. La consistencia entre tramos importa mas
  que el resultado agregado.
* **Monte Carlo**: remuestrea las operaciones para estimar la distribucion de
  drawdowns posibles. La curva observada es solo *una* realizacion; lo que
  hunde cuentas es la cola.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldbot.backtest.engine import BacktestEngine, BacktestResult
from goldbot.backtest.metrics import PerformanceMetrics, compute_metrics
from goldbot.config import Config
from goldbot.strategies.base import Strategy
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Resultado de un unico tramo walk-forward."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics

    @property
    def efficiency(self) -> float:
        """Ratio OOS/IS del retorno anualizado.

        Cerca de 1 significa que la estrategia se comporta fuera de muestra
        como dentro. Muy por debajo de 0.5 huele a sobreajuste.
        """
        is_ret = self.in_sample.annual_return
        oos_ret = self.out_of_sample.annual_return
        if abs(is_ret) < 1e-9:
            return 0.0
        return float(oos_ret / is_ret)


@dataclass
class WalkForwardResult:
    """Agregado de todos los tramos."""

    folds: list[FoldResult] = field(default_factory=list)
    combined_oos: PerformanceMetrics | None = None
    combined_equity: pd.Series | None = None

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def profitable_folds(self) -> int:
        return sum(1 for f in self.folds if f.out_of_sample.total_return > 0)

    @property
    def profitable_ratio(self) -> float:
        return self.profitable_folds / self.n_folds if self.n_folds else 0.0

    @property
    def mean_efficiency(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.efficiency for f in self.folds]))

    @property
    def oos_return_std(self) -> float:
        """Dispersion de los retornos OOS entre tramos: mide la fiabilidad."""
        if len(self.folds) < 2:
            return 0.0
        return float(np.std([f.out_of_sample.total_return for f in self.folds], ddof=1))

    @property
    def mean_oos_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        values = [f.out_of_sample.sharpe for f in self.folds]
        return float(np.mean([v for v in values if np.isfinite(v)] or [0.0]))

    @property
    def worst_oos_drawdown(self) -> float:
        if not self.folds:
            return 1.0
        return float(max(f.out_of_sample.max_drawdown for f in self.folds))

    def summary(self) -> str:
        return (
            f"WF: {self.profitable_folds}/{self.n_folds} tramos en verde | "
            f"Sharpe OOS medio={self.mean_oos_sharpe:.2f} | "
            f"eficiencia={self.mean_efficiency:.2f} | "
            f"peor DD OOS={self.worst_oos_drawdown:.1%}"
        )


class WalkForwardAnalyzer:
    """Ejecuta el analisis walk-forward de una estrategia."""

    def __init__(self, config: Config, engine: BacktestEngine | None = None) -> None:
        self.config = config
        self.engine = engine or BacktestEngine(config)

    # ------------------------------------------------------------------ #
    def make_folds(self, n_bars: int) -> list[tuple[slice, slice]]:
        """Ventanas deslizantes (train, test) con purga entre ambas.

        Se usa ventana deslizante y no anclada: en intradia el pasado muy
        lejano tiene un regimen distinto y contamina mas de lo que aporta.
        """
        cfg = self.config.backtest
        n_folds = max(1, cfg.walk_forward_folds)
        purge = cfg.purge_bars

        # Cada tramo = train + test; los tramos avanzan por el tamano del test.
        total_windows = n_folds + 1
        window = n_bars // total_windows
        if window < 200:
            logger.warning("Historico corto (%d barras): se reduce a 2 tramos", n_bars)
            n_folds = 2
            total_windows = 3
            window = n_bars // total_windows

        train_size = int(window * total_windows * cfg.train_ratio / n_folds)
        train_size = max(train_size, window)
        test_size = window

        folds: list[tuple[slice, slice]] = []
        for k in range(n_folds):
            test_end = n_bars - (n_folds - 1 - k) * test_size
            test_start = test_end - test_size
            train_end = test_start - purge
            train_start = max(0, train_end - train_size)

            if train_end - train_start < 200 or test_end - test_start < 50:
                continue
            folds.append((slice(train_start, train_end), slice(test_start, test_end)))

        return folds

    # ------------------------------------------------------------------ #
    def run(
        self,
        strategy: Strategy,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        refit: callable | None = None,
    ) -> WalkForwardResult:
        """Evalua ``strategy`` tramo a tramo.

        ``refit`` permite reajustar la estrategia con los datos de train de
        cada tramo (es lo que usa la capa de ML para reentrenar su filtro sin
        contaminar el test). Si es ``None``, la estrategia se evalua fija.
        """
        result = WalkForwardResult()
        n_bars = len(ohlcv)
        if n_bars < 500:
            logger.warning("Datos insuficientes para walk-forward (%d barras)", n_bars)
            return result

        folds = self.make_folds(n_bars)
        if not folds:
            logger.warning("No se pudo construir ningun tramo walk-forward")
            return result

        oos_equities: list[pd.Series] = []

        for k, (train_slice, test_slice) in enumerate(folds):
            train_ohlcv = ohlcv.iloc[train_slice]
            train_features = features.iloc[train_slice]
            test_ohlcv = ohlcv.iloc[test_slice]
            test_features = features.iloc[test_slice]

            fold_strategy = strategy
            if refit is not None:
                try:
                    fold_strategy = refit(strategy, train_ohlcv, train_features)
                except Exception as exc:
                    logger.warning("Reajuste fallido en el tramo %d: %s", k, exc)
                    fold_strategy = strategy

            is_result = self._evaluate(fold_strategy, train_ohlcv, train_features)
            oos_result = self._evaluate(fold_strategy, test_ohlcv, test_features)

            result.folds.append(
                FoldResult(
                    fold=k,
                    train_start=train_ohlcv.index[0],
                    train_end=train_ohlcv.index[-1],
                    test_start=test_ohlcv.index[0],
                    test_end=test_ohlcv.index[-1],
                    in_sample=is_result.metrics,
                    out_of_sample=oos_result.metrics,
                )
            )
            oos_equities.append(oos_result.equity)

        result.combined_equity = _chain_equities(oos_equities, self.config.risk.initial_balance)
        if result.combined_equity is not None and len(result.combined_equity) > 1:
            result.combined_oos = compute_metrics(
                equity=result.combined_equity,
                trades=None,
                initial_balance=self.config.risk.initial_balance,
                timeframe_minutes=self.config.data.timeframe_minutes,
            )

        logger.debug("%s", result.summary())
        return result

    def _evaluate(self, strategy: Strategy, ohlcv: pd.DataFrame, features: pd.DataFrame) -> BacktestResult:
        signals = strategy.generate_signals(features, ohlcv)
        confidence = strategy.confidence(features, ohlcv)
        exit_rules = getattr(strategy, "exit_rules", None)
        return self.engine.run(ohlcv, signals, exit_rules, size_multiplier=confidence)


def _chain_equities(equities: list[pd.Series], initial_balance: float) -> pd.Series | None:
    """Encadena las curvas OOS de cada tramo en una sola curva continua.

    Cada tramo arranca con el capital que dejo el anterior, de modo que la
    curva resultante refleja el efecto compuesto real de operar la estrategia
    de forma continuada.
    """
    if not equities:
        return None

    chained: list[pd.Series] = []
    capital = initial_balance
    for equity in equities:
        if equity is None or len(equity) < 2:
            continue
        scaled = equity / equity.iloc[0] * capital
        chained.append(scaled)
        capital = float(scaled.iloc[-1])

    if not chained:
        return None
    out = pd.concat(chained)
    return out[~out.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass
class MonteCarloResult:
    """Distribucion de resultados bajo remuestreo de operaciones."""

    runs: int
    median_return: float
    return_p05: float
    return_p95: float
    median_drawdown: float
    drawdown_p95: float          # el drawdown "malo pero posible"
    worst_drawdown: float
    probability_of_loss: float
    probability_of_ruin: float   # perder mas del limite de drawdown configurado
    risk_of_ruin_threshold: float

    def summary(self) -> str:
        return (
            f"MC({self.runs}): retorno mediano={self.median_return:+.1%} "
            f"[p05={self.return_p05:+.1%}] | DD p95={self.drawdown_p95:.1%} "
            f"| P(perdida)={self.probability_of_loss:.1%} "
            f"| P(ruina)={self.probability_of_ruin:.1%}"
        )


def monte_carlo_analysis(
    trades: pd.DataFrame,
    initial_balance: float = 10_000.0,
    runs: int = 500,
    ruin_threshold: float = 0.20,
    seed: int = 42,
) -> MonteCarloResult:
    """Bootstrap del orden de las operaciones.

    Se remuestrea con reemplazo la secuencia de P&L. La rentabilidad final
    apenas cambia, pero el drawdown maximo varia enormemente: es exactamente la
    incertidumbre que un backtest unico esconde.
    """
    empty = MonteCarloResult(0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, ruin_threshold)
    if trades is None or len(trades) < 10:
        return empty

    pnl = trades["pnl"].to_numpy(dtype="float64")
    n = len(pnl)
    rng = np.random.default_rng(seed)

    final_returns = np.empty(runs)
    max_drawdowns = np.empty(runs)

    for i in range(runs):
        sample = rng.choice(pnl, size=n, replace=True)
        equity = initial_balance + np.cumsum(sample)

        final_returns[i] = equity[-1] / initial_balance - 1.0

        running_max = np.maximum.accumulate(np.maximum(equity, 1e-9))
        drawdown = (running_max - equity) / running_max
        max_drawdowns[i] = float(drawdown.max())

    return MonteCarloResult(
        runs=runs,
        median_return=float(np.median(final_returns)),
        return_p05=float(np.percentile(final_returns, 5)),
        return_p95=float(np.percentile(final_returns, 95)),
        median_drawdown=float(np.median(max_drawdowns)),
        drawdown_p95=float(np.percentile(max_drawdowns, 95)),
        worst_drawdown=float(max_drawdowns.max()),
        probability_of_loss=float((final_returns < 0).mean()),
        probability_of_ruin=float((max_drawdowns >= ruin_threshold).mean()),
        risk_of_ruin_threshold=ruin_threshold,
    )


def parameter_sensitivity(
    evaluate_fn: callable,
    base_params: dict[str, float],
    perturbation: float = 0.15,
    n_samples: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Robustez ante perturbaciones de los parametros.

    Una estrategia sana sigue funcionando si mueves su stop un 15%. Si el
    rendimiento se desploma, el optimizador encontro un pico estrecho de ruido
    y no una regularidad del mercado.
    """
    rng = np.random.default_rng(seed)
    base_score = evaluate_fn(base_params)
    scores = []

    for _ in range(n_samples):
        perturbed = {
            key: float(value * (1 + rng.uniform(-perturbation, perturbation)))
            for key, value in base_params.items()
        }
        try:
            scores.append(evaluate_fn(perturbed))
        except Exception:
            scores.append(float("-inf"))

    finite = [s for s in scores if np.isfinite(s)]
    if not finite:
        return {"base_score": base_score, "mean_score": 0.0, "degradation": 1.0, "stability": 0.0}

    mean_score = float(np.mean(finite))
    std_score = float(np.std(finite))
    degradation = float((base_score - mean_score) / abs(base_score)) if abs(base_score) > 1e-9 else 1.0

    return {
        "base_score": float(base_score),
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": float(np.min(finite)),
        "degradation": degradation,
        # Estabilidad alta = el vecindario rinde parecido al optimo.
        "stability": float(np.clip(1.0 - abs(degradation), 0.0, 1.0)),
    }
