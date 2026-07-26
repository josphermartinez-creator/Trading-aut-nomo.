"""Funcion de fitness: que significa que una estrategia sea "buena".

Este es el fichero mas delicado del proyecto. El algoritmo genetico optimizara
*exactamente* lo que aqui se premie, incluidos los atajos que no se hayan
previsto. Una funcion de fitness ingenua (maximizar el retorno, o el Sharpe a
secas) produce siempre lo mismo: estrategias con tres operaciones afortunadas,
o con un drawdown que reventaria la cuenta antes de recuperarse.

Principios aplicados:

1. **Actividad minima obligatoria.** Menos de N operaciones = muestra sin valor
   estadistico, por muy bonita que sea la curva.
2. **El drawdown es multiplicativo, no un sumando.** No hay retorno que
   compense arruinarse.
3. **Se premia la regularidad**, no el total: R^2 de la equity y proporcion de
   meses en verde.
4. **Se penaliza la concentracion.** Si una sola operacion aporta la mitad del
   beneficio, eso fue suerte.
5. **Se penaliza la complejidad.** A igualdad de resultados, gana la regla
   simple: generaliza mejor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldbot.backtest.engine import BacktestEngine, BacktestResult
from goldbot.backtest.metrics import PerformanceMetrics
from goldbot.config import Config
from goldbot.strategies.genome import StrategyGenome
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)

# Puntuacion asignada a un genoma invalido o inoperante. Muy negativa para que
# nunca sobreviva a la seleccion, pero finita para no romper las estadisticas.
INVALID_SCORE = -100.0


@dataclass
class FitnessScore:
    """Puntuacion y su desglose, para poder auditar por que gano un genoma."""

    score: float = INVALID_SCORE
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 1.0
    profit_factor: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    equity_r2: float = 0.0
    concentration: float = 1.0
    complexity: int = 0
    penalties: dict[str, float] = field(default_factory=dict)
    valid: bool = False
    reason: str = ""

    def __lt__(self, other: FitnessScore) -> bool:
        return self.score < other.score

    def summary(self) -> str:
        if not self.valid:
            return f"INVALIDO ({self.reason})"
        return (
            f"score={self.score:.3f} sharpe={self.sharpe:.2f} "
            f"DD={self.max_drawdown:.1%} PF={self.profit_factor:.2f} "
            f"ops={self.trades} R2={self.equity_r2:.2f}"
        )


class FitnessEvaluator:
    """Evalua genomas y cachea resultados por huella estructural.

    El cache no es un lujo: en una poblacion de 120 individuos a lo largo de 40
    generaciones se repiten muchisimos genomas identicos (elitismo, mutaciones
    que revierten). Ahorra tipicamente entre el 30% y el 50% de los backtests.
    """

    def __init__(
        self,
        config: Config,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        engine: BacktestEngine | None = None,
        signal_filter: callable | None = None,
    ) -> None:
        self.config = config
        self.ohlcv = ohlcv
        self.features = features
        self.engine = engine or BacktestEngine(config)
        # Gancho para que la capa de ML module el tamano segun su confianza.
        self.signal_filter = signal_filter
        self._cache: dict[str, FitnessScore] = {}
        self.evaluations = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ #
    def evaluate(self, genome: StrategyGenome, use_cache: bool = True) -> FitnessScore:
        """Puntua un genoma."""
        fingerprint = genome.fingerprint()
        if use_cache and fingerprint in self._cache:
            self.cache_hits += 1
            return self._cache[fingerprint]

        try:
            result = self.backtest(genome)
            score = self.score_result(result, genome)
        except Exception as exc:
            logger.debug("Genoma %s fallo al evaluarse: %s", genome.genome_id, exc)
            score = FitnessScore(reason=f"excepcion: {type(exc).__name__}")

        self.evaluations += 1
        if use_cache:
            self._cache[fingerprint] = score
        return score

    def backtest(self, genome: StrategyGenome) -> BacktestResult:
        """Ejecuta el backtest del genoma sobre los datos del evaluador."""
        signals = genome.generate_signals(self.features, self.ohlcv)
        size_multiplier = None
        if self.signal_filter is not None:
            size_multiplier = self.signal_filter(signals, self.features, self.ohlcv)
        return self.engine.run(
            self.ohlcv,
            signals,
            genome.exit_rules,
            size_multiplier=size_multiplier,
            n_strategies_tried=max(1, self.evaluations),
        )

    # ------------------------------------------------------------------ #
    def score_result(self, result: BacktestResult, genome: StrategyGenome) -> FitnessScore:
        """Traduce un backtest a una unica puntuacion escalar."""
        m: PerformanceMetrics = result.metrics
        cfg = self.config
        min_trades = cfg.evolution.min_trades

        score = FitnessScore(
            sharpe=_finite(m.sharpe),
            total_return=_finite(m.total_return),
            max_drawdown=_finite(m.max_drawdown, 1.0),
            profit_factor=_finite(m.profit_factor, 0.0, cap=10.0),
            trades=m.total_trades,
            win_rate=_finite(m.win_rate),
            equity_r2=_finite(m.equity_r2),
            complexity=genome.complexity(),
        )

        # --- descalificaciones ------------------------------------------ #
        if m.total_trades == 0:
            score.reason = "sin operaciones"
            score.score = INVALID_SCORE
            return score

        if m.total_trades < min_trades:
            # Puntuacion negativa pero con pendiente: guia al GA hacia genomas
            # mas activos en lugar de dejarlo en una meseta plana.
            score.reason = f"muy pocas operaciones ({m.total_trades} < {min_trades})"
            score.score = INVALID_SCORE / 2 * (1 - m.total_trades / min_trades)
            return score

        if result.halted:
            score.reason = f"cortacircuitos: {result.halt_reason}"
            score.score = -10.0
            return score

        # --- concentracion de beneficios -------------------------------- #
        score.concentration = _profit_concentration(result.trades)
        if score.concentration > 0.6:
            score.reason = f"beneficio concentrado en pocas operaciones ({score.concentration:.0%})"
            score.score = -5.0
            return score

        # --- puntuacion base -------------------------------------------- #
        # Con Sharpe negativo devolvemos el propio Sharpe: mantiene un orden
        # coherente entre perdedores sin que los factores lo distorsionen.
        sharpe = score.sharpe
        if sharpe <= 0:
            score.valid = True
            score.reason = "sharpe no positivo"
            score.score = float(np.clip(sharpe, -10.0, 0.0))
            return score

        base = min(sharpe, 5.0)  # techo: por encima de 5 casi siempre es artefacto

        # --- factores multiplicativos ----------------------------------- #
        penalties: dict[str, float] = {}

        # Drawdown: cae a 0 al alcanzar el limite tolerado.
        dd_limit = max(cfg.stability.max_drawdown_pct, 1e-6)
        dd_factor = float(np.clip(1.0 - (score.max_drawdown / dd_limit) ** 1.5, 0.0, 1.0))
        penalties["drawdown"] = dd_factor

        # Profit factor: 1.0 es el punto muerto; por debajo no hay negocio.
        pf_factor = float(np.clip((score.profit_factor - 1.0) / 0.5, 0.0, 1.0))
        penalties["profit_factor"] = pf_factor

        # Regularidad: mezcla linealidad de la equity y meses ganadores.
        consistency = 0.6 * score.equity_r2 + 0.4 * _finite(m.monthly_win_rate)
        consistency_factor = float(np.clip(0.35 + 0.65 * consistency, 0.0, 1.0))
        penalties["consistency"] = consistency_factor

        # Concentracion: penaliza que el beneficio dependa de pocas operaciones.
        concentration_factor = float(np.clip(1.0 - score.concentration, 0.3, 1.0))
        penalties["concentration"] = concentration_factor

        # Actividad: sube hasta 4x el minimo de operaciones y ahi satura.
        activity_factor = float(np.clip(m.total_trades / (min_trades * 4), 0.5, 1.0))
        penalties["activity"] = activity_factor

        # Complejidad: -4% por condicion sobre las 3 primeras.
        excess_complexity = max(0, score.complexity - 3)
        complexity_factor = float(0.96**excess_complexity)
        penalties["complexity"] = complexity_factor

        # Coste de oportunidad: operar cada 5 minutos multiplica los costes y
        # la fragilidad frente a la latencia real.
        overtrading_factor = 1.0
        if m.trades_per_day > 12:
            overtrading_factor = float(np.clip(12 / m.trades_per_day, 0.4, 1.0))
        penalties["overtrading"] = overtrading_factor

        final = base
        for factor in penalties.values():
            final *= factor

        score.penalties = penalties
        score.score = float(final)
        score.valid = True
        score.reason = "ok"
        return score

    # ------------------------------------------------------------------ #
    @property
    def cache_efficiency(self) -> float:
        total = self.evaluations + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def clear_cache(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------------- #
def _finite(value: float, default: float = 0.0, cap: float | None = None) -> float:
    """Convierte NaN/inf en un valor utilizable."""
    if value is None or not np.isfinite(value):
        return default if cap is None else cap
    return float(value if cap is None else min(value, cap))


def _profit_concentration(trades: pd.DataFrame) -> float:
    """Fraccion del beneficio bruto que aportan las 5 mejores operaciones.

    Es el detector de suerte mas simple y eficaz que conozco: si de 200
    operaciones cinco explican el 70% del beneficio, la estrategia no tiene una
    ventaja, tuvo un buen dia.
    """
    if trades is None or len(trades) < 5:
        return 1.0

    pnl = trades["pnl"].to_numpy(dtype="float64")
    profits = pnl[pnl > 0]
    if profits.size == 0:
        return 1.0

    total_profit = float(profits.sum())
    if total_profit <= 0:
        return 1.0

    top_n = min(5, profits.size)
    top_profit = float(np.sort(profits)[-top_n:].sum())
    return float(np.clip(top_profit / total_profit, 0.0, 1.0))


def evaluate_population(
    genomes: list[StrategyGenome],
    evaluator: FitnessEvaluator,
    n_jobs: int = 1,
) -> list[FitnessScore]:
    """Evalua una poblacion entera.

    La paralelizacion con joblib solo compensa cuando el dataset es grande: por
    debajo, serializar el DataFrame a cada worker cuesta mas que el backtest.
    """
    if n_jobs == 1 or len(genomes) < 8:
        return [evaluator.evaluate(g) for g in genomes]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [evaluator.evaluate(g) for g in genomes]

    # El cache vive en el proceso padre: primero resolvemos los conocidos y solo
    # repartimos entre workers los genomas realmente nuevos.
    pending: list[tuple[int, StrategyGenome]] = []
    scores: list[FitnessScore | None] = [None] * len(genomes)

    for i, genome in enumerate(genomes):
        fingerprint = genome.fingerprint()
        cached = evaluator._cache.get(fingerprint)
        if cached is not None:
            evaluator.cache_hits += 1
            scores[i] = cached
        else:
            pending.append((i, genome))

    if pending:
        computed = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_evaluate_single)(evaluator, genome) for _, genome in pending
        )
        for (i, genome), score in zip(pending, computed, strict=True):
            scores[i] = score
            evaluator._cache[genome.fingerprint()] = score
            evaluator.evaluations += 1

    return [s if s is not None else FitnessScore(reason="no evaluado") for s in scores]


def _evaluate_single(evaluator: FitnessEvaluator, genome: StrategyGenome) -> FitnessScore:
    return evaluator.evaluate(genome, use_cache=False)
