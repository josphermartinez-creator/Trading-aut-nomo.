"""La puerta de estabilidad: que estrategia merece operar con dinero.

Este es el filtro que da sentido a "hasta conseguir la que se mantenga
estable". El motor evolutivo genera muchos candidatos con buen aspecto; casi
todos son ruido bien vestido. Aqui se les exige demostrar que no lo son.

Una estrategia debe superar **cinco pruebas independientes**, y basta fallar
una para quedar descartada. Son independientes a proposito: el sobreajuste
suele pasar una prueba por casualidad, rara vez cinco.

1. **Rendimiento absoluto**: Sharpe, profit factor, drawdown y numero de
   operaciones por encima de los minimos.
2. **Consistencia walk-forward**: gana en la mayoria de los tramos fuera de
   muestra, no solo en el agregado.
3. **Robustez Monte Carlo**: reordenando las operaciones, la probabilidad de
   ruina sigue siendo baja. El orden real fue una casualidad entre muchas.
4. **Estabilidad parametrica**: sigue funcionando al mover sus parametros. Si
   depende de que el stop sea 2.13 y no 2.30, se ajusto al ruido.
5. **Regularidad temporal**: la ganancia se reparte en el tiempo, no procede de
   un unico mes afortunado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldbot.backtest.engine import BacktestEngine, BacktestResult
from goldbot.backtest.metrics import PerformanceMetrics
from goldbot.backtest.walkforward import (
    MonteCarloResult,
    WalkForwardAnalyzer,
    WalkForwardResult,
    monte_carlo_analysis,
)
from goldbot.config import Config
from goldbot.strategies.genome import StrategyGenome
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """Resultado de una prueba individual."""

    name: str
    passed: bool
    detail: str
    value: float = 0.0
    threshold: float = 0.0

    def __str__(self) -> str:
        mark = "OK  " if self.passed else "FALLA"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class StabilityVerdict:
    """Veredicto completo sobre un candidato."""

    genome_id: str
    passed: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    score: float = 0.0
    metrics: PerformanceMetrics | None = None
    walkforward: WalkForwardResult | None = None
    montecarlo: MonteCarloResult | None = None
    parameter_stability: float = 0.0

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def rejection_reason(self) -> str:
        failed = self.failed_checks
        return "; ".join(c.detail for c in failed) if failed else ""

    def report(self) -> str:
        lines = [
            f"Veredicto de estabilidad para {self.genome_id}: "
            f"{'APTA' if self.passed else 'DESCARTADA'} (puntuacion {self.score:.3f})"
        ]
        lines.extend(f"  {check}" for check in self.checks)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "passed": self.passed,
            "score": self.score,
            "parameter_stability": self.parameter_stability,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail,
                 "value": c.value, "threshold": c.threshold}
                for c in self.checks
            ],
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "walkforward": {
                "folds": self.walkforward.n_folds,
                "profitable_ratio": self.walkforward.profitable_ratio,
                "mean_oos_sharpe": self.walkforward.mean_oos_sharpe,
                "mean_efficiency": self.walkforward.mean_efficiency,
            } if self.walkforward else {},
            "montecarlo": {
                "probability_of_ruin": self.montecarlo.probability_of_ruin,
                "drawdown_p95": self.montecarlo.drawdown_p95,
                "probability_of_loss": self.montecarlo.probability_of_loss,
            } if self.montecarlo else {},
        }


class StabilityGate:
    """Aplica las cinco pruebas a un candidato."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = BacktestEngine(config)
        self.analyzer = WalkForwardAnalyzer(config, self.engine)

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        genome: StrategyGenome,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        size_multiplier: pd.Series | None = None,
        run_walkforward: bool = True,
    ) -> StabilityVerdict:
        """Somete ``genome`` a todas las pruebas."""
        verdict = StabilityVerdict(genome_id=genome.genome_id)
        cfg = self.config.stability

        signals = genome.generate_signals(features, ohlcv)
        result = self.engine.run(ohlcv, signals, genome.exit_rules, size_multiplier=size_multiplier)
        verdict.metrics = result.metrics

        # --- 1) rendimiento absoluto --- #
        verdict.checks.extend(self._check_performance(result.metrics, result))

        # Sin operaciones suficientes el resto de pruebas carece de sentido.
        if result.metrics.total_trades < cfg.min_trades:
            verdict.passed = False
            verdict.score = 0.0
            logger.info("%s", verdict.report())
            return verdict

        # --- 2) consistencia walk-forward --- #
        if run_walkforward:
            wf = self.analyzer.run(genome, ohlcv, features)
            verdict.walkforward = wf
            verdict.checks.extend(self._check_walkforward(wf))

        # --- 3) robustez Monte Carlo --- #
        mc = monte_carlo_analysis(
            trades=result.trades,
            initial_balance=self.config.risk.initial_balance,
            runs=self.config.backtest.monte_carlo_runs,
            ruin_threshold=cfg.max_drawdown_pct,
        )
        verdict.montecarlo = mc
        verdict.checks.extend(self._check_montecarlo(mc))

        # --- 4) estabilidad parametrica --- #
        stability, check = self._check_parameter_stability(genome, ohlcv, features)
        verdict.parameter_stability = stability
        verdict.checks.append(check)

        # --- 5) regularidad temporal --- #
        verdict.checks.append(self._check_temporal_consistency(result))

        verdict.passed = all(c.passed for c in verdict.checks)
        verdict.score = self._composite_score(verdict)

        logger.info("%s", verdict.report())
        return verdict

    # ------------------------------------------------------------------ #
    def _check_performance(
        self, m: PerformanceMetrics, result: BacktestResult
    ) -> list[CheckResult]:
        cfg = self.config.stability
        checks: list[CheckResult] = []

        checks.append(CheckResult(
            "operaciones", m.total_trades >= cfg.min_trades,
            f"{m.total_trades} operaciones (minimo {cfg.min_trades})",
            float(m.total_trades), float(cfg.min_trades),
        ))

        sharpe_ok = np.isfinite(m.sharpe) and m.sharpe >= cfg.min_sharpe
        checks.append(CheckResult(
            "sharpe", sharpe_ok,
            f"Sharpe {m.sharpe:.2f} (minimo {cfg.min_sharpe})",
            float(m.sharpe), cfg.min_sharpe,
        ))

        pf_ok = np.isfinite(m.profit_factor) and m.profit_factor >= cfg.min_profit_factor
        checks.append(CheckResult(
            "profit_factor", pf_ok,
            f"PF {m.profit_factor:.2f} (minimo {cfg.min_profit_factor})",
            float(m.profit_factor) if np.isfinite(m.profit_factor) else 0.0,
            cfg.min_profit_factor,
        ))

        checks.append(CheckResult(
            "drawdown", m.max_drawdown <= cfg.max_drawdown_pct,
            f"drawdown maximo {m.max_drawdown:.1%} (limite {cfg.max_drawdown_pct:.1%})",
            m.max_drawdown, cfg.max_drawdown_pct,
        ))

        checks.append(CheckResult(
            "tasa_acierto", m.win_rate >= cfg.min_win_rate,
            f"aciertos {m.win_rate:.1%} (minimo {cfg.min_win_rate:.1%})",
            m.win_rate, cfg.min_win_rate,
        ))

        checks.append(CheckResult(
            "sin_cortacircuito", not result.halted,
            "no se activo el cortacircuitos" if not result.halted else f"detenida: {result.halt_reason}",
        ))

        return checks

    def _check_walkforward(self, wf: WalkForwardResult) -> list[CheckResult]:
        cfg = self.config.stability
        checks: list[CheckResult] = []

        if wf.n_folds == 0:
            checks.append(CheckResult(
                "walkforward", False, "no se pudo ejecutar el analisis walk-forward"
            ))
            return checks

        checks.append(CheckResult(
            "wf_tramos_rentables", wf.profitable_ratio >= cfg.min_folds_profitable,
            f"{wf.profitable_folds}/{wf.n_folds} tramos OOS en verde "
            f"({wf.profitable_ratio:.0%}, minimo {cfg.min_folds_profitable:.0%})",
            wf.profitable_ratio, cfg.min_folds_profitable,
        ))

        # Eficiencia: cuanto del rendimiento in-sample sobrevive fuera de muestra.
        efficiency_ok = wf.mean_efficiency >= self.config.backtest.min_oos_ratio
        checks.append(CheckResult(
            "wf_eficiencia", efficiency_ok,
            f"eficiencia OOS/IS {wf.mean_efficiency:.2f} "
            f"(minimo {self.config.backtest.min_oos_ratio})",
            wf.mean_efficiency, self.config.backtest.min_oos_ratio,
        ))

        dispersion_ok = wf.oos_return_std <= cfg.max_return_std_across_folds
        checks.append(CheckResult(
            "wf_dispersion", dispersion_ok,
            f"dispersion entre tramos {wf.oos_return_std:.2f} "
            f"(maximo {cfg.max_return_std_across_folds})",
            wf.oos_return_std, cfg.max_return_std_across_folds,
        ))

        return checks

    def _check_montecarlo(self, mc: MonteCarloResult) -> list[CheckResult]:
        checks: list[CheckResult] = []

        if mc.runs == 0:
            checks.append(CheckResult("montecarlo", False, "operaciones insuficientes para Monte Carlo"))
            return checks

        # Un 5% de probabilidad de ruina es mucho: significa 1 de cada 20
        # universos posibles termina en el limite de drawdown.
        ruin_ok = mc.probability_of_ruin <= 0.05
        checks.append(CheckResult(
            "mc_riesgo_ruina", ruin_ok,
            f"probabilidad de ruina {mc.probability_of_ruin:.1%} (maximo 5%)",
            mc.probability_of_ruin, 0.05,
        ))

        loss_ok = mc.probability_of_loss <= 0.30
        checks.append(CheckResult(
            "mc_probabilidad_perdida", loss_ok,
            f"probabilidad de acabar en perdidas {mc.probability_of_loss:.1%} (maximo 30%)",
            mc.probability_of_loss, 0.30,
        ))

        dd_ok = mc.drawdown_p95 <= self.config.stability.max_drawdown_pct * 1.5
        checks.append(CheckResult(
            "mc_drawdown_p95", dd_ok,
            f"drawdown p95 {mc.drawdown_p95:.1%} "
            f"(maximo {self.config.stability.max_drawdown_pct * 1.5:.1%})",
            mc.drawdown_p95, self.config.stability.max_drawdown_pct * 1.5,
        ))

        return checks

    def _check_parameter_stability(
        self, genome: StrategyGenome, ohlcv: pd.DataFrame, features: pd.DataFrame
    ) -> tuple[float, CheckResult]:
        """Perturba los parametros de salida y mide cuanto aguanta.

        Solo se tocan las reglas de salida (stop, objetivo, trailing, tiempo):
        son continuas y su efecto es directo. Perturbar los umbrales de entrada
        cambiaria el numero de operaciones y mezclaria dos efectos distintos.
        """
        from goldbot.backtest.engine import ExitRules

        baseline_signals = genome.generate_signals(features, ohlcv)
        baseline = self.engine.run(ohlcv, baseline_signals, genome.exit_rules)
        baseline_sharpe = baseline.metrics.sharpe

        if not np.isfinite(baseline_sharpe) or baseline_sharpe <= 0:
            return 0.0, CheckResult(
                "estabilidad_parametrica", False, "Sharpe base no positivo; no evaluable"
            )

        rng = np.random.default_rng(2024)
        sharpes: list[float] = []
        base = genome.exit_rules

        for _ in range(10):
            perturbed = ExitRules(
                stop_atr=base.stop_atr * float(rng.uniform(0.85, 1.15)),
                target_atr=base.target_atr * float(rng.uniform(0.85, 1.15)),
                trail_atr=base.trail_atr * float(rng.uniform(0.85, 1.15)) if base.trail_atr else 0.0,
                breakeven_atr=(
                    base.breakeven_atr * float(rng.uniform(0.85, 1.15)) if base.breakeven_atr else 0.0
                ),
                max_bars=int(base.max_bars * float(rng.uniform(0.85, 1.15))),
                exit_on_reverse=base.exit_on_reverse,
            ).validate()

            try:
                perturbed_result = self.engine.run(ohlcv, baseline_signals, perturbed)
                value = perturbed_result.metrics.sharpe
                sharpes.append(float(value) if np.isfinite(value) else 0.0)
            except Exception:
                sharpes.append(0.0)

        mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
        stability = float(np.clip(mean_sharpe / baseline_sharpe, 0.0, 2.0))

        # Se exige conservar al menos el 60% del rendimiento en el vecindario.
        passed = stability >= 0.6
        return stability, CheckResult(
            "estabilidad_parametrica", passed,
            f"conserva el {stability:.0%} del Sharpe al perturbar +-15% (minimo 60%)",
            stability, 0.6,
        )

    def _check_temporal_consistency(self, result: BacktestResult) -> CheckResult:
        """Comprueba que la ganancia no proceda de un unico periodo."""
        m = result.metrics

        if m.total_return <= 0:
            return CheckResult("regularidad_temporal", False, "rendimiento total no positivo")

        # R^2 de la equity: mide lo recta que es la curva de capital.
        r2_ok = m.equity_r2 >= 0.55
        monthly_ok = m.monthly_win_rate >= 0.5

        passed = r2_ok and monthly_ok
        return CheckResult(
            "regularidad_temporal", passed,
            f"R2 de la equity {m.equity_r2:.2f} (minimo 0.55), "
            f"meses en verde {m.monthly_win_rate:.0%} (minimo 50%)",
            m.equity_r2, 0.55,
        )

    # ------------------------------------------------------------------ #
    def _composite_score(self, verdict: StabilityVerdict) -> float:
        """Puntuacion final para ordenar entre candidatos que ya pasaron la puerta.

        Se pondera lo que predice el comportamiento futuro (rendimiento fuera de
        muestra y robustez) por encima del resultado historico agregado.
        """
        if verdict.metrics is None:
            return 0.0

        m = verdict.metrics
        sharpe = float(np.clip(m.sharpe, 0, 4)) / 4

        wf_component = 0.0
        if verdict.walkforward and verdict.walkforward.n_folds:
            wf_component = float(
                0.6 * verdict.walkforward.profitable_ratio
                + 0.4 * np.clip(verdict.walkforward.mean_efficiency, 0, 1)
            )

        mc_component = 0.0
        if verdict.montecarlo and verdict.montecarlo.runs:
            mc_component = float(
                0.5 * (1 - verdict.montecarlo.probability_of_ruin)
                + 0.5 * (1 - verdict.montecarlo.probability_of_loss)
            )

        stability_component = float(np.clip(verdict.parameter_stability, 0, 1))
        consistency_component = float(np.clip(m.equity_r2, 0, 1))
        drawdown_component = float(
            np.clip(1 - m.max_drawdown / max(self.config.stability.max_drawdown_pct, 1e-6), 0, 1)
        )

        return float(
            0.20 * sharpe
            + 0.25 * wf_component
            + 0.20 * mc_component
            + 0.15 * stability_component
            + 0.10 * consistency_component
            + 0.10 * drawdown_component
        )
