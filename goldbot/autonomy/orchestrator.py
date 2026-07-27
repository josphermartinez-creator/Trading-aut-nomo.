"""El ciclo diario de aprendizaje: el cerebro autonomo del bot.

Se ejecuta una vez al dia (tipicamente tras el cierre de Nueva York) y encadena
siempre las mismas etapas:

    1. Actualizar datos          -> el historico crece cada dia
    2. Recalcular features       -> con el mercado mas reciente incluido
    3. Vigilar deriva            -> ¿ha cambiado el regimen? ¿se degrado el campeon?
    4. Descubrir (si toca)       -> el genetico inventa estrategias nuevas
    5. Refinar con Optuna        -> ajuste fino de los finalistas
    6. Puerta de estabilidad     -> cinco pruebas independientes
    7. Reentrenar el meta-modelo -> el filtro de ML del campeon
    8. Gestionar la incubadora   -> quien entra, quien sale
    9. Promover / degradar       -> quien opera manana

Cada etapa es independiente y falla de forma aislada: si el descubrimiento
revienta, el resto del ciclo continua y el campeon sigue operando. Un bot
autonomo que se cae entero porque una etapa fallo no es autonomo.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import date

from goldbot.autonomy.champion import ChampionRegistry, PromotionDecision
from goldbot.autonomy.stability import StabilityGate
from goldbot.backtest.engine import BacktestEngine
from goldbot.config import Config
from goldbot.data.pipeline import MarketData
from goldbot.evolution.genetic import GeneticEngine
from goldbot.features.engineering import FeatureBuilder, build_features
from goldbot.ml.drift import DriftDetector
from goldbot.ml.trainer import MLTrainer
from goldbot.storage.db import Database
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import now_utc

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Resultado de una etapa del ciclo."""

    name: str
    ok: bool
    detail: str = ""
    elapsed: float = 0.0
    error: str = ""

    def __str__(self) -> str:
        mark = "OK " if self.ok else "ERR"
        return f"[{mark}] {self.name} ({self.elapsed:.1f}s): {self.detail or self.error}"


@dataclass
class DailyReport:
    """Informe del ciclo diario completo."""

    run_date: str
    started_at: str
    stages: list[StageResult] = field(default_factory=list)
    bars_available: int = 0
    strategies_discovered: int = 0
    strategies_validated: int = 0
    promotion: PromotionDecision | None = None
    drift_recommendation: str = "sin_accion"
    champion_id: str | None = None
    elapsed_seconds: float = 0.0
    success: bool = False

    def add(self, stage: StageResult) -> None:
        self.stages.append(stage)
        logger.info("%s", stage)

    @property
    def failed_stages(self) -> list[StageResult]:
        return [s for s in self.stages if not s.ok]

    def summary(self) -> str:
        lines = [
            f"=== Ciclo de aprendizaje {self.run_date} ===",
            f"Duracion: {self.elapsed_seconds:.0f}s | barras: {self.bars_available}",
            f"Descubiertas: {self.strategies_discovered} | validadas: {self.strategies_validated}",
            f"Deriva: {self.drift_recommendation}",
            f"Campeon: {self.champion_id or 'ninguno'}",
        ]
        if self.promotion:
            lines.append(f"Decision: {self.promotion.summary()}")
        if self.failed_stages:
            lines.append(f"Etapas con error: {len(self.failed_stages)}")
            lines.extend(f"  - {s.name}: {s.error}" for s in self.failed_stages)
        return "\n".join(lines)


class Orchestrator:
    """Coordina el ciclo diario de aprendizaje."""

    def __init__(self, config: Config, db: Database | None = None) -> None:
        self.config = config
        self.db = db or Database(config.path(config.db_path))
        self.market_data = MarketData(config)
        self.registry = ChampionRegistry(config, self.db)
        self.gate = StabilityGate(config)
        self.trainer = MLTrainer(config, self.db)
        self.drift_detector = DriftDetector()
        self.engine = BacktestEngine(config)

    # ------------------------------------------------------------------ #
    def run_daily_cycle(self, force_discovery: bool = False) -> DailyReport:
        """Ejecuta el ciclo completo. Nunca lanza excepciones hacia arriba."""
        started = time.time()
        today = date.today().isoformat()

        report = DailyReport(run_date=today, started_at=now_utc().isoformat())
        run_id = self.db.start_run(today)

        logger.info("=" * 70)
        logger.info("INICIO DEL CICLO DE APRENDIZAJE - %s", today)
        logger.info("=" * 70)

        try:
            # --- 1) datos --- #
            ohlcv, features, catalog = self._stage_data(report, run_id)
            if ohlcv is None:
                report.elapsed_seconds = time.time() - started
                self.db.finish_run(run_id, "failed", report.summary(), "sin datos utilizables")
                return report

            # --- 2) deriva --- #
            self._stage_drift(report, run_id, ohlcv, features)

            # --- 3) descubrimiento --- #
            should_discover = force_discovery or self._should_discover(report)
            if should_discover:
                self._stage_discovery(report, run_id, ohlcv, features, catalog)
            else:
                report.add(StageResult(
                    "descubrimiento", True,
                    "omitido (no toca segun discovery_every_days y sin deriva relevante)",
                ))

            # --- 4) reentrenamiento del meta-modelo --- #
            self._stage_retrain(report, run_id, ohlcv, features)

            # --- 5) incubadora --- #
            self._stage_incubation(report, run_id)

            # --- 6) promocion / degradacion --- #
            self._stage_promotion(report, run_id)

            champion = self.registry.get_champion()
            report.champion_id = champion.id if champion else None
            report.success = len(report.failed_stages) == 0
            report.elapsed_seconds = time.time() - started

            self.db.finish_run(run_id, "ok" if report.success else "failed", report.summary())

        except Exception as exc:
            report.elapsed_seconds = time.time() - started
            report.success = False
            error = f"{type(exc).__name__}: {exc}"
            logger.error("Ciclo abortado: %s\n%s", error, traceback.format_exc())
            self.db.finish_run(run_id, "failed", report.summary(), error)

        logger.info("\n%s", report.summary())
        return report

    # ------------------------------------------------------------------ #
    # Etapas
    # ------------------------------------------------------------------ #
    def _stage_data(self, report: DailyReport, run_id: int):
        """Actualiza el historico y construye las features."""
        start = time.time()
        self.db.update_run(run_id, "datos")

        try:
            # Volcado desde el broker cuando el cache aun no da para entrenar.
            # Es la via por la que el ciclo de aprendizaje obtiene las 5.000
            # velas reales la primera vez que se ejecuta contra XM o Vantage.
            if (
                self.config.data.mt5_bootstrap_on_connect
                and len(self.market_data.cache.load()) < self.config.data.min_bars_required
            ):
                self.market_data.bootstrap_from_mt5()

            raw = self.market_data.get(refresh=True)
            if raw.empty:
                report.add(StageResult("datos", False, error="no se obtuvieron velas", elapsed=time.time() - start))
                return None, None, None

            ohlcv, features, catalog = build_features(raw, FeatureBuilder.from_config(self.config))
            report.bars_available = len(ohlcv)

            if len(ohlcv) < self.config.data.min_bars_required:
                report.add(StageResult(
                    "datos", False,
                    error=f"solo {len(ohlcv)} barras utiles (minimo {self.config.data.min_bars_required})",
                    elapsed=time.time() - start,
                ))
                return None, None, None

            report.add(StageResult(
                "datos", True,
                f"{len(ohlcv)} barras [{ohlcv.index[0]:%Y-%m-%d} -> {ohlcv.index[-1]:%Y-%m-%d}], "
                f"{len(catalog)} features",
                time.time() - start,
            ))
            return ohlcv, features, catalog

        except Exception as exc:
            report.add(StageResult("datos", False, error=str(exc), elapsed=time.time() - start))
            return None, None, None

    def _stage_drift(self, report: DailyReport, run_id: int, ohlcv, features) -> None:
        """Compara el mes reciente con el trimestre anterior."""
        start = time.time()
        self.db.update_run(run_id, "deriva")

        try:
            bars_per_day = int(24 * 60 / self.config.data.timeframe_minutes)
            recent_bars = 20 * bars_per_day
            reference_bars = 60 * bars_per_day

            if len(features) < recent_bars + reference_bars:
                report.add(StageResult(
                    "deriva", True, "historico insuficiente para comparar regimenes",
                    time.time() - start,
                ))
                return

            current = features.iloc[-recent_bars:]
            reference = features.iloc[-(recent_bars + reference_bars) : -recent_bars]

            champion_sharpe = reference_sharpe = None
            champion = self.registry.get_champion()
            if champion is not None:
                genome = champion.to_genome()
                recent_result = self.engine.run(
                    ohlcv.iloc[-recent_bars:],
                    genome.generate_signals(current, ohlcv.iloc[-recent_bars:]),
                    genome.exit_rules,
                )
                champion_sharpe = recent_result.metrics.sharpe
                reference_sharpe = float(
                    champion.metrics.get("stability", {}).get("metrics", {}).get("sharpe", 0.0) or 0.0
                )

            drift = self.drift_detector.detect(
                reference=reference,
                current=current,
                reference_sharpe=reference_sharpe,
                current_sharpe=champion_sharpe,
            )
            report.drift_recommendation = drift.recommendation

            # La degradacion sostenida del campeon se cuenta aqui.
            if champion is not None and champion_sharpe is not None:
                self._track_champion_health(champion.id, champion_sharpe, reference_sharpe or 0.0)

            report.add(StageResult("deriva", True, drift.summary(), time.time() - start))

        except Exception as exc:
            report.add(StageResult("deriva", False, error=str(exc), elapsed=time.time() - start))

    def _stage_discovery(self, report: DailyReport, run_id: int, ohlcv, features, catalog) -> None:
        """Lanza el genetico, refina y somete los finalistas a la puerta."""
        start = time.time()
        self.db.update_run(run_id, "descubrimiento")

        try:
            # El filtro de ML del campeon, si existe, guia la busqueda.
            labeler = self.trainer.load_active()
            signal_filter = self.trainer.build_signal_filter(labeler)

            genetic = GeneticEngine(self.config, ohlcv, features, catalog, signal_filter)
            budget = self.config.autonomy.max_runtime_minutes * 60 * 0.6
            evolution = genetic.run(max_seconds=budget)

            report.strategies_discovered = len(evolution.best)
            if not evolution.best:
                report.add(StageResult(
                    "descubrimiento", True, "el genetico no produjo finalistas", time.time() - start
                ))
                return

            # Refinamiento con Optuna de los mejores candidatos.
            finalists = evolution.best[: self.config.autonomy.keep_top_strategies]
            if self.config.optuna.enabled:
                finalists = self._refine(finalists, genetic, catalog)

            # Puerta de estabilidad sobre el historico completo.
            validated = 0
            size_multiplier = (
                labeler.size_multiplier(features) if labeler and labeler.is_ready else None
            )

            for individual in finalists:
                genome = individual.genome
                try:
                    verdict = self.gate.evaluate(genome, ohlcv, features, size_multiplier)
                except Exception as exc:
                    logger.warning("Puerta de estabilidad fallo en %s: %s", genome.genome_id, exc)
                    continue

                registered = self.registry.register_candidate(
                    genome,
                    verdict,
                    metrics={
                        "train": individual.fitness.summary(),
                        "validation": individual.validation.summary() if individual.validation else "",
                        "combined_score": individual.combined_score,
                    },
                )
                if registered and verdict.passed:
                    validated += 1

            report.strategies_validated = validated
            report.add(StageResult(
                "descubrimiento", True,
                f"{len(evolution.best)} finalistas, {validated} superaron la puerta de estabilidad "
                f"({evolution.total_evaluations} evaluaciones)",
                time.time() - start,
            ))

        except Exception as exc:
            logger.debug("Traza del fallo de descubrimiento:\n%s", traceback.format_exc())
            report.add(StageResult("descubrimiento", False, error=str(exc), elapsed=time.time() - start))

    def _refine(self, finalists, genetic: GeneticEngine, catalog):
        """Refina con Optuna, conservando el original si no hay mejora real."""
        from goldbot.evolution.optuna_opt import refine_with_optuna

        refined = []
        for individual in finalists:
            try:
                result = refine_with_optuna(
                    individual.genome,
                    genetic.train_evaluator,
                    genetic.val_evaluator,
                    self.config,
                    catalog,
                )
                if result.improved:
                    logger.info("%s: %s", individual.genome.genome_id, result.summary())
                    individual.genome = result.genome
            except Exception as exc:
                logger.warning("Refinamiento fallo en %s: %s", individual.genome.genome_id, exc)
            refined.append(individual)
        return refined

    def _stage_retrain(self, report: DailyReport, run_id: int, ohlcv, features) -> None:
        """Reentrena el meta-etiquetador del campeon con los datos de hoy."""
        start = time.time()
        self.db.update_run(run_id, "reentrenamiento")

        if not self.config.ml.enabled:
            report.add(StageResult("reentrenamiento", True, "ML desactivado", time.time() - start))
            return

        champion = self.registry.get_champion()
        if champion is None:
            report.add(StageResult(
                "reentrenamiento", True, "sin campeon al que asociar un modelo", time.time() - start
            ))
            return

        try:
            genome = champion.to_genome()
            result = self.trainer.train_for_strategy(
                genome, ohlcv, features, strategy_id=champion.id, save=True
            )

            if result.is_useful:
                from goldbot.ml.trainer import evaluate_ml_contribution

                contribution = evaluate_ml_contribution(
                    self.config, genome, ohlcv, features, result.labeler
                )
                self.db.save_evaluation(
                    champion.id, "ml_contribution", contribution, contribution["delta_sharpe"]
                )
                detail = f"{result.report.summary()} | aporta {contribution['delta_sharpe']:+.2f} de Sharpe"
            else:
                detail = result.summary()

            report.add(StageResult("reentrenamiento", True, detail, time.time() - start))

        except Exception as exc:
            report.add(StageResult("reentrenamiento", False, error=str(exc), elapsed=time.time() - start))

    def _stage_incubation(self, report: DailyReport, run_id: int) -> None:
        """Mete en la incubadora a las validadas que esperan hueco."""
        start = time.time()
        self.db.update_run(run_id, "incubacion")

        try:
            waiting = self.registry.get_validated()
            # Las mejores primero: los huecos de incubadora son escasos.
            waiting.sort(
                key=lambda r: float(r.metrics.get("stability", {}).get("score", 0.0)), reverse=True
            )

            started = 0
            for record in waiting:
                if self.registry.start_incubation(record.id):
                    started += 1

            statuses = [self.registry.check_incubation(r.id) for r in self.registry.get_incubating()]
            ready = sum(1 for s in statuses if s.ready)

            detail = f"{started} nuevas en incubacion, {len(statuses)} incubando, {ready} listas"
            if statuses:
                detail += " | " + "; ".join(s.summary() for s in statuses[:3])

            report.add(StageResult("incubacion", True, detail, time.time() - start))

        except Exception as exc:
            report.add(StageResult("incubacion", False, error=str(exc), elapsed=time.time() - start))

    def _stage_promotion(self, report: DailyReport, run_id: int) -> None:
        """Decide quien opera manana."""
        start = time.time()
        self.db.update_run(run_id, "promocion")

        try:
            # Primero se comprueba si el campeon actual debe retirarse.
            champion = self.registry.get_champion()
            if champion is not None:
                bad_days = int(self.db.get_state(f"bad_days:{champion.id}", 0) or 0)
                recent_sharpe = float(self.db.get_state(f"recent_sharpe:{champion.id}", 0.0) or 0.0)

                demotion = self.registry.evaluate_demotion(recent_sharpe, bad_days)
                if demotion.action == "retirar":
                    self.registry.demote(demotion.strategy_id, demotion.reason)
                    self.db.set_state(f"bad_days:{champion.id}", 0)
                    report.promotion = demotion
                    report.add(StageResult(
                        "promocion", True, f"campeon retirado: {demotion.reason}", time.time() - start
                    ))
                    # Tras retirar, se busca sustituto de inmediato.

            decision = self.registry.evaluate_promotion()
            if decision.action == "promover" and decision.strategy_id:
                self.registry.promote(
                    decision.strategy_id, decision.previous_champion, decision.reason
                )

            report.promotion = decision
            report.add(StageResult("promocion", True, decision.summary(), time.time() - start))

        except Exception as exc:
            report.add(StageResult("promocion", False, error=str(exc), elapsed=time.time() - start))

    # ------------------------------------------------------------------ #
    # Auxiliares
    # ------------------------------------------------------------------ #
    def _should_discover(self, report: DailyReport) -> bool:
        """Decide si toca lanzar el descubrimiento completo.

        Es la etapa mas cara (minutos de CPU). Se lanza por calendario, o de
        inmediato si no hay campeon o si la deriva lo aconseja.
        """
        if self.registry.get_champion() is None:
            logger.info("Sin campeon: se fuerza el descubrimiento")
            return True

        if report.drift_recommendation == "redescubrir_estrategias":
            logger.info("Deriva detectada: se fuerza el descubrimiento")
            return True

        last = self.db.get_state("last_discovery")
        if not last:
            return True

        try:
            from datetime import datetime

            last_date = datetime.fromisoformat(last).date()
        except (ValueError, TypeError):
            return True

        days_since = (date.today() - last_date).days
        due = days_since >= self.config.autonomy.discovery_every_days

        if due:
            self.db.set_state("last_discovery", date.today().isoformat())
        else:
            logger.info(
                "Descubrimiento en %d dias (ultimo hace %d)",
                self.config.autonomy.discovery_every_days - days_since,
                days_since,
            )
        return due

    def _track_champion_health(self, champion_id: str, recent_sharpe: float, expected: float) -> None:
        """Lleva la cuenta de dias consecutivos por debajo de lo esperado."""
        self.db.set_state(f"recent_sharpe:{champion_id}", float(recent_sharpe))

        threshold = expected - self.config.stability.degradation_sharpe_drop
        bad_days = int(self.db.get_state(f"bad_days:{champion_id}", 0) or 0)

        if recent_sharpe < threshold:
            bad_days += 1
            logger.warning(
                "Campeon %s por debajo de lo esperado (%.2f < %.2f): %d dias consecutivos",
                champion_id, recent_sharpe, threshold, bad_days,
            )
        else:
            # Un solo dia bueno reinicia el contador: exigimos degradacion continua.
            bad_days = 0

        self.db.set_state(f"bad_days:{champion_id}", bad_days)

    # ------------------------------------------------------------------ #
    def bootstrap(self) -> DailyReport:
        """Arranque en frio: descubre desde cero y prepara el primer campeon."""
        logger.info("Arranque inicial: se fuerza el descubrimiento completo")
        self.db.set_state("last_discovery", date.today().isoformat())
        return self.run_daily_cycle(force_discovery=True)

    def status(self) -> dict:
        """Estado global del sistema."""
        registry = self.registry.summary()
        last_run = self.db.last_successful_run()
        start, end, bars = self.market_data.cache.coverage()

        return {
            "registro": registry,
            "ultimo_ciclo_ok": last_run["finished_at"] if last_run else None,
            "datos": {
                "barras": bars,
                "desde": start.isoformat() if start else None,
                "hasta": end.isoformat() if end else None,
            },
            "modo_ejecucion": self.config.execution.mode,
            "dry_run": self.config.execution.dry_run,
        }
