"""Motor evolutivo: el componente que *inventa* las estrategias.

Algoritmo genetico con las salvaguardas que la practica exige:

* **Elitismo** para no perder nunca lo mejor encontrado.
* **Nicho por huella estructural**: si media poblacion converge al mismo
  genoma, la busqueda se ha detenido aunque el fitness siga subiendo. Se
  penaliza la duplicidad para mantener diversidad.
* **Mutacion adaptativa**: fuerte al principio (explorar), suave al final
  (refinar), y con repunte automatico si la poblacion se estanca.
* **Validacion en dos bloques**: el fitness se calcula sobre el tramo de
  entrenamiento, pero el salon de la fama se ordena por el resultado en un
  bloque reservado que la evolucion nunca ve. Es la diferencia entre
  seleccionar la mejor estrategia y seleccionar la mejor casualidad.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldbot.backtest.engine import BacktestEngine
from goldbot.config import Config
from goldbot.evolution.fitness import INVALID_SCORE, FitnessEvaluator, FitnessScore
from goldbot.features.engineering import FeatureCatalog
from goldbot.strategies.genome import StrategyGenome, crossover, mutate, random_genome
from goldbot.strategies.seeds import seed_population
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Individual:
    """Genoma junto a sus puntuaciones."""

    genome: StrategyGenome
    fitness: FitnessScore = field(default_factory=FitnessScore)
    validation: FitnessScore | None = None

    @property
    def score(self) -> float:
        return self.fitness.score

    @property
    def validation_score(self) -> float:
        return self.validation.score if self.validation else INVALID_SCORE

    @property
    def combined_score(self) -> float:
        """Criterio de ranking final del salon de la fama.

        Se pondera la validacion por encima del entrenamiento y se castiga la
        divergencia entre ambos: una estrategia que rinde 3.0 dentro y 0.2
        fuera es peor que una que rinde 1.0 y 0.9.
        """
        if self.validation is None:
            return self.score
        train, val = self.score, self.validation.score
        if train <= 0 or val <= 0:
            return min(train, val)
        divergence = abs(train - val) / max(train, val)
        return float((0.35 * train + 0.65 * val) * (1.0 - 0.5 * divergence))


@dataclass
class EvolutionReport:
    """Resumen de una ejecucion completa del motor evolutivo."""

    best: list[Individual] = field(default_factory=list)
    generations_run: int = 0
    total_evaluations: int = 0
    cache_efficiency: float = 0.0
    elapsed_seconds: float = 0.0
    history: list[dict] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""

    @property
    def champion(self) -> Individual | None:
        return self.best[0] if self.best else None

    def summary(self) -> str:
        champion = self.champion
        head = (
            f"Evolucion: {self.generations_run} generaciones, "
            f"{self.total_evaluations} evaluaciones "
            f"(cache {self.cache_efficiency:.0%}) en {self.elapsed_seconds:.0f}s"
        )
        if self.stopped_early:
            head += f" [parada anticipada: {self.stop_reason}]"
        if champion is None:
            return head + " | sin campeon"
        return (
            f"{head}\n  Campeon {champion.genome.genome_id}: "
            f"train={champion.fitness.summary()}\n"
            f"    validacion={champion.validation.summary() if champion.validation else 'n/d'}"
        )

    def to_history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


class GeneticEngine:
    """Algoritmo genetico sobre :class:`StrategyGenome`."""

    def __init__(
        self,
        config: Config,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        catalog: FeatureCatalog,
        signal_filter: callable | None = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.rng = np.random.default_rng(config.evolution.random_seed)

        # Particion train/validacion. El bloque de validacion queda al final
        # (el pasado mas reciente) porque es el que mas se parece al futuro.
        n = len(ohlcv)
        split = int(n * 0.75)
        purge = config.backtest.purge_bars

        self.train_ohlcv = ohlcv.iloc[:split]
        self.train_features = features.iloc[:split]
        self.val_ohlcv = ohlcv.iloc[split + purge :]
        self.val_features = features.iloc[split + purge :]

        engine = BacktestEngine(config)
        self.train_evaluator = FitnessEvaluator(
            config, self.train_ohlcv, self.train_features, engine, signal_filter
        )
        self.val_evaluator = FitnessEvaluator(
            config, self.val_ohlcv, self.val_features, engine, signal_filter
        )

        logger.info(
            "Motor evolutivo listo: %d barras de entrenamiento, %d de validacion, %d features",
            len(self.train_ohlcv),
            len(self.val_ohlcv),
            len(catalog),
        )

    # ------------------------------------------------------------------ #
    def run(self, max_seconds: float | None = None) -> EvolutionReport:
        """Ejecuta la evolucion completa."""
        cfg = self.config.evolution
        start_time = time.time()
        report = EvolutionReport()

        population = [
            Individual(genome=g)
            for g in seed_population(self.catalog, cfg.population_size, self.rng, cfg.max_conditions)
        ]

        best_ever = -np.inf
        generations_without_improvement = 0
        hall_of_fame: dict[str, Individual] = {}

        for generation in range(cfg.generations):
            self._evaluate(population)

            population.sort(key=lambda ind: ind.score, reverse=True)
            self._update_hall_of_fame(hall_of_fame, population)

            best_score = population[0].score
            improved = best_score > best_ever + 1e-6
            if improved:
                best_ever = best_score
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            stats = self._generation_stats(generation, population, best_score)
            report.history.append(stats)

            if generation % 5 == 0 or improved:
                logger.info(
                    "Gen %02d/%d | mejor=%.3f media=%.3f diversidad=%.0f%% validos=%d/%d",
                    generation,
                    cfg.generations,
                    best_score,
                    stats["mean_score"],
                    stats["diversity"] * 100,
                    stats["valid"],
                    len(population),
                )

            # --- criterios de parada --- #
            if generations_without_improvement >= cfg.early_stop_generations:
                report.stopped_early = True
                report.stop_reason = f"{generations_without_improvement} generaciones sin mejora"
                report.generations_run = generation + 1
                break
            if max_seconds and (time.time() - start_time) > max_seconds:
                report.stopped_early = True
                report.stop_reason = f"limite de tiempo ({max_seconds:.0f}s)"
                report.generations_run = generation + 1
                break

            report.generations_run = generation + 1
            population = self._next_generation(population, generation, generations_without_improvement)

        # Evaluacion final en validacion de los mejores candidatos.
        finalists = self._collect_finalists(hall_of_fame, population)
        self._validate(finalists)
        finalists.sort(key=lambda ind: ind.combined_score, reverse=True)

        report.best = finalists[: self.config.evolution.hall_of_fame]
        report.total_evaluations = self.train_evaluator.evaluations
        report.cache_efficiency = self.train_evaluator.cache_efficiency
        report.elapsed_seconds = time.time() - start_time

        logger.info("%s", report.summary())
        return report

    # ------------------------------------------------------------------ #
    def _evaluate(self, population: list[Individual]) -> None:
        for individual in population:
            if individual.fitness.valid or individual.fitness.score > INVALID_SCORE:
                # Ya evaluado en una generacion previa (elite): no repetimos.
                if individual.fitness.reason:
                    continue
            individual.fitness = self.train_evaluator.evaluate(individual.genome)

    def _validate(self, individuals: list[Individual]) -> None:
        """Evalua en el bloque reservado. Solo se hace con los finalistas."""
        for individual in individuals:
            if individual.validation is None:
                individual.validation = self.val_evaluator.evaluate(individual.genome)

    def _update_hall_of_fame(self, hall: dict[str, Individual], population: list[Individual]) -> None:
        """Guarda los mejores de todos los tiempos, deduplicados por huella."""
        limit = self.config.evolution.hall_of_fame * 3
        for individual in population:
            if not individual.fitness.valid:
                continue
            fingerprint = individual.genome.fingerprint()
            existing = hall.get(fingerprint)
            if existing is None or individual.score > existing.score:
                hall[fingerprint] = individual

        if len(hall) > limit:
            best = sorted(hall.items(), key=lambda kv: kv[1].score, reverse=True)[:limit]
            hall.clear()
            hall.update(best)

    def _collect_finalists(
        self, hall: dict[str, Individual], population: list[Individual]
    ) -> list[Individual]:
        """Une salon de la fama y poblacion final, sin duplicados."""
        pool: dict[str, Individual] = dict(hall)
        for individual in population:
            if individual.fitness.valid:
                pool.setdefault(individual.genome.fingerprint(), individual)

        finalists = sorted(pool.values(), key=lambda ind: ind.score, reverse=True)
        # Validar es caro: limitamos a un multiplo del salon de la fama.
        return finalists[: self.config.evolution.hall_of_fame * 2]

    # ------------------------------------------------------------------ #
    def _next_generation(
        self, population: list[Individual], generation: int, stagnation: int
    ) -> list[Individual]:
        """Construye la siguiente generacion."""
        cfg = self.config.evolution
        size = cfg.population_size

        # Fuerza de mutacion: decae con el progreso, pero repunta si nos
        # estancamos (mecanismo de escape de optimos locales).
        progress = generation / max(cfg.generations - 1, 1)
        strength = float(np.clip(1.0 - 0.7 * progress, 0.3, 1.0))
        if stagnation >= max(3, cfg.early_stop_generations // 3):
            strength = 1.0
            logger.debug("Estancamiento: mutacion elevada al maximo")

        next_population: list[Individual] = []

        # 1. Elite intacta.
        for individual in population[: cfg.elitism]:
            elite = Individual(genome=individual.genome, fitness=individual.fitness)
            next_population.append(elite)

        # 2. Inmigracion aleatoria: sangre nueva contra la convergencia prematura.
        n_immigrants = max(2, int(size * (0.10 if stagnation < 3 else 0.20)))
        for _ in range(n_immigrants):
            next_population.append(
                Individual(genome=random_genome(self.catalog, self.rng, cfg.max_conditions, generation + 1))
            )

        # 3. Descendencia por cruce y mutacion.
        seen = {ind.genome.fingerprint() for ind in next_population}
        attempts = 0
        max_attempts = size * 12

        while len(next_population) < size and attempts < max_attempts:
            attempts += 1
            parent_a = self._tournament(population)
            parent_b = self._tournament(population)

            if self.rng.random() < cfg.crossover_prob:
                child_a, child_b = crossover(
                    parent_a.genome, parent_b.genome, self.rng, cfg, self.catalog
                )
                children = [child_a, child_b]
            else:
                children = [parent_a.genome.clone()]

            for child in children:
                if self.rng.random() < cfg.mutation_prob:
                    child = mutate(child, self.catalog, self.rng, cfg, strength)

                fingerprint = child.fingerprint()
                # Clones exactos no aportan informacion: se descartan.
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                next_population.append(Individual(genome=child))
                if len(next_population) >= size:
                    break

        # Si la deduplicacion no llego a llenar la poblacion, completamos con
        # individuos aleatorios (senal de que la diversidad se ha agotado).
        while len(next_population) < size:
            next_population.append(
                Individual(genome=random_genome(self.catalog, self.rng, cfg.max_conditions, generation + 1))
            )

        return next_population

    def _tournament(self, population: list[Individual]) -> Individual:
        """Seleccion por torneo: presion selectiva ajustable y barata."""
        k = min(self.config.evolution.tournament_size, len(population))
        indices = self.rng.integers(0, len(population), size=k)
        best = population[int(indices[0])]
        for idx in indices[1:]:
            candidate = population[int(idx)]
            if candidate.score > best.score:
                best = candidate
        return best

    # ------------------------------------------------------------------ #
    def _generation_stats(
        self, generation: int, population: list[Individual], best_score: float
    ) -> dict:
        scores = np.array([ind.score for ind in population], dtype="float64")
        valid = [ind for ind in population if ind.fitness.valid]
        fingerprints = {ind.genome.fingerprint() for ind in population}

        return {
            "generation": generation,
            "best_score": float(best_score),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
            "valid": len(valid),
            # Diversidad = genomas estructuralmente distintos / tamano.
            "diversity": len(fingerprints) / len(population),
            "mean_trades": float(np.mean([ind.fitness.trades for ind in valid])) if valid else 0.0,
            "mean_complexity": float(np.mean([ind.fitness.complexity for ind in population])),
        }
