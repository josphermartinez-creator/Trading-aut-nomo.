"""Refinamiento fino con Optuna (optimizacion bayesiana).

El genetico es bueno encontrando la *forma* de una estrategia (que features,
que operadores, que estructura) pero tosco ajustando numeros. Optuna hace lo
contrario: no puede inventar reglas, pero afina umbrales y parametros de salida
mucho mejor que la mutacion aleatoria.

Se usa como segunda pasada sobre los finalistas del genetico.

Precaucion importante: refinar es la fase donde mas facil resulta sobreajustar,
porque se busca el maximo de una superficie ya explorada. Por eso el objetivo
no es el fitness del bloque de entrenamiento, sino una **combinacion penalizada
por divergencia** entre entrenamiento y validacion; y ademas se comprueba que
el optimo no sea un pico estrecho de ruido.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goldbot.config import Config
from goldbot.evolution.fitness import FitnessEvaluator
from goldbot.features.engineering import FeatureCatalog
from goldbot.strategies.genome import StrategyGenome
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefinementResult:
    """Resultado del refinamiento."""

    genome: StrategyGenome
    improved: bool
    baseline_score: float
    refined_score: float
    n_trials: int
    neighborhood_stability: float = 0.0

    def summary(self) -> str:
        arrow = "mejora" if self.improved else "sin mejora"
        return (
            f"Optuna ({self.n_trials} pruebas): {self.baseline_score:.3f} -> "
            f"{self.refined_score:.3f} [{arrow}] "
            f"| estabilidad del entorno={self.neighborhood_stability:.2f}"
        )


def refine_with_optuna(
    genome: StrategyGenome,
    train_evaluator: FitnessEvaluator,
    val_evaluator: FitnessEvaluator,
    config: Config,
    catalog: FeatureCatalog | None = None,
) -> RefinementResult:
    """Afina los parametros numericos de ``genome`` sin tocar su estructura."""
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna no instalado; se omite el refinamiento")
        return RefinementResult(genome, False, 0.0, 0.0, 0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = config.optuna
    baseline = _combined_score(genome, train_evaluator, val_evaluator)

    # Recogemos los parametros ajustables: umbrales constantes + salidas.
    tunable = _collect_tunable(genome, catalog)
    if not tunable:
        logger.debug("Genoma %s sin parametros ajustables", genome.genome_id)
        return RefinementResult(genome, False, baseline, baseline, 0)

    def objective(trial: optuna.Trial) -> float:
        candidate = _apply_params(genome, tunable, trial)
        return _combined_score(candidate, train_evaluator, val_evaluator)

    sampler = optuna.samplers.TPESampler(
        seed=config.evolution.random_seed,
        n_startup_trials=cfg.n_startup_trials,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # El baseline entra como primera prueba: Optuna nunca devolvera algo peor
    # que el genoma de partida.
    study.enqueue_trial({name: spec["current"] for name, spec in tunable.items()})

    try:
        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout_seconds,
            catch=(Exception,),
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning("Optuna fallo en %s: %s", genome.genome_id, exc)
        return RefinementResult(genome, False, baseline, baseline, 0)

    if not study.trials or study.best_value <= baseline:
        logger.debug("Optuna no mejoro el genoma %s", genome.genome_id)
        return RefinementResult(genome, False, baseline, baseline, len(study.trials))

    refined = _apply_params(genome, tunable, None, study.best_params)
    refined.metadata["refined_by"] = "optuna"
    refined.metadata["refinement_gain"] = float(study.best_value - baseline)

    # Un optimo que se desmorona al moverlo un 10% es ruido, no una ventaja.
    stability = _neighborhood_stability(
        refined, tunable, study.best_params, train_evaluator, val_evaluator, study.best_value
    )
    if stability < 0.5:
        logger.info(
            "Genoma %s: el optimo de Optuna es inestable (%.2f); se conserva el original",
            genome.genome_id,
            stability,
        )
        return RefinementResult(genome, False, baseline, study.best_value, len(study.trials), stability)

    return RefinementResult(
        refined, True, baseline, float(study.best_value), len(study.trials), stability
    )


# --------------------------------------------------------------------------- #
def _collect_tunable(genome: StrategyGenome, catalog: FeatureCatalog | None) -> dict[str, dict]:
    """Enumera los parametros continuos del genoma y su rango de busqueda."""
    tunable: dict[str, dict] = {}

    rulesets = {
        "long": genome.long_rules,
        "short": genome.short_rules,
        "regime": genome.regime_filter,
    }
    for prefix, rules in rulesets.items():
        for i, condition in enumerate(rules.conditions):
            if condition.target_type != "const":
                continue
            spec = catalog.get(condition.feature) if catalog else None
            if spec is not None and spec.kind == "binary":
                continue  # un umbral binario no admite ajuste fino

            current = float(condition.threshold)
            if spec is not None:
                span = max(abs(spec.high - spec.low) * 0.4, abs(current) * 0.5, 1e-9)
            else:
                span = max(abs(current) * 0.5, 1e-6)

            tunable[f"{prefix}_{i}_threshold"] = {
                "kind": "threshold",
                "low": current - span,
                "high": current + span,
                "current": current,
            }

    e = genome.exit_rules
    tunable["stop_atr"] = {"kind": "exit", "low": max(0.3, e.stop_atr * 0.5),
                           "high": min(6.0, e.stop_atr * 1.8), "current": e.stop_atr}
    tunable["target_atr"] = {"kind": "exit", "low": max(0.3, e.target_atr * 0.5),
                             "high": min(12.0, e.target_atr * 1.8), "current": e.target_atr}
    if e.trail_atr > 0:
        tunable["trail_atr"] = {"kind": "exit", "low": max(0.3, e.trail_atr * 0.5),
                                "high": min(6.0, e.trail_atr * 1.8), "current": e.trail_atr}
    if e.breakeven_atr > 0:
        tunable["breakeven_atr"] = {"kind": "exit", "low": max(0.3, e.breakeven_atr * 0.5),
                                    "high": min(4.0, e.breakeven_atr * 1.8), "current": e.breakeven_atr}
    tunable["max_bars"] = {"kind": "int", "low": max(6, int(e.max_bars * 0.5)),
                           "high": min(576, int(e.max_bars * 2.0)), "current": e.max_bars}
    return tunable


def _apply_params(
    genome: StrategyGenome,
    tunable: dict[str, dict],
    trial: object | None,
    params: dict | None = None,
) -> StrategyGenome:
    """Crea una copia del genoma con los parametros propuestos aplicados."""
    candidate = genome.clone()

    def _value(name: str, spec: dict):
        if params is not None:
            return params.get(name, spec["current"])
        if spec["kind"] == "int":
            return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        return trial.suggest_float(name, spec["low"], spec["high"])

    rulesets = {
        "long": candidate.long_rules,
        "short": candidate.short_rules,
        "regime": candidate.regime_filter,
    }

    for name, spec in tunable.items():
        value = _value(name, spec)
        if spec["kind"] == "threshold":
            prefix, index, _ = name.split("_", 2)
            rules = rulesets[prefix]
            idx = int(index)
            if idx < len(rules.conditions):
                rules.conditions[idx].threshold = float(value)
        elif name == "max_bars":
            candidate.exit_rules.max_bars = int(value)
        else:
            setattr(candidate.exit_rules, name, float(value))

    candidate.exit_rules.validate()
    return candidate


def _combined_score(
    genome: StrategyGenome, train_evaluator: FitnessEvaluator, val_evaluator: FitnessEvaluator
) -> float:
    """Objetivo: entrenamiento y validacion, penalizando la divergencia.

    Optimizar solo el entrenamiento produce un genoma perfecto para el pasado e
    inutil para el futuro; optimizar solo la validacion la convierte en un
    segundo conjunto de entrenamiento. La media penalizada es el compromiso.
    """
    train = train_evaluator.evaluate(genome).score
    val = val_evaluator.evaluate(genome).score

    if train <= 0 or val <= 0:
        return float(min(train, val))

    divergence = abs(train - val) / max(train, val)
    return float((0.4 * train + 0.6 * val) * (1.0 - 0.5 * divergence))


def _neighborhood_stability(
    genome: StrategyGenome,
    tunable: dict[str, dict],
    best_params: dict,
    train_evaluator: FitnessEvaluator,
    val_evaluator: FitnessEvaluator,
    best_score: float,
    n_samples: int = 8,
    perturbation: float = 0.10,
) -> float:
    """Cuanto se conserva el rendimiento al mover los parametros un +-10%.

    Devuelve la razon entre el rendimiento medio del vecindario y el del
    optimo. Cerca de 1 = meseta ancha (bueno). Cerca de 0 = pico de ruido.
    """
    if best_score <= 0:
        return 0.0

    rng = np.random.default_rng(12345)
    scores = []

    for _ in range(n_samples):
        perturbed = {}
        for name, value in best_params.items():
            spec = tunable.get(name)
            if spec is None:
                perturbed[name] = value
                continue
            factor = 1.0 + rng.uniform(-perturbation, perturbation)
            new_value = value * factor
            if spec["kind"] == "int":
                perturbed[name] = int(np.clip(round(new_value), spec["low"], spec["high"]))
            else:
                perturbed[name] = float(np.clip(new_value, spec["low"], spec["high"]))

        candidate = _apply_params(genome, tunable, None, perturbed)
        scores.append(_combined_score(candidate, train_evaluator, val_evaluator))

    if not scores:
        return 0.0
    mean_score = float(np.mean(scores))
    return float(np.clip(mean_score / best_score, 0.0, 1.0))
