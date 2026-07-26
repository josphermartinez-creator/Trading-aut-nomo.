"""Poblacion semilla: arquetipos clasicos de trading.

Arrancar el algoritmo genetico desde ruido puro funciona, pero desperdicia
muchas generaciones redescubriendo ideas que el oficio ya conoce. Sembrar la
poblacion con arquetipos sensatos (seguimiento de tendencia, reversion a la
media, ruptura, compresion de volatilidad...) acelera la convergencia sin
restringir el espacio de busqueda: el GA sigue siendo libre de destrozarlos,
recombinarlos o descartarlos por completo.

Cada semilla se construye de forma defensiva: si el catalogo actual no incluye
alguna de las features que necesita, esa semilla simplemente se omite.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from goldbot.backtest.engine import ExitRules
from goldbot.features.engineering import FeatureCatalog
from goldbot.strategies.genome import Condition, RuleSet, StrategyGenome, random_genome
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


def _cond(feature: str, operator: str, threshold: float = 0.0, target: str | None = None) -> Condition:
    return Condition(
        feature=feature,
        operator=operator,
        target_type="feature" if target else "const",
        threshold=threshold,
        target_feature=target,
    )


def _requires(catalog: FeatureCatalog, *names: str) -> bool:
    return all(name in catalog for name in names)


# --------------------------------------------------------------------------- #
# Arquetipos
# --------------------------------------------------------------------------- #
def _trend_following(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Cruce de medias filtrado por fuerza de tendencia (ADX)."""
    if not _requires(catalog, "ema_cross_9_50", "adx_14"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("ema_cross_9_50", "gt", 0.0)], "and"),
        short_rules=RuleSet([_cond("ema_cross_9_50", "lt", 0.0)], "and"),
        regime_filter=RuleSet([_cond("adx_14", "gt", 22.0)], "and"),
        exit_rules=ExitRules(stop_atr=2.0, target_atr=4.0, trail_atr=2.5, max_bars=144),
        metadata={"archetype": "seguimiento_tendencia"},
    )


def _mean_reversion(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Bandas de Bollinger + RSI, operando solo en rango (ADX bajo)."""
    if not _requires(catalog, "bb_pct_20", "rsi_14", "adx_14"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("bb_pct_20", "lt", 0.05), _cond("rsi_14", "lt", 30.0)], "and"),
        short_rules=RuleSet([_cond("bb_pct_20", "gt", 0.95), _cond("rsi_14", "gt", 70.0)], "and"),
        regime_filter=RuleSet([_cond("adx_14", "lt", 20.0)], "and"),
        exit_rules=ExitRules(stop_atr=1.5, target_atr=1.5, max_bars=48, exit_on_reverse=False),
        metadata={"archetype": "reversion_media"},
    )


def _breakout(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Ruptura de canal de Donchian en la sesion liquida."""
    if not _requires(catalog, "donchian_pos_20", "atr_rank_14"):
        return None
    regime = RuleSet([_cond("atr_rank_14", "gt", 0.5)], "and")
    if "in_overlap" in catalog:
        regime.conditions.append(_cond("in_overlap", "gt", 0.5))
    return StrategyGenome(
        long_rules=RuleSet([_cond("donchian_pos_20", "cross_above", 0.95)], "and"),
        short_rules=RuleSet([_cond("donchian_pos_20", "cross_below", 0.05)], "and"),
        regime_filter=regime,
        exit_rules=ExitRules(stop_atr=1.8, target_atr=3.6, trail_atr=2.0, max_bars=96),
        metadata={"archetype": "ruptura"},
    )


def _volatility_squeeze(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Compresion de bandas seguida de expansion direccional."""
    if not _requires(catalog, "bb_squeeze", "macd_hist_norm"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("macd_hist_norm", "cross_above", 0.0)], "and"),
        short_rules=RuleSet([_cond("macd_hist_norm", "cross_below", 0.0)], "and"),
        regime_filter=RuleSet([_cond("bb_squeeze", "lt", 0.25)], "and"),
        exit_rules=ExitRules(stop_atr=1.5, target_atr=4.5, trail_atr=1.5, breakeven_atr=1.0, max_bars=120),
        metadata={"archetype": "compresion_volatilidad"},
    )


def _momentum_pullback(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Retroceso dentro de tendencia mayor: compra caidas en tendencia alcista."""
    if not _requires(catalog, "trend_h4", "rsi_7", "ema_dist_21"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("trend_h4", "gt", 0.001), _cond("rsi_7", "lt", 35.0)], "and"),
        short_rules=RuleSet([_cond("trend_h4", "lt", -0.001), _cond("rsi_7", "gt", 65.0)], "and"),
        regime_filter=RuleSet(),
        exit_rules=ExitRules(stop_atr=2.2, target_atr=3.3, breakeven_atr=1.2, max_bars=96),
        metadata={"archetype": "retroceso_momento"},
    )


def _supertrend_follow(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Seguimiento puro de SuperTrend con confirmacion de pendiente."""
    if not _requires(catalog, "supertrend_dir_10", "linreg_slope_20"):
        return None
    return StrategyGenome(
        long_rules=RuleSet(
            [_cond("supertrend_dir_10", "cross_above", 0.0), _cond("linreg_slope_20", "gt", 0.0)], "and"
        ),
        short_rules=RuleSet(
            [_cond("supertrend_dir_10", "cross_below", 0.0), _cond("linreg_slope_20", "lt", 0.0)], "and"
        ),
        regime_filter=RuleSet(),
        exit_rules=ExitRules(stop_atr=2.5, target_atr=5.0, trail_atr=2.5, max_bars=288),
        metadata={"archetype": "supertrend"},
    )


def _session_bias(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Continuacion en la apertura de Londres, la ventana mas direccional del oro."""
    if not _requires(catalog, "in_london", "roc_10", "atr_pct_14"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("roc_10", "gt", 0.08)], "and"),
        short_rules=RuleSet([_cond("roc_10", "lt", -0.08)], "and"),
        regime_filter=RuleSet([_cond("in_london", "gt", 0.5), _cond("atr_pct_14", "gt", 0.0008)], "and"),
        exit_rules=ExitRules(stop_atr=1.5, target_atr=2.5, max_bars=60, exit_on_reverse=False),
        metadata={"archetype": "sesgo_sesion"},
    )


def _vwap_reversion(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Reversion al VWAP de la sesion (solo si el feed trae volumen)."""
    if not _requires(catalog, "vwap_dist", "stoch_k_14"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("vwap_dist", "lt", -0.002), _cond("stoch_k_14", "lt", 20.0)], "and"),
        short_rules=RuleSet([_cond("vwap_dist", "gt", 0.002), _cond("stoch_k_14", "gt", 80.0)], "and"),
        regime_filter=RuleSet(),
        exit_rules=ExitRules(stop_atr=1.8, target_atr=1.8, max_bars=36, exit_on_reverse=False),
        metadata={"archetype": "reversion_vwap"},
    )


def _hurst_regime(catalog: FeatureCatalog) -> StrategyGenome | None:
    """Cambia de logica segun el regimen: persistente => tendencia."""
    if not _requires(catalog, "hurst_100", "ema_cross_14_50"):
        return None
    return StrategyGenome(
        long_rules=RuleSet([_cond("ema_cross_14_50", "cross_above", 0.0)], "and"),
        short_rules=RuleSet([_cond("ema_cross_14_50", "cross_below", 0.0)], "and"),
        regime_filter=RuleSet([_cond("hurst_100", "gt", 0.52)], "and"),
        exit_rules=ExitRules(stop_atr=2.0, target_atr=4.0, trail_atr=2.0, max_bars=200),
        metadata={"archetype": "regimen_hurst"},
    )


ARCHETYPES: tuple[Callable[[FeatureCatalog], StrategyGenome | None], ...] = (
    _trend_following,
    _mean_reversion,
    _breakout,
    _volatility_squeeze,
    _momentum_pullback,
    _supertrend_follow,
    _session_bias,
    _vwap_reversion,
    _hurst_regime,
)


# --------------------------------------------------------------------------- #
def build_archetypes(catalog: FeatureCatalog) -> list[StrategyGenome]:
    """Instancia todos los arquetipos compatibles con el catalogo dado."""
    genomes: list[StrategyGenome] = []
    for factory in ARCHETYPES:
        try:
            genome = factory(catalog)
        except Exception as exc:  # una semilla rota no debe tumbar el arranque
            logger.warning("Semilla %s fallo: %s", factory.__name__, exc)
            continue
        if genome is not None:
            genomes.append(genome)
    logger.info("Semillas disponibles: %d de %d arquetipos", len(genomes), len(ARCHETYPES))
    return genomes


def seed_population(
    catalog: FeatureCatalog,
    size: int,
    rng: np.random.Generator,
    max_conditions: int = 4,
    seed_ratio: float = 0.35,
) -> list[StrategyGenome]:
    """Poblacion inicial: arquetipos + variantes suyas + individuos aleatorios.

    ``seed_ratio`` controla cuanta poblacion procede de arquetipos. Demasiado
    alto y la busqueda se queda anclada en lo conocido; demasiado bajo y se
    tarda en despegar. Un tercio es un equilibrio razonable.
    """
    from goldbot.config import EvolutionConfig
    from goldbot.strategies.genome import mutate

    archetypes = build_archetypes(catalog)
    population: list[StrategyGenome] = []

    if archetypes:
        # Los arquetipos puros entran tal cual.
        population.extend(g.clone() for g in archetypes[:size])

        # Variantes mutadas: exploran el vecindario de cada idea conocida.
        target_seeded = int(size * seed_ratio)
        cfg = EvolutionConfig(max_conditions=max_conditions)
        while len(population) < target_seeded:
            base = archetypes[int(rng.integers(len(archetypes)))]
            variant = mutate(base, catalog, rng, cfg, strength=1.0)
            variant.metadata["archetype"] = base.metadata.get("archetype", "?") + "_variante"
            population.append(variant)

    # El resto, aleatorio puro: es de donde salen las ideas que nadie busco.
    while len(population) < size:
        population.append(random_genome(catalog, rng, max_conditions))

    logger.info(
        "Poblacion inicial: %d individuos (%d desde arquetipos, %d aleatorios)",
        len(population),
        sum(1 for g in population if "archetype" in g.metadata),
        sum(1 for g in population if "archetype" not in g.metadata),
    )
    return population[:size]
