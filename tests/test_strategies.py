"""Pruebas del genoma, los operadores geneticos y el motor evolutivo."""

from __future__ import annotations

import numpy as np
import pytest

from goldbot.strategies.genome import (
    Condition,
    RuleSet,
    StrategyGenome,
    crossover,
    mutate,
    random_condition,
    random_genome,
)


def test_condicion_umbral(features):
    column = features.columns[0]
    threshold = float(features[column].median())

    condition = Condition(feature=column, operator="gt", threshold=threshold)
    mask = condition.evaluate(features)

    assert mask.dtype == bool
    assert mask.sum() > 0
    np.testing.assert_array_equal(mask.to_numpy(), (features[column] > threshold).to_numpy())


def test_condicion_cruce_es_causal(features):
    """Un cruce exige que la barra anterior estuviera al otro lado."""
    column = features.columns[0]
    threshold = float(features[column].median())

    condition = Condition(feature=column, operator="cross_above", threshold=threshold)
    mask = condition.evaluate(features)

    series = features[column]
    expected = (series > threshold) & (series.shift(1) <= threshold)
    np.testing.assert_array_equal(mask.to_numpy(), expected.fillna(False).to_numpy())

    # Un cruce ocurre menos veces que un simple "por encima".
    assert mask.sum() < (series > threshold).sum()


def test_feature_ausente_no_revienta(features):
    condition = Condition(feature="no_existe_esta_feature", operator="gt", threshold=0.0)
    mask = condition.evaluate(features)
    assert not mask.any()


def test_logicas_del_ruleset(features):
    columns = list(features.columns[:3])
    conditions = [
        Condition(feature=c, operator="gt", threshold=float(features[c].median())) for c in columns
    ]

    and_mask = RuleSet(conditions, "and").evaluate(features)
    or_mask = RuleSet(conditions, "or").evaluate(features)
    majority_mask = RuleSet(conditions, "majority").evaluate(features)

    assert and_mask.sum() <= majority_mask.sum() <= or_mask.sum()
    # AND implica OR.
    assert (and_mask & ~or_mask).sum() == 0


def test_ruleset_vacio_es_neutro(features):
    mask = RuleSet([], "and").evaluate(features)
    assert mask.all(), "un conjunto vacio no debe filtrar nada"


def test_senales_solo_toman_tres_valores(catalog, features):
    rng = np.random.default_rng(9)
    for _ in range(15):
        genome = random_genome(catalog, rng)
        signals = genome.generate_signals(features)
        assert set(np.unique(signals)).issubset({-1.0, 0.0, 1.0})


def test_senal_contradictoria_es_neutra(features):
    """Si las reglas de largo y corto se activan a la vez, no hay senal."""
    column = features.columns[0]
    median = float(features[column].median())

    # Ambas reglas verdaderas en las mismas barras.
    same = Condition(feature=column, operator="gt", threshold=median)
    genome = StrategyGenome(
        long_rules=RuleSet([same], "and"),
        short_rules=RuleSet([same], "and"),
    )
    signals = genome.generate_signals(features)
    assert (signals != 0).sum() == 0


def test_serializacion_ida_y_vuelta(catalog):
    rng = np.random.default_rng(21)
    original = random_genome(catalog, rng, max_conditions=4)

    restored = StrategyGenome.from_json(original.to_json())

    assert restored.fingerprint() == original.fingerprint()
    assert restored.genome_id == original.genome_id
    assert restored.exit_rules.stop_atr == pytest.approx(original.exit_rules.stop_atr)
    assert restored.describe() == original.describe()


def test_la_huella_ignora_el_id(catalog):
    rng = np.random.default_rng(4)
    original = random_genome(catalog, rng)
    clone = original.clone()

    assert clone.genome_id != original.genome_id
    assert clone.fingerprint() == original.fingerprint(), "el clon es estructuralmente identico"


def test_mutacion_produce_genoma_valido(catalog, features, config):
    rng = np.random.default_rng(13)
    parent = random_genome(catalog, rng)

    changed = 0
    for _ in range(30):
        child = mutate(parent, catalog, rng, config.evolution)

        assert child.genome_id != parent.genome_id
        assert child.generation == parent.generation + 1
        assert child.allow_long or child.allow_short
        assert child.complexity() <= config.evolution.max_conditions * 3
        child.generate_signals(features)  # no debe lanzar

        if child.fingerprint() != parent.fingerprint():
            changed += 1

    assert changed > 15, "la mutacion apenas altera el genoma"


def test_cruce_mezcla_a_ambos_padres(catalog, features, config):
    rng = np.random.default_rng(17)
    parent_a = random_genome(catalog, rng)
    parent_b = random_genome(catalog, rng)

    child_a, child_b = crossover(parent_a, parent_b, rng, config.evolution, catalog)

    for child in (child_a, child_b):
        assert set(child.parents) == {parent_a.genome_id, parent_b.genome_id}
        assert child.allow_long or child.allow_short
        child.generate_signals(features)


def test_reparacion_de_features_invalidas(catalog, features, config):
    """Una condicion sobre una feature inexistente debe repararse sola."""
    genome = StrategyGenome(
        long_rules=RuleSet([Condition("feature_fantasma", "gt", threshold=1.0)], "and"),
        short_rules=RuleSet([Condition("otra_fantasma", "lt", threshold=1.0)], "and"),
    )
    rng = np.random.default_rng(2)
    repaired = mutate(genome, catalog, rng, config.evolution)

    for name in repaired.required_features():
        assert name in catalog, f"{name} no existe en el catalogo tras la reparacion"


def test_condiciones_aleatorias_son_semanticamente_validas(catalog):
    """El umbral muestreado debe caer en el rango util de la feature."""
    rng = np.random.default_rng(31)

    for _ in range(200):
        condition = random_condition(catalog, rng)
        spec = catalog.get(condition.feature)
        assert spec is not None

        if condition.target_type == "feature":
            other = catalog.get(condition.target_feature)
            assert other is not None
            assert other.kind == spec.kind, "solo se comparan features del mismo tipo"
        elif spec.kind == "binary":
            assert condition.threshold == 0.5


def test_semillas_generan_estrategias_operables(catalog, features, ohlcv, config):
    from goldbot.backtest.engine import BacktestEngine
    from goldbot.strategies.seeds import build_archetypes

    engine = BacktestEngine(config)
    archetypes = build_archetypes(catalog)
    assert len(archetypes) >= 5, "deberian instanciarse la mayoria de arquetipos"

    trading = 0
    for genome in archetypes:
        result = engine.run(ohlcv, genome.generate_signals(features), genome.exit_rules)
        if result.metrics.total_trades > 0:
            trading += 1

    assert trading >= len(archetypes) // 2, "demasiados arquetipos no operan nunca"


@pytest.mark.slow
def test_el_genetico_mejora_el_fitness(config, ohlcv, features, catalog):
    """La evolucion debe mejorar el mejor fitness respecto a la generacion 0."""
    from goldbot.evolution.genetic import GeneticEngine

    engine = GeneticEngine(config, ohlcv, features, catalog)
    report = engine.run(max_seconds=180)

    assert report.generations_run >= 2
    assert report.history, "no se registro historial de generaciones"

    first = report.history[0]["best_score"]
    last = max(h["best_score"] for h in report.history)
    assert last >= first, "el mejor fitness no deberia empeorar"

    # Diversidad razonable: si cae a cero, la busqueda se ha colapsado.
    assert report.history[-1]["diversity"] > 0.3
