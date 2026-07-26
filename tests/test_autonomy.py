"""Pruebas del ciclo autonomo: puerta de estabilidad, registro y persistencia."""

from __future__ import annotations

import numpy as np

from goldbot.storage.db import (
    STATUS_CHAMPION,
    STATUS_INCUBATING,
    STATUS_VALIDATED,
)


# --------------------------------------------------------------------------- #
# Base de datos
# --------------------------------------------------------------------------- #
def test_guardar_y_recuperar_estrategia(tmp_db, catalog):
    from goldbot.strategies.genome import random_genome

    genome = random_genome(catalog, np.random.default_rng(1))
    tmp_db.save_strategy(genome, status=STATUS_VALIDATED, metrics={"stability": {"score": 0.7}})

    record = tmp_db.get_strategy(genome.genome_id)
    assert record is not None
    assert record.status == STATUS_VALIDATED
    assert record.metrics["stability"]["score"] == 0.7

    restored = record.to_genome()
    assert restored.fingerprint() == genome.fingerprint()


def test_busqueda_por_huella(tmp_db, catalog):
    from goldbot.strategies.genome import random_genome

    genome = random_genome(catalog, np.random.default_rng(2))
    tmp_db.save_strategy(genome)

    found = tmp_db.find_by_fingerprint(genome.fingerprint())
    assert found is not None and found.id == genome.genome_id
    assert tmp_db.find_by_fingerprint("huella_inexistente") is None


def test_solo_hay_un_campeon(tmp_db, catalog):
    from goldbot.strategies.genome import random_genome

    rng = np.random.default_rng(3)
    first = random_genome(catalog, rng)
    second = random_genome(catalog, rng)

    tmp_db.save_strategy(first, status=STATUS_CHAMPION)
    champion = tmp_db.get_champion()
    assert champion.id == first.genome_id

    tmp_db.save_strategy(second, status=STATUS_VALIDATED)
    assert tmp_db.get_champion().id == first.genome_id


def test_estado_clave_valor(tmp_db):
    tmp_db.set_state("prueba", {"a": 1, "b": [1, 2]})
    assert tmp_db.get_state("prueba") == {"a": 1, "b": [1, 2]}
    assert tmp_db.get_state("no_existe", "por_defecto") == "por_defecto"


def test_registro_de_operaciones(tmp_db):
    trade_id = tmp_db.record_trade(
        strategy_id="s1", mode="paper", entry_time="2024-01-01T00:00:00+00:00",
        direction=1, entry_price=2000.0, lots=0.1,
    )
    tmp_db.close_trade(trade_id, "2024-01-01T01:00:00+00:00", 2010.0, 100.0, "take_profit")

    trades = tmp_db.get_trades(strategy_id="s1", mode="paper")
    assert len(trades) == 1
    assert trades[0]["pnl"] == 100.0
    assert trades[0]["exit_reason"] == "take_profit"


def test_ciclo_de_aprendizaje(tmp_db):
    run_id = tmp_db.start_run("2024-01-01")
    tmp_db.update_run(run_id, "descubrimiento")
    tmp_db.finish_run(run_id, "ok", "todo correcto")

    last = tmp_db.last_successful_run()
    assert last is not None and last["status"] == "ok"


# --------------------------------------------------------------------------- #
# Puerta de estabilidad
# --------------------------------------------------------------------------- #
def test_la_puerta_rechaza_estrategias_sin_operaciones(config, ohlcv, features):
    from goldbot.autonomy.stability import StabilityGate
    from goldbot.strategies.genome import Condition, RuleSet, StrategyGenome

    # Condicion imposible: nunca se cumple.
    column = features.columns[0]
    impossible = Condition(feature=column, operator="gt", threshold=1e12)
    genome = StrategyGenome(
        long_rules=RuleSet([impossible], "and"),
        short_rules=RuleSet([impossible], "and"),
    )

    verdict = StabilityGate(config).evaluate(genome, ohlcv, features, run_walkforward=False)

    assert not verdict.passed
    assert verdict.score == 0.0
    assert any("operaciones" in c.name for c in verdict.failed_checks)


def test_la_puerta_rechaza_estrategias_perdedoras(config, ohlcv, features, catalog):
    """Sobre datos sinteticos sin ventaja, casi nada debe pasar la puerta."""
    from goldbot.autonomy.stability import StabilityGate
    from goldbot.strategies.genome import random_genome

    gate = StabilityGate(config)
    rng = np.random.default_rng(77)

    passed = 0
    for _ in range(12):
        genome = random_genome(catalog, rng, max_conditions=3)
        verdict = gate.evaluate(genome, ohlcv, features, run_walkforward=False)
        if verdict.passed:
            passed += 1

    assert passed <= 1, (
        f"{passed} de 12 estrategias aleatorias pasaron la puerta de estabilidad. "
        f"El filtro es demasiado permisivo o hay una fuga."
    )


def test_el_veredicto_es_serializable(config, ohlcv, features, catalog):
    import json

    from goldbot.autonomy.stability import StabilityGate
    from goldbot.strategies.genome import random_genome

    genome = random_genome(catalog, np.random.default_rng(5))
    verdict = StabilityGate(config).evaluate(genome, ohlcv, features, run_walkforward=False)

    payload = json.dumps(verdict.to_dict(), default=str)
    assert json.loads(payload)["genome_id"] == genome.genome_id


# --------------------------------------------------------------------------- #
# Registro campeon/retador
# --------------------------------------------------------------------------- #
def test_no_se_promociona_sin_incubar(config, tmp_db, catalog):
    """Una estrategia validada no puede saltar directamente a campeona."""
    from goldbot.autonomy.champion import ChampionRegistry
    from goldbot.strategies.genome import random_genome

    registry = ChampionRegistry(config, tmp_db)
    genome = random_genome(catalog, np.random.default_rng(6))
    tmp_db.save_strategy(genome, status=STATUS_VALIDATED, metrics={"stability": {"score": 0.9}})

    assert registry.promote(genome.genome_id) is False
    assert tmp_db.get_champion() is None


def test_flujo_completo_de_incubacion(config, tmp_db, catalog):
    from goldbot.autonomy.champion import ChampionRegistry
    from goldbot.strategies.genome import random_genome

    registry = ChampionRegistry(config, tmp_db)
    genome = random_genome(catalog, np.random.default_rng(7))
    tmp_db.save_strategy(genome, status=STATUS_VALIDATED, metrics={"stability": {"score": 0.9}})

    assert registry.start_incubation(genome.genome_id)
    assert tmp_db.get_strategy(genome.genome_id).status == STATUS_INCUBATING

    # Recien empezada: no puede estar lista.
    status = registry.check_incubation(genome.genome_id)
    assert not status.ready
    assert "dias" in status.blocked_reason

    # Ahora si se puede promocionar (viene de incubacion).
    assert registry.promote(genome.genome_id, reason="prueba")
    assert tmp_db.get_champion().id == genome.genome_id


def test_los_duplicados_no_se_registran_dos_veces(config, tmp_db, catalog):
    from goldbot.autonomy.champion import ChampionRegistry
    from goldbot.autonomy.stability import StabilityVerdict
    from goldbot.strategies.genome import random_genome

    registry = ChampionRegistry(config, tmp_db)
    genome = random_genome(catalog, np.random.default_rng(8))

    verdict = StabilityVerdict(genome_id=genome.genome_id, passed=True, score=0.8)

    assert registry.register_candidate(genome, verdict) == genome.genome_id
    # Un clon tiene id distinto pero la misma huella: debe rechazarse.
    assert registry.register_candidate(genome.clone(), verdict) is None


def test_promocion_exige_mejora_sustancial(config, tmp_db, catalog):
    """Un retador marginalmente mejor no debe destronar al campeon."""
    from goldbot.autonomy.champion import ChampionRegistry
    from goldbot.strategies.genome import random_genome

    registry = ChampionRegistry(config, tmp_db)
    rng = np.random.default_rng(9)

    champion = random_genome(catalog, rng)
    tmp_db.save_strategy(champion, status=STATUS_CHAMPION, metrics={"stability": {"score": 0.80}})

    challenger = random_genome(catalog, rng)
    # Solo un 2.5% mejor: por debajo del 15% exigido.
    tmp_db.save_strategy(challenger, status=STATUS_INCUBATING, metrics={"stability": {"score": 0.82}})
    tmp_db.set_state(f"incubation_start:{challenger.genome_id}", "2020-01-01T00:00:00+00:00")

    for i in range(config.stability.incubation_min_trades):
        trade_id = tmp_db.record_trade(
            strategy_id=challenger.genome_id, mode="paper",
            entry_time=f"2020-01-0{i % 9 + 1}T00:00:00+00:00",
            direction=1, entry_price=2000.0, lots=0.1,
        )
        tmp_db.close_trade(trade_id, f"2020-01-0{i % 9 + 1}T01:00:00+00:00", 2001.0, 10.0, "tp")

    decision = registry.evaluate_promotion()
    assert decision.action == "mantener", f"no deberia promocionar: {decision.reason}"


def test_degradacion_exige_deterioro_sostenido(config, tmp_db, catalog):
    from goldbot.autonomy.champion import ChampionRegistry
    from goldbot.strategies.genome import random_genome

    registry = ChampionRegistry(config, tmp_db)
    genome = random_genome(catalog, np.random.default_rng(10))
    tmp_db.save_strategy(
        genome, status=STATUS_CHAMPION,
        metrics={"stability": {"score": 0.8, "metrics": {"sharpe": 2.0}}},
    )

    # Un solo dia malo no basta.
    assert registry.evaluate_demotion(recent_sharpe=0.5, consecutive_bad_days=1).action == "mantener"

    # Deterioro sostenido si.
    decision = registry.evaluate_demotion(
        recent_sharpe=0.5, consecutive_bad_days=config.stability.consecutive_bad_days
    )
    assert decision.action == "retirar"


# --------------------------------------------------------------------------- #
# Deriva
# --------------------------------------------------------------------------- #
def test_sin_deriva_entre_muestras_identicas(features):
    from goldbot.ml.drift import DriftDetector

    half = len(features) // 2
    report = DriftDetector().detect(features.iloc[:half], features.iloc[half : half * 2])

    assert not report.covariate_drift
    assert report.recommendation == "sin_accion"


def test_deriva_detectada_al_desplazar_la_distribucion(features):
    from goldbot.ml.drift import DriftDetector

    half = len(features) // 2
    reference = features.iloc[:half]
    # Desplazamiento fuerte y artificial de todas las features.
    shifted = features.iloc[half:] + features.iloc[half:].std() * 4

    report = DriftDetector().detect(reference, shifted)
    assert report.covariate_drift, "deberia detectarse deriva de covariables"
    assert report.psi_total > 0.25
