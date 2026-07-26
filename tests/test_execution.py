"""Pruebas de la capa de ejecucion y del bucle en vivo."""

from __future__ import annotations

import pytest

from goldbot.execution.base import Order, OrderSide, OrderStatus
from goldbot.execution.paper import PaperBroker


# --------------------------------------------------------------------------- #
# Broker de papel
# --------------------------------------------------------------------------- #
def test_ciclo_completo_de_orden(config):
    broker = PaperBroker(config)
    broker.connect()
    broker.set_price(2000.0)

    order = Order(side=OrderSide.BUY, volume=0.1, symbol="XAUUSD", stop_loss=1990.0, take_profit=2020.0)
    filled = broker.place_order(order)

    assert filled.is_filled
    # Al comprar se paga el ask: por encima del medio.
    assert filled.filled_price > 2000.0

    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].side is OrderSide.BUY

    # Precio al alza: la posicion larga gana.
    broker.set_price(2010.0)
    close = broker.close_position(positions[0], "prueba")

    assert close.is_filled
    assert close.metadata["pnl"] > 0
    assert broker.balance > config.risk.initial_balance
    assert broker.get_positions() == []


def test_los_costes_hacen_perder_en_ida_y_vuelta_inmediata(config):
    """Abrir y cerrar al mismo precio debe perder dinero: se cruza el spread."""
    broker = PaperBroker(config)
    broker.connect()
    broker.set_price(2000.0)

    broker.place_order(Order(side=OrderSide.BUY, volume=1.0, symbol="XAUUSD"))
    broker.close_position(broker.get_positions()[0], "inmediato")

    assert broker.balance < config.risk.initial_balance


def test_orden_contraria_cierra_la_posicion(config):
    broker = PaperBroker(config)
    broker.connect()
    broker.set_price(2000.0)

    broker.place_order(Order(side=OrderSide.BUY, volume=0.1, symbol="XAUUSD"))
    broker.place_order(Order(side=OrderSide.SELL, volume=0.1, symbol="XAUUSD"))

    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].side is OrderSide.SELL, "deberia haber invertido la posicion"


def test_volumen_por_debajo_del_minimo_se_rechaza(config):
    broker = PaperBroker(config)
    broker.connect()
    broker.set_price(2000.0)

    order = broker.place_order(Order(side=OrderSide.BUY, volume=0.0001, symbol="XAUUSD"))
    assert order.status is OrderStatus.REJECTED


def test_stop_y_objetivo_en_la_misma_vela_asume_el_stop(config):
    """Ante la ambiguedad intrabarra, siempre el resultado pesimista."""
    broker = PaperBroker(config)
    broker.connect()
    broker.set_price(2000.0)

    broker.place_order(
        Order(side=OrderSide.BUY, volume=0.1, symbol="XAUUSD", stop_loss=1990.0, take_profit=2010.0)
    )
    # Vela que abarca ambos niveles.
    closed = broker.check_stops(high=2015.0, low=1985.0)

    assert len(closed) == 1
    assert closed[0].metadata["reason"] == "stop_loss"


def test_take_profit_se_dispara(config):
    broker = PaperBroker(config)
    broker.connect()
    broker.set_price(2000.0)

    broker.place_order(
        Order(side=OrderSide.BUY, volume=0.1, symbol="XAUUSD", stop_loss=1990.0, take_profit=2010.0)
    )
    closed = broker.check_stops(high=2012.0, low=1998.0)

    assert len(closed) == 1
    assert closed[0].metadata["reason"] == "take_profit"
    assert closed[0].metadata["pnl"] > 0


# --------------------------------------------------------------------------- #
# Salvaguarda dry_run
# --------------------------------------------------------------------------- #
def test_dry_run_fuerza_el_broker_de_papel(config):
    """Con dry_run activo jamas debe devolverse un broker real."""
    import copy

    from goldbot.execution import build_broker

    for mode in ("ccxt", "mt5"):
        cfg = copy.deepcopy(config)
        cfg.execution.mode = mode
        cfg.execution.dry_run = True

        broker = build_broker(cfg)
        assert isinstance(broker, PaperBroker), f"dry_run no protegio el modo {mode}"


# --------------------------------------------------------------------------- #
# Gestion de riesgo
# --------------------------------------------------------------------------- #
def test_el_tamano_respeta_el_riesgo_configurado(config):
    from goldbot.risk.manager import RiskManager

    manager = RiskManager(config)
    manager.new_day(__import__("datetime").date(2024, 1, 1), 10_000.0)

    plan = manager.build_plan(
        direction=1, entry_price=2000.0, atr=2.0, equity=10_000.0, stop_atr=2.0, target_atr=4.0
    )

    assert plan.approved
    objetivo = 10_000.0 * config.risk.risk_per_trade
    # El redondeo al paso de lote solo puede reducir el riesgo, nunca aumentarlo.
    assert plan.risk_amount <= objetivo * 1.01
    assert plan.reward_risk == pytest.approx(2.0, rel=0.01)


def test_el_limite_diario_bloquea_nuevas_operaciones(config):
    import datetime

    from goldbot.risk.manager import RiskManager

    manager = RiskManager(config)
    manager.new_day(datetime.date(2024, 1, 1), 10_000.0)

    # Perdida por encima del limite diario configurado.
    perdida = 10_000.0 * (config.risk.max_daily_loss_pct + 0.01)
    equity = 10_000.0 - perdida

    allowed, reason = manager.can_trade(equity)
    assert not allowed
    assert "diaria" in reason


def test_sin_atr_no_hay_operacion(config):
    import datetime

    from goldbot.risk.manager import RiskManager

    manager = RiskManager(config)
    manager.new_day(datetime.date(2024, 1, 1), 10_000.0)

    plan = manager.build_plan(direction=1, entry_price=2000.0, atr=float("nan"), equity=10_000.0)
    assert not plan.approved
    assert "ATR" in plan.reason


# --------------------------------------------------------------------------- #
# Cortacircuitos
# --------------------------------------------------------------------------- #
def test_el_cortacircuitos_salta_por_drawdown(config):
    from goldbot.risk.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(config)
    assert breaker.allows_trading

    equity = 10_000.0 * (1 - config.risk.max_drawdown_pct - 0.01)
    assert not breaker.check_equity(equity, peak_equity=10_000.0, day_start_equity=10_000.0)
    assert not breaker.allows_trading


def test_el_cortacircuitos_salta_por_racha_perdedora(config):
    from goldbot.risk.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(config)
    for _ in range(breaker.max_consecutive_losses):
        breaker.check_trade_result(-100.0)

    assert not breaker.allows_trading


def test_una_ganancia_reinicia_la_racha(config):
    from goldbot.risk.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(config)
    for _ in range(breaker.max_consecutive_losses - 1):
        breaker.check_trade_result(-100.0)

    breaker.check_trade_result(50.0)
    for _ in range(breaker.max_consecutive_losses - 1):
        breaker.check_trade_result(-100.0)

    assert breaker.allows_trading, "la racha deberia haberse reiniciado"


def test_el_cortacircuitos_detecta_precios_absurdos(config):
    from goldbot.risk.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(config)
    assert breaker.check_price_sanity(2000.0, 1999.0)

    breaker2 = CircuitBreaker(config)
    assert not breaker2.check_price_sanity(2200.0, 2000.0), "un salto del 10% debe parar el bot"

    breaker3 = CircuitBreaker(config)
    assert not breaker3.check_price_sanity(-5.0, 2000.0)


# --------------------------------------------------------------------------- #
# Bucle en vivo
# --------------------------------------------------------------------------- #
def test_el_runner_opera_con_un_campeon(config, tmp_path, catalog, ohlcv_raw, monkeypatch):
    """El bucle completo debe ejecutar ciclos y registrar equity sin reventar."""
    import copy

    from goldbot.live.runner import LiveRunner
    from goldbot.storage.db import STATUS_CHAMPION, Database
    from goldbot.strategies.genome import Condition, RuleSet, StrategyGenome

    cfg = copy.deepcopy(config)
    cfg.db_path = str(tmp_path / "live.db")
    cfg.execution.mode = "paper"
    cfg.data.providers = ["synthetic"]

    db = Database(cfg.db_path)

    # Campeon que opera con frecuencia, para que el ciclo haga algo real.
    genome = StrategyGenome(
        long_rules=RuleSet([Condition("rsi_14", "lt", threshold=45.0)], "and"),
        short_rules=RuleSet([Condition("rsi_14", "gt", threshold=55.0)], "and"),
    )
    db.save_strategy(genome, status=STATUS_CHAMPION, metrics={"stability": {"score": 0.8}})

    runner = LiveRunner(cfg, db)
    # Se evita la red: el runner usa el historico sintetico ya generado.
    monkeypatch.setattr(runner.market_data, "update", lambda *a, **k: ohlcv_raw)

    assert runner.broker.connect()
    runner._load_champion()
    assert runner._champion is not None

    runner.state.running = True
    runner.run_cycle()

    assert runner.state.last_bar_time is not None
    curve = db.get_equity_curve(genome.genome_id, "paper")
    assert len(curve) == 1, "el ciclo debe registrar un punto de equity"

    # Un segundo ciclo con la misma vela no debe duplicar trabajo.
    runner.run_cycle()
    assert len(db.get_equity_curve(genome.genome_id, "paper")) == 1


def test_el_runner_sin_campeon_no_revienta(config, tmp_path, ohlcv_raw, monkeypatch):
    """Sin campeon el bot debe seguir vivo, limitandose a incubar."""
    import copy

    from goldbot.live.runner import LiveRunner
    from goldbot.storage.db import Database

    cfg = copy.deepcopy(config)
    cfg.db_path = str(tmp_path / "empty.db")
    cfg.execution.mode = "paper"

    runner = LiveRunner(cfg, Database(cfg.db_path))
    monkeypatch.setattr(runner.market_data, "update", lambda *a, **k: ohlcv_raw)

    runner.broker.connect()
    runner._load_champion()
    assert runner._champion is None

    runner.state.running = True
    runner.run_cycle()  # no debe lanzar
