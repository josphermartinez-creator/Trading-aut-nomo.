"""Pruebas del motor de backtesting: contabilidad, salidas y limites de riesgo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldbot.backtest.costs import CostModel
from goldbot.backtest.engine import BacktestEngine, ExitRules
from goldbot.backtest.metrics import compute_metrics


def test_sin_senales_no_hay_operaciones(config, ohlcv):
    engine = BacktestEngine(config)
    result = engine.run(ohlcv, pd.Series(0.0, index=ohlcv.index))

    assert result.metrics.total_trades == 0
    assert result.equity.iloc[-1] == pytest.approx(config.risk.initial_balance)


def test_la_equity_cuadra_con_el_pnl(config, ohlcv):
    """El saldo final debe ser exactamente el inicial mas la suma de P&L."""
    engine = BacktestEngine(config)

    rng = np.random.default_rng(3)
    signals = pd.Series(rng.choice([0.0, 1.0, -1.0], len(ohlcv), p=[0.97, 0.015, 0.015]), index=ohlcv.index)

    result = engine.run(ohlcv, signals, ExitRules(stop_atr=2.0, target_atr=3.0, max_bars=48))
    assert result.metrics.total_trades > 0

    expected = config.risk.initial_balance + result.trades["pnl"].sum()
    assert float(result.equity.iloc[-1]) == pytest.approx(expected, rel=1e-6)


def test_el_stop_limita_la_perdida(config, ohlcv):
    """Ninguna perdida individual debe superar mucho el riesgo planificado."""
    engine = BacktestEngine(config)

    rng = np.random.default_rng(5)
    signals = pd.Series(rng.choice([0.0, 1.0, -1.0], len(ohlcv), p=[0.98, 0.01, 0.01]), index=ohlcv.index)

    result = engine.run(ohlcv, signals, ExitRules(stop_atr=2.0, target_atr=4.0, max_bars=96))
    if result.trades.empty:
        pytest.skip("no se generaron operaciones")

    max_loss = float(result.trades["pnl"].min())
    # Margen amplio: los huecos de apertura pueden superar el stop teorico.
    budget = config.risk.initial_balance * config.risk.risk_per_trade * 3

    assert abs(max_loss) < budget, (
        f"perdida maxima {abs(max_loss):.2f} USD frente a un presupuesto de {budget:.2f}"
    )


def test_cortacircuito_por_drawdown(config, ohlcv):
    """Con un limite de drawdown minusculo, el motor debe detenerse."""
    import copy

    strict = copy.deepcopy(config)
    strict.risk.max_drawdown_pct = 0.005
    strict.risk.max_daily_loss_pct = 0.004
    strict.risk.risk_per_trade = 0.02

    engine = BacktestEngine(strict)
    signals = pd.Series(1.0, index=ohlcv.index)  # siempre largo
    result = engine.run(ohlcv, signals, ExitRules(stop_atr=1.0, target_atr=1.0, max_bars=12))

    if result.metrics.max_drawdown >= 0.005:
        assert result.halted, "deberia haberse activado el cortacircuitos"
        # Tras detenerse, la equity queda plana.
        tail = result.equity.iloc[-10:]
        assert tail.nunique() == 1


def test_limite_de_operaciones_diarias(config, ohlcv):
    """No se puede superar max_trades_per_day."""
    import copy

    limited = copy.deepcopy(config)
    limited.risk.max_trades_per_day = 2
    limited.risk.cooldown_bars_after_loss = 0

    engine = BacktestEngine(limited)
    signals = pd.Series(1.0, index=ohlcv.index)
    result = engine.run(ohlcv, signals, ExitRules(stop_atr=1.0, target_atr=1.0, max_bars=3))

    if result.trades.empty:
        pytest.skip("sin operaciones")

    per_day = result.trades.groupby(result.trades["entry_time"].dt.date).size()
    assert per_day.max() <= 2, f"se abrieron {per_day.max()} operaciones en un dia"


def test_los_costes_reducen_el_beneficio(config, ohlcv):
    """La misma senal debe rendir menos con costes altos que sin ellos."""
    import copy

    signals = pd.Series(0.0, index=ohlcv.index)
    signals.iloc[::200] = 1.0
    rules = ExitRules(stop_atr=2.0, target_atr=3.0, max_bars=48)

    free = copy.deepcopy(config)
    free.costs.spread_points = 0.0
    free.costs.commission_per_lot = 0.0
    free.costs.slippage_points = 0.0

    expensive = copy.deepcopy(config)
    expensive.costs.spread_points = 2.0
    expensive.costs.commission_per_lot = 30.0

    result_free = BacktestEngine(free).run(ohlcv, signals, rules)
    result_expensive = BacktestEngine(expensive).run(ohlcv, signals, rules)

    assert result_expensive.metrics.total_return < result_free.metrics.total_return


def test_coste_ida_y_vuelta():
    costs = CostModel(spread_points=0.30, commission_per_lot=7.0, slippage_points=0.10, contract_size=100.0)
    # spread 0.30 + 2x slippage 0.20 + comision 7/100 = 0.07 -> 0.57
    assert costs.round_trip_cost_points() == pytest.approx(0.57, abs=1e-9)


def test_operacion_ganadora_y_perdedora():
    costs = CostModel(spread_points=0.0, commission_per_lot=0.0, slippage_points=0.0, contract_size=100.0)
    # Largo de 1 lote, +10 USD/onza => +1000 USD
    assert costs.pnl(entry=2000.0, exit_=2010.0, lots=1.0, direction=1) == pytest.approx(1000.0)
    # Corto de 1 lote con el precio subiendo => -1000 USD
    assert costs.pnl(entry=2000.0, exit_=2010.0, lots=1.0, direction=-1) == pytest.approx(-1000.0)


def test_metricas_de_curva_creciente():
    """Una equity que crece de forma lineal debe dar R2 alto y drawdown nulo."""
    index = pd.date_range("2024-01-01", periods=3000, freq="5min", tz="UTC")
    equity = pd.Series(np.linspace(10_000, 12_000, 3000), index=index)

    metrics = compute_metrics(equity, None, initial_balance=10_000)

    assert metrics.total_return == pytest.approx(0.2, rel=1e-6)
    assert metrics.max_drawdown == pytest.approx(0.0, abs=1e-9)
    assert metrics.equity_r2 > 0.99


def test_metricas_con_drawdown():
    index = pd.date_range("2024-01-01", periods=300, freq="5min", tz="UTC")
    values = np.concatenate([
        np.linspace(10_000, 12_000, 100),
        np.linspace(12_000, 9_000, 100),   # -25% desde el maximo
        np.linspace(9_000, 11_000, 100),
    ])
    metrics = compute_metrics(pd.Series(values, index=index), None, initial_balance=10_000)

    assert metrics.max_drawdown == pytest.approx(0.25, rel=1e-3)
    assert metrics.max_drawdown_duration_bars > 50


def test_datos_insuficientes_no_revientan(config):
    engine = BacktestEngine(config)
    tiny = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
    )
    result = engine.run(tiny, pd.Series([0.0], index=tiny.index))
    assert result.metrics.total_trades == 0
