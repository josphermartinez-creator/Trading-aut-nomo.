"""Pruebas de ausencia de look-ahead: las mas importantes del proyecto.

Un backtest con fuga de informacion futura no es un backtest optimista, es
ficcion. Y el algoritmo genetico es un buscador de fugas extraordinariamente
eficaz: si existe una, la encontrara y construira toda su "estrategia" sobre
ella. Durante el desarrollo de este motor el GA encontro tres.

La bateria se apoya en dos controles:

* **Control negativo**: sobre un paseo aleatorio con costes, ninguna estrategia
  puede ganar de forma sistematica. Si alguna lo hace, hay una fuga.
* **Control positivo**: una estrategia con acceso explicito al futuro *debe*
  ganar mucho. Si no gana, el motor esta roto en el otro sentido y las pruebas
  negativas no valdrian nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from goldbot.backtest.engine import BacktestEngine, ExitRules
from goldbot.features import indicators as ta


def test_indicadores_son_causales(ohlcv):
    """Ningun indicador puede cambiar de valor al anadir barras futuras.

    Se calcula cada indicador sobre la serie completa y sobre un prefijo; los
    valores del prefijo deben coincidir exactamente. Si difieren, el indicador
    esta mirando hacia adelante.
    """
    cut = len(ohlcv) // 2
    prefix = ohlcv.iloc[:cut]

    checks = {
        "ema_21": lambda d: ta.ema(d["close"], 21),
        "sma_50": lambda d: ta.sma(d["close"], 50),
        "rsi_14": lambda d: ta.rsi(d["close"], 14),
        "atr_14": lambda d: ta.atr(d["high"], d["low"], d["close"], 14),
        "macd": lambda d: ta.macd(d["close"])[0],
        "bb_pct": lambda d: ta.bollinger_pct(d["close"], 20),
        "adx": lambda d: ta.adx(d["high"], d["low"], d["close"], 14)[0],
        "supertrend": lambda d: ta.supertrend(d["high"], d["low"], d["close"])[1],
        "donchian": lambda d: ta.donchian(d["high"], d["low"], 20)[0],
        "zscore": lambda d: ta.zscore(d["close"], 20),
        "kama": lambda d: ta.kama(d["close"], 20),
        "linreg": lambda d: ta.linreg_slope(d["close"], 20),
    }

    for name, fn in checks.items():
        full = fn(ohlcv).iloc[:cut]
        partial = fn(prefix)

        both_valid = full.notna() & partial.notna()
        assert both_valid.sum() > 100, f"{name}: muy pocos valores validos para comparar"

        np.testing.assert_allclose(
            full[both_valid].to_numpy(),
            partial[both_valid].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{name} NO es causal: cambia al conocer el futuro",
        )


def test_features_son_causales(ohlcv_raw):
    """La matriz de features completa debe ser causal en todas sus columnas."""
    from goldbot.features.engineering import FeatureBuilder

    builder = FeatureBuilder()
    cut = len(ohlcv_raw) // 2

    full, _ = builder.build(ohlcv_raw)
    partial, _ = builder.build(ohlcv_raw.iloc[:cut])

    common_index = full.index[:cut].intersection(partial.index)
    # El VWAP se reinicia por sesion: al cortar a mitad de dia el ultimo dia
    # queda truncado, asi que se compara todo salvo esa jornada parcial.
    last_day = partial.index[-1].normalize()
    comparable = common_index[common_index < last_day]

    assert len(comparable) > 500, "muestra insuficiente para validar causalidad"

    offenders = []
    for column in full.columns:
        if column not in partial.columns:
            continue
        a = full.loc[comparable, column]
        b = partial.loc[comparable, column]
        valid = a.notna() & b.notna()
        if valid.sum() < 50:
            continue
        if not np.allclose(a[valid], b[valid], rtol=1e-7, atol=1e-9):
            offenders.append(column)

    assert not offenders, f"Features con look-ahead: {offenders}"


def test_control_negativo_paseo_aleatorio(config):
    """Sobre un paseo aleatorio puro con costes, nadie gana de forma sistematica.

    Se prueban muchas estrategias aleatorias; con costes reales la mediana debe
    perder dinero. Si la mediana ganase, el motor estaria regalando dinero.
    """
    from goldbot.features.engineering import build_features
    from goldbot.strategies.genome import random_genome

    rng = np.random.default_rng(42)
    n = 30_000

    # Paseo aleatorio estricto: sin tendencia, sin reversion, sin estructura.
    index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    returns = rng.normal(0, 0.0008, n)
    close = 2000 * np.exp(np.cumsum(returns))
    noise = np.abs(rng.normal(0, 0.0004, n)) * close
    frame = pd.DataFrame(
        {
            "open": np.concatenate([[2000.0], close[:-1]]),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": rng.gamma(2.0, 500.0, n),
        },
        index=index,
    )

    aligned, feats, cat = build_features(frame)
    engine = BacktestEngine(config)

    returns_observed = []
    for _ in range(40):
        genome = random_genome(cat, rng, max_conditions=3)
        result = engine.run(aligned, genome.generate_signals(feats), genome.exit_rules)
        if result.metrics.total_trades >= 20:
            returns_observed.append(result.metrics.total_return)

    assert len(returns_observed) >= 10, "muy pocas estrategias operaron para concluir"

    median_return = float(np.median(returns_observed))
    assert median_return < 0, (
        f"FUGA DETECTADA: la estrategia mediana gana {median_return:.2%} sobre un paseo "
        f"aleatorio. Con costes eso es imposible sin look-ahead."
    )


def test_control_positivo_estrategia_oraculo(config, ohlcv, features):
    """Una estrategia que conoce el futuro DEBE ganar mucho.

    Es la contraprueba del control negativo: confirma que el motor si permite
    ganar cuando existe una ventaja real, de modo que un resultado negativo en
    las otras pruebas signifique algo.
    """
    engine = BacktestEngine(config)

    # Senal oraculo: mira el retorno de las 12 barras siguientes.
    future_return = ohlcv["close"].shift(-12) / ohlcv["close"] - 1
    oracle = pd.Series(np.sign(future_return), index=ohlcv.index).fillna(0.0)

    result = engine.run(
        ohlcv, oracle, ExitRules(stop_atr=3.0, target_atr=3.0, max_bars=12, exit_on_reverse=False)
    )

    assert result.metrics.total_trades > 50, "el oraculo apenas opero"
    assert result.metrics.total_return > 0.10, (
        f"El oraculo solo gano {result.metrics.total_return:.2%}. "
        f"El motor podria estar impidiendo ganar incluso con informacion perfecta."
    )
    assert result.metrics.win_rate > 0.6, "el oraculo deberia acertar la mayoria"


def test_senal_se_ejecuta_en_la_barra_siguiente(config, ohlcv):
    """La senal de la barra t se ejecuta en la apertura de t+1, nunca antes."""
    engine = BacktestEngine(config)

    # Una unica senal, en una barra concreta.
    target = 500
    signals = pd.Series(0.0, index=ohlcv.index)
    signals.iloc[target] = 1.0

    result = engine.run(ohlcv, signals, ExitRules(max_bars=5, exit_on_reverse=False))

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == ohlcv.index[target + 1], (
        "la entrada debe producirse en la barra SIGUIENTE a la senal"
    )

    # Y el precio debe partir de la apertura de esa barra (mas costes).
    expected_open = float(ohlcv["open"].iloc[target + 1])
    assert trade["entry_price"] >= expected_open, "la compra debe pagar el ask, no el mid"


def test_stop_no_puede_ser_rentable(config, ohlcv, features):
    """Una salida etiquetada 'stop_loss' nunca debe ganar dinero de media.

    Este es exactamente el sintoma que delato la fuga del trailing stop durante
    el desarrollo: cientos de operaciones cerradas en stop, con beneficio medio
    positivo.
    """
    from goldbot.strategies.seeds import seed_population

    rng = np.random.default_rng(11)
    engine = BacktestEngine(config)

    for genome in seed_population(catalog_from(features), 12, rng):
        result = engine.run(ohlcv, genome.generate_signals(features), genome.exit_rules)
        trades = result.trades
        if trades.empty:
            continue

        stops = trades[trades["exit_reason"] == "stop_loss"]
        if len(stops) < 10:
            continue

        assert stops["pnl"].mean() < 0, (
            f"Las salidas por stop_loss ganan {stops['pnl'].mean():.2f} USD de media "
            f"en el genoma {genome.genome_id}. Un stop rentable indica una fuga."
        )


def test_trailing_minimo_se_respeta():
    """Un trailing por debajo del minimo se desactiva en lugar de aplicarse.

    Sin este limite, el optimizador elige trailings sub-ruido que equivalen a
    "vender en el maximo de la vela": imposible de ejecutar en real.
    """
    rules = ExitRules(stop_atr=2.0, target_atr=3.0, trail_atr=0.16).validate()
    assert rules.trail_atr == 0.0, "un trailing sub-minimo debe quedar desactivado"

    rules = ExitRules(stop_atr=0.1, target_atr=3.0).validate()
    assert rules.stop_atr >= ExitRules.MIN_STOP_ATR, "el stop debe respetar el minimo"


def catalog_from(features):
    """Reconstruye el catalogo a partir de las columnas presentes."""
    from goldbot.features.engineering import FeatureBuilder

    builder = FeatureBuilder()
    dummy = pd.DataFrame(
        {
            "open": [1.0] * 400, "high": [1.1] * 400,
            "low": [0.9] * 400, "close": [1.0] * 400, "volume": [1.0] * 400,
        },
        index=pd.date_range("2024-01-01", periods=400, freq="5min", tz="UTC"),
    )
    _, cat = builder.build(dummy)
    return cat.restrict(list(features.columns))
