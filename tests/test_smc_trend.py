"""Pruebas de SMC, filtro de tendencia, instrumentos y Telegram.

La mas importante del fichero es :func:`test_los_swings_son_causales`: casi
todas las implementaciones de SMC que circulan marcan un swing en su propia
vela usando ``rolling(center=True)``, lo que introduce N barras de futuro en
cada senal. Sobre esa fuga, el motor evolutivo construye estrategias
espectaculares e irreproducibles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldbot.features import smc
from goldbot.features.trend import TREND_COLUMN, TrendFilter, apply_trend_veto


# --------------------------------------------------------------------------- #
# Causalidad de SMC
# --------------------------------------------------------------------------- #
def test_los_swings_son_causales(ohlcv):
    """Un swing confirmado no puede cambiar al conocer velas posteriores."""
    cut = len(ohlcv) // 2

    full = smc.swing_points(ohlcv["high"], ohlcv["low"])
    partial = smc.swing_points(ohlcv["high"].iloc[:cut], ohlcv["low"].iloc[:cut])

    for name, a, b in zip(
        ("maximo", "minimo", "confirma_max", "confirma_min"), full, partial, strict=True
    ):
        left = a.iloc[:cut]
        right = b
        valid = left.notna() & right.notna()
        assert valid.sum() > 50, f"{name}: muestra insuficiente"
        np.testing.assert_allclose(
            left[valid].to_numpy(dtype="float64"),
            right[valid].to_numpy(dtype="float64"),
            err_msg=f"El swing '{name}' NO es causal",
        )


def test_el_bloque_smc_completo_es_causal(ohlcv):
    from goldbot.features import indicators as ta

    cut = len(ohlcv) // 2
    o, h, lo, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
    atr = ta.atr(h, lo, c, 14)

    full = smc.compute_smc(o, h, lo, c, atr)
    partial = smc.compute_smc(
        o.iloc[:cut], h.iloc[:cut], lo.iloc[:cut], c.iloc[:cut], atr.iloc[:cut]
    )

    offenders = []
    for name, series in full.items():
        a, b = series.iloc[:cut], partial[name]
        valid = a.notna() & b.notna()
        if valid.sum() < 50:
            continue
        if not np.allclose(a[valid], b[valid], rtol=1e-7, atol=1e-9):
            offenders.append(name)

    assert not offenders, f"Features SMC con look-ahead: {offenders}"


def test_el_barrido_de_liquidez_exige_que_el_cierre_vuelva():
    """Un barrido no es una ruptura: el cierre debe regresar al lado de partida."""
    index = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    high = pd.Series([100, 100, 100, 100, 100.0], index=index)
    low = pd.Series([90, 90, 90, 85, 90.0], index=index)   # la barra 3 perfora
    close = pd.Series([95, 95, 95, 95, 95.0], index=index)  # y cierra dentro
    last_high = pd.Series([101.0] * 5, index=index)
    last_low = pd.Series([90.0] * 5, index=index)

    bull, bear = smc.liquidity_sweep(high, low, close, last_high, last_low)
    assert bull.iloc[3] == 1.0, "deberia detectarse el barrido bajo el minimo"
    assert bear.sum() == 0.0

    # Si el cierre se queda por debajo, es ruptura y no barrido.
    close_broken = pd.Series([95, 95, 95, 84, 84.0], index=index)
    bull2, _ = smc.liquidity_sweep(high, low, close_broken, last_high, last_low)
    assert bull2.iloc[3] == 0.0, "una ruptura real no es un barrido"


def test_premium_discount_esta_normalizado(ohlcv):
    last_high, last_low, _, _ = smc.swing_points(ohlcv["high"], ohlcv["low"])
    position = smc.premium_discount(ohlcv["close"], last_high, last_low).dropna()

    assert len(position) > 100
    assert position.between(-0.5, 1.5).all()


# --------------------------------------------------------------------------- #
# Filtro de tendencia
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["ema_stack", "ema_slope", "structure", "combined"])
def test_los_metodos_de_tendencia_devuelven_direcciones_validas(ohlcv, method):
    trend = TrendFilter(method=method).direction(ohlcv)

    assert len(trend) == len(ohlcv)
    assert set(np.unique(trend)).issubset({-1.0, 0.0, 1.0})
    # Debe identificar ambos lados: si solo ve una direccion, no sirve de filtro.
    assert (trend > 0).any() and (trend < 0).any()


def test_el_filtro_de_tendencia_es_causal(ohlcv):
    cut = len(ohlcv) // 2
    tf = TrendFilter(method="combined")

    full = tf.direction(ohlcv).iloc[:cut]
    partial = tf.direction(ohlcv.iloc[:cut])

    np.testing.assert_allclose(
        full.to_numpy(), partial.to_numpy(), err_msg="El filtro de tendencia mira al futuro"
    )


def test_el_veto_anula_las_entradas_a_contracorriente():
    index = pd.date_range("2024-01-01", periods=6, freq="5min", tz="UTC")
    signals = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], index=index)
    trend = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0, 0.0], index=index)

    out = apply_trend_veto(signals, trend)

    assert out.iloc[0] == 1.0    # largo en tendencia alcista: pasa
    assert out.iloc[1] == 0.0    # corto en tendencia alcista: vetado
    assert out.iloc[2] == 0.0    # largo en tendencia bajista: vetado
    assert out.iloc[3] == -1.0   # corto en tendencia bajista: pasa
    assert out.iloc[4] == 0.0    # sin tendencia: no se opera
    assert out.iloc[5] == 0.0


def test_la_columna_de_tendencia_no_es_visible_para_la_evolucion(config, ohlcv_raw):
    """El genetico no puede construir condiciones sobre el veto ni eliminarlo."""
    from goldbot.features.engineering import FeatureBuilder, build_features

    _, features, catalog = build_features(ohlcv_raw, FeatureBuilder.from_config(config))

    assert TREND_COLUMN in features.columns, "el veto debe estar disponible"
    assert TREND_COLUMN not in catalog, "pero NO debe aparecer en el catalogo"
    assert TREND_COLUMN not in catalog.names


def test_ninguna_estrategia_puede_operar_contra_la_tendencia(config, ohlcv_raw):
    """La prueba que respalda el requisito principal del usuario."""
    from goldbot.features.engineering import FeatureBuilder, build_features
    from goldbot.strategies.genome import random_genome

    _, features, catalog = build_features(ohlcv_raw, FeatureBuilder.from_config(config))
    trend = features[TREND_COLUMN]

    rng = np.random.default_rng(99)
    emitted = violations = 0

    for _ in range(30):
        genome = random_genome(catalog, rng, max_conditions=4)
        signals = genome.generate_signals(features)
        emitted += int((signals != 0).sum())
        violations += int(((signals > 0) & (trend <= 0)).sum())
        violations += int(((signals < 0) & (trend >= 0)).sum())

    assert emitted > 1000, "las estrategias apenas operaron; la prueba no concluye"
    assert violations == 0, f"{violations} entradas contra tendencia se colaron"


# --------------------------------------------------------------------------- #
# Instrumentos
# --------------------------------------------------------------------------- #
def test_resolucion_de_alias_de_instrumento():
    from goldbot.instruments import EURUSD, XAUUSD, get_instrument

    for alias in ("XAUUSD", "GOLD", "XAUUSD+", "XAUUSDm", "oro", "xauusd.a"):
        assert get_instrument(alias) is XAUUSD, alias
    for alias in ("EURUSD", "eurusd", "EUR/USD", "EURUSD+", "EURUSD.a"):
        assert get_instrument(alias) is EURUSD, alias

    with pytest.raises(ValueError):
        get_instrument("BITCOIN_LUNAR")


def test_los_rangos_de_precio_distinguen_los_instrumentos():
    from goldbot.instruments import EURUSD, XAUUSD

    assert XAUUSD.is_plausible_price(2650.0)
    assert not XAUUSD.is_plausible_price(1.085)

    assert EURUSD.is_plausible_price(1.085)
    assert not EURUSD.is_plausible_price(2650.0)


def test_el_contrato_equivocado_se_detecta(config):
    """El fallo que multiplicaria el tamano de las posiciones por mil."""
    import copy

    cfg = copy.deepcopy(config)
    cfg.data.symbol = "EURUSD"
    cfg.costs.contract_size = 100.0  # el del oro

    with pytest.raises(ValueError, match="contract_size"):
        cfg.validate()

    cfg.apply_instrument_defaults()
    cfg.validate()
    assert cfg.costs.contract_size == 100_000.0


def test_los_costes_se_adaptan_al_instrumento(config):
    import copy

    cfg = copy.deepcopy(config)
    cfg.data.symbol = "EURUSD"
    cfg.apply_instrument_defaults()

    # Un spread de 0.25 en EUR/USD serian 2500 pips: absurdo.
    assert cfg.costs.spread_points < 0.001
    assert cfg.instrument.name == "EURUSD"


def test_el_dimensionamiento_es_coherente_entre_instrumentos(config):
    """Mismo riesgo en dolares, tamanos de lote muy distintos."""
    import copy
    import datetime

    from goldbot.risk.manager import RiskManager

    oro = copy.deepcopy(config)
    oro.apply_instrument_defaults()
    m_oro = RiskManager(oro)
    m_oro.new_day(datetime.date(2024, 1, 1), 10_000.0)
    # ATR del oro ~2 USD sobre un precio de 2650.
    plan_oro = m_oro.build_plan(1, 2650.0, atr=2.0, equity=10_000.0, stop_atr=2.0, target_atr=4.0)

    euro = copy.deepcopy(config)
    euro.data.symbol = "EURUSD"
    euro.apply_instrument_defaults()
    m_euro = RiskManager(euro)
    m_euro.new_day(datetime.date(2024, 1, 1), 10_000.0)
    # ATR del euro ~0.0006 sobre un precio de 1.085.
    plan_euro = m_euro.build_plan(1, 1.085, atr=0.0006, equity=10_000.0, stop_atr=2.0, target_atr=4.0)

    assert plan_oro.approved and plan_euro.approved

    presupuesto = 10_000.0 * config.risk.risk_per_trade
    for plan in (plan_oro, plan_euro):
        assert plan.risk_amount <= presupuesto * 1.05


# --------------------------------------------------------------------------- #
# MT5
# --------------------------------------------------------------------------- #
def test_la_resolucion_de_simbolo_encuentra_el_alias_del_broker():
    """Simula el catalogo de XM (GOLD) y el de Vantage (XAUUSD+)."""
    from goldbot.data.mt5_provider import resolve_symbol
    from goldbot.instruments import EURUSD, XAUUSD

    class FakeSymbol:
        def __init__(self, name, bid):
            self.name = name
            self.bid = bid
            self.visible = True
            self.digits = 2
            self.point = 0.01
            self.trade_contract_size = 100.0
            self.volume_min, self.volume_max, self.volume_step = 0.01, 100.0, 0.01
            self.description = name
            self.spread = 25

    class FakeMT5:
        def __init__(self, catalog: dict[str, float]):
            self._catalog = {n: FakeSymbol(n, p) for n, p in catalog.items()}

        def symbol_info(self, name):
            return self._catalog.get(name)

        def symbol_info_tick(self, name):
            return self._catalog.get(name)

        def symbol_select(self, name, enable):
            return name in self._catalog

        def symbols_get(self):
            return list(self._catalog.values())

    # XM: el oro se llama GOLD.
    xm = FakeMT5({"GOLD": 2650.0, "EURUSD": 1.085, "USDJPY": 150.0})
    assert resolve_symbol(xm, XAUUSD).name == "GOLD"
    assert resolve_symbol(xm, EURUSD).name == "EURUSD"

    # Vantage: sufijo + en cuentas STP.
    vantage = FakeMT5({"XAUUSD+": 2650.0, "EURUSD+": 1.085})
    assert resolve_symbol(vantage, XAUUSD).name == "XAUUSD+"
    assert resolve_symbol(vantage, EURUSD).name == "EURUSD+"

    # Un simbolo con nombre correcto pero precio imposible se descarta: es la
    # senal de que el broker esta cotizando otra cosa.
    corrupto = FakeMT5({"XAUUSD": 1.085})
    assert resolve_symbol(corrupto, XAUUSD) is None


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def test_telegram_sin_configurar_no_hace_nada():
    from goldbot.notifications.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    assert not notifier.is_configured
    assert notifier.send("hola") is False
    # Los mensajes de dominio tampoco deben lanzar.
    notifier.trade_opened("XAUUSD", 1, 0.1, 2650.0, 2645.0, 2660.0, "abc")
    notifier.circuit_breaker("prueba", "detalle")


def test_telegram_escapa_el_html():
    from goldbot.notifications.telegram import esc

    assert esc("<script>") == "&lt;script&gt;"
    assert esc("a & b") == "a &amp; b"


def test_telegram_ignora_los_chats_no_autorizados():
    """El bot puede cerrar posiciones: solo obedece al chat configurado."""
    from goldbot.notifications.telegram import TelegramBot, TelegramNotifier

    notifier = TelegramNotifier(token="x", chat_id="12345", enabled=True)
    bot = TelegramBot(notifier)

    ejecutado = []
    bot.register("estado", lambda args: ejecutado.append(True) or "ok")

    # El manejador solo se invoca desde _handle, al que _poll_once solo llega
    # tras comprobar el chat_id. Se verifica que el comando existe y que la
    # comprobacion de autorizacion esta en el camino.
    import inspect

    source = inspect.getsource(bot._poll_once)
    assert "chat_id != str(self.notifier.chat_id)" in source
    assert "continue" in source
