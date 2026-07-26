"""Construccion de la matriz de features y su catalogo semantico.

Aqui no solo se calculan columnas: cada feature se registra con metadatos
(:class:`FeatureSpec`) que describen su *tipo* y su rango util. Ese catalogo es
lo que permite al algoritmo genetico inventar reglas con sentido -- comparar el
RSI con 70 es razonable, compararlo con el precio del oro no lo es.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldbot.features import indicators as ta
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import in_overlap, in_session

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Catalogo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    """Descripcion de una feature para que el GA sepa como usarla."""

    name: str
    kind: str          # bounded_100 | bounded_1 | zscore | signed | ratio | binary
    group: str         # trend | momentum | volatility | volume | structure | time
    low: float         # extremo bajo tipico (para muestrear umbrales)
    high: float        # extremo alto tipico
    comparable: bool = True   # si puede compararse contra otra feature del mismo kind

    def sample_threshold(self, rng: np.random.Generator) -> float:
        """Umbral aleatorio plausible para esta feature."""
        if self.kind == "binary":
            return 0.5
        return float(rng.uniform(self.low, self.high))


class FeatureCatalog:
    """Indice de features disponibles, consultable por tipo o por grupo."""

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, name: str) -> FeatureSpec | None:
        return self._specs.get(name)

    def by_kind(self, kind: str) -> list[FeatureSpec]:
        return [s for s in self._specs.values() if s.kind == kind]

    def by_group(self, group: str) -> list[FeatureSpec]:
        return [s for s in self._specs.values() if s.group == group]

    def comparable_with(self, name: str) -> list[str]:
        """Features contra las que ``name`` puede compararse directamente."""
        spec = self._specs.get(name)
        if spec is None or not spec.comparable:
            return []
        return [s.name for s in self._specs.values() if s.kind == spec.kind and s.name != name]

    def restrict(self, available: list[str]) -> FeatureCatalog:
        """Sub-catalogo con las features realmente presentes en un DataFrame."""
        keep = set(available)
        return FeatureCatalog([s for s in self._specs.values() if s.name in keep])


# --------------------------------------------------------------------------- #
# Constructor
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBuilder:
    """Calcula todas las features y devuelve tambien su catalogo.

    ``fast_periods`` / ``slow_periods`` definen la rejilla de ventanas. Cuantas
    mas, mas rico es el espacio de busqueda del GA, pero mas lento todo.
    """

    fast_periods: tuple[int, ...] = (5, 9, 14)
    slow_periods: tuple[int, ...] = (21, 50, 100)
    include_time: bool = True
    include_multiscale: bool = True
    _specs: list[FeatureSpec] = field(default_factory=list, init=False, repr=False)

    # -- helpers de registro -------------------------------------------- #
    def _add(
        self,
        out: pd.DataFrame,
        name: str,
        series: pd.Series,
        kind: str,
        group: str,
        low: float,
        high: float,
        comparable: bool = True,
    ) -> None:
        out[name] = series.astype("float64")
        self._specs.append(FeatureSpec(name, kind, group, low, high, comparable))

    # -- API ------------------------------------------------------------- #
    def build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureCatalog]:
        """Devuelve (features, catalogo). El indice coincide con ``df``."""
        if df.empty:
            return pd.DataFrame(index=df.index), FeatureCatalog([])

        self._specs = []
        o, h, lo, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
        out = pd.DataFrame(index=df.index)

        self._build_trend(out, o, h, lo, c)
        self._build_momentum(out, o, h, lo, c)
        self._build_volatility(out, o, h, lo, c)
        self._build_volume(out, h, lo, c, v)
        self._build_structure(out, o, h, lo, c)
        if self.include_multiscale:
            self._build_multiscale(out, c)
        if self.include_time:
            self._build_time(out, df.index)

        # Infinitos: aparecen en divisiones por rangos nulos. Se tratan como NaN.
        out = out.replace([np.inf, -np.inf], np.nan)
        catalog = FeatureCatalog(self._specs)
        logger.debug("Construidas %d features", out.shape[1])
        return out, catalog

    # -- bloques ---------------------------------------------------------- #
    def _build_trend(self, out, o, h, lo, c) -> None:
        for p in (*self.fast_periods, *self.slow_periods):
            # Distancia relativa a la media: escala-invariante, comparable entre epocas.
            self._add(out, f"ema_dist_{p}", (c - ta.ema(c, p)) / c, "ratio", "trend", -0.01, 0.01)
            self._add(out, f"sma_dist_{p}", (c - ta.sma(c, p)) / c, "ratio", "trend", -0.01, 0.01)

        for fast in self.fast_periods:
            for slow in self.slow_periods:
                if fast >= slow:
                    continue
                spread = (ta.ema(c, fast) - ta.ema(c, slow)) / c
                self._add(out, f"ema_cross_{fast}_{slow}", spread, "ratio", "trend", -0.008, 0.008)

        self._add(out, "hma_dist_21", (c - ta.hma(c, 21)) / c, "ratio", "trend", -0.01, 0.01)
        self._add(out, "kama_dist_20", (c - ta.kama(c, 20)) / c, "ratio", "trend", -0.01, 0.01)

        for p in (14, 25):
            adx_line, plus_di, minus_di = ta.adx(h, lo, c, p)
            self._add(out, f"adx_{p}", adx_line, "bounded_100", "trend", 15, 45)
            self._add(out, f"di_diff_{p}", plus_di - minus_di, "signed", "trend", -25, 25)

        for p, m in ((10, 3.0), (20, 2.0)):
            _, direction = ta.supertrend(h, lo, c, p, m)
            self._add(out, f"supertrend_dir_{p}", direction, "signed", "trend", -1, 1, comparable=False)

        for p in (20, 50):
            self._add(out, f"linreg_slope_{p}", ta.linreg_slope(c, p), "signed", "trend", -2e-4, 2e-4)

        aroon_up, aroon_down = ta.aroon(h, lo, 25)
        self._add(out, "aroon_osc", aroon_up - aroon_down, "signed", "trend", -80, 80)

        tenkan, kijun, span_a, span_b = ta.ichimoku(h, lo)
        self._add(out, "ichimoku_tk_diff", (tenkan - kijun) / c, "ratio", "trend", -0.006, 0.006)
        self._add(out, "ichimoku_cloud_pos", (c - (span_a + span_b) / 2) / c, "ratio", "trend", -0.01, 0.01)

    def _build_momentum(self, out, o, h, lo, c) -> None:
        for p in (7, 14, 21):
            self._add(out, f"rsi_{p}", ta.rsi(c, p), "bounded_100", "momentum", 20, 80)

        for p in (14, 21):
            k, d = ta.stochastic(h, lo, c, p)
            self._add(out, f"stoch_k_{p}", k, "bounded_100", "momentum", 15, 85)
            self._add(out, f"stoch_diff_{p}", k - d, "signed", "momentum", -15, 15)

        self._add(out, "williams_r_14", ta.williams_r(h, lo, c, 14), "signed", "momentum", -90, -10)
        self._add(out, "cci_20", ta.cci(h, lo, c, 20), "signed", "momentum", -150, 150)
        self._add(out, "tsi", ta.tsi(c), "signed", "momentum", -30, 30)

        macd_line, macd_signal, macd_hist = ta.macd(c)
        self._add(out, "macd_norm", macd_line / c, "ratio", "momentum", -0.004, 0.004)
        self._add(out, "macd_hist_norm", macd_hist / c, "ratio", "momentum", -0.002, 0.002)
        self._add(out, "macd_above_signal", (macd_line > macd_signal).astype("float64"),
                  "binary", "momentum", 0, 1, comparable=False)

        for p in (5, 10, 20):
            self._add(out, f"roc_{p}", ta.roc(c, p), "signed", "momentum", -0.8, 0.8)

        for p in (20, 50):
            self._add(out, f"ret_z_{p}", ta.zscore(c.pct_change(), p), "zscore", "momentum", -2.5, 2.5)

    def _build_volatility(self, out, o, h, lo, c) -> None:
        for p in (14, 50):
            atr_ = ta.atr(h, lo, c, p)
            # ATR normalizado por precio: comparable con el oro a 1800 o a 3000.
            self._add(out, f"atr_pct_{p}", atr_ / c, "ratio", "volatility", 0.0005, 0.004)
            self._add(out, f"atr_rank_{p}", ta.percent_rank(atr_, 200), "bounded_1", "volatility", 0.2, 0.8)

        for p, m in ((20, 2.0), (50, 2.5)):
            self._add(out, f"bb_pct_{p}", ta.bollinger_pct(c, p, m), "bounded_1", "volatility", 0.05, 0.95)
            self._add(out, f"bb_width_{p}", ta.bollinger_width(c, p, m), "ratio", "volatility", 0.002, 0.02)

        self._add(out, "bb_squeeze", ta.percent_rank(ta.bollinger_width(c, 20, 2.0), 120),
                  "bounded_1", "volatility", 0.05, 0.5)

        upper_k, _, lower_k = ta.keltner(h, lo, c, 20, 2.0)
        self._add(out, "keltner_pos", (c - lower_k) / (upper_k - lower_k).replace(0, np.nan),
                  "bounded_1", "volatility", 0.05, 0.95)

        for p in (20, 60):
            self._add(out, f"realized_vol_{p}", ta.realized_volatility(c, p),
                      "ratio", "volatility", 0.0003, 0.003)
        self._add(out, "parkinson_vol_20", ta.parkinson_volatility(h, lo, 20),
                  "ratio", "volatility", 0.0003, 0.003)
        self._add(out, "vol_of_vol", ta.zscore(ta.realized_volatility(c, 20), 100),
                  "zscore", "volatility", -2, 2)

    def _build_volume(self, out, h, lo, c, v) -> None:
        has_volume = float(v.sum()) > 0
        if not has_volume:
            # Algunos feeds de XAUUSD no traen volumen: no inventamos columnas
            # que serian constantes y solo servirian para sobreajustar.
            logger.debug("Sin volumen en el feed: se omiten features de volumen")
            return

        self._add(out, "volume_z_50", ta.volume_zscore(v, 50), "zscore", "volume", -1.5, 2.5)
        self._add(out, "mfi_14", ta.mfi(h, lo, c, v, 14), "bounded_100", "volume", 20, 80)

        obv_ = ta.obv(c, v)
        self._add(out, "obv_slope", ta.linreg_slope(obv_.abs() + 1, 20), "signed", "volume", -0.02, 0.02)

        vwap = ta.vwap_session(h, lo, c, v)
        self._add(out, "vwap_dist", (c - vwap) / c, "ratio", "volume", -0.005, 0.005)

    def _build_structure(self, out, o, h, lo, c) -> None:
        self._add(out, "body_ratio", ta.candle_body_ratio(o, h, lo, c), "bounded_1", "structure", 0.1, 0.8)
        self._add(out, "upper_wick", ta.upper_wick_ratio(o, h, lo, c), "bounded_1", "structure", 0.1, 0.7)
        self._add(out, "lower_wick", ta.lower_wick_ratio(o, h, lo, c), "bounded_1", "structure", 0.1, 0.7)

        for p in (20, 50):
            upper, _, lower = ta.donchian(h, lo, p)
            span = (upper - lower).replace(0, np.nan)
            self._add(out, f"donchian_pos_{p}", (c - lower) / span, "bounded_1", "structure", 0.05, 0.95)

        for p in (50, 200):
            self._add(out, f"dist_high_{p}", ta.distance_from_high(c, p), "ratio", "structure", -0.03, 0.0)
            self._add(out, f"dist_low_{p}", ta.distance_from_low(c, p), "ratio", "structure", 0.0, 0.03)

        self._add(out, "fractal_eff_30", ta.fractal_dimension(h, lo, 30), "bounded_1", "structure", 0.15, 0.6)
        self._add(out, "hurst_100", ta.hurst_exponent(c, 100), "bounded_1", "structure", 0.4, 0.6)
        self._add(out, "rolling_sharpe_100", ta.rolling_sharpe(c.pct_change(), 100),
                  "zscore", "structure", -0.15, 0.15)

        # Gap respecto al cierre anterior: en el oro marca reacciones a noticias.
        self._add(out, "gap_pct", (o - c.shift(1)) / c.shift(1), "ratio", "structure", -0.002, 0.002)

        for p in (3, 5):
            streak = np.sign(c.diff()).rolling(p, min_periods=p).sum()
            self._add(out, f"streak_{p}", streak, "signed", "structure", -float(p), float(p))

    def _build_multiscale(self, out, c: pd.Series) -> None:
        """Contexto de temporalidades superiores, sin salir del indice M5.

        Se calcula con ventanas equivalentes (12 barras M5 = 1h, 288 = 1 dia)
        en lugar de remuestrear y reindexar, que es donde suele colarse el
        look-ahead.
        """
        for label, bars in (("h1", 12), ("h4", 48), ("d1", 288)):
            self._add(out, f"trend_{label}", (c - ta.ema(c, bars)) / c, "ratio", "trend", -0.015, 0.015)
            self._add(out, f"rsi_{label}", ta.rsi(c, bars), "bounded_100", "momentum", 25, 75)

    def _build_time(self, out: pd.DataFrame, index: pd.DatetimeIndex) -> None:
        hour = index.hour.to_numpy(dtype="float64")
        # Codificacion ciclica: las 23:00 y las 00:00 quedan contiguas.
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        self._specs.append(FeatureSpec("hour_sin", "signed", "time", -1, 1, comparable=False))
        self._specs.append(FeatureSpec("hour_cos", "signed", "time", -1, 1, comparable=False))

        for session in ("tokyo", "london", "newyork"):
            name = f"in_{session}"
            out[name] = in_session(index, session).astype("float64")
            self._specs.append(FeatureSpec(name, "binary", "time", 0, 1, comparable=False))

        out["in_overlap"] = in_overlap(index).astype("float64")
        self._specs.append(FeatureSpec("in_overlap", "binary", "time", 0, 1, comparable=False))

        dow = index.dayofweek.to_numpy(dtype="float64")
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        self._specs.append(FeatureSpec("dow_sin", "signed", "time", -1, 1, comparable=False))
        self._specs.append(FeatureSpec("dow_cos", "signed", "time", -1, 1, comparable=False))


def build_features(
    df: pd.DataFrame, builder: FeatureBuilder | None = None, dropna: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureCatalog]:
    """Construye features y alinea el OHLCV con ellas.

    Devuelve ``(ohlcv_alineado, features, catalogo)``. Si ``dropna`` esta
    activo se recortan las primeras barras (las que aun no tienen ventana
    completa), garantizando que ninguna feature entre al modelo a medio
    calentar.
    """
    builder = builder or FeatureBuilder()
    features, catalog = builder.build(df)

    if dropna and not features.empty:
        # Descartamos columnas casi enteramente vacias antes de recortar filas:
        # de lo contrario una sola feature mala se lleva por delante el dataset.
        valid_ratio = features.notna().mean()
        weak = valid_ratio[valid_ratio < 0.5].index.tolist()
        if weak:
            logger.debug("Descartadas features con demasiados NaN: %s", weak)
            features = features.drop(columns=weak)

        before = len(features)
        features = features.dropna()
        logger.debug("Recorte por warm-up: %d -> %d barras", before, len(features))

    aligned = df.loc[features.index]
    catalog = catalog.restrict(list(features.columns))
    return aligned, features, catalog
