"""Fixtures compartidas."""

from __future__ import annotations

from datetime import timedelta

import pytest

from goldbot.config import Config
from goldbot.data.providers import SyntheticProvider
from goldbot.features.engineering import FeatureBuilder, build_features
from goldbot.utils.timeutils import now_utc


@pytest.fixture(scope="session")
def config() -> Config:
    """Configuracion de pruebas: rapida y determinista."""
    cfg = Config()
    cfg.evolution.population_size = 20
    cfg.evolution.generations = 4
    cfg.evolution.min_trades = 10
    cfg.evolution.random_seed = 123
    cfg.backtest.monte_carlo_runs = 100
    cfg.backtest.walk_forward_folds = 3
    cfg.data.min_bars_required = 1000
    cfg.optuna.n_trials = 10
    cfg.optuna.timeout_seconds = 30
    return cfg


@pytest.fixture(scope="session")
def ohlcv_raw():
    """90 dias de velas sinteticas M5."""
    end = now_utc()
    return SyntheticProvider(seed=7).fetch(end - timedelta(days=90), end, "5m")


@pytest.fixture(scope="session")
def dataset(ohlcv_raw):
    """(ohlcv, features, catalogo) ya alineados."""
    return build_features(ohlcv_raw, FeatureBuilder())


@pytest.fixture(scope="session")
def ohlcv(dataset):
    return dataset[0]


@pytest.fixture(scope="session")
def features(dataset):
    return dataset[1]


@pytest.fixture(scope="session")
def catalog(dataset):
    return dataset[2]


@pytest.fixture
def tmp_db(tmp_path):
    """Base de datos limpia por test."""
    from goldbot.storage.db import Database

    return Database(tmp_path / "test.db")
