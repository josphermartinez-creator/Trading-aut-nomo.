"""Entrenamiento diario del meta-etiquetador.

Orquesta el ciclo completo: genera las senales de la estrategia, las etiqueta
con triple barrera, entrena el clasificador con un split purgado y decide si el
modelo resultante merece usarse.

El punto sutil: **solo se entrena sobre las barras en las que la estrategia da
senal**. Entrenar sobre todas las barras diluiria el problema con miles de
momentos en los que no se iba a operar de todos modos.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goldbot.config import Config
from goldbot.features.labeling import purged_train_test_split, triple_barrier_labels
from goldbot.ml.models import MetaLabeler, ModelReport
from goldbot.storage.db import Database
from goldbot.strategies.base import Strategy
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Salida del entrenamiento diario."""

    labeler: MetaLabeler | None
    report: ModelReport
    model_path: Path | None = None
    n_signals: int = 0

    @property
    def is_useful(self) -> bool:
        return self.labeler is not None and self.labeler.is_ready

    def summary(self) -> str:
        return f"{self.report.summary()} | senales etiquetadas={self.n_signals}"


class MLTrainer:
    """Entrena y persiste el meta-etiquetador de una estrategia."""

    def __init__(self, config: Config, db: Database | None = None) -> None:
        self.config = config
        self.db = db
        self.artifacts_dir = config.path(config.artifacts_dir) / "models"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def train_for_strategy(
        self,
        strategy: Strategy,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        strategy_id: str | None = None,
        save: bool = True,
    ) -> TrainingResult:
        """Entrena el filtro de calidad para las senales de ``strategy``."""
        cfg = self.config.ml
        if not cfg.enabled:
            return TrainingResult(None, ModelReport(reason="ML desactivado en configuracion"))

        signals = strategy.generate_signals(features, ohlcv)
        active = signals[signals != 0]

        if len(active) < 200:
            report = ModelReport(reason=f"solo {len(active)} senales; insuficientes para entrenar")
            logger.info("Sin ML para %s: %s", getattr(strategy, "name", "?"), report.reason)
            return TrainingResult(None, report, n_signals=len(active))

        # Meta-etiquetado: ¿esta senal concreta habria ganado dinero?
        # Se exige superar el coste de ida y vuelta, no solo quedar en positivo.
        from goldbot.backtest.costs import CostModel

        costs = CostModel.from_config(self.config.costs)
        min_return = costs.round_trip_cost_points() / float(ohlcv["close"].median())

        labels_result = triple_barrier_labels(
            ohlcv=ohlcv,
            lookahead_bars=cfg.lookahead_bars,
            profit_take_atr=cfg.profit_take_atr,
            stop_loss_atr=cfg.stop_loss_atr,
            side=signals,
            min_return=min_return,
        )

        # Solo las barras con senal entran al entrenamiento.
        mask = signals != 0
        X = features.loc[mask]
        y = labels_result.labels.loc[mask]
        weights = labels_result.sample_weight.loc[mask]

        # Las ultimas barras no tienen etiqueta fiable (su horizonte todavia no
        # ha terminado). Incluirlas seria etiquetar con informacion incompleta.
        if len(X) > cfg.lookahead_bars:
            X = X.iloc[: -cfg.lookahead_bars]
            y = y.iloc[: -cfg.lookahead_bars]
            weights = weights.iloc[: -cfg.lookahead_bars]

        logger.info(
            "Entrenando meta-etiquetador: %d senales, %.1f%% positivas",
            len(X),
            100 * float(y.mean()) if len(y) else 0.0,
        )

        train_idx, test_idx = purged_train_test_split(
            X.index,
            train_ratio=cfg.train_test_split,
            purge_bars=cfg.purge_bars,
            embargo_bars=cfg.embargo_bars,
        )

        labeler = MetaLabeler(cfg)
        report = labeler.fit(X, y, weights, train_idx, test_idx)

        model_path: Path | None = None
        if save and report.is_useful:
            model_id = uuid.uuid4().hex[:12]
            model_path = self.artifacts_dir / f"metalabeler_{model_id}.joblib"
            labeler.save(model_path)

            if self.db is not None:
                self.db.save_model(
                    model_id=model_id,
                    path=str(model_path),
                    model_type=cfg.model_type,
                    auc=report.auc,
                    accuracy=report.accuracy,
                    n_samples=report.n_train,
                    features=labeler.feature_names,
                    strategy_id=strategy_id,
                    active=True,
                )

        if report.feature_importance:
            top = ", ".join(f"{k}={v:.3f}" for k, v in report.top_features(5))
            logger.info("Features mas influyentes: %s", top)

        return TrainingResult(
            labeler=labeler if report.is_useful else None,
            report=report,
            model_path=model_path,
            n_signals=len(X),
        )

    # ------------------------------------------------------------------ #
    def load_active(self, strategy_id: str | None = None) -> MetaLabeler | None:
        """Recupera el modelo activo persistido, si existe y sigue siendo valido."""
        if self.db is None:
            return None

        record = self.db.get_active_model(strategy_id)
        if not record:
            return None

        path = Path(record["path"])
        if not path.exists():
            logger.warning("El modelo activo %s ya no esta en disco", path)
            return None

        try:
            labeler = MetaLabeler.load(path, self.config.ml)
            logger.info("Modelo cargado: %s (AUC %.3f)", path.name, record.get("auc", 0.0))
            return labeler
        except Exception as exc:
            logger.warning("No se pudo cargar el modelo %s: %s", path, exc)
            return None

    def build_signal_filter(self, labeler: MetaLabeler | None):
        """Adaptador para inyectar el modelo en el evaluador de fitness.

        Devuelve una funcion con la firma que espera
        :class:`~goldbot.evolution.fitness.FitnessEvaluator`, que convierte la
        probabilidad del modelo en un multiplicador de tamano por barra.
        """
        if labeler is None or not labeler.is_ready:
            return None

        def _filter(signals: pd.Series, features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.Series:
            multiplier = labeler.size_multiplier(features)
            # Fuera de las barras con senal el multiplicador es irrelevante,
            # pero lo dejamos a 1.0 para no confundir a quien lo inspeccione.
            return multiplier.where(signals != 0, 1.0)

        return _filter


def evaluate_ml_contribution(
    config: Config,
    strategy: Strategy,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    labeler: MetaLabeler,
) -> dict[str, float]:
    """Compara la estrategia con y sin el filtro de ML.

    Si el filtro no mejora el Sharpe, sobra: cada capa adicional es una
    oportunidad mas de sobreajustar, y solo se justifica si paga su coste.
    """
    from goldbot.backtest.engine import BacktestEngine

    engine = BacktestEngine(config)
    signals = strategy.generate_signals(features, ohlcv)
    exit_rules = getattr(strategy, "exit_rules", None)

    baseline = engine.run(ohlcv, signals, exit_rules)
    multiplier = labeler.size_multiplier(features).where(signals != 0, 1.0)
    filtered = engine.run(ohlcv, signals, exit_rules, size_multiplier=multiplier)

    delta_sharpe = filtered.metrics.sharpe - baseline.metrics.sharpe
    logger.info(
        "Aportacion del ML: Sharpe %.2f -> %.2f (%+.2f), operaciones %d -> %d",
        baseline.metrics.sharpe,
        filtered.metrics.sharpe,
        delta_sharpe,
        baseline.metrics.total_trades,
        filtered.metrics.total_trades,
    )

    return {
        "baseline_sharpe": float(baseline.metrics.sharpe),
        "filtered_sharpe": float(filtered.metrics.sharpe),
        "delta_sharpe": float(delta_sharpe),
        "baseline_trades": int(baseline.metrics.total_trades),
        "filtered_trades": int(filtered.metrics.total_trades),
        "baseline_drawdown": float(baseline.metrics.max_drawdown),
        "filtered_drawdown": float(filtered.metrics.max_drawdown),
        "improves": bool(delta_sharpe > 0.05),
    }
