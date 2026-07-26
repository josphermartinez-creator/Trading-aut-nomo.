"""Meta-etiquetado: el modelo que decide *cuando hacer caso* a la estrategia.

Un error muy extendido es pedirle a un modelo que prediga la direccion del
precio. Con datos de 5 minutos y una relacion señal/ruido pesima, eso produce
AUC de 0.51 y modelos inutiles.

El enfoque de meta-etiquetado (Lopez de Prado) es distinto y mucho mas
tratable: la *direccion* la decide el genoma; el modelo solo responde a una
pregunta binaria y mucho mas facil -- "dada esta señal concreta y el contexto
actual, ¿acabara ganando?". El modelo no toma posiciones, modula el tamaño: si
la probabilidad es baja, se opera pequeño o no se opera.

Esto mejora el ratio de Sharpe sin tocar el numero de aciertos direccionales,
porque reduce la exposicion justo en las señales de peor calidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goldbot.config import MLConfig
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelReport:
    """Resultado del entrenamiento."""

    model_type: str = ""
    auc: float = 0.5
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    n_train: int = 0
    n_test: int = 0
    positive_rate: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    is_useful: bool = False
    reason: str = ""

    def summary(self) -> str:
        verdict = "UTIL" if self.is_useful else f"DESCARTADO ({self.reason})"
        return (
            f"{self.model_type}: AUC={self.auc:.3f} acc={self.accuracy:.3f} "
            f"prec={self.precision:.3f} n={self.n_train}/{self.n_test} [{verdict}]"
        )

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:n]


class MetaLabeler:
    """Clasificador binario que puntua la calidad de cada senal."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.model: Any = None
        self.scaler: Any = None
        self.feature_names: list[str] = []
        self.report = ModelReport()
        self._trained = False

    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        """Solo se usa el modelo si entreno bien Y aporta algo sobre el azar."""
        return self._trained and self.model is not None and self.report.is_useful

    # ------------------------------------------------------------------ #
    def _build_model(self):
        """Instancia el estimador segun la configuracion.

        Los hiperparametros estan deliberadamente conservadores (arboles poco
        profundos, regularizacion alta): con etiquetas ruidosas y muestras
        solapadas, un modelo flexible memoriza el ruido sin remedio.
        """
        kind = self.config.model_type

        if kind == "gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=200,
                learning_rate=0.05,
                min_samples_leaf=50,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
            )

        if kind == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=50,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )

        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=0.1, max_iter=1000, class_weight="balanced", random_state=42
            )

        if kind == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            )

        if kind == "torch_mlp":
            return _build_torch_model()

        raise ValueError(f"Tipo de modelo desconocido: {kind}")

    # ------------------------------------------------------------------ #
    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series | None = None,
        train_idx: np.ndarray | None = None,
        test_idx: np.ndarray | None = None,
    ) -> ModelReport:
        """Entrena el meta-etiquetador.

        ``train_idx``/``test_idx`` deben venir de un split **purgado**
        (:func:`goldbot.features.labeling.purged_train_test_split`). Un split
        aleatorio corriente filtra informacion del futuro a traves del solape
        de las etiquetas y da un AUC ficticio.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        report = ModelReport(model_type=self.config.model_type)

        # Solo entrenan las barras con senal (etiqueta definida y no nula).
        mask = labels.notna() & features.notna().all(axis=1)
        X_all = features.loc[mask]
        y_all = labels.loc[mask].astype(int)
        weights_all = sample_weight.loc[mask] if sample_weight is not None else None

        if len(X_all) < 200:
            report.reason = f"muestras insuficientes ({len(X_all)})"
            self.report = report
            return report

        if y_all.nunique() < 2:
            report.reason = "una sola clase en las etiquetas"
            self.report = report
            return report

        # Seleccion de features: demasiadas columnas con pocas muestras
        # efectivas es la receta del sobreajuste.
        selected = self._select_features(X_all, y_all)
        X_all = X_all[selected]
        self.feature_names = selected

        if train_idx is None or test_idx is None:
            from goldbot.features.labeling import purged_train_test_split

            train_idx, test_idx = purged_train_test_split(
                X_all.index,
                train_ratio=self.config.train_test_split,
                purge_bars=self.config.purge_bars,
                embargo_bars=self.config.embargo_bars,
            )

        train_idx = train_idx[train_idx < len(X_all)]
        test_idx = test_idx[test_idx < len(X_all)]

        if len(train_idx) < 100 or len(test_idx) < 50:
            report.reason = f"split insuficiente (train={len(train_idx)}, test={len(test_idx)})"
            self.report = report
            return report

        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_test, y_test = X_all.iloc[test_idx], y_all.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            report.reason = "una sola clase tras el split temporal"
            self.report = report
            return report

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = self._build_model()

        fit_kwargs: dict[str, Any] = {}
        if weights_all is not None:
            w = weights_all.iloc[train_idx].to_numpy(dtype="float64")
            if np.all(np.isfinite(w)) and w.sum() > 0:
                try:
                    self.model.fit(X_train_scaled, y_train, sample_weight=w)
                    fit_kwargs["used_weights"] = True
                except TypeError:
                    self.model.fit(X_train_scaled, y_train)
            else:
                self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)

        probabilities = self._predict_proba(X_test_scaled)
        predictions = (probabilities >= 0.5).astype(int)

        report.n_train = len(X_train)
        report.n_test = len(X_test)
        report.positive_rate = float(y_train.mean())
        report.auc = float(roc_auc_score(y_test, probabilities))
        report.accuracy = float(accuracy_score(y_test, predictions))
        report.precision = float(precision_score(y_test, predictions, zero_division=0))
        report.recall = float(recall_score(y_test, predictions, zero_division=0))
        report.feature_importance = self._feature_importance(X_train_scaled, y_train)

        # Puerta de calidad: un AUC de 0.52 fuera de muestra no es una senal,
        # es ruido con suerte. Mejor no usar el modelo que usar uno malo.
        if report.auc < self.config.min_auc:
            report.is_useful = False
            report.reason = f"AUC {report.auc:.3f} < minimo {self.config.min_auc}"
        else:
            report.is_useful = True
            report.reason = "ok"

        self._trained = True
        self.report = report
        logger.info("Meta-etiquetador entrenado: %s", report.summary())
        return report

    # ------------------------------------------------------------------ #
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Selecciona las features mas informativas por informacion mutua.

        Se usa informacion mutua y no correlacion porque captura relaciones no
        lineales, que es justo donde un arbol puede aportar algo.
        """
        max_features = self.config.max_features
        if X.shape[1] <= max_features:
            return list(X.columns)

        try:
            from sklearn.feature_selection import mutual_info_classif

            # Submuestreo: la informacion mutua es cara y no necesita todo.
            sample_size = min(len(X), 20_000)
            step = max(1, len(X) // sample_size)
            X_sample, y_sample = X.iloc[::step], y.iloc[::step]

            scores = mutual_info_classif(X_sample, y_sample, random_state=42, n_neighbors=5)
            ranked = pd.Series(scores, index=X.columns).sort_values(ascending=False)
            selected = ranked.head(max_features).index.tolist()
            logger.debug("Seleccionadas %d de %d features", len(selected), X.shape[1])
            return selected
        except Exception as exc:
            logger.warning("Seleccion de features fallo (%s); se usan todas", exc)
            return list(X.columns)[:max_features]

    def _predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Probabilidad de la clase positiva, sea cual sea el backend."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)[:, 1]
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_scaled)
            return 1.0 / (1.0 + np.exp(-scores))
        return self.model.predict(X_scaled).astype("float64")

    def _feature_importance(self, X_scaled: np.ndarray, y: pd.Series) -> dict[str, float]:
        """Importancia de cada feature; por permutacion si no es nativa."""
        try:
            if hasattr(self.model, "feature_importances_"):
                values = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                values = np.abs(self.model.coef_.ravel())
            else:
                from sklearn.inspection import permutation_importance

                result = permutation_importance(
                    self.model, X_scaled, y, n_repeats=3, random_state=42, n_jobs=-1
                )
                values = result.importances_mean

            total = float(np.sum(np.abs(values))) or 1.0
            return {
                name: float(abs(v) / total)
                for name, v in zip(self.feature_names, values, strict=False)
            }
        except Exception as exc:
            logger.debug("No se pudo calcular la importancia de features: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Probabilidad de exito por barra. Devuelve 1.0 si el modelo no sirve.

        Devolver 1.0 (y no 0.5) cuando el modelo no esta listo es intencionado:
        significa "no tengo criterio, no estorbo", de modo que la estrategia
        opera con su tamano normal en lugar de quedarse paralizada.
        """
        if not self.is_ready:
            return pd.Series(1.0, index=features.index)

        missing = [f for f in self.feature_names if f not in features.columns]
        if missing:
            logger.warning("Faltan %d features para el modelo; se omite el filtro", len(missing))
            return pd.Series(1.0, index=features.index)

        X = features[self.feature_names]
        valid = X.notna().all(axis=1)

        out = pd.Series(1.0, index=features.index)
        if not valid.any():
            return out

        X_scaled = self.scaler.transform(X.loc[valid])
        out.loc[valid] = self._predict_proba(X_scaled)
        return out

    def size_multiplier(self, features: pd.DataFrame) -> pd.Series:
        """Traduce la probabilidad a un multiplicador de tamano en [0, 1].

        Por debajo del umbral no se opera; por encima, el tamano crece de forma
        lineal hasta el maximo. Es una rampa y no un interruptor para que la
        exposicion no salte bruscamente ante cambios minusculos de probabilidad.
        """
        if not self.is_ready:
            return pd.Series(1.0, index=features.index)

        proba = self.predict_proba(features)
        threshold = self.config.min_probability

        scaled = (proba - threshold) / max(1.0 - threshold, 1e-6)
        return scaled.clip(lower=0.0, upper=1.0).where(proba >= threshold, 0.0)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "report": self.report,
                "config": self.config,
            },
            path,
        )
        logger.debug("Modelo guardado en %s", path)

    @classmethod
    def load(cls, path: str | Path, config: MLConfig | None = None) -> MetaLabeler:
        import joblib

        payload = joblib.load(Path(path))
        instance = cls(config or payload["config"])
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.report = payload["report"]
        instance._trained = True
        return instance


# --------------------------------------------------------------------------- #
def _build_torch_model():
    """MLP en PyTorch envuelto en la interfaz de scikit-learn.

    Es opcional: si PyTorch no esta instalado se cae a un MLP de sklearn. Para
    tabulares de este tamano, el boosting de gradiente suele ganar de todas
    formas; esta rama existe por si se quiere experimentar con arquitecturas
    secuenciales mas adelante.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.warning("PyTorch no instalado; se usa MLPClassifier de sklearn")
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=400,
            early_stopping=True, random_state=42,
        )

    from sklearn.base import BaseEstimator, ClassifierMixin

    class TorchMLP(BaseEstimator, ClassifierMixin):
        """MLP pequeno con dropout, compatible con la API de sklearn."""

        def __init__(self, hidden: int = 32, epochs: int = 120, lr: float = 1e-3, dropout: float = 0.3):
            self.hidden = hidden
            self.epochs = epochs
            self.lr = lr
            self.dropout = dropout
            self.net_: Any = None
            self.classes_ = np.array([0, 1])

        def fit(self, X, y, sample_weight=None):
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
            y_t = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)

            self.net_ = nn.Sequential(
                nn.Linear(X_t.shape[1], self.hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, self.hidden // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden // 2, 1),
            )

            optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
            if sample_weight is not None:
                w_t = torch.tensor(np.asarray(sample_weight), dtype=torch.float32).view(-1, 1)
                criterion = nn.BCEWithLogitsLoss(reduction="none")
            else:
                w_t = None
                criterion = nn.BCEWithLogitsLoss()

            self.net_.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                logits = self.net_(X_t)
                loss = criterion(logits, y_t)
                if w_t is not None:
                    loss = (loss * w_t).mean()
                loss.backward()
                optimizer.step()
            return self

        def predict_proba(self, X):
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(torch.tensor(np.asarray(X), dtype=torch.float32))
                p = torch.sigmoid(logits).numpy().ravel()
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    return TorchMLP()
