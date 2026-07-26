"""Deteccion de deriva (concept drift).

Una estrategia deja de funcionar por dos motivos distintos y conviene
distinguirlos:

* **Deriva de covariables**: el mercado cambia de regimen (la volatilidad se
  duplica, la sesion asiatica se despierta). Las features entran en rangos que
  el modelo nunca vio. Suele ser recuperable reentrenando.
* **Degradacion de rendimiento**: las senales siguen apareciendo en contextos
  familiares, pero ya no ganan. Eso significa que la ventaja se ha agotado, y
  reentrenar no la devuelve.

El bot reacciona distinto a cada caso: ante la primera reentrena; ante la
segunda, retira la estrategia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from goldbot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftReport:
    """Diagnostico de deriva."""

    covariate_drift: bool = False
    performance_drift: bool = False
    drifted_features: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)
    psi_total: float = 0.0
    performance_delta: float = 0.0
    recommendation: str = "sin_accion"
    details: str = ""

    @property
    def needs_action(self) -> bool:
        return self.covariate_drift or self.performance_drift

    def summary(self) -> str:
        return (
            f"Deriva: covariables={'SI' if self.covariate_drift else 'no'} "
            f"(PSI={self.psi_total:.3f}, {len(self.drifted_features)} features) | "
            f"rendimiento={'SI' if self.performance_drift else 'no'} "
            f"({self.performance_delta:+.2f}) -> {self.recommendation}"
        )


class DriftDetector:
    """Compara la ventana reciente con la de referencia."""

    def __init__(
        self,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.01,
        max_drifted_ratio: float = 0.30,
        performance_drop_threshold: float = 0.5,
    ) -> None:
        # PSI > 0.25 es el umbral clasico de "cambio importante" en riesgo de credito.
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.max_drifted_ratio = max_drifted_ratio
        self.performance_drop_threshold = performance_drop_threshold

    # ------------------------------------------------------------------ #
    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        reference_sharpe: float | None = None,
        current_sharpe: float | None = None,
        feature_subset: list[str] | None = None,
    ) -> DriftReport:
        """Compara distribuciones y rendimiento entre dos periodos."""
        report = DriftReport()

        if reference.empty or current.empty:
            report.details = "ventanas vacias"
            return report

        columns = feature_subset or [c for c in current.columns if c in reference.columns]
        if not columns:
            report.details = "sin features comunes"
            return report

        psi_values: list[float] = []
        for column in columns:
            ref_series = reference[column].dropna()
            cur_series = current[column].dropna()
            if len(ref_series) < 100 or len(cur_series) < 50:
                continue

            psi = _population_stability_index(ref_series.to_numpy(), cur_series.to_numpy())
            report.drift_scores[column] = float(psi)
            psi_values.append(psi)

            if psi > self.psi_threshold:
                report.drifted_features.append(column)
                continue

            # El PSI puede pasar por alto cambios de forma sin cambio de rango;
            # Kolmogorov-Smirnov los captura.
            try:
                _, p_value = stats.ks_2samp(ref_series, cur_series)
                if p_value < self.ks_alpha and psi > self.psi_threshold * 0.5:
                    report.drifted_features.append(column)
            except Exception:
                pass

        if psi_values:
            report.psi_total = float(np.mean(psi_values))

        drifted_ratio = len(report.drifted_features) / max(len(columns), 1)
        report.covariate_drift = drifted_ratio > self.max_drifted_ratio

        # --- degradacion de rendimiento --- #
        if reference_sharpe is not None and current_sharpe is not None:
            report.performance_delta = float(current_sharpe - reference_sharpe)
            report.performance_drift = report.performance_delta < -self.performance_drop_threshold

        report.recommendation = self._recommend(report, drifted_ratio)
        report.details = (
            f"{len(report.drifted_features)}/{len(columns)} features con deriva "
            f"({drifted_ratio:.0%})"
        )

        if report.needs_action:
            logger.warning("%s", report.summary())
        else:
            logger.debug("%s", report.summary())
        return report

    def _recommend(self, report: DriftReport, drifted_ratio: float) -> str:
        """Traduce el diagnostico a una accion concreta."""
        if report.performance_drift and report.covariate_drift:
            # El mercado cambio Y la estrategia dejo de funcionar: es lo mas
            # probable que sea recuperable con una busqueda nueva.
            return "redescubrir_estrategias"
        if report.performance_drift:
            # Contexto igual, resultados peores: la ventaja se agoto.
            return "retirar_estrategia"
        if report.covariate_drift:
            # Contexto nuevo pero sin dano medible todavia: reentrenar el filtro.
            return "reentrenar_modelo"
        return "sin_accion"


# --------------------------------------------------------------------------- #
def _population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre dos muestras.

    Se construyen los cortes con los cuantiles de la referencia; asi el PSI mide
    cuanto se ha desplazado la masa de probabilidad respecto al periodo base.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size < 10 or current.size < 10:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0  # feature practicamente constante: no hay deriva que medir

    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Suavizado para que un bin vacio no produzca un PSI infinito.
    epsilon = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + epsilon
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + epsilon

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def rolling_performance_drift(
    equity: pd.Series, window_days: int = 10, timeframe_minutes: int = 5
) -> pd.Series:
    """Sharpe movil de la equity: sirve para vigilar la degradacion en vivo."""
    bars_per_day = int(24 * 60 / timeframe_minutes)
    window = max(50, window_days * bars_per_day)

    returns = equity.pct_change().fillna(0.0)
    mean = returns.rolling(window, min_periods=window // 2).mean()
    std = returns.rolling(window, min_periods=window // 2).std(ddof=0).replace(0, np.nan)
    annualization = np.sqrt(bars_per_day * 252)
    return (mean / std * annualization).fillna(0.0)
