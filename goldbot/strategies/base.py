"""Contrato comun de una estrategia.

Cualquier cosa que produzca senales -- un genoma evolucionado, un modelo de ML
o una regla escrita a mano -- implementa esta interfaz. El resto del sistema
(backtest, incubadora, ejecucion en vivo) solo conoce este contrato.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategySignal:
    """Senal para una unica barra, tal como la consume el ejecutor en vivo."""

    direction: int              # +1 largo, -1 corto, 0 plano
    confidence: float = 1.0     # [0, 1]; escala el tamano de la posicion
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        return self.direction == 0


class Strategy(ABC):
    """Interfaz de estrategia."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame, ohlcv: pd.DataFrame | None = None) -> pd.Series:
        """Serie de senales (+1/-1/0) alineada con ``features``.

        Cada valor debe calcularse **solo** con informacion disponible hasta el
        cierre de esa barra. El motor de backtest se encarga de retrasar la
        ejecucion a la apertura siguiente.
        """

    def confidence(self, features: pd.DataFrame, ohlcv: pd.DataFrame | None = None) -> pd.Series:
        """Confianza por barra en [0, 1]. Por defecto, maxima."""
        return pd.Series(1.0, index=features.index)

    def latest_signal(self, features: pd.DataFrame, ohlcv: pd.DataFrame | None = None) -> StrategySignal:
        """Senal de la ultima barra cerrada, para operar en vivo."""
        signals = self.generate_signals(features, ohlcv)
        if signals.empty:
            return StrategySignal(direction=0, reason="sin datos")

        confidences = self.confidence(features, ohlcv)
        direction = int(signals.iloc[-1])
        conf = float(confidences.iloc[-1]) if not confidences.empty else 1.0
        return StrategySignal(
            direction=direction,
            confidence=conf,
            reason=f"{self.name} @ {signals.index[-1]}",
            metadata={"bar": str(signals.index[-1])},
        )

    def required_features(self) -> list[str]:
        """Features que la estrategia necesita; permite validar antes de operar."""
        return []
