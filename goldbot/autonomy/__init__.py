"""Autonomia: puerta de estabilidad, registro campeon/retador y ciclo diario."""

from goldbot.autonomy.champion import ChampionRegistry
from goldbot.autonomy.orchestrator import DailyReport, Orchestrator
from goldbot.autonomy.stability import StabilityGate, StabilityVerdict

__all__ = [
    "ChampionRegistry",
    "DailyReport",
    "Orchestrator",
    "StabilityGate",
    "StabilityVerdict",
]
