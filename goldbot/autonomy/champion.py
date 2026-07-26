"""Registro campeon/retador y ciclo de vida de las estrategias.

El bot mantiene, como mucho, **un campeon** operando y varios **retadores** en
incubacion. Una estrategia recorre siempre el mismo camino:

    candidata -> validada -> en incubacion -> campeona -> retirada

Ninguna estrategia salta pasos. En concreto, ninguna llega a campeona sin haber
pasado antes por la incubadora en papel durante el numero de dias configurado:
un backtest excelente y una ejecucion real decepcionante son perfectamente
compatibles, y la incubadora es lo unico que distingue ambos casos antes de que
cueste dinero.

Reemplazar al campeon exige una **mejora sustancial**, no una mejora cualquiera.
Sin ese margen, el bot cambiaria de estrategia constantemente persiguiendo ruido
estadistico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from goldbot.config import Config
from goldbot.storage.db import (
    STATUS_CHAMPION,
    STATUS_INCUBATING,
    STATUS_REJECTED,
    STATUS_RETIRED,
    STATUS_VALIDATED,
    Database,
    StrategyRecord,
)
from goldbot.strategies.genome import StrategyGenome
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import now_utc

logger = get_logger(__name__)


@dataclass
class IncubationStatus:
    """Progreso de una estrategia en la incubadora de papel."""

    strategy_id: str
    days_elapsed: int
    trades: int
    pnl: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    ready: bool = False
    blocked_reason: str = ""

    def summary(self) -> str:
        state = "LISTA" if self.ready else f"en curso ({self.blocked_reason})"
        return (
            f"{self.strategy_id}: {self.days_elapsed}d, {self.trades} ops, "
            f"P&L {self.pnl:+.2f}, Sharpe {self.sharpe:.2f}, DD {self.max_drawdown:.1%} [{state}]"
        )


@dataclass
class PromotionDecision:
    """Decision razonada sobre un cambio de campeon."""

    action: str  # promover | mantener | retirar | sin_campeon
    strategy_id: str | None = None
    previous_champion: str | None = None
    reason: str = ""
    improvement: float = 0.0
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return f"[{self.action.upper()}] {self.strategy_id or '-'}: {self.reason}"


class ChampionRegistry:
    """Gestiona quien opera, quien incuba y quien se retira."""

    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db

    # ------------------------------------------------------------------ #
    # Consultas
    # ------------------------------------------------------------------ #
    def get_champion(self) -> StrategyRecord | None:
        return self.db.get_champion()

    def get_champion_genome(self) -> StrategyGenome | None:
        record = self.get_champion()
        return record.to_genome() if record else None

    def get_incubating(self) -> list[StrategyRecord]:
        return self.db.list_strategies(status=STATUS_INCUBATING, limit=50)

    def get_validated(self) -> list[StrategyRecord]:
        return self.db.list_strategies(status=STATUS_VALIDATED, limit=50)

    # ------------------------------------------------------------------ #
    # Registro de candidatos
    # ------------------------------------------------------------------ #
    def register_candidate(
        self, genome: StrategyGenome, verdict, metrics: dict | None = None
    ) -> str | None:
        """Registra un candidato tras pasar por la puerta de estabilidad.

        Devuelve el id si se registro, o ``None`` si era un duplicado
        estructural de una estrategia ya conocida.
        """
        fingerprint = genome.fingerprint()
        existing = self.db.find_by_fingerprint(fingerprint)

        if existing is not None and existing.status not in {STATUS_REJECTED, STATUS_RETIRED}:
            logger.debug(
                "Candidato %s descartado: duplicado de %s (%s)",
                genome.genome_id, existing.id, existing.status,
            )
            return None

        payload = dict(metrics or {})
        payload["stability"] = verdict.to_dict()

        status = STATUS_VALIDATED if verdict.passed else STATUS_REJECTED
        notes = "apta para incubacion" if verdict.passed else verdict.rejection_reason[:500]

        self.db.save_strategy(genome, status=status, metrics=payload, notes=notes)
        self.db.save_evaluation(
            strategy_id=genome.genome_id,
            kind="stability",
            metrics=verdict.to_dict(),
            score=verdict.score,
        )

        if verdict.passed:
            logger.info(
                "Nueva estrategia validada: %s (puntuacion %.3f)", genome.genome_id, verdict.score
            )
        return genome.genome_id

    # ------------------------------------------------------------------ #
    # Incubacion
    # ------------------------------------------------------------------ #
    def start_incubation(self, strategy_id: str) -> bool:
        """Pasa una estrategia validada a incubacion en papel."""
        record = self.db.get_strategy(strategy_id)
        if record is None:
            logger.warning("No existe la estrategia %s", strategy_id)
            return False

        if record.status != STATUS_VALIDATED:
            logger.warning(
                "%s no se puede incubar desde el estado '%s'", strategy_id, record.status
            )
            return False

        slots = self.config.autonomy.challenger_slots
        current = self.get_incubating()
        if len(current) >= slots:
            # Se libera el hueco de la peor incubada, no de la mas antigua:
            # el objetivo es maximizar la calidad de la cola, no rotarla.
            worst = min(current, key=lambda r: float(r.metrics.get("stability", {}).get("score", 0.0)))
            new_score = float(record.metrics.get("stability", {}).get("score", 0.0))
            worst_score = float(worst.metrics.get("stability", {}).get("score", 0.0))

            if new_score <= worst_score:
                logger.info(
                    "Incubadora llena (%d/%d) y %s no supera a la peor (%s)",
                    len(current), slots, strategy_id, worst.id,
                )
                return False

            self.db.update_status(worst.id, STATUS_REJECTED, "desplazada por un candidato mejor")

        self.db.update_status(strategy_id, STATUS_INCUBATING, "inicio de incubacion en papel")
        self.db.set_state(f"incubation_start:{strategy_id}", now_utc().isoformat())
        logger.info("Estrategia %s en incubacion (papel)", strategy_id)
        return True

    def check_incubation(self, strategy_id: str) -> IncubationStatus:
        """Evalua si una estrategia incubada ya reune evidencia suficiente."""
        cfg = self.config.stability

        started_raw = self.db.get_state(f"incubation_start:{strategy_id}")
        started = _parse_datetime(started_raw) or now_utc()
        days = max(0, (now_utc() - started).days)

        trades = self.db.get_trades(strategy_id=strategy_id, mode="paper")
        closed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]

        pnls = np.array([float(t["pnl"]) for t in closed], dtype="float64")
        n_trades = len(pnls)

        status = IncubationStatus(
            strategy_id=strategy_id,
            days_elapsed=days,
            trades=n_trades,
            pnl=float(pnls.sum()) if n_trades else 0.0,
            sharpe=0.0,
            win_rate=float((pnls > 0).mean()) if n_trades else 0.0,
            max_drawdown=0.0,
        )

        if n_trades >= 2:
            std = float(pnls.std(ddof=1))
            if std > 0:
                # Sharpe por operacion anualizado con el ritmo observado.
                trades_per_year = n_trades / max(days, 1) * 252
                status.sharpe = float(pnls.mean() / std * np.sqrt(max(trades_per_year, 1)))

            equity = self.config.risk.initial_balance + np.cumsum(pnls)
            running_max = np.maximum.accumulate(equity)
            status.max_drawdown = float(np.max((running_max - equity) / running_max))

        # --- criterios para salir de la incubadora --- #
        if days < cfg.incubation_days:
            status.blocked_reason = f"faltan {cfg.incubation_days - days} dias"
        elif n_trades < cfg.incubation_min_trades:
            status.blocked_reason = f"faltan {cfg.incubation_min_trades - n_trades} operaciones"
        elif status.pnl <= 0:
            status.blocked_reason = f"P&L acumulado negativo ({status.pnl:+.2f})"
        elif status.max_drawdown > cfg.max_drawdown_pct:
            status.blocked_reason = f"drawdown en papel {status.max_drawdown:.1%} excesivo"
        else:
            status.ready = True

        return status

    # ------------------------------------------------------------------ #
    # Promocion y degradacion
    # ------------------------------------------------------------------ #
    def evaluate_promotion(self) -> PromotionDecision:
        """Decide si algun retador debe sustituir al campeon."""
        cfg = self.config.autonomy
        champion = self.get_champion()

        ready: list[tuple[StrategyRecord, IncubationStatus]] = []
        for record in self.get_incubating():
            status = self.check_incubation(record.id)
            if status.ready:
                ready.append((record, status))

        if not ready:
            if champion is None:
                return PromotionDecision("sin_campeon", reason="ningun retador listo todavia")
            return PromotionDecision(
                "mantener", champion.id, reason="ningun retador ha completado la incubacion"
            )

        # El mejor retador por puntuacion de estabilidad.
        ready.sort(key=lambda pair: _stability_score(pair[0]), reverse=True)
        best_record, best_status = ready[0]
        best_score = _stability_score(best_record)

        if champion is None:
            return PromotionDecision(
                "promover",
                strategy_id=best_record.id,
                reason=f"primer campeon (puntuacion {best_score:.3f}, {best_status.trades} ops en papel)",
                improvement=best_score,
                details={"incubation": best_status.summary()},
            )

        champion_score = self._live_score(champion)
        if champion_score <= 0:
            return PromotionDecision(
                "promover",
                strategy_id=best_record.id,
                previous_champion=champion.id,
                reason=f"el campeon actual rinde por debajo de cero ({champion_score:.3f})",
                improvement=best_score - champion_score,
            )

        improvement = (best_score - champion_score) / abs(champion_score)

        if improvement >= cfg.promote_min_improvement:
            return PromotionDecision(
                "promover",
                strategy_id=best_record.id,
                previous_champion=champion.id,
                reason=(
                    f"mejora del {improvement:.1%} sobre el campeon "
                    f"(minimo exigido {cfg.promote_min_improvement:.0%})"
                ),
                improvement=improvement,
            )

        return PromotionDecision(
            "mantener",
            strategy_id=champion.id,
            reason=(
                f"el mejor retador solo mejora un {improvement:.1%}, "
                f"por debajo del {cfg.promote_min_improvement:.0%} exigido"
            ),
            improvement=improvement,
        )

    def promote(self, strategy_id: str, previous_champion: str | None = None, reason: str = "") -> bool:
        """Asciende una estrategia a campeona y retira a la anterior."""
        if not self.config.autonomy.auto_promote:
            logger.info("Promocion automatica desactivada; %s queda pendiente", strategy_id)
            return False

        record = self.db.get_strategy(strategy_id)
        if record is None:
            return False

        if record.status != STATUS_INCUBATING:
            logger.warning(
                "%s no puede promocionarse desde el estado '%s' (debe incubar primero)",
                strategy_id, record.status,
            )
            return False

        current = self.get_champion()
        if current is not None:
            self.db.update_status(
                current.id, STATUS_RETIRED, f"sustituida por {strategy_id}: {reason}"
            )

        self.db.update_status(strategy_id, STATUS_CHAMPION, reason)
        self.db.set_state("champion_since", now_utc().isoformat())
        self.db.set_state("champion_id", strategy_id)

        logger.info(
            "NUEVO CAMPEON: %s%s | %s",
            strategy_id,
            f" (sustituye a {current.id})" if current else "",
            reason,
        )
        return True

    def evaluate_demotion(self, recent_sharpe: float, consecutive_bad_days: int) -> PromotionDecision:
        """Decide si el campeon debe retirarse por degradacion.

        Se exige degradacion **sostenida**: unos dias malos son parte del juego
        y retirar una buena estrategia por una racha normal es tan destructivo
        como mantener una mala.
        """
        champion = self.get_champion()
        if champion is None:
            return PromotionDecision("sin_campeon", reason="no hay campeon activo")

        if not self.config.autonomy.auto_demote:
            return PromotionDecision("mantener", champion.id, reason="degradacion automatica desactivada")

        cfg = self.config.stability
        expected = _expected_sharpe(champion)
        drop = expected - recent_sharpe

        if consecutive_bad_days >= cfg.consecutive_bad_days and drop >= cfg.degradation_sharpe_drop:
            return PromotionDecision(
                "retirar",
                strategy_id=champion.id,
                reason=(
                    f"degradacion sostenida: Sharpe {recent_sharpe:.2f} frente a {expected:.2f} "
                    f"esperado durante {consecutive_bad_days} dias"
                ),
                improvement=-drop,
            )

        return PromotionDecision(
            "mantener",
            champion.id,
            reason=f"rendimiento dentro de lo esperado (Sharpe {recent_sharpe:.2f})",
        )

    def demote(self, strategy_id: str, reason: str) -> bool:
        """Retira al campeon. El bot queda sin operar hasta tener sustituto."""
        self.db.update_status(strategy_id, STATUS_RETIRED, reason)
        self.db.set_state("champion_id", None)
        logger.warning("Campeon %s RETIRADO: %s", strategy_id, reason)
        return True

    # ------------------------------------------------------------------ #
    def _live_score(self, champion: StrategyRecord) -> float:
        """Puntuacion del campeon combinando su validacion y su operativa real.

        Mientras haya pocas operaciones reales manda la validacion historica;
        segun se acumula evidencia en vivo, esta va pesando mas. Asi se evita
        tanto juzgar por 3 operaciones como ignorar 300.
        """
        historical = _stability_score(champion)

        trades = self.db.get_trades(strategy_id=champion.id, mode="live")
        closed = [t for t in trades if t.get("pnl") is not None]

        if len(closed) < 20:
            return historical

        pnls = np.array([float(t["pnl"]) for t in closed], dtype="float64")
        std = float(pnls.std(ddof=1))
        if std <= 0:
            return historical

        live_sharpe = float(pnls.mean() / std * np.sqrt(252))
        live_score = float(np.clip(live_sharpe / 4.0, 0.0, 1.0))

        weight = float(np.clip(len(closed) / 200, 0.0, 0.7))
        return (1 - weight) * historical + weight * live_score

    def summary(self) -> dict:
        """Estado del registro, para el panel de control y los logs."""
        champion = self.get_champion()
        incubating = self.get_incubating()
        return {
            "champion": champion.id if champion else None,
            "champion_since": self.db.get_state("champion_since"),
            "champion_score": _stability_score(champion) if champion else 0.0,
            "incubating": [r.id for r in incubating],
            "validated_waiting": len(self.get_validated()),
            "total_discovered": len(self.db.list_strategies(limit=10_000)),
        }


# --------------------------------------------------------------------------- #
def _stability_score(record: StrategyRecord | None) -> float:
    if record is None:
        return 0.0
    return float(record.metrics.get("stability", {}).get("score", 0.0))


def _expected_sharpe(record: StrategyRecord) -> float:
    """Sharpe que la validacion prometia; es la referencia contra la que medir."""
    stability = record.metrics.get("stability", {})
    metrics = stability.get("metrics", {})
    value = metrics.get("sharpe", 0.0)
    try:
        return float(value) if np.isfinite(float(value)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(raw) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
