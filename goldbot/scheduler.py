"""Planificador: trading en vivo y aprendizaje diario en un solo proceso.

Ejecuta el bucle de trading en el hilo principal y lanza el ciclo de
aprendizaje en un hilo aparte a la hora configurada (por defecto, 22:00 UTC,
tras el cierre de Nueva York).

Se usa ``threading`` y no ``cron`` porque asi el aprendizaje puede avisar al
runner de que el campeon ha cambiado, sin necesidad de reiniciar el proceso ni
de coordinar dos servicios. En un VPS basta con mantener vivo un unico proceso.

Para quien prefiera separar responsabilidades, ``deploy/`` incluye tambien las
unidades systemd para correr ``run`` y ``learn`` por separado.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta

from goldbot.config import Config
from goldbot.storage.db import Database
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import now_utc

logger = get_logger(__name__)


class Scheduler:
    """Coordina el runner en vivo y el ciclo diario de aprendizaje."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.path(config.db_path))
        self.runner = None
        self._stop_event = threading.Event()
        self._learning_thread: threading.Thread | None = None
        self._last_learning_date: date | None = None

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Arranca todo. Bloquea hasta que se detiene el proceso."""
        from goldbot.live.runner import LiveRunner

        logger.info("=" * 70)
        logger.info("PLANIFICADOR EN MARCHA")
        logger.info(
            "Aprendizaje diario a las %02d:00 UTC | descubrimiento cada %d dias",
            self.config.autonomy.daily_learning_hour_utc,
            self.config.autonomy.discovery_every_days,
        )
        logger.info("=" * 70)

        # Si nunca se ha aprendido, se arranca en frio antes de operar: sin
        # campeon no hay nada que ejecutar.
        if self.db.last_successful_run() is None:
            logger.info("Sin ciclos previos: se lanza el arranque en frio")
            self._run_learning(bootstrap=True)

        self.runner = LiveRunner(self.config, self.db)

        self._learning_thread = threading.Thread(
            target=self._learning_loop, name="aprendizaje", daemon=True
        )
        self._learning_thread.start()

        try:
            self.runner.start()
        except KeyboardInterrupt:
            logger.info("Interrupcion recibida")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self.runner is not None:
            self.runner.stop()
        if self._learning_thread is not None and self._learning_thread.is_alive():
            logger.info("Esperando a que termine el ciclo de aprendizaje...")
            self._learning_thread.join(timeout=60)
        logger.info("Planificador detenido")

    # ------------------------------------------------------------------ #
    def _learning_loop(self) -> None:
        """Vigila la hora y dispara el aprendizaje una vez al dia."""
        target_hour = self.config.autonomy.daily_learning_hour_utc

        while not self._stop_event.is_set():
            now = now_utc()
            today = now.date()

            if now.hour == target_hour and self._last_learning_date != today:
                self._run_learning()
                self._last_learning_date = today

            # Comprobacion cada minuto: coste despreciable y precision de sobra.
            self._stop_event.wait(60)

    def _run_learning(self, bootstrap: bool = False) -> None:
        """Ejecuta el ciclo de aprendizaje y avisa al runner si cambio el campeon."""
        from goldbot.autonomy.orchestrator import Orchestrator

        logger.info("--- Lanzando ciclo de aprendizaje ---")
        try:
            orchestrator = Orchestrator(self.config, self.db)
            report = orchestrator.bootstrap() if bootstrap else orchestrator.run_daily_cycle()

            # El runner debe recoger el campeon nuevo sin reiniciar el proceso.
            if self.runner is not None:
                self.runner.reload_champion()

            logger.info(
                "Ciclo de aprendizaje terminado: %s",
                "correcto" if report.success else f"con {len(report.failed_stages)} errores",
            )
        except Exception as exc:
            logger.error("El ciclo de aprendizaje fallo: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    def next_learning_time(self) -> datetime:
        """Proxima ejecucion prevista del aprendizaje."""
        now = now_utc()
        target = now.replace(
            hour=self.config.autonomy.daily_learning_hour_utc, minute=0, second=0, microsecond=0
        )
        if target <= now:
            target += timedelta(days=1)
        return target
