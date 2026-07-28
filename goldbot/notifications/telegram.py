"""Bot de Telegram: avisos en tiempo real y control remoto.

Un bot que opera solo en un VPS es una caja negra hasta que abres el terminal.
Telegram resuelve las dos necesidades reales: enterarse de lo que pasa sin
estar delante, y poder pararlo desde el movil cuando algo va mal.

Se habla directamente con la API HTTP de Telegram en lugar de usar
``python-telegram-bot``. La razon es de despliegue: esa libreria arrastra
asyncio y un arbol de dependencias considerable para lo que aqui son cuatro
llamadas HTTP. Con ``requests`` el modulo es autocontenido y no puede romper el
resto del sistema al actualizarse.

Seguridad, que aqui no es un detalle menor -- este bot puede cerrar posiciones:

* Solo se atienden mensajes del ``chat_id`` autorizado. Cualquier otro se
  registra y se ignora.
* El token nunca vive en el YAML: se lee de la variable de entorno.
* Los comandos destructivos (parar el bot, cerrar todo) exigen confirmacion.
"""

from __future__ import annotations

import html
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from goldbot.utils.logging import get_logger

logger = get_logger(__name__)

API_BASE = "https://api.telegram.org/bot{token}/{method}"

# Telegram corta los mensajes a 4096 caracteres.
MAX_MESSAGE_LENGTH = 4000


@dataclass
class TelegramNotifier:
    """Envio de mensajes. Nunca lanza excepciones hacia el bucle de trading.

    Un fallo de red al mandar un aviso no puede tumbar la operativa, asi que
    todos los errores se registran y se tragan. Si Telegram esta caido, el bot
    sigue operando a ciegas, que es preferible a que se detenga.
    """

    token: str = ""
    chat_id: str = ""
    enabled: bool = True
    parse_mode: str = "HTML"
    timeout: float = 10.0
    _failures: int = field(default=0, init=False, repr=False)

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.token and self.chat_id)

    # ------------------------------------------------------------------ #
    def send(self, text: str, silent: bool = False) -> bool:
        """Envia un mensaje. Devuelve ``True`` si Telegram lo acepto."""
        if not self.is_configured:
            return False

        # Tras varios fallos seguidos dejamos de intentarlo para no gastar un
        # segundo de timeout en cada vela.
        if self._failures >= 10:
            return False

        try:
            import requests
        except ImportError:
            logger.warning("requests no disponible: sin notificaciones de Telegram")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": _truncate(text),
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
            "disable_notification": silent,
        }

        try:
            response = requests.post(
                API_BASE.format(token=self.token, method="sendMessage"),
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                self._failures = 0
                return True

            self._failures += 1
            logger.warning("Telegram devolvio %d: %s", response.status_code, response.text[:200])
        except Exception as exc:
            self._failures += 1
            logger.warning("No se pudo enviar el mensaje de Telegram: %s", exc)
        return False

    # ------------------------------------------------------------------ #
    # Mensajes de dominio
    # ------------------------------------------------------------------ #
    def trade_opened(
        self, symbol: str, direction: int, lots: float, price: float,
        stop: float, target: float, strategy: str, confidence: float = 1.0,
    ) -> None:
        side = "COMPRA" if direction > 0 else "VENTA"
        emoji = "🟢" if direction > 0 else "🔴"
        risk = abs(price - stop)
        reward = abs(target - price)
        ratio = reward / risk if risk > 0 else 0.0

        self.send(
            f"{emoji} <b>{side} {esc(symbol)}</b>\n"
            f"Volumen: {lots:.2f} lotes\n"
            f"Entrada: {price:.2f}\n"
            f"Stop: {stop:.2f}  |  Objetivo: {target:.2f}\n"
            f"Ratio R:B: {ratio:.2f}\n"
            f"Confianza ML: {confidence:.0%}\n"
            f"<i>Estrategia {esc(strategy)}</i>"
        )

    def trade_closed(
        self, symbol: str, pnl: float, reason: str, balance: float, duration_minutes: float = 0.0
    ) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        self.send(
            f"{emoji} <b>Cerrada {esc(symbol)}</b>\n"
            f"Resultado: <b>{pnl:+.2f} USD</b>\n"
            f"Motivo: {esc(reason)}\n"
            f"Duracion: {duration_minutes:.0f} min\n"
            f"Saldo: {balance:.2f} USD"
        )

    def daily_report(self, report: dict) -> None:
        """Resumen del ciclo diario de aprendizaje."""
        lines = ["📊 <b>Informe diario</b>", ""]
        for key, value in report.items():
            lines.append(f"<b>{esc(str(key))}:</b> {esc(str(value))}")
        self.send("\n".join(lines), silent=True)

    def champion_changed(self, new_id: str, previous: str | None, reason: str) -> None:
        self.send(
            f"👑 <b>Nuevo campeon</b>\n"
            f"Estrategia: <code>{esc(new_id)}</code>\n"
            + (f"Sustituye a: <code>{esc(previous)}</code>\n" if previous else "")
            + f"Motivo: {esc(reason)}"
        )

    def circuit_breaker(self, reason: str, detail: str, resume_at: str | None = None) -> None:
        """Aviso critico: nunca silencioso."""
        text = (
            f"🚨 <b>CORTACIRCUITOS ACTIVADO</b>\n"
            f"Motivo: {esc(reason)}\n"
            f"Detalle: {esc(detail)}\n"
        )
        if resume_at:
            text += f"Reanudacion prevista: {esc(resume_at)}"
        self.send(text)

    def error(self, message: str) -> None:
        self.send(f"⚠️ <b>Error</b>\n<code>{esc(message[:500])}</code>")

    def startup(self, mode: str, dry_run: bool, champion: str | None, symbol: str) -> None:
        self.send(
            f"🤖 <b>GoldBot en marcha</b>\n"
            f"Modo: {esc(mode)}  |  Simulacion: {'SI' if dry_run else '<b>NO - DINERO REAL</b>'}\n"
            f"Simbolo: {esc(symbol)}\n"
            f"Campeon: <code>{esc(champion or 'ninguno')}</code>\n"
            f"<i>Escribe /ayuda para ver los comandos</i>"
        )


# --------------------------------------------------------------------------- #
# Control remoto
# --------------------------------------------------------------------------- #
class TelegramBot:
    """Escucha comandos por long polling y los ejecuta.

    Corre en su propio hilo para no bloquear el bucle de trading. Las acciones
    concretas se inyectan como funciones (``handlers``), de modo que este modulo
    no necesita conocer el runner ni el orquestador: solo traduce mensajes a
    llamadas.
    """

    def __init__(self, notifier: TelegramNotifier, poll_interval: float = 3.0) -> None:
        self.notifier = notifier
        self.poll_interval = poll_interval
        self.handlers: dict[str, Callable[[list[str]], str]] = {}
        self._offset = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending_confirmation: str | None = None

        self.register("ayuda", self._cmd_help)
        self.register("help", self._cmd_help)
        self.register("start", self._cmd_help)

    # ------------------------------------------------------------------ #
    def register(self, command: str, handler: Callable[[list[str]], str]) -> None:
        """Asocia un comando (sin la barra) a una funcion que devuelve texto."""
        self.handlers[command.lower()] = handler

    def start(self) -> bool:
        """Arranca el hilo de escucha."""
        if not self.notifier.is_configured:
            logger.info("Telegram sin configurar: control remoto desactivado")
            return False

        if not self._verify_token():
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="telegram", daemon=True)
        self._thread.start()
        logger.info("Control remoto de Telegram activo (%d comandos)", len(self.handlers))
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    def _verify_token(self) -> bool:
        """Comprueba que el token es valido antes de arrancar el hilo."""
        try:
            import requests

            response = requests.get(
                API_BASE.format(token=self.notifier.token, method="getMe"), timeout=10
            )
            if response.status_code != 200:
                logger.error("Token de Telegram invalido (HTTP %d)", response.status_code)
                return False
            username = response.json().get("result", {}).get("username", "?")
            logger.info("Bot de Telegram conectado: @%s", username)
            return True
        except Exception as exc:
            logger.error("No se pudo verificar el bot de Telegram: %s", exc)
            return False

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                logger.warning("Fallo en el sondeo de Telegram: %s", exc)
                # Espera larga tras un error para no martillear la API.
                self._stop.wait(30)
                continue
            self._stop.wait(self.poll_interval)

    def _poll_once(self) -> None:
        import requests

        response = requests.get(
            API_BASE.format(token=self.notifier.token, method="getUpdates"),
            params={"offset": self._offset, "timeout": 20},
            timeout=30,
        )
        if response.status_code != 200:
            return

        for update in response.json().get("result", []):
            self._offset = update["update_id"] + 1
            message = update.get("message") or update.get("edited_message")
            if not message:
                continue

            chat_id = str(message.get("chat", {}).get("id", ""))
            text = (message.get("text") or "").strip()
            if not text:
                continue

            # Unica puerta de entrada: solo el chat autorizado da ordenes.
            if chat_id != str(self.notifier.chat_id):
                sender = message.get("from", {}).get("username", "?")
                logger.warning(
                    "Mensaje ignorado de un chat no autorizado (%s, @%s): %r",
                    chat_id, sender, text[:60],
                )
                continue

            self._handle(text)

    def _handle(self, text: str) -> None:
        parts = text.split()
        command = parts[0].lstrip("/").lower()
        # Telegram anade @nombre_del_bot en los grupos.
        command = command.split("@")[0]
        args = parts[1:]

        # Confirmacion pendiente de una accion destructiva.
        if self._pending_confirmation:
            pending, self._pending_confirmation = self._pending_confirmation, None
            if command in {"si", "confirmar", "yes", "confirm"}:
                handler = self.handlers.get(pending)
                if handler:
                    self.notifier.send(handler(["--confirmado"]))
                return
            self.notifier.send("Accion cancelada.")
            return

        handler = self.handlers.get(command)
        if handler is None:
            self.notifier.send(
                f"Comando desconocido: <code>{esc(command)}</code>\n"
                f"Escribe /ayuda para ver la lista."
            )
            return

        # Los comandos que cierran posiciones o paran el bot piden confirmacion.
        if command in {"parar", "cerrartodo"} and "--confirmado" not in args:
            self._pending_confirmation = command
            self.notifier.send(
                f"⚠️ <b>{esc(command.upper())}</b> es una accion irreversible.\n"
                f"Responde <b>si</b> para confirmar, cualquier otra cosa la cancela."
            )
            return

        try:
            self.notifier.send(handler(args))
        except Exception as exc:
            logger.error("El comando /%s fallo: %s", command, exc, exc_info=True)
            self.notifier.send(f"El comando fallo: <code>{esc(str(exc)[:200])}</code>")

    # ------------------------------------------------------------------ #
    def _cmd_help(self, _args: list[str]) -> str:
        known = {
            "estado": "situacion actual del bot y de la cuenta",
            "posiciones": "posiciones abiertas ahora mismo",
            "hoy": "resultado del dia",
            "estrategias": "estrategias descubiertas y su estado",
            "campeon": "detalle de la estrategia que esta operando",
            "pausar": "deja de abrir posiciones nuevas",
            "reanudar": "vuelve a operar",
            "cerrartodo": "cierra todas las posiciones (pide confirmacion)",
            "parar": "detiene el bot por completo (pide confirmacion)",
            "ayuda": "esta lista",
        }
        lines = ["🤖 <b>Comandos disponibles</b>", ""]
        lines.extend(
            f"/{name} — {description}"
            for name, description in known.items()
            if name in self.handlers or name == "ayuda"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def build_command_handlers(runner, orchestrator=None) -> dict[str, Callable[[list[str]], str]]:
    """Conecta los comandos con el runner en vivo.

    Se define aqui, y no dentro de :class:`TelegramBot`, para que el modulo de
    notificaciones no dependa del resto del sistema y siga siendo probable de
    forma aislada.
    """

    def estado(_args: list[str]) -> str:
        status = runner.status()
        breaker = status["breaker"]
        return (
            f"📊 <b>Estado</b>\n"
            f"Operando: {'si' if status['running'] else 'no'}\n"
            f"Ciclos: {status['cycles']}\n"
            f"Campeon: <code>{esc(str(status['champion']))}</code>\n"
            f"Equity: {status['equity']:.2f} USD\n"
            f"Saldo: {status['balance']:.2f} USD\n"
            f"Ordenes enviadas: {status['orders_sent']}\n"
            f"Errores: {status['errors']}\n"
            f"Cortacircuitos: {esc(breaker['state'])}\n"
            f"Ultima vela: {esc(str(status['last_bar']))}"
        )

    def posiciones(_args: list[str]) -> str:
        try:
            positions = runner.broker.get_positions()
        except Exception as exc:
            return f"No se pudieron leer las posiciones: {esc(str(exc))}"

        if not positions:
            return "Sin posiciones abiertas."

        lines = ["<b>Posiciones abiertas</b>", ""]
        for p in positions:
            side = "COMPRA" if p.direction > 0 else "VENTA"
            lines.append(
                f"{side} {p.volume:.2f} {esc(p.symbol)} @ {p.entry_price:.2f}\n"
                f"  P&L flotante: {p.unrealized_pnl:+.2f} USD"
            )
        return "\n".join(lines)

    def hoy(_args: list[str]) -> str:
        stats = runner.risk.stats()
        return (
            f"<b>Resultado del dia</b>\n"
            f"Operaciones: {stats['trades_today']}\n"
            f"P&L realizado: {stats['realized_pnl_today']:+.2f} USD\n"
            f"Equity al abrir: {stats['day_start_equity']:.2f} USD\n"
            f"Maximo historico: {stats['peak_equity']:.2f} USD"
        )

    def pausar(_args: list[str]) -> str:
        runner.breaker.trip("pausa_manual", "solicitada desde Telegram", cooldown_minutes=10_000)
        return "⏸ Bot pausado. No se abriran posiciones nuevas.\nUsa /reanudar para volver."

    def reanudar(_args: list[str]) -> str:
        runner.breaker.reset("reanudado desde Telegram")
        return "▶️ Bot reanudado."

    def cerrartodo(_args: list[str]) -> str:
        try:
            orders = runner.broker.close_all(reason="cierre manual desde Telegram")
            return f"Cerradas {len(orders)} posiciones."
        except Exception as exc:
            return f"El cierre fallo: {esc(str(exc))}"

    def parar(_args: list[str]) -> str:
        runner.state.running = False
        return "🛑 Deteniendo el bot. Habra que reiniciarlo desde el servidor."

    def estrategias(_args: list[str]) -> str:
        records = runner.db.list_strategies(limit=10)
        if not records:
            return "Todavia no hay estrategias descubiertas."

        lines = ["<b>Estrategias</b>", ""]
        for r in records:
            score = r.metrics.get("stability", {}).get("score", 0.0)
            lines.append(f"<code>{esc(r.id)}</code> — {esc(r.status)} (punt. {score:.3f})")
        return "\n".join(lines)

    def campeon(_args: list[str]) -> str:
        record = runner.registry.get_champion()
        if record is None:
            return "No hay campeon activo. El bot solo esta incubando candidatas."

        stability = record.metrics.get("stability", {})
        metrics = stability.get("metrics", {})
        return (
            f"👑 <b>Campeon</b> <code>{esc(record.id)}</code>\n"
            f"Desde: {esc(str(record.promoted_at or '?')[:19])}\n"
            f"Puntuacion: {stability.get('score', 0):.3f}\n"
            f"Sharpe: {metrics.get('sharpe', 0):.2f}  |  "
            f"DD: {metrics.get('max_drawdown', 0):.1%}\n\n"
            f"<pre>{esc(record.description)}</pre>"
        )

    handlers = {
        "estado": estado,
        "posiciones": posiciones,
        "hoy": hoy,
        "pausar": pausar,
        "reanudar": reanudar,
        "cerrartodo": cerrartodo,
        "parar": parar,
        "estrategias": estrategias,
        "campeon": campeon,
    }

    if orchestrator is not None:
        def aprender(_args: list[str]) -> str:
            # El ciclo tarda minutos: se lanza en segundo plano y se avisa al
            # terminar, en vez de dejar a Telegram esperando una respuesta.
            def _run() -> None:
                report = orchestrator.run_daily_cycle()
                runner.reload_champion()
                handlers_notifier.daily_report({
                    "Duracion": f"{report.elapsed_seconds:.0f}s",
                    "Descubiertas": report.strategies_discovered,
                    "Validadas": report.strategies_validated,
                    "Campeon": report.champion_id or "ninguno",
                })

            handlers_notifier = runner.notifier
            threading.Thread(target=_run, name="aprendizaje-telegram", daemon=True).start()
            return "🧠 Ciclo de aprendizaje lanzado. Te aviso al terminar."

        handlers["aprender"] = aprender

    return handlers


# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    """Escapa el texto para el modo HTML de Telegram."""
    return html.escape(str(text), quote=False)


def _truncate(text: str) -> str:
    if len(text) <= MAX_MESSAGE_LENGTH:
        return text
    return text[: MAX_MESSAGE_LENGTH - 20] + "\n<i>[...truncado]</i>"
