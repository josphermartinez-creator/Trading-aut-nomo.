"""Bucle de trading en vivo.

Despierta justo despues de cada cierre de vela M5, recalcula las features con
los datos frescos, pregunta al campeon que hacer y ejecuta. Entre vela y vela
duerme: no tiene sentido consultar el precio 300 veces por minuto cuando las
decisiones se toman una vez cada cinco.

Todas las estrategias en incubacion se evaluan en paralelo contra el broker de
papel en el mismo ciclo, de modo que acumulan historial real sin arriesgar un
dolar.

Principio rector: **ante la duda, no operar**. Cualquier anomalia -- datos
obsoletos, precio incoherente, broker que no responde, cortacircuitos abierto
-- se resuelve quedandose fuera del mercado.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from goldbot.autonomy.champion import ChampionRegistry
from goldbot.config import Config
from goldbot.data.pipeline import MarketData
from goldbot.execution import build_broker
from goldbot.execution.base import Broker, Order, OrderSide, Position
from goldbot.execution.paper import PaperBroker
from goldbot.features import indicators as ta
from goldbot.features.engineering import FeatureBuilder, build_features
from goldbot.ml.trainer import MLTrainer
from goldbot.notifications.telegram import TelegramBot, TelegramNotifier, build_command_handlers
from goldbot.risk.circuit_breaker import CircuitBreaker
from goldbot.risk.manager import RiskManager
from goldbot.storage.db import Database
from goldbot.strategies.genome import StrategyGenome
from goldbot.utils.logging import get_logger
from goldbot.utils.timeutils import (
    market_is_open,
    now_utc,
    seconds_until_next_close,
)

logger = get_logger(__name__)


@dataclass
class RunnerState:
    """Estado del bucle en vivo."""

    running: bool = False
    cycles: int = 0
    last_bar_time: datetime | None = None
    last_price: float | None = None
    orders_sent: int = 0
    errors: int = 0
    started_at: datetime | None = None
    current_day: date | None = None
    incubators: dict[str, PaperBroker] = field(default_factory=dict)


class LiveRunner:
    """Bucle principal de operativa."""

    def __init__(self, config: Config, db: Database | None = None) -> None:
        self.config = config
        self.db = db or Database(config.path(config.db_path))
        self.market_data = MarketData(config)
        self.registry = ChampionRegistry(config, self.db)
        self.risk = RiskManager(config)
        self.breaker = CircuitBreaker(config)
        self.trainer = MLTrainer(config, self.db)
        self.feature_builder = FeatureBuilder.from_config(config)

        self.broker: Broker = build_broker(config, self.market_data)
        self.state = RunnerState()

        # Telegram: avisos y control remoto. Si no esta configurado, el
        # notificador se queda inerte y no estorba.
        self.notifier = TelegramNotifier(
            token=config.telegram_token or "",
            chat_id=str(config.telegram_chat_id or ""),
            enabled=config.telegram.enabled,
        )
        self.telegram: TelegramBot | None = None

        self._champion: StrategyGenome | None = None
        self._champion_id: str | None = None
        self._labeler = None
        self._open_trade_ids: dict[str, int] = {}

        self._install_signal_handlers()

    # ------------------------------------------------------------------ #
    def _install_signal_handlers(self) -> None:
        """Parada limpia ante SIGINT/SIGTERM.

        Importa mucho en un VPS: al reiniciar el servicio hay que cerrar la
        conexion con el broker de forma ordenada, no dejarla colgando.
        """
        def _handler(signum, _frame):
            logger.warning("Senal %s recibida: deteniendo el bucle...", signum)
            self.state.running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # En hilos secundarios no se pueden instalar manejadores.
                pass

    # ------------------------------------------------------------------ #
    def start(self, max_cycles: int | None = None) -> None:
        """Arranca el bucle. Bloquea hasta que se detiene."""
        if not self.broker.connect():
            logger.error("No se pudo conectar con el broker; abortando")
            return

        self._load_champion()

        self.state.running = True
        self.state.started_at = now_utc()

        logger.info("=" * 70)
        logger.info(
            "BOT EN MARCHA | broker=%s | modo=%s | dry_run=%s",
            self.broker.name, self.config.execution.mode, self.config.execution.dry_run,
        )
        logger.info("Campeon: %s", self._champion_id or "NINGUNO (solo incubacion)")
        logger.info("=" * 70)

        if self.notifier.is_configured:
            if self.config.telegram.allow_remote_control:
                self.telegram = TelegramBot(
                    self.notifier, self.config.telegram.poll_interval_seconds
                )
                for name, handler in build_command_handlers(self).items():
                    self.telegram.register(name, handler)
                self.telegram.start()
            self.notifier.startup(
                mode=self.config.execution.mode,
                dry_run=self.config.execution.dry_run,
                champion=self._champion_id,
                symbol=self._symbol(),
            )

        try:
            while self.state.running:
                if max_cycles is not None and self.state.cycles >= max_cycles:
                    logger.info("Alcanzado el limite de %d ciclos", max_cycles)
                    break

                if not market_is_open():
                    logger.info("Mercado cerrado; esperando 15 minutos")
                    self._sleep(900)
                    continue

                try:
                    self.run_cycle()
                    self.breaker.clear_errors()
                except Exception as exc:
                    self.state.errors += 1
                    logger.error("Error en el ciclo: %s", exc, exc_info=True)
                    self.breaker.check_error(exc)

                self.state.cycles += 1
                self._wait_for_next_bar()

        finally:
            self.stop()

    def stop(self) -> None:
        """Detiene el bucle y cierra la conexion."""
        self.state.running = False
        if self.telegram is not None:
            self.telegram.stop()
        try:
            self.broker.disconnect()
        except Exception as exc:
            logger.warning("Error al desconectar del broker: %s", exc)
        logger.info(
            "Bot detenido tras %d ciclos, %d ordenes, %d errores",
            self.state.cycles, self.state.orders_sent, self.state.errors,
        )

    # ------------------------------------------------------------------ #
    def run_cycle(self) -> None:
        """Un ciclo completo: datos -> senal -> riesgo -> ejecucion."""
        # --- 1) datos frescos --- #
        ohlcv = self.market_data.update()
        if ohlcv.empty or len(ohlcv) < 300:
            logger.warning("Datos insuficientes (%d barras); se omite el ciclo", len(ohlcv))
            return

        last_bar = ohlcv.index[-1].to_pydatetime()
        last_close = float(ohlcv["close"].iloc[-1])

        if not self.breaker.check_data_freshness(last_bar, max_delay_minutes=15):
            return
        if not self.breaker.check_price_sanity(last_close, self.state.last_price):
            return

        # Vela ya procesada: el proveedor aun no ha publicado la siguiente.
        if self.state.last_bar_time is not None and last_bar <= self.state.last_bar_time:
            logger.debug("Sin vela nueva (ultima: %s)", last_bar)
            return

        self.state.last_bar_time = last_bar
        self.state.last_price = last_close

        # --- 2) features --- #
        aligned, features, _ = build_features(ohlcv, self.feature_builder)
        if features.empty:
            logger.warning("No se pudieron calcular las features")
            return

        # --- 3) contabilidad del dia y de la cuenta --- #
        account = self.broker.get_account()
        today = last_bar.date()
        if self.state.current_day != today:
            self.state.current_day = today
            self.risk.new_day(today, account.equity)

        if not self.breaker.check_equity(
            account.equity, self.risk._peak_equity, self.risk._day_start_equity
        ):
            self._emergency_close("cortacircuitos disparado")
            return

        # --- 4) incubadoras en papel --- #
        self._run_incubators(aligned, features)

        # --- 5) campeon --- #
        if self._champion is None:
            logger.debug("Sin campeon activo; solo se incuba")
            self._record_equity(account)
            return

        if not self.breaker.allows_trading:
            logger.info("Cortacircuitos abierto; no se opera")
            return

        self._trade_champion(aligned, features, account, last_close)
        self._record_equity(account)

    # ------------------------------------------------------------------ #
    def _trade_champion(
        self, ohlcv: pd.DataFrame, features: pd.DataFrame, account, last_close: float
    ) -> None:
        """Aplica la estrategia campeona: gestiona lo abierto y abre si toca."""
        genome = self._champion
        assert genome is not None

        signals = genome.generate_signals(features, ohlcv)
        if signals.empty:
            return

        direction = int(signals.iloc[-1])
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        # ATR de la barra anterior: coherente con el motor de backtest.
        atr = float(atr_series.iloc[-2]) if len(atr_series) >= 2 else float("nan")

        confidence = 1.0
        if self._labeler is not None and self._labeler.is_ready:
            multiplier = self._labeler.size_multiplier(features)
            confidence = float(multiplier.iloc[-1])

        positions = self.broker.get_positions(self._symbol())

        # --- gestion de la posicion abierta --- #
        if positions:
            position = positions[0]
            if genome.exit_rules.exit_on_reverse and direction != 0 and direction != position.direction:
                self._close(position, "senal_contraria")
                positions = []
            else:
                self._update_trailing(position, atr, last_close)
                return  # con una posicion viva no se abre otra

        # --- apertura --- #
        if direction == 0:
            return

        if confidence <= 0:
            logger.info("Senal %+d descartada por el filtro de ML (confianza %.2f)", direction, confidence)
            return

        allowed, reason = self.risk.can_trade(account.equity)
        if not allowed:
            logger.info("Operacion bloqueada por gestion de riesgo: %s", reason)
            return

        try:
            bid, ask = self.broker.get_price(self._symbol())
        except Exception as exc:
            logger.warning("No se pudo obtener el precio: %s", exc)
            self.breaker.check_error(exc)
            return

        entry_reference = ask if direction > 0 else bid

        plan = self.risk.build_plan(
            direction=direction,
            entry_price=entry_reference,
            atr=atr,
            equity=account.equity,
            stop_atr=genome.exit_rules.stop_atr,
            target_atr=genome.exit_rules.target_atr,
            confidence=confidence * self.breaker.size_multiplier,
        )

        if not plan.approved:
            logger.info("Plan rechazado: %s", plan.reason)
            return

        logger.info("Ejecutando: %s", plan.summary())

        order = Order(
            side=OrderSide.from_direction(direction),
            volume=plan.lots,
            symbol=self._symbol(),
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            comment=f"goldbot:{self._champion_id}",
        )
        filled = self.broker.place_order(order)

        if filled.is_filled:
            self.state.orders_sent += 1
            self.risk.register_entry()
            trade_id = self.db.record_trade(
                strategy_id=self._champion_id or "unknown",
                mode="live" if not isinstance(self.broker, PaperBroker) else "paper",
                entry_time=now_utc().isoformat(),
                direction=direction,
                entry_price=filled.filled_price or entry_reference,
                lots=filled.filled_volume or plan.lots,
                order_id=filled.order_id,
                metadata={"confidence": confidence, "atr": atr, "plan": plan.summary()},
            )
            self._open_trade_ids[self._symbol()] = trade_id

            if self.config.telegram.notify_trades:
                self.notifier.trade_opened(
                    symbol=self._symbol(),
                    direction=direction,
                    lots=filled.filled_volume or plan.lots,
                    price=filled.filled_price or entry_reference,
                    stop=plan.stop_loss,
                    target=plan.take_profit,
                    strategy=self._champion_id or "?",
                    confidence=confidence,
                )
        else:
            logger.error("Orden no ejecutada: %s", filled.error)
            self.breaker.check_error(filled.error or "orden rechazada")

    def _update_trailing(self, position: Position, atr: float, last_close: float) -> None:
        """Mueve el stop segun las reglas de trailing/break-even del genoma."""
        genome = self._champion
        if genome is None or not np.isfinite(atr) or atr <= 0:
            return

        rules = genome.exit_rules
        if rules.trail_atr <= 0 and rules.breakeven_atr <= 0:
            return

        direction = position.direction
        gain = (last_close - position.entry_price) * direction
        new_stop = position.stop_loss

        if rules.breakeven_atr > 0 and gain >= rules.breakeven_atr * atr:
            candidate = position.entry_price
            if new_stop is None or (candidate - new_stop) * direction > 0:
                new_stop = candidate

        if rules.trail_atr > 0:
            candidate = last_close - direction * rules.trail_atr * atr
            if new_stop is None or (candidate - new_stop) * direction > 0:
                new_stop = candidate

        if new_stop is not None and new_stop != position.stop_loss:
            if self.broker.modify_position(position, stop_loss=new_stop):
                logger.info(
                    "Stop movido a %.2f (era %.2f)", new_stop, position.stop_loss or 0.0
                )

    def _close(self, position: Position, reason: str) -> None:
        order = self.broker.close_position(position, reason)
        if not order.is_filled:
            logger.error("No se pudo cerrar la posicion: %s", order.error)
            return

        pnl = float(order.metadata.get("pnl", 0.0))
        account = self.broker.get_account()

        self.risk.register_fill(pnl, account.equity)
        self.breaker.check_trade_result(pnl)

        trade_id = self._open_trade_ids.pop(position.symbol, None)
        if trade_id is not None:
            self.db.close_trade(
                trade_id, now_utc().isoformat(), order.filled_price or 0.0, pnl, reason
            )

        if self.config.telegram.notify_trades:
            duration = (now_utc() - position.opened_at).total_seconds() / 60
            self.notifier.trade_closed(
                symbol=position.symbol, pnl=pnl, reason=reason,
                balance=account.balance, duration_minutes=duration,
            )

    def _emergency_close(self, reason: str) -> None:
        """Cierra todo. Solo lo invocan los cortacircuitos."""
        logger.error("CIERRE DE EMERGENCIA: %s", reason)
        status = self.breaker.status()
        self.notifier.circuit_breaker(
            reason=status.get("last_reason", reason),
            detail=reason,
            resume_at=status.get("reset_at"),
        )
        try:
            for order in self.broker.close_all(self._symbol(), reason):
                logger.info("Cerrada: %s", order.summary())
        except Exception as exc:
            logger.critical("¡El cierre de emergencia fallo! %s", exc)

    # ------------------------------------------------------------------ #
    def _run_incubators(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> None:
        """Opera en papel todas las estrategias en incubacion.

        Es lo que convierte la incubadora en algo mas que una espera: cada
        candidata acumula operaciones reales contra precios reales, y esa es la
        evidencia que decide si llega a campeona.
        """
        incubating = self.registry.get_incubating()
        if not incubating:
            return

        high = float(ohlcv["high"].iloc[-1])
        low = float(ohlcv["low"].iloc[-1])
        close = float(ohlcv["close"].iloc[-1])

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        atr = float(atr_series.iloc[-2]) if len(atr_series) >= 2 else float("nan")
        if not np.isfinite(atr) or atr <= 0:
            return

        for record in incubating:
            broker = self.state.incubators.get(record.id)
            if broker is None:
                broker = PaperBroker(self.config, self.market_data)
                broker.connect()
                self.state.incubators[record.id] = broker

            try:
                genome = record.to_genome()
                broker.set_price(close)

                # Primero se comprueban stops/objetivos contra la vela cerrada.
                for order in broker.check_stops(high, low):
                    self._record_paper_close(record.id, order)

                signals = genome.generate_signals(features, ohlcv)
                if signals.empty:
                    continue
                direction = int(signals.iloc[-1])

                positions = broker.get_positions()
                if positions:
                    position = positions[0]
                    if genome.exit_rules.exit_on_reverse and direction != 0 and direction != position.direction:
                        self._record_paper_close(record.id, broker.close_position(position, "senal_contraria"))
                    else:
                        continue

                if direction == 0:
                    continue

                account = broker.get_account()
                plan = self.risk.build_plan(
                    direction=direction,
                    entry_price=close,
                    atr=atr,
                    equity=account.equity,
                    stop_atr=genome.exit_rules.stop_atr,
                    target_atr=genome.exit_rules.target_atr,
                )
                if not plan.approved:
                    continue

                order = Order(
                    side=OrderSide.from_direction(direction),
                    volume=plan.lots,
                    symbol=self._symbol(),
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    comment=f"incubacion:{record.id}",
                )
                filled = broker.place_order(order)
                if filled.is_filled:
                    self.db.record_trade(
                        strategy_id=record.id,
                        mode="paper",
                        entry_time=now_utc().isoformat(),
                        direction=direction,
                        entry_price=filled.filled_price or close,
                        lots=filled.filled_volume or plan.lots,
                        order_id=filled.order_id,
                    )

            except Exception as exc:
                logger.warning("Incubadora %s fallo: %s", record.id, exc)

    def _record_paper_close(self, strategy_id: str, order: Order) -> None:
        """Registra en la BD el cierre de una operacion de incubacion."""
        if not order.is_filled:
            return
        pnl = float(order.metadata.get("pnl", 0.0))
        trades = self.db.get_trades(strategy_id=strategy_id, mode="paper")
        open_trades = [t for t in trades if not t.get("exit_time")]
        if open_trades:
            self.db.close_trade(
                int(open_trades[-1]["id"]),
                now_utc().isoformat(),
                order.filled_price or 0.0,
                pnl,
                order.metadata.get("reason", "cierre"),
            )

    # ------------------------------------------------------------------ #
    def _load_champion(self) -> None:
        """Carga el campeon vigente y su modelo de ML."""
        record = self.registry.get_champion()
        if record is None:
            self._champion = None
            self._champion_id = None
            logger.warning(
                "No hay campeon. El bot incubara candidatas pero no operara en real. "
                "Ejecuta 'goldbot learn --bootstrap' para descubrir estrategias."
            )
            return

        self._champion = record.to_genome()
        self._champion_id = record.id
        self._labeler = self.trainer.load_active(record.id)

        logger.info("Campeon cargado:\n%s", self._champion.describe())
        if self._labeler and self._labeler.is_ready:
            logger.info("Filtro de ML activo (AUC %.3f)", self._labeler.report.auc)

    def reload_champion(self) -> None:
        """Recarga el campeon; el ciclo diario puede haberlo cambiado."""
        previous = self._champion_id
        self._load_champion()
        if previous != self._champion_id:
            logger.info("Campeon actualizado: %s -> %s", previous, self._champion_id)

    def _record_equity(self, account) -> None:
        if self._champion_id is None:
            return
        self.db.record_equity(
            strategy_id=self._champion_id,
            mode="live" if not isinstance(self.broker, PaperBroker) else "paper",
            timestamp=now_utc().isoformat(),
            equity=account.equity,
            balance=account.balance,
            open_positions=len(self.broker.get_positions(self._symbol())),
        )

    def _symbol(self) -> str:
        if self.config.execution.mode == "mt5":
            return self.config.execution.mt5_symbol
        if self.config.execution.mode == "ccxt":
            return self.config.execution.symbol
        return self.config.data.symbol

    def _wait_for_next_bar(self) -> None:
        """Duerme hasta poco despues del proximo cierre de vela."""
        delay = seconds_until_next_close(self.config.data.timeframe_minutes, offset_seconds=5.0)
        logger.debug("Esperando %.0fs hasta el proximo cierre de vela", delay)
        self._sleep(delay)

    def _sleep(self, seconds: float) -> None:
        """Duerme troceado para poder atender una senal de parada al instante."""
        end = time.time() + seconds
        while self.state.running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        try:
            account = self.broker.get_account()
            equity, balance = account.equity, account.balance
        except Exception:
            equity = balance = 0.0

        return {
            "running": self.state.running,
            "cycles": self.state.cycles,
            "champion": self._champion_id,
            "last_bar": self.state.last_bar_time.isoformat() if self.state.last_bar_time else None,
            "equity": equity,
            "balance": balance,
            "orders_sent": self.state.orders_sent,
            "errors": self.state.errors,
            "breaker": self.breaker.status(),
            "risk": self.risk.stats(),
            "incubators": list(self.state.incubators),
        }
