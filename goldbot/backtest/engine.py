"""Motor de backtesting orientado a eventos, sobre arrays de numpy.

Prioridades del diseno, en este orden:

1. **Ausencia de look-ahead.** La senal de la barra ``t`` se ejecuta en la
   apertura de ``t+1``. El motor aplica ese desplazamiento internamente, asi
   que una estrategia no puede hacer trampa aunque quiera.
2. **Pesimismo ante la ambiguedad.** Si dentro de una vela se tocan stop y
   objetivo, se asume el stop: sin datos de tick no hay forma de saber el
   orden, y equivocarse hacia el lado optimista arruina cuentas reales.
3. **Velocidad.** El motor evolutivo lo llama miles de veces, asi que todo el
   bucle trabaja sobre arrays planos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goldbot.backtest.costs import CostModel, estimate_volatility_factor
from goldbot.backtest.metrics import PerformanceMetrics, compute_metrics
from goldbot.config import Config
from goldbot.features import indicators as ta
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)

# Codigos de motivo de salida (enteros para que el bucle no toque objetos Python).
EXIT_NONE = 0
EXIT_STOP = 1
EXIT_TARGET = 2
EXIT_TIME = 3
EXIT_SIGNAL = 4
EXIT_TRAIL = 5
EXIT_EOD = 6
EXIT_CIRCUIT = 7

EXIT_NAMES = {
    EXIT_NONE: "none",
    EXIT_STOP: "stop_loss",
    EXIT_TARGET: "take_profit",
    EXIT_TIME: "time_exit",
    EXIT_SIGNAL: "signal_exit",
    EXIT_TRAIL: "trailing_stop",
    EXIT_EOD: "end_of_data",
    EXIT_CIRCUIT: "circuit_breaker",
}


@dataclass
class ExitRules:
    """Como se cierra una posicion. Forma parte del genoma de la estrategia."""

    stop_atr: float = 2.0
    target_atr: float = 3.0
    trail_atr: float = 0.0        # 0 = sin trailing
    breakeven_atr: float = 0.0    # mover stop a BE tras N ATR a favor; 0 = off
    max_bars: int = 96            # 8 horas en M5
    exit_on_reverse: bool = True  # cerrar si la senal se invierte

    # Distancia minima ejecutable de un stop, en multiplos de ATR. Por debajo,
    # la distancia es inferior al ruido y a la horquilla, y el "trailing"
    # degenera en una orden de vender justo en el maximo de la vela: imposible
    # de replicar en real y una fuente garantizada de backtests fantasticos.
    # Si no se acota, el optimizador encuentra siempre estos valores.
    MIN_STOP_ATR: float = 0.5

    def validate(self) -> ExitRules:
        self.stop_atr = float(max(self.MIN_STOP_ATR, self.stop_atr))
        self.target_atr = float(max(0.2, self.target_atr))
        # Por debajo del minimo el trailing se desactiva en vez de recortarse:
        # conserva la intencion del genoma ("sin trailing") sin distorsionarla.
        self.trail_atr = float(self.trail_atr) if self.trail_atr >= self.MIN_STOP_ATR else 0.0
        self.breakeven_atr = (
            float(self.breakeven_atr) if self.breakeven_atr >= self.MIN_STOP_ATR else 0.0
        )
        self.max_bars = int(max(1, self.max_bars))
        return self


@dataclass
class Trade:
    """Una operacion cerrada."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    lots: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str
    mae: float  # maximum adverse excursion (USD)
    mfe: float  # maximum favourable excursion (USD)


@dataclass
class BacktestResult:
    """Salida completa de una simulacion."""

    equity: pd.Series
    trades: pd.DataFrame
    positions: pd.Series
    metrics: PerformanceMetrics
    initial_balance: float
    halted: bool = False
    halt_reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.trades) == 0

    def summary(self) -> str:
        base = self.metrics.summary()
        return f"{base} [DETENIDO: {self.halt_reason}]" if self.halted else base


class BacktestEngine:
    """Simulador de una estrategia sobre velas OHLCV."""

    def __init__(self, config: Config, cost_model: CostModel | None = None) -> None:
        self.config = config
        self.costs = cost_model or CostModel.from_config(config.costs)

    # ------------------------------------------------------------------ #
    def run(
        self,
        ohlcv: pd.DataFrame,
        signals: pd.Series,
        exit_rules: ExitRules | None = None,
        atr: pd.Series | None = None,
        size_multiplier: pd.Series | None = None,
        initial_balance: float | None = None,
        n_strategies_tried: int = 1,
    ) -> BacktestResult:
        """Simula ``signals`` sobre ``ohlcv``.

        Parameters
        ----------
        signals:
            +1 largo, -1 corto, 0 plano. Calculada con informacion hasta el
            cierre de cada barra; el motor la ejecuta en la apertura siguiente.
        size_multiplier:
            Escalado opcional del tamano por barra en [0, 1]. Es el canal por
            el que la capa de ML modula la exposicion segun su confianza.
        """
        rules = (exit_rules or ExitRules()).validate()
        balance = float(initial_balance if initial_balance is not None else self.config.risk.initial_balance)

        if ohlcv.empty or len(ohlcv) < 10:
            return self._empty_result(ohlcv, balance)

        signals = signals.reindex(ohlcv.index).fillna(0.0)
        if atr is None:
            atr = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        atr = atr.reindex(ohlcv.index).ffill()

        multiplier = (
            pd.Series(1.0, index=ohlcv.index)
            if size_multiplier is None
            else size_multiplier.reindex(ohlcv.index).fillna(0.0).clip(0.0, 1.0)
        )

        arrays = _Arrays(
            open_=ohlcv["open"].to_numpy(dtype="float64"),
            high=ohlcv["high"].to_numpy(dtype="float64"),
            low=ohlcv["low"].to_numpy(dtype="float64"),
            close=ohlcv["close"].to_numpy(dtype="float64"),
            atr=atr.to_numpy(dtype="float64"),
            signal=signals.to_numpy(dtype="float64"),
            multiplier=multiplier.to_numpy(dtype="float64"),
            day_id=_day_ids(ohlcv.index),
            vol_factor=estimate_volatility_factor((atr / ohlcv["close"]).to_numpy(dtype="float64")),
        )

        state = self._simulate(arrays, rules, balance)

        equity = pd.Series(state.equity, index=ohlcv.index, name="equity")
        positions = pd.Series(state.positions, index=ohlcv.index, name="position")
        trades = _trades_to_frame(state.trades, ohlcv.index)

        metrics = compute_metrics(
            equity=equity,
            trades=trades,
            initial_balance=balance,
            timeframe_minutes=self.config.data.timeframe_minutes,
            n_strategies_tried=n_strategies_tried,
        )

        return BacktestResult(
            equity=equity,
            trades=trades,
            positions=positions,
            metrics=metrics,
            initial_balance=balance,
            halted=state.halted,
            halt_reason=state.halt_reason,
            extra={"exit_rules": rules, "bars": len(ohlcv)},
        )

    # ------------------------------------------------------------------ #
    def _simulate(self, a: _Arrays, rules: ExitRules, balance: float) -> _State:
        """Bucle principal barra a barra.

        Orden de eventos dentro de la barra ``i``, elegido para que ninguna
        posicion disfrute de informacion o de inmunidad que no tendria en real:

        1. Salida por senal contraria, ejecutada en la apertura.
        2. Apertura de posicion nueva, tambien en la apertura.
        3. Gestion de la posicion **dentro de la misma barra** i: hueco contra
           la apertura, stop, objetivo, trailing y salida por tiempo.
        4. Valoracion a mercado y cortacircuitos.

        Que la gestion ocurra en la misma barra que la entrada es lo que impide
        el clasico "la vela de entrada nunca me salta el stop". Y todos los
        niveles se calculan con el ATR de la barra **anterior**, porque el ATR
        de la barra en curso contiene su maximo y su minimo: usarlo seria mirar
        al futuro.
        """
        risk_cfg = self.config.risk
        costs = self.costs
        contract = costs.contract_size

        n = len(a.close)
        equity = np.full(n, balance, dtype="float64")
        positions = np.zeros(n, dtype="float64")
        trades: list[Trade] = []

        cash = balance
        peak_equity = balance

        in_position = False
        direction = 0
        entry_price = 0.0
        lots = 0.0
        stop_level = 0.0
        target_level = 0.0
        entry_bar = 0
        best_price = 0.0
        worst_price = 0.0
        breakeven_done = False

        current_day = a.day_id[0]
        day_start_equity = balance
        trades_today = 0
        cooldown = 0

        halted = False
        halt_reason = ""

        def _close_position(bar: int, mid_price: float, code: int) -> float:
            """Cierra la posicion abierta y registra la operacion. Devuelve el P&L."""
            nonlocal in_position, direction, lots, cash
            fill = costs.exit_price(mid_price, direction, a.vol_factor[bar])
            nights = _count_nights(a.day_id, entry_bar, bar)
            pnl = costs.pnl(entry_price, fill, lots, direction, nights)
            cash += pnl
            trades.append(
                Trade(
                    entry_time=entry_bar,
                    exit_time=bar,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=fill,
                    lots=lots,
                    pnl=pnl,
                    pnl_pct=pnl / max(cash - pnl, 1e-9),
                    bars_held=bar - entry_bar,
                    exit_reason=EXIT_NAMES[code],
                    mae=(worst_price - entry_price) * direction * lots * contract,
                    mfe=(best_price - entry_price) * direction * lots * contract,
                )
            )
            in_position = False
            direction = 0
            lots = 0.0
            return pnl

        for i in range(1, n):
            # ---- cambio de dia: se reinician los limites diarios ----
            if a.day_id[i] != current_day:
                current_day = a.day_id[i]
                day_start_equity = equity[i - 1]
                trades_today = 0

            price_open = a.open_[i]
            high_i, low_i, close_i = a.high[i], a.low[i], a.close[i]

            # ATR de la barra previa: la unica volatilidad ya observable.
            atr_ref = a.atr[i - 1]
            has_atr = np.isfinite(atr_ref) and atr_ref > 0

            # ---- 1) salida por senal contraria, en la apertura ----
            if in_position and rules.exit_on_reverse and a.signal[i - 1] * direction < 0:
                if _close_position(i, price_open, EXIT_SIGNAL) < 0:
                    cooldown = risk_cfg.cooldown_bars_after_loss

            # ---- 2) apertura de posicion ----
            daily_loss = (
                (day_start_equity - equity[i - 1]) / day_start_equity if day_start_equity > 0 else 0.0
            )
            can_open = (
                not in_position
                and cooldown <= 0
                and has_atr
                and daily_loss < risk_cfg.max_daily_loss_pct
                and trades_today < risk_cfg.max_trades_per_day
            )

            if can_open:
                desired = a.signal[i - 1]
                size_scale = a.multiplier[i - 1]
                if desired != 0 and size_scale > 0:
                    new_direction = 1 if desired > 0 else -1
                    stop_distance = rules.stop_atr * atr_ref
                    candidate_lots = self._position_size(equity[i - 1], stop_distance, size_scale)

                    if stop_distance > 0 and candidate_lots >= risk_cfg.min_lot_size:
                        lots = candidate_lots
                        direction = new_direction
                        entry_price = costs.entry_price(price_open, direction, a.vol_factor[i])
                        stop_level = entry_price - direction * stop_distance
                        target_level = entry_price + direction * rules.target_atr * atr_ref
                        entry_bar = i
                        best_price = worst_price = entry_price
                        breakeven_done = False
                        in_position = True
                        trades_today += 1

            # ---- 3) gestion dentro de la barra ----
            if in_position:
                best_price = max(best_price, high_i) if direction > 0 else min(best_price, low_i)
                worst_price = min(worst_price, low_i) if direction > 0 else max(worst_price, high_i)

                exit_code = EXIT_NONE
                exit_mid = 0.0

                # 3a) Hueco de apertura: si la vela abre ya pasado un nivel, el
                #     fill real es la apertura, no el nivel teorico.
                if entry_bar < i:
                    gapped_stop = price_open <= stop_level if direction > 0 else price_open >= stop_level
                    gapped_target = (
                        price_open >= target_level if direction > 0 else price_open <= target_level
                    )
                    if gapped_stop:
                        exit_code, exit_mid = EXIT_STOP, price_open
                    elif gapped_target:
                        exit_code, exit_mid = EXIT_TARGET, price_open

                # 3b) Recorrido dentro de la vela. Ante la duda, gana el stop:
                #     sin datos de tick no se puede saber el orden real.
                if exit_code == EXIT_NONE:
                    hit_stop = low_i <= stop_level if direction > 0 else high_i >= stop_level
                    hit_target = high_i >= target_level if direction > 0 else low_i <= target_level
                    if hit_stop:
                        exit_code, exit_mid = EXIT_STOP, stop_level
                    elif hit_target:
                        exit_code, exit_mid = EXIT_TARGET, target_level

                # 3c) Break-even y trailing. Se actualizan con los extremos de
                #     ESTA vela y se comprueban acto seguido contra ella misma:
                #     de lo contrario la posicion tendria una barra de ventaja
                #     gratuita entre que el trailing sube y puede saltar.
                if exit_code == EXIT_NONE and has_atr:
                    if rules.breakeven_atr > 0 and not breakeven_done:
                        if (best_price - entry_price) * direction >= rules.breakeven_atr * atr_ref:
                            stop_level = (
                                max(stop_level, entry_price)
                                if direction > 0
                                else min(stop_level, entry_price)
                            )
                            breakeven_done = True

                    if rules.trail_atr > 0:
                        trail = best_price - direction * rules.trail_atr * atr_ref
                        stop_level = max(stop_level, trail) if direction > 0 else min(stop_level, trail)

                    retraced = low_i <= stop_level if direction > 0 else high_i >= stop_level
                    if retraced:
                        exit_code, exit_mid = EXIT_TRAIL, stop_level

                # 3d) Barrera temporal.
                if exit_code == EXIT_NONE and (i - entry_bar) >= rules.max_bars:
                    exit_code, exit_mid = EXIT_TIME, close_i

                if exit_code != EXIT_NONE:
                    if _close_position(i, exit_mid, exit_code) < 0:
                        cooldown = risk_cfg.cooldown_bars_after_loss

            # ---- 4) valoracion a mercado ----
            if in_position:
                equity[i] = cash + (close_i - entry_price) * direction * lots * contract
                positions[i] = direction * lots
            else:
                equity[i] = cash
                positions[i] = 0.0

            peak_equity = max(peak_equity, equity[i])
            if cooldown > 0:
                cooldown -= 1

            # ---- cortacircuito por drawdown maximo ----
            drawdown = (peak_equity - equity[i]) / peak_equity if peak_equity > 0 else 0.0
            if drawdown >= risk_cfg.max_drawdown_pct:
                if in_position:
                    _close_position(i, close_i, EXIT_CIRCUIT)
                equity[i:] = cash
                positions[i:] = 0.0
                halted, halt_reason = True, f"drawdown maximo {drawdown:.1%}"
                break

        # ---- cierre forzoso al agotarse los datos ----
        if in_position:
            last = n - 1
            _close_position(last, a.close[last], EXIT_EOD)
            equity[last] = cash
            positions[last] = 0.0

        return _State(
            equity=equity, positions=positions, trades=trades, halted=halted, halt_reason=halt_reason
        )

    # ------------------------------------------------------------------ #
    def _position_size(self, equity: float, stop_distance: float, scale: float) -> float:
        """Tamano por riesgo fijo: se arriesga siempre el mismo % del capital.

        Es lo que hace que la curva sea geometrica sin exponerse a la ruina:
        el tamano se adapta a la distancia del stop, no al reves.
        """
        cfg = self.config.risk
        if equity <= 0 or stop_distance <= 0:
            return 0.0

        risk_amount = equity * cfg.risk_per_trade * scale
        raw_lots = risk_amount / (stop_distance * self.costs.contract_size)

        # Redondeo al paso de lote del broker (siempre hacia abajo).
        stepped = np.floor(raw_lots / cfg.lot_step) * cfg.lot_step
        return float(np.clip(stepped, 0.0, cfg.max_lot_size))

    def _empty_result(self, ohlcv: pd.DataFrame, balance: float) -> BacktestResult:
        index = ohlcv.index if not ohlcv.empty else pd.DatetimeIndex([], tz="UTC")
        equity = pd.Series(balance, index=index, dtype="float64", name="equity")
        return BacktestResult(
            equity=equity,
            trades=_empty_trades(),
            positions=pd.Series(0.0, index=index, name="position"),
            metrics=PerformanceMetrics(notes=["datos insuficientes"]),
            initial_balance=balance,
        )


# --------------------------------------------------------------------------- #
# Estructuras internas
# --------------------------------------------------------------------------- #
@dataclass
class _Arrays:
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    signal: np.ndarray
    multiplier: np.ndarray
    day_id: np.ndarray
    vol_factor: np.ndarray


@dataclass
class _State:
    equity: np.ndarray
    positions: np.ndarray
    trades: list[Trade]
    halted: bool
    halt_reason: str


def _day_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Ordinal de dia natural (UTC), para los limites diarios y los swaps.

    Se usa ``to_period('D')`` y no aritmetica sobre el entero subyacente porque
    la resolucion de los datetime cambia entre versiones de pandas (ns en 2.x,
    us en 3.x): dividir por una constante en nanosegundos da resultados
    silenciosamente erroneos.
    """
    # tz_localize(None) explicito: el indice ya esta en UTC, asi que descartar
    # la zona es intencionado y evita el aviso de pandas al pasar a Period.
    return index.tz_localize(None).to_period("D").astype("int64").to_numpy()


def _count_nights(day_id: np.ndarray, entry_bar: int, exit_bar: int) -> int:
    """Numero de cambios de dia mientras la posicion estuvo abierta (swaps)."""
    return int(day_id[exit_bar] - day_id[entry_bar])


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "lots", "pnl", "pnl_pct", "bars_held", "exit_reason", "mae", "mfe",
        ]
    )


def _trades_to_frame(trades: list[Trade], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Convierte las operaciones (con indices enteros) a un DataFrame con fechas."""
    if not trades:
        return _empty_trades()

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["entry_time"] = index[df["entry_time"].to_numpy(dtype="int64")]
    df["exit_time"] = index[df["exit_time"].to_numpy(dtype="int64")]
    return df
