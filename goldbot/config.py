"""Configuracion tipada del sistema.

Toda la configuracion vive en un YAML (ver ``configs/default.yaml``). Las
credenciales NUNCA se guardan en el YAML: se leen de variables de entorno.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class DataConfig:
    """De donde salen las velas y como se cachean."""

    symbol: str = "XAUUSD"
    timeframe: str = "5m"
    timeframe_minutes: int = 5
    # Orden de preferencia de proveedores; el primero que devuelva datos gana.
    providers: list[str] = field(default_factory=lambda: ["yfinance", "ccxt", "csv"])
    yfinance_symbol: str = "GC=F"          # futuros del oro (COMEX)
    yfinance_fallback: str = "XAUUSD=X"    # spot FX como respaldo
    ccxt_exchange: str = "binance"
    ccxt_symbol: str = "PAXG/USDT"         # token respaldado por oro fisico 1:1
    csv_path: str = "data/XAUUSD_M5.csv"
    cache_dir: str = "data/cache"
    history_days: int = 720                # historico objetivo para entrenar
    min_bars_required: int = 5_000         # por debajo de esto no se entrena
    max_gap_minutes: int = 120             # hueco maximo tolerado sin marcar sesion rota


@dataclass
class CostConfig:
    """Costes de transaccion. Son la diferencia entre backtest y realidad."""

    spread_points: float = 0.25      # spread tipico XAUUSD en USD por onza
    commission_per_lot: float = 7.0  # USD ida y vuelta por lote estandar
    slippage_points: float = 0.10    # deslizamiento medio por ejecucion
    contract_size: float = 100.0     # onzas por lote estandar
    swap_long: float = -0.5          # USD/lote/noche
    swap_short: float = 0.2


@dataclass
class RiskConfig:
    """Limites duros. El bot nunca los sobrepasa, gane lo que gane."""

    initial_balance: float = 10_000.0
    risk_per_trade: float = 0.005        # 0.5% del capital por operacion
    max_risk_per_trade: float = 0.02
    max_concurrent_positions: int = 1
    max_daily_loss_pct: float = 0.03     # corta el dia al -3%
    max_drawdown_pct: float = 0.15       # apaga el bot al -15%
    max_lot_size: float = 5.0
    min_lot_size: float = 0.01
    lot_step: float = 0.01
    use_kelly: bool = True
    kelly_fraction: float = 0.25         # Kelly fraccional: 1/4 del optimo teorico
    atr_stop_multiplier: float = 2.0
    atr_target_multiplier: float = 3.0
    max_trades_per_day: int = 20
    cooldown_bars_after_loss: int = 3


@dataclass
class EvolutionConfig:
    """Parametros del algoritmo genetico que inventa estrategias."""

    population_size: int = 120
    generations: int = 40
    crossover_prob: float = 0.6
    mutation_prob: float = 0.35
    tournament_size: int = 4
    elitism: int = 6
    hall_of_fame: int = 25
    max_conditions: int = 4           # complejidad maxima de una regla
    random_seed: int | None = 42
    n_jobs: int = -1
    early_stop_generations: int = 12  # paciencia sin mejora
    min_trades: int = 40              # menos operaciones => muestra no significativa


@dataclass
class OptunaConfig:
    """Refinamiento fino de los mejores genomas."""

    enabled: bool = True
    n_trials: int = 80
    timeout_seconds: int = 900
    n_startup_trials: int = 15
    pruner_warmup_steps: int = 5


@dataclass
class MLConfig:
    """Capa de meta-etiquetado: filtra las senales del genoma."""

    enabled: bool = True
    model_type: str = "gradient_boosting"  # gradient_boosting | random_forest | logistic | mlp | torch_mlp
    lookahead_bars: int = 24               # horizonte del triple barrier (2h en M5)
    profit_take_atr: float = 1.5
    stop_loss_atr: float = 1.0
    min_probability: float = 0.55          # umbral para aceptar la senal
    train_test_split: float = 0.75
    purge_bars: int = 24                   # purga para evitar solapamiento de etiquetas
    embargo_bars: int = 12
    max_features: int = 40
    retrain_every_days: int = 1
    min_auc: float = 0.52                  # por debajo, el modelo no se usa


@dataclass
class BacktestConfig:
    """Validacion: walk-forward + Monte Carlo."""

    walk_forward_folds: int = 6
    train_ratio: float = 0.7
    purge_bars: int = 24
    monte_carlo_runs: int = 500
    monte_carlo_confidence: float = 0.95
    min_oos_ratio: float = 0.45          # rendimiento OOS / IS minimo aceptable


@dataclass
class StabilityConfig:
    """Puerta de calidad. Solo lo que la atraviesa opera con dinero."""

    min_sharpe: float = 1.0
    min_profit_factor: float = 1.15
    max_drawdown_pct: float = 0.20
    min_trades: int = 50
    min_win_rate: float = 0.35
    min_folds_profitable: float = 0.65    # fraccion de folds walk-forward en verde
    max_return_std_across_folds: float = 2.5
    incubation_days: int = 10             # dias en paper antes de tocar dinero real
    incubation_min_trades: int = 25
    degradation_sharpe_drop: float = 0.5  # caida que dispara la degradacion
    consecutive_bad_days: int = 5


@dataclass
class ExecutionConfig:
    """Conexion con el broker."""

    mode: str = "paper"               # paper | ccxt | mt5
    exchange: str = "binance"
    symbol: str = "PAXG/USDT"
    testnet: bool = True
    mt5_symbol: str = "XAUUSD"
    mt5_magic: int = 20250726
    order_type: str = "market"
    max_slippage_pct: float = 0.002
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    dry_run: bool = True              # con True jamas se manda una orden real


@dataclass
class AutonomyConfig:
    """El ciclo diario de aprendizaje."""

    daily_learning_hour_utc: int = 22      # tras el cierre de NY
    discovery_every_days: int = 3          # cada cuanto se lanza la evolucion completa
    keep_top_strategies: int = 10
    challenger_slots: int = 3
    promote_min_improvement: float = 0.15  # el retador debe superar al campeon un 15%
    auto_promote: bool = True
    auto_demote: bool = True
    max_runtime_minutes: int = 120


@dataclass
class Config:
    """Configuracion raiz."""

    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)

    log_level: str = "INFO"
    log_file: str = "logs/goldbot.log"
    db_path: str = "data/goldbot.db"
    artifacts_dir: str = "artifacts"

    # --- credenciales, siempre desde el entorno ---
    @property
    def api_key(self) -> str | None:
        return os.getenv("GOLDBOT_API_KEY")

    @property
    def api_secret(self) -> str | None:
        return os.getenv("GOLDBOT_API_SECRET")

    @property
    def mt5_login(self) -> int | None:
        raw = os.getenv("GOLDBOT_MT5_LOGIN")
        return int(raw) if raw and raw.isdigit() else None

    @property
    def mt5_password(self) -> str | None:
        return os.getenv("GOLDBOT_MT5_PASSWORD")

    @property
    def mt5_server(self) -> str | None:
        return os.getenv("GOLDBOT_MT5_SERVER")

    # --- serializacion ---
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return _build(cls, raw or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Carga la configuracion, aplicando los overrides de entorno."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        cfg._apply_env_overrides()
        cfg.validate()
        return cfg

    def _apply_env_overrides(self) -> None:
        """Permite ajustar cosas puntuales sin tocar el YAML (util en Docker)."""
        if mode := os.getenv("GOLDBOT_MODE"):
            self.execution.mode = mode
        if (dry := os.getenv("GOLDBOT_DRY_RUN")) is not None:
            self.execution.dry_run = dry.strip().lower() in {"1", "true", "yes", "on"}
        if level := os.getenv("GOLDBOT_LOG_LEVEL"):
            self.log_level = level
        if balance := os.getenv("GOLDBOT_INITIAL_BALANCE"):
            try:
                self.risk.initial_balance = float(balance)
            except ValueError:
                pass

    def validate(self) -> None:
        """Comprueba invariantes que, de romperse, arruinarian la cuenta."""
        r = self.risk
        if not 0 < r.risk_per_trade <= r.max_risk_per_trade:
            raise ValueError(
                f"risk_per_trade ({r.risk_per_trade}) debe estar en (0, max_risk_per_trade={r.max_risk_per_trade}]"
            )
        if r.max_risk_per_trade > 0.05:
            raise ValueError("max_risk_per_trade > 5% es temerario; revisa la configuracion")
        if not 0 < r.max_daily_loss_pct < r.max_drawdown_pct:
            raise ValueError("Se requiere 0 < max_daily_loss_pct < max_drawdown_pct")
        if r.min_lot_size <= 0 or r.max_lot_size < r.min_lot_size:
            raise ValueError("Tamanos de lote incoherentes")
        if r.initial_balance <= 0:
            raise ValueError("initial_balance debe ser positivo")
        if self.data.timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes debe ser positivo")
        if not 0 < self.ml.min_probability < 1:
            raise ValueError("ml.min_probability debe estar en (0, 1)")
        if not 0 < self.backtest.train_ratio < 1:
            raise ValueError("backtest.train_ratio debe estar en (0, 1)")
        if self.execution.mode not in {"paper", "ccxt", "mt5"}:
            raise ValueError(f"Modo de ejecucion desconocido: {self.execution.mode}")
        if self.execution.mode != "paper" and self.execution.dry_run is False:
            # No es un error, pero merece quedar registrado de forma explicita.
            os.environ.setdefault("GOLDBOT_LIVE_ACK", "0")

    # --- rutas derivadas ---
    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Construye recursivamente dataclasses anidadas ignorando claves extra."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if is_dataclass(f.type) and isinstance(value, dict) or isinstance(value, dict) and hasattr(f.type, "__dataclass_fields__"):
            kwargs[f.name] = _build(f.type, value)
        else:
            kwargs[f.name] = value
    # Las anotaciones pueden llegar como cadenas (``from __future__ import annotations``),
    # asi que resolvemos las secciones anidadas por nombre conocido.
    nested = {
        "data": DataConfig,
        "costs": CostConfig,
        "risk": RiskConfig,
        "evolution": EvolutionConfig,
        "optuna": OptunaConfig,
        "ml": MLConfig,
        "backtest": BacktestConfig,
        "stability": StabilityConfig,
        "execution": ExecutionConfig,
        "autonomy": AutonomyConfig,
    }
    if cls is Config:
        for key, sub_cls in nested.items():
            if isinstance(raw.get(key), dict):
                kwargs[key] = _build(sub_cls, raw[key])
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Atajo de conveniencia."""
    return Config.load(path)
