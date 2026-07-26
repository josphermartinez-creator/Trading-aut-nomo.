"""Interfaz de linea de comandos.

Todo el sistema se maneja desde aqui:

    goldbot data --update           descargar y cachear velas
    goldbot learn --bootstrap       descubrir estrategias desde cero
    goldbot learn                   ciclo diario de aprendizaje
    goldbot backtest <id>           backtest de una estrategia guardada
    goldbot run                     bucle de trading en vivo
    goldbot schedule                aprendizaje diario + trading, todo en uno
    goldbot status                  estado del sistema
    goldbot strategies              listar estrategias descubiertas
    goldbot report <id>             informe detallado de una estrategia
"""

from __future__ import annotations

import argparse
import json
import sys

from goldbot import __version__
from goldbot.config import Config, load_config
from goldbot.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
def cmd_data(args, config: Config) -> int:
    """Descarga y diagnostica el historico."""
    from goldbot.data.pipeline import MarketData

    market = MarketData(config)

    if args.update:
        df = market.update(lookback_days=args.days)
    else:
        df = market.load()

    if df.empty:
        print("No hay datos disponibles.")
        return 1

    quality = market.assess(df)
    print(f"\nSimbolo    : {config.data.symbol} {config.data.timeframe}")
    print(f"Barras     : {quality.total_bars:,}")
    print(f"Periodo    : {quality.start:%Y-%m-%d %H:%M} -> {quality.end:%Y-%m-%d %H:%M}")
    print(f"Calidad    : {quality.summary()}")
    print(f"Utilizable : {'SI' if quality.is_usable else 'NO'}")
    if quality.notes:
        print("Avisos     :")
        for note in quality.notes:
            print(f"  - {note}")

    print(f"\nCache      : {market.cache.path}")
    return 0


def cmd_learn(args, config: Config) -> int:
    """Ejecuta el ciclo de aprendizaje."""
    from goldbot.autonomy.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    report = orchestrator.bootstrap() if args.bootstrap else orchestrator.run_daily_cycle(
        force_discovery=args.force_discovery
    )

    print("\n" + report.summary())
    return 0 if report.success else 1


def cmd_backtest(args, config: Config) -> int:
    """Backtest de una estrategia guardada, o del campeon."""
    from goldbot.backtest.engine import BacktestEngine
    from goldbot.backtest.walkforward import monte_carlo_analysis
    from goldbot.data.pipeline import MarketData
    from goldbot.features.engineering import build_features
    from goldbot.storage.db import Database

    db = Database(config.path(config.db_path))
    record = db.get_champion() if args.strategy_id == "champion" else db.get_strategy(args.strategy_id)

    if record is None:
        print(f"No se encontro la estrategia '{args.strategy_id}'.")
        return 1

    genome = record.to_genome()
    print(f"\n{genome.describe()}\n")

    ohlcv, features, _ = build_features(MarketData(config).get(refresh=args.refresh))
    result = BacktestEngine(config).run(ohlcv, genome.generate_signals(features, ohlcv), genome.exit_rules)

    m = result.metrics
    print(f"Periodo            : {ohlcv.index[0]:%Y-%m-%d} -> {ohlcv.index[-1]:%Y-%m-%d} ({len(ohlcv):,} barras)")
    print(f"Rentabilidad total : {m.total_return:+.2%}")
    print(f"Rentabilidad anual : {m.annual_return:+.2%}")
    print(f"Sharpe / Sortino   : {m.sharpe:.2f} / {m.sortino:.2f}")
    print(f"Calmar             : {m.calmar:.2f}")
    print(f"Drawdown maximo    : {m.max_drawdown:.2%}")
    print(f"Operaciones        : {m.total_trades} ({m.trades_per_day:.1f}/dia)")
    print(f"Aciertos           : {m.win_rate:.1%}")
    print(f"Profit factor      : {m.profit_factor:.2f}")
    print(f"Esperanza          : {m.expectancy:+.2f} USD/operacion")
    print(f"R2 de la equity    : {m.equity_r2:.3f}")
    print(f"Meses en verde     : {m.monthly_win_rate:.0%}")
    print(f"Sharpe deflactado  : {m.deflated_sharpe:.3f}")

    if result.halted:
        print(f"\n*** DETENIDA: {result.halt_reason} ***")

    if not result.trades.empty:
        print("\nSalidas por tipo:")
        for reason, count in result.trades["exit_reason"].value_counts().items():
            subset = result.trades[result.trades["exit_reason"] == reason]
            print(f"  {reason:<18} {count:>5}  P&L total {subset['pnl'].sum():>+10.2f}")

        mc = monte_carlo_analysis(
            result.trades, config.risk.initial_balance, config.backtest.monte_carlo_runs,
            config.stability.max_drawdown_pct,
        )
        print(f"\n{mc.summary()}")

    if args.walkforward:
        from goldbot.backtest.walkforward import WalkForwardAnalyzer

        wf = WalkForwardAnalyzer(config).run(genome, ohlcv, features)
        print(f"\n{wf.summary()}")
        for fold in wf.folds:
            print(
                f"  Tramo {fold.fold}: OOS {fold.test_start:%Y-%m-%d}->{fold.test_end:%Y-%m-%d} "
                f"| ret {fold.out_of_sample.total_return:+.2%} "
                f"| Sharpe {fold.out_of_sample.sharpe:.2f} "
                f"| eficiencia {fold.efficiency:.2f}"
            )

    return 0


def cmd_run(args, config: Config) -> int:
    """Arranca el bucle de trading en vivo."""
    from goldbot.live.runner import LiveRunner

    if not config.execution.dry_run and config.execution.mode != "paper":
        print("\n" + "!" * 70)
        print("ATENCION: dry_run esta DESACTIVADO. Se enviaran ordenes REALES.")
        print(f"Modo: {config.execution.mode} | Simbolo: {config.execution.symbol}")
        print("!" * 70)
        if not args.yes:
            answer = input("\nEscribe 'OPERAR EN REAL' para continuar: ")
            if answer.strip() != "OPERAR EN REAL":
                print("Cancelado.")
                return 1

    LiveRunner(config).start(max_cycles=args.max_cycles)
    return 0


def cmd_schedule(args, config: Config) -> int:
    """Trading en vivo + ciclo de aprendizaje diario, en un solo proceso."""
    from goldbot.scheduler import Scheduler

    Scheduler(config).start()
    return 0


def cmd_status(args, config: Config) -> int:
    """Estado del sistema."""
    from goldbot.autonomy.orchestrator import Orchestrator

    status = Orchestrator(config).status()

    print("\n=== ESTADO DE GOLDBOT ===")
    print(f"Version        : {__version__}")
    print(f"Modo ejecucion : {status['modo_ejecucion']} (dry_run={status['dry_run']})")

    data = status["datos"]
    print(f"\nDatos          : {data['barras']:,} barras")
    if data["desde"]:
        print(f"                 {data['desde'][:10]} -> {data['hasta'][:10]}")

    registry = status["registro"]
    print(f"\nCampeon        : {registry['champion'] or 'NINGUNO'}")
    if registry["champion"]:
        print(f"                 desde {(registry['champion_since'] or '')[:19]}")
        print(f"                 puntuacion de estabilidad {registry['champion_score']:.3f}")
    print(f"En incubacion  : {len(registry['incubating'])} {registry['incubating']}")
    print(f"Validadas      : {registry['validated_waiting']} esperando hueco")
    print(f"Descubiertas   : {registry['total_discovered']} en total")
    print(f"\nUltimo ciclo OK: {status['ultimo_ciclo_ok'] or 'nunca'}")

    if args.json:
        print("\n" + json.dumps(status, indent=2, default=str))
    return 0


def cmd_strategies(args, config: Config) -> int:
    """Lista las estrategias descubiertas."""
    from goldbot.storage.db import Database

    db = Database(config.path(config.db_path))
    records = db.list_strategies(status=args.status, limit=args.limit)

    if not records:
        print("No hay estrategias registradas todavia. Ejecuta 'goldbot learn --bootstrap'.")
        return 0

    print(f"\n{'ID':<14} {'ESTADO':<12} {'PUNT.':>7} {'SHARPE':>7} {'DD':>7} {'OPS':>6}  CREADA")
    print("-" * 80)
    for record in records:
        stability = record.metrics.get("stability", {})
        metrics = stability.get("metrics", {})
        print(
            f"{record.id:<14} {record.status:<12} "
            f"{stability.get('score', 0.0):>7.3f} "
            f"{metrics.get('sharpe', 0.0):>7.2f} "
            f"{metrics.get('max_drawdown', 0.0):>6.1%} "
            f"{int(metrics.get('total_trades', 0)):>6}  "
            f"{record.created_at[:10]}"
        )
    return 0


def cmd_report(args, config: Config) -> int:
    """Informe detallado de una estrategia."""
    from goldbot.storage.db import Database

    db = Database(config.path(config.db_path))
    record = db.get_champion() if args.strategy_id == "champion" else db.get_strategy(args.strategy_id)

    if record is None:
        print(f"No se encontro la estrategia '{args.strategy_id}'.")
        return 1

    print(f"\n=== ESTRATEGIA {record.id} ===")
    print(f"Estado    : {record.status}")
    print(f"Creada    : {record.created_at[:19]}")
    if record.promoted_at:
        print(f"Promovida : {record.promoted_at[:19]}")
    if record.notes:
        print(f"Notas     : {record.notes}")

    print(f"\n{record.description}\n")

    stability = record.metrics.get("stability", {})
    if stability:
        print(f"Puntuacion de estabilidad: {stability.get('score', 0):.3f} "
              f"({'APTA' if stability.get('passed') else 'DESCARTADA'})")
        print("\nPruebas superadas:")
        for check in stability.get("checks", []):
            mark = "OK  " if check["passed"] else "FALLA"
            print(f"  [{mark}] {check['name']}: {check['detail']}")

    trades = db.get_trades(strategy_id=record.id)
    if trades:
        closed = [t for t in trades if t.get("pnl") is not None]
        total_pnl = sum(float(t["pnl"]) for t in closed)
        wins = sum(1 for t in closed if float(t["pnl"]) > 0)
        print(f"\nOperativa real: {len(closed)} operaciones cerradas")
        print(f"  P&L acumulado : {total_pnl:+.2f} USD")
        if closed:
            print(f"  Aciertos      : {wins / len(closed):.1%}")

    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldbot",
        description="Bot autonomo de trading de oro (XAU/USD) en M5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", help="ruta al YAML de configuracion")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--version", action="version", version=f"goldbot {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="descargar y diagnosticar el historico")
    p.add_argument("--update", action="store_true", help="descargar velas nuevas")
    p.add_argument("--days", type=int, default=None, help="dias de historico a solicitar")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("learn", help="ciclo de aprendizaje")
    p.add_argument("--bootstrap", action="store_true", help="arranque en frio (descubre desde cero)")
    p.add_argument("--force-discovery", action="store_true", help="forzar la etapa de descubrimiento")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("backtest", help="backtest de una estrategia")
    p.add_argument("strategy_id", nargs="?", default="champion", help="id o 'champion'")
    p.add_argument("--walkforward", action="store_true", help="incluir analisis walk-forward")
    p.add_argument("--refresh", action="store_true", help="actualizar datos antes")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("run", help="bucle de trading en vivo")
    p.add_argument("--max-cycles", type=int, default=None, help="detenerse tras N ciclos")
    p.add_argument("--yes", "-y", action="store_true", help="omitir la confirmacion de operativa real")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("schedule", help="trading + aprendizaje diario en un proceso")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("status", help="estado del sistema")
    p.add_argument("--json", action="store_true", help="salida en JSON")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("strategies", help="listar estrategias")
    p.add_argument("--status", default=None, help="filtrar por estado")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser("report", help="informe de una estrategia")
    p.add_argument("strategy_id", nargs="?", default="champion", help="id o 'champion'")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 2

    setup_logging(
        level=args.log_level or config.log_level,
        log_file=config.path(config.log_file),
    )

    try:
        return args.func(args, config)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        return 130
    except Exception as exc:
        logger.error("Fallo del comando: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
