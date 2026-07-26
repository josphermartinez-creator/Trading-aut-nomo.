"""Persistencia en SQLite.

SQLite y no Postgres a proposito: el bot corre en un unico VPS, escribe poco y
lee menos, y un fichero unico se respalda con un ``scp``. Añadir un servidor de
base de datos seria complejidad operativa sin ninguna contrapartida.

Guarda el historial completo: cada estrategia descubierta, cada evaluacion,
cada operacion y cada ciclo diario de aprendizaje. Ese historial es lo que
permite responder a "por que el bot esta operando esto" meses despues.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from goldbot.utils.logging import get_logger

logger = get_logger(__name__)

# Estados del ciclo de vida de una estrategia.
STATUS_CANDIDATE = "candidate"    # recien descubierta, sin validar
STATUS_VALIDATED = "validated"    # supero walk-forward y Monte Carlo
STATUS_INCUBATING = "incubating"  # operando en paper, acumulando evidencia
STATUS_CHAMPION = "champion"      # operando en produccion
STATUS_RETIRED = "retired"        # degradada por perdida de rendimiento
STATUS_REJECTED = "rejected"      # no supero la puerta de calidad

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id            TEXT PRIMARY KEY,
    fingerprint   TEXT NOT NULL,
    genome        TEXT NOT NULL,
    status        TEXT NOT NULL,
    generation    INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    promoted_at   TEXT,
    retired_at    TEXT,
    description   TEXT,
    metrics       TEXT,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
CREATE INDEX IF NOT EXISTS idx_strategies_fingerprint ON strategies(fingerprint);

CREATE TABLE IF NOT EXISTS evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    kind         TEXT NOT NULL,          -- train | validation | walkforward | montecarlo | paper | live
    created_at   TEXT NOT NULL,
    period_start TEXT,
    period_end   TEXT,
    score        REAL,
    metrics      TEXT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);
CREATE INDEX IF NOT EXISTS idx_eval_strategy ON evaluations(strategy_id, kind);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    mode         TEXT NOT NULL,          -- paper | live
    entry_time   TEXT NOT NULL,
    exit_time    TEXT,
    direction    INTEGER NOT NULL,
    entry_price  REAL NOT NULL,
    exit_price   REAL,
    lots         REAL NOT NULL,
    pnl          REAL,
    exit_reason  TEXT,
    order_id     TEXT,
    metadata     TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id, mode);
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_time);

CREATE TABLE IF NOT EXISTS equity_curve (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    mode        TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    equity      REAL NOT NULL,
    balance     REAL,
    open_positions INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_equity_lookup ON equity_curve(strategy_id, mode, timestamp);

CREATE TABLE IF NOT EXISTS learning_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,           -- running | ok | failed
    stage       TEXT,
    summary     TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_date ON learning_runs(run_date);

CREATE TABLE IF NOT EXISTS ml_models (
    id          TEXT PRIMARY KEY,
    strategy_id TEXT,
    path        TEXT NOT NULL,
    model_type  TEXT NOT NULL,
    trained_at  TEXT NOT NULL,
    auc         REAL,
    accuracy    REAL,
    n_samples   INTEGER,
    features    TEXT,
    active      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kv_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class StrategyRecord:
    """Fila de la tabla ``strategies`` ya deserializada."""

    id: str
    fingerprint: str
    genome: dict[str, Any]
    status: str
    generation: int = 0
    created_at: str = ""
    updated_at: str = ""
    promoted_at: str | None = None
    retired_at: str | None = None
    description: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_genome(self):
        """Reconstruye el objeto :class:`StrategyGenome`."""
        from goldbot.strategies.genome import StrategyGenome

        return StrategyGenome.from_dict(self.genome)


class Database:
    """Acceso a SQLite, seguro entre hilos."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        """Una conexion por hilo: sqlite3 no permite compartirlas."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            # WAL permite que el runner en vivo escriba mientras el ciclo de
            # aprendizaje lee, sin bloquearse mutuamente.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(SCHEMA)
        logger.debug("Esquema de base de datos listo en %s", self.path)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ #
    # Estrategias
    # ------------------------------------------------------------------ #
    def save_strategy(
        self,
        genome,
        status: str = STATUS_CANDIDATE,
        metrics: dict | None = None,
        notes: str = "",
    ) -> str:
        """Inserta o actualiza una estrategia. Devuelve su id."""
        now = _now()
        payload = genome.to_dict()
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM strategies WHERE id = ?", (genome.genome_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE strategies SET status=?, metrics=?, notes=?, updated_at=?,
                       genome=?, description=? WHERE id=?""",
                    (
                        status,
                        json.dumps(metrics or {}),
                        notes,
                        now,
                        json.dumps(payload),
                        genome.describe(),
                        genome.genome_id,
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO strategies
                       (id, fingerprint, genome, status, generation, created_at, updated_at,
                        description, metrics, notes)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        genome.genome_id,
                        genome.fingerprint(),
                        json.dumps(payload),
                        status,
                        genome.generation,
                        now,
                        now,
                        genome.describe(),
                        json.dumps(metrics or {}),
                        notes,
                    ),
                )
        return genome.genome_id

    def update_status(self, strategy_id: str, status: str, notes: str = "") -> None:
        now = _now()
        column = (
            "promoted_at"
            if status == STATUS_CHAMPION
            else ("retired_at" if status == STATUS_RETIRED else None)
        )
        with self.transaction() as conn:
            if column:
                conn.execute(
                    f"UPDATE strategies SET status=?, updated_at=?, {column}=?, notes=? WHERE id=?",
                    (status, now, now, notes, strategy_id),
                )
            else:
                conn.execute(
                    "UPDATE strategies SET status=?, updated_at=?, notes=? WHERE id=?",
                    (status, now, notes, strategy_id),
                )
        logger.info("Estrategia %s -> %s %s", strategy_id, status, f"({notes})" if notes else "")

    def get_strategy(self, strategy_id: str) -> StrategyRecord | None:
        row = self._connect().execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        return _row_to_strategy(row) if row else None

    def find_by_fingerprint(self, fingerprint: str) -> StrategyRecord | None:
        """Evita reevaluar una estrategia estructuralmente identica ya conocida."""
        row = self._connect().execute(
            "SELECT * FROM strategies WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return _row_to_strategy(row) if row else None

    def list_strategies(self, status: str | None = None, limit: int = 100) -> list[StrategyRecord]:
        query = "SELECT * FROM strategies"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params = (*params, limit)
        rows = self._connect().execute(query, params).fetchall()
        return [_row_to_strategy(r) for r in rows]

    def get_champion(self) -> StrategyRecord | None:
        """La estrategia que esta operando ahora mismo (como mucho una)."""
        rows = self.list_strategies(status=STATUS_CHAMPION, limit=1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ #
    # Evaluaciones
    # ------------------------------------------------------------------ #
    def save_evaluation(
        self,
        strategy_id: str,
        kind: str,
        metrics: dict,
        score: float | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO evaluations
                   (strategy_id, kind, created_at, period_start, period_end, score, metrics)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    strategy_id,
                    kind,
                    _now(),
                    period_start,
                    period_end,
                    score,
                    json.dumps(metrics, default=str),
                ),
            )

    def get_evaluations(self, strategy_id: str, kind: str | None = None, limit: int = 50) -> list[dict]:
        query = "SELECT * FROM evaluations WHERE strategy_id = ?"
        params: list = [strategy_id]
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._connect().execute(query, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else {}
            out.append(item)
        return out

    # ------------------------------------------------------------------ #
    # Operaciones y equity
    # ------------------------------------------------------------------ #
    def record_trade(
        self,
        strategy_id: str,
        mode: str,
        entry_time: str,
        direction: int,
        entry_price: float,
        lots: float,
        exit_time: str | None = None,
        exit_price: float | None = None,
        pnl: float | None = None,
        exit_reason: str | None = None,
        order_id: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (strategy_id, mode, entry_time, exit_time, direction, entry_price,
                    exit_price, lots, pnl, exit_reason, order_id, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    strategy_id, mode, entry_time, exit_time, direction, entry_price,
                    exit_price, lots, pnl, exit_reason, order_id,
                    json.dumps(metadata or {}, default=str),
                ),
            )
            return int(cursor.lastrowid)

    def close_trade(
        self, trade_id: int, exit_time: str, exit_price: float, pnl: float, exit_reason: str
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE trades SET exit_time=?, exit_price=?, pnl=?, exit_reason=? WHERE id=?",
                (exit_time, exit_price, pnl, exit_reason, trade_id),
            )

    def get_trades(
        self, strategy_id: str | None = None, mode: str | None = None, since: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if since:
            query += " AND entry_time >= ?"
            params.append(since)
        query += " ORDER BY entry_time"
        return [dict(r) for r in self._connect().execute(query, params).fetchall()]

    def record_equity(
        self,
        strategy_id: str,
        mode: str,
        timestamp: str,
        equity: float,
        balance: float | None = None,
        open_positions: int = 0,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO equity_curve
                   (strategy_id, mode, timestamp, equity, balance, open_positions)
                   VALUES (?,?,?,?,?,?)""",
                (strategy_id, mode, timestamp, equity, balance, open_positions),
            )

    def get_equity_curve(self, strategy_id: str, mode: str, since: str | None = None) -> list[dict]:
        query = "SELECT timestamp, equity, balance FROM equity_curve WHERE strategy_id=? AND mode=?"
        params: list = [strategy_id, mode]
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp"
        return [dict(r) for r in self._connect().execute(query, params).fetchall()]

    # ------------------------------------------------------------------ #
    # Ciclos de aprendizaje
    # ------------------------------------------------------------------ #
    def start_run(self, run_date: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO learning_runs (run_date, started_at, status, stage) VALUES (?,?,?,?)",
                (run_date, _now(), "running", "inicio"),
            )
            return int(cursor.lastrowid)

    def update_run(self, run_id: int, stage: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE learning_runs SET stage=? WHERE id=?", (stage, run_id))

    def finish_run(
        self, run_id: int, status: str, summary: str = "", error: str = ""
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE learning_runs SET finished_at=?, status=?, summary=?, error=? WHERE id=?",
                (_now(), status, summary, error, run_id),
            )

    def last_successful_run(self) -> dict | None:
        row = self._connect().execute(
            "SELECT * FROM learning_runs WHERE status='ok' ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_runs(self, limit: int = 30) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM learning_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Modelos de ML
    # ------------------------------------------------------------------ #
    def save_model(
        self,
        model_id: str,
        path: str,
        model_type: str,
        auc: float,
        accuracy: float,
        n_samples: int,
        features: list[str],
        strategy_id: str | None = None,
        active: bool = True,
    ) -> None:
        with self.transaction() as conn:
            if active:
                # Solo un modelo activo a la vez para una estrategia dada.
                conn.execute(
                    "UPDATE ml_models SET active=0 WHERE strategy_id IS ? OR strategy_id = ?",
                    (strategy_id, strategy_id),
                )
            conn.execute(
                """INSERT OR REPLACE INTO ml_models
                   (id, strategy_id, path, model_type, trained_at, auc, accuracy,
                    n_samples, features, active)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_id, strategy_id, path, model_type, _now(), auc, accuracy,
                    n_samples, json.dumps(features), int(active),
                ),
            )

    def get_active_model(self, strategy_id: str | None = None) -> dict | None:
        row = self._connect().execute(
            "SELECT * FROM ml_models WHERE active=1 AND (strategy_id IS ? OR strategy_id=?) "
            "ORDER BY trained_at DESC LIMIT 1",
            (strategy_id, strategy_id),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["features"] = json.loads(item["features"]) if item["features"] else []
        return item

    # ------------------------------------------------------------------ #
    # Estado clave-valor
    # ------------------------------------------------------------------ #
    def set_state(self, key: str, value: Any) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_state (key, value, updated_at) VALUES (?,?,?)",
                (key, json.dumps(value, default=str), _now()),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._connect().execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default


# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _row_to_strategy(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        id=row["id"],
        fingerprint=row["fingerprint"],
        genome=json.loads(row["genome"]),
        status=row["status"],
        generation=row["generation"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        promoted_at=row["promoted_at"],
        retired_at=row["retired_at"],
        description=row["description"] or "",
        metrics=json.loads(row["metrics"]) if row["metrics"] else {},
        notes=row["notes"] or "",
    )
