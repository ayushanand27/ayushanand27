"""Crash-proof result storage.

Two stores, one purpose each:
  * SQLite (WAL, synchronous=FULL)  - the durable checkpoint ledger. Every result is committed here
                                      FIRST, so a kill -9 between two issues loses nothing.
  * DuckDB (results/benchmark.duckdb) - the analytical store the report queries.

On open, `reconcile()` replays any ledger rows missing from DuckDB (crash between the two writes,
or DuckDB file deleted), so the two never diverge. Only status='ok' rows count as "completed";
error rows are kept for diagnostics but are retried on the next run.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import duckdb

from .models import RunRecord

log = logging.getLogger("tokenslicer.storage")

COLUMNS: list[str] = [
    "instance_id", "pipeline_name", "status", "error", "model",
    "input_tokens", "cached_input_tokens", "output_tokens",
    "context_tokens_raw", "context_tokens_pruned", "tokens_saved_pct",
    "cost_usd", "cost_inr", "latency_sec",
    "valid_git_diff", "git_apply_error", "gold_file_match", "gold_in_context",
    "usage_estimated", "raw_truncated", "retrieved_files", "raw_patch", "created_at",
]

_DUCKDB_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    instance_id           VARCHAR   NOT NULL,
    pipeline_name         VARCHAR   NOT NULL,
    status                VARCHAR   NOT NULL,
    error                 VARCHAR,
    model                 VARCHAR,
    input_tokens          BIGINT,
    cached_input_tokens   BIGINT,
    output_tokens         BIGINT,
    context_tokens_raw    BIGINT,
    context_tokens_pruned BIGINT,
    tokens_saved_pct      DOUBLE,
    cost_usd              DOUBLE,
    cost_inr              DOUBLE,
    latency_sec           DOUBLE,
    valid_git_diff        BOOLEAN,
    git_apply_error       VARCHAR,
    gold_file_match       BOOLEAN,
    gold_in_context       BOOLEAN,
    usage_estimated       BOOLEAN,
    raw_truncated         BOOLEAN,
    retrieved_files       VARCHAR,
    raw_patch             VARCHAR,
    created_at            TIMESTAMP,
    PRIMARY KEY (instance_id, pipeline_name)
)
"""

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS checkpoint (
    instance_id   TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    status        TEXT NOT NULL,
    payload       TEXT NOT NULL,
    PRIMARY KEY (instance_id, pipeline_name)
)
"""


def _row(record: RunRecord) -> list[object]:
    data = record.model_dump()
    return [data[c] for c in COLUMNS]


class ResultStore:
    """Thread-safe. All methods are synchronous and fast; call via `asyncio.to_thread` from async code."""

    def __init__(self, duckdb_path: Path, sqlite_path: Path | None = None) -> None:
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.duckdb_path = duckdb_path
        self.sqlite_path = sqlite_path or duckdb_path.with_name("checkpoint.sqlite")
        self._lock = threading.RLock()

        self._sqlite = sqlite3.connect(self.sqlite_path, check_same_thread=False, isolation_level=None)
        self._sqlite.execute("PRAGMA journal_mode=WAL")
        self._sqlite.execute("PRAGMA synchronous=FULL")
        self._sqlite.execute(_SQLITE_DDL)

        self._duck = duckdb.connect(str(duckdb_path))
        self._duck.execute(_DUCKDB_DDL)
        self.reconcile()

    # ------------------------------------------------------------------ writes
    def save(self, record: RunRecord) -> None:
        """Durably persist one result. Never raises for a DuckDB failure (ledger already has it)."""
        payload = record.model_dump_json()
        with self._lock:
            self._sqlite.execute(
                "INSERT OR REPLACE INTO checkpoint (instance_id, pipeline_name, status, payload) VALUES (?,?,?,?)",
                (record.instance_id, record.pipeline_name, record.status, payload),
            )
            try:
                self._upsert_duck(record)
            except Exception as exc:  # reconcile() will heal this on the next start
                log.error(
                    "duckdb write failed; checkpoint ledger holds the row",
                    extra={"instance_id": record.instance_id, "pipeline": record.pipeline_name, "error": str(exc)},
                )

    def _upsert_duck(self, record: RunRecord) -> None:
        placeholders = ",".join("?" for _ in COLUMNS)
        self._duck.execute(
            f"INSERT OR REPLACE INTO runs ({','.join(COLUMNS)}) VALUES ({placeholders})", _row(record)
        )

    def reconcile(self) -> int:
        """Replay ledger rows that DuckDB lacks (or holds with a different status). Returns rows healed."""
        with self._lock:
            duck_status = {
                (r[0], r[1]): r[2]
                for r in self._duck.execute("SELECT instance_id, pipeline_name, status FROM runs").fetchall()
            }
            healed = 0
            for iid, pipeline, status, payload in self._sqlite.execute(
                "SELECT instance_id, pipeline_name, status, payload FROM checkpoint"
            ).fetchall():
                if duck_status.get((iid, pipeline)) == status:
                    continue
                try:
                    self._upsert_duck(RunRecord.model_validate_json(payload))
                    healed += 1
                except Exception as exc:
                    log.error("cannot replay checkpoint row", extra={"instance_id": iid, "error": str(exc)})
            if healed:
                log.warning("reconciled checkpoint ledger into DuckDB", extra={"rows": healed})
            return healed

    # ------------------------------------------------------------------ reads
    def completed_pairs(self) -> set[tuple[str, str]]:
        with self._lock:
            rows = self._sqlite.execute(
                "SELECT instance_id, pipeline_name FROM checkpoint WHERE status='ok'"
            ).fetchall()
        return {(r[0], r[1]) for r in rows}

    def get_record(self, instance_id: str, pipeline_name: str) -> RunRecord | None:
        with self._lock:
            row = self._sqlite.execute(
                "SELECT payload FROM checkpoint WHERE instance_id=? AND pipeline_name=?",
                (instance_id, pipeline_name),
            ).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def query(self, sql: str, params: list[object] | None = None) -> list[dict[str, object]]:
        """Run a read-only analytical query against DuckDB; returns rows as dicts."""
        with self._lock:
            cur = self._duck.execute(sql, params or [])
            cols = [d[0] for d in cur.description or []]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            try:
                self._duck.close()
            finally:
                self._sqlite.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
