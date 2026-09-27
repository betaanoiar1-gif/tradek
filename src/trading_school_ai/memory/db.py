"""SQLite experience memory (local filesystem only, WAL mode, append-oriented)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    counter       INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    split         TEXT NOT NULL,
    genome_hash   TEXT NOT NULL,
    genome_id     TEXT NOT NULL,
    feature_hash  TEXT NOT NULL,
    dataset_hash  TEXT,
    dataset_source TEXT,
    config_hash   TEXT NOT NULL,
    code_version  TEXT NOT NULL,
    git_commit    TEXT,
    fitness       REAL NOT NULL,
    fitness_accepted INTEGER NOT NULL,
    rejection_reason TEXT,
    formula_version  TEXT,
    status        TEXT NOT NULL,
    ledger_path   TEXT,
    genome_json   TEXT NOT NULL,
    metrics_json  TEXT NOT NULL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_genome ON experiments(genome_hash);
CREATE INDEX IF NOT EXISTS idx_exp_fitness ON experiments(fitness DESC);
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    concept       TEXT,
    hypothesis    TEXT,
    expected_effect TEXT,
    confidence    REAL,
    payload_json  TEXT NOT NULL,
    status        TEXT NOT NULL,
    experiment_id TEXT
);
CREATE TABLE IF NOT EXISTS robustness (
    experiment_id TEXT NOT NULL,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    PRIMARY KEY (experiment_id, kind)
);
"""


class Memory:
    """Append-oriented experiment store. Never place this file on Google Drive."""

    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        if "/drive/" in str(path).replace("\\", "/").lower():
            raise ValueError(
                "Refusing to open the runtime SQLite database on a Drive mount: "
                f"{path}. Use a local runtime directory and sync backups instead."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(_DDL)
        self._migrate()
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_info(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _migrate(self) -> None:
        """Additive migrations only; existing rows are never rewritten."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(experiments)")}
        if "dataset_source" not in cols:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN dataset_source TEXT")
        self.conn.commit()

    # ---------------------------------------------------------------- writes
    def record_experiment(self, result: Any) -> None:
        row = result.to_row() if hasattr(result, "to_row") else dict(result)
        self.conn.execute(
            """INSERT OR REPLACE INTO experiments VALUES
               (:experiment_id,:counter,:seed,:split,:genome_hash,:genome_id,
                :feature_hash,:dataset_hash,:dataset_source,:config_hash,:code_version,:git_commit,
                :fitness,:fitness_accepted,:rejection_reason,:formula_version,:status,
                :ledger_path,:genome_json,:metrics_json,:error)""",
            {
                **{k: row.get(k) for k in (
                    "experiment_id","counter","seed","split","genome_hash","genome_id",
                    "feature_hash","dataset_hash","dataset_source","config_hash","code_version",
                    "git_commit","fitness","rejection_reason","formula_version",
                    "status","ledger_path","genome_json","error")},
                "fitness_accepted": int(bool(row.get("fitness_accepted"))),
                "metrics_json": json.dumps(row.get("metrics", {}), sort_keys=True),
            },
        )
        self.conn.commit()

    def record_hypothesis(self, hypothesis_id: str, source: str, payload: dict,
                          status: str, experiment_id: Optional[str] = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO hypotheses VALUES
               (?,?,?,?,?,?,?,?,?)""",
            (hypothesis_id, source, payload.get("concept"), payload.get("hypothesis"),
             payload.get("expected_effect"), payload.get("confidence"),
             json.dumps(payload, sort_keys=True), status, experiment_id),
        )
        self.conn.commit()

    def record_robustness(self, experiment_id: str, kind: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO robustness VALUES (?,?,?)",
            (experiment_id, kind, json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()

    # ----------------------------------------------------------------- reads
    def next_counter(self) -> int:
        cur = self.conn.execute("SELECT COALESCE(MAX(counter), -1) + 1 FROM experiments")
        return int(cur.fetchone()[0])

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def top_experiments(self, limit: int = 10, split: str = "train") -> list[dict]:
        cur = self.conn.execute(
            """SELECT * FROM experiments WHERE split=? AND fitness_accepted=1
               ORDER BY fitness DESC, experiment_id ASC LIMIT ?""",
            (split, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    def failures(self, limit: int = 20) -> list[dict]:
        cur = self.conn.execute(
            """SELECT rejection_reason, COUNT(*) AS n FROM experiments
               WHERE fitness_accepted=0 GROUP BY rejection_reason
               ORDER BY n DESC LIMIT ?""", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def summary(self) -> dict:
        return {
            "db_path": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "experiments": self.count(),
            "accepted": int(self.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE fitness_accepted=1").fetchone()[0]),
            "hypotheses": int(self.conn.execute(
                "SELECT COUNT(*) FROM hypotheses").fetchone()[0]),
            "failure_modes": self.failures(),
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
