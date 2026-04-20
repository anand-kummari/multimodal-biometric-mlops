"""SQLite-backed experiment run tracker.

Provides a lightweight, zero-dependency alternative to MLflow for
persisting training run metadata locally.  Every call to ``record_run``
inserts a row with hyperparameters and final metrics into a local
``experiment_runs.db`` file so that past runs can be queried with plain
SQL.

Usage::

    tracker = RunTracker("experiment_runs.db")
    run_id = tracker.start_run(
        experiment="biometric-v1",
        params={"lr": 0.001, "optimizer": "adam", "batch_size": 16},
    )
    tracker.log_epoch(run_id, epoch=0, metrics={"train_loss": 3.81, "val_loss": 3.82})
    tracker.log_epoch(run_id, epoch=1, metrics={"train_loss": 2.89, "val_loss": 4.37})
    tracker.finish_run(run_id, final_metrics={"best_val_loss": 3.82, "best_epoch": 0})

    # Query later
    for row in tracker.query_runs("SELECT * FROM runs ORDER BY created_at DESC LIMIT 5"):
        print(row)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    experiment    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    params_json   TEXT,
    metrics_json  TEXT
);

CREATE TABLE IF NOT EXISTS epoch_metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    epoch      INTEGER NOT NULL,
    metrics_json TEXT,
    logged_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_epoch_run ON epoch_metrics(run_id, epoch);
"""


class RunTracker:
    """Lightweight SQLite experiment tracker.

    Args:
        db_path: Path to the SQLite database file.  Created if it
            does not exist.
    """

    def __init__(self, db_path: str | Path = "experiment_runs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("RunTracker initialised: %s", self.db_path)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment: str = "default",
        params: dict[str, Any] | None = None,
    ) -> str:
        """Create a new run and return its unique ID."""
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO runs (run_id, experiment, created_at, params_json) VALUES (?, ?, ?, ?)",
            (run_id, experiment, now, json.dumps(params or {})),
        )
        self._conn.commit()
        logger.info("Run started: %s (experiment=%s)", run_id, experiment)
        return run_id

    def log_epoch(
        self,
        run_id: str,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Append per-epoch metrics for a run."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO epoch_metrics"
            " (run_id, epoch, metrics_json, logged_at)"
            " VALUES (?, ?, ?, ?)",
            (run_id, epoch, json.dumps(metrics), now),
        )
        self._conn.commit()

    def finish_run(
        self,
        run_id: str,
        final_metrics: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> None:
        """Mark a run as finished and store summary metrics."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, metrics_json = ? WHERE run_id = ?",
            (now, status, json.dumps(final_metrics or {}), run_id),
        )
        self._conn.commit()
        logger.info("Run finished: %s (status=%s)", run_id, status)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_runs(
        self,
        sql: str = "SELECT * FROM runs ORDER BY created_at DESC",
    ) -> list[dict[str, Any]]:
        """Execute a SQL query against the database and return results as dicts."""
        cursor = self._conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by ID."""
        cursor = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_epoch_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all epoch metrics for a run, ordered by epoch."""
        cursor = self._conn.execute(
            "SELECT * FROM epoch_metrics WHERE run_id = ? ORDER BY epoch",
            (run_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_best_run(
        self,
        experiment: str,
        metric_key: str = "best_val_loss",
        mode: str = "min",
    ) -> dict[str, Any] | None:
        """Find the run with the best value for a given metric.

        Args:
            experiment: Experiment name to filter on.
            metric_key: JSON key within ``metrics_json`` to compare.
            mode: ``'min'`` or ``'max'``.

        Returns:
            The best run as a dict, or None if no completed runs exist.
        """
        rows = self.query_runs(
            "SELECT * FROM runs" f" WHERE experiment = '{experiment}'" " AND status = 'completed'"
        )
        if not rows:
            return None

        def _extract(row: dict[str, Any]) -> float | None:
            try:
                metrics = json.loads(row.get("metrics_json", "{}"))
                val = metrics.get(metric_key)
                return float(val) if val is not None else None
            except (json.JSONDecodeError, TypeError, ValueError):
                return None

        valid: list[tuple[dict[str, Any], float]] = [
            (r, v) for r in rows if (v := _extract(r)) is not None
        ]
        if not valid:
            return None

        if mode == "min":
            return min(valid, key=lambda x: x[1])[0]
        return max(valid, key=lambda x: x[1])[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
