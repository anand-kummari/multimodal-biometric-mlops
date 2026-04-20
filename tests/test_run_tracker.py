"""Tests for the SQLite run tracker."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from biometric.training.run_tracker import RunTracker


@pytest.fixture
def tracker(tmp_path: Path) -> RunTracker:  # type: ignore[misc]
    db = tmp_path / "test_runs.db"
    t = RunTracker(db_path=db)
    yield t
    t.close()


class TestRunTracker:
    """Tests for RunTracker lifecycle, querying, and best-run selection."""

    def test_start_and_get_run(self, tracker: RunTracker) -> None:
        run_id = tracker.start_run(experiment="test-exp", params={"lr": 0.001})
        run = tracker.get_run(run_id)
        assert run is not None
        assert run["experiment"] == "test-exp"
        assert run["status"] == "running"
        params = json.loads(run["params_json"])
        assert params["lr"] == 0.001

    def test_log_epoch(self, tracker: RunTracker) -> None:
        run_id = tracker.start_run(experiment="test-exp")
        tracker.log_epoch(run_id, epoch=0, metrics={"train_loss": 3.81, "val_loss": 3.82})
        tracker.log_epoch(run_id, epoch=1, metrics={"train_loss": 2.50, "val_loss": 3.10})
        rows = tracker.get_epoch_metrics(run_id)
        assert len(rows) == 2
        assert json.loads(rows[0]["metrics_json"])["train_loss"] == 3.81
        assert json.loads(rows[1]["metrics_json"])["val_loss"] == 3.10

    def test_finish_run(self, tracker: RunTracker) -> None:
        run_id = tracker.start_run(experiment="test-exp")
        tracker.finish_run(run_id, final_metrics={"best_val_loss": 3.10}, status="completed")
        run = tracker.get_run(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert run["finished_at"] is not None
        metrics = json.loads(run["metrics_json"])
        assert metrics["best_val_loss"] == 3.10

    def test_finish_run_failed(self, tracker: RunTracker) -> None:
        run_id = tracker.start_run(experiment="test-exp")
        tracker.finish_run(run_id, status="failed")
        run = tracker.get_run(run_id)
        assert run is not None
        assert run["status"] == "failed"

    def test_query_runs(self, tracker: RunTracker) -> None:
        tracker.start_run(experiment="exp-a")
        tracker.start_run(experiment="exp-b")
        runs = tracker.query_runs()
        assert len(runs) == 2

    def test_get_best_run(self, tracker: RunTracker) -> None:
        r1 = tracker.start_run(experiment="exp")
        tracker.finish_run(r1, final_metrics={"best_val_loss": 3.50})
        r2 = tracker.start_run(experiment="exp")
        tracker.finish_run(r2, final_metrics={"best_val_loss": 2.10})
        r3 = tracker.start_run(experiment="exp")
        tracker.finish_run(r3, final_metrics={"best_val_loss": 4.00})
        best = tracker.get_best_run("exp", metric_key="best_val_loss", mode="min")
        assert best is not None
        assert best["run_id"] == r2

    def test_get_best_run_max(self, tracker: RunTracker) -> None:
        r1 = tracker.start_run(experiment="exp")
        tracker.finish_run(r1, final_metrics={"best_val_acc": 0.85})
        r2 = tracker.start_run(experiment="exp")
        tracker.finish_run(r2, final_metrics={"best_val_acc": 0.92})
        best = tracker.get_best_run("exp", metric_key="best_val_acc", mode="max")
        assert best is not None
        assert best["run_id"] == r2

    def test_get_nonexistent_run(self, tracker: RunTracker) -> None:
        assert tracker.get_run("nonexistent") is None

    def test_get_best_run_no_runs(self, tracker: RunTracker) -> None:
        assert tracker.get_best_run("nope") is None

    def test_schema_tables_exist(self, tracker: RunTracker) -> None:
        conn = sqlite3.connect(str(tracker.db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "runs" in tables
        assert "epoch_metrics" in tables
