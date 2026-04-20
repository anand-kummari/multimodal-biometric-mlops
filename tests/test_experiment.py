"""Tests for MLflow experiment tracking integration."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biometric.training.experiment import (
    end_run,
    init_experiment,
    log_artifact,
    log_metrics,
    log_params,
)


class TestMLflowIntegration:
    """Test MLflow experiment tracking."""

    @pytest.fixture(autouse=True)
    def mock_mlflow(self) -> Generator[MagicMock, None, None]:
        """Mock MLflow to avoid actual tracking during tests."""
        mock = MagicMock()
        mock.active_run.return_value = MagicMock(info=MagicMock(run_id="test_run_123"))

        with (
            patch("biometric.training.experiment._mlflow", mock),
            patch("biometric.training.experiment._ensure_mlflow", return_value=True),
        ):
            yield mock

    def test_init_experiment_creates_run(self, mock_mlflow: MagicMock) -> None:
        """Test that init_experiment starts an MLflow run."""
        init_experiment(experiment_name="test_exp", run_name="test_run")

        mock_mlflow.set_experiment.assert_called_once_with("test_exp")
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")

    def test_log_params_single(self, mock_mlflow: MagicMock) -> None:
        """Test logging a single parameter."""
        init_experiment("test_exp")
        log_params({"learning_rate": 0.001})

        mock_mlflow.log_params.assert_called_once_with({"learning_rate": 0.001})

    def test_log_params_multiple(self, mock_mlflow: MagicMock) -> None:
        """Test logging multiple parameters."""
        init_experiment("test_exp")
        params = {"lr": 0.001, "batch_size": 32, "epochs": 10}
        log_params(params)

        mock_mlflow.log_params.assert_called_once_with(params)

    def test_log_metrics_single_step(self, mock_mlflow: MagicMock) -> None:
        """Test logging metrics at a single step."""
        init_experiment("test_exp")
        log_metrics({"train_loss": 0.5}, step=0)

        mock_mlflow.log_metrics.assert_called_once_with({"train_loss": 0.5}, step=0)

    def test_log_metrics_multiple_steps(self, mock_mlflow: MagicMock) -> None:
        """Test logging metrics across multiple steps."""
        init_experiment("test_exp")

        for step in range(3):
            log_metrics({"val_acc": 0.8 + step * 0.05}, step=step)

        assert mock_mlflow.log_metrics.call_count == 3

    def test_log_artifact_file(self, mock_mlflow: MagicMock) -> None:
        """Test logging a file artifact."""
        init_experiment("test_exp")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            log_artifact(str(temp_path))
            mock_mlflow.log_artifact.assert_called_once_with(str(temp_path))
        finally:
            temp_path.unlink()

    def test_end_run_closes_active_run(self, mock_mlflow: MagicMock) -> None:
        """Test that end_run closes the active MLflow run."""
        init_experiment("test_exp")
        end_run()

        mock_mlflow.end_run.assert_called_once()

    def test_log_without_active_run_no_error(self, mock_mlflow: MagicMock) -> None:
        """Test that logging without an active run doesn't raise errors."""
        mock_mlflow.active_run.return_value = None

        log_params({"test": 1})
        log_metrics({"test": 0.5}, step=0)

    def test_init_experiment_with_tags(self, mock_mlflow: MagicMock) -> None:
        """Test initializing experiment with custom tags."""
        init_experiment("test_exp", run_name="test", tags={"env": "test", "version": "1.0"})

        mock_mlflow.start_run.assert_called_once()
        call_kwargs = mock_mlflow.start_run.call_args[1]
        assert call_kwargs["run_name"] == "test"

    def test_multiple_metrics_same_step(self, mock_mlflow: MagicMock) -> None:
        """Test logging multiple metrics at the same step."""
        init_experiment("test_exp")

        log_metrics({"train_loss": 0.5, "train_acc": 0.8, "val_loss": 0.6}, step=0)

        mock_mlflow.log_metrics.assert_called_once_with(
            {"train_loss": 0.5, "train_acc": 0.8, "val_loss": 0.6}, step=0
        )


class TestMLflowGracefulDegradation:
    """Test that code works when MLflow is not installed."""

    @patch("biometric.training.experiment._ensure_mlflow", return_value=False)
    def test_operations_without_mlflow(self, mock_ensure: MagicMock) -> None:
        """Test that all operations work when MLflow is unavailable."""
        init_experiment("test")
        log_params({"test": 1})
        log_metrics({"test": 0.5}, step=0)
        log_artifact("fake_path.txt")
        end_run()
