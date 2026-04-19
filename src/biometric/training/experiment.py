"""MLflow integration for experiment tracking."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_mlflow: Any = None
_MISSING = object()  # sentinel for "tried import and failed"


def _ensure_mlflow() -> bool:
    """Attempt to import mlflow once and cache the result."""
    global _mlflow
    if _mlflow is _MISSING:
        return False
    if _mlflow is not None:
        return True
    try:
        import mlflow

        _mlflow = mlflow
        return True
    except ImportError:
        _mlflow = _MISSING
        return False


def init_experiment(
    experiment_name: str = "multimodal-biometric",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Start an MLflow run, creating the experiment if needed.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: Optional tracking server URI.  When *None*,
            MLflow falls back to the ``MLFLOW_TRACKING_URI``
            environment variable or local ``./mlruns``.
        run_name: Human-readable name for this run.
        tags: Extra key-value tags attached to the run.
    """
    if not _ensure_mlflow():
        logger.debug("mlflow not installed — experiment tracking disabled")
        return

    if tracking_uri:
        _mlflow.set_tracking_uri(tracking_uri)

    _mlflow.set_experiment(experiment_name)
    _mlflow.start_run(run_name=run_name)
    if tags:
        _mlflow.set_tags(tags)
    logger.info("MLflow run started: experiment=%s", experiment_name)


def log_params(params: dict[str, Any]) -> None:
    """Log a batch of hyperparameters to the active run."""
    if not _ensure_mlflow():
        return
    _mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a batch of numeric metrics (typically called once per epoch)."""
    if not _ensure_mlflow():
        return
    _mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str) -> None:
    """Upload a local file as an artifact to the active run."""
    if not _ensure_mlflow():
        return
    _mlflow.log_artifact(local_path)


def end_run() -> None:
    """Finalise the active MLflow run."""
    if not _ensure_mlflow():
        return
    _mlflow.end_run()
    logger.info("MLflow run ended")
