"""Training callbacks for early stopping and model checkpointing."""

from __future__ import annotations

import abc
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingCallback(abc.ABC):
    """Abstract base class for training callbacks."""

    @abc.abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of computed epoch metrics.
            model: The model being trained.
        """

    @property
    def should_stop(self) -> bool:
        """Return True if training should be stopped."""
        return False


class EarlyStopping(TrainingCallback):
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        metric: Metric name to monitor (e.g., 'val_loss').
        mode: 'min' if lower is better, 'max' if higher is better.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self._best_value: float | None = None
        self._counter: int = 0
        self._stop: bool = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        """Check if the metric has improved."""
        if self.metric not in metrics:
            logger.warning("EarlyStopping: metric '%s' not found in epoch metrics", self.metric)
            return

        current = metrics[self.metric]

        if self._best_value is None:
            self._best_value = current
            return

        if self._is_improvement(current):
            self._best_value = current
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs (best=%s: %.4f)",
                self._counter,
                self.patience,
                self.metric,
                self._best_value,
            )

            if self._counter >= self.patience:
                self._stop = True
                logger.info(
                    "EarlyStopping triggered at epoch %d (best %s: %.4f)",
                    epoch,
                    self.metric,
                    self._best_value,
                )

    def _is_improvement(self, current: float) -> bool:
        """Determine if the current value is an improvement over the best."""
        if self._best_value is None:
            return True
        if self.mode == "min":
            return current < (self._best_value - self.min_delta)
        return current > (self._best_value + self.min_delta)

    @property
    def should_stop(self) -> bool:
        """Return True if patience has been exhausted."""
        return self._stop


class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints based on a monitored metric.

    Args:
        checkpoint_dir: Directory to save checkpoint files.
        metric: Metric name to monitor.
        mode: 'min' if lower is better, 'max' if higher is better.
        save_best: Save the best model.
        save_last: Always save the last epoch's model.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        metric: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last

        self._best_value: float | None = None

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        """Save checkpoints based on metric performance."""
        if self.save_last:
            self._save(model, epoch, metrics, "last")

        if self.save_best and self.metric in metrics:
            current = metrics[self.metric]
            if self._is_best(current):
                self._best_value = current
                self._save(model, epoch, metrics, "best")
                logger.info(
                    "New best model saved (epoch %d, %s=%.4f)",
                    epoch,
                    self.metric,
                    current,
                )

    def _is_best(self, current: float) -> bool:
        """Check if the current value is the best so far."""
        if self._best_value is None:
            return True
        if self.mode == "min":
            return current < self._best_value
        return current > self._best_value

    def _save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, float],
        tag: str,
    ) -> None:
        """Save a checkpoint to disk.

        Checkpoint includes model state, epoch, and metrics for full
        reproducibility of the saved state.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved: %s", path)
