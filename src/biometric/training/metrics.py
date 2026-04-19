"""Metric tracking for training and evaluation."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Aggregated metrics for a single epoch.

    Attributes:
        epoch: Epoch number (0-indexed).
        metrics: Dictionary of metric name -> computed value.
    """

    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EpochMetrics(epoch={self.epoch}, {metrics_str})"


class MetricTracker:
    """Tracks and aggregates metrics across batches and epochs.

    Accumulates per-batch values during an epoch and computes averages
    (or other aggregations) at epoch end. Maintains a full history for
    plotting and analysis.

    Usage::

        tracker = MetricTracker()
        for batch in dataloader:
            loss = train_step(batch)
            tracker.update("train_loss", loss.item())
            tracker.update("train_acc", accuracy)

        epoch_metrics = tracker.compute_epoch(epoch=0)
        tracker.reset()  # Reset for next epoch
    """

    def __init__(self) -> None:
        self._batch_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._epoch_history: list[EpochMetrics] = []

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Record a metric value for the current batch.

        Args:
            name: Metric name (e.g., 'train_loss', 'val_accuracy').
            value: Metric value for this batch.
            count: Number of samples this value represents (for weighted avg).
        """
        self._batch_values[name].append((value, count))

    def compute_epoch(self, epoch: int) -> EpochMetrics:
        """Compute epoch-level metrics from accumulated batch values.

        Args:
            epoch: Current epoch number.

        Returns:
            EpochMetrics with averaged values for each tracked metric.
        """
        computed: dict[str, float] = {}
        for name, entries in self._batch_values.items():
            if entries:
                total_weighted = sum(v * c for v, c in entries)
                total_count = sum(c for _, c in entries)
                computed[name] = total_weighted / max(total_count, 1)

        epoch_metrics = EpochMetrics(epoch=epoch, metrics=computed)
        self._epoch_history.append(epoch_metrics)

        logger.info("Epoch %d metrics: %s", epoch, epoch_metrics)
        return epoch_metrics

    def reset(self) -> None:
        """Reset batch accumulators for a new epoch."""
        self._batch_values.clear()

    @property
    def history(self) -> list[EpochMetrics]:
        """Full epoch history."""
        return self._epoch_history

    def get_best(self, metric: str, mode: str = "min") -> EpochMetrics | None:
        """Find the epoch with the best value for a given metric.

        Args:
            metric: Metric name to optimize.
            mode: 'min' for loss-like metrics, 'max' for accuracy-like.

        Returns:
            EpochMetrics for the best epoch, or None if no history.
        """
        if not self._epoch_history:
            return None

        relevant = [e for e in self._epoch_history if metric in e.metrics]
        if not relevant:
            return None

        if mode == "min":
            return min(relevant, key=lambda e: e.metrics[metric])
        return max(relevant, key=lambda e: e.metrics[metric])

    def to_dict(self) -> list[dict[str, Any]]:
        """Export history as a list of dictionaries (for JSON serialization)."""
        return [{"epoch": em.epoch, **em.metrics} for em in self._epoch_history]
