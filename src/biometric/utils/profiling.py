"""Performance profiling utilities for data loading and training.

Provides lightweight instrumentation to measure throughput, latency,
and identify bottlenecks in the data pipeline.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Stores timing measurements for a profiled operation.

    Attributes:
        name: Identifier for the profiled operation.
        elapsed_seconds: Total wall-clock time in seconds.
        iterations: Number of iterations (e.g., batches) processed.
        samples_processed: Total number of samples processed.
    """

    name: str
    elapsed_seconds: float
    iterations: int = 0
    samples_processed: int = 0

    @property
    def throughput(self) -> float:
        """Samples per second."""
        if self.elapsed_seconds == 0:
            return float("inf")
        return self.samples_processed / self.elapsed_seconds

    @property
    def avg_batch_time(self) -> float:
        """Average time per iteration in milliseconds."""
        if self.iterations == 0:
            return 0.0
        return (self.elapsed_seconds / self.iterations) * 1000

    def summary(self) -> str:
        """Return a human-readable summary of the timing result."""
        lines = [
            f"--- {self.name} ---",
            f"  Total time:       {self.elapsed_seconds:.3f}s",
            f"  Iterations:       {self.iterations}",
            f"  Samples:          {self.samples_processed}",
            f"  Throughput:       {self.throughput:.1f} samples/sec",
            f"  Avg batch time:   {self.avg_batch_time:.2f}ms",
        ]
        return "\n".join(lines)


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("data_loading") as t:
            for batch in dataloader:
                process(batch)
        print(t.result.summary())
    """

    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self._start: float = 0.0
        self._end: float = 0.0
        self.result: TimingResult = TimingResult(name=name, elapsed_seconds=0.0)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()
        self.result.elapsed_seconds = self._end - self._start

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds (available after context exit)."""
        return self.result.elapsed_seconds


@dataclass
class DataLoaderProfile:
    """Aggregated profiling results for a DataLoader.

    Attributes:
        config_name: Label describing the DataLoader configuration.
        results: Individual timing results per epoch.
    """

    config_name: str
    results: list[TimingResult] = field(default_factory=list)

    @property
    def avg_throughput(self) -> float:
        """Average throughput across all profiled epochs."""
        if not self.results:
            return 0.0
        return sum(r.throughput for r in self.results) / len(self.results)

    @property
    def avg_batch_time_ms(self) -> float:
        """Average batch time in milliseconds across all epochs."""
        if not self.results:
            return 0.0
        return sum(r.avg_batch_time for r in self.results) / len(self.results)


def profile_dataloader(
    dataloader: DataLoader[Any],
    num_epochs: int = 3,
    name: str = "dataloader",
) -> DataLoaderProfile:
    """Profile a DataLoader over multiple epochs to measure loading performance.

    Iterates through the entire DataLoader for the specified number of epochs,
    collecting timing metrics for each epoch. The first epoch is included
    (cold-start measurement) — compare with subsequent epochs for warm-cache behavior.

    Args:
        dataloader: The PyTorch DataLoader to profile.
        num_epochs: Number of full passes over the DataLoader.
        name: Label for the profiling configuration.

    Returns:
        DataLoaderProfile with per-epoch timing results.
    """
    profile = DataLoaderProfile(config_name=name)
    batch_size = dataloader.batch_size or 1

    for epoch in range(num_epochs):
        iterations = 0
        samples = 0

        with Timer(f"{name}_epoch_{epoch}") as timer:
            for batch in dataloader:
                iterations += 1
                # Handle both tuple/list batches and dict batches
                if isinstance(batch, (tuple, list)):
                    current_batch_size = _infer_batch_size(batch[0])
                elif isinstance(batch, dict):
                    first_val = next(iter(batch.values()))
                    current_batch_size = _infer_batch_size(first_val)
                else:
                    current_batch_size = batch_size
                samples += current_batch_size

        timer.result.iterations = iterations
        timer.result.samples_processed = samples
        profile.results.append(timer.result)

        logger.info(
            "Epoch %d/%d: %s", epoch + 1, num_epochs, timer.result.summary()
        )

    return profile


def _infer_batch_size(tensor_or_value: Any) -> int:
    """Infer batch size from the first dimension of a tensor or nested structure."""
    if isinstance(tensor_or_value, torch.Tensor):
        return tensor_or_value.shape[0]
    if isinstance(tensor_or_value, (list, tuple)):
        return len(tensor_or_value)
    return 1
