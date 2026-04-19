"""High-performance DataLoader creation with configurable optimizations.

Centralizes DataLoader construction with production-ready defaults:
- Pinned memory for faster host-to-device transfers
- Persistent workers to avoid re-spawning overhead
- Configurable prefetch factor for overlapping I/O and compute
- Reproducible sampling via seeded generators
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility.

    PyTorch DataLoader workers inherit the parent process's RNG state,
    which can lead to correlated augmentation patterns. This function
    ensures each worker has a unique, deterministic seed.
    """
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    dataset: Dataset[Any],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, DataLoader[Any]]:
    """Create train/val/test DataLoaders with reproducible splits.

    Splits the dataset into train/val/test subsets and wraps each in a
    DataLoader with optimized settings for throughput.

    Args:
        dataset: The full dataset to split.
        batch_size: Batch size for all loaders.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Use pinned memory for faster GPU transfers.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        drop_last: Drop the last incomplete batch.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for the split.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader instances.
    """
    # Compute split sizes
    assert hasattr(dataset, "__len__"), "Dataset must implement __len__"
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    logger.info(
        "Splitting dataset: total=%d, train=%d, val=%d, test=%d",
        total,
        train_size,
        val_size,
        test_size,
    )

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Common DataLoader kwargs
    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
    }

    # Persistent workers require num_workers > 0
    if num_workers > 0:
        common_kwargs["persistent_workers"] = persistent_workers
        common_kwargs["prefetch_factor"] = prefetch_factor

    # Separate generator per loader for reproducibility
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    logger.info(
        "DataLoaders created: batch_size=%d, num_workers=%d, pin_memory=%s",
        batch_size,
        num_workers,
        pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
