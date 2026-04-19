"""High-performance DataLoader creation with configurable optimizations."""

from __future__ import annotations

import logging
import random as _random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

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


def split_subjects(
    data_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split subject directories into train/val/test at the **subject level**.

    This prevents data leakage: no subject's images appear in more than
    one split, so the model is evaluated on genuinely unseen identities.

    Args:
        data_dir: Root directory containing subject sub-directories.
        train_ratio: Fraction of subjects for training.
        val_ratio: Fraction of subjects for validation.
        seed: Random seed for reproducible splits.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'``, each mapping to
        a sorted list of subject directory **names** (not full paths).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"train": [], "val": [], "test": []}

    subject_names = sorted(d.name for d in data_path.iterdir() if d.is_dir())
    total = len(subject_names)

    # Deterministic shuffle
    rng = _random.Random(seed)
    shuffled = list(subject_names)
    rng.shuffle(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }

    logger.info(
        "Subject-level split: total=%d, train=%d, val=%d, test=%d",
        total,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


def create_dataloaders(
    datasets: Mapping[str, Dataset[Any]],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    seed: int = 42,
) -> dict[str, DataLoader[Any]]:
    """Create train/val/test DataLoaders from pre-split datasets.

    Each dataset must already contain the correct split of data with
    appropriate transforms (train gets augmentation, val/test do not).
    This function only wraps them in DataLoaders with optimised settings.

    Args:
        datasets: Dict with 'train', 'val', 'test' Dataset instances.
        batch_size: Batch size for all loaders.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Use pinned memory for faster GPU transfers.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        drop_last: Drop the last incomplete batch.
        seed: Random seed for the train sampler.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader instances.
    """
    for name in ("train", "val", "test"):
        ds = datasets[name]
        if not hasattr(ds, "__len__"):
            raise TypeError(f"Dataset '{name}' must implement __len__")

    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    logger.info(
        "Creating DataLoaders: train=%d, val=%d, test=%d samples",
        len(train_ds),  # type: ignore[arg-type]
        len(val_ds),  # type: ignore[arg-type]
        len(test_ds),  # type: ignore[arg-type]
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
        train_ds,
        shuffle=True,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
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
