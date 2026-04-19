"""Shared test fixtures for the biometric test suite.

Provides synthetic data generators and common test configurations
that avoid dependency on the real Kaggle dataset during CI/CD.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def device() -> torch.device:
    """Provide the best available test device (CPU for CI stability)."""
    return torch.device("cpu")


@pytest.fixture
def seed() -> int:
    """Standard test seed for reproducibility."""
    return 42


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor | bool]:
    """Create a synthetic batch mimicking the dataset output.

    Returns a batch of 4 samples with all three modalities.
    """
    batch_size = 4
    return {
        "iris_left": torch.randn(batch_size, 3, 224, 224),
        "iris_right": torch.randn(batch_size, 3, 224, 224),
        "fingerprint": torch.randn(batch_size, 1, 224, 224),
        "label": torch.randint(0, 45, (batch_size,)),
        "has_iris_left": True,
        "has_iris_right": True,
        "has_fingerprint": True,
    }


@pytest.fixture
def tmp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with synthetic biometric data.

    Generates a small dataset with 3 subjects, each having iris and
    fingerprint images, mimicking the real dataset structure.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    num_subjects = 3
    images_per_modality = 2

    for subj_idx in range(num_subjects):
        subject_dir = tmp_dir / f"subject_{subj_idx:03d}"

        # Iris left
        iris_left_dir = subject_dir / "iris_left"
        iris_left_dir.mkdir(parents=True)
        for img_idx in range(images_per_modality):
            img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
            img.save(iris_left_dir / f"img_{img_idx:03d}.png")

        # Iris right
        iris_right_dir = subject_dir / "iris_right"
        iris_right_dir.mkdir(parents=True)
        for img_idx in range(images_per_modality):
            img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
            img.save(iris_right_dir / f"img_{img_idx:03d}.png")

        # Fingerprint
        fp_dir = subject_dir / "fingerprint"
        fp_dir.mkdir(parents=True)
        for img_idx in range(images_per_modality):
            img = Image.fromarray(np.random.randint(0, 255, (128, 128), dtype=np.uint8))
            img.save(fp_dir / f"finger_{img_idx:03d}.png")

    yield tmp_dir

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def tmp_cache_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for cache files."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def tmp_checkpoint_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for model checkpoints."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
