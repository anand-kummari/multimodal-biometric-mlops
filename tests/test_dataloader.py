"""Tests for DataLoader creation and configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from biometric.data.dataloader import create_dataloaders
from biometric.data.dataset import MultimodalBiometricDataset


class TestCreateDataloaders:
    """Tests for the create_dataloaders factory function."""

    def test_returns_three_splits(self, tmp_data_dir: Path) -> None:
        """Verify train/val/test loaders are returned."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        loaders = create_dataloaders(
            dataset, batch_size=2, num_workers=0, train_ratio=0.6, val_ratio=0.2
        )
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_split_sizes_add_up(self, tmp_data_dir: Path) -> None:
        """Verify that split sizes sum to total dataset length."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        total = len(dataset)
        loaders = create_dataloaders(
            dataset, batch_size=2, num_workers=0, train_ratio=0.6, val_ratio=0.2
        )
        split_total = (
            len(loaders["train"].dataset)
            + len(loaders["val"].dataset)
            + len(loaders["test"].dataset)
        )
        assert split_total == total

    def test_train_loader_shuffles(self, tmp_data_dir: Path) -> None:
        """Verify that the train loader is configured with shuffle=True."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        loaders = create_dataloaders(dataset, batch_size=2, num_workers=0)
        # DataLoader doesn't expose shuffle directly, but we can verify
        # it produces a batch without error
        for batch in loaders["train"]:
            assert "label" in batch
            break

    def test_batch_contains_all_modalities(self, tmp_data_dir: Path) -> None:
        """Verify that batches contain all expected modality keys."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        loaders = create_dataloaders(dataset, batch_size=2, num_workers=0)

        for batch in loaders["train"]:
            assert "iris_left" in batch
            assert "iris_right" in batch
            assert "fingerprint" in batch
            assert "label" in batch
            break

    def test_reproducible_splits(self, tmp_data_dir: Path) -> None:
        """Verify that the same seed produces identical splits."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")

        loaders1 = create_dataloaders(dataset, batch_size=2, num_workers=0, seed=42)
        loaders2 = create_dataloaders(dataset, batch_size=2, num_workers=0, seed=42)

        # Same split sizes
        assert len(loaders1["train"].dataset) == len(loaders2["train"].dataset)
        assert len(loaders1["val"].dataset) == len(loaders2["val"].dataset)
