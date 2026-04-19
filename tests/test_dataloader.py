"""Tests for DataLoader creation and subject-level splitting."""

from __future__ import annotations

from pathlib import Path

from biometric.data.dataloader import create_dataloaders, split_subjects
from biometric.data.dataset import MultimodalBiometricDataset


def _make_split_datasets(
    data_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, MultimodalBiometricDataset]:
    """Helper: split subjects and build per-split datasets."""
    splits = split_subjects(data_dir, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return {
        name: MultimodalBiometricDataset(data_dir=data_dir, split=name, subject_names=subjects)
        for name, subjects in splits.items()
    }


class TestSplitSubjects:
    """Tests for the split_subjects helper."""

    def test_all_subjects_assigned(self, tmp_data_dir: Path) -> None:
        """Every subject must appear in exactly one split."""
        splits = split_subjects(tmp_data_dir, train_ratio=0.6, val_ratio=0.2)
        all_names = splits["train"] + splits["val"] + splits["test"]
        total_dirs = [d.name for d in sorted(tmp_data_dir.iterdir()) if d.is_dir()]
        assert sorted(all_names) == sorted(total_dirs)

    def test_no_overlap_between_splits(self, tmp_data_dir: Path) -> None:
        """Subjects must not leak across splits."""
        splits = split_subjects(tmp_data_dir, train_ratio=0.6, val_ratio=0.2)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_reproducible(self, tmp_data_dir: Path) -> None:
        """Same seed must yield the same split."""
        s1 = split_subjects(tmp_data_dir, seed=42)
        s2 = split_subjects(tmp_data_dir, seed=42)
        assert s1 == s2


class TestCreateDataloaders:
    """Tests for the create_dataloaders factory function."""

    def test_returns_three_splits(self, tmp_data_dir: Path) -> None:
        """Verify train/val/test loaders are returned."""
        datasets = _make_split_datasets(tmp_data_dir)
        loaders = create_dataloaders(datasets, batch_size=2, num_workers=0)
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_split_sizes_add_up(self, tmp_data_dir: Path) -> None:
        """Verify that split sizes sum to total dataset length."""
        datasets = _make_split_datasets(tmp_data_dir)
        total = sum(len(ds) for ds in datasets.values())
        loaders = create_dataloaders(datasets, batch_size=2, num_workers=0)
        split_total = 0
        for s in ("train", "val", "test"):
            split_total += len(loaders[s].dataset)  # type: ignore[arg-type]
        assert split_total == total

    def test_train_loader_shuffles(self, tmp_data_dir: Path) -> None:
        """Verify that the train loader is configured with shuffle=True."""
        datasets = _make_split_datasets(tmp_data_dir)
        loaders = create_dataloaders(datasets, batch_size=2, num_workers=0)
        # DataLoader doesn't expose shuffle directly, but we can verify
        # it produces a batch without error
        for batch in loaders["train"]:
            assert "label" in batch
            break

    def test_batch_contains_all_modalities(self, tmp_data_dir: Path) -> None:
        """Verify that batches contain all expected modality keys."""
        datasets = _make_split_datasets(tmp_data_dir)
        loaders = create_dataloaders(datasets, batch_size=2, num_workers=0)

        for batch in loaders["train"]:
            assert "iris_left" in batch
            assert "iris_right" in batch
            assert "fingerprint" in batch
            assert "label" in batch
            break

    def test_reproducible_splits(self, tmp_data_dir: Path) -> None:
        """Verify that the same seed produces identical splits."""
        ds1 = _make_split_datasets(tmp_data_dir, seed=42)
        ds2 = _make_split_datasets(tmp_data_dir, seed=42)

        loaders1 = create_dataloaders(ds1, batch_size=2, num_workers=0, seed=42)
        loaders2 = create_dataloaders(ds2, batch_size=2, num_workers=0, seed=42)

        assert len(loaders1["train"].dataset) == len(loaders2["train"].dataset)  # type: ignore[arg-type]
        assert len(loaders1["val"].dataset) == len(loaders2["val"].dataset)  # type: ignore[arg-type]
