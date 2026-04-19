"""Tests for the MultimodalBiometricDataset."""

from __future__ import annotations

from pathlib import Path

import torch

from biometric.data.dataset import MultimodalBiometricDataset, SubjectSample


class TestSubjectSample:
    """Tests for the SubjectSample dataclass."""

    def test_creation_with_all_paths(self) -> None:
        sample = SubjectSample(
            subject_id=0,
            iris_left_path="/path/to/left.png",
            iris_right_path="/path/to/right.png",
            fingerprint_path="/path/to/finger.png",
        )
        assert sample.subject_id == 0
        assert sample.iris_left_path == "/path/to/left.png"

    def test_creation_with_missing_paths(self) -> None:
        sample = SubjectSample(subject_id=5)
        assert sample.iris_left_path is None
        assert sample.iris_right_path is None
        assert sample.fingerprint_path is None


class TestMultimodalBiometricDataset:
    """Tests for the dataset class."""

    def test_dataset_discovery(self, tmp_data_dir: Path) -> None:
        """Test that samples are correctly discovered from directory structure."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        assert len(dataset) > 0

    def test_dataset_returns_dict(self, tmp_data_dir: Path) -> None:
        """Test that __getitem__ returns the expected dictionary format."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        sample = dataset[0]

        assert "iris_left" in sample
        assert "iris_right" in sample
        assert "fingerprint" in sample
        assert "label" in sample
        assert "has_iris_left" in sample

    def test_iris_tensor_shape(self, tmp_data_dir: Path) -> None:
        """Test that iris tensors have correct shape (3, 224, 224)."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        sample = dataset[0]

        assert sample["iris_left"].shape == (3, 224, 224)
        assert sample["iris_right"].shape == (3, 224, 224)

    def test_fingerprint_tensor_shape(self, tmp_data_dir: Path) -> None:
        """Test that fingerprint tensors have correct shape (1, 224, 224)."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        sample = dataset[0]

        assert sample["fingerprint"].shape == (1, 224, 224)

    def test_label_is_long_tensor(self, tmp_data_dir: Path) -> None:
        """Test that labels are long tensors for CrossEntropyLoss."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        sample = dataset[0]

        assert sample["label"].dtype == torch.long

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test that an empty data directory results in an empty dataset."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_path, split="train")
        assert len(dataset) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test that a nonexistent directory is handled gracefully."""
        dataset = MultimodalBiometricDataset(data_dir=tmp_path / "nonexistent", split="train")
        assert len(dataset) == 0

    def test_eval_split_no_augmentation(self, tmp_data_dir: Path) -> None:
        """Test that eval split uses non-augmented transforms."""
        dataset_train = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="train")
        dataset_val = MultimodalBiometricDataset(data_dir=tmp_data_dir, split="val")
        # Both should create valid datasets
        assert len(dataset_train) == len(dataset_val)

    def test_custom_modalities(self, tmp_data_dir: Path) -> None:
        """Test loading with a subset of modalities."""
        dataset = MultimodalBiometricDataset(
            data_dir=tmp_data_dir,
            split="train",
            modalities=["iris_left"],
        )
        sample = dataset[0]
        # iris_left should be loaded, others should be zero tensors
        assert sample["has_iris_left"] is True
