"""Tests for the dataset validation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from biometric.data.validation import ValidationReport, validate_dataset


class TestValidateDataset:
    """Verify the validation scanner against synthetic directory layouts."""

    def test_healthy_dataset(self, tmp_data_dir: Path) -> None:
        report = validate_dataset(tmp_data_dir)
        assert report.total_subjects == 3
        assert report.total_images > 0
        assert report.corrupt_images == []
        assert report.is_healthy

    def test_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        report = validate_dataset(tmp_path)
        assert report.total_subjects == 0
        assert report.total_images == 0

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        report = validate_dataset(tmp_path / "no_such_dir")
        assert report.total_subjects == 0

    def test_missing_modality_detected(self, tmp_path: Path) -> None:
        subj = tmp_path / "subject_000" / "iris_left"
        subj.mkdir(parents=True)
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img.save(subj / "img.png")

        report = validate_dataset(tmp_path)
        assert "subject_000" in report.missing_modalities
        assert "fingerprint" in report.missing_modalities["subject_000"]

    def test_corrupt_image_detected(self, tmp_path: Path) -> None:
        subj = tmp_path / "subject_000"
        for mod in ("iris_left", "iris_right", "fingerprint"):
            d = subj / mod
            d.mkdir(parents=True)
            (d / "bad.png").write_bytes(b"not-a-real-image")

        report = validate_dataset(tmp_path)
        assert len(report.corrupt_images) > 0
        assert not report.is_healthy


class TestValidationReport:
    """Unit tests for the report dataclass itself."""

    def test_healthy_when_no_issues(self) -> None:
        report = ValidationReport(total_subjects=3, total_images=30)
        assert report.is_healthy

    def test_not_healthy_with_corrupt(self) -> None:
        report = ValidationReport(corrupt_images=["bad.png"])
        assert not report.is_healthy

    def test_summary_output(self) -> None:
        report = ValidationReport(total_subjects=5, total_images=100)
        text = report.summary()
        assert "Subjects:" in text
        assert "100" in text
