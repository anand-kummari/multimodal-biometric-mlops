"""Pre-training dataset validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@dataclass
class ValidationReport:
    """Aggregated result of a dataset health check.

    Attributes:
        total_subjects: Number of subject directories found.
        total_images: Total number of image files discovered.
        corrupt_images: Paths to images that could not be opened.
        missing_modalities: Per-subject list of missing modality folders.
        class_counts: Number of samples per subject index.
    """

    total_subjects: int = 0
    total_images: int = 0
    corrupt_images: list[str] = field(default_factory=list)
    missing_modalities: dict[str, list[str]] = field(default_factory=dict)
    class_counts: dict[int, int] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """True when no corrupt images and no subjects are missing all modalities."""
        return len(self.corrupt_images) == 0 and len(self.missing_modalities) == 0

    def summary(self) -> str:
        """Return a human-readable summary suitable for logging."""
        lines = [
            f"Subjects:           {self.total_subjects}",
            f"Total images:       {self.total_images}",
            f"Corrupt images:     {len(self.corrupt_images)}",
            f"Subjects w/ gaps:   {len(self.missing_modalities)}",
        ]
        if self.class_counts:
            counts = sorted(self.class_counts.values())
            lines.append(f"Samples/class:      min={counts[0]}, max={counts[-1]}")
        return "\n".join(lines)


def validate_dataset(
    data_dir: str | Path,
    expected_modalities: list[str] | None = None,
) -> ValidationReport:
    """Walk the dataset tree and check for common problems.

    Verifies that every subject directory contains the expected modality
    sub-folders, that each image file can be opened by PIL, and collects
    per-class sample counts for imbalance detection.

    Args:
        data_dir: Root directory laid out as ``subject_XXX/<modality>/<image>``.
        expected_modalities: Modality folder names to look for in each
            subject directory.  Defaults to ``["iris_left", "iris_right",
            "fingerprint"]``.

    Returns:
        A :class:`ValidationReport` summarising the findings.
    """
    data_dir = Path(data_dir)
    modalities = expected_modalities or ["iris_left", "iris_right", "fingerprint"]
    report = ValidationReport()

    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return report

    subject_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    report.total_subjects = len(subject_dirs)

    for subject_idx, subject_dir in enumerate(subject_dirs):
        sample_count = 0

        # Check that every expected modality folder is present.
        missing: list[str] = []
        for modality in modalities:
            modality_dir = subject_dir / modality
            if not modality_dir.exists() or not any(modality_dir.iterdir()):
                missing.append(modality)
                continue

            images = [f for f in modality_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS]
            sample_count += len(images)

            for img_path in images:
                report.total_images += 1
                if not _can_open_image(img_path):
                    report.corrupt_images.append(str(img_path))

        if missing:
            report.missing_modalities[subject_dir.name] = missing

        report.class_counts[subject_idx] = sample_count

    logger.info("Dataset validation complete:\n%s", report.summary())
    return report


def _can_open_image(path: Path) -> bool:
    """Try to open and verify an image file with PIL."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
