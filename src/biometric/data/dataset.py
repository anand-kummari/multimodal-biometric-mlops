"""Multimodal biometric dataset implementation.

Provides a PyTorch Dataset that loads iris and fingerprint images per subject,
returning a dictionary of modality tensors. Supports:
- Multiple modalities (left iris, right iris, fingerprint)
- Per-modality transforms
- Missing modality handling (graceful degradation)
- Integration with the PyArrow caching layer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from biometric.data.registry import DatasetRegistry
from biometric.data.transforms import (
    fingerprint_eval_transform,
    fingerprint_train_transform,
    iris_eval_transform,
    iris_train_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class SubjectSample:
    """A single training sample linking modality image paths to a subject label.

    Attributes:
        subject_id: Integer label for the subject (0-indexed).
        iris_left_path: Path to a left-eye iris image.
        iris_right_path: Path to a right-eye iris image.
        fingerprint_path: Path to a fingerprint image.
    """

    subject_id: int
    iris_left_path: str | None = None
    iris_right_path: str | None = None
    fingerprint_path: str | None = None


@DatasetRegistry.register("multimodal_biometric")
class MultimodalBiometricDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for multimodal biometric data.

    Loads iris (left/right) and fingerprint images for each sample,
    applying modality-specific transforms. Returns a dictionary of
    tensors keyed by modality name, suitable for the fusion model.

    Args:
        data_dir: Root directory containing the organized dataset.
        split: One of 'train', 'val', 'test'.
        iris_size: Target size for iris images as (H, W).
        fingerprint_size: Target size for fingerprint images as (H, W).
        modalities: List of modalities to load.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        iris_size: tuple[int, int] = (224, 224),
        fingerprint_size: tuple[int, int] = (224, 224),
        modalities: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ["iris_left", "iris_right", "fingerprint"]
        self._iris_size = iris_size
        self._fingerprint_size = fingerprint_size

        # Build modality-specific transforms
        is_train = split == "train"
        self.iris_transform = (
            iris_train_transform(iris_size) if is_train else iris_eval_transform(iris_size)
        )
        self.fingerprint_transform = (
            fingerprint_train_transform(fingerprint_size)
            if is_train
            else fingerprint_eval_transform(fingerprint_size)
        )

        # Discover and index all samples
        self.samples: list[SubjectSample] = self._discover_samples()
        logger.info(
            "MultimodalBiometricDataset: split=%s, samples=%d, modalities=%s",
            split,
            len(self.samples),
            self.modalities,
        )

    def _discover_samples(self) -> list[SubjectSample]:
        """Scan the data directory to build a list of samples.

        Each iris image paired with a fingerprint creates one sample.
        This approach maximizes the number of training samples by creating
        combinations of iris and fingerprint images per subject.

        Returns:
            List of SubjectSample dataclass instances.
        """
        samples: list[SubjectSample] = []

        if not self.data_dir.exists():
            logger.warning("Data directory does not exist: %s", self.data_dir)
            return samples

        subject_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        for subject_idx, subject_dir in enumerate(subject_dirs):
            iris_left_dir = subject_dir / "iris_left"
            iris_right_dir = subject_dir / "iris_right"
            fingerprint_dir = subject_dir / "fingerprint"

            iris_left_images: list[Path | None] = list(self._list_images(iris_left_dir))
            iris_right_images: list[Path | None] = list(self._list_images(iris_right_dir))
            fingerprint_images: list[Path | None] = list(self._list_images(fingerprint_dir))

            # Create combinatorial samples: each iris pair + fingerprint
            if not iris_left_images:
                iris_left_images = [None]
            if not iris_right_images:
                iris_right_images = [None]
            if not fingerprint_images:
                fingerprint_images = [None]

            # Pair images: zip iris left+right, combine with fingerprints
            num_iris_pairs = max(len(iris_left_images), len(iris_right_images))
            for i in range(num_iris_pairs):
                left = iris_left_images[i % len(iris_left_images)]
                right = iris_right_images[i % len(iris_right_images)]

                for fp in fingerprint_images:
                    samples.append(
                        SubjectSample(
                            subject_id=subject_idx,
                            iris_left_path=str(left) if left else None,
                            iris_right_path=str(right) if right else None,
                            fingerprint_path=str(fp) if fp else None,
                        )
                    )

        return samples

    @staticmethod
    def _list_images(directory: Path) -> list[Path]:
        """List image files in a directory sorted by name."""
        if not directory.exists():
            return []
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        return sorted([f for f in directory.iterdir() if f.suffix.lower() in extensions])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and transform a multimodal sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                - 'iris_left': Transformed iris tensor (C, H, W) or zero tensor
                - 'iris_right': Transformed iris tensor (C, H, W) or zero tensor
                - 'fingerprint': Transformed fingerprint tensor (C, H, W) or zero tensor
                - 'label': Subject ID as a long tensor
                - 'has_iris_left': Boolean indicating modality availability
                - 'has_iris_right': Boolean indicating modality availability
                - 'has_fingerprint': Boolean indicating modality availability
        """
        sample = self.samples[idx]
        result: dict[str, Any] = {"label": torch.tensor(sample.subject_id, dtype=torch.long)}

        ih, iw = self._iris_size
        fh, fw = self._fingerprint_size

        # Load iris left
        if "iris_left" in self.modalities and sample.iris_left_path:
            result["iris_left"] = self._load_and_transform(
                sample.iris_left_path, self.iris_transform
            )
            result["has_iris_left"] = True
        else:
            result["iris_left"] = torch.zeros(3, ih, iw)
            result["has_iris_left"] = False

        # Load iris right
        if "iris_right" in self.modalities and sample.iris_right_path:
            result["iris_right"] = self._load_and_transform(
                sample.iris_right_path, self.iris_transform
            )
            result["has_iris_right"] = True
        else:
            result["iris_right"] = torch.zeros(3, ih, iw)
            result["has_iris_right"] = False

        # Load fingerprint
        if "fingerprint" in self.modalities and sample.fingerprint_path:
            result["fingerprint"] = self._load_and_transform(
                sample.fingerprint_path,
                self.fingerprint_transform,
                mode="L",
            )
            result["has_fingerprint"] = True
        else:
            result["fingerprint"] = torch.zeros(1, fh, fw)
            result["has_fingerprint"] = False

        return result

    @staticmethod
    def _load_and_transform(
        image_path: str,
        transform: transforms.Compose,
        mode: str = "RGB",
    ) -> torch.Tensor:
        """Load an image from disk and apply transforms.

        Args:
            image_path: Path to the image file.
            transform: Transform pipeline to apply.
            mode: PIL colour mode (``RGB`` for iris, ``L`` for
                grayscale fingerprints).

        Returns:
            Transformed image tensor.
        """
        image = Image.open(image_path).convert(mode)
        result: torch.Tensor = transform(image)
        return result
