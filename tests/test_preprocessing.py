"""Tests for the parallel preprocessing pipeline."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from biometric.preprocessing.parallel_processor import (
    ParallelPreprocessor,
    PreprocessingResult,
    _process_single_image,
)


@pytest.fixture
def tmp_image_dir() -> Path:
    """Create temporary directory with test images."""
    tmp_dir = Path(tempfile.mkdtemp())
    for i in range(5):
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        img.save(tmp_dir / f"test_{i:03d}.png")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def tmp_output_dir() -> Path:
    """Create temporary output directory."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestProcessSingleImage:
    """Tests for the single image processing function."""

    def test_resize_image(self, tmp_image_dir: Path, tmp_output_dir: Path) -> None:
        src = str(tmp_image_dir / "test_000.png")
        out = str(tmp_output_dir / "output.png")
        result = _process_single_image(src, out, (224, 224))

        assert result.success
        assert Path(out).exists()

        img = Image.open(out)
        assert img.size == (224, 224)

    def test_nonexistent_source(self, tmp_output_dir: Path) -> None:
        result = _process_single_image(
            "/nonexistent/image.png",
            str(tmp_output_dir / "output.png"),
            (224, 224),
        )
        assert not result.success
        assert result.error is not None

    def test_timing_recorded(self, tmp_image_dir: Path, tmp_output_dir: Path) -> None:
        src = str(tmp_image_dir / "test_000.png")
        out = str(tmp_output_dir / "output.png")
        result = _process_single_image(src, out, (224, 224))

        assert result.elapsed_ms > 0


class TestParallelPreprocessor:
    """Tests for the ParallelPreprocessor."""

    def test_sequential_processing(
        self, tmp_image_dir: Path, tmp_output_dir: Path
    ) -> None:
        """Test sequential fallback (no Ray)."""
        processor = ParallelPreprocessor(use_ray=False)
        results = processor.process_directory(
            source_dir=tmp_image_dir,
            output_dir=tmp_output_dir,
            target_size=(224, 224),
        )

        assert len(results) == 5
        assert all(r.success for r in results)

    def test_empty_directory(self, tmp_output_dir: Path) -> None:
        """Test with empty source directory."""
        empty_dir = tmp_output_dir / "empty"
        empty_dir.mkdir()
        processor = ParallelPreprocessor(use_ray=False)
        results = processor.process_directory(
            source_dir=empty_dir,
            output_dir=tmp_output_dir / "output",
            target_size=(224, 224),
        )
        assert len(results) == 0

    def test_output_files_created(
        self, tmp_image_dir: Path, tmp_output_dir: Path
    ) -> None:
        processor = ParallelPreprocessor(use_ray=False)
        processor.process_directory(
            source_dir=tmp_image_dir,
            output_dir=tmp_output_dir,
            target_size=(224, 224),
        )

        output_images = list(tmp_output_dir.glob("*.png"))
        assert len(output_images) == 5
