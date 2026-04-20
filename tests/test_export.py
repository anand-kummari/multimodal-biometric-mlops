"""Tests for ONNX model export functionality."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from biometric.models.export import export_to_onnx
from biometric.models.fusion import MultimodalFusionNet


class TestONNXExport:
    """Test ONNX export functionality."""

    @pytest.fixture
    def model(self) -> MultimodalFusionNet:
        """Create a small model for testing."""
        return MultimodalFusionNet(num_classes=10)

    @pytest.fixture
    def temp_output_path(self) -> Generator[Path, None, None]:
        """Create a temporary output path."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    def test_export_creates_file(
        self,
        model: MultimodalFusionNet,
        temp_output_path: Path,
    ) -> None:
        """Test that export creates an ONNX file."""
        model.eval()
        result_path = export_to_onnx(model, temp_output_path)

        assert result_path.exists()
        assert result_path.suffix == ".onnx"
        assert result_path.stat().st_size > 0

    def test_export_with_custom_shapes(
        self,
        model: MultimodalFusionNet,
        temp_output_path: Path,
    ) -> None:
        """Test export with custom input shapes."""
        model.eval()
        custom_shapes = {
            "iris_left": (2, 3, 128, 128),
            "iris_right": (2, 3, 128, 128),
            "fingerprint": (2, 1, 128, 128),
        }

        export_to_onnx(model, temp_output_path, input_shapes=custom_shapes)

        assert temp_output_path.exists()

    def test_onnx_inference_single_sample(
        self,
        model: MultimodalFusionNet,
        temp_output_path: Path,
    ) -> None:
        """Test ONNX model can run inference on single sample."""
        model.eval()
        export_to_onnx(model, temp_output_path)

        session = ort.InferenceSession(str(temp_output_path))
        inputs = {
            "iris_left": np.random.randn(1, 3, 224, 224).astype(np.float32),
            "iris_right": np.random.randn(1, 3, 224, 224).astype(np.float32),
            "fingerprint": np.random.randn(1, 1, 224, 224).astype(np.float32),
        }

        outputs = session.run(None, inputs)

        assert len(outputs) == 1
        assert outputs[0].shape == (1, 10)

    def test_onnx_dynamic_batch(
        self,
        model: MultimodalFusionNet,
        temp_output_path: Path,
    ) -> None:
        """Test ONNX model supports dynamic batch sizes."""
        model.eval()
        export_to_onnx(model, temp_output_path, dynamic_batch=True)

        session = ort.InferenceSession(str(temp_output_path))

        for batch_size in [1, 4, 8]:
            inputs = {
                "iris_left": np.random.randn(batch_size, 3, 224, 224).astype(np.float32),
                "iris_right": np.random.randn(batch_size, 3, 224, 224).astype(np.float32),
                "fingerprint": np.random.randn(batch_size, 1, 224, 224).astype(np.float32),
            }

            outputs = session.run(None, inputs)
            assert outputs[0].shape == (batch_size, 10)

    def test_export_creates_parent_directories(self, model: MultimodalFusionNet) -> None:
        """Test that export creates parent directories if they don't exist."""
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "model.onnx"
            export_to_onnx(model, nested_path)

            assert nested_path.exists()

    def test_export_with_different_opset(
        self,
        model: MultimodalFusionNet,
        temp_output_path: Path,
    ) -> None:
        """Test export with different ONNX opset versions."""
        model.eval()
        export_to_onnx(model, temp_output_path, opset_version=14)

        session = ort.InferenceSession(str(temp_output_path))
        assert session is not None
