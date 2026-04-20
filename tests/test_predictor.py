"""Tests for the inference predictor pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from biometric.inference.predictor import Predictor
from biometric.models.fusion import MultimodalFusionNet


@pytest.fixture
def saved_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal checkpoint file on disk."""
    model = MultimodalFusionNet(num_classes=45)
    ckpt = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "metrics": {"val_loss": 0.42, "val_acc": 0.88},
    }
    path = tmp_path / "checkpoint_best.pt"
    torch.save(ckpt, path)
    return path


class TestPredictorInit:
    """Verify the Predictor loads a checkpoint and sets up transforms."""

    def test_loads_checkpoint(self, saved_checkpoint: Path) -> None:
        predictor = Predictor(checkpoint_path=saved_checkpoint, device=torch.device("cpu"))
        assert predictor.model is not None
        assert predictor.device == torch.device("cpu")

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Predictor(
                checkpoint_path=tmp_path / "nonexistent.pt",
                device=torch.device("cpu"),
            )


class TestPredictSingle:
    """Verify single-sample prediction output structure."""

    def test_predict_with_no_images(self, saved_checkpoint: Path) -> None:
        predictor = Predictor(checkpoint_path=saved_checkpoint, device=torch.device("cpu"))
        result = predictor.predict()
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert isinstance(result["predicted_class"], int)
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["probabilities"]) == 45


class TestPredictBatch:
    """Verify batch prediction from a synthetic DataLoader-style dict."""

    def test_batch_output_shape(self, saved_checkpoint: Path) -> None:
        predictor = Predictor(checkpoint_path=saved_checkpoint, device=torch.device("cpu"))
        batch = {
            "iris_left": torch.randn(4, 3, 224, 224),
            "iris_right": torch.randn(4, 3, 224, 224),
            "fingerprint": torch.randn(4, 1, 224, 224),
        }
        result = predictor.predict_batch(batch)
        assert len(result["predictions"]) == 4
        assert len(result["confidences"]) == 4

    def test_batch_with_partial_modalities(self, saved_checkpoint: Path) -> None:
        """Test batch prediction with missing modalities."""
        predictor = Predictor(checkpoint_path=saved_checkpoint, device=torch.device("cpu"))
        batch = {
            "iris_left": torch.randn(4, 3, 224, 224),
            "iris_right": torch.zeros(4, 3, 224, 224),
            "fingerprint": torch.zeros(4, 1, 224, 224),
            "has_iris_left": torch.tensor([True, True, True, True]),
            "has_iris_right": torch.tensor([False, False, False, False]),
            "has_fingerprint": torch.tensor([False, False, False, False]),
        }
        result = predictor.predict_batch(batch)
        assert len(result["predictions"]) == 4
        assert len(result["confidences"]) == 4


class TestPredictorValidation:
    """Test predictor validation and error handling."""

    def test_num_classes_mismatch_raises(self, saved_checkpoint: Path) -> None:
        """Test that loading a checkpoint with mismatched num_classes raises error."""
        with pytest.raises(
            (ValueError, RuntimeError), match="(num_classes mismatch|size mismatch)"
        ):
            Predictor(
                checkpoint_path=saved_checkpoint,
                model_config={"num_classes": 10},
                device=torch.device("cpu"),
            )

    def test_missing_checkpoint_file_raises(self, tmp_path: Path) -> None:
        """Test that missing checkpoint file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Predictor(
                checkpoint_path=tmp_path / "missing.pt",
                device=torch.device("cpu"),
            )
