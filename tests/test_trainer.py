"""Tests for the Trainer and callbacks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from biometric.models.fusion import MultimodalFusionNet
from biometric.training.callbacks import EarlyStopping, ModelCheckpoint
from biometric.training.metrics import MetricTracker
from biometric.training.trainer import Trainer


def _create_dummy_loader(num_samples: int = 20, batch_size: int = 4) -> DataLoader:
    """Create a DataLoader with synthetic multimodal data."""

    class DummyDataset(Dataset):  # type: ignore[type-arg]
        def __init__(self, size: int) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "iris_left": torch.randn(3, 224, 224),
                "iris_right": torch.randn(3, 224, 224),
                "fingerprint": torch.randn(1, 224, 224),
                "label": torch.tensor(idx % 45, dtype=torch.long),
            }

    return DataLoader(DummyDataset(num_samples), batch_size=batch_size, num_workers=0)


class TestTrainer:
    """Tests for the Trainer class."""

    def test_single_epoch(self, device: torch.device) -> None:
        """Test that a single training epoch runs without error."""
        model = MultimodalFusionNet(num_classes=45)
        trainer = Trainer(
            model=model,
            device=device,
            mixed_precision=False,
            gradient_clip_max_norm=None,
        )

        train_loader = _create_dummy_loader(num_samples=8)
        val_loader = _create_dummy_loader(num_samples=4)

        tracker = trainer.fit(train_loader, val_loader, epochs=1)
        assert len(tracker.history) == 1

    def test_metrics_tracked(self, device: torch.device) -> None:
        """Test that training and validation metrics are recorded."""
        model = MultimodalFusionNet(num_classes=45)
        trainer = Trainer(model=model, device=device, mixed_precision=False)

        train_loader = _create_dummy_loader(num_samples=8)
        val_loader = _create_dummy_loader(num_samples=4)

        tracker = trainer.fit(train_loader, val_loader, epochs=2)
        assert len(tracker.history) == 2

    def test_save_training_config(self, device: torch.device, tmp_path: Path) -> None:
        """Test that training config is saved as JSON."""
        model = MultimodalFusionNet(num_classes=45)
        trainer = Trainer(model=model, device=device, mixed_precision=False)

        config_path = tmp_path / "config.json"
        trainer.save_training_config(config_path)
        assert config_path.exists()

    def test_invalid_optimizer(self, device: torch.device) -> None:
        """Test that an invalid optimizer raises ValueError."""
        model = MultimodalFusionNet(num_classes=45)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            Trainer(model=model, device=device, optimizer_name="invalid")


class TestEarlyStopping:
    """Tests for the EarlyStopping callback."""

    def test_no_stop_when_improving(self) -> None:
        cb = EarlyStopping(patience=3, metric="val_loss", mode="min")
        model = MultimodalFusionNet(num_classes=45)

        cb.on_epoch_end(0, {"val_loss": 1.0}, model)
        cb.on_epoch_end(1, {"val_loss": 0.9}, model)
        cb.on_epoch_end(2, {"val_loss": 0.8}, model)
        assert not cb.should_stop

    def test_stop_after_patience(self) -> None:
        cb = EarlyStopping(patience=2, metric="val_loss", mode="min")
        model = MultimodalFusionNet(num_classes=45)

        cb.on_epoch_end(0, {"val_loss": 1.0}, model)
        cb.on_epoch_end(1, {"val_loss": 1.1}, model)
        cb.on_epoch_end(2, {"val_loss": 1.2}, model)
        assert cb.should_stop

    def test_max_mode(self) -> None:
        cb = EarlyStopping(patience=2, metric="val_acc", mode="max")
        model = MultimodalFusionNet(num_classes=45)

        cb.on_epoch_end(0, {"val_acc": 0.5}, model)
        cb.on_epoch_end(1, {"val_acc": 0.6}, model)
        assert not cb.should_stop


class TestModelCheckpoint:
    """Tests for the ModelCheckpoint callback."""

    def test_saves_best_checkpoint(self, tmp_checkpoint_dir: Path) -> None:
        cb = ModelCheckpoint(checkpoint_dir=tmp_checkpoint_dir, metric="val_loss", mode="min")
        model = MultimodalFusionNet(num_classes=45)

        cb.on_epoch_end(0, {"val_loss": 1.0}, model)
        assert (tmp_checkpoint_dir / "checkpoint_best.pt").exists()
        assert (tmp_checkpoint_dir / "checkpoint_last.pt").exists()

    def test_updates_best_on_improvement(self, tmp_checkpoint_dir: Path) -> None:
        cb = ModelCheckpoint(checkpoint_dir=tmp_checkpoint_dir, metric="val_loss", mode="min")
        model = MultimodalFusionNet(num_classes=45)

        cb.on_epoch_end(0, {"val_loss": 1.0}, model)
        cb.on_epoch_end(1, {"val_loss": 0.5}, model)

        checkpoint = torch.load(tmp_checkpoint_dir / "checkpoint_best.pt", weights_only=True)
        assert checkpoint["epoch"] == 1


class TestMetricTracker:
    """Tests for the MetricTracker."""

    def test_update_and_compute(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0)
        tracker.update("loss", 0.5)
        result = tracker.compute_epoch(0)
        assert result.metrics["loss"] == pytest.approx(0.75)

    def test_reset_clears_batch_values(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0)
        tracker.reset()
        result = tracker.compute_epoch(0)
        assert len(result.metrics) == 0

    def test_get_best_min(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0)
        tracker.compute_epoch(0)
        tracker.reset()
        tracker.update("loss", 0.5)
        tracker.compute_epoch(1)

        best = tracker.get_best("loss", mode="min")
        assert best is not None
        assert best.epoch == 1

    def test_history(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0)
        tracker.compute_epoch(0)
        tracker.reset()
        tracker.update("loss", 0.5)
        tracker.compute_epoch(1)

        assert len(tracker.history) == 2

    def test_to_dict(self) -> None:
        tracker = MetricTracker()
        tracker.update("loss", 1.0)
        tracker.compute_epoch(0)
        exported = tracker.to_dict()
        assert len(exported) == 1
        assert "loss" in exported[0]
