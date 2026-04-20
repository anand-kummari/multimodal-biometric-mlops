"""End-to-end integration tests for the full MLOps pipeline.

These tests verify the complete workflow from data loading through
training, export, and inference. They are marked as slow and should
be run selectively in CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from biometric.data.dataloader import create_dataloaders, split_subjects
from biometric.data.dataset import MultimodalBiometricDataset
from biometric.inference.predictor import Predictor
from biometric.models.export import export_to_onnx
from biometric.models.fusion import MultimodalFusionNet
from biometric.training.callbacks import EarlyStopping, ModelCheckpoint
from biometric.training.trainer import Trainer


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic dataset for integration testing."""
    data_dir = tmp_path / "data"

    for subject_id in range(1, 4):
        subject_dir = data_dir / f"subject_{subject_id:03d}"

        for modality in ["iris_left", "iris_right", "fingerprint"]:
            modality_dir = subject_dir / modality
            modality_dir.mkdir(parents=True, exist_ok=True)

            for img_idx in range(2):
                if modality == "fingerprint":
                    img = Image.fromarray(
                        np.random.randint(0, 255, (128, 128), dtype=np.uint8),
                        mode="L",
                    )
                else:
                    img = Image.fromarray(
                        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                        mode="RGB",
                    )

                img.save(modality_dir / f"img_{img_idx}.png")

    return data_dir


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for the complete pipeline."""

    def test_full_training_pipeline(self, synthetic_dataset_dir: Path) -> None:
        """Test complete training pipeline from data to checkpoints."""
        splits = split_subjects(synthetic_dataset_dir, train_ratio=0.6, val_ratio=0.2, seed=42)

        datasets = {
            "train": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="train",
                subject_names=splits["train"],
            ),
            "val": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="val",
                subject_names=splits["val"],
            ),
        }

        loaders = create_dataloaders(
            datasets=datasets,
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
        )

        model = MultimodalFusionNet(num_classes=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                device="cpu",
                callbacks=[
                    ModelCheckpoint(
                        checkpoint_dir=tmpdir,
                        save_best=True,
                        save_last=True,
                    ),
                    EarlyStopping(patience=2, metric="val_loss"),
                ],
            )

            metrics = trainer.fit(
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                epochs=2,
            )

            assert metrics is not None
            assert (Path(tmpdir) / "checkpoint_best.pt").exists()
            assert (Path(tmpdir) / "checkpoint_last.pt").exists()

    def test_train_export_infer_pipeline(self, synthetic_dataset_dir: Path) -> None:
        """Test full pipeline: train → export → inference."""
        splits = split_subjects(synthetic_dataset_dir, train_ratio=0.6, val_ratio=0.2, seed=42)

        datasets = {
            "train": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="train",
                subject_names=splits["train"],
            ),
            "val": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="val",
                subject_names=splits["val"],
            ),
        }

        loaders = create_dataloaders(
            datasets=datasets,
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
        )

        model = MultimodalFusionNet(num_classes=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            trainer = Trainer(
                model=model,
                device="cpu",
                callbacks=[
                    ModelCheckpoint(
                        checkpoint_dir=tmpdir,
                        save_best=True,
                    )
                ],
            )

            trainer.fit(
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                epochs=1,
            )

            checkpoint_path = tmpdir_path / "checkpoint_best.pt"
            assert checkpoint_path.exists()

            onnx_path = tmpdir_path / "model.onnx"
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            export_to_onnx(model, onnx_path)

            assert onnx_path.exists()

            predictor = Predictor(
                checkpoint_path=checkpoint_path,
                model_config={"num_classes": 3},
                device=torch.device("cpu"),
            )

            result = predictor.predict()

            assert "predicted_class" in result
            assert 0 <= result["predicted_class"] < 3
            assert 0.0 <= result["confidence"] <= 1.0

    def test_checkpoint_resume_continues_training(
        self,
        synthetic_dataset_dir: Path,
    ) -> None:
        """Test that resuming from checkpoint continues training correctly."""
        splits = split_subjects(synthetic_dataset_dir, train_ratio=0.6, val_ratio=0.2, seed=42)

        datasets = {
            "train": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="train",
                subject_names=splits["train"],
            ),
            "val": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="val",
                subject_names=splits["val"],
            ),
        }

        loaders = create_dataloaders(
            datasets=datasets,
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
        )

        model = MultimodalFusionNet(num_classes=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer1 = Trainer(
                model=model,
                device="cpu",
                callbacks=[ModelCheckpoint(checkpoint_dir=tmpdir, save_last=True)],
            )

            trainer1.fit(
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                epochs=2,
            )

            model2 = MultimodalFusionNet(num_classes=3)
            trainer2 = Trainer(model=model2, device="cpu")
            trainer2.resume_from_checkpoint(str(Path(tmpdir) / "checkpoint_last.pt"))

            assert trainer2._start_epoch == 2

            trainer2.fit(
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                epochs=2,
            )

    def test_batch_inference_on_real_data(self, synthetic_dataset_dir: Path) -> None:
        """Test batch inference using real dataset samples."""
        splits = split_subjects(synthetic_dataset_dir, train_ratio=0.6, val_ratio=0.2, seed=42)

        val_dataset = MultimodalBiometricDataset(
            data_dir=synthetic_dataset_dir,
            split="val",
            subject_names=splits["val"],
        )

        loaders = create_dataloaders(
            datasets={"val": val_dataset},
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
        )

        model = MultimodalFusionNet(num_classes=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": model.state_dict(),
                    "metrics": {},
                },
                checkpoint_path,
            )

            predictor = Predictor(
                checkpoint_path=checkpoint_path,
                model_config={"num_classes": 3},
                device=torch.device("cpu"),
            )

            batch = next(iter(loaders["val"]))
            result = predictor.predict_batch(batch)

            assert len(result["predictions"]) == len(batch["label"])
            assert len(result["confidences"]) == len(batch["label"])
            assert all(0 <= p < 3 for p in result["predictions"])
            assert all(0.0 <= c <= 1.0 for c in result["confidences"])


@pytest.mark.slow
@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for data loading and preprocessing."""

    def test_dataset_split_no_overlap(self, synthetic_dataset_dir: Path) -> None:
        """Test that train/val/test splits have no subject overlap."""
        splits = split_subjects(synthetic_dataset_dir, train_ratio=0.6, val_ratio=0.2, seed=42)

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_dataloader_iteration(self, synthetic_dataset_dir: Path) -> None:
        """Test that dataloaders can be iterated without errors."""
        splits = split_subjects(synthetic_dataset_dir, train_ratio=0.6, val_ratio=0.2, seed=42)

        datasets = {
            "train": MultimodalBiometricDataset(
                data_dir=synthetic_dataset_dir,
                split="train",
                subject_names=splits["train"],
            )
        }

        loaders = create_dataloaders(
            datasets=datasets,
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
        )

        batch_count = 0
        for batch in loaders["train"]:
            assert "iris_left" in batch
            assert "iris_right" in batch
            assert "fingerprint" in batch
            assert "label" in batch
            batch_count += 1

        assert batch_count > 0
