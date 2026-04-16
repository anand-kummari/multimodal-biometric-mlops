"""Inference pipeline for biometric identity prediction.

Provides a clean, production-ready interface for loading a trained model
checkpoint and running predictions on new multimodal biometric samples.
Separates inference concerns from training to support deployment scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from biometric.data.transforms import iris_eval_transform, fingerprint_eval_transform
from biometric.models.fusion import MultimodalFusionNet

logger = logging.getLogger(__name__)


class Predictor:
    """Loads a trained model and runs inference on biometric samples.

    Encapsulates checkpoint loading, device management, and transform
    application so that consumers only need to provide image paths.

    Args:
        checkpoint_path: Path to the saved model checkpoint (.pt file).
        device: Compute device. If None, auto-selects.
        model_config: Model configuration dict (must match training config).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        model_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Build model from config
        cfg = model_config or {}
        self.model = MultimodalFusionNet(
            num_classes=cfg.get("num_classes", 45),
            iris_encoder_cfg=cfg.get("iris_encoder"),
            fingerprint_encoder_cfg=cfg.get("fingerprint_encoder"),
            fusion_cfg=cfg.get("fusion"),
        )

        # Load checkpoint
        self._load_checkpoint()

        # Eval transforms (no augmentation)
        self.iris_transform = iris_eval_transform()
        self.fingerprint_transform = fingerprint_eval_transform()

        logger.info(
            "Predictor initialized: checkpoint=%s, device=%s",
            self.checkpoint_path,
            self.device,
        )

    def _load_checkpoint(self) -> None:
        """Load model weights from checkpoint file."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        epoch = checkpoint.get("epoch", "unknown")
        metrics = checkpoint.get("metrics", {})
        logger.info(
            "Loaded checkpoint from epoch %s, metrics: %s", epoch, metrics
        )

    @torch.no_grad()
    def predict(
        self,
        iris_left_path: Optional[str] = None,
        iris_right_path: Optional[str] = None,
        fingerprint_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run inference on a single multimodal sample.

        Args:
            iris_left_path: Path to left iris image.
            iris_right_path: Path to right iris image.
            fingerprint_path: Path to fingerprint image.

        Returns:
            Dictionary with:
                - 'predicted_class': Predicted subject ID.
                - 'confidence': Softmax probability of the predicted class.
                - 'probabilities': Full probability distribution over classes.
        """
        # Load and transform each modality
        iris_left = self._load_modality(iris_left_path, self.iris_transform, channels=3)
        iris_right = self._load_modality(iris_right_path, self.iris_transform, channels=3)
        fingerprint = self._load_modality(
            fingerprint_path, self.fingerprint_transform, channels=1
        )

        # Add batch dimension and move to device
        modality_inputs = {
            "iris_left": iris_left.unsqueeze(0).to(self.device),
            "iris_right": iris_right.unsqueeze(0).to(self.device),
            "fingerprint": fingerprint.unsqueeze(0).to(self.device),
        }

        # Forward pass
        logits = self.model(modality_inputs)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.cpu().numpy().tolist(),
        }

    @torch.no_grad()
    def predict_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Run inference on a batch from a DataLoader.

        Args:
            batch: Dictionary with modality tensors from the dataset.

        Returns:
            Dictionary with batch predictions and confidences.
        """
        modality_inputs = {
            "iris_left": batch["iris_left"].to(self.device),
            "iris_right": batch["iris_right"].to(self.device),
            "fingerprint": batch["fingerprint"].to(self.device),
        }

        logits = self.model(modality_inputs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values

        return {
            "predictions": predictions.cpu().numpy().tolist(),
            "confidences": confidences.cpu().numpy().tolist(),
        }

    def _load_modality(
        self,
        image_path: Optional[str],
        transform: transforms.Compose,
        channels: int,
    ) -> torch.Tensor:
        """Load and transform a single modality image.

        If the path is None, returns a zero tensor (missing modality).
        """
        if image_path is None:
            return torch.zeros(channels, 224, 224)

        image = Image.open(image_path).convert("RGB")
        return transform(image)
