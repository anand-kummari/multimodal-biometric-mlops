"""Training orchestrator with reproducibility and callback support."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from biometric.training.callbacks import ModelCheckpoint, TrainingCallback
from biometric.training.experiment import log_metrics, log_params
from biometric.training.metrics import MetricTracker

logger = logging.getLogger(__name__)

_OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}


class Trainer:
    """Orchestrates the training loop for multimodal biometric models.

    Handles epoch iteration, forward/backward passes, metric collection,
    callback invocation, mixed precision, gradient clipping, and learning
    rate scheduling.

    Args:
        model: The PyTorch model to train.
        device: Device to train on.
        optimizer_name: Optimizer type ('adam', 'adamw', 'sgd').
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization strength.
        scheduler_type: LR scheduler type ('cosine', 'step', 'plateau').
        warmup_epochs: Number of warmup epochs for the scheduler.
        min_lr: Minimum learning rate for cosine annealing.
        mixed_precision: Enable automatic mixed precision training.
        gradient_clip_max_norm: Max gradient norm for clipping (None to disable).
        callbacks: List of training callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        min_lr: float = 1e-5,
        mixed_precision: bool = True,
        gradient_clip_max_norm: float | None = 1.0,
        callbacks: list[TrainingCallback] | None = None,
        max_epochs: int = 50,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision and device.type == "cuda"
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.callbacks = callbacks or []
        self._start_epoch = 0

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer_name not in _OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name!r}. Available: {list(_OPTIMIZERS.keys())}"
            )
        self.optimizer = _OPTIMIZERS[optimizer_name](
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler(scheduler_type, warmup_epochs, min_lr, max_epochs)

        # Mixed precision scaler
        self.scaler = GradScaler("cuda", enabled=self.mixed_precision)

        # Metric tracking
        self.metric_tracker = MetricTracker()

        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.attach_training_state(self.optimizer, self.scheduler, self.scaler)

        log_params(
            {
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "scheduler": scheduler_type,
                "mixed_precision": str(self.mixed_precision),
                "max_epochs": max_epochs,
            }
        )

        logger.info(
            "Trainer initialized: optimizer=%s, lr=%.6f, mixed_precision=%s, device=%s",
            optimizer_name,
            learning_rate,
            self.mixed_precision,
            device,
        )

    def _create_scheduler(
        self,
        scheduler_type: str,
        warmup_epochs: int,
        min_lr: float,
        max_epochs: int = 50,
    ) -> Any:
        """Create a learning rate scheduler with optional linear warmup.

        When *warmup_epochs* > 0 and a non-plateau scheduler is requested,
        the first *warmup_epochs* epochs use a ``LinearLR`` warmup that
        ramps from ``min_lr`` to the optimizer's initial LR.  The remaining
        epochs follow the requested decay schedule via ``SequentialLR``.
        """

        def _with_warmup(main_scheduler: Any) -> Any:
            if warmup_epochs <= 0:
                return main_scheduler
            warmup = LinearLR(
                self.optimizer,
                start_factor=min_lr / self.optimizer.defaults["lr"],
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_epochs],
            )

        if scheduler_type == "cosine":
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr
            )
            return _with_warmup(cosine)
        elif scheduler_type == "step":
            step = StepLR(self.optimizer, step_size=max(max_epochs // 3, 1), gamma=0.1)
            return _with_warmup(step)
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        else:
            logger.warning("Unknown scheduler '%s', using cosine", scheduler_type)
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr
            )
            return _with_warmup(cosine)

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        epochs: int = 50,
    ) -> MetricTracker:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of training epochs.

        Returns:
            MetricTracker with full training history.
        """
        logger.info("Starting training for %d epochs", epochs)
        total_start = time.perf_counter()

        for epoch in range(self._start_epoch, epochs):
            epoch_start = time.perf_counter()

            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation phase
            val_metrics = self._validate_epoch(val_loader, epoch)

            # Merge metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Compute epoch metrics
            self.metric_tracker.compute_epoch(epoch)
            self.metric_tracker.reset()

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(all_metrics.get("val_loss", 0))
            else:
                self.scheduler.step()

            # Invoke callbacks
            stop = False
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, all_metrics, self.model)
                if callback.should_stop:
                    stop = True

            epoch_time = time.perf_counter() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Publish epoch metrics to the experiment tracker.
            log_metrics({**all_metrics, "lr": current_lr}, step=epoch)

            logger.info(
                "Epoch %d/%d completed in %.1fs | lr=%.6f | %s",
                epoch + 1,
                epochs,
                epoch_time,
                current_lr,
                " | ".join(f"{k}={v:.4f}" for k, v in all_metrics.items()),
            )

            if stop:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        total_time = time.perf_counter() - total_start
        logger.info("Training complete in %.1fs", total_time)
        return self.metric_tracker

    def _train_epoch(self, dataloader: DataLoader[Any], epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            dataloader: Training DataLoader.
            epoch: Current epoch number.

        Returns:
            Dictionary of averaged training metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _batch_idx, batch in enumerate(dataloader):
            # Move data to device
            modality_inputs = {
                "iris_left": batch["iris_left"].to(self.device, non_blocking=True),
                "iris_right": batch["iris_right"].to(self.device, non_blocking=True),
                "fingerprint": batch["fingerprint"].to(self.device, non_blocking=True),
            }
            labels = batch["label"].to(self.device, non_blocking=True)

            # Build modality masks if available in the batch
            modality_masks = self._extract_masks(batch, device=self.device)

            # Forward pass with optional mixed precision
            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(modality_inputs, modality_masks=modality_masks)
                loss = self.criterion(logits, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip_max_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            self.metric_tracker.update("train_loss", loss.item(), count=labels.size(0))

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return {"train_loss": avg_loss, "train_acc": accuracy}

    @torch.no_grad()
    def _validate_epoch(self, dataloader: DataLoader[Any], epoch: int) -> dict[str, float]:
        """Run one validation epoch.

        Args:
            dataloader: Validation DataLoader.
            epoch: Current epoch number.

        Returns:
            Dictionary of averaged validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            modality_inputs = {
                "iris_left": batch["iris_left"].to(self.device, non_blocking=True),
                "iris_right": batch["iris_right"].to(self.device, non_blocking=True),
                "fingerprint": batch["fingerprint"].to(self.device, non_blocking=True),
            }
            labels = batch["label"].to(self.device, non_blocking=True)
            modality_masks = self._extract_masks(batch, device=self.device)

            with autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(modality_inputs, modality_masks=modality_masks)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            self.metric_tracker.update("val_loss", loss.item(), count=labels.size(0))

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return {"val_loss": avg_loss, "val_acc": accuracy}

    @staticmethod
    def _extract_masks(
        batch: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor] | None:
        """Build a modality mask dict from ``has_*`` keys in the batch.

        Returns ``None`` when no mask keys are present (e.g. synthetic test batches).
        """
        mask_map = {
            "iris_left": "has_iris_left",
            "iris_right": "has_iris_right",
            "fingerprint": "has_fingerprint",
        }
        masks: dict[str, torch.Tensor] = {}
        for modality, key in mask_map.items():
            if key in batch:
                masks[modality] = batch[key].to(device, non_blocking=True)
        return masks if masks else None

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Restore training state from a previously saved checkpoint.

        Loads model weights, optimizer momentum buffers, scheduler step
        count so that training continues exactly
        where it was interrupted.

        Args:
            path: Path to the checkpoint file (.pt).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self._start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(
            "Resumed from checkpoint %s (epoch %d)",
            ckpt_path,
            checkpoint.get("epoch", -1),
        )

    def save_training_config(self, path: str | Path) -> None:
        """Save the training configuration for reproducibility."""
        config = {
            "optimizer": type(self.optimizer).__name__,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0),
            "mixed_precision": self.mixed_precision,
            "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
        logger.info("Training config saved to %s", path)
