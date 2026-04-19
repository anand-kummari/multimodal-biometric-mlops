"""Training entry point with Hydra configuration.

This is the main script for launching training runs. All hyperparameters,
data paths, and model configuration are driven by Hydra YAML configs.

Usage:
    # Default training
    python scripts/train.py

    # Quick debug run
    python scripts/train.py training=quick

    # Override specific parameters
    python scripts/train.py training.epochs=10 data.dataloader.batch_size=32

    # Multirun sweep
    python scripts/train.py -m training.learning_rate=0.001,0.0001,0.00001
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main training function orchestrated by Hydra.

    Args:
        cfg: Hydra-composed configuration from YAML files.
    """
    from biometric.data.dataloader import create_dataloaders
    from biometric.data.dataset import MultimodalBiometricDataset
    from biometric.models.fusion import MultimodalFusionNet
    from biometric.training.callbacks import EarlyStopping, ModelCheckpoint, TrainingCallback
    from biometric.training.experiment import end_run, init_experiment
    from biometric.training.trainer import Trainer
    from biometric.utils.logging import setup_logging
    from biometric.utils.reproducibility import get_device, set_seed

    # Setup
    setup_logging(level=cfg.logging.level, format_str=cfg.logging.format)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    # Start experiment tracking (no-op if MLflow is not installed)
    init_experiment(
        experiment_name=cfg.project.name,
        run_name=f"{cfg.project.name}_{Path.cwd().name}",
    )

    # Data
    data_cfg = cfg.data
    dataset = MultimodalBiometricDataset(
        data_dir=cfg.storage.processed_dir,
        split="train",
        iris_size=tuple(data_cfg.dataset.iris_size),
        fingerprint_size=tuple(data_cfg.dataset.fingerprint_size),
        modalities=list(data_cfg.dataset.modalities),
    )

    dataloaders = create_dataloaders(
        dataset=dataset,
        batch_size=data_cfg.dataloader.batch_size,
        num_workers=data_cfg.dataloader.num_workers,
        pin_memory=data_cfg.dataloader.pin_memory,
        persistent_workers=data_cfg.dataloader.persistent_workers,
        prefetch_factor=data_cfg.dataloader.prefetch_factor,
        drop_last=data_cfg.dataloader.drop_last,
        train_ratio=data_cfg.dataloader.split.train,
        val_ratio=data_cfg.dataloader.split.val,
        seed=cfg.project.seed,
    )

    # Model
    model_cfg = cfg.model.model
    iris_cfg = dict(OmegaConf.to_container(model_cfg.iris_encoder, resolve=True))  # type: ignore[arg-type]
    fp_cfg = dict(OmegaConf.to_container(model_cfg.fingerprint_encoder, resolve=True))  # type: ignore[arg-type]
    fusion_cfg_dict = dict(OmegaConf.to_container(model_cfg.fusion, resolve=True))  # type: ignore[arg-type]
    model = MultimodalFusionNet(
        num_classes=model_cfg.num_classes,
        iris_encoder_cfg=iris_cfg,
        fingerprint_encoder_cfg=fp_cfg,
        fusion_cfg=fusion_cfg_dict,
    )

    # Callbacks
    train_cfg = cfg.training.training
    callbacks: list[TrainingCallback] = []

    if train_cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                patience=train_cfg.early_stopping.patience,
                metric=train_cfg.early_stopping.metric,
                mode=train_cfg.early_stopping.mode,
            )
        )

    if train_cfg.checkpointing.enabled:
        callbacks.append(
            ModelCheckpoint(
                checkpoint_dir=cfg.storage.checkpoint_dir,
                metric=train_cfg.checkpointing.metric,
                mode=train_cfg.checkpointing.mode,
                save_best=train_cfg.checkpointing.save_best,
                save_last=train_cfg.checkpointing.save_last,
            )
        )

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer_name=train_cfg.optimizer,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        scheduler_type=train_cfg.scheduler.type,
        warmup_epochs=train_cfg.scheduler.warmup_epochs,
        min_lr=train_cfg.scheduler.min_lr,
        mixed_precision=train_cfg.mixed_precision,
        gradient_clip_max_norm=(
            train_cfg.gradient_clip.max_norm if train_cfg.gradient_clip.enabled else None
        ),
        callbacks=callbacks,
    )

    # Save training config for reproducibility
    trainer.save_training_config("training_config.json")

    # Train
    metric_tracker = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=train_cfg.epochs,
    )

    # Log final results
    best = metric_tracker.get_best("val_loss", mode="min")
    if best:
        logger.info("Best epoch: %s", best)

    logger.info("Training complete. Outputs saved to: %s", Path.cwd())
    end_run()


if __name__ == "__main__":
    main()
