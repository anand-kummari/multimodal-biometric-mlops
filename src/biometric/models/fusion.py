"""Multimodal fusion network.

Combines encoded features from iris and fingerprint modalities using
configurable fusion strategies. The Strategy Pattern allows switching
between concatenation, attention-based, and gated fusion without
modifying the training pipeline.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from biometric.data.registry import ModelRegistry
from biometric.models.base import BaseFusionModel
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.iris_encoder import IrisEncoder

logger = logging.getLogger(__name__)


class ConcatenationFusion(nn.Module):
    """Concatenation-based feature fusion.

    The simplest fusion strategy: concatenates feature vectors from all
    modalities along the feature dimension and projects to a shared space.

    Args:
        input_dims: Dictionary mapping modality names to their feature dimensions.
        hidden_dim: Dimensionality of the fused representation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.modality_names = list(input_dims.keys())
        total_dim = sum(input_dims.values())
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features via concatenation.

        Args:
            features: Dict of modality name -> feature tensor (B, D_i).

        Returns:
            Fused feature tensor of shape (B, hidden_dim).
        """
        concatenated = torch.cat([features[n] for n in self.modality_names], dim=1)
        out: torch.Tensor = self.projection(concatenated)
        return out


class AttentionFusion(nn.Module):
    """Attention-based feature fusion.

    Learns modality-specific attention weights to dynamically weight
    the contribution of each modality. This is beneficial when modality
    quality varies across samples (e.g., blurry iris but clear fingerprint).

    Args:
        input_dims: Dictionary mapping modality names to their feature dimensions.
        hidden_dim: Dimensionality of the fused representation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.modality_names = list(input_dims.keys())

        # Project all modalities to the same dimension
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in input_dims.items()}
        )

        # Attention weight network
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), len(input_dims)),
            nn.Softmax(dim=1),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features using learned attention weights.

        Args:
            features: Dict of modality name -> feature tensor (B, D_i).

        Returns:
            Fused feature tensor of shape (B, hidden_dim).
        """
        # Project all modalities to same dimension
        projected = {name: self.projections[name](features[name]) for name in self.modality_names}

        # Compute attention weights
        concat_for_attn = torch.cat([projected[n] for n in self.modality_names], dim=1)
        attn_weights = self.attention(concat_for_attn)  # (B, num_modalities)

        # Weighted sum
        stacked = torch.stack([projected[n] for n in self.modality_names], dim=1)  # (B, M, hidden)
        attn_weights = attn_weights.unsqueeze(-1)  # (B, M, 1)
        fused = (stacked * attn_weights).sum(dim=1)  # (B, hidden)

        out: torch.Tensor = self.output_projection(fused)
        return out


# Mapping of fusion strategy names to their classes
_FUSION_STRATEGIES: dict[str, type[nn.Module]] = {
    "concatenation": ConcatenationFusion,
    "attention": AttentionFusion,
}


@ModelRegistry.register("multimodal_fusion_net")
class MultimodalFusionNet(BaseFusionModel):
    """Complete multimodal biometric recognition network.

    Orchestrates modality-specific encoders and a configurable fusion
    strategy to produce identity predictions from multimodal biometric data.

    Args:
        num_classes: Number of subjects/identities.
        iris_encoder_cfg: Configuration dict for the iris encoder.
        fingerprint_encoder_cfg: Configuration dict for the fingerprint encoder.
        fusion_cfg: Configuration dict for the fusion strategy.
    """

    def __init__(
        self,
        num_classes: int = 45,
        iris_encoder_cfg: dict | None = None,
        fingerprint_encoder_cfg: dict | None = None,
        fusion_cfg: dict | None = None,
    ) -> None:
        super().__init__(num_classes=num_classes)

        iris_cfg = iris_encoder_cfg or {}
        fp_cfg = fingerprint_encoder_cfg or {}
        fuse_cfg = fusion_cfg or {}

        # Modality-specific encoders
        self.iris_encoder = IrisEncoder(
            in_channels=iris_cfg.get("in_channels", 3),
            feature_dim=iris_cfg.get("feature_dim", 128),
            dropout=iris_cfg.get("dropout", 0.3),
        )

        self.fingerprint_encoder = FingerprintEncoder(
            in_channels=fp_cfg.get("in_channels", 1),
            feature_dim=fp_cfg.get("feature_dim", 128),
            dropout=fp_cfg.get("dropout", 0.3),
        )

        # Feature dimensions for fusion
        iris_feat_dim = iris_cfg.get("feature_dim", 128)
        fp_feat_dim = fp_cfg.get("feature_dim", 128)

        input_dims = {
            "iris_left": iris_feat_dim,
            "iris_right": iris_feat_dim,
            "fingerprint": fp_feat_dim,
        }

        # Fusion strategy (Strategy Pattern)
        strategy_name = fuse_cfg.get("strategy", "concatenation")
        if strategy_name not in _FUSION_STRATEGIES:
            available = ", ".join(sorted(_FUSION_STRATEGIES.keys()))
            raise ValueError(
                f"Unknown fusion strategy: {strategy_name!r}. Available: [{available}]"
            )

        fusion_cls = _FUSION_STRATEGIES[strategy_name]
        self.fusion = fusion_cls(
            input_dims=input_dims,
            hidden_dim=fuse_cfg.get("hidden_dim", 256),
            dropout=fuse_cfg.get("dropout", 0.4),
        )

        # Classification head
        hidden_dim = fuse_cfg.get("hidden_dim", 256)
        classifier_dropout = fuse_cfg.get("classifier_dropout", 0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        logger.info(
            "MultimodalFusionNet initialized: %d classes, strategy=%s, params=%d",
            num_classes,
            strategy_name,
            self.count_parameters(),
        )

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        modality_masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass through encoders, fusion, and classifier.

        Args:
            modality_features: Dictionary with keys:
                - 'iris_left': (B, C, H, W) iris images
                - 'iris_right': (B, C, H, W) iris images
                - 'fingerprint': (B, C, H, W) fingerprint images
            modality_masks: Optional dict mapping modality names to boolean
                tensors of shape ``(B,)`` indicating which samples in the
                batch actually have this modality.  When a sample's mask is
                ``False`` the corresponding encoder output is zeroed out so
                the model cannot learn to detect zero-input artifacts.

        Returns:
            Class logits of shape (B, num_classes).
        """
        # Encode each modality (iris encoder is shared for left and right)
        encoded = {
            "iris_left": self.iris_encoder(modality_features["iris_left"]),
            "iris_right": self.iris_encoder(modality_features["iris_right"]),
            "fingerprint": self.fingerprint_encoder(modality_features["fingerprint"]),
        }

        # Mask out absent modalities so the model cannot learn zero-input artifacts
        if modality_masks is not None:
            for name, mask in modality_masks.items():
                if name in encoded:
                    # mask shape: (B,) -> (B, 1) for broadcasting with (B, D)
                    encoded[name] = encoded[name] * mask.unsqueeze(-1).float()

        # Fuse and classify
        fused = self.fusion(encoded)
        logits: torch.Tensor = self.classifier(fused)
        return logits
