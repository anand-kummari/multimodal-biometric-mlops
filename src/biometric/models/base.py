"""Abstract base classes for model components."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, abc.ABC):
    """Abstract encoder that extracts feature embeddings from a single modality.

    All modality-specific encoders (iris, fingerprint) must implement this
    interface to be compatible with the fusion model.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale).
        feature_dim: Dimensionality of the output feature vector.
    """

    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Feature tensor of shape (batch_size, feature_dim).
        """

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseFusionModel(nn.Module, abc.ABC):
    """Abstract fusion model that combines features from multiple modalities.

    The fusion model receives encoded features from individual modality
    encoders and produces a unified prediction. Different fusion strategies
    (early, mid, late, attention-based) can be implemented by subclassing.

    Args:
        num_classes: Number of output classes for classification.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multimodal features and produce class logits.

        Args:
            modality_features: Dictionary mapping modality names to their
                encoded feature tensors, each of shape (batch_size, feature_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
