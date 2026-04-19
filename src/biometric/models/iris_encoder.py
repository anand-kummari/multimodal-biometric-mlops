"""Iris modality encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from biometric.data.registry import ModelRegistry
from biometric.models.base import BaseEncoder


@ModelRegistry.register("iris_encoder")
class IrisEncoder(BaseEncoder):
    """CNN-based feature extractor for iris images.

    Architecture: 4 convolutional blocks with batch normalization and
    adaptive average pooling, followed by a linear projection to the
    target feature dimension.

    Args:
        in_channels: Number of input channels (default: 3 for RGB iris images).
        feature_dim: Output feature vector dimensionality.
        dropout: Dropout probability for regularization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(in_channels=in_channels, feature_dim=feature_dim)

        self.features = nn.Sequential(
            # Block 1: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract iris features.

        Args:
            x: Iris image tensor of shape (B, C, H, W).

        Returns:
            Feature vector of shape (B, feature_dim).
        """
        x = self.features(x)
        out: torch.Tensor = self.projection(x)
        return out
