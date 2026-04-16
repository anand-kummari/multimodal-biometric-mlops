"""Fingerprint modality encoder.

A lightweight CNN encoder for extracting feature embeddings from fingerprint
images. Fingerprints are typically grayscale, so the default input channel is 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from biometric.data.registry import ModelRegistry
from biometric.models.base import BaseEncoder


@ModelRegistry.register("fingerprint_encoder")
class FingerprintEncoder(BaseEncoder):
    """CNN-based feature extractor for fingerprint images.

    Architecture mirrors the IrisEncoder but accepts single-channel
    (grayscale) input by default. The shared architectural pattern makes
    it easy to swap encoders or experiment with different backbones.

    Args:
        in_channels: Number of input channels (default: 1 for grayscale).
        feature_dim: Output feature vector dimensionality.
        dropout: Dropout probability for regularization.
    """

    def __init__(
        self,
        in_channels: int = 1,
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
        """Extract fingerprint features.

        Args:
            x: Fingerprint image tensor of shape (B, C, H, W).

        Returns:
            Feature vector of shape (B, feature_dim).
        """
        x = self.features(x)
        return self.projection(x)
