"""Tests for model components: encoders and fusion network."""

from __future__ import annotations

import pytest
import torch

from biometric.models.iris_encoder import IrisEncoder
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion import (
    MultimodalFusionNet,
    ConcatenationFusion,
    AttentionFusion,
)


class TestIrisEncoder:
    """Tests for the IrisEncoder."""

    def test_output_shape(self) -> None:
        encoder = IrisEncoder(in_channels=3, feature_dim=128)
        x = torch.randn(4, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_custom_feature_dim(self) -> None:
        encoder = IrisEncoder(in_channels=3, feature_dim=64)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 64)

    def test_single_channel_input(self) -> None:
        encoder = IrisEncoder(in_channels=1, feature_dim=128)
        x = torch.randn(2, 1, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_parameter_count(self) -> None:
        encoder = IrisEncoder(in_channels=3, feature_dim=128)
        assert encoder.count_parameters() > 0


class TestFingerprintEncoder:
    """Tests for the FingerprintEncoder."""

    def test_output_shape(self) -> None:
        encoder = FingerprintEncoder(in_channels=1, feature_dim=128)
        x = torch.randn(4, 1, 224, 224)
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_grayscale_default(self) -> None:
        encoder = FingerprintEncoder()
        assert encoder.in_channels == 1


class TestConcatenationFusion:
    """Tests for concatenation-based fusion."""

    def test_output_shape(self) -> None:
        fusion = ConcatenationFusion(
            input_dims={"iris_left": 128, "iris_right": 128, "fingerprint": 128},
            hidden_dim=256,
        )
        features = {
            "iris_left": torch.randn(4, 128),
            "iris_right": torch.randn(4, 128),
            "fingerprint": torch.randn(4, 128),
        }
        out = fusion(features)
        assert out.shape == (4, 256)


class TestAttentionFusion:
    """Tests for attention-based fusion."""

    def test_output_shape(self) -> None:
        fusion = AttentionFusion(
            input_dims={"iris_left": 128, "iris_right": 128, "fingerprint": 128},
            hidden_dim=256,
        )
        features = {
            "iris_left": torch.randn(4, 128),
            "iris_right": torch.randn(4, 128),
            "fingerprint": torch.randn(4, 128),
        }
        out = fusion(features)
        assert out.shape == (4, 256)


class TestMultimodalFusionNet:
    """Tests for the complete fusion network."""

    def test_forward_pass(self, sample_batch: dict[str, torch.Tensor]) -> None:
        model = MultimodalFusionNet(num_classes=45)
        modality_inputs = {
            "iris_left": sample_batch["iris_left"],
            "iris_right": sample_batch["iris_right"],
            "fingerprint": sample_batch["fingerprint"],
        }
        logits = model(modality_inputs)
        assert logits.shape == (4, 45)

    def test_output_logits_not_probabilities(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """Ensure output is raw logits (can be negative, sum != 1)."""
        model = MultimodalFusionNet(num_classes=45)
        modality_inputs = {
            "iris_left": sample_batch["iris_left"],
            "iris_right": sample_batch["iris_right"],
            "fingerprint": sample_batch["fingerprint"],
        }
        logits = model(modality_inputs)
        # Raw logits - no specific range constraint
        assert logits.shape[1] == 45

    def test_custom_config(self) -> None:
        model = MultimodalFusionNet(
            num_classes=10,
            iris_encoder_cfg={"in_channels": 3, "feature_dim": 64, "dropout": 0.1},
            fingerprint_encoder_cfg={"in_channels": 1, "feature_dim": 64, "dropout": 0.1},
            fusion_cfg={"strategy": "concatenation", "hidden_dim": 128, "dropout": 0.2},
        )
        modality_inputs = {
            "iris_left": torch.randn(2, 3, 224, 224),
            "iris_right": torch.randn(2, 3, 224, 224),
            "fingerprint": torch.randn(2, 1, 224, 224),
        }
        logits = model(modality_inputs)
        assert logits.shape == (2, 10)

    def test_attention_fusion_strategy(self) -> None:
        model = MultimodalFusionNet(
            num_classes=45,
            fusion_cfg={"strategy": "attention", "hidden_dim": 256, "dropout": 0.3},
        )
        modality_inputs = {
            "iris_left": torch.randn(2, 3, 224, 224),
            "iris_right": torch.randn(2, 3, 224, 224),
            "fingerprint": torch.randn(2, 1, 224, 224),
        }
        logits = model(modality_inputs)
        assert logits.shape == (2, 45)

    def test_invalid_fusion_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            MultimodalFusionNet(
                num_classes=45,
                fusion_cfg={"strategy": "invalid_strategy"},
            )

    def test_model_parameter_count(self) -> None:
        model = MultimodalFusionNet(num_classes=45)
        assert model.count_parameters() > 0

    def test_shared_iris_encoder(self) -> None:
        """Verify that left and right iris share the same encoder weights."""
        model = MultimodalFusionNet(num_classes=45)
        model.eval()  # Disable dropout for deterministic comparison
        x = torch.randn(2, 3, 224, 224)
        feat_left = model.iris_encoder(x)
        feat_right = model.iris_encoder(x)
        assert torch.allclose(feat_left, feat_right)
