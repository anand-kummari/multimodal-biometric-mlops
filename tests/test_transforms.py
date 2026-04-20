"""Tests for modality-specific image transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from biometric.data.registry import TransformRegistry
from biometric.data.transforms import (
    fingerprint_eval_transform,
    fingerprint_train_transform,
    iris_eval_transform,
    iris_train_transform,
)


class TestIrisTransforms:
    """Test iris image transforms."""

    @pytest.fixture
    def rgb_image(self) -> Image.Image:
        """Create a synthetic RGB image."""
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_iris_train_transform_output_shape(self, rgb_image: Image.Image) -> None:
        """Test that iris train transform produces correct output shape."""
        transform = iris_train_transform(image_size=(224, 224))
        output = transform(rgb_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_iris_eval_transform_output_shape(self, rgb_image: Image.Image) -> None:
        """Test that iris eval transform produces correct output shape."""
        transform = iris_eval_transform(image_size=(224, 224))
        output = transform(rgb_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_iris_train_transform_custom_size(self, rgb_image: Image.Image) -> None:
        """Test iris train transform with custom image size."""
        transform = iris_train_transform(image_size=(128, 128))
        output = transform(rgb_image)

        assert output.shape == (3, 128, 128)

    def test_iris_transform_normalization(self, rgb_image: Image.Image) -> None:
        """Test that iris transforms normalize values to [-1, 1] range."""
        transform = iris_eval_transform()
        output = transform(rgb_image)

        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_iris_train_has_augmentation(self, rgb_image: Image.Image) -> None:
        """Test that train transform applies augmentation (outputs differ)."""
        transform = iris_train_transform()

        output1 = transform(rgb_image)
        output2 = transform(rgb_image)

        assert not torch.allclose(output1, output2)

    def test_iris_eval_deterministic(self, rgb_image: Image.Image) -> None:
        """Test that eval transform is deterministic (no augmentation)."""
        transform = iris_eval_transform()

        output1 = transform(rgb_image)
        output2 = transform(rgb_image)

        assert torch.allclose(output1, output2)


class TestFingerprintTransforms:
    """Test fingerprint image transforms."""

    @pytest.fixture
    def grayscale_image(self) -> Image.Image:
        """Create a synthetic grayscale image."""
        arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture
    def rgb_image(self) -> Image.Image:
        """Create a synthetic RGB image (to test grayscale conversion)."""
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_fingerprint_train_transform_output_shape(
        self,
        grayscale_image: Image.Image,
    ) -> None:
        """Test that fingerprint train transform produces correct output shape."""
        transform = fingerprint_train_transform(image_size=(224, 224))
        output = transform(grayscale_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 224, 224)

    def test_fingerprint_eval_transform_output_shape(
        self,
        grayscale_image: Image.Image,
    ) -> None:
        """Test that fingerprint eval transform produces correct output shape."""
        transform = fingerprint_eval_transform(image_size=(224, 224))
        output = transform(grayscale_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 224, 224)

    def test_fingerprint_converts_rgb_to_grayscale(self, rgb_image: Image.Image) -> None:
        """Test that fingerprint transform converts RGB to grayscale."""
        transform = fingerprint_eval_transform()
        output = transform(rgb_image)

        assert output.shape[0] == 1

    def test_fingerprint_train_transform_custom_size(
        self,
        grayscale_image: Image.Image,
    ) -> None:
        """Test fingerprint train transform with custom image size."""
        transform = fingerprint_train_transform(image_size=(128, 128))
        output = transform(grayscale_image)

        assert output.shape == (1, 128, 128)

    def test_fingerprint_transform_normalization(
        self,
        grayscale_image: Image.Image,
    ) -> None:
        """Test that fingerprint transforms normalize values."""
        transform = fingerprint_eval_transform()
        output = transform(grayscale_image)

        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_fingerprint_train_has_augmentation(
        self,
        grayscale_image: Image.Image,
    ) -> None:
        """Test that train transform applies augmentation."""
        transform = fingerprint_train_transform()

        output1 = transform(grayscale_image)
        output2 = transform(grayscale_image)

        assert not torch.allclose(output1, output2)

    def test_fingerprint_eval_deterministic(self, grayscale_image: Image.Image) -> None:
        """Test that eval transform is deterministic."""
        transform = fingerprint_eval_transform()

        output1 = transform(grayscale_image)
        output2 = transform(grayscale_image)

        assert torch.allclose(output1, output2)


class TestTransformRegistry:
    """Test transform registry integration."""

    def test_iris_train_registered(self) -> None:
        """Test that iris_train transform is registered."""
        assert "iris_train" in TransformRegistry

    def test_iris_eval_registered(self) -> None:
        """Test that iris_eval transform is registered."""
        assert "iris_eval" in TransformRegistry

    def test_fingerprint_train_registered(self) -> None:
        """Test that fingerprint_train transform is registered."""
        assert "fingerprint_train" in TransformRegistry

    def test_fingerprint_eval_registered(self) -> None:
        """Test that fingerprint_eval transform is registered."""
        assert "fingerprint_eval" in TransformRegistry

    def test_get_iris_train_from_registry(self) -> None:
        """Test retrieving iris_train transform from registry."""
        transform_fn = TransformRegistry.get("iris_train")
        transform = transform_fn(image_size=(224, 224))

        assert transform is not None

    def test_get_fingerprint_eval_from_registry(self) -> None:
        """Test retrieving fingerprint_eval transform from registry."""
        transform_fn = TransformRegistry.get("fingerprint_eval")
        transform = transform_fn(image_size=(224, 224))

        assert transform is not None
