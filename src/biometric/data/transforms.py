"""Modality-specific image transforms.

Defines separate transform pipelines for iris and fingerprint images,
reflecting the different characteristics of each modality. Transform
pipelines are registered via the TransformRegistry for config-driven selection.
"""

from __future__ import annotations

from torchvision import transforms

from biometric.data.registry import TransformRegistry


def _build_iris_transforms(
    image_size: tuple[int, int] = (224, 224),
    augment: bool = False,
) -> transforms.Compose:
    """Build transform pipeline for iris images.

    Iris images are RGB and benefit from color jitter and slight geometric
    augmentation during training.

    Args:
        image_size: Target (height, width) for resizing.
        augment: If True, include training-time augmentations.

    Returns:
        Composed transform pipeline.
    """
    transform_list: list[transforms.Transform] = []

    if augment:
        transform_list.extend(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform_list.extend(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    return transforms.Compose(transform_list)


def _build_fingerprint_transforms(
    image_size: tuple[int, int] = (224, 224),
    augment: bool = False,
) -> transforms.Compose:
    """Build transform pipeline for fingerprint images.

    Fingerprint images are grayscale. Augmentations are more conservative
    since ridge orientation is a key feature that should be preserved.

    Args:
        image_size: Target (height, width) for resizing.
        augment: If True, include training-time augmentations.

    Returns:
        Composed transform pipeline.
    """
    transform_list: list[transforms.Transform] = []

    if augment:
        transform_list.extend(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        transform_list.extend(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    return transforms.Compose(transform_list)


@TransformRegistry.register("iris_train")
def iris_train_transform(image_size: tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Training transforms for iris images."""
    return _build_iris_transforms(image_size=image_size, augment=True)


@TransformRegistry.register("iris_eval")
def iris_eval_transform(image_size: tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Evaluation transforms for iris images (no augmentation)."""
    return _build_iris_transforms(image_size=image_size, augment=False)


@TransformRegistry.register("fingerprint_train")
def fingerprint_train_transform(
    image_size: tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """Training transforms for fingerprint images."""
    return _build_fingerprint_transforms(image_size=image_size, augment=True)


@TransformRegistry.register("fingerprint_eval")
def fingerprint_eval_transform(
    image_size: tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """Evaluation transforms for fingerprint images (no augmentation)."""
    return _build_fingerprint_transforms(image_size=image_size, augment=False)
