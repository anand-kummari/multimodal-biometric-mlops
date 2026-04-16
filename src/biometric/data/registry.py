"""Registry pattern for dynamic registration of datasets and transforms.

The Registry pattern enables config-driven instantiation of components without
hard-coding class references. This is essential for scalable systems where new
datasets or transform pipelines can be added without modifying existing code.

Usage:
    @DatasetRegistry.register("multimodal_biometric")
    class MultimodalBiometricDataset(BaseDataset):
        ...

    dataset_cls = DatasetRegistry.get("multimodal_biometric")
    dataset = dataset_cls(**config)
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for dynamically registering and retrieving classes.

    This pattern decouples component creation from component usage, enabling
    config-driven instantiation via Hydra or similar configuration systems.

    Args:
        name: Human-readable name for this registry (used in error messages).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, key: str) -> Any:
        """Decorator to register a class under the given key.

        Args:
            key: Unique identifier for the class in this registry.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Raises:
            ValueError: If the key is already registered.
        """

        def decorator(cls: type[T]) -> type[T]:
            if key in self._registry:
                raise ValueError(
                    f"'{key}' is already registered in {self._name} registry. "
                    f"Existing: {self._registry[key].__name__}, New: {cls.__name__}"
                )
            self._registry[key] = cls
            logger.debug("Registered '%s' in %s registry: %s", key, self._name, cls.__name__)
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Retrieve a registered class by key.

        Args:
            key: The identifier used during registration.

        Returns:
            The registered class.

        Raises:
            KeyError: If the key is not found in the registry.
        """
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{key}' not found in {self._name} registry. "
                f"Available: [{available}]"
            )
        return self._registry[key]

    def list_registered(self) -> list[str]:
        """Return all registered keys.

        Returns:
            Sorted list of registered keys.
        """
        return sorted(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={self.list_registered()})"


# Global registries for the application
DatasetRegistry: Registry[Any] = Registry("Dataset")
TransformRegistry: Registry[Any] = Registry("Transform")
ModelRegistry: Registry[Any] = Registry("Model")
