"""Factory for creating storage backend instances from configuration.

Uses the Factory Pattern to instantiate the correct storage backend
based on the Hydra configuration, keeping consumer code decoupled
from specific backend implementations.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from biometric.storage.base import StorageBackend
from biometric.storage.local import LocalStorageBackend

logger = logging.getLogger(__name__)

_BACKEND_MAP: dict[str, type[StorageBackend]] = {
    "local": LocalStorageBackend,
    # "azure": AzureBlobStorageBackend,  # Enable when azure-storage-blob is installed
}


def create_storage_backend(cfg: DictConfig | dict[str, Any]) -> StorageBackend:
    """Create a storage backend instance from configuration.

    Args:
        cfg: Configuration dict or DictConfig with keys:
            - backend: Storage backend type ('local' or 'azure').
            - base_path: Root path for the storage backend.
            - Additional backend-specific keys.

    Returns:
        Configured StorageBackend instance.

    Raises:
        ValueError: If the backend type is not recognized.
    """
    backend_type = cfg.get("backend", "local") if isinstance(cfg, dict) else cfg.backend
    base_path = cfg.get("base_path", "./data") if isinstance(cfg, dict) else cfg.base_path

    if backend_type not in _BACKEND_MAP:
        available = ", ".join(sorted(_BACKEND_MAP.keys()))
        raise ValueError(
            f"Unknown storage backend: {backend_type!r}. Available: [{available}]"
        )

    backend_cls = _BACKEND_MAP[backend_type]

    if backend_type == "local":
        backend = backend_cls(base_path=base_path)
    else:
        # Azure backend would need connection_string and container_name
        raise ValueError(
            f"Backend '{backend_type}' requires additional configuration. "
            "See docs/adr/005-storage-abstraction.md for setup instructions."
        )

    logger.info("Created storage backend: %s", type(backend).__name__)
    return backend
