"""Factory for creating storage backend instances from configuration."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from biometric.storage.azure import AzureBlobStorageBackend
from biometric.storage.base import StorageBackend
from biometric.storage.local import LocalStorageBackend

logger = logging.getLogger(__name__)

_BACKEND_MAP: dict[str, type[StorageBackend]] = {
    "local": LocalStorageBackend,
    "azure": AzureBlobStorageBackend,
}


def create_storage_backend(cfg: DictConfig | dict[str, Any]) -> StorageBackend:
    """Create a storage backend instance from configuration.

    Args:
        cfg: Configuration dict or DictConfig with keys:
            - backend: Storage backend type ('local' or 'azure').
            - base_path: Root path for local backend.
            - connection_string: Azure connection string (azure backend).
            - container_name: Azure container name (azure backend).

    Returns:
        Configured StorageBackend instance.

    Raises:
        ValueError: If the backend type is not recognized.
    """
    backend_type = cfg.get("backend", "local") if isinstance(cfg, dict) else cfg.backend

    if backend_type not in _BACKEND_MAP:
        available = ", ".join(sorted(_BACKEND_MAP.keys()))
        raise ValueError(f"Unknown storage backend: {backend_type!r}. Available: [{available}]")

    backend: StorageBackend
    if backend_type == "local":
        base_path = cfg.get("base_path", "./data") if isinstance(cfg, dict) else cfg.base_path
        backend = LocalStorageBackend(base_path=base_path)
    elif backend_type == "azure":
        conn_str = cfg.get("connection_string") if isinstance(cfg, dict) else cfg.connection_string
        container = cfg.get("container_name") if isinstance(cfg, dict) else cfg.container_name
        if not conn_str or not container:
            raise ValueError(
                "Azure backend requires 'connection_string' and 'container_name' in config."
            )
        backend = AzureBlobStorageBackend(
            connection_string=conn_str,
            container_name=container,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend_type!r}")

    logger.info("Created storage backend: %s", type(backend).__name__)
    return backend
