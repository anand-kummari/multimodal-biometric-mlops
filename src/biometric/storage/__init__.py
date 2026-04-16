"""Storage backend abstractions for local and cloud-based data access."""

from biometric.storage.base import StorageBackend
from biometric.storage.local import LocalStorageBackend
from biometric.storage.factory import create_storage_backend

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "create_storage_backend",
]
