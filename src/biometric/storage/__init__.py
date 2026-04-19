"""Storage backend abstractions for local and cloud-based data access."""

from biometric.storage.azure import AzureBlobStorageBackend
from biometric.storage.base import StorageBackend
from biometric.storage.factory import create_storage_backend
from biometric.storage.local import LocalStorageBackend

__all__ = [
    "AzureBlobStorageBackend",
    "LocalStorageBackend",
    "StorageBackend",
    "create_storage_backend",
]
