"""Azure Blob Storage backend (stub implementation)."""

from __future__ import annotations

import logging
from typing import BinaryIO

from biometric.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class AzureBlobStorageBackend(StorageBackend):
    """Storage backend for Azure Blob Storage.

    Args:
        connection_string: Azure Storage account connection string.
        container_name: Name of the blob container.

    Example production usage::

        storage = AzureBlobStorageBackend(
            connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            container_name="training-data",
        )
        data = storage.read_bytes("datasets/biometric/iris/subject_01/left_1.png")
    """

    def __init__(self, connection_string: str, container_name: str) -> None:
        self._connection_string = connection_string
        self._container_name = container_name
        # In production:
        # from azure.storage.blob import BlobServiceClient
        # self._client = BlobServiceClient.from_connection_string(connection_string)
        # self._container = self._client.get_container_client(container_name)
        logger.info(
            "AzureBlobStorageBackend initialized (stub) for container: %s",
            container_name,
        )
        raise NotImplementedError("Install azure-storage-blob and implement for production use. ")

    def exists(self, path: str) -> bool:
        """Check if a blob exists."""
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        """Read blob contents."""
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes) -> None:
        """Upload bytes to a blob."""
        raise NotImplementedError

    def list_files(self, directory: str, pattern: str = "*", recursive: bool = False) -> list[str]:
        """List blobs with a given prefix."""
        raise NotImplementedError

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open a blob as a stream."""
        raise NotImplementedError

    def resolve_path(self, path: str) -> str:
        """Return the full blob URL."""
        raise NotImplementedError

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """No-op for blob storage (no directory concept)."""
