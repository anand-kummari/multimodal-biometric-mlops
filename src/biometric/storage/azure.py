"""Azure Blob Storage backend."""

from __future__ import annotations

import fnmatch
import io
import logging
from typing import Any, BinaryIO, cast

from biometric.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class AzureBlobStorageBackend(StorageBackend):
    """Storage backend for Azure Blob Storage.

    Args:
        connection_string: Azure Storage account connection string.
        container_name: Name of the blob container.

    Example::

        storage = AzureBlobStorageBackend(
            connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            container_name="training-data",
        )
        data = storage.read_bytes("datasets/biometric/iris/subject_01/left_1.png")
    """

    def __init__(self, connection_string: str, container_name: str) -> None:
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "azure-storage-blob is required for AzureBlobStorageBackend. "
                "Install with: pip install .[azure]"
            ) from exc

        self._connection_string = connection_string
        self._container_name = container_name
        self._client: Any = BlobServiceClient.from_connection_string(connection_string)
        self._container: Any = self._client.get_container_client(container_name)
        logger.info(
            "AzureBlobStorageBackend initialized for container: %s",
            container_name,
        )

    def exists(self, path: str) -> bool:
        """Check if a blob exists in the container.

        Args:
            path: Relative blob path within the container.

        Returns:
            True if the blob exists, False otherwise.
        """
        blob_client = self._container.get_blob_client(path)
        try:
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    def read_bytes(self, path: str) -> bytes:
        """Download and return the full contents of a blob.

        Args:
            path: Relative blob path within the container.

        Returns:
            Blob contents as bytes.

        Raises:
            FileNotFoundError: If the blob does not exist.
        """
        blob_client = self._container.get_blob_client(path)
        try:
            stream = blob_client.download_blob()
            data: bytes = stream.readall()
            return data
        except Exception as exc:
            raise FileNotFoundError(f"Blob not found: {path}") from exc

    def write_bytes(self, path: str, data: bytes) -> None:
        """Upload bytes to a blob, overwriting if it already exists.

        Args:
            path: Relative blob path within the container.
            data: Bytes to upload.
        """
        blob_client = self._container.get_blob_client(path)
        blob_client.upload_blob(data, overwrite=True)
        logger.debug("Uploaded %d bytes to %s", len(data), path)

    def list_files(self, directory: str, pattern: str = "*", recursive: bool = False) -> list[str]:
        """List blobs under a prefix matching a glob pattern.

        Args:
            directory: Blob path prefix (acts as a virtual directory).
            pattern: Glob pattern to filter blob names (e.g., '*.png').
            recursive: If True, list blobs in all virtual subdirectories.

        Returns:
            List of relative blob paths matching the pattern.
        """
        prefix = directory.rstrip("/") + "/" if directory else ""
        blobs: list[str] = []

        for blob in self._container.list_blobs(name_starts_with=prefix):
            blob_name: str = blob.name
            relative = blob_name[len(prefix) :] if prefix else blob_name

            if not recursive and "/" in relative:
                continue

            filename = relative.rsplit("/", 1)[-1] if "/" in relative else relative
            if fnmatch.fnmatch(filename, pattern):
                blobs.append(blob_name)

        return sorted(blobs)

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open a blob and return a file-like object.

        For read modes, downloads the blob into an in-memory buffer.
        For write modes, returns a buffer that is uploaded on close.

        Args:
            path: Relative blob path within the container.
            mode: File open mode ('rb' for read, 'wb' for write).

        Returns:
            A file-like binary object.
        """
        if "r" in mode:
            data = self.read_bytes(path)
            buffer = io.BytesIO(data)
            buffer.seek(0)
            return cast(BinaryIO, buffer)

        if "w" in mode or "a" in mode:
            return cast(BinaryIO, _AzureWriteBuffer(self, path))

        raise ValueError(f"Unsupported file mode: {mode}")

    def resolve_path(self, path: str) -> str:
        """Return the full Azure Blob URL for a given path.

        Args:
            path: Relative blob path within the container.

        Returns:
            Full HTTPS URL to the blob.
        """
        account_url = self._client.url.rstrip("/")
        return f"{account_url}/{self._container_name}/{path}"

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """No-op for blob storage — Azure has no directory concept.

        Args:
            path: Directory path (ignored).
            exist_ok: Ignored for blob storage.
        """


class _AzureWriteBuffer(io.BytesIO):
    """In-memory buffer that uploads to Azure Blob Storage on close.

    Args:
        backend: The AzureBlobStorageBackend instance.
        path: Blob path to upload to when the buffer is closed.
    """

    def __init__(self, backend: AzureBlobStorageBackend, path: str) -> None:
        super().__init__()
        self._backend = backend
        self._path = path

    def close(self) -> None:
        """Upload buffer contents to Azure and then close."""
        if not self.closed:
            self._backend.write_bytes(self._path, self.getvalue())
        super().close()
