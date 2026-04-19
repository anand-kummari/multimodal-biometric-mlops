"""Tests for the Azure Blob Storage backend using mocked SDK clients."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from biometric.storage.azure import AzureBlobStorageBackend


@pytest.fixture
def mock_blob_service() -> MagicMock:
    """Build a fake BlobServiceClient with a nested container/blob hierarchy."""
    service = MagicMock()
    service.url = "https://myaccount.blob.core.windows.net"
    return service


@pytest.fixture
def azure_backend(mock_blob_service: MagicMock) -> AzureBlobStorageBackend:
    """Construct an AzureBlobStorageBackend without calling __init__."""
    backend = object.__new__(AzureBlobStorageBackend)
    backend._connection_string = "DefaultEndpointsProtocol=https;AccountName=test"
    backend._container_name = "test-container"
    backend._client = mock_blob_service
    backend._container = mock_blob_service.get_container_client("test-container")
    return backend


class TestAzureExists:
    """Verify the exists() delegation to blob properties."""

    def test_returns_true_when_blob_present(self, azure_backend: AzureBlobStorageBackend) -> None:
        azure_backend._container.get_blob_client.return_value.get_blob_properties.return_value = {}
        assert azure_backend.exists("data/file.bin") is True

    def test_returns_false_when_blob_missing(self, azure_backend: AzureBlobStorageBackend) -> None:
        azure_backend._container.get_blob_client.return_value.get_blob_properties.side_effect = (
            Exception("BlobNotFound")
        )
        assert azure_backend.exists("no/such/blob.bin") is False


class TestAzureReadBytes:
    """Verify read_bytes downloads and returns raw content."""

    def test_successful_download(self, azure_backend: AzureBlobStorageBackend) -> None:
        stream = MagicMock()
        stream.readall.return_value = b"hello azure"
        blob_client = azure_backend._container.get_blob_client.return_value
        blob_client.download_blob.return_value = stream

        assert azure_backend.read_bytes("path/to/blob") == b"hello azure"

    def test_missing_blob_raises(self, azure_backend: AzureBlobStorageBackend) -> None:
        blob_client = azure_backend._container.get_blob_client.return_value
        blob_client.download_blob.side_effect = Exception("NotFound")

        with pytest.raises(FileNotFoundError):
            azure_backend.read_bytes("missing.bin")


class TestAzureWriteBytes:
    """Verify write_bytes calls upload_blob correctly."""

    def test_upload(self, azure_backend: AzureBlobStorageBackend) -> None:
        azure_backend.write_bytes("output/data.bin", b"payload")
        blob_client = azure_backend._container.get_blob_client.return_value
        blob_client.upload_blob.assert_called_once_with(b"payload", overwrite=True)


class TestAzureListFiles:
    """Verify listing with prefix and pattern filtering."""

    def test_list_with_pattern(self, azure_backend: AzureBlobStorageBackend) -> None:
        blob1 = MagicMock()
        blob1.name = "images/a.png"
        blob2 = MagicMock()
        blob2.name = "images/b.txt"
        azure_backend._container.list_blobs.return_value = [blob1, blob2]

        result = azure_backend.list_files("images", pattern="*.png")
        assert result == ["images/a.png"]

    def test_list_non_recursive_skips_nested(self, azure_backend: AzureBlobStorageBackend) -> None:
        blob_deep = MagicMock()
        blob_deep.name = "images/sub/deep.png"
        azure_backend._container.list_blobs.return_value = [blob_deep]

        result = azure_backend.list_files("images", pattern="*.png", recursive=False)
        assert result == []

    def test_list_recursive_includes_nested(self, azure_backend: AzureBlobStorageBackend) -> None:
        blob_deep = MagicMock()
        blob_deep.name = "images/sub/deep.png"
        azure_backend._container.list_blobs.return_value = [blob_deep]

        result = azure_backend.list_files("images", pattern="*.png", recursive=True)
        assert result == ["images/sub/deep.png"]


class TestAzureOpen:
    """Verify the file-like open interface."""

    def test_read_mode_returns_bytesio(self, azure_backend: AzureBlobStorageBackend) -> None:
        stream = MagicMock()
        stream.readall.return_value = b"content"
        blob_client = azure_backend._container.get_blob_client.return_value
        blob_client.download_blob.return_value = stream

        fobj = azure_backend.open("path.bin", "rb")
        assert fobj.read() == b"content"

    def test_write_mode_uploads_on_close(self, azure_backend: AzureBlobStorageBackend) -> None:
        fobj = azure_backend.open("out.bin", "wb")
        fobj.write(b"data")
        fobj.close()

        blob_client = azure_backend._container.get_blob_client.return_value
        blob_client.upload_blob.assert_called_once()

    def test_unsupported_mode_raises(self, azure_backend: AzureBlobStorageBackend) -> None:
        with pytest.raises(ValueError, match="Unsupported file mode"):
            azure_backend.open("x.bin", "x")


class TestAzureResolvePath:
    """Verify URL construction."""

    def test_url_format(self, azure_backend: AzureBlobStorageBackend) -> None:
        url = azure_backend.resolve_path("data/file.bin")
        assert url == "https://myaccount.blob.core.windows.net/test-container/data/file.bin"


class TestAzureMakedirs:
    """makedirs is a no-op for blob storage but should not raise."""

    def test_no_op(self, azure_backend: AzureBlobStorageBackend) -> None:
        azure_backend.makedirs("any/path")  # should not raise
