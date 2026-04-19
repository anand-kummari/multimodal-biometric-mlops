"""Abstract base class for storage backends.

1. Local development with fast filesystem access
2. Cloud deployment with Azure Blob Storage integration
3. Easy addition of new storage backends (S3, GCS) without modifying consumers
"""

from __future__ import annotations

import abc
from typing import BinaryIO


class StorageBackend(abc.ABC):
    """Abstract interface for storage operations.

    All data access in the pipeline goes through this interface, making the
    system storage-agnostic. Implementations must handle path resolution,
    file I/O, and listing operations for their respective storage medium.
    """

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists at the given path.

        Args:
            path: Relative path within the storage backend.

        Returns:
            True if the path exists, False otherwise.
        """

    @abc.abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read raw bytes from a file.

        Args:
            path: Relative path to the file.

        Returns:
            File contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """

    @abc.abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file, creating parent directories as needed.

        Args:
            path: Relative path for the output file.
            data: Bytes to write.
        """

    @abc.abstractmethod
    def list_files(self, directory: str, pattern: str = "*", recursive: bool = False) -> list[str]:
        """List files in a directory matching a glob pattern.

        Args:
            directory: Relative path to the directory.
            pattern: Glob pattern to filter files (e.g., '*.png').
            recursive: If True, search recursively in subdirectories.

        Returns:
            List of relative file paths matching the pattern.
        """

    @abc.abstractmethod
    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open a file and return a file-like object.

        Args:
            path: Relative path to the file.
            mode: File open mode (default: 'rb').

        Returns:
            File-like object.
        """

    @abc.abstractmethod
    def resolve_path(self, path: str) -> str:
        """Resolve a relative path to its absolute/full form.

        Args:
            path: Relative path within the storage backend.

        Returns:
            Absolute path string.
        """

    @abc.abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory and any necessary parent directories.

        Args:
            path: Directory path to create.
            exist_ok: If True, don't raise error if directory exists.
        """
