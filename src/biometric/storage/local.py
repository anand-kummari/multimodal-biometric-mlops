"""Local filesystem storage backend.

This is the default storage backend for development and testing.
All paths are resolved relative to a configurable base directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, cast

from biometric.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """Storage backend for local filesystem access.

    Args:
        base_path: Root directory for all storage operations. All relative
            paths passed to methods are resolved against this base.
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path).resolve()
        logger.info("Initialized LocalStorageBackend at: %s", self._base_path)

    @property
    def base_path(self) -> Path:
        """Return the resolved base path."""
        return self._base_path

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path against the base path."""
        return self._base_path / path

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        return self._resolve(path).exists()

    def read_bytes(self, path: str) -> bytes:
        """Read raw bytes from a file."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        return resolved.read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file, creating parent directories."""
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(data)

    def list_files(self, directory: str, pattern: str = "*", recursive: bool = False) -> list[str]:
        """List files matching a glob pattern."""
        dir_path = self._resolve(directory)
        if not dir_path.exists():
            return []

        glob_pattern = f"**/{pattern}" if recursive else pattern

        return [
            str(p.relative_to(self._base_path))
            for p in sorted(dir_path.glob(glob_pattern))
            if p.is_file()
        ]

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open a file and return a file-like object."""
        resolved = self._resolve(path)
        if "w" in mode or "a" in mode:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return cast(BinaryIO, open(resolved, mode))

    def resolve_path(self, path: str) -> str:
        """Resolve a relative path to its absolute form."""
        return str(self._resolve(path))

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory and any necessary parent directories."""
        self._resolve(path).mkdir(parents=True, exist_ok=exist_ok)
