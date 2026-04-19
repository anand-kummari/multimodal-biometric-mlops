"""Tests for storage backends and factory."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from biometric.storage.factory import create_storage_backend
from biometric.storage.local import LocalStorageBackend


@pytest.fixture
def tmp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storage tests."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestLocalStorageBackend:
    """Tests for the local filesystem storage backend."""

    def test_write_and_read_bytes(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        backend.write_bytes("test/data.bin", b"hello world")
        data = backend.read_bytes("test/data.bin")
        assert data == b"hello world"

    def test_exists_true(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        backend.write_bytes("exists.txt", b"data")
        assert backend.exists("exists.txt")

    def test_exists_false(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        assert not backend.exists("nonexistent.txt")

    def test_read_nonexistent_raises(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        with pytest.raises(FileNotFoundError):
            backend.read_bytes("no_such_file.bin")

    def test_list_files(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        backend.write_bytes("images/a.png", b"img1")
        backend.write_bytes("images/b.png", b"img2")
        backend.write_bytes("images/c.txt", b"txt")

        png_files = backend.list_files("images", pattern="*.png")
        assert len(png_files) == 2

    def test_list_files_recursive(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        backend.write_bytes("a/b/deep.png", b"img")
        backend.write_bytes("a/shallow.png", b"img")

        all_files = backend.list_files("a", pattern="*.png", recursive=True)
        assert len(all_files) == 2

    def test_list_files_empty_dir(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        files = backend.list_files("nonexistent_dir")
        assert files == []

    def test_open_read(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        backend.write_bytes("readable.bin", b"content")

        with backend.open("readable.bin", "rb") as f:
            assert f.read() == b"content"

    def test_open_write(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)

        with backend.open("writable.bin", "wb") as f:
            f.write(b"new content")

        assert backend.read_bytes("writable.bin") == b"new content"

    def test_resolve_path(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        resolved = backend.resolve_path("sub/dir/file.txt")
        # Use resolve() to handle macOS /private/var symlinks
        assert Path(resolved).name == "file.txt"
        assert "sub/dir/file.txt" in resolved

    def test_makedirs(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        backend.makedirs("deep/nested/dir")
        assert (tmp_storage_dir / "deep" / "nested" / "dir").is_dir()

    def test_base_path_property(self, tmp_storage_dir: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_storage_dir)
        # resolve() may add /private prefix on macOS
        assert backend.base_path.name == tmp_storage_dir.name


class TestStorageFactory:
    """Tests for the create_storage_backend factory."""

    def test_create_local_backend(self, tmp_storage_dir: Path) -> None:
        backend = create_storage_backend(
            {
                "backend": "local",
                "base_path": str(tmp_storage_dir),
            }
        )
        assert isinstance(backend, LocalStorageBackend)

    def test_create_default_is_local(self, tmp_storage_dir: Path) -> None:
        backend = create_storage_backend(
            {
                "base_path": str(tmp_storage_dir),
            }
        )
        assert isinstance(backend, LocalStorageBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage_backend(
                {
                    "backend": "s3",
                    "base_path": "/tmp",
                }
            )
