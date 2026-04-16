"""Tests for the PyArrow caching layer."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch

from biometric.data.arrow_cache import (
    ArrowCacheWriter,
    ArrowCacheReader,
    compute_cache_key,
    _tensor_to_bytes,
    _bytes_to_tensor,
)


@pytest.fixture
def tmp_cache_dir() -> Generator[Path, None, None]:
    """Temporary directory for cache files."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestTensorSerialization:
    """Tests for tensor serialization/deserialization."""

    def test_roundtrip_float_tensor(self) -> None:
        original = torch.randn(3, 224, 224)
        data = _tensor_to_bytes(original)
        restored = _bytes_to_tensor(data, str(list(original.shape)), str(original.dtype))
        assert torch.allclose(original, restored)

    def test_roundtrip_long_tensor(self) -> None:
        original = torch.tensor(42, dtype=torch.long)
        data = _tensor_to_bytes(original)
        restored = _bytes_to_tensor(data, str(list(original.shape)), str(original.dtype))
        assert original.item() == restored.item()

    def test_roundtrip_batch_tensor(self) -> None:
        original = torch.randn(4, 3, 64, 64)
        data = _tensor_to_bytes(original)
        restored = _bytes_to_tensor(data, str(list(original.shape)), str(original.dtype))
        assert torch.allclose(original, restored)


class TestArrowCacheWriter:
    """Tests for the ArrowCacheWriter."""

    def test_write_single_sample(self, tmp_cache_dir: Path) -> None:
        writer = ArrowCacheWriter(cache_dir=tmp_cache_dir, batch_size=10)
        sample = {
            "iris_left": torch.randn(3, 64, 64),
            "label": 5,
            "has_iris": True,
        }
        writer.add_sample(sample)
        writer.finalize()

        # Should have created at least one parquet file
        parquet_files = list(tmp_cache_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_write_multiple_batches(self, tmp_cache_dir: Path) -> None:
        writer = ArrowCacheWriter(cache_dir=tmp_cache_dir, batch_size=5)
        for i in range(12):
            sample = {
                "data": torch.randn(3, 32, 32),
                "label": i % 3,
            }
            writer.add_sample(sample)
        writer.finalize()

        # 12 samples with batch_size=5 -> 2 full flushes + 1 final = 3 files
        parquet_files = list(tmp_cache_dir.glob("*.parquet"))
        assert len(parquet_files) == 3

    def test_compression_options(self, tmp_cache_dir: Path) -> None:
        for compression in ["snappy", "zstd", "none"]:
            sub_dir = tmp_cache_dir / compression
            writer = ArrowCacheWriter(
                cache_dir=sub_dir, compression=compression, batch_size=5
            )
            writer.add_sample({"data": torch.randn(3, 32, 32), "label": 0})
            writer.finalize()
            assert len(list(sub_dir.glob("*.parquet"))) == 1


class TestArrowCacheReader:
    """Tests for the ArrowCacheReader."""

    def _write_test_cache(self, cache_dir: Path, num_samples: int = 10) -> None:
        writer = ArrowCacheWriter(cache_dir=cache_dir, batch_size=5)
        for i in range(num_samples):
            sample = {
                "iris_left": torch.randn(3, 64, 64),
                "label": i % 5,
                "has_iris": True,
            }
            writer.add_sample(sample)
        writer.finalize()

    def test_read_length(self, tmp_cache_dir: Path) -> None:
        self._write_test_cache(tmp_cache_dir, num_samples=10)
        reader = ArrowCacheReader(tmp_cache_dir)
        assert len(reader) == 10

    def test_read_sample(self, tmp_cache_dir: Path) -> None:
        self._write_test_cache(tmp_cache_dir, num_samples=5)
        reader = ArrowCacheReader(tmp_cache_dir)
        sample = reader[0]
        assert "iris_left" in sample
        assert isinstance(sample["iris_left"], torch.Tensor)

    def test_read_all_samples(self, tmp_cache_dir: Path) -> None:
        self._write_test_cache(tmp_cache_dir, num_samples=12)
        reader = ArrowCacheReader(tmp_cache_dir)
        for i in range(len(reader)):
            sample = reader[i]
            assert "iris_left" in sample

    def test_index_out_of_range(self, tmp_cache_dir: Path) -> None:
        self._write_test_cache(tmp_cache_dir, num_samples=5)
        reader = ArrowCacheReader(tmp_cache_dir)
        with pytest.raises(IndexError):
            _ = reader[10]

    def test_negative_index_raises(self, tmp_cache_dir: Path) -> None:
        self._write_test_cache(tmp_cache_dir, num_samples=5)
        reader = ArrowCacheReader(tmp_cache_dir)
        with pytest.raises(IndexError):
            _ = reader[-1]

    def test_empty_cache(self, tmp_cache_dir: Path) -> None:
        reader = ArrowCacheReader(tmp_cache_dir)
        assert len(reader) == 0
        assert not reader.is_valid

    def test_is_valid(self, tmp_cache_dir: Path) -> None:
        self._write_test_cache(tmp_cache_dir, num_samples=5)
        reader = ArrowCacheReader(tmp_cache_dir)
        assert reader.is_valid


class TestComputeCacheKey:
    """Tests for cache key generation."""

    def test_deterministic(self) -> None:
        files = ["a.png", "b.png"]
        config = {"size": 224}
        key1 = compute_cache_key(files, config)
        key2 = compute_cache_key(files, config)
        assert key1 == key2

    def test_different_files_different_key(self) -> None:
        config = {"size": 224}
        key1 = compute_cache_key(["a.png"], config)
        key2 = compute_cache_key(["b.png"], config)
        assert key1 != key2

    def test_different_config_different_key(self) -> None:
        files = ["a.png"]
        key1 = compute_cache_key(files, {"size": 224})
        key2 = compute_cache_key(files, {"size": 128})
        assert key1 != key2

    def test_order_independent_for_files(self) -> None:
        config = {"size": 224}
        key1 = compute_cache_key(["b.png", "a.png"], config)
        key2 = compute_cache_key(["a.png", "b.png"], config)
        assert key1 == key2

    def test_key_length(self) -> None:
        key = compute_cache_key(["a.png"], {"size": 224})
        assert len(key) == 16
