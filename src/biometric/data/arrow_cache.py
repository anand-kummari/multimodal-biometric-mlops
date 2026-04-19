"""PyArrow-based caching layer for preprocessed image data."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

logger = logging.getLogger(__name__)


class ArrowCacheWriter:
    """Writes preprocessed tensor data to Arrow/Parquet format.

    Collects samples in batches and flushes to Parquet files for
    efficient sequential reads during training.

    Args:
        cache_dir: Directory where cache files will be stored.
        compression: Compression codec ('snappy', 'zstd', or 'none').
        batch_size: Number of samples to accumulate before flushing.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        compression: str = "snappy",
        batch_size: int = 100,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression if compression != "none" else None
        self.batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._file_index: int = 0

    def add_sample(self, sample: dict[str, Any]) -> None:
        """Add a preprocessed sample to the write buffer.

        Args:
            sample: Dictionary with modality tensors and metadata.
                Tensors are serialized to bytes for storage.
        """
        serialized: dict[str, Any] = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                serialized[f"{key}_data"] = _tensor_to_bytes(value)
                serialized[f"{key}_shape"] = json.dumps(list(value.shape))
                serialized[f"{key}_dtype"] = str(value.dtype)
            elif isinstance(value, (bool, int, float)):
                serialized[key] = value
            else:
                serialized[key] = str(value)

        self._buffer.append(serialized)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def finalize(self) -> None:
        """Flush any remaining samples and write metadata."""
        if self._buffer:
            self._flush()
        logger.info("Cache finalized: %d files written to %s", self._file_index, self.cache_dir)

    def _flush(self) -> None:
        """Write buffered samples to a Parquet file."""
        if not self._buffer:
            return

        table = pa.Table.from_pylist(self._buffer)
        output_path = self.cache_dir / f"shard_{self._file_index:05d}.parquet"
        pq.write_table(table, output_path, compression=self.compression)

        logger.debug("Flushed %d samples to %s", len(self._buffer), output_path)
        self._buffer.clear()
        self._file_index += 1


class ArrowCacheReader:
    """Reads preprocessed data from Arrow/Parquet cache.

    Provides random access to cached samples with zero-copy reads
    where possible, significantly reducing data loading latency.

    Args:
        cache_dir: Directory containing cached Parquet files.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._tables: list[pa.Table] = []
        self._cumulative_lengths: list[int] = []
        self._total_length: int = 0
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all Parquet shards into memory."""
        shard_files = sorted(self.cache_dir.glob("shard_*.parquet"))
        if not shard_files:
            logger.warning("No cache shards found in %s", self.cache_dir)
            return

        cumulative = 0
        for shard_path in shard_files:
            table = pq.read_table(shard_path)
            self._tables.append(table)
            cumulative += len(table)
            self._cumulative_lengths.append(cumulative)

        self._total_length = cumulative
        logger.info(
            "Loaded %d cache shards (%d total samples) from %s",
            len(self._tables),
            self._total_length,
            self.cache_dir,
        )

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieve a cached sample by index.

        Args:
            idx: Global sample index.

        Returns:
            Dictionary with deserialized tensors and metadata.
        """
        if idx < 0 or idx >= self._total_length:
            raise IndexError(f"Index {idx} out of range [0, {self._total_length})")

        # Find the correct shard
        shard_idx = 0
        local_idx = idx
        for i, cum_len in enumerate(self._cumulative_lengths):
            if idx < cum_len:
                shard_idx = i
                local_idx = idx - (self._cumulative_lengths[i - 1] if i > 0 else 0)
                break

        row = self._tables[shard_idx].slice(local_idx, 1).to_pydict()
        return self._deserialize_row(row)

    @staticmethod
    def _deserialize_row(row: dict[str, list[Any]]) -> dict[str, Any]:
        """Deserialize a single row from the Parquet table.

        Reconstructs tensors from serialized bytes, shapes, and dtypes.
        """
        result: dict[str, Any] = {}
        processed_keys: set[str] = set()

        for key in row:
            if key.endswith("_data") and key not in processed_keys:
                base_key = key[:-5]  # Remove '_data' suffix
                data_bytes = row[key][0]
                shape_str = row.get(f"{base_key}_shape", [None])[0]
                dtype_str = row.get(f"{base_key}_dtype", [None])[0]

                if data_bytes and shape_str and dtype_str:
                    tensor = _bytes_to_tensor(data_bytes, shape_str, dtype_str)
                    result[base_key] = tensor
                    processed_keys.update({key, f"{base_key}_shape", f"{base_key}_dtype"})
            elif key not in processed_keys:
                value = row[key][0]
                result[key] = value

        return result

    @property
    def is_valid(self) -> bool:
        """Check if the cache contains data."""
        return self._total_length > 0


def compute_cache_key(
    file_list: list[str],
    transform_config: dict[str, Any],
) -> str:
    """Compute a deterministic cache key for invalidation.

    The cache key is a hash of the sorted file list and transform configuration.
    If either changes, the cache is invalidated and rebuilt.

    Args:
        file_list: List of source file paths.
        transform_config: Transform configuration dictionary.

    Returns:
        Hex digest string for cache identification.
    """
    hasher = hashlib.sha256()
    hasher.update(json.dumps(sorted(file_list), sort_keys=True).encode())
    hasher.update(json.dumps(transform_config, sort_keys=True, default=str).encode())
    return hasher.hexdigest()[:16]


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes using numpy's buffer protocol."""
    np_array = tensor.numpy()
    buffer = io.BytesIO()
    np.save(buffer, np_array)
    return buffer.getvalue()


def _bytes_to_tensor(data: bytes, shape_str: str, dtype_str: str) -> torch.Tensor:
    """Deserialize bytes back to a tensor."""
    buffer = io.BytesIO(data)
    np_array = np.load(buffer)
    return torch.from_numpy(np_array)
