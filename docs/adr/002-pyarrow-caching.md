# ADR-002: PyArrow/Parquet for Data Caching

## Status
Accepted

## Context
Training deep learning models requires iterating over the dataset many times (50+ epochs). Each iteration involves:
1. Reading image files from disk
2. Decoding image bytes (PNG/JPEG → pixel array)
3. Applying transforms (resize, normalize, augment)

Steps 1-2 are expensive and repeated identically for non-augmented operations. For augmented training, only step 3 varies. We need a caching strategy to eliminate redundant I/O.

## Decision
Use **Apache Arrow (via PyArrow)** with **Parquet** file format to cache preprocessed image tensors.

## Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **No caching** | Simple | Repeated I/O, 3-5x slower |
| **HDF5** | Mature, random access | Poor compression, no columnar filtering |
| **LMDB** | Fast key-value reads | Complex API, fixed DB size |
| **TFRecord** | Designed for ML pipelines | TensorFlow-specific, not Pythonic |
| **PyArrow/Parquet** | Columnar, compressed, zero-copy reads, SQL-compatible | Serialization overhead on first write |

## Rationale
- **Columnar format**: Can read only specific modalities (e.g., just iris) without loading fingerprints
- **Compression** (snappy/zstd): 2-4x disk reduction; snappy for speed, zstd for ratio
- **Sharded files**: Multiple Parquet shards enable parallel reads and efficient delta updates
- **Cache invalidation**: Hash of source file list + transform config determines cache validity
- **Ecosystem fit**: PyArrow integrates with Ray Data, Pandas, and SQL — all relevant to the Bosch VIPer stack
- **Delta handling**: New subjects can be added as new Parquet shards without reprocessing existing data

## Performance Impact
- First run (cold cache): ~10x overhead (preprocessing + serialization)
- Subsequent runs (warm cache): **3-5x faster** data loading vs raw image decode
- Memory: Arrow's zero-copy reads minimize memory overhead during deserialization

## Consequences
- Cache files stored in `data/cache/` (gitignored)
- Cache key changes automatically when source data or transforms change
- Preprocessing script builds the cache: `python scripts/preprocess.py`
