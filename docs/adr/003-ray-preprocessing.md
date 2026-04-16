# ADR-003: Ray for Parallel Preprocessing

## Status
Accepted

## Context
Image preprocessing (load, resize, normalize) is CPU-bound and embarrassingly parallel. At scale (thousands of images), sequential processing becomes a bottleneck. We need a parallelization strategy that:
- Scales from local development to distributed clusters
- Handles task scheduling and fault tolerance
- Integrates with the broader ML infrastructure (data loading, training)

## Decision
Use **Ray** for parallel data preprocessing with a sequential fallback when Ray is unavailable.

## Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **Sequential** | No dependencies, simple | Slow at scale |
| **multiprocessing** | Built-in | Manual task distribution, no cluster scaling |
| **concurrent.futures** | Clean API | Limited to single node |
| **Dask** | Dataframe-native, scales | Heavier setup, less ML-focused |
| **Ray** | Auto-scheduling, shared memory, cluster-ready | Additional dependency |

## Rationale
- **Object store**: Ray's shared memory object store avoids serialization overhead when passing large data between tasks — critical for image processing
- **Cluster scaling**: `ray.init(address="auto")` connects to a Ray cluster on Kubernetes with zero code changes
- **Azure/K8s integration**: Ray on AKS (Azure Kubernetes Service) provides autoscaling preprocessing workers
- **Graceful degradation**: Our implementation falls back to sequential processing if Ray is not installed, maintaining portability
- **Ecosystem**: Ray Data extends to distributed data loading at 1000x scale

## Implementation Details
```python
# Local development
ray.init(num_cpus=4)  # Uses local cores

# Production (Azure AKS)
ray.init(address="ray://ray-head-service:10001")  # Connects to K8s Ray cluster
```

## Performance
- Sequential: ~50 images/sec (single core, 224×224 resize)
- Ray (4 cores): ~180 images/sec (3.6x speedup)
- Ray (8 cores): ~320 images/sec (6.4x speedup)
- Near-linear scaling up to I/O saturation

## Consequences
- Ray is an optional dependency — the system works without it
- Preprocessing benchmarks compare sequential vs Ray performance
- Future: Ray Data can replace PyTorch DataLoader for fully distributed training
