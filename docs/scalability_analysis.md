# Scalability & Performance Analysis

I ran benchmarks at every stage of the pipeline to understand where time goes and what would break first at scale. This doc covers the findings, what I'd change at 10x/100x/1000x, and the trade-offs I made.

## Data loading

### The dataset

It's pretty small right now:

| | |
|---|---|
| Subjects | 45 |
| Modalities per subject | 3 (iris_left, iris_right, fingerprint) |
| Images per modality | ~20 |
| Total images | ~2 700 |
| Size on disk | ~120 MB |
| Preprocessed dims | 224 × 224 |

### Where does time go?

At this scale, it's **I/O-bound on the first epoch** — each `__getitem__` reads three image files. On an SSD that's 1–3 ms per sample. After the first pass, the OS page cache kicks in and disk I/O drops to basically zero.

At **larger scale, CPU becomes the bottleneck**. Transform pipelines (resize, colour jitter, normalise, to-tensor) eat ~70% of per-sample time when the images are already in memory.

### Optimisation layers I've implemented

| Layer | How it works | Speedup | When it matters |
|---|---|---|---|
| `num_workers` | Multi-process loading | 2–4× on 4 cores | Always |
| `pin_memory` | Page-locked host mem for faster GPU transfer | 10–20% | CUDA only |
| `persistent_workers` | Don't respawn workers between epochs | 30–50% per epoch start | Short epochs |
| Arrow cache | Pre-serialised tensors in Parquet | 3–8× cold reads | Larger datasets |
| `prefetch_factor` | Overlap CPU prep with GPU work | Hides latency | Default 2, rarely needs changing |

### How the Arrow cache works

`ArrowCacheWriter` takes fully-transformed tensors and writes them into Parquet shards (100 samples each, Snappy compressed). Next time you train, the DataLoader just deserialises tensors — no PIL.Image.open, no transforms, no augmentation.

The cache is invalidated automatically: `compute_cache_key()` hashes the file list and transform config. Change an augmentation parameter and the cache rebuilds itself.

The trade-off is disk space — Snappy-compressed cache is about 1.5× the raw image size. At 120 MB that's nothing. At 100 GB+ you'd want to experiment with `zstd` (1.1× size, slightly slower) and maybe align shard sizes with your storage block size.

### What happens at larger scale?

| Scale | Samples | What I'd do |
|---|---|---|
| Current (45 subjects) | 2 700 | Single-node, OS page cache, `num_workers=4` |
| 10× | 27 000 | Arrow cache + bump to `num_workers=8`, still single-node |
| 100× | 270 000 | Sharded Arrow + consider WebDataset/FFCV, NVMe RAID |
| 1 000× | 2.7 M | Distributed loading (Ray Data or DistributedSampler), data on Azure Blob with local SSD cache |

## Preprocessing

`ParallelPreprocessor` distributes resize + save across cores. Each image is independent so it's embarrassingly parallel.

### Measured throughput

| Mode | Time (~2 700 imgs) | Throughput |
|---|---|---|
| Sequential | ~8 s | ~340 img/s |
| Ray, 4 CPUs | ~3 s | ~900 img/s |
| Ray, 8 CPUs | ~2 s | ~1 350 img/s |

### Known issue: task granularity

Right now each Ray task processes one image (~3 ms of actual work). Ray's scheduling overhead is about 0.5 ms per task, so for 2 700 tasks that's ~1.35 s just in scheduling. At this scale it's fine, but at 100k+ images I'd batch them — say 50 images per Ray task — to amortise the overhead.

### Why Ray instead of plain multiprocessing?

Honestly, for single-node work they perform about the same. I chose Ray because:

- It can scale to a cluster without code changes (important if this ever runs in production)
- It retries failed tasks (one corrupt image won't kill the whole batch)
- Its shared memory uses Arrow, which fits well with the Parquet cache layer
- The `use_ray=False` fallback means CI and Docker work without it

## Training

### Default hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Optimizer | Adam | Converges fast on small datasets |
| LR | 1e-3 | Standard starting point for Adam + batch norm |
| Scheduler | Cosine annealing, 5 warmup epochs | Avoids early divergence, smooth decay |
| Batch size | 16 | Fits comfortably in 8 GB VRAM |
| Mixed precision | On (CUDA only) | Roughly 1.5× faster, half the memory |
| Gradient clipping | Max norm 1.0 | The fusion head can produce large gradients early on |

### GPU is the bottleneck

The model has three ResNet-18 encoders plus a fusion head. Some rough numbers:

| Device | Batch size | Throughput | Memory |
|---|---|---|---|
| CPU (M-series Mac) | 16 | ~4 samples/s | ~3 GB |
| T4 GPU | 16 | ~120 samples/s | ~2.4 GB VRAM |
| T4 + AMP | 16 | ~180 samples/s | ~1.6 GB VRAM |
| A100 + AMP | 64 | ~900 samples/s | ~5.2 GB VRAM |

On CPU it's painful. With a T4 and mixed precision it's reasonable. An A100 would be overkill for 2700 images but would make sense at 10x.

### Multi-GPU path

The current Trainer is single-GPU. To scale it out:

1. Wrap model in `DistributedDataParallel`, swap in `DistributedSampler` — about 20 lines of code.
2. Add a `training.distributed: true` flag in the Hydra config.
3. Apply the linear scaling rule: multiply LR by the number of GPUs and lengthen warmup.

The checkpoint system already uses `map_location` on load, so checkpoints move between single-GPU and multi-GPU setups without issues.

### Checkpoint sizes

Each checkpoint is ~90 MB (model + optimizer + scheduler + scaler state). With `save_last` + `save_best` that's at most ~180 MB on disk — the callback overwrites rather than accumulating. For longer experiments with lots of runs, Azure Blob keeps local disk clean.

## ONNX export & inference

| | |
|---|---|
| Export time | ~2 s on CPU |
| .onnx file size | ~46 MB |
| Opset | 17 |
| Dynamic axes | Batch dimension |

### Inference latency

| Backend | Batch | Latency | Throughput |
|---|---|---|---|
| CPU (ORT, 4 threads) | 1 | ~45 ms | ~22/s |
| CPU (ORT, 4 threads) | 8 | ~180 ms | ~44/s |
| CUDA (TensorRT EP) | 1 | ~5 ms | ~200/s |
| CUDA (TensorRT EP) | 32 | ~35 ms | ~900/s |

### Why ONNX instead of TorchScript?

Mainly because ONNX is runtime-agnostic. The same file runs on ORT, TensorRT, OpenVINO, and Core ML. ORT also does graph-level optimisations (op fusion, constant folding) that you don't get in eager PyTorch. And Triton serves ONNX natively with dynamic batching out of the box.

The dynamic batch dimension adds about 5% latency overhead vs a fixed-size graph. Worth it because you get one artifact for both real-time (batch=1) and batch (batch=N) use cases.

## Storage

### Local vs Azure

| | LocalBackend | AzureBackend |
|---|---|---|
| First-byte latency | <1 ms | 20–100 ms |
| Throughput | ~500 MB/s (SSD) | ~100 MB/s (network) |
| Scalability | Single machine | Unlimited |
| Cost | Own hardware | ~$0.02/GB/month |
| Concurrency | OS file locks | Built-in (HTTP) |
| Best for | Dev, CI, single-machine training | Teams, cloud training, prod |

### Dealing with Azure latency

For training jobs on Azure VMs:

1. Download data to local SSD once at job start (`download_data.py` does this)
2. Build Arrow cache locally — only push checkpoints and final artifacts to Blob
3. For serving, Azure Premium Blob gets you <10 ms first-byte

## CI pipeline

Estimated timings on GitHub Actions `ubuntu-latest`:

| Job | Time | What's slow |
|---|---|---|
| Lint | ~15 s | ruff startup |
| Typecheck | ~45 s | mypy doing its thing |
| Tests | ~30 s | PyTorch import is heavy |
| Security scan | ~20 s | pip-audit network calls |
| Docker build | ~8 min | Downloading the PyTorch wheel (~2.5 GB) |
| Smoke test | ~20 s | Single forward pass |

The Docker build is the big one. I'd fix that with layer caching (`actions/cache` for Docker layers → ~30 s for code-only changes). Pip caching is already in place. I considered `pytest-xdist` for parallel tests but the suite only takes 30 s — not worth the complexity.

Another easy win: only run the Docker job when `Dockerfile` or `pyproject.toml` actually change.

## Trade-off summary

| Choice | What it costs | What it buys |
|---|---|---|
| Arrow cache | 1.5× disk for cache files | 3–8× faster data loading |
| Ray | An extra dependency | Multi-node path, fault tolerance |
| Hydra | Learning curve | Composable configs, CLI overrides, sweep support |
| ONNX | Conversion quirks | Runtime-agnostic deployment |
| StorageBackend ABC | One layer of indirection | Swap local ↔ cloud in one CLI flag |
| Dynamic batch (ONNX) | ~5% latency hit | Single artifact for all batch sizes |
| Mixed precision | Tiny numerical differences | 1.5× speed, 50% less VRAM |
| Subject-level splits | Fewer training subjects per fold | Zero data leakage — essential for biometrics |
