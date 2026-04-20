# Design Decisions

Why I chose each technology and what I considered instead. I'm keeping this as a living doc — if I revisit a decision, I'll update the relevant section.

## Hydra for config management

The pipeline has a lot of knobs: batch size, learning rate, image size, scheduler type, storage backend, etc. I needed all of these to be changeable without editing Python, composable across experiments, and automatically recorded for reproducibility.

**What I considered:**

| | Pros | Cons |
|---|---|---|
| argparse | Zero deps, everyone knows it | Flat namespace, can't compose, no YAML |
| click / typer | Nice CLI with type validation | Still need to wire up config file loading manually |
| Plain YAML + PyYAML | Simple | No merging, no CLI overrides |
| **Hydra** | Composable YAML, CLI overrides, multirun, output dirs | Learning curve |
| OmegaConf alone | Lighter weight | No CLI, no sweeps |

**Why Hydra won:** The `defaults` list lets me compose data/model/training/storage configs cleanly. Adding a new model variant is one YAML file, zero Python. CLI overrides (`training.epochs=50`) and multirun sweeps (`-m training.learning_rate=0.001,0.01`) work out of the box. And Hydra snapshots the merged config per run, so every experiment is reproducible by default.

**The downsides I accepted:** Scripts use `@hydra.main()` which can be awkward for testing. Tests either need `hydra.initialize()` or should just mock the config object directly.

## PyArrow/Parquet for data caching

Loading raw images + applying transforms costs 3–5 ms per sample. At 2700 images that's ~8 s per epoch. Annoying but bearable. At 100× scale it's 800 s per epoch — not acceptable.

**What I considered:**

| | Pros | Cons |
|---|---|---|
| No caching | Simple | Pay transform cost every epoch |
| `torch.save` per sample | PyTorch-native | One file per sample = inode nightmare at scale |
| LMDB | Fast random reads | Binary blobs, hard to inspect or debug |
| HDF5 | Common in science | GIL contention with multi-worker DataLoader |
| **Parquet (PyArrow)** | Columnar, compressed, schema-aware | Tensors go in as byte columns |
| WebDataset | Good for huge datasets | Overkill here |

**Why Parquet:** Sharding (100 samples per shard) keeps file counts sane. Snappy compression is free in terms of CPU. You can inspect shards with pandas or duckdb, which is great for debugging. Cache invalidation is automatic via `compute_cache_key()` — change a transform parameter and the cache rebuilds.

**The cost:** Tensors get serialised through numpy → bytes → Parquet binary column, which adds ~1.5× disk overhead. At 120 MB who cares. At 100 GB I'd benchmark zstd (1.1× overhead) vs Snappy.

## Ray for parallel preprocessing

Preprocessing is embarrassingly parallel — each image is independent. Sequential processing at current scale (2700 images, ~8 s) is fine, but at 100k+ it'd be painful.

**What I considered:**

| | Pros | Cons |
|---|---|---|
| ThreadPoolExecutor | Simple, stdlib | GIL kills CPU-bound perf |
| multiprocessing.Pool | Stdlib, real parallelism | Single-node only, no retry, pickle overhead |
| joblib | Simple API | Single-node |
| Dask | Distributed | Heavier than needed for map-style tasks |
| **Ray** | Cluster-ready, fault tolerant, Arrow-native | Big dependency, startup cost |

**Why Ray:** Honestly, on a single machine `multiprocessing.Pool` would be just as fast. I picked Ray because it gives a migration path to cluster-scale processing without rewriting anything. It also retries failed tasks (so one corrupt image doesn't kill the batch), and its shared memory is Arrow-based, which pairs nicely with the Parquet cache.

The `use_ray=False` flag is important — CI and Docker don't need Ray installed.

**Known limitation:** Each task processes one image (~3 ms of work) and Ray's scheduling overhead is ~0.5 ms per task. At 100k+ images I'd batch 50 images per task to amortise that.

## Abstract storage backend

During dev everything is on local disk. In production it might be Azure Blob. I didn't want every script to have if/else logic for this.

**What I considered:**

| | Pros | Cons |
|---|---|---|
| Hardcode `Path()` | Simple | Cloud migration = touch every file |
| Env var switching | Minimal code | Scattered conditionals |
| fsspec | Many backends | Heavy dep, not all ops we need |
| **Custom ABC + factory** | Lightweight, explicit | More upfront code |

**Why a custom ABC:** We only need 4 methods: `upload`, `download`, `list_files`, `exists`. That's 18 lines of abstract code. Adding fsspec would bring in a bunch of transitive deps for backends we'll never use. The factory pattern means adding S3 or GCS later is just one new class.

Config-driven selection:
```yaml
# configs/storage/local.yaml
backend: local
root_dir: data/

# configs/storage/azure.yaml
backend: azure
connection_string: ${oc.env:AZURE_STORAGE_CONNECTION_STRING}
container_name: biometric-data
```

Switching is `storage=azure` on the command line. Azure SDK is optional (`pip install -e ".[azure]"`).

## ONNX for production export

The trained model needs to run in prod without a full PyTorch installation (~2.5 GB).

**What I considered:**

| | Pros | Cons |
|---|---|---|
| Raw PyTorch serving | No conversion | 2.5 GB runtime, slower inference |
| TorchScript | PyTorch-native | Only runs on PyTorch, limited optimisation |
| **ONNX** | Runtime-agnostic, graph opts, TensorRT path | Conversion can be finicky |
| TensorRT direct | Best GPU perf | NVIDIA-only |

**Why ONNX:** One file runs on ONNX Runtime, TensorRT, OpenVINO, Core ML. ORT does graph-level optimisations (op fusion, constant folding) automatically. Triton serves ONNX natively. And the ONNX file (~46 MB) is smaller than the checkpoint (~90 MB) since it doesn't include optimizer state.

I had to write a `_DictInputWrapper` to convert our dict-based forward signature to positional args — ONNX doesn't support dict inputs. I also stuck with the legacy TorchScript exporter (`dynamo=False`) because the dynamo path doesn't support `dynamic_axes` yet.

## Subject-level data splitting

This one is domain-specific and easy to get wrong. The dataset has ~20 images per subject per modality. A random image-level split would leak identity information across train/test — the model would just memorise texture patterns instead of learning generalisable features.

`split_subjects()` puts all images from a subject into exactly one split. With 45 subjects at a 70/15/15 ratio, that's 31/7/7 subjects per split. It's not a lot of test subjects, and metric variance will be high. For a real system I'd use k-fold cross-validation, but for demonstrating the architecture this is sufficient.

## Multi-stage Docker build

A naive single-stage build with PyTorch produces a 5+ GB image. I split it into two stages:

1. **Builder** — has build tools, compiles everything, builds the wheel
2. **Runtime** — slim base, copies only the installed packages

I also added a few production hardening bits:
- Non-root user (`biometric:biometric`)
- `tini` as PID 1 for proper signal handling
- Health check (`python -c "import biometric"`)
- `.dockerignore` to keep `.venv`, `.git`, `data/` out of the build context

The image still ends up at ~3.2 GB because PyTorch is enormous. Not much you can do about that without switching to CPU-only wheels. Build time is ~8 min cold, ~30 s with layer caching.

## MLflow — optional by design

I wanted experiment tracking but didn't want to force everyone to run a tracking server just to train a model. So MLflow is optional: `experiment.py` checks for it at import time and if it's not installed, all calls (`log_params`, `log_metrics`, etc.) become silent no-ops.

```python
def _ensure_mlflow() -> bool:
    if not MLFLOW_AVAILABLE:
        return False
    ...
```

`pip install -e ".[tracking]"` turns it on. CI tests mock MLflow entirely. In production, you'd set `MLFLOW_TRACKING_URI` as an env var and the same code logs to your central server.

## Pre-commit hooks + Ruff

I wanted quality enforcement to happen *before* code hits CI, not after a 5-minute pipeline run. Pre-commit hooks catch issues at `git commit` time.

The hook chain: trailing whitespace → ruff lint → ruff format → mypy → large file check.

**Why Ruff instead of Black + Flake8 + isort?** One tool does all four jobs, and it's 10–100× faster (written in Rust). It's also what most new Python projects are converging on.

The only downside is developers need to run `pre-commit install` once after cloning. CI runs the same checks as a safety net in case someone bypasses hooks.
