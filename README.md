# Multimodal Biometric Recognition System

End-to-end MLOps infrastructure for multimodal biometric recognition using iris and fingerprint images. The project covers the full lifecycle from raw data ingestion through parallel preprocessing, model training with experiment tracking, to ONNX export for production serving.

Built on PyTorch with a config-driven architecture so that every knob (batch size, fusion strategy, storage backend, learning rate schedule) can be changed from YAML files or the command line without touching source code.


## Project structure

```
.
├── configs/                  # Hydra YAML configurations
│   ├── config.yaml           # Root config (composes sub-configs)
│   ├── data/                 # Data loading and preprocessing
│   ├── model/                # Model architecture (fusion_net)
│   ├── storage/              # Storage backends (local / azure)
│   └── training/             # Hyperparameters, scheduling, callbacks
├── src/biometric/            # Main Python package
│   ├── data/                 # Dataset, DataLoader, transforms, Arrow cache, validation
│   ├── models/               # Encoders (iris, fingerprint), fusion network, ONNX export
│   ├── training/             # Trainer, callbacks, metrics, experiment tracking
│   ├── inference/            # Single-sample and batch prediction
│   ├── preprocessing/        # Ray-based parallel image processing
│   ├── storage/              # Abstract backend + local + Azure Blob implementations
│   └── utils/                # Logging, reproducibility, profiling
├── tests/                    # 184+ unit & integration tests (pytest, 91% coverage)
├── benchmarks/               # DataLoader throughput benchmarks
├── scripts/                  # CLI entry points (download, preprocess, train)
├── Dockerfile                # Multi-stage, non-root, tini-based production image
├── Makefile                  # Common dev commands
├── .github/workflows/ci.yml  # Lint, typecheck, test, security scan, Docker build
└── docs/                     # Architecture, scalability analysis, ADRs, deployment
    ├── architecture.md        # System design with diagrams and data flow
    ├── scalability_analysis.md # Bottleneck analysis and scaling projections
    ├── decisions.md           # Architecture Decision Records (ADRs)
    └── deployment.md          # ONNX serving with Triton and Kubernetes
```

## Getting started

### 1. Installation

```bash
git clone <repo-url>
cd multimodal-biometric-mlops
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"       # Core + dev tools
pre-commit install            # Enable git hooks
```

Optional extras:

```bash
pip install -e ".[azure]"     # Azure Blob Storage support
pip install -e ".[tracking]"  # MLflow experiment tracking
pip install -e ".[export]"    # ONNX model export
```

### 2. Download and organise the dataset

```bash
pip install kaggle            # One-time Kaggle CLI setup
python scripts/download_data.py --output-dir data/raw
```

Dataset: [Multimodal Iris & Fingerprint Biometric Data](https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data) (45 subjects, ~2 700 images).

### 3. Preprocess

```bash
python scripts/preprocess.py                # Ray parallel + Arrow cache
python scripts/preprocess.py --no-ray       # Sequential fallback
python scripts/preprocess.py --cache-only   # Rebuild cache without reprocessing
```

### 4. Train

```bash
python scripts/train.py                                        # Defaults
python scripts/train.py training.epochs=20 data.dataloader.batch_size=32
python scripts/train.py model.model.fusion.strategy=attention  # Attention fusion
python scripts/train.py storage=azure                          # Azure Blob backend
```

Hydra writes outputs (checkpoints, logs, config snapshot) to `outputs/<date>/<time>/`.

### 5. Resume training

```bash
python scripts/train.py +resume_from=outputs/2024-01-15/10-30-00/checkpoints/checkpoint_last.pt
```

Checkpoints include optimizer momentum, scheduler state, and gradient-scaler state, so the resumed run picks up exactly where it left off.

### 6. Export to ONNX

```python
from biometric.models.export import export_to_onnx
from biometric.models.fusion import MultimodalFusionNet

model = MultimodalFusionNet(num_classes=45)
# ... load weights ...
export_to_onnx(model, "model.onnx")
```

### 7. Run benchmarks

```bash
python benchmarks/benchmark_dataloader.py --data-dir data/processed
```

Compares throughput across different `num_workers`, `pin_memory`, and caching configurations. Results are saved to `benchmarks/results/`.

### 8. Hyperparameter sweep

```bash
python scripts/train.py -m training.learning_rate=0.001,0.01,0.1 training.optimizer=adam,sgd
```

Hydra multirun launches one training run per combination and writes results to `multirun/<date>/<time>/`.

## Data validation

Before training starts you can run a dataset health check that flags corrupt images, missing modality folders, and class-imbalance statistics:

```python
from biometric.data.validation import validate_dataset

report = validate_dataset("data/raw")
print(report.summary())
# Subjects:           45
# Total images:       2700
# Corrupt images:     0
# Subjects w/ gaps:   0
```

## Storage backends

The entire pipeline reads and writes through an abstract `StorageBackend` interface. Switching between local disk and Azure Blob is a single config change.

| Backend | Config | When to use |
|---|---|---|
| `local` (default) | `storage=local` | Development, CI, local training |
| `azure` | `storage=azure` | Cloud training, shared team datasets |

For Azure, set the connection string in an environment variable:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
python scripts/train.py storage=azure
```

## Experiment tracking

When MLflow is installed, every training run logs hyperparameters and per-epoch metrics automatically. No code changes needed.

```bash
pip install -e ".[tracking]"
mlflow ui                     # Open http://localhost:5000
python scripts/train.py       # Metrics appear in the UI
```

Without MLflow the tracking calls are silently skipped.

## CI/CD pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:

1. **Lint & format** (ruff) across the entire codebase
2. **Type check** (mypy) with strict config
3. **Unit tests** (pytest) with 85% minimum coverage gate (currently 91%)
4. **Security scan** (pip-audit) for known vulnerabilities
5. **Docker build** verification
6. **Smoke test** with a synthetic forward pass

## Docker

```bash
docker build -t multimodal-biometric-mlops .
docker run --gpus all -v ./data:/app/data multimodal-biometric-mlops \
    python scripts/train.py training.epochs=5
```

The image uses a multi-stage build (builder + slim runtime), runs as a non-root user, and includes a health check.

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System design, component diagrams, data flow |
| [`docs/scalability_analysis.md`](docs/scalability_analysis.md) | Bottleneck analysis, performance benchmarks, scaling projections |
| [`docs/decisions.md`](docs/decisions.md) | Architecture Decision Records — *why* each technology was chosen |
| [`docs/deployment.md`](docs/deployment.md) | ONNX serving via Triton, Kubernetes manifests, monitoring |

## Benchmark Results

DataLoader throughput measured with `benchmarks/benchmark_dataloader.py`:

| Configuration | Throughput (samples/s) | Avg batch time |
|---|---|---|
| `num_workers=0`, no pin | 211 | 75.5 ms |
| `num_workers=2`, pin + persistent | 425 | 38.1 ms |
| `num_workers=4`, pin + persistent | 756 | 21.8 ms |
| `num_workers=8`, pin + persistent | 1 328 | 13.8 ms |

See [`docs/scalability_analysis.md`](docs/scalability_analysis.md) for full analysis.

## Development

```bash
make help          # Show all targets
make lint          # Ruff lint
make format        # Ruff auto-format
make typecheck     # Mypy strict
make test          # Pytest
make test-cov      # Pytest with coverage report
make all           # lint + typecheck + test
make docker-build  # Build Docker image
```
