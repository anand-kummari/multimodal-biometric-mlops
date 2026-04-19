"""Benchmark suite for data loading performance.

Systematically measures DataLoader throughput under different configurations
to identify optimal settings and bottlenecks. Results are saved as JSON
and plotted for inclusion in documentation.

Benchmarked dimensions:
    - num_workers: 0, 2, 4, 8
    - pin_memory: True / False
    - persistent_workers: True / False
    - With/without PyArrow cache
    - Sequential vs Ray preprocessing

Usage:
    python benchmarks/benchmark_dataloader.py --data-dir data/processed
    python benchmarks/benchmark_dataloader.py --data-dir data/processed --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def benchmark_num_workers(dataset_path: Path, num_epochs: int = 3) -> list[dict]:
    """Benchmark DataLoader with different num_workers settings.

    Args:
        dataset_path: Path to processed dataset.
        num_epochs: Number of epochs to average over.

    Returns:
        List of benchmark result dictionaries.
    """
    from torch.utils.data import DataLoader

    from biometric.data.dataset import MultimodalBiometricDataset
    from biometric.utils.profiling import profile_dataloader

    real_dataset = MultimodalBiometricDataset(data_dir=dataset_path, split="train")

    dataset: Dataset = real_dataset  # type: ignore[type-arg]
    if len(real_dataset) == 0:
        logger.warning("Dataset is empty. Generating synthetic benchmark data.")
        dataset = _create_synthetic_dataset(num_samples=200)

    configs = [
        {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        {"num_workers": 2, "pin_memory": True, "persistent_workers": True},
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True},
        {"num_workers": 8, "pin_memory": True, "persistent_workers": True},
    ]

    results = []
    for config in configs:
        name = f"workers={config['num_workers']}_pin={config['pin_memory']}"
        logger.info("Benchmarking: %s", name)

        num_workers = int(config["num_workers"])
        pin_memory = bool(config["pin_memory"])
        persistent = bool(config.get("persistent_workers", False)) and num_workers > 0

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        profile = profile_dataloader(loader, num_epochs=num_epochs, name=name)

        results.append(
            {
                "config": config,
                "name": name,
                "avg_throughput_sps": profile.avg_throughput,
                "avg_batch_time_ms": profile.avg_batch_time_ms,
                "epoch_results": [
                    {
                        "epoch": i,
                        "throughput": r.throughput,
                        "batch_time_ms": r.avg_batch_time,
                        "total_seconds": r.elapsed_seconds,
                    }
                    for i, r in enumerate(profile.results)
                ],
            }
        )

    return results


def benchmark_preprocessing(raw_dir: Path, processed_dir: Path) -> dict:
    """Benchmark sequential vs Ray parallel preprocessing.

    Args:
        raw_dir: Directory with raw images.
        processed_dir: Output directory for processed images.

    Returns:
        Dictionary with timing comparisons.
    """
    from biometric.preprocessing.parallel_processor import ParallelPreprocessor

    results = {}

    # Sequential
    logger.info("Benchmarking sequential preprocessing...")
    proc_seq = ParallelPreprocessor(use_ray=False)
    start = time.perf_counter()
    seq_results = proc_seq.process_directory(
        source_dir=raw_dir,
        output_dir=processed_dir / "seq_bench",
        target_size=(224, 224),
    )
    seq_time = time.perf_counter() - start
    results["sequential"] = {
        "elapsed_seconds": seq_time,
        "images_processed": len(seq_results),
        "images_per_second": len(seq_results) / max(seq_time, 0.001),
    }

    # Ray parallel
    logger.info("Benchmarking Ray parallel preprocessing...")
    proc_ray = ParallelPreprocessor(use_ray=True)
    start = time.perf_counter()
    ray_results = proc_ray.process_directory(
        source_dir=raw_dir,
        output_dir=processed_dir / "ray_bench",
        target_size=(224, 224),
    )
    ray_time = time.perf_counter() - start
    proc_ray.shutdown()
    results["ray_parallel"] = {
        "elapsed_seconds": ray_time,
        "images_processed": len(ray_results),
        "images_per_second": len(ray_results) / max(ray_time, 0.001),
    }

    # Speedup
    speedup = seq_time / max(ray_time, 0.001) if seq_time > 0 else 0.0
    return {**results, "speedup": speedup}


def _create_synthetic_dataset(num_samples: int = 200) -> Dataset:  # type: ignore[type-arg]
    """Create a synthetic dataset for benchmarking when real data is unavailable."""
    import torch
    from torch.utils.data import Dataset

    class SyntheticDataset(Dataset):
        """Generates random tensors mimicking the multimodal biometric format."""

        def __init__(self, size: int) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int) -> dict:
            return {
                "iris_left": torch.randn(3, 224, 224),
                "iris_right": torch.randn(3, 224, 224),
                "fingerprint": torch.randn(1, 224, 224),
                "label": torch.tensor(idx % 45, dtype=torch.long),
                "has_iris_left": True,
                "has_iris_right": True,
                "has_fingerprint": True,
            }

    return SyntheticDataset(num_samples)


def save_results(results: dict, output_dir: Path) -> None:
    """Save benchmark results as JSON.

    Args:
        results: Benchmark results dictionary.
        output_dir: Directory to save results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


def print_summary(results: dict) -> None:
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print("DATALOADER BENCHMARK RESULTS")
    print("=" * 70)

    if "dataloader" in results:
        print("\n--- DataLoader Configurations ---")
        for entry in results["dataloader"]:
            print(
                f"  {entry['name']:40s} | "
                f"{entry['avg_throughput_sps']:8.1f} samples/s | "
                f"{entry['avg_batch_time_ms']:8.2f} ms/batch"
            )

    if "preprocessing" in results:
        print("\n--- Preprocessing ---")
        for method, data in results["preprocessing"].items():
            if isinstance(data, dict):
                print(
                    f"  {method:40s} | "
                    f"{data.get('images_per_second', 0):8.1f} images/s | "
                    f"{data.get('elapsed_seconds', 0):8.2f}s total"
                )
        if "speedup" in results["preprocessing"]:
            print(f"\n  Ray speedup: {results['preprocessing']['speedup']:.2f}x")

    print("=" * 70)


def main() -> None:
    """Entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark data loading performance")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed dataset",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw dataset (for preprocessing benchmarks)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for dataloader benchmarks",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing benchmarks",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    all_results: dict = {}

    # DataLoader benchmarks
    logger.info("Running DataLoader benchmarks...")
    all_results["dataloader"] = benchmark_num_workers(
        dataset_path=Path(args.data_dir),
        num_epochs=args.epochs,
    )

    # Preprocessing benchmarks
    if not args.skip_preprocessing and Path(args.raw_dir).exists():
        logger.info("Running preprocessing benchmarks...")
        all_results["preprocessing"] = benchmark_preprocessing(
            raw_dir=Path(args.raw_dir),
            processed_dir=Path(args.data_dir),
        )

    # Save and display
    save_results(all_results, Path(args.output_dir))
    print_summary(all_results)


if __name__ == "__main__":
    main()
