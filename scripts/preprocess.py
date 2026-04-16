"""Preprocess raw biometric images using parallel processing.

Runs Ray-based parallel preprocessing to resize and normalize images,
then optionally builds a PyArrow cache for fast training data loading.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --no-ray       # Sequential fallback
    python scripts/preprocess.py --cache-only   # Only build Arrow cache
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_preprocessing(
    raw_dir: Path,
    processed_dir: Path,
    target_size: tuple[int, int] = (224, 224),
    use_ray: bool = True,
    num_cpus: int | None = None,
) -> None:
    """Run parallel preprocessing on raw images.

    Args:
        raw_dir: Directory containing raw organized data.
        processed_dir: Output directory for processed images.
        target_size: Target image size (width, height).
        use_ray: Whether to use Ray for parallel processing.
        num_cpus: Number of CPUs to use (None = auto-detect).
    """
    from biometric.preprocessing.parallel_processor import ParallelPreprocessor

    processor = ParallelPreprocessor(num_cpus=num_cpus, use_ray=use_ray)

    logger.info("Starting preprocessing: %s -> %s", raw_dir, processed_dir)
    start = time.perf_counter()

    results = processor.process_directory(
        source_dir=raw_dir,
        output_dir=processed_dir,
        target_size=target_size,
    )

    elapsed = time.perf_counter() - start
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)

    logger.info(
        "Preprocessing complete: %d images in %.2fs (%d success, %d failed)",
        len(results),
        elapsed,
        successes,
        failures,
    )

    # Log any failures
    for r in results:
        if not r.success:
            logger.warning("Failed: %s -> %s", r.source_path, r.error)

    processor.shutdown()


def build_arrow_cache(
    processed_dir: Path,
    cache_dir: Path,
    compression: str = "snappy",
) -> None:
    """Build PyArrow cache from processed images.

    Args:
        processed_dir: Directory with processed images.
        cache_dir: Output directory for Arrow cache files.
        compression: Compression codec ('snappy', 'zstd', 'none').
    """
    from biometric.data.arrow_cache import ArrowCacheWriter
    from biometric.data.dataset import MultimodalBiometricDataset

    logger.info("Building Arrow cache: %s -> %s", processed_dir, cache_dir)
    start = time.perf_counter()

    # Create dataset (eval mode, no augmentation for caching)
    dataset = MultimodalBiometricDataset(
        data_dir=processed_dir,
        split="train",  # Cache all data, split at DataLoader level
    )

    writer = ArrowCacheWriter(
        cache_dir=cache_dir,
        compression=compression,
        batch_size=50,
    )

    for idx in range(len(dataset)):
        sample = dataset[idx]
        writer.add_sample(sample)

        if (idx + 1) % 100 == 0:
            logger.info("Cached %d/%d samples", idx + 1, len(dataset))

    writer.finalize()

    elapsed = time.perf_counter() - start
    logger.info(
        "Arrow cache built: %d samples in %.2fs at %s",
        len(dataset),
        elapsed,
        cache_dir,
    )


def main() -> None:
    """Entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess biometric images")
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Raw data directory",
    )
    parser.add_argument(
        "--processed-dir", type=str, default="data/processed",
        help="Processed data output directory",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/cache",
        help="Arrow cache output directory",
    )
    parser.add_argument(
        "--target-size", type=int, nargs=2, default=[224, 224],
        help="Target image size (width height)",
    )
    parser.add_argument(
        "--no-ray", action="store_true",
        help="Disable Ray (use sequential processing)",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=None,
        help="Number of CPUs for Ray (default: auto)",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Only build Arrow cache (skip image preprocessing)",
    )
    parser.add_argument(
        "--compression", type=str, default="snappy",
        choices=["snappy", "zstd", "none"],
        help="Arrow cache compression",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    cache_dir = Path(args.cache_dir)

    if not args.cache_only:
        run_preprocessing(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            target_size=tuple(args.target_size),
            use_ray=not args.no_ray,
            num_cpus=args.num_cpus,
        )

    build_arrow_cache(
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
