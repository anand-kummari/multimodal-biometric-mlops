"""Ray-based parallel preprocessing for image data.

Distributes image preprocessing (resize, normalize, augment) across
multiple CPU cores using Ray for significant speedup on large datasets.

Architecture Decision:
    Ray was chosen over Python multiprocessing because:
    1. Automatic serialization of complex objects (PIL images, transforms)
    2. Built-in task scheduling and load balancing
    3. Seamless scaling from local to distributed clusters (Azure, K8s)
    4. Shared memory via the object store reduces data copying

    See docs/adr/003-ray-preprocessing.md for full rationale.

Scalability Notes:
    - Current: Single-node Ray with auto-detected CPUs
    - Scale to cluster: Change ray.init() to connect to Ray cluster on K8s
    - Azure integration: Ray on AKS with autoscaling node pools
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of a preprocessing operation.

    Attributes:
        source_path: Original file path.
        output_path: Processed file output path.
        success: Whether preprocessing succeeded.
        error: Error message if failed.
        elapsed_ms: Processing time in milliseconds.
    """

    source_path: str
    output_path: str
    success: bool
    error: Optional[str] = None
    elapsed_ms: float = 0.0


def _process_single_image(
    source_path: str,
    output_path: str,
    target_size: tuple[int, int],
) -> PreprocessingResult:
    """Process a single image: load, resize, save.

    This is the unit of work that gets distributed across Ray workers.

    Args:
        source_path: Path to the source image.
        output_path: Path to save the processed image.
        target_size: Target (width, height) for resizing.

    Returns:
        PreprocessingResult with success/failure details.
    """
    start = time.perf_counter()
    try:
        img = Image.open(source_path)
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        elapsed = (time.perf_counter() - start) * 1000
        return PreprocessingResult(
            source_path=source_path,
            output_path=output_path,
            success=True,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return PreprocessingResult(
            source_path=source_path,
            output_path=output_path,
            success=False,
            error=str(e),
            elapsed_ms=elapsed,
        )


class ParallelPreprocessor:
    """Orchestrates parallel image preprocessing using Ray.

    Scans a source directory for images, distributes preprocessing tasks
    across Ray workers, and collects results with progress tracking.

    Args:
        num_cpus: Number of CPUs to allocate. None = auto-detect.
        use_ray: If True, use Ray for parallelism. If False, fall back to sequential.
    """

    def __init__(
        self,
        num_cpus: Optional[int] = None,
        use_ray: bool = True,
    ) -> None:
        self.num_cpus = num_cpus
        self.use_ray = use_ray
        self._ray_initialized = False

    def _init_ray(self) -> None:
        """Initialize Ray runtime if not already active."""
        if not self.use_ray or self._ray_initialized:
            return

        try:
            import ray

            if not ray.is_initialized():
                ray.init(
                    num_cpus=self.num_cpus,
                    log_to_driver=False,
                    ignore_reinit_error=True,
                )
            self._ray_initialized = True
            logger.info(
                "Ray initialized: %d CPUs available",
                int(ray.cluster_resources().get("CPU", 0)),
            )
        except ImportError:
            logger.warning("Ray not installed. Falling back to sequential processing.")
            self.use_ray = False

    def process_directory(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        target_size: tuple[int, int] = (224, 224),
        extensions: Optional[set[str]] = None,
    ) -> list[PreprocessingResult]:
        """Preprocess all images in a directory tree.

        Args:
            source_dir: Root directory containing source images.
            output_dir: Root directory for processed outputs.
            target_size: Target (width, height) for resizing.
            extensions: Set of valid file extensions (default: common image formats).

        Returns:
            List of PreprocessingResult for each processed image.
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        extensions = extensions or {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        # Discover all image files
        image_files = [
            f for f in source_dir.rglob("*") if f.suffix.lower() in extensions
        ]

        if not image_files:
            logger.warning("No image files found in %s", source_dir)
            return []

        logger.info("Found %d images to preprocess in %s", len(image_files), source_dir)

        # Build task list
        tasks = []
        for img_path in image_files:
            relative = img_path.relative_to(source_dir)
            out_path = output_dir / relative.with_suffix(".png")
            tasks.append((str(img_path), str(out_path), target_size))

        # Process with Ray or sequentially
        if self.use_ray:
            results = self._process_with_ray(tasks)
        else:
            results = self._process_sequential(tasks)

        # Report summary
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        total_time = sum(r.elapsed_ms for r in results)
        logger.info(
            "Preprocessing complete: %d success, %d failures, %.1fms total",
            successes,
            failures,
            total_time,
        )

        return results

    def _process_with_ray(
        self, tasks: list[tuple[str, str, tuple[int, int]]]
    ) -> list[PreprocessingResult]:
        """Distribute tasks across Ray workers."""
        self._init_ray()
        import ray

        remote_fn = ray.remote(_process_single_image)
        futures = [
            remote_fn.remote(src, out, size) for src, out, size in tasks
        ]
        return ray.get(futures)

    @staticmethod
    def _process_sequential(
        tasks: list[tuple[str, str, tuple[int, int]]]
    ) -> list[PreprocessingResult]:
        """Process tasks sequentially (fallback when Ray is unavailable)."""
        results = []
        for src, out, size in tasks:
            result = _process_single_image(src, out, size)
            results.append(result)
        return results

    def shutdown(self) -> None:
        """Shutdown Ray runtime."""
        if self._ray_initialized:
            import ray

            ray.shutdown()
            self._ray_initialized = False
            logger.info("Ray runtime shut down")
