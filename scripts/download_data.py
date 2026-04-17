"""Download and organize the Kaggle multimodal biometric dataset.

This script handles:
1. Downloading the dataset from Kaggle (requires kaggle CLI configured)
2. Extracting the archive
3. Reorganizing into the standardized directory structure expected by the dataset class

Expected output structure:
    data/raw/
        subject_001/
            iris_left/
                img_001.png, img_002.png, ...
            iris_right/
                img_001.png, img_002.png, ...
            fingerprint/
                thumb_left.png, index_left.png, ...

Usage:
    python scripts/download_data.py --output-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "ninadmehendale/multimodal-iris-fingerprint-biometric-data"


def download_from_kaggle(output_dir: Path) -> Path:
    """Download the dataset using the Kaggle CLI.

    Args:
        output_dir: Directory to download the zip file to.

    Returns:
        Path to the downloaded zip file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset from Kaggle: %s", KAGGLE_DATASET)
    # Resolve kaggle binary from the same directory as the running Python interpreter
    # so it works correctly inside virtual environments.
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    kaggle_cmd = str(kaggle_bin) if kaggle_bin.exists() else "kaggle"

    try:
        subprocess.run(
            [
                kaggle_cmd,
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.error(
            "kaggle CLI not found. Install it with: pip install kaggle\n"
            "Then configure credentials: https://www.kaggle.com/docs/api"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Kaggle download failed: %s", e.stderr)
        sys.exit(1)

    # Find the downloaded zip
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        logger.error("No zip file found after download in %s", output_dir)
        sys.exit(1)

    return zip_files[0]


def extract_archive(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the dataset zip archive.

    Args:
        zip_path: Path to the zip file.
        extract_dir: Directory to extract to.

    Returns:
        Path to the extracted content root.
    """
    logger.info("Extracting %s to %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def reorganize_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Reorganize the extracted dataset into the standardized structure.

    The Kaggle dataset has varying structures. This function normalizes
    it into per-subject directories with modality subdirectories.

    Args:
        raw_dir: Directory containing the extracted raw data.
        output_dir: Target directory for the organized structure.
    """
    logger.info("Reorganizing dataset from %s to %s", raw_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files recursively
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    all_images = [f for f in raw_dir.rglob("*") if f.suffix.lower() in image_extensions]

    logger.info("Found %d image files", len(all_images))

    # Attempt to parse subject and modality from directory structure
    # Common patterns:
    #   .../Iris/Left/subject_01/img_001.png
    #   .../Fingerprint/subject_01/thumb.png
    for img_path in all_images:
        parts = img_path.relative_to(raw_dir).parts

        subject_id = _extract_subject_id(parts, img_path.stem)
        modality = _extract_modality(parts)

        if subject_id is None or modality is None:
            logger.debug("Skipping unrecognized path: %s", img_path)
            continue

        # Create target path
        target_dir = output_dir / f"subject_{subject_id:03d}" / modality
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / img_path.name

        if not target_path.exists():
            shutil.copy2(img_path, target_path)

    # Report results
    subjects = [d for d in output_dir.iterdir() if d.is_dir()]
    logger.info("Organized %d subjects into %s", len(subjects), output_dir)


def _extract_subject_id(parts: tuple[str, ...], stem: str) -> int | None:
    """Try to extract a subject ID from the path components.

    The Kaggle dataset structure uses numeric directory names as subject IDs:
        IRIS and FINGERPRINT DATASET/1/left/aeval1.bmp  → subject 1
        IRIS and FINGERPRINT DATASET/1/Fingerprint/1__M_Left_index_finger.BMP → subject 1

    We look for directory parts that are purely numeric (e.g., '1', '23', '045').
    Named prefixes like 'subject_01' are also supported for pre-organized data.
    """
    for part in parts:
        part_lower = part.lower()
        # First try: purely numeric directory names ("1", "23", "045")
        if part_lower.isdigit():
            return int(part_lower)
        # Second try: prefixed patterns ("subject_01", "person_01", "s01")
        for prefix in ("subject_", "person_", "s"):
            if part_lower.startswith(prefix):
                try:
                    num_str = part_lower.removeprefix(prefix).lstrip("0") or "0"
                    return int(num_str)
                except ValueError:
                    continue
    return None


def _extract_modality(parts: tuple[str, ...]) -> str | None:
    """Determine the modality from the immediate parent directory name.

    The Kaggle dataset structure is:
        <optional_parents>/<subject_id>/Fingerprint/<image>.BMP
        <optional_parents>/<subject_id>/left/<image>.bmp        (left iris)
        <optional_parents>/<subject_id>/right/<image>.bmp       (right iris)

    The modality is always the *immediate parent directory* of the image
    file (i.e., `parts[-2]`).
    """
    if len(parts) < 2:
        return None

    # The immediate parent directory of the image file
    parent_dir = parts[-2].lower()

    if parent_dir == "fingerprint" or parent_dir == "finger":
        return "fingerprint"
    if parent_dir == "left":
        return "iris_left"
    if parent_dir == "right":
        return "iris_right"
    if parent_dir in ("iris", "eye"):
        return "iris_left"  # Default to left if unspecified

    return None


def main() -> None:
    """Entry point for dataset download and organization."""
    parser = argparse.ArgumentParser(
        description="Download and organize the multimodal biometric dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to already-extracted dataset (skips download/extract)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use if data is already downloaded)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    output_dir = Path(args.output_dir)

    if args.source_dir:
        # Reorganize from a user-specified already-extracted directory
        source = Path(args.source_dir)
        if not source.exists():
            logger.error("Source directory does not exist: %s", source)
            return
        reorganize_dataset(source, output_dir)
    elif not args.skip_download:
        temp_dir = output_dir.parent / "temp_download"
        zip_path = download_from_kaggle(temp_dir)
        extract_archive(zip_path, temp_dir / "extracted")
        reorganize_dataset(temp_dir / "extracted", output_dir)
        # Clean up temp
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        logger.info("Skipping download; using existing data in %s", output_dir)

    logger.info("Dataset ready at %s", output_dir)


if __name__ == "__main__":
    main()
