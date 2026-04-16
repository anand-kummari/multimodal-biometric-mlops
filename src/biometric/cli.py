"""CLI entry points for the biometric package.

These functions are registered as console scripts in pyproject.toml
and can be invoked as: biometric-train, biometric-infer, biometric-preprocess.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def train() -> None:
    """Launch the training pipeline via Hydra."""
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "train.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))


def infer() -> None:
    """Launch the inference pipeline."""
    print("Inference CLI not yet implemented. Use: python scripts/infer.py")
    sys.exit(0)


def preprocess() -> None:
    """Launch the preprocessing pipeline."""
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "preprocess.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
