"""Reproducibility utilities for deterministic training.

Ensures consistent results across runs by controlling all sources of
randomness: Python, NumPy, PyTorch (CPU and CUDA), and cuDNN.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for reproducibility.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required for certain CUDA operations to be deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("Random seed set to %d (deterministic mode enabled)", seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device based on preference and availability.

    Args:
        preference: One of 'auto', 'cpu', 'cuda', 'mps'.
            - 'auto': Selects CUDA > MPS > CPU based on availability.
            - 'cuda': Requires CUDA; raises if unavailable.
            - 'mps': Requires Apple MPS; raises if unavailable.
            - 'cpu': Always uses CPU.

    Returns:
        Resolved torch.device.

    Raises:
        RuntimeError: If the requested device is not available.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        device = torch.device("mps")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {preference!r}")

    logger.info("Using device: %s", device)
    return device
