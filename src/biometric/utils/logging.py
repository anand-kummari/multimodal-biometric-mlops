"""Structured logging configuration.

Provides a centralized logging setup that integrates with Hydra's
logging configuration while allowing standalone usage.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(
    level: str = "INFO",
    format_str: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    log_file: str | None = None,
) -> None:
    """Configure the root logger with consistent formatting.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_str: Format string for log messages.
        log_file: Optional path to a log file. If None, logs only to stdout.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ("PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
