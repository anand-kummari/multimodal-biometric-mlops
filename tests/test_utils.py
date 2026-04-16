"""Tests for utility modules: reproducibility, logging, and profiling."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from biometric.utils.reproducibility import set_seed, get_device
from biometric.utils.logging import setup_logging
from biometric.utils.profiling import Timer, TimingResult, DataLoaderProfile, profile_dataloader


class TestSetSeed:
    """Tests for the set_seed function."""

    def test_torch_determinism(self) -> None:
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_numpy_determinism(self) -> None:
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        assert np.allclose(a, b)

    def test_different_seeds_different_output(self) -> None:
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b)


class TestGetDevice:
    """Tests for the get_device function."""

    def test_cpu_device(self) -> None:
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_returns_device(self) -> None:
        device = get_device("auto")
        assert isinstance(device, torch.device)

    def test_invalid_preference_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device preference"):
            get_device("tpu")


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_sets_log_level(self) -> None:
        setup_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_file_logging(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name

        setup_logging(level="INFO", log_file=log_path)
        logger = logging.getLogger("test_file_logging")
        logger.info("test message")

        content = Path(log_path).read_text()
        assert "test message" in content

        Path(log_path).unlink(missing_ok=True)

    def test_console_only(self) -> None:
        setup_logging(level="WARNING")
        root = logging.getLogger()
        # Should have at least one handler (console)
        assert len(root.handlers) >= 1


class TestTimingResult:
    """Tests for the TimingResult dataclass."""

    def test_throughput(self) -> None:
        result = TimingResult(
            name="test", elapsed_seconds=2.0, iterations=10, samples_processed=100
        )
        assert result.throughput == pytest.approx(50.0)

    def test_throughput_zero_time(self) -> None:
        result = TimingResult(
            name="test", elapsed_seconds=0.0, samples_processed=10
        )
        assert result.throughput == float("inf")

    def test_avg_batch_time(self) -> None:
        result = TimingResult(
            name="test", elapsed_seconds=1.0, iterations=10
        )
        assert result.avg_batch_time == pytest.approx(100.0)

    def test_avg_batch_time_zero_iterations(self) -> None:
        result = TimingResult(name="test", elapsed_seconds=1.0, iterations=0)
        assert result.avg_batch_time == 0.0

    def test_summary(self) -> None:
        result = TimingResult(
            name="bench", elapsed_seconds=1.5, iterations=5, samples_processed=50
        )
        summary = result.summary()
        assert "bench" in summary
        assert "1.500s" in summary


class TestTimer:
    """Tests for the Timer context manager."""

    def test_measures_time(self) -> None:
        import time

        with Timer("sleep_test") as t:
            time.sleep(0.05)
        assert t.elapsed > 0.04
        assert t.result.elapsed_seconds > 0.04

    def test_name_preserved(self) -> None:
        with Timer("my_op") as t:
            pass
        assert t.result.name == "my_op"


class TestDataLoaderProfile:
    """Tests for the DataLoaderProfile dataclass."""

    def test_avg_throughput(self) -> None:
        profile = DataLoaderProfile(
            config_name="test",
            results=[
                TimingResult(name="e0", elapsed_seconds=1.0, samples_processed=100),
                TimingResult(name="e1", elapsed_seconds=1.0, samples_processed=200),
            ],
        )
        assert profile.avg_throughput == pytest.approx(150.0)

    def test_empty_results(self) -> None:
        profile = DataLoaderProfile(config_name="empty")
        assert profile.avg_throughput == 0.0
        assert profile.avg_batch_time_ms == 0.0


class _SimpleDataset(Dataset):
    """Minimal dataset for profiling tests."""

    def __init__(self, size: int = 20) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(3, 32, 32)


class TestProfileDataloader:
    """Tests for the profile_dataloader function."""

    def test_returns_profile(self) -> None:
        loader = DataLoader(_SimpleDataset(20), batch_size=5, num_workers=0)
        profile = profile_dataloader(loader, num_epochs=2, name="bench")
        assert len(profile.results) == 2
        assert profile.config_name == "bench"

    def test_samples_counted(self) -> None:
        loader = DataLoader(_SimpleDataset(20), batch_size=5, num_workers=0)
        profile = profile_dataloader(loader, num_epochs=1, name="count_test")
        assert profile.results[0].samples_processed == 20

    def test_iterations_counted(self) -> None:
        loader = DataLoader(_SimpleDataset(20), batch_size=5, num_workers=0)
        profile = profile_dataloader(loader, num_epochs=1, name="iter_test")
        assert profile.results[0].iterations == 4  # 20 / 5

    def test_dict_batch_handling(self) -> None:
        """Test profiling with dict-returning datasets."""

        class DictDataset(Dataset):
            def __len__(self) -> int:
                return 10

            def __getitem__(self, idx: int) -> dict:
                return {"x": torch.randn(3, 32, 32), "y": torch.tensor(0)}

        loader = DataLoader(DictDataset(), batch_size=5, num_workers=0)
        profile = profile_dataloader(loader, num_epochs=1, name="dict_test")
        assert profile.results[0].samples_processed == 10
