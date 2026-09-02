"""Shared fixtures for unit and integration tests (no GPU required)."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data import synthetic_cifar


@pytest.fixture
def tiny_config(tmp_path) -> PipelineConfig:
    cfg = PipelineConfig(
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "benchmarks",
        epochs=1,
        batch_size=8,
        max_samples=32,
        seed=0,
        device="cpu",
        frameworks=("pytorch",),
        benchmark_iterations=2,
        warmup_iterations=1,
        skip_train=False,
        use_synthetic=True,
        log_level="WARNING",
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def tiny_dataset():
    return synthetic_cifar(n_train=32, n_test=16, seed=0)


@pytest.fixture
def unit_batch() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((4, 32, 32, 3), dtype=np.float32)
