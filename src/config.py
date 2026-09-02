"""Shared configuration for the GPU-accelerated ML pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODEL_DIR = ROOT_DIR / "models"
BENCHMARK_DIR = ROOT_DIR / "benchmark_results"

NUM_CLASSES = 10
IMAGE_SIZE = 32
NUM_CHANNELS = 3
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DeviceName = Literal["auto", "cpu", "cuda"]
FrameworkName = Literal["pytorch", "tensorflow", "jax", "onnx"]


@dataclass
class PipelineConfig:
    """Runtime knobs for training, inference, and benchmarking."""

    data_dir: Path = DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    model_dir: Path = MODEL_DIR
    output_dir: Path = BENCHMARK_DIR
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_samples: int = 4000
    num_workers: int = 0
    seed: int = 42
    device: DeviceName = "auto"
    frameworks: tuple[str, ...] = ("pytorch", "tensorflow", "jax", "onnx")
    brightness_factor: float = 1.15
    noise_prob: float = 0.02
    benchmark_iterations: int = 50
    warmup_iterations: int = 10
    skip_train: bool = False
    skip_onnx: bool = False
    use_synthetic: bool = False
    log_level: str = "INFO"

    extra: dict = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.checkpoint_dir, self.model_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


def configure_process_env() -> None:
    """Reduce GPU memory fights when PyTorch, TensorFlow, and JAX share a device."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
