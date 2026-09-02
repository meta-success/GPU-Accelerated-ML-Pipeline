"""CIFAR-10 (or synthetic) data loading shared by all frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from src.config import CIFAR10_MEAN, CIFAR10_STD, IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, PipelineConfig
from src.logging_utils import get_logger

logger = get_logger("data")


@dataclass
class ArrayDataset:
    """Framework-agnostic image arrays in NHWC float32 [0, 1] and int32 labels."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def num_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def num_test(self) -> int:
        return int(self.x_test.shape[0])


def set_seed(seed: int) -> None:
    """Seed NumPy (and PyTorch / TF when they are imported)."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _normalize(images: np.ndarray) -> np.ndarray:
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    return ((images - mean) / std).astype(np.float32)


def denormalize(images: np.ndarray) -> np.ndarray:
    """Undo CIFAR-10 normalization for visualization / CUDA augmentations in [0, 1]."""
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    return np.clip(images * std + mean, 0.0, 1.0).astype(np.float32)


def to_nchw(images: np.ndarray) -> np.ndarray:
    return np.transpose(images, (0, 3, 1, 2))


def to_nhwc(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == NUM_CHANNELS:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def synthetic_cifar(n_train: int, n_test: int, seed: int = 42) -> ArrayDataset:
    """Deterministic fake CIFAR-10 tensors so tests and demos work offline."""
    rng = np.random.default_rng(seed)
    x_train = rng.random((n_train, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    x_test = rng.random((n_test, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    y_train = rng.integers(0, NUM_CLASSES, size=(n_train,), dtype=np.int32)
    y_test = rng.integers(0, NUM_CLASSES, size=(n_test,), dtype=np.int32)
    return ArrayDataset(x_train, y_train, x_test, y_test)


def _from_torchvision(data_dir: Path, max_samples: int) -> ArrayDataset:
    from torchvision import datasets, transforms

    data_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=to_tensor)
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=to_tensor)

    def dump(dataset, limit: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
        n = len(dataset) if limit is None else min(limit, len(dataset))
        images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
        labels = np.empty((n,), dtype=np.int32)
        for i in range(n):
            img, label = dataset[i]
            images[i] = np.transpose(img.numpy(), (1, 2, 0))
            labels[i] = int(label)
        return images, labels

    x_train, y_train = dump(train_set, max_samples)
    test_limit = max(500, min(2000, max_samples // 2)) if max_samples else None
    x_test, y_test = dump(test_set, test_limit)
    return ArrayDataset(x_train, y_train, x_test, y_test)


def load_dataset(config: PipelineConfig) -> ArrayDataset:
    """Load CIFAR-10, falling back to synthetic data if download is unavailable."""
    set_seed(config.seed)
    if config.use_synthetic:
        n_test = max(200, config.max_samples // 5)
        logger.info("Using synthetic CIFAR-shaped data (%s train / %s test)", config.max_samples, n_test)
        return synthetic_cifar(config.max_samples, n_test, seed=config.seed)

    try:
        logger.info("Downloading / loading CIFAR-10 into %s", config.data_dir)
        dataset = _from_torchvision(config.data_dir, config.max_samples)
        logger.info("Loaded CIFAR-10: %s train, %s test", dataset.num_train, dataset.num_test)
        return dataset
    except Exception as exc:
        logger.warning("CIFAR-10 download failed (%s). Falling back to synthetic data.", exc)
        n_test = max(200, config.max_samples // 5)
        return synthetic_cifar(config.max_samples, n_test, seed=config.seed)


def iterate_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = images.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        sl = indices[start : start + batch_size]
        yield images[sl], labels[sl]


def normalized_copy(dataset: ArrayDataset) -> ArrayDataset:
    return ArrayDataset(
        x_train=_normalize(dataset.x_train),
        y_train=dataset.y_train.copy(),
        x_test=_normalize(dataset.x_test),
        y_test=dataset.y_test.copy(),
    )
