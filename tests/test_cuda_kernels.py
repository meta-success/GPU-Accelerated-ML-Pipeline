from __future__ import annotations

import numpy as np

from src.cuda_kernels import (
    apply_brightness_cpu,
    apply_salt_pepper_cpu,
    augment_batch,
    cuda_status,
)


def test_brightness_cpu_clips_and_scales():
    images = np.full((2, 4, 4, 3), 0.5, dtype=np.float32)
    out = apply_brightness_cpu(images, 3.0)
    assert out.max() <= 1.0 + 1e-6
    assert np.allclose(out, 1.0)


def test_salt_pepper_cpu_is_deterministic():
    images = np.full((1, 8, 8, 3), 0.4, dtype=np.float32)
    a = apply_salt_pepper_cpu(images, 0.2, seed=7)
    b = apply_salt_pepper_cpu(images, 0.2, seed=7)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isclose(a, 0.0) | np.isclose(a, 0.4) | np.isclose(a, 1.0))


def test_augment_batch_keeps_shape(unit_batch):
    out = augment_batch(unit_batch, brightness_factor=1.1, noise_prob=0.05, seed=1)
    arr = np.asarray(out.detach().cpu() if hasattr(out, "detach") else out)
    assert arr.shape == unit_batch.shape
    assert arr.dtype == np.float32 or arr.dtype == np.float64
    assert arr.min() >= -1e-5
    assert arr.max() <= 1.0 + 1e-5


def test_cuda_status_reports_backend():
    status = cuda_status()
    assert status.backend in {"numba_cuda", "torch_cuda", "cpu"}
    assert isinstance(status.device_name, str)
