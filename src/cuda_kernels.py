"""Custom CUDA kernels for GPU image preprocessing (Numba) with CPU/PyTorch fallbacks.

Two kernels are provided:

* brightness — per-pixel multiply + clamp to [0, 1]
* salt-and-pepper — in-place impulse noise with a cheap LCG RNG

When Numba CUDA (and optionally CuPy / DLPack) is available the kernels run on
the GPU. Otherwise the same math is executed with vectorized PyTorch (GPU or
CPU) so the rest of the pipeline still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.logging_utils import get_logger

logger = get_logger("cuda")

try:
    from numba import cuda as numba_cuda

    _NUMBA_CUDA_AVAILABLE = numba_cuda.is_available()
except Exception:
    numba_cuda = None  # type: ignore[assignment]
    _NUMBA_CUDA_AVAILABLE = False

try:
    import torch
    from torch.utils.dlpack import from_dlpack as torch_from_dlpack
    from torch.utils.dlpack import to_dlpack as torch_to_dlpack

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    torch_from_dlpack = None
    torch_to_dlpack = None
    _TORCH_AVAILABLE = False

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


THREADS_PER_BLOCK = 256


if numba_cuda is not None:

    @numba_cuda.jit
    def brightness_kernel(images, brightness_factor, n):
        """In-place brightness: pixel = min(pixel * factor, 1.0)."""
        idx = numba_cuda.grid(1)
        if idx < n:
            val = images[idx] * brightness_factor
            images[idx] = val if val < 1.0 else 1.0

    @numba_cuda.jit
    def salt_pepper_kernel(images, prob, seed, n):
        """In-place salt-and-pepper noise using a per-thread linear congruential RNG."""
        idx = numba_cuda.grid(1)
        if idx < n:
            res = (idx * 1103515245 + 12345 + seed) & 0x7FFFFFFF
            random_val = res / 0x7FFFFFFF
            half = prob * 0.5
            if random_val < half:
                images[idx] = 0.0
            elif random_val < prob:
                images[idx] = 1.0

else:

    def brightness_kernel(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError("Numba CUDA is not available")

    def salt_pepper_kernel(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError("Numba CUDA is not available")


@dataclass
class CudaStatus:
    numba_cuda: bool
    cupy: bool
    torch_cuda: bool
    device_name: str
    backend: str


def cuda_status() -> CudaStatus:
    torch_cuda = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
    device_name = "cpu"
    if torch_cuda:
        device_name = torch.cuda.get_device_name(0)
    elif _NUMBA_CUDA_AVAILABLE:
        try:
            device_name = str(numba_cuda.get_current_device().name)
        except Exception:
            device_name = "cuda"

    if _NUMBA_CUDA_AVAILABLE:
        backend = "numba_cuda"
    elif torch_cuda:
        backend = "torch_cuda"
    else:
        backend = "cpu"

    return CudaStatus(
        numba_cuda=_NUMBA_CUDA_AVAILABLE,
        cupy=_CUPY_AVAILABLE,
        torch_cuda=torch_cuda,
        device_name=device_name,
        backend=backend,
    )


def _grid(n: int) -> tuple[int, int]:
    blocks = (int(n) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    return max(blocks, 1), THREADS_PER_BLOCK


def torch_to_cupy(tensor):
    """Zero-copy PyTorch CUDA tensor -> CuPy array via DLPack."""
    if not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not installed")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    try:
        return cp.from_dlpack(tensor)
    except Exception:
        return cp.from_dlpack(torch_to_dlpack(tensor))


def cupy_to_torch(array):
    """Zero-copy CuPy array -> PyTorch tensor via DLPack."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    try:
        return torch.from_dlpack(array)
    except Exception:
        return torch_from_dlpack(array.toDlpack())


def _as_torch(images, device: Optional[str] = None):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor preprocessing")
    if isinstance(images, torch.Tensor):
        tensor = images
    else:
        tensor = torch.as_tensor(np.asarray(images, dtype=np.float32))
    tensor = tensor.float().contiguous()
    if device:
        tensor = tensor.to(device)
    return tensor


def apply_brightness_cpu(images: np.ndarray, brightness_factor: float) -> np.ndarray:
    return np.clip(images.astype(np.float32) * float(brightness_factor), 0.0, 1.0)


def apply_salt_pepper_cpu(images: np.ndarray, prob: float, seed: int = 0) -> np.ndarray:
    """Vectorized CPU salt-and-pepper that matches the kernel's LCG in spirit."""
    flat = images.astype(np.float32).reshape(-1).copy()
    n = flat.size
    idx = np.arange(n, dtype=np.int64)
    res = (idx * 1103515245 + 12345 + int(seed)) & 0x7FFFFFFF
    random_val = res.astype(np.float64) / 0x7FFFFFFF
    half = float(prob) * 0.5
    flat[random_val < half] = 0.0
    mask_salt = (random_val >= half) & (random_val < float(prob))
    flat[mask_salt] = 1.0
    return flat.reshape(images.shape)


def _launch_numba(flat_cuda_array, kernel, *scalar_args) -> None:
    n = int(flat_cuda_array.size)
    blocks, threads = _grid(n)
    kernel[blocks, threads](flat_cuda_array, *scalar_args, n)
    if numba_cuda is not None:
        numba_cuda.synchronize()


def _apply_numba_on_cupy(array, brightness_factor: Optional[float], noise_prob: Optional[float], seed: int):
    flat = array.ravel()
    if brightness_factor is not None:
        _launch_numba(flat, brightness_kernel, np.float32(brightness_factor))
    if noise_prob is not None and noise_prob > 0:
        _launch_numba(flat, salt_pepper_kernel, np.float32(noise_prob), np.int32(seed))
    return array


def apply_brightness(images, brightness_factor: float = 1.15):
    """Apply brightness on GPU when possible. Accepts ndarray or torch.Tensor."""
    status = cuda_status()
    if status.backend == "numba_cuda" and _CUPY_AVAILABLE and _TORCH_AVAILABLE:
        tensor = _as_torch(images)
        was_cpu = tensor.device.type == "cpu"
        if was_cpu:
            tensor = tensor.cuda()
        cupy_arr = torch_to_cupy(tensor)
        _apply_numba_on_cupy(cupy_arr, brightness_factor, None, 0)
        out = cupy_to_torch(cupy_arr).view_as(tensor)
        return out.cpu() if was_cpu else out

    if _TORCH_AVAILABLE:
        tensor = _as_torch(images)
        if status.torch_cuda and tensor.device.type != "cuda":
            tensor = tensor.cuda()
        return torch.clamp(tensor * float(brightness_factor), 0.0, 1.0)

    arr = np.asarray(images, dtype=np.float32)
    return apply_brightness_cpu(arr, brightness_factor)


def apply_salt_pepper(images, prob: float = 0.02, seed: int = 0):
    """Apply salt-and-pepper noise on GPU when possible."""
    status = cuda_status()
    if status.backend == "numba_cuda" and _CUPY_AVAILABLE and _TORCH_AVAILABLE:
        tensor = _as_torch(images)
        was_cpu = tensor.device.type == "cpu"
        if was_cpu:
            tensor = tensor.cuda()
        cupy_arr = torch_to_cupy(tensor)
        _apply_numba_on_cupy(cupy_arr, None, prob, seed)
        out = cupy_to_torch(cupy_arr).view_as(tensor)
        return out.cpu() if was_cpu else out

    if _TORCH_AVAILABLE:
        tensor = _as_torch(images)
        if status.torch_cuda and tensor.device.type != "cuda":
            tensor = tensor.cuda()
        n = tensor.numel()
        idx = torch.arange(n, device=tensor.device, dtype=torch.int64)
        res = (idx * 1103515245 + 12345 + int(seed)) & 0x7FFFFFFF
        random_val = res.float() / 0x7FFFFFFF
        half = float(prob) * 0.5
        flat = tensor.reshape(-1)
        flat = torch.where(random_val < half, torch.zeros_like(flat), flat)
        flat = torch.where((random_val >= half) & (random_val < float(prob)), torch.ones_like(flat), flat)
        return flat.view_as(tensor)

    return apply_salt_pepper_cpu(np.asarray(images, dtype=np.float32), prob, seed)


def augment_batch(
    images,
    brightness_factor: float = 1.15,
    noise_prob: float = 0.02,
    seed: int = 0,
):
    """Run brightness then salt-and-pepper. Returns the same type family as ``images``."""
    status = cuda_status()
    logger.debug(
        "Augmenting batch with backend=%s factor=%.2f noise=%.3f",
        status.backend,
        brightness_factor,
        noise_prob,
    )

    if status.backend == "numba_cuda" and _CUPY_AVAILABLE and _TORCH_AVAILABLE:
        tensor = _as_torch(images)
        was_cpu = tensor.device.type == "cpu"
        if was_cpu:
            tensor = tensor.cuda()
        cupy_arr = torch_to_cupy(tensor)
        _apply_numba_on_cupy(cupy_arr, brightness_factor, noise_prob, seed)
        out = cupy_to_torch(cupy_arr).view_as(tensor)
        return out.cpu() if was_cpu else out

    bright = apply_brightness(images, brightness_factor)
    return apply_salt_pepper(bright, noise_prob, seed)


def log_cuda_banner() -> CudaStatus:
    status = cuda_status()
    logger.info(
        "CUDA status | backend=%s | numba=%s | cupy=%s | torch_cuda=%s | device=%s",
        status.backend,
        status.numba_cuda,
        status.cupy,
        status.torch_cuda,
        status.device_name,
    )
    return status
