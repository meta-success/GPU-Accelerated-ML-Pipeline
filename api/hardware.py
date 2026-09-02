"""Detect NVIDIA GPUs and CUDA availability for the console UI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


def find_nvidia_smi() -> Optional[Path]:
    which = shutil.which("nvidia-smi")
    if which:
        return Path(which)
    for candidate in (
        Path(r"C:\Windows\System32\nvidia-smi.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _query_nvidia_smi(smi: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            str(smi),
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    line = (proc.stdout or "").strip().splitlines()
    if proc.returncode != 0 or not line:
        return {"smi_ok": False, "smi_error": (proc.stderr or proc.stdout or "").strip()[:400]}
    name, driver, memory = [part.strip() for part in line[0].split(",", 2)]
    cuda = None
    banner = subprocess.run(
        [str(smi)],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    for row in (banner.stdout or "").splitlines():
        if "CUDA Version" in row:
            cuda = row.split("CUDA Version:")[-1].split()[0]
            break
    return {
        "smi_ok": True,
        "name": name,
        "driver": driver,
        "memory_mb": float(memory) if memory else None,
        "cuda_driver": cuda,
        "smi_path": str(smi),
    }


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "smi_ok": False,
        "name": None,
        "driver": None,
        "memory_mb": None,
        "cuda_driver": None,
        "smi_path": None,
        "torch_cuda": False,
        "torch_device": "cpu",
        "recommended_device": "cpu",
    }
    smi = find_nvidia_smi()
    if smi is not None:
        info.update(_query_nvidia_smi(smi))
    try:
        import torch

        info["torch_cuda"] = bool(torch.cuda.is_available())
        if info["torch_cuda"]:
            info["torch_device"] = torch.cuda.get_device_name(0)
            info["recommended_device"] = "cuda"
        elif info.get("smi_ok"):
            info["recommended_device"] = "cpu"
            info["torch_note"] = (
                "NVIDIA GPU is visible to the driver, but this Python build cannot use CUDA. "
                "Install a CUDA PyTorch wheel and/or update the NVIDIA driver."
            )
    except Exception as exc:
        info["torch_note"] = f"PyTorch not importable: {exc}"
    if info.get("smi_ok") and info["name"] is None:
        info["name"] = info.get("torch_device")
    return info
