"""CPU vs GPU and cross-framework benchmarking with CSV / JSON / plot export."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.logging_utils import get_logger

logger = get_logger("benchmark")


@dataclass
class BenchmarkRow:
    framework: str
    device: str
    phase: str
    batch_size: int
    iterations: int
    latency_ms: float
    throughput_img_s: float
    memory_mb: float
    notes: str = ""


@dataclass
class BenchmarkSuite:
    rows: list[BenchmarkRow] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, row: BenchmarkRow) -> None:
        self.rows.append(row)
        logger.info(
            "%s/%s/%s | %.3f ms | %.1f img/s | %.1f MB | %s",
            row.framework,
            row.device,
            row.phase,
            row.latency_ms,
            row.throughput_img_s,
            row.memory_mb,
            row.notes,
        )


def _sync_torch(device: str) -> None:
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def gpu_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        pass
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except Exception:
        return 0.0


def reset_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def time_callable(
    fn: Callable[[], Any],
    warmup: int = 10,
    iterations: int = 50,
    sync_device: str = "cpu",
) -> float:
    """Return average seconds per call after warmup."""
    for _ in range(max(warmup, 0)):
        fn()
    _sync_torch(sync_device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync_torch(sync_device)
    return (time.perf_counter() - start) / max(iterations, 1)


def benchmark_transform(
    name: str,
    fn: Callable[[], Any],
    images_per_call: int,
    device: str,
    warmup: int,
    iterations: int,
    notes: str = "",
) -> BenchmarkRow:
    reset_gpu_memory()
    avg_s = time_callable(fn, warmup=warmup, iterations=iterations, sync_device=device)
    mem = gpu_memory_mb()
    return BenchmarkRow(
        framework="cuda_kernels" if "cuda" in name.lower() or "numba" in name.lower() else name,
        device=device,
        phase=name,
        batch_size=images_per_call,
        iterations=iterations,
        latency_ms=avg_s * 1000.0,
        throughput_img_s=images_per_call / avg_s if avg_s > 0 else 0.0,
        memory_mb=mem,
        notes=notes,
    )


def benchmark_inference(
    framework: str,
    device: str,
    fn: Callable[[], Any],
    batch_size: int,
    warmup: int,
    iterations: int,
    notes: str = "",
) -> BenchmarkRow:
    reset_gpu_memory()
    avg_s = time_callable(fn, warmup=warmup, iterations=iterations, sync_device=device if device == "cuda" else "cpu")
    mem = gpu_memory_mb()
    return BenchmarkRow(
        framework=framework,
        device=device,
        phase="inference",
        batch_size=batch_size,
        iterations=iterations,
        latency_ms=avg_s * 1000.0,
        throughput_img_s=batch_size / avg_s if avg_s > 0 else 0.0,
        memory_mb=mem,
        notes=notes,
    )


def write_csv(rows: list[BenchmarkRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    logger.info("Wrote benchmark CSV -> %s", path)
    return path


def write_json(suite: BenchmarkSuite, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": suite.meta, "rows": [asdict(r) for r in suite.rows]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote benchmark JSON -> %s", path)
    return path


def plot_results(rows: list[BenchmarkRow], path: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable, skipping plot: %s", exc)
        return None

    inference = [r for r in rows if r.phase == "inference"]
    if not inference:
        inference = rows
    labels = [f"{r.framework}\n{r.device}" for r in inference]
    values = [r.throughput_img_s for r in inference]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#3b82f6")
    ax.set_ylabel("Throughput (images / second)")
    ax.set_title("GPU-Accelerated ML Pipeline — Inference Throughput")
    ax.tick_params(axis="x", labelrotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Wrote throughput plot -> %s", path)
    return path


def speedup_table(rows: list[BenchmarkRow]) -> list[dict[str, Any]]:
    """Pair CPU/GPU rows that share framework + phase."""
    by_key: dict[tuple[str, str], dict[str, BenchmarkRow]] = {}
    for row in rows:
        by_key.setdefault((row.framework, row.phase), {})[row.device] = row

    out = []
    for (framework, phase), devices in sorted(by_key.items()):
        cpu = devices.get("cpu")
        gpu = devices.get("cuda") or devices.get("gpu")
        if cpu and gpu and cpu.latency_ms > 0:
            out.append(
                {
                    "framework": framework,
                    "phase": phase,
                    "cpu_ms": cpu.latency_ms,
                    "gpu_ms": gpu.latency_ms,
                    "speedup": cpu.latency_ms / gpu.latency_ms if gpu.latency_ms else 0.0,
                    "cpu_img_s": cpu.throughput_img_s,
                    "gpu_img_s": gpu.throughput_img_s,
                }
            )
    return out
