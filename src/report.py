"""Markdown + HTML performance report generated from benchmark rows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark import BenchmarkRow, speedup_table
from src.logging_utils import get_logger

logger = get_logger("report")


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body]) if rows else head + "\n" + sep + "\n| _(no rows)_ |"


def write_performance_report(
    rows: list[BenchmarkRow],
    meta: dict[str, Any],
    path: Path,
) -> Path:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    speedups = speedup_table(rows)

    bench_rows = [
        [
            r.framework,
            r.device,
            r.phase,
            r.batch_size,
            f"{r.latency_ms:.3f}",
            f"{r.throughput_img_s:.1f}",
            f"{r.memory_mb:.1f}",
            r.notes,
        ]
        for r in rows
    ]
    speed_rows = [
        [
            s["framework"],
            s["phase"],
            f"{s['cpu_ms']:.3f}",
            f"{s['gpu_ms']:.3f}",
            f"{s['speedup']:.2f}x",
            f"{s['cpu_img_s']:.1f}",
            f"{s['gpu_img_s']:.1f}",
        ]
        for s in speedups
    ]

    accuracy = meta.get("accuracy", {})
    acc_rows = [[k, f"{v:.3f}"] for k, v in accuracy.items()]

    md = f"""# GPU-Accelerated ML Pipeline — Performance Report

Generated: **{generated}**

## Environment

| Key | Value |
| --- | --- |
| Host | {meta.get("host", "unknown")} |
| Python | {meta.get("python", "unknown")} |
| CUDA backend | {meta.get("cuda_backend", "unknown")} |
| GPU | {meta.get("gpu_name", "none")} |
| Dataset | {meta.get("dataset", "CIFAR-10")} |
| Train samples | {meta.get("train_samples", "n/a")} |
| Epochs | {meta.get("epochs", "n/a")} |
| Batch size | {meta.get("batch_size", "n/a")} |

## Validation accuracy

{_md_table(["Framework", "Val accuracy"], acc_rows)}

## Benchmarks

{_md_table(["Framework", "Device", "Phase", "Batch", "Latency (ms)", "img/s", "Memory (MB)", "Notes"], bench_rows)}

## CPU vs GPU speedup

{_md_table(["Framework", "Phase", "CPU ms", "GPU ms", "Speedup", "CPU img/s", "GPU img/s"], speed_rows)}

## How to read this

- **Latency** is the average time for one batched call after warmup.
- **Throughput** is `batch_size / latency`.
- CUDA kernel numbers compare Numba CUDA (or the PyTorch CUDA fallback) against a NumPy CPU path using the same brightness + salt-and-pepper math.
- ONNX Runtime uses `TensorrtExecutionProvider` when present, otherwise `CUDAExecutionProvider`, otherwise CPU.
- Absolute speedups depend on GPU SKU, driver, batch size, and whether TensorFlow/JAX actually attached to CUDA (on native Windows they often stay on CPU — use Docker/WSL2).

## Reproduce

```bash
python -m src --epochs 3 --batch-size 128 --max-samples 4000
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    logger.info("Wrote performance report -> %s", path)
    return path
