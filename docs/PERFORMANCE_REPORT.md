# Performance report (template)

This file is the **narrative** you attach to REWORK. After you run the pipeline on a real GPU, replace the tables with the generated copy at `benchmark_results/PERFORMANCE_REPORT.md`.

## Machine (fill in)

| Item | Value |
| --- | --- |
| GPU | e.g. NVIDIA RTX 4070 Laptop |
| Driver / CUDA | from `nvidia-smi` |
| OS | Windows 11 + WSL2 Ubuntu 22.04 / Docker |
| Python | 3.10.x |
| PyTorch | 2.x + cu118 |
| TensorFlow | 2.15 |
| JAX | 0.4.x cuda |
| ORT providers | TensorrtExecutionProvider, CUDAExecutionProvider |

## Pipeline settings

- Dataset: CIFAR-10 subset (`--max-samples 4000`)
- Epochs: 3
- Batch size: 128
- Augmentation: brightness 1.15, salt-and-pepper p=0.02 (Numba CUDA)

## Example numbers

These rows match `benchmark_results/sample_results.csv` and are **not** a substitute for your run.

| Framework | Device | Phase | Latency (ms) | img/s |
| --- | --- | --- | --- | --- |
| CUDA kernels | CPU | preprocess | 12.4 | 10.3k |
| CUDA kernels | GPU | preprocess | 0.21 | 610k |
| PyTorch | CPU | inference | 18.2 | 7.0k |
| PyTorch | GPU | inference | 2.85 | 44.9k |
| TensorFlow | GPU | inference | 4.10 | 31.2k |
| JAX | GPU | inference | 1.94 | 66.0k |
| ONNX Runtime | GPU | inference | 1.21 | 106k |

### Speedups implied by the sample

- Preprocess GPU vs CPU: **~59×** (matches the “batch processing” talking point when kernels land on a real GPU)
- PyTorch inference GPU vs CPU: **~6.4×**
- ONNX GPU vs PyTorch GPU: **~2.4×** on this toy CNN (TensorRT EP helps more on larger models)

## Accuracy (expect this to be modest)

A 4k-image, 3-epoch demo is for **skill evidence**, not SOTA. Val accuracy often lands in the 0.25–0.55 range depending on seed and subset. Train `--max-samples 50000 --epochs 20` if you want a stronger accuracy slide.

## Memory

Peak allocated bytes come from `torch.cuda.max_memory_allocated` between stages. JAX/TF may allocate extra pools; the pipeline sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and TF memory growth to keep three frameworks from killing each other.

## How to regenerate

```bash
python -m src --epochs 3 --batch-size 128 --max-samples 4000
```

Commit `benchmark_results/results.csv`, `throughput.png`, and `PERFORMANCE_REPORT.md` with the Loom.
