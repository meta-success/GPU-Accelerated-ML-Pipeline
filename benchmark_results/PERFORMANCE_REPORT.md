# GPU-Accelerated ML Pipeline — Performance Report

Generated: **2026-09-02 16:28 UTC**

## Environment

| Key | Value |
| --- | --- |
| Host | DESKTOP-VOMUVJU |
| Python | 3.11.0 |
| CUDA backend | cpu |
| GPU | cpu |
| Dataset | synthetic |
| Train samples | 256 |
| Epochs | 1 |
| Batch size | 64 |

## Validation accuracy

| Framework | Val accuracy |
| --- | --- |
| pytorch | 0.115 |
| tensorflow | 0.100 |
| jax | 0.125 |
| onnx | 0.115 |

## Benchmarks

| Framework | Device | Phase | Batch | Latency (ms) | img/s | Memory (MB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preprocess_cpu | cpu | preprocess_cpu | 64 | 4.535 | 14112.0 | 527.6 | NumPy brightness + LCG salt-pepper |
| preprocess_gpu | cpu | preprocess_gpu | 64 | 1.145 | 55888.4 | 537.5 | backend=cpu |
| pytorch | cpu | inference | 64 | 23.272 | 2750.1 | 708.1 | CIFARCNN forward |
| tensorflow | cpu | inference | 64 | 64.403 | 993.7 | 914.9 | Keras Model.predict |
| jax | cpu | inference | 64 | 10.720 | 5970.3 | 1048.2 | JIT cnn_forward |
| onnxruntime | cpu | inference | 64 | 14.582 | 4389.0 | 1023.7 | CPUExecutionProvider |

## CPU vs GPU speedup

| Framework | Phase | CPU ms | GPU ms | Speedup | CPU img/s | GPU img/s |
| --- | --- | --- | --- | --- | --- | --- |
| _(no rows)_ |

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
