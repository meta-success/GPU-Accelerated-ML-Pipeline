# GPU-Accelerated ML Pipeline — Performance Report

Generated: **2026-09-02 19:27 UTC**

## Environment

| Key | Value |
| --- | --- |
| Host | DESKTOP-VOMUVJU |
| Python | 3.11.0 |
| CUDA backend | cpu |
| GPU | cpu |
| Dataset | CIFAR-10 |
| Train samples | 4000 |
| Epochs | 1 |
| Batch size | 64 |

## Validation accuracy

| Framework | Val accuracy |
| --- | --- |
| pytorch | 0.248 |
| tensorflow | 0.153 |
| jax | 0.156 |
| onnx | 0.248 |

## Benchmarks

| Framework | Device | Phase | Batch | Latency (ms) | img/s | Memory (MB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preprocess_cpu | cpu | preprocess_cpu | 64 | 4.097 | 15621.7 | 336.1 | NumPy brightness + LCG salt-pepper |
| preprocess_gpu | cpu | preprocess_gpu | 64 | 1.181 | 54190.4 | 346.3 | backend=cpu |
| pytorch | cpu | inference | 64 | 22.403 | 2856.8 | 527.1 | CIFARCNN forward |
| tensorflow | cpu | inference | 64 | 75.214 | 850.9 | 1097.0 | Keras Model.predict |
| jax | cpu | inference | 64 | 12.220 | 5237.1 | 1243.8 | JIT cnn_forward |
| onnxruntime | cpu | inference | 64 | 14.583 | 4388.7 | 1253.2 | CPUExecutionProvider |

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
