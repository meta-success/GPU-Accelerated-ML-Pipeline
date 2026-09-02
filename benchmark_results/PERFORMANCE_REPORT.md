# GPU-Accelerated ML Pipeline — Performance Report

Generated: **2026-09-02 20:26 UTC**

## Environment

| Key | Value |
| --- | --- |
| Host | DESKTOP-VOMUVJU |
| Python | 3.11.0 |
| CUDA backend | cpu |
| GPU | cpu |
| Dataset | synthetic |
| Train samples | 400 |
| Epochs | 3 |
| Batch size | 64 |

## Validation accuracy

| Framework | Val accuracy |
| --- | --- |
| pytorch | 0.095 |
| tensorflow | 0.110 |
| jax | 0.065 |
| onnx | 0.095 |

## Benchmarks

| Framework | Device | Phase | Batch | Latency (ms) | img/s | Memory (MB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preprocess_cpu | cpu | preprocess_cpu | 64 | 3.332 | 19209.4 | 530.0 | NumPy brightness + LCG salt-pepper |
| preprocess_gpu | cpu | preprocess_gpu | 64 | 1.136 | 56352.9 | 540.0 | backend=cpu |
| pytorch | cpu | inference | 64 | 23.048 | 2776.8 | 699.6 | CIFARCNN forward |
| tensorflow | cpu | inference | 64 | 71.613 | 893.7 | 948.8 | Keras Model.predict |
| jax | cpu | inference | 64 | 13.893 | 4606.6 | 1090.1 | JIT cnn_forward |
| onnxruntime | cpu | inference | 64 | 12.981 | 4930.3 | 1091.1 | CPUExecutionProvider |

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
