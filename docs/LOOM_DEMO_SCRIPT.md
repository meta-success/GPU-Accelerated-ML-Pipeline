# Loom demo script — GPU-Accelerated ML Pipeline

**Target length:** 8–12 minutes  
**Goal:** Show CUDA, JAX, PyTorch, TensorFlow, and ONNX actually running, plus CPU vs GPU numbers.

Record at 1080p. Keep a terminal, `nvidia-smi -l 1` (or Task Manager → GPU), and the repo open.

---

## 0. Before you hit record (2 min, off-camera)

```bash
nvidia-smi
python scripts/check_gpu.py
python -m src --synthetic --epochs 1 --max-samples 256 --benchmark-iterations 5 --warmup 1
```

Confirm at least PyTorch CUDA is `True`. If TensorFlow/JAX stay on CPU, say so on camera and still run them — honesty beats a fake GPU.

Have a second terminal ready:

```bash
nvidia-smi -l 1
```

---

## 1. Hook (30 s)

> “This repo is one CIFAR-10 pipeline that hits five REWORK skills: custom CUDA kernels, JAX JIT/vmap/pmap, PyTorch GPU training, TensorFlow GPU training, and ONNX Runtime inference, with a CPU vs GPU benchmark at the end.”

Show `README.md` architecture diagram (scroll, don’t read every line).

---

## 2. Environment (1 min)

- `nvidia-smi` — driver, GPU name, memory
- `python scripts/check_gpu.py` — read out backend lines
- Optional: `nvcc --version`

---

## 3. CUDA kernels (2 min)

Open `src/cuda_kernels.py`. Point at:

1. `brightness_kernel` / `salt_pepper_kernel` (`@cuda.jit`)
2. `torch_to_cupy` / `cupy_to_torch` (DLPack)
3. CPU fallback (so reviewers know you handled missing CUDA)

In a Python REPL or notebook cell:

```python
import numpy as np, time
from src.cuda_kernels import apply_brightness_cpu, augment_batch, log_cuda_banner
log_cuda_banner()
x = np.random.rand(256, 32, 32, 3).astype("float32")
t0 = time.perf_counter(); apply_brightness_cpu(x, 1.2); print("cpu", time.perf_counter()-t0)
t0 = time.perf_counter(); augment_batch(x, 1.2, 0.02, 0); print("gpu", time.perf_counter()-t0)
```

Glance at `nvidia-smi` while the GPU call runs (memory / util spike).

---

## 4. Train + export (4–5 min)

```bash
python -m src --epochs 2 --batch-size 128 --max-samples 4000 --log-level INFO
```

Narrate as logs appear:

| Log line | What to say |
| --- | --- |
| `CUDA status \| backend=numba_cuda` | Custom kernels (or “PyTorch CUDA fallback if Numba isn’t linking”) |
| `PyTorch training on cuda` | Model + optimizer on GPU, checkpoint path |
| `TensorFlow GPUs:` | Keras residual CNN |
| `JAX backend=gpu` + `pmap smoke-test` | JIT training, vmap, pmap even on 1 GPU |
| `Exported ONNX` + `providers=` | Checker passed; CUDA or TensorRT EP |

If a framework fails, keep recording, skip to the next, mention it. Partial GPU evidence is better than cutting the video.

---

## 5. Results (2 min)

Open:

- `benchmark_results/results.csv`
- `benchmark_results/PERFORMANCE_REPORT.md`
- `benchmark_results/throughput.png` if generated

Call out **one speedup number** (e.g. “PyTorch inference 6.4× vs CPU on this 128-image batch”) and **ONNX vs eager PyTorch**.

Optional API (30 s):

```bash
uvicorn api.app:app --port 8000
# browser: http://127.0.0.1:8000/docs  → POST /predict
```

---

## 6. Close (20 s)

> “Repo README has setup, Docker, and tests. The CSV and this Loom are the REWORK evidence pack.”

Show GitHub repo URL on screen.

---

## Talking points if you freeze

- Why Numba instead of raw `nvcc`: Python-native kernels, still compiled for the GPU, easy DLPack into PyTorch.
- Why ONNX: one export, many runtimes (ORT, TensorRT, future Triton).
- Why a train *subset*: 4k images × 2 epochs finishes inside a Loom; architecture is the same as a 50k run (`--max-samples 50000`).
