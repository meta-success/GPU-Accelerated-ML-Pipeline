# Reading pipeline benchmarks

After a run, the console can download these files, or you can copy them from `benchmark_results/`:

| File | Contents |
| --- | --- |
| `results.csv` | One row per framework / device / phase |
| `results.json` | Same numbers plus run metadata |
| `throughput.png` | Throughput chart |
| `PERFORMANCE_REPORT.md` | Markdown summary generated on your machine |

`sample_results.csv` is a **worked example**, not a measurement of the computer you are using. Treat the console table as sample data until `results.csv` exists.

## What the columns mean

- **Latency (ms)** — average time for one batched call after warmup (`cuda.synchronize` / `block_until_ready` where it applies).
- **Throughput (img/s)** — `batch_size / latency`.
- **Memory (MB)** — peak allocated bytes from `torch.cuda.max_memory_allocated` when CUDA is active.

Typical pattern on a mid-range NVIDIA GPU (not a guarantee):

- CUDA preprocess: large GPU vs CPU gap on a full batch
- Training: GPU wall-clock often a few times faster than CPU
- Inference: ONNX Runtime / TensorRT is often the fastest serving path
- JAX: first JIT call is compile time; ignore it and read the warmed loop

## Accuracy

A 4k-image, few-epoch run is for wiring and speed, not leaderboard accuracy. Validation often lands roughly in the 0.25–0.55 range. For a stronger model:

```bash
python -m src --max-samples 50000 --epochs 20
```

## Memory when three frameworks share one GPU

The process sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and TensorFlow memory growth so PyTorch, TensorFlow, and JAX are less likely to OOM each other. If you still OOM, run fewer stages (for example PyTorch + ONNX only).

## Reproduce from the CLI

```bash
python -m src --epochs 3 --batch-size 128 --max-samples 4000
```

Or open the web console (`python -m api`) and use **Run pipeline** → **all results (.zip)**.
