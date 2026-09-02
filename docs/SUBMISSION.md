# REWORK submission notes

Use this as a packing list, not as extra theory.

## Repo

- Push this project to a public GitHub repository.
- README Quick start must work from a clean clone (document WSL2/Docker if you are on Windows).
- Do **not** commit `.venv`, `data/`, or huge checkpoints. **Do** commit `src/`, `tests/`, `docker/`, `docs/`, and your **measured** `benchmark_results/results.csv`.

## Loom

Record with [docs/LOOM_DEMO_SCRIPT.md](LOOM_DEMO_SCRIPT.md). Upload unlisted Loom, paste URL in the REWORK evidence form.

## Performance report

Attach `benchmark_results/PERFORMANCE_REPORT.md` produced by `python -m src`. The template in this folder is only a placeholder.

## Skills mapping (paste into the form)

- **CUDA:** Numba `@cuda.jit` brightness and salt-and-pepper kernels; DLPack with PyTorch; CPU vs GPU preprocess bench.
- **JAX:** `jit` SGD step, `vmap` forward, `pmap` + `pmean`, XLA matmul demo.
- **PyTorch:** Residual CIFAR CNN, CUDA train loop, `state_dict` checkpoint, GPU inference bench.
- **TensorFlow:** Keras residual CNN, `set_memory_growth`, `fit` / `predict` on GPU when available.
- **ONNX:** `torch.onnx.export`, `onnx.checker`, ONNX Runtime providers (TensorRT EP when present).

## Optional live demo

`uvicorn api.app:app --port 8000` after a pipeline run that wrote `models/cifar_cnn.onnx`. Screenshot Swagger `/predict`.
