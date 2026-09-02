"""End-to-end GPU-accelerated ML pipeline CLI.

Steps:
  1. Load CIFAR-10 (or synthetic data)
  2. CUDA-accelerated brightness + salt-and-pepper preprocessing
  3. Train PyTorch / TensorFlow / JAX CNNs
  4. Export PyTorch -> ONNX and run ONNX Runtime
  5. Benchmark CPU vs GPU and write CSV + report
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from src.benchmark import (
    BenchmarkSuite,
    benchmark_inference,
    benchmark_transform,
    plot_results,
    write_csv,
    write_json,
)
from src.config import PipelineConfig, configure_process_env
from src.cuda_kernels import (
    apply_brightness_cpu,
    apply_salt_pepper_cpu,
    augment_batch,
    cuda_status,
    log_cuda_banner,
)
from src.data import load_dataset, normalized_copy
from src.logging_utils import get_logger, setup_logging
from src.report import write_performance_report

logger = get_logger("pipeline")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated CIFAR-10 pipeline (CUDA, JAX, PyTorch, TensorFlow, ONNX).",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=4000, help="Train subset size for a fast demo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["pytorch", "tensorflow", "jax", "onnx"],
        choices=["pytorch", "tensorflow", "jax", "onnx"],
    )
    parser.add_argument("--brightness", type=float, default=1.15)
    parser.add_argument("--noise-prob", type=float, default=0.02)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Skip CIFAR-10 download; use random images.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        seed=args.seed,
        device=args.device,
        frameworks=tuple(args.frameworks),
        brightness_factor=args.brightness,
        noise_prob=args.noise_prob,
        benchmark_iterations=args.benchmark_iterations,
        warmup_iterations=args.warmup,
        skip_train=args.skip_train,
        skip_onnx=args.skip_onnx,
        use_synthetic=args.synthetic,
        log_level=args.log_level,
    )
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.checkpoint_dir = args.output_dir / "checkpoints"
        cfg.model_dir = args.output_dir / "models"
    return cfg


def _maybe_to_numpy(result) -> np.ndarray:
    try:
        import torch

        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(result, dtype=np.float32)


def run_preprocessing_benchmarks(raw_images: np.ndarray, cfg: PipelineConfig, suite: BenchmarkSuite) -> np.ndarray:
    """Benchmark CPU vs GPU augmentations and return the GPU-augmented training images."""
    status = log_cuda_banner()
    batch = raw_images[: min(cfg.batch_size, raw_images.shape[0])].copy()
    gpu_device = "cuda" if status.backend in {"numba_cuda", "torch_cuda"} else "cpu"

    def cpu_fn():
        out = apply_brightness_cpu(batch, cfg.brightness_factor)
        return apply_salt_pepper_cpu(out, cfg.noise_prob, cfg.seed)

    def gpu_fn():
        return augment_batch(batch, cfg.brightness_factor, cfg.noise_prob, cfg.seed)

    suite.add(
        benchmark_transform(
            "preprocess_cpu",
            cpu_fn,
            images_per_call=batch.shape[0],
            device="cpu",
            warmup=cfg.warmup_iterations,
            iterations=cfg.benchmark_iterations,
            notes="NumPy brightness + LCG salt-pepper",
        )
    )
    suite.add(
        benchmark_transform(
            "preprocess_gpu",
            gpu_fn,
            images_per_call=batch.shape[0],
            device=gpu_device,
            warmup=max(3, cfg.warmup_iterations // 2),
            iterations=cfg.benchmark_iterations,
            notes=f"backend={status.backend}",
        )
    )

    logger.info("Applying CUDA/GPU augmentation to the full training set (%s images)", raw_images.shape[0])
    augmented = _maybe_to_numpy(augment_batch(raw_images, cfg.brightness_factor, cfg.noise_prob, cfg.seed))
    return augmented.astype(np.float32)


def run_pytorch(dataset, cfg: PipelineConfig, suite: BenchmarkSuite) -> dict:
    from src.pytorch_model import CIFARCNN, infer_pytorch, resolve_torch_device, train_pytorch
    import torch

    result = None
    if not cfg.skip_train:
        result = train_pytorch(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, cfg)
        model = result.model
        device = resolve_torch_device(cfg.device)
    else:
        device = resolve_torch_device(cfg.device)
        model = CIFARCNN().to(device)
        logger.info("Skipping PyTorch training; using randomly initialized weights for inference benches")

    batch = dataset.x_test[: cfg.batch_size]
    from src.data import to_nchw

    nchw = torch.from_numpy(to_nchw(batch)).to(device)
    model.eval()

    def infer():
        with torch.no_grad():
            return model(nchw)

    suite.add(
        benchmark_inference(
            "pytorch",
            "cuda" if device.type == "cuda" else "cpu",
            infer,
            batch_size=batch.shape[0],
            warmup=cfg.warmup_iterations,
            iterations=cfg.benchmark_iterations,
            notes="CIFARCNN forward",
        )
    )

    if device.type == "cuda":
        cpu_model = model.cpu().eval()
        nchw_cpu = nchw.cpu()

        def infer_cpu():
            with torch.no_grad():
                return cpu_model(nchw_cpu)

        suite.add(
            benchmark_inference(
                "pytorch",
                "cpu",
                infer_cpu,
                batch_size=batch.shape[0],
                warmup=max(3, cfg.warmup_iterations // 2),
                iterations=max(10, cfg.benchmark_iterations // 2),
                notes="same weights on CPU",
            )
        )
        model.to(device)

    return {
        "model": model,
        "device": device,
        "accuracy": None if result is None else result.accuracy,
        "checkpoint": None if result is None else str(result.checkpoint_path),
    }


def run_tensorflow(dataset, cfg: PipelineConfig, suite: BenchmarkSuite) -> dict:
    from src.tensorflow_model import build_tf_model, infer_tensorflow, train_tensorflow

    if cfg.skip_train:
        model = build_tf_model(learning_rate=cfg.learning_rate)
        acc = None
        ckpt = None
    else:
        result = train_tensorflow(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, cfg)
        model = result.model
        acc = result.accuracy
        ckpt = str(result.checkpoint_path)

    batch = dataset.x_test[: cfg.batch_size]

    def infer():
        return infer_tensorflow(model, batch, batch_size=batch.shape[0])

    import tensorflow as tf

    device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
    suite.add(
        benchmark_inference(
            "tensorflow",
            device,
            infer,
            batch_size=batch.shape[0],
            warmup=max(3, cfg.warmup_iterations // 2),
            iterations=max(10, cfg.benchmark_iterations // 2),
            notes="Keras Model.predict",
        )
    )
    return {"model": model, "accuracy": acc, "checkpoint": ckpt, "device": device}


def run_jax(dataset, cfg: PipelineConfig, suite: BenchmarkSuite) -> dict:
    from src.jax_pipeline import cnn_forward, init_params, jax_device_summary, train_jax
    import jax
    import jax.numpy as jnp

    summary = jax_device_summary()
    if cfg.skip_train:
        params = init_params(jax.random.PRNGKey(cfg.seed))
        acc = None
        ckpt = None
        matmul = None
    else:
        result = train_jax(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, cfg)
        params = result.params
        acc = result.accuracy
        ckpt = str(result.checkpoint_path)
        matmul = result.matmul

    batch = jnp.asarray(dataset.x_test[: cfg.batch_size])
    forward = jax.jit(cnn_forward)
    forward(params, batch).block_until_ready()

    def infer():
        return forward(params, batch).block_until_ready()

    device = "cuda" if summary.get("backend") in {"gpu", "cuda"} else str(summary.get("backend") or "cpu")
    if device == "gpu":
        device = "cuda"
    suite.add(
        benchmark_inference(
            "jax",
            device if device in {"cpu", "cuda"} else "cpu",
            infer,
            batch_size=int(batch.shape[0]),
            warmup=cfg.warmup_iterations,
            iterations=cfg.benchmark_iterations,
            notes="JIT cnn_forward",
        )
    )
    return {"params": params, "accuracy": acc, "checkpoint": ckpt, "device": device, "matmul": matmul}


def run_onnx(pytorch_bundle: dict, dataset, cfg: PipelineConfig, suite: BenchmarkSuite) -> dict:
    from src.onnx_export import create_session, export_pytorch_to_onnx, infer_onnx, onnx_accuracy, validate_onnx

    if pytorch_bundle.get("model") is None:
        raise RuntimeError("ONNX export requires a trained (or instantiated) PyTorch model")

    onnx_path = cfg.model_dir / "cifar_cnn.onnx"
    export_pytorch_to_onnx(pytorch_bundle["model"], onnx_path)
    validate_onnx(onnx_path)
    session, info = create_session(onnx_path)
    acc = onnx_accuracy(session, info, dataset.x_test, dataset.y_test, batch_size=cfg.batch_size)

    batch = dataset.x_test[: cfg.batch_size]
    from src.data import to_nchw

    chunk = np.ascontiguousarray(to_nchw(batch.astype(np.float32)))

    def infer():
        return session.run([info.output_name], {info.input_name: chunk})

    providers = info.providers
    device = "cuda" if any("CUDA" in p or "Tensorrt" in p for p in providers) else "cpu"
    suite.add(
        benchmark_inference(
            "onnxruntime",
            device,
            infer,
            batch_size=batch.shape[0],
            warmup=cfg.warmup_iterations,
            iterations=cfg.benchmark_iterations,
            notes=",".join(providers),
        )
    )
    return {"path": str(onnx_path), "accuracy": acc, "providers": providers}


def run_pipeline(cfg: PipelineConfig) -> BenchmarkSuite:
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)
    configure_process_env()

    logger.info("=== GPU-Accelerated ML Pipeline ===")
    logger.info("frameworks=%s epochs=%s batch=%s samples=%s", cfg.frameworks, cfg.epochs, cfg.batch_size, cfg.max_samples)

    raw = load_dataset(cfg)
    suite = BenchmarkSuite(
        meta={
            "host": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dataset": "synthetic" if cfg.use_synthetic else "CIFAR-10",
            "train_samples": raw.num_train,
            "test_samples": raw.num_test,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "accuracy": {},
        }
    )
    status = cuda_status()
    suite.meta["cuda_backend"] = status.backend
    suite.meta["gpu_name"] = status.device_name

    augmented_train = run_preprocessing_benchmarks(raw.x_train, cfg, suite)
    raw.x_train = augmented_train
    dataset = normalized_copy(raw)

    pytorch_bundle: dict = {}
    if "pytorch" in cfg.frameworks:
        try:
            pytorch_bundle = run_pytorch(dataset, cfg, suite)
            if pytorch_bundle.get("accuracy") is not None:
                suite.meta["accuracy"]["pytorch"] = pytorch_bundle["accuracy"]
        except Exception:
            logger.error("PyTorch stage failed:\n%s", traceback.format_exc())

    if "tensorflow" in cfg.frameworks:
        try:
            tf_bundle = run_tensorflow(dataset, cfg, suite)
            if tf_bundle.get("accuracy") is not None:
                suite.meta["accuracy"]["tensorflow"] = tf_bundle["accuracy"]
        except Exception:
            logger.error("TensorFlow stage failed:\n%s", traceback.format_exc())

    if "jax" in cfg.frameworks:
        try:
            jax_bundle = run_jax(dataset, cfg, suite)
            if jax_bundle.get("accuracy") is not None:
                suite.meta["accuracy"]["jax"] = jax_bundle["accuracy"]
            if jax_bundle.get("matmul"):
                suite.meta["jax_matmul"] = jax_bundle["matmul"]
        except Exception:
            logger.error("JAX stage failed:\n%s", traceback.format_exc())

    if "onnx" in cfg.frameworks and not cfg.skip_onnx:
        try:
            if not pytorch_bundle:
                pytorch_bundle = run_pytorch(dataset, cfg, suite)
            onnx_bundle = run_onnx(pytorch_bundle, dataset, cfg, suite)
            if onnx_bundle.get("accuracy") is not None:
                suite.meta["accuracy"]["onnx"] = onnx_bundle["accuracy"]
        except Exception:
            logger.error("ONNX stage failed:\n%s", traceback.format_exc())

    csv_path = write_csv(suite.rows, cfg.output_dir / "results.csv")
    write_json(suite, cfg.output_dir / "results.json")
    plot_results(suite.rows, cfg.output_dir / "throughput.png")
    report_path = write_performance_report(suite.rows, suite.meta, cfg.output_dir / "PERFORMANCE_REPORT.md")
    suite.meta["csv"] = str(csv_path)
    suite.meta["report"] = str(report_path)
    logger.info("Pipeline complete. Report: %s", report_path)
    return suite


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    cfg = config_from_args(args)
    try:
        run_pipeline(cfg)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except Exception:
        logger.error("Pipeline crashed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
