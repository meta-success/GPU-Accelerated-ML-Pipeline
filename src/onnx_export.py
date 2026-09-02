"""PyTorch -> ONNX export, validation, and ONNX Runtime GPU/CPU inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.data import to_nchw
from src.logging_utils import get_logger

logger = get_logger("onnx")


@dataclass
class ONNXSessionInfo:
    path: Path
    providers: list[str]
    input_name: str
    output_name: str


def _require_onnx() -> None:
    """Import onnx in a TF 2.15-compatible pin (1.16.x, not 1.19+)."""
    try:
        import onnx  # noqa: F401
    except AttributeError as exc:
        raise RuntimeError(
            "Installed onnx is too new for ml-dtypes 0.3.x (TensorFlow 2.15). "
            "In the venv run: pip uninstall -y onnxscript onnx-ir && pip install \"onnx==1.16.2\""
        ) from exc


def export_pytorch_to_onnx(
    model,
    path: Path,
    opset: int = 17,
    batch_size: int = 1,
) -> Path:
    """Export a PyTorch CIFAR CNN to ONNX (NCHW, float32, dynamic batch)."""
    import torch

    _require_onnx()
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)

    export_kwargs = dict(
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    # PyTorch 2.8+ defaults to the onnxscript exporter. Prefer the classic path
    # so a CPU install still exports without that extra package.
    try:
        import inspect

        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
    except (TypeError, ValueError):
        pass

    torch.onnx.export(model, dummy, str(path), **export_kwargs)
    logger.info("Exported ONNX model -> %s", path)
    return path


def validate_onnx(path: Path) -> None:
    _require_onnx()
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    logger.info("ONNX checker passed for %s (IR=%s opset=%s)", path, model.ir_version, model.opset_import[0].version)


def _select_providers(prefer_trt: bool = True) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    ordered: list[str] = []
    if prefer_trt and "TensorrtExecutionProvider" in available:
        ordered.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available:
        ordered.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        ordered.append("CPUExecutionProvider")
    if not ordered:
        ordered = available
    logger.info("ONNX Runtime providers available=%s selected=%s", available, ordered)
    return ordered


def create_session(path: Path, prefer_trt: bool = True) -> tuple[object, ONNXSessionInfo]:
    import onnxruntime as ort

    providers = _select_providers(prefer_trt=prefer_trt)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    inputs = session.get_inputs()[0]
    outputs = session.get_outputs()[0]
    info = ONNXSessionInfo(
        path=path,
        providers=session.get_providers(),
        input_name=inputs.name,
        output_name=outputs.name,
    )
    logger.info("ONNX session ready | providers=%s | input=%s %s", info.providers, info.input_name, inputs.shape)
    return session, info


def infer_onnx(session, info: ONNXSessionInfo, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """images are NHWC [0,1] or normalized; converted to NCHW float32."""
    x = to_nchw(images.astype(np.float32))
    outs = []
    for start in range(0, x.shape[0], batch_size):
        chunk = np.ascontiguousarray(x[start : start + batch_size])
        logits = session.run([info.output_name], {info.input_name: chunk})[0]
        outs.append(logits)
    return np.concatenate(outs, axis=0)


def export_and_load(
    model,
    config: PipelineConfig,
    filename: str = "cifar_cnn.onnx",
) -> tuple[object, ONNXSessionInfo]:
    path = config.model_dir / filename
    export_pytorch_to_onnx(model, path)
    validate_onnx(path)
    return create_session(path)


def onnx_accuracy(session, info: ONNXSessionInfo, images: np.ndarray, labels: np.ndarray, batch_size: int = 128) -> float:
    logits = infer_onnx(session, info, images, batch_size=batch_size)
    pred = logits.argmax(axis=1)
    return float((pred == labels.astype(np.int64)).mean())
