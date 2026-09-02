from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.onnx_export import create_session, export_pytorch_to_onnx, infer_onnx, validate_onnx
from src.pytorch_model import CIFARCNN


def test_onnx_export_validate_and_infer(tmp_path, tiny_dataset):
    model = CIFARCNN().eval()
    path = tmp_path / "model.onnx"
    export_pytorch_to_onnx(model, path, opset=17, batch_size=1)
    assert path.exists()
    validate_onnx(path)

    session, info = create_session(path, prefer_trt=False)
    logits = infer_onnx(session, info, tiny_dataset.x_test[:8], batch_size=4)
    assert logits.shape == (8, 10)
    assert np.isfinite(logits).all()
    assert "CPUExecutionProvider" in info.providers or "CUDAExecutionProvider" in info.providers
