from __future__ import annotations

import numpy as np
import pytest

from src.data import load_dataset, synthetic_cifar, to_nchw, to_nhwc
from src.pipeline import run_pipeline


def test_synthetic_shapes_and_layout():
    ds = synthetic_cifar(20, 10, seed=1)
    assert ds.x_train.shape == (20, 32, 32, 3)
    assert ds.y_test.max() < 10
    nchw = to_nchw(ds.x_train)
    assert nchw.shape == (20, 3, 32, 32)
    np.testing.assert_allclose(to_nhwc(nchw), ds.x_train, atol=1e-6)


def test_load_dataset_synthetic_flag(tiny_config):
    tiny_config.use_synthetic = True
    ds = load_dataset(tiny_config)
    assert ds.num_train == tiny_config.max_samples


@pytest.mark.integration
def test_end_to_end_pytorch_onnx_cpu(tiny_config):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    tiny_config.frameworks = ("pytorch", "onnx")
    tiny_config.epochs = 1
    tiny_config.max_samples = 32
    tiny_config.batch_size = 8
    tiny_config.benchmark_iterations = 2
    tiny_config.warmup_iterations = 1
    tiny_config.use_synthetic = True
    tiny_config.skip_onnx = False

    suite = run_pipeline(tiny_config)
    assert (tiny_config.output_dir / "results.csv").exists()
    assert (tiny_config.output_dir / "PERFORMANCE_REPORT.md").exists()
    phases = {(r.framework, r.phase) for r in suite.rows}
    assert any(p[1].startswith("preprocess") for p in phases)
    assert ("pytorch", "inference") in phases
    assert ("onnxruntime", "inference") in phases
    assert suite.meta["accuracy"].get("pytorch") is not None or tiny_config.skip_train
