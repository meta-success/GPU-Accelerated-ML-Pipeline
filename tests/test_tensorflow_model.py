from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from src.tensorflow_model import build_tf_model, configure_tensorflow_gpu, infer_tensorflow


def test_tensorflow_gpu_config_returns_device_string():
    device = configure_tensorflow_gpu()
    assert device in {"gpu", "cpu", "unavailable"}


def test_tensorflow_model_forward(tiny_dataset):
    model = build_tf_model()
    logits = infer_tensorflow(model, tiny_dataset.x_test[:4], batch_size=4)
    assert logits.shape == (4, 10)
    n_params = int(model.count_params())
    assert n_params > 1_000
