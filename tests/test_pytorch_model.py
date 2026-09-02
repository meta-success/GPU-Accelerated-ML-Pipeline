from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.pytorch_model import CIFARCNN, count_parameters, evaluate, infer_pytorch, train_pytorch


def test_pytorch_forward_shape():
    model = CIFARCNN()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
    assert count_parameters(model) > 1_000


def test_pytorch_train_one_epoch(tiny_config, tiny_dataset):
    tiny_config.epochs = 1
    tiny_config.batch_size = 8
    result = train_pytorch(
        tiny_dataset.x_train,
        tiny_dataset.y_train,
        tiny_dataset.x_test,
        tiny_dataset.y_test,
        tiny_config,
    )
    assert result.checkpoint_path.exists()
    assert 0.0 <= result.accuracy <= 1.0
    device = next(result.model.parameters()).device
    acc = evaluate(result.model, tiny_dataset.x_test, tiny_dataset.y_test, device, 8)
    assert 0.0 <= acc <= 1.0
    probs = infer_pytorch(result.model, tiny_dataset.x_test[:8], device, 8)
    assert probs.shape == (8, 10)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)
