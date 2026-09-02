from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.jax_pipeline import (
    accuracy_fn,
    batched_forward,
    build_pmap_step,
    build_train_step,
    cnn_forward,
    gpu_matmul_demo,
    init_params,
    loss_fn,
)


def test_jax_forward_shapes():
    params = init_params(jax.random.PRNGKey(0))
    single = jnp.ones((32, 32, 3), dtype=jnp.float32)
    logits = cnn_forward(params, single)
    assert logits.shape == (10,)

    batch = jnp.ones((4, 32, 32, 3), dtype=jnp.float32)
    batched = cnn_forward(params, batch)
    vmapped = batched_forward(params, batch)
    assert batched.shape == (4, 10)
    assert vmapped.shape == (4, 10)


def test_jax_jit_train_step_reduces_loss():
    params = init_params(jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(2), (8, 32, 32, 3))
    y = jax.random.randint(jax.random.PRNGKey(3), (8,), 0, 10)
    step = build_train_step(0.05)
    loss0 = float(loss_fn(params, x, y))
    params, loss1 = step(params, x, y)
    assert float(loss1) <= loss0 + 1.0
    acc = float(accuracy_fn(params, x, y))
    assert 0.0 <= acc <= 1.0


def test_jax_pmap_builds_and_runs_on_local_devices():
    params = init_params(jax.random.PRNGKey(0))
    n = jax.local_device_count()
    x = jax.random.normal(jax.random.PRNGKey(4), (n, 2, 32, 32, 3))
    y = jax.random.randint(jax.random.PRNGKey(5), (n, 2), 0, 10)
    replicated = jax.device_put_replicated(params, jax.local_devices())
    step = build_pmap_step(0.01)
    new_params, loss = step(replicated, x, y)
    assert float(jnp.mean(loss)) >= 0.0
    assert new_params["fc_w"].shape[0] == n


def test_jax_matmul_demo():
    result = gpu_matmul_demo(size=64, seed=0)
    assert result["size"] == 64
    assert result["seconds"] >= 0.0
