"""JAX CIFAR-10 CNN with JIT, vmap, pmap, and GPU matrix operations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from src.config import NUM_CLASSES, PipelineConfig
from src.data import iterate_batches
from src.logging_utils import get_logger

logger = get_logger("jax")

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _JAX_AVAILABLE = False
    logger.warning("JAX import failed: %s", exc)


def _import_jax():
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed")
    return jax


def jax_device_summary() -> dict[str, Any]:
    if not _JAX_AVAILABLE:
        return {"available": False, "devices": [], "backend": "none"}
    devices = [str(d) for d in jax.devices()]
    backend = jax.default_backend()
    logger.info("JAX backend=%s devices=%s", backend, devices)
    return {"available": True, "devices": devices, "backend": backend, "count": len(jax.devices())}


def _he_normal(key, shape: tuple[int, ...]):
    fan_in = shape[0] * shape[1] * shape[2] if len(shape) == 4 else shape[0]
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / max(fan_in, 1))


def init_params(rng, num_classes: int = NUM_CLASSES) -> dict:
    """Initialize a small NHWC CNN: conv -> conv -> dense."""
    keys = jax.random.split(rng, 8)
    return {
        "conv1_w": _he_normal(keys[0], (3, 3, 3, 32)),
        "conv1_b": jnp.zeros((32,)),
        "conv2_w": _he_normal(keys[1], (3, 3, 32, 64)),
        "conv2_b": jnp.zeros((64,)),
        "conv3_w": _he_normal(keys[2], (3, 3, 64, 128)),
        "conv3_b": jnp.zeros((128,)),
        "fc_w": _he_normal(keys[3], (128, num_classes)),
        "fc_b": jnp.zeros((num_classes,)),
    }


def _conv(x, w, b, stride: int = 1):
    return (
        jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(stride, stride),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + b
    )


def _max_pool(x):
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )


def cnn_forward(params: dict, x):
    """Single-example or batched NHWC forward. x shape (H, W, C) or (N, H, W, C)."""
    squeezed = False
    if x.ndim == 3:
        x = x[None, ...]
        squeezed = True
    x = jax.nn.relu(_conv(x, params["conv1_w"], params["conv1_b"]))
    x = _max_pool(x)
    x = jax.nn.relu(_conv(x, params["conv2_w"], params["conv2_b"], stride=2))
    x = jax.nn.relu(_conv(x, params["conv3_w"], params["conv3_b"], stride=2))
    x = jnp.mean(x, axis=(1, 2))
    logits = x @ params["fc_w"] + params["fc_b"]
    return logits[0] if squeezed else logits


def loss_fn(params: dict, x, y) -> Any:
    logits = cnn_forward(params, x)
    y_onehot = jax.nn.one_hot(y, NUM_CLASSES)
    if logits.ndim == 1:
        logits = logits[None, :]
        y_onehot = y_onehot[None, :]
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y_onehot * log_probs, axis=-1))


def accuracy_fn(params: dict, x, y) -> Any:
    logits = cnn_forward(params, x)
    return jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32))


def build_train_step(learning_rate: float = 1e-3):
    """JIT-compiled SGD step. Gradients are vectorized over the batch with vmap-ready loss."""

    @partial(jax.jit, static_argnames=())
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        return new_params, loss

    return train_step


def batched_forward(params, images):
    """vmap over the batch axis while sharing parameters."""
    return jax.vmap(lambda img: cnn_forward(params, img))(images)


def build_pmap_step(learning_rate: float = 1e-3):
    """Multi-device training step. Works with 1 device; shards the leading axis."""

    def device_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        return new_params, loss

    return jax.pmap(device_step, axis_name="devices")


def _shard(array: np.ndarray, n_devices: int) -> np.ndarray:
    """Pad then reshape leading dim to (devices, local_batch, ...)."""
    n = array.shape[0]
    pad = (-n) % n_devices
    if pad:
        array = np.concatenate([array, array[:pad]], axis=0)
    local = array.shape[0] // n_devices
    return array.reshape((n_devices, local) + array.shape[1:])


def gpu_matmul_demo(size: int = 1024, seed: int = 0) -> dict[str, float]:
    """JIT GPU/CPU matmul to show XLA kernel generation without hand-written CUDA."""
    _import_jax()
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed")

    import time

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (size, size), dtype=jnp.float32)
    b = jax.random.normal(k2, (size, size), dtype=jnp.float32)

    @jax.jit
    def matmul(x, y):
        return x @ y

    matmul(a, b).block_until_ready()
    start = time.perf_counter()
    out = matmul(a, b).block_until_ready()
    elapsed = time.perf_counter() - start
    logger.info("JAX JIT matmul %sx%s in %.4fs | mean=%.4f", size, size, elapsed, float(out.mean()))
    return {"size": size, "seconds": elapsed, "mean": float(out.mean())}


@dataclass
class JAXTrainResult:
    params: dict
    history: list[dict]
    checkpoint_path: Path
    device: str
    accuracy: float
    matmul: dict


def save_jax_params(params: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_tree = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    np.savez(path, **{k: numpy_tree[k] for k in numpy_tree})
    logger.info("Saved JAX params -> %s", path)
    return path


def load_jax_params(path: Path) -> dict:
    _import_jax()
    blob = np.load(path)
    return {k: jnp.asarray(blob[k]) for k in blob.files}


def train_jax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> JAXTrainResult:
    _import_jax()
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed")

    summary = jax_device_summary()
    rng = jax.random.PRNGKey(config.seed)
    params = init_params(rng)
    train_step = build_train_step(config.learning_rate)
    n_devices = max(int(summary.get("count") or 1), 1)
    use_pmap = n_devices > 1
    pmap_step = build_pmap_step(config.learning_rate) if use_pmap else None
    replicated = jax.device_put_replicated(params, jax.devices()) if use_pmap else None

    logger.info(
        "JAX training | backend=%s | devices=%s | pmap=%s | train=%s | epochs=%s",
        summary.get("backend"),
        n_devices,
        use_pmap,
        x_train.shape[0],
        config.epochs,
    )

    history: list[dict] = []
    jit_eval = jax.jit(accuracy_fn)

    # pmap works with one device; run a smoke step so single-GPU machines still
    # exercise XLA sharding.
    try:
        smoke = build_pmap_step(config.learning_rate)
        replicated_smoke = jax.device_put_replicated(params, jax.devices())
        xb, yb = next(iterate_batches(x_train, y_train, max(n_devices * 2, 2), shuffle=False))
        _, pmap_loss = smoke(replicated_smoke, _shard(xb, n_devices), _shard(yb, n_devices))
        logger.info("pmap smoke-test loss=%.4f on %s device(s)", float(np.mean(np.asarray(pmap_loss))), n_devices)
    except Exception as exc:
        logger.warning("pmap smoke-test skipped: %s", exc)

    for epoch in range(1, config.epochs + 1):
        epoch_loss = []
        for xb, yb in iterate_batches(
            x_train, y_train, config.batch_size, shuffle=True, seed=config.seed + epoch
        ):
            if use_pmap and pmap_step is not None:
                x_sharded = _shard(xb, n_devices)
                y_sharded = _shard(yb, n_devices)
                replicated, loss = pmap_step(replicated, x_sharded, y_sharded)
                epoch_loss.append(float(np.mean(np.asarray(loss))))
            else:
                params, loss = train_step(params, xb, yb)
                epoch_loss.append(float(loss))

        if use_pmap:
            params = jax.device_get(jax.tree_util.tree_map(lambda z: z[0], replicated))

        acc = float(jit_eval(params, x_test, y_test))
        mean_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        history.append({"epoch": epoch, "loss": mean_loss, "val_accuracy": acc})
        logger.info("JAX epoch %s/%s | loss=%.4f | val_acc=%.3f", epoch, config.epochs, mean_loss, acc)

    ckpt = save_jax_params(params, config.checkpoint_dir / "jax_cifar_cnn.npz")
    matmul = gpu_matmul_demo(size=512, seed=config.seed)
    return JAXTrainResult(
        params=params,
        history=history,
        checkpoint_path=ckpt,
        device=str(summary.get("backend")),
        accuracy=history[-1]["val_accuracy"] if history else 0.0,
        matmul=matmul,
    )


def infer_jax(params: dict, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
    _import_jax()
    forward = jax.jit(cnn_forward)
    outs = []
    dummy = np.zeros((images.shape[0],), dtype=np.int32)
    for xb, _ in iterate_batches(images, dummy, batch_size, shuffle=False):
        logits = forward(params, xb)
        outs.append(np.asarray(jax.nn.softmax(logits)))
    return np.concatenate(outs, axis=0)
