"""TensorFlow/Keras CIFAR-10 CNN with GPU training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import NUM_CLASSES, PipelineConfig
from src.logging_utils import get_logger

logger = get_logger("tensorflow")

_TF = None
_TF_AVAILABLE = False


def _import_tf():
    global _TF, _TF_AVAILABLE
    if _TF is not None:
        return _TF
    try:
        import tensorflow as tf

        _TF = tf
        _TF_AVAILABLE = True
    except Exception as exc:  # pragma: no cover
        logger.warning("TensorFlow import failed: %s", exc)
        _TF = None
        _TF_AVAILABLE = False
    return _TF


def configure_tensorflow_gpu() -> str:
    """Enable memory growth on every visible GPU. Safe to call more than once."""
    tf = _import_tf()
    if tf is None:
        return "unavailable"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        if gpus:
            names = [gpu.name for gpu in gpus]
            logger.info("TensorFlow GPUs: %s", names)
            return "gpu"
        logger.info("TensorFlow running on CPU")
        return "cpu"
    except Exception as exc:
        logger.warning("TensorFlow GPU configuration failed: %s", exc)
        return "cpu"


def build_tf_model(num_classes: int = NUM_CLASSES, learning_rate: float = 1e-3):
    """Keras CNN that mirrors the PyTorch residual architecture at a high level."""
    tf = _import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow is not installed")

    from tensorflow.keras import layers, models, optimizers

    def residual_block(x, channels: int, name: str):
        skip = x
        y = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
        y = layers.BatchNormalization(name=f"{name}_bn1")(y)
        y = layers.ReLU(name=f"{name}_relu1")(y)
        y = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"{name}_conv2")(y)
        y = layers.BatchNormalization(name=f"{name}_bn2")(y)
        y = layers.Add(name=f"{name}_add")([skip, y])
        return layers.ReLU(name=f"{name}_out")(y)

    inputs = layers.Input(shape=(32, 32, 3), name="input")
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = residual_block(x, 32, "res32")
    x = layers.MaxPool2D(2, name="pool1")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False, name="down1")(x)
    x = layers.BatchNormalization(name="down1_bn")(x)
    x = layers.ReLU(name="down1_relu")(x)
    x = residual_block(x, 64, "res64")
    x = layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False, name="down2")(x)
    x = layers.BatchNormalization(name="down2_bn")(x)
    x = layers.ReLU(name="down2_relu")(x)
    x = residual_block(x, 128, "res128")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    model = models.Model(inputs, outputs, name="cifar_cnn_tf")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


@dataclass
class TFTrainResult:
    model: object
    history: dict
    checkpoint_path: Path
    device: str
    accuracy: float


def train_tensorflow(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> TFTrainResult:
    tf = _import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow is not installed")

    device = configure_tensorflow_gpu()
    model = build_tf_model(learning_rate=config.learning_rate)
    logger.info(
        "TensorFlow training on %s | params=%s | train=%s | epochs=%s",
        device,
        f"{model.count_params():,}",
        x_train.shape[0],
        config.epochs,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]
    history = model.fit(
        x_train,
        y_train.astype(np.int32),
        validation_data=(x_test, y_test.astype(np.int32)),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    ckpt = config.checkpoint_dir / "tensorflow_cifar_cnn.keras"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(ckpt)
    logger.info("Saved TensorFlow model -> %s", ckpt)

    val_acc = float(history.history.get("val_accuracy", [0.0])[-1])
    return TFTrainResult(
        model=model,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        checkpoint_path=ckpt,
        device=device,
        accuracy=val_acc,
    )


def load_tf_model(path: Path):
    tf = _import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow is not installed")
    return tf.keras.models.load_model(path)


def infer_tensorflow(model, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
    logits = model.predict(images, batch_size=batch_size, verbose=0)
    return logits


def evaluate_tensorflow(model, images: np.ndarray, labels: np.ndarray, batch_size: int = 128) -> float:
    _loss, acc = model.evaluate(images, labels.astype(np.int32), batch_size=batch_size, verbose=0)
    return float(acc)
