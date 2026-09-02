from __future__ import annotations

"""Print which GPU backends this machine actually exposes."""

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import configure_process_env
from src.cuda_kernels import log_cuda_banner
from src.logging_utils import setup_logging


def main() -> None:
    setup_logging("INFO")
    configure_process_env()
    print(f"Python {platform.python_version()} on {platform.platform()}")
    log_cuda_banner()

    try:
        import torch

        print(f"PyTorch {torch.__version__} | cuda={torch.cuda.is_available()} | device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    except Exception as exc:
        print(f"PyTorch: unavailable ({exc})")

    try:
        import tensorflow as tf
        from src.tensorflow_model import configure_tensorflow_gpu

        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow {tf.__version__} | gpus={gpus} | config={configure_tensorflow_gpu()}")
    except Exception as exc:
        print(f"TensorFlow: unavailable ({exc})")

    try:
        from src.jax_pipeline import jax_device_summary

        print(f"JAX {jax_device_summary()}")
    except Exception as exc:
        print(f"JAX: unavailable ({exc})")

    try:
        import onnx
        import onnxruntime as ort

        print(f"ONNX {onnx.__version__} | ORT {ort.__version__} | providers={ort.get_available_providers()}")
    except Exception as exc:
        print(f"ONNX: unavailable ({exc})")


if __name__ == "__main__":
    main()
