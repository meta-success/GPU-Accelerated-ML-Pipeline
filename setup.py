"""Installable package wrapper for the GPU-accelerated ML pipeline."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="gpu-accelerated-ml-pipeline",
    version="1.0.0",
    description="GPU-accelerated image classification pipeline (CUDA, JAX, PyTorch, TensorFlow, ONNX).",
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    include_package_data=True,
    package_data={"api": ["static/*"]},
    entry_points={
        "console_scripts": [
            "gpu-pipeline=src.pipeline:main",
            "gpu-pipeline-ui=api.__main__:main",
        ]
    },
)
