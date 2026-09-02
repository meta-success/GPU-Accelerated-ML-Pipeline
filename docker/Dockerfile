# CUDA 11.8 + cuDNN 8 — works with PyTorch cu118, TF 2.14–2.15, JAX CUDA 11/12.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Framework wheels that must come from CUDA indexes (order matters).
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    && pip install tensorflow==2.15.1 \
    && pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install onnx onnxruntime-gpu \
    && pip install cupy-cuda11x numba \
    && pip install numpy pillow tqdm psutil pandas matplotlib pytest fastapi uvicorn python-multipart

COPY . /app
RUN pip install -e . --no-deps

RUN mkdir -p /app/data /app/checkpoints /app/models /app/benchmark_results

ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["--epochs", "3", "--batch-size", "128", "--max-samples", "4000"]
