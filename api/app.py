"""Optional FastAPI inference service backed by the exported ONNX model."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from src.config import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, MODEL_DIR, NUM_CLASSES
from src.logging_utils import setup_logging
from src.onnx_export import create_session

setup_logging("INFO")

ONNX_PATH = MODEL_DIR / "cifar_cnn.onnx"
app = FastAPI(title="GPU-Accelerated CIFAR-10 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = None
_info = None


class PredictResponse(BaseModel):
    label: str
    class_id: int = Field(ge=0, lt=NUM_CLASSES)
    confidence: float
    probabilities: List[float]


def _load_session():
    global _session, _info
    if _session is not None:
        return _session, _info
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {ONNX_PATH}. Run `python -m src` first to export it."
        )
    _session, _info = create_session(ONNX_PATH)
    return _session, _info


def _preprocess_image(raw: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std = np.array(CIFAR10_STD, dtype=np.float32)
    arr = (arr - mean) / std
    nchw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return np.ascontiguousarray(nchw)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "onnx_exists": ONNX_PATH.exists()}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    try:
        session, info = _load_session()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        tensor = _preprocess_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    logits = session.run([info.output_name], {info.input_name: tensor})[0][0]
    logits = logits - logits.max()
    exp = np.exp(logits)
    probs = exp / exp.sum()
    class_id = int(probs.argmax())
    return PredictResponse(
        label=CIFAR10_CLASSES[class_id],
        class_id=class_id,
        confidence=float(probs[class_id]),
        probabilities=[float(p) for p in probs],
    )
