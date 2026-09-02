"""Local console + inference API for the GPU-accelerated ML pipeline."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.hardware import gpu_info
from api.runner import PipelineRunner, RunRequest
from src.config import BENCHMARK_DIR, CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, MODEL_DIR, NUM_CLASSES, ROOT_DIR
from src.logging_utils import setup_logging

setup_logging("INFO")

STATIC_DIR = Path(__file__).resolve().parent / "static"
ONNX_PATH = MODEL_DIR / "cifar_cnn.onnx"

ARTIFACTS = {
    "csv": BENCHMARK_DIR / "results.csv",
    "json": BENCHMARK_DIR / "results.json",
    "report": BENCHMARK_DIR / "PERFORMANCE_REPORT.md",
    "plot": BENCHMARK_DIR / "throughput.png",
    "sample": BENCHMARK_DIR / "sample_results.csv",
}

app = FastAPI(title="GPU Pipeline Console", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = None
_info = None
_runner: PipelineRunner


def reset_onnx_session(_code: int = 0) -> None:
    global _session, _info
    _session = None
    _info = None


_runner = PipelineRunner(on_complete=reset_onnx_session)


class RunBody(BaseModel):
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    frameworks: List[str] = Field(default_factory=lambda: ["pytorch", "tensorflow", "jax", "onnx"])
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=128, ge=1, le=1024)
    max_samples: int = Field(default=4000, ge=8, le=50000)
    synthetic: bool = False
    skip_train: bool = False
    brightness: float = Field(default=1.15, ge=0.1, le=3.0)
    noise_prob: float = Field(default=0.02, ge=0.0, le=0.5)
    benchmark_iterations: int = Field(default=50, ge=1, le=500)
    warmup: int = Field(default=10, ge=0, le=100)


class PredictResponse(BaseModel):
    label: str
    class_id: int = Field(ge=0, lt=NUM_CLASSES)
    confidence: float
    probabilities: List[float]


def _artifact_flags() -> dict[str, bool]:
    return {key: path.is_file() for key, path in ARTIFACTS.items()} | {"onnx": ONNX_PATH.is_file()}


def _load_session():
    global _session, _info
    if _session is not None:
        return _session, _info
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {ONNX_PATH}. Run the pipeline with ONNX enabled first."
        )
    from src.onnx_export import create_session

    _session, _info = create_session(ONNX_PATH)
    return _session, _info


def _preprocess_image(raw: bytes):
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std = np.array(CIFAR10_STD, dtype=np.float32)
    arr = (arr - mean) / std
    nchw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return np.ascontiguousarray(nchw)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@app.get("/")
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="Console UI files are missing")
    return FileResponse(page)


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "gpu": gpu_info(),
        "job": _runner.snapshot(),
        "artifacts": _artifact_flags(),
        "classes": list(CIFAR10_CLASSES),
        "root": str(ROOT_DIR),
    }


@app.get("/api/logs")
def logs(cursor: int = 0) -> dict[str, Any]:
    next_cursor, lines = _runner.logs_since(cursor)
    snap = _runner.snapshot()
    return {"cursor": next_cursor, "lines": lines, "status": snap["status"]}


@app.post("/api/run")
def run_pipeline(body: RunBody) -> dict[str, Any]:
    req = RunRequest(
        device=body.device,
        frameworks=tuple(body.frameworks),
        epochs=body.epochs,
        batch_size=body.batch_size,
        max_samples=body.max_samples,
        synthetic=body.synthetic,
        skip_train=body.skip_train,
        brightness=body.brightness,
        noise_prob=body.noise_prob,
        benchmark_iterations=body.benchmark_iterations,
        warmup=body.warmup,
    )
    try:
        _runner.start(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "job": _runner.snapshot()}


@app.post("/api/stop")
def stop_pipeline() -> dict[str, Any]:
    try:
        _runner.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "job": _runner.snapshot()}


@app.get("/api/results")
def results() -> dict[str, Any]:
    csv_path = ARTIFACTS["csv"] if ARTIFACTS["csv"].is_file() else ARTIFACTS["sample"]
    source = "measured" if ARTIFACTS["csv"].is_file() else "sample"
    rows = _read_csv_rows(csv_path) if csv_path.is_file() else []
    return {"source": source, "rows": rows, "artifacts": _artifact_flags()}


@app.get("/api/download/{name}")
def download(name: str):
    if name == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for key, path in ARTIFACTS.items():
                if key == "sample":
                    continue
                if path.is_file():
                    zf.write(path, arcname=path.name)
            if ONNX_PATH.is_file():
                zf.write(ONNX_PATH, arcname=ONNX_PATH.name)
        if buffer.tell() == 0:
            raise HTTPException(status_code=404, detail="No result files yet. Run the pipeline first.")
        buffer.seek(0)
        headers = {"Content-Disposition": "attachment; filename=gpu-pipeline-results.zip"}
        return StreamingResponse(buffer, media_type="application/zip", headers=headers)

    path = ARTIFACTS.get(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No {name} file yet. Run the pipeline first.")
    media = {
        "csv": "text/csv",
        "sample": "text/csv",
        "json": "application/json",
        "report": "text/markdown",
        "plot": "image/png",
    }.get(name, "application/octet-stream")
    disposition = "inline" if name == "plot" else "attachment"
    return FileResponse(
        path,
        media_type=media,
        filename=path.name,
        content_disposition_type=disposition,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "onnx_exists": ONNX_PATH.exists(), "job": _runner.snapshot()["status"]}


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

    import numpy as np

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


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
