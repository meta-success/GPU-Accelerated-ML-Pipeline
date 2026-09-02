"""Background subprocess runner for the training / benchmark pipeline."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from src.config import ROOT_DIR


OnComplete = Optional[Callable[[int], None]]


@dataclass
class RunRequest:
    device: str = "auto"
    frameworks: tuple[str, ...] = ("pytorch", "tensorflow", "jax", "onnx")
    epochs: int = 3
    batch_size: int = 128
    max_samples: int = 4000
    synthetic: bool = False
    skip_train: bool = False
    brightness: float = 1.15
    noise_prob: float = 0.02
    benchmark_iterations: int = 50
    warmup: int = 10


@dataclass
class JobState:
    status: str = "idle"
    device: Optional[str] = None
    command: list[str] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    log_count: int = 0


def build_argv(req: RunRequest) -> list[str]:
    if req.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    frameworks = [f.strip().lower() for f in req.frameworks if f.strip()]
    allowed = {"pytorch", "tensorflow", "jax", "onnx"}
    unknown = [f for f in frameworks if f not in allowed]
    if unknown:
        raise ValueError(f"unknown frameworks: {unknown}")
    if not frameworks:
        raise ValueError("select at least one framework")
    if req.epochs < 1 or req.batch_size < 1 or req.max_samples < 8:
        raise ValueError("epochs, batch size, and max samples are too small")

    argv = [
        sys.executable,
        "-u",
        "-m",
        "src",
        "--device",
        req.device,
        "--frameworks",
        *frameworks,
        "--epochs",
        str(int(req.epochs)),
        "--batch-size",
        str(int(req.batch_size)),
        "--max-samples",
        str(int(req.max_samples)),
        "--brightness",
        str(req.brightness),
        "--noise-prob",
        str(req.noise_prob),
        "--benchmark-iterations",
        str(int(req.benchmark_iterations)),
        "--warmup",
        str(int(req.warmup)),
    ]
    if req.synthetic:
        argv.append("--synthetic")
    if req.skip_train:
        argv.append("--skip-train")
    return argv


class PipelineRunner:
    def __init__(self, on_complete: OnComplete = None) -> None:
        self._lock = threading.Lock()
        self._logs: deque[str] = deque(maxlen=8000)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._on_complete = on_complete
        self.state = JobState()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.state.status,
                "device": self.state.device,
                "command": self.state.command,
                "started_at": self.state.started_at,
                "finished_at": self.state.finished_at,
                "returncode": self.state.returncode,
                "error": self.state.error,
                "log_count": len(self._logs),
            }

    def logs_since(self, cursor: int) -> tuple[int, list[str]]:
        with self._lock:
            lines = list(self._logs)
        start = max(0, int(cursor))
        chunk = lines[start:]
        return start + len(chunk), chunk

    def start(self, req: RunRequest) -> JobState:
        argv = build_argv(req)
        with self._lock:
            if self.state.status == "running":
                raise RuntimeError("a pipeline run is already in progress")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            kwargs: dict = {
                "cwd": str(ROOT_DIR),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "env": env,
                "bufsize": 1,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._proc = subprocess.Popen(argv, **kwargs)
            self._logs.clear()
            self.state = JobState(
                status="running",
                device=req.device,
                command=argv,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self._append_unlocked(" ".join(argv))
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            return self.state

    def stop(self) -> JobState:
        with self._lock:
            proc = self._proc
            if proc is None or self.state.status != "running":
                raise RuntimeError("no running pipeline to stop")
            try:
                if sys.platform == "win32":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except Exception:
                proc.kill()
            self._append_unlocked("stop requested")
            return self.state

    def _append_unlocked(self, line: str) -> None:
        text = line.rstrip("\r\n")
        if text:
            self._logs.append(text)

    def _pump(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, ""):
                with self._lock:
                    self._append_unlocked(line)
        finally:
            code = proc.wait()
            finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
            callback = self._on_complete
            with self._lock:
                self.state.returncode = code
                self.state.finished_at = finished
                self.state.status = "succeeded" if code == 0 else "failed"
                if code != 0:
                    self.state.error = f"pipeline exited with code {code}"
                self._append_unlocked(f"process exited with code {code}")
                self._proc = None
            if callback is not None:
                try:
                    callback(code)
                except Exception:
                    pass
