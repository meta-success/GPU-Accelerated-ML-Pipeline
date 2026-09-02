"""Console runner and hardware helpers (no GPU required)."""

from __future__ import annotations

import pytest

from api.runner import RunRequest, build_argv


def test_build_argv_cpu_synthetic():
    argv = build_argv(
        RunRequest(
            device="cpu",
            frameworks=("pytorch", "onnx"),
            epochs=1,
            batch_size=32,
            max_samples=128,
            synthetic=True,
        )
    )
    assert "-m" in argv and "src" in argv
    assert argv[argv.index("--device") + 1] == "cpu"
    assert "--synthetic" in argv
    assert "pytorch" in argv and "onnx" in argv


def test_build_argv_rejects_bad_device():
    with pytest.raises(ValueError, match="device"):
        build_argv(RunRequest(device="tpu"))


def test_build_argv_requires_framework():
    with pytest.raises(ValueError, match="framework"):
        build_argv(RunRequest(frameworks=()))
