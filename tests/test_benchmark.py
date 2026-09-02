from __future__ import annotations

from src.benchmark import BenchmarkRow, BenchmarkSuite, speedup_table, time_callable, write_csv, write_json


def test_time_callable_positive():
    n = {"i": 0}

    def fn():
        n["i"] += 1

    avg = time_callable(fn, warmup=1, iterations=3, sync_device="cpu")
    assert avg >= 0.0
    assert n["i"] == 4


def test_speedup_and_export(tmp_path):
    rows = [
        BenchmarkRow("pytorch", "cpu", "inference", 32, 10, 10.0, 3200.0, 100.0, "cpu"),
        BenchmarkRow("pytorch", "cuda", "inference", 32, 10, 2.0, 16000.0, 400.0, "gpu"),
    ]
    table = speedup_table(rows)
    assert table[0]["speedup"] == 5.0

    suite = BenchmarkSuite(rows=rows, meta={"gpu": "test"})
    csv_path = write_csv(rows, tmp_path / "results.csv")
    json_path = write_json(suite, tmp_path / "results.json")
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert json_path.exists() and "pytorch" in json_path.read_text(encoding="utf-8")
