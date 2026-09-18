from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "resident_colmeans_scaling.py"
SPEC = importlib.util.spec_from_file_location("resident_colmeans_scaling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_stats_use_interpolated_percentiles() -> None:
    assert benchmark.sample_stats([4.0, 1.0, 3.0, 2.0]) == {
        "query_median_sec": 2.5,
        "query_p10_sec": pytest.approx(1.3),
        "query_p90_sec": pytest.approx(3.7),
        "query_min_sec": 1.0,
        "query_max_sec": 4.0,
    }
    with pytest.raises(ValueError, match="at least one"):
        benchmark.sample_stats([])
    with pytest.raises(ValueError, match="finite"):
        benchmark.sample_stats([float("nan")])


def test_cli_defaults_and_validation() -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(["run", "--output-dir", "out"])
    assert args.sizes == (100_000, 300_000, 1_000_000, 3_000_000)
    assert args.backends == ("numpy", "duckdb", "datafusion")
    assert args.repeats == 5
    assert args.threads == 2
    assert args.engine_memory_gib == 1.0
    assert args.rss_limit_gib == 5.0
    assert args.timeout_sec == 300.0

    extra = parser.parse_args(["run", "--output-dir", "out", "--sizes", "10000000"])
    assert extra.sizes == (10_000_000,)
    for arguments in (
        ["run", "--output-dir", "out", "--repeats", "101"],
        ["run", "--output-dir", "out", "--sizes", "2,2"],
        ["run", "--output-dir", "out", "--backends", "unknown"],
        ["run", "--output-dir", "out", "--rss-limit-gib", "0"],
        ["plot", "--input", "in.json"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)


def _successful_row(size: int, backend: str) -> dict[str, object]:
    return {
        "size": size,
        "backend": backend,
        "status": "ok",
        "error": "",
        "rows": benchmark.ROWS,
        "cols": size,
        "nnz": size,
        "density": 1 / benchmark.ROWS,
        "query_median_sec": 0.2,
        "query_p10_sec": 0.1,
        "query_p90_sec": 0.3,
        "output_length": size,
        "output_bytes": size * 8,
    }


def test_incremental_persistence_refuses_overwrite_and_resumes_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results"
    argv = [
        "run",
        "--output-dir",
        str(output),
        "--sizes",
        "11",
        "--backends",
        "numpy",
    ]
    calls: list[tuple[int, str]] = []

    def fake_run(size: int, backend: str, **kwargs):
        calls.append((size, backend))
        assert kwargs["threads"] == 2
        return _successful_row(size, backend)

    monkeypatch.setattr(benchmark, "run_case", fake_run)
    args = benchmark.build_parser().parse_args(argv)
    assert benchmark.run_benchmark(args) == 0
    assert calls == [(11, "numpy")]
    document = json.loads((output / "results.json").read_text())
    assert document["results"] == [_successful_row(11, "numpy")]
    csv_text = (output / "results.csv").read_text()
    assert "query_samples_json" in csv_text

    with pytest.raises(FileExistsError, match="--resume"):
        benchmark.run_benchmark(args)
    calls.clear()
    resumed = benchmark.build_parser().parse_args([*argv, "--resume"])
    assert benchmark.run_benchmark(resumed) == 0
    assert calls == []

    mismatched = benchmark.build_parser().parse_args(
        [*argv, "--resume", "--threads", "3"]
    )
    with pytest.raises(ValueError, match="configuration"):
        benchmark.run_benchmark(mismatched)


def test_resume_rejects_partial_or_duplicate_canonical_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results"
    output.mkdir()
    args = benchmark.build_parser().parse_args(
        [
            "run",
            "--output-dir",
            str(output),
            "--sizes",
            "11",
            "--backends",
            "numpy",
            "--resume",
        ]
    )
    (output / "results.csv").write_text("partial")
    with pytest.raises(ValueError, match="canonical"):
        benchmark.run_benchmark(args)

    row = _successful_row(11, "numpy")
    document = {
        "config": benchmark._config(args),
        "results": [row, row],
    }
    (output / "results.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.run_benchmark(args)


def test_parent_memory_limit_precedes_queued_child_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PipeEnd:
        def close(self) -> None:
            pass

    class Receiving(PipeEnd):
        def poll(self, timeout: float) -> bool:
            return True

        def recv(self) -> dict[str, str]:
            return {"status": "ok"}

    class Process:
        pid = 123
        exitcode = -15

        def __init__(self) -> None:
            self.alive = True

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            pass

        def terminate(self) -> None:
            self.alive = False

        def kill(self) -> None:
            self.alive = False

    class Context:
        def Pipe(self, *, duplex: bool):  # noqa: N802
            assert duplex is False
            return Receiving(), PipeEnd()

        def Process(self, *, target, args) -> Process:  # noqa: N802
            return Process()

    monkeypatch.setattr(
        benchmark.multiprocessing, "get_context", lambda name: Context()
    )
    monkeypatch.setattr(benchmark, "read_rss_bytes", lambda pid: 2048)
    row = benchmark.run_case(
        11,
        "numpy",
        repeats=5,
        threads=2,
        engine_memory_gib=1.0,
        rss_limit_bytes=1024,
        rss_limit_gib=1 / 1024**3,
        timeout_sec=300,
        output_dir=tmp_path,
    )
    assert row["status"] == "memory_limit"
    assert "RSS exceeded" in row["error"]
    assert row["peak_rss_bytes"] == 2048


def test_svg_is_accessible_xml_with_width_axis_and_failure_markers(
    tmp_path: Path,
) -> None:
    rows = [
        _successful_row(100_000, "numpy"),
        {
            "backend": "duckdb",
            "status": "timeout",
            "cols": 300_000,
        },
        {
            "backend": "datafusion",
            "status": "memory_limit",
            "cols": 1_000_000,
        },
    ]
    output = tmp_path / "resident.svg"
    benchmark.write_svg(rows, output)
    svg = output.read_text()
    ET.fromstring(svg)
    assert "aria-labelledby" in svg
    assert "one NNZ per output column" in svg
    assert "Result columns / output vector length (log scale)" in svg
    assert "not Figure 1 MTX density" in svg
    assert "duckdb timeout" in svg
    assert "datafusion memory_limit" in svg
    assert "Whiskers: p10–p90" in svg
    with pytest.raises(FileExistsError, match="overwrite"):
        benchmark.write_svg(rows, output)
