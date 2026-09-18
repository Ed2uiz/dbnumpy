from __future__ import annotations

import gzip
import importlib.util
import inspect
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "real_matrix_scaling.py"
SPEC = importlib.util.spec_from_file_location("real_matrix_scaling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def write_header(
    path: Path,
    dimensions: str = "20000 1000 17",
    *,
    storage_format: str = "coordinate",
    field: str = "real",
    symmetry: str = "general",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="ascii") as stream:
        stream.write(f"%%MatrixMarket matrix {storage_format} {field} {symmetry}\n")
        stream.write("% generated fixture\n")
        stream.write(f"{dimensions}\n")
        stream.write("1 1 2.0\n")


def test_discover_reads_headers_and_ignores_near_matches(tmp_path: Path) -> None:
    matrix = tmp_path / "10x_synth_20000g_1000c_sp99" / "matrix.mtx.gz"
    write_header(matrix)
    write_header(tmp_path / "10x_synth_20000g_3c_sp90" / "matrix.mtx.gz")
    write_header(tmp_path / "other" / "matrix.mtx.gz")

    datasets = benchmark.discover_datasets(tmp_path)

    assert list(datasets) == [1000]
    assert datasets[1000].path == matrix.resolve()
    assert (datasets[1000].rows, datasets[1000].cols, datasets[1000].nnz) == (
        20_000,
        1000,
        17,
    )
    assert datasets[1000].density == 17 / 20_000_000
    assert datasets[1000].compressed_bytes == matrix.stat().st_size


def test_sample_stats_uses_interpolated_percentiles() -> None:
    stats = benchmark.sample_stats([4.0, 1.0, 3.0, 2.0])

    assert stats == {
        "query_median_sec": 2.5,
        "query_p10_sec": pytest.approx(1.3),
        "query_p90_sec": pytest.approx(3.7),
        "query_min_sec": 1.0,
        "query_max_sec": 4.0,
    }


def test_prepare_plain_mtx_is_bounded_reusable_and_never_overwrites(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mtx.gz"
    destination = tmp_path / "prepared" / "matrix.mtx"
    write_header(source, "20000 1000 1")

    prepared, elapsed = benchmark.prepare_plain_mtx(source, destination, reuse=False)

    assert prepared == destination.resolve()
    assert elapsed >= 0
    assert benchmark.read_mtx_header(prepared) == (20_000, 1000, 1)
    with pytest.raises(FileExistsError, match="reuse-prepared-mtx"):
        benchmark.prepare_plain_mtx(source, destination, reuse=False)
    reused, reuse_elapsed = benchmark.prepare_plain_mtx(source, destination, reuse=True)
    assert reused == prepared
    assert reuse_elapsed == 0.0


def test_native_worker_branches_before_scipy_and_reports_all_phases() -> None:
    source = inspect.getsource(benchmark._worker)

    assert source.index('backend_name == "duckdb-native"') < source.index(
        "from scipy.io import mmread"
    )
    for column in (
        "prepare_sec",
        "checkpoint_sec",
        "reopen_sec",
        "wrap_sec",
        "cold_query_sec",
        "host_coo_materialized",
        "scipy_loaded",
        "last_phase",
    ):
        assert column in benchmark.RESULT_COLUMNS


@pytest.mark.parametrize(
    ("field", "symmetry", "message"),
    [("complex", "general", "real or integer"), ("real", "symmetric", "general")],
)
def test_header_rejects_unsupported_field_and_symmetry(
    tmp_path: Path, field: str, symmetry: str, message: str
) -> None:
    path = tmp_path / "matrix.mtx.gz"
    write_header(path, field=field, symmetry=symmetry)

    with pytest.raises(ValueError, match=message):
        benchmark.read_mtx_header(path)


def test_header_rejects_array_format(tmp_path: Path) -> None:
    path = tmp_path / "matrix.mtx.gz"
    write_header(path, storage_format="array")

    with pytest.raises(ValueError, match="coordinate"):
        benchmark.read_mtx_header(path)


def test_run_refuses_overwrite_and_resume_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, output_dir = tmp_path / "data", tmp_path / "output"
    write_header(data_dir / "10x_synth_20000g_1000c_sp99" / "matrix.mtx.gz")
    args = benchmark.build_parser().parse_args(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--sizes",
            "1000",
            "--backends",
            "numpy-host",
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "run_case",
        lambda *positional, **keywords: {
            **benchmark._base_row(
                benchmark.discover_datasets(data_dir)[1000], "numpy-host"
            ),
            "status": "ok",
            "error": "",
            "requested_spill_dir": str(output_dir / "worker-tmp" / "requested"),
            "effective_spill_dir": str(output_dir / "worker-tmp" / "requested"),
            "threads": keywords["threads"],
            "engine_memory_gb": keywords["engine_memory_gb"],
            "process_rss_limit_gb": keywords["process_rss_limit_gb"],
        },
    )
    assert benchmark.run_benchmark(args) == 0
    with pytest.raises(FileExistsError, match="--resume"):
        benchmark.run_benchmark(args)

    resume_args = benchmark.build_parser().parse_args(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--sizes",
            "1000",
            "--backends",
            "numpy-host",
            "--resume",
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "run_case",
        lambda *args, **kwargs: pytest.fail("completed case should be skipped"),
    )
    assert benchmark.run_benchmark(resume_args) == 0
    document = json.loads((output_dir / "results.json").read_text())
    assert len(document["results"]) == 1
    assert document["results"][0]["requested_spill_dir"].startswith(str(output_dir))
    assert document["results"][0]["threads"] == 2
    assert document["results"][0]["engine_memory_gb"] == 1.0
    assert document["results"][0]["process_rss_limit_gb"] == 5.0


def test_resume_retries_and_replaces_failed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, output_dir = tmp_path / "data", tmp_path / "output"
    write_header(data_dir / "10x_synth_20000g_1000c_sp99" / "matrix.mtx.gz")
    base_argv = [
        "run",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--sizes",
        "1000",
        "--backends",
        "numpy-host",
    ]
    dataset = benchmark.discover_datasets(data_dir)[1000]
    monkeypatch.setattr(
        benchmark,
        "run_case",
        lambda *args, **kwargs: {
            **benchmark._base_row(dataset, "numpy-host"),
            "status": "error",
            "error": "synthetic failure",
        },
    )
    assert benchmark.run_benchmark(benchmark.build_parser().parse_args(base_argv)) == 1
    (output_dir / "results.csv").write_text("stale csv\n")

    monkeypatch.setattr(
        benchmark,
        "run_case",
        lambda *args, **kwargs: {
            **benchmark._base_row(dataset, "numpy-host"),
            "status": "ok",
            "error": "",
        },
    )
    resume_args = benchmark.build_parser().parse_args([*base_argv, "--resume"])
    assert benchmark.run_benchmark(resume_args) == 0
    document = json.loads((output_dir / "results.json").read_text())
    assert [(row["backend"], row["status"]) for row in document["results"]] == [
        ("numpy-host", "ok")
    ]
    assert "stale csv" not in (output_dir / "results.csv").read_text()


def test_backend_order_rotates_by_dataset_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, output_dir = tmp_path / "data", tmp_path / "output"
    write_header(data_dir / "10x_synth_20000g_1000c_sp99" / "matrix.mtx.gz")
    write_header(
        data_dir / "10x_synth_20000g_3000c_sp99" / "matrix.mtx.gz",
        "20000 3000 19",
    )
    observed: list[tuple[int, str, int]] = []

    def fake_run(dataset, backend, **kwargs):
        observed.append((dataset.size, backend, kwargs["case_order"]))
        return {
            **benchmark._base_row(dataset, backend),
            "status": "ok",
            "error": "",
            "case_order": kwargs["case_order"],
            "case_order_policy": kwargs["case_order_policy"],
        }

    monkeypatch.setattr(benchmark, "run_case", fake_run)
    args = benchmark.build_parser().parse_args(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--sizes",
            "1000,3000",
        ]
    )

    assert benchmark.run_benchmark(args) == 0
    assert observed == [
        (1000, "numpy-host", 0),
        (1000, "duckdb-host", 1),
        (1000, "datafusion-host", 2),
        (1000, "duckdb-native", 3),
        (3000, "duckdb-host", 4),
        (3000, "datafusion-host", 5),
        (3000, "duckdb-native", 6),
        (3000, "numpy-host", 7),
    ]


def test_parent_rss_guard_overrides_queued_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PipeEnd:
        def close(self) -> None:
            pass

    class Receiving(PipeEnd):
        available = True

        def poll(self, timeout: float) -> bool:
            return self.available

        def recv(self) -> dict[str, str]:
            self.available = False
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
    dataset = benchmark.Dataset(1, tmp_path / "input", 1, 1, 1, 1)

    row = benchmark.run_case(
        dataset,
        "numpy-host",
        repeats=5,
        threads=2,
        engine_memory_gb=1.0,
        rss_limit_bytes=1024,
        process_rss_limit_gb=1 / 1024**3,
        timeout_sec=300,
        output_dir=tmp_path,
        case_order=0,
        case_order_policy="test",
        native_input="gzip",
        reuse_prepared_mtx=False,
    )

    assert row["status"] == "rss_guard"
    assert "RSS exceeded" in row["error"]
    assert "worker_starting" in row["error"]
    assert row["last_phase"] == "worker_starting"
    assert "sampled_via_ps" in row["rss_scope"]


def test_svg_plot_contains_panels_whiskers_and_failure_markers(tmp_path: Path) -> None:
    rows = [
        {
            "backend": "numpy-host",
            "status": "ok",
            "nnz": 100,
            "ready_sec": 0.2,
            "query_median_sec": 0.02,
            "query_p10_sec": 0.01,
            "query_p90_sec": 0.03,
        },
        {
            "backend": "duckdb-host",
            "status": "timeout",
            "nnz": 1000,
        },
        {
            "backend": "datafusion-host",
            "status": "rss_guard",
            "nnz": 10_000,
        },
        {
            "backend": "duckdb-native",
            "status": "error",
            "nnz": 3000,
        },
    ]
    output = tmp_path / "plot.svg"

    benchmark.write_svg(rows, output)

    svg = output.read_text()
    assert "Execution path → ready (preparation excluded)" in svg
    assert "Warm colMeans time" in svg
    assert "aria-labelledby" in svg
    assert "Whiskers show" in svg
    assert "duckdb-host timeout" in svg
    assert "datafusion-host RSS guard" in svg
    assert "duckdb-native error" in svg
    ET.fromstring(svg)
    with pytest.raises(FileExistsError):
        benchmark.write_svg(rows, output)


def test_plot_loader_concatenates_json_and_csv_inputs(tmp_path: Path) -> None:
    json_path = tmp_path / "small.json"
    csv_path = tmp_path / "large.csv"
    json_path.write_text(
        json.dumps(
            {"results": [{"size": 100_000, "backend": "numpy-host", "status": "ok"}]}
        )
    )
    csv_path.write_text("size,backend,status\n3000000,datafusion-host,rss_guard\n")

    rows = benchmark._load_plot_rows([json_path, csv_path])

    assert [(int(row["size"]), row["backend"]) for row in rows] == [
        (100_000, "numpy-host"),
        (3_000_000, "datafusion-host"),
    ]
    assert benchmark._load_plot_rows(json_path) == rows[:1]


def test_plot_loader_rejects_duplicate_points_across_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.csv"
    first.write_text(
        json.dumps(
            {"results": [{"size": 1000, "backend": "duckdb-host", "status": "ok"}]}
        )
    )
    second.write_text("size,backend,status\n1000,duckdb-host,timeout\n")

    with pytest.raises(
        ValueError, match=r"duplicate plot point size=1000 backend='duckdb-host'"
    ):
        benchmark._load_plot_rows([first, second])


def test_plot_and_resume_contract_reject_legacy_ambiguous_results(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "config": {"schema_version": 2},
                "results": [{"size": 1000, "backend": "duckdb", "status": "ok"}],
            }
        )
    )

    with pytest.raises(ValueError, match="incompatible benchmark schema"):
        benchmark._load_plot_rows(legacy)

    ambiguous_csv = tmp_path / "legacy.csv"
    ambiguous_csv.write_text("size,backend,status\n1000,duckdb,ok\n")
    with pytest.raises(ValueError, match="explicit host/native label"):
        benchmark._load_plot_rows(ambiguous_csv)


def test_command_defaults_and_validation() -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(["run", "--data-dir", "data", "--output-dir", "out"])
    assert args.sizes == benchmark.DEFAULT_SIZES
    assert args.backends == benchmark.DEFAULT_BACKENDS
    assert args.repeats == 5
    assert args.threads == 2
    assert args.engine_memory_gb == 1.0
    assert args.rss_limit_gb == 5.0
    assert args.timeout_sec == 300.0
    assert args.native_input == "gzip"
    assert args.reuse_prepared_mtx is False

    plot_args = parser.parse_args(
        ["plot", "--input", "one.json", "two.csv", "--output", "plot.svg"]
    )
    assert plot_args.input == [Path("one.json"), Path("two.csv")]

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--data-dir",
                "data",
                "--output-dir",
                "out",
                "--backends",
                "numpy,unknown",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--data-dir",
                "data",
                "--output-dir",
                "out",
                "--repeats",
                "101",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--data-dir",
                "data",
                "--output-dir",
                "out",
                "--rss-limit-gb",
                "0",
            ]
        )
