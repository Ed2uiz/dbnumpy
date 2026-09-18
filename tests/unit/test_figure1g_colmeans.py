from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "figure1g_colmeans.py"
SPEC = importlib.util.spec_from_file_location("figure1g_colmeans", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_figure1g_series_is_an_explicit_audit_contract() -> None:
    assert [
        (spec.columns, spec.sparsity, spec.nnz, spec.expected_mean)
        for spec in benchmark.FIGURE1G_SERIES
    ] == [
        (1_000, "sp95", 1_000_000, 0.05),
        (3_000, "sp95", 3_000_000, 0.05),
        (10_000, "sp95", 10_000_000, 0.05),
        (30_000, "sp95", 30_000_000, 0.05),
        (100_000, "sp95", 100_000_000, 0.05),
        (1_000_000, "sp99", 200_000_000, 0.01),
        (3_000_000, "sp99", 600_000_000, 0.01),
        (10_000_000, "sp99", 2_000_000_000, 0.01),
    ]


def test_parse_sizes_supports_stages_and_largest_case() -> None:
    assert benchmark.parse_sizes("1000,10000") == (1_000, 10_000)
    assert benchmark.parse_sizes("all")[-1] == 10_000_000
    with pytest.raises(ValueError, match="unknown Figure 1g sizes"):
        benchmark.parse_sizes("2000")
    with pytest.raises(ValueError, match="unique"):
        benchmark.parse_sizes("1000,1000")


def test_emergency_threshold_is_not_the_engine_budget() -> None:
    physical = 16 * 1024**3
    assert benchmark.default_emergency_rss_bytes(physical) == int(physical * 0.8)
    args = benchmark.parse_args(
        ["--data-dir", "/data", "--output-dir", "/results"]
    )
    assert args.engine_memory_gb == 1.0
    assert args.duckdb_ingest_memory_gb is None
    assert benchmark.resolve_duckdb_ingest_memory_gb(1.0, None) == 1.0
    assert benchmark.resolve_duckdb_ingest_memory_gb(1.0, 8.0) == 8.0
    assert args.emergency_rss_gb is None


def test_macos_swap_snapshot_tracks_system_swap_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "total = 21504.00M  used = 18907.31M  free = 2596.69M"
        ),
    )

    assert benchmark.swap_snapshot() == {
        "total": 21_504 * 1024**2,
        "used": int(18_907.31 * 1024**2),
        "free": int(2_596.69 * 1024**2),
    }


def test_swap_snapshot_handles_units_platform_and_sysctl_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda *args, **kwargs: "total = 2.00G used = 512.00M free = 1.50G",
    )
    assert benchmark.swap_snapshot() == {
        "total": 2 * 1024**3,
        "used": 512 * 1024**2,
        "free": int(1.5 * 1024**3),
    }

    monkeypatch.setattr(benchmark.platform, "system", lambda: "Linux")
    assert benchmark.swap_snapshot() is None

    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")

    def fail(*args: object, **kwargs: object) -> str:
        raise benchmark.subprocess.CalledProcessError(1, "sysctl")

    monkeypatch.setattr(benchmark.subprocess, "check_output", fail)
    assert benchmark.swap_snapshot() is None


def test_swap_safety_requires_high_use_and_low_free_space() -> None:
    gib = 1024**3
    limits = {"minimum_free_bytes": gib, "minimum_used_bytes": 12 * gib}
    assert not benchmark.swap_safety_limit_reached(
        {"total": 0, "used": 0, "free": 0}, **limits
    )
    assert not benchmark.swap_safety_limit_reached(
        {"total": 2 * gib, "used": gib, "free": gib // 2}, **limits
    )
    assert not benchmark.swap_safety_limit_reached(
        {"total": 20 * gib, "used": 18 * gib, "free": 2 * gib}, **limits
    )
    assert benchmark.swap_safety_limit_reached(
        {"total": 20 * gib, "used": 19 * gib, "free": gib // 2}, **limits
    )


def test_discovery_requires_exact_figure1g_path_and_header(tmp_path: Path) -> None:
    spec = benchmark.SPEC_BY_COLUMNS[1_000]
    directory = tmp_path / spec.directory_name
    directory.mkdir()
    matrix = directory / "matrix.mtx"
    matrix.write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "% generated fixture\n"
        f"20000 1000 {spec.nnz}\n"
    )
    [dataset] = benchmark.discover_datasets(tmp_path, (1_000,))
    assert dataset.path == matrix.resolve()
    assert dataset.spec is spec

    matrix.write_text(
        "%%MatrixMarket matrix coordinate integer general\n20000 1000 999999\n"
    )
    with pytest.raises(ValueError, match="header"):
        benchmark.discover_datasets(tmp_path, (1_000,))


def test_structural_zero_reference_checks_full_column_vector() -> None:
    spec = benchmark.Figure1GSpec(3, "sp95", 3_000, 0.05)
    dataset = benchmark.Dataset(spec, Path("unused.mtx"), 0)
    result = benchmark._validate_result(np.array([0.05, 0.05, 0.05]), dataset)
    assert result["max_abs_error"] == 0
    assert result["reference_semantics"] == "SUM(x) / 20000; structural zeros included"
    with pytest.raises(ValueError, match="max error"):
        benchmark._validate_result(np.array([1.0, 1.0, 1.0]), dataset)


def test_manifest_reuse_requires_an_exact_match_and_never_overwrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.manifest.json"
    expected = {"source": "/matrix.mtx", "nnz": 10}
    benchmark._write_manifest(path, expected)
    benchmark._load_matching_manifest(path, expected)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        benchmark._write_manifest(path, expected)
    with pytest.raises(ValueError, match="does not match"):
        benchmark._load_matching_manifest(path, {"source": "/other.mtx"})


def test_results_include_json_and_csv_without_losing_nested_telemetry(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "columns": 1_000,
            "backend": "duckdb-mtx-table",
            "status": "success",
            "disk_before": {"disk_free_bytes": 123},
            "warm_query_samples_sec": [1.0] * 5,
        }
    ]
    run_config = {"engine_memory_budget_bytes": 1024**3, "timeout_seconds": 60}
    benchmark.write_results(rows, tmp_path, run_config)
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload == {
        "schema_version": 1,
        "run_config": run_config,
        "results": rows,
    }
    csv_text = (tmp_path / "results.csv").read_text()
    assert '""disk_free_bytes"": 123' in csv_text


def test_backend_labels_describe_the_full_ingestion_path() -> None:
    assert "SciPy sparse" in benchmark.BACKEND_LABELS["scipy-host"]
    assert "MTX ingested" in benchmark.BACKEND_LABELS["duckdb-mtx-table"]
    assert "external canonical Parquet" in benchmark.BACKEND_LABELS[
        "datafusion-parquet-scan"
    ]


def test_scipy_two_billion_payload_lower_bounds_are_explicit() -> None:
    bounds = benchmark.scipy_memory_lower_bounds(2_000_000_000, 10_000_000)
    assert bounds == {
        "scipy_mmread_coo_lower_bound_bytes": 32_000_000_000,
        "scipy_csc_lower_bound_bytes": 24_040_000_004,
    }


def test_parquet_parts_are_regular_files_in_deterministic_order(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "canonical.parquet"
    directory.mkdir()
    second = directory / "part-02.parquet"
    first = directory / "part-01.parquet"
    second.write_bytes(b"PAR1")
    first.write_bytes(b"PAR1")
    assert benchmark._parquet_files(directory) == (first, second)
    assert benchmark._parquet_files(first) == (first,)


def test_dependency_free_plot_uses_direct_failure_text(tmp_path: Path) -> None:
    rows = [
        {
            "backend": "scipy-host",
            "columns": 10_000_000,
            "nnz": 2_000_000_000,
            "status": "system-emergency-safety-stop",
            **benchmark.scipy_memory_lower_bounds(2_000_000_000, 10_000_000),
        }
    ]
    destination = tmp_path / "plot.svg"
    benchmark.plot_results(
        rows,
        destination,
        backends=("scipy-host",),
        title="Figure 1g test",
    )
    svg = destination.read_text()
    assert "SciPy payload minimum" in svg
    assert "system-emergency-safety-stop" in svg
    assert "No operation timing" in svg
    assert 'aria-labelledby="operation-title operation-desc"' in svg
    assert "<svg" in svg
    ET.fromstring(svg)


def test_operation_plot_uses_full_direct_path_labels_without_legend(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "backend": backend,
            "columns": 1_000,
            "nnz": 1_000_000,
            "status": "success",
            "cold_query_sec": 0.1 + index,
            "warm_query_median_sec": 0.05 + index,
        }
        for index, backend in enumerate(benchmark.BACKENDS)
    ]
    destination = tmp_path / "operation.svg"
    benchmark.plot_results(
        rows,
        destination,
        backends=benchmark.BACKENDS,
        title="Column means: test paths",
    )
    svg = destination.read_text()
    for label in benchmark.BACKEND_LABELS.values():
        assert label in svg
    assert "Operation time only" in svg
    assert "both axes logarithmic" in svg
    assert "<legend" not in svg
    ET.fromstring(svg)


def test_preparation_plot_keeps_reuse_and_operation_semantics_separate(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "backend": "duckdb-parquet-scan",
            "nnz": 1_000_000,
            "status": "success",
            "preparation_sec": 2.5,
            "preparation_reused": False,
        },
        {
            "backend": "datafusion-parquet-scan",
            "nnz": 1_000_000,
            "status": "success",
            "preparation_sec": 0.0,
            "preparation_reused": True,
        },
    ]
    destination = tmp_path / "preparation.svg"
    benchmark.plot_preparation_results(
        rows,
        destination,
        backends=("duckdb-parquet-scan", "datafusion-parquet-scan"),
    )
    svg = destination.read_text()
    assert "Input preparation time" in svg
    assert "reused artifact case(s); preparation not retimed" in svg
    assert "mean(axis=0) and operation spill are not shown" in svg
    assert "zero-second measurements" in svg
    assert "warm_query" not in svg
    ET.fromstring(svg)


def test_duckdb_table_uses_ingest_then_query_budget_and_reuse_skips_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbnumpy import DuckDBBackend

    source = tmp_path / "matrix.mtx"
    source.write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "20000 3 3\n"
        "1 1 1\n2 2 1\n3 3 1\n"
    )
    dataset = benchmark.Dataset(
        benchmark.Figure1GSpec(3, "test", 3, 0.00005),
        source,
        source.stat().st_size,
    )
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    calls: list[str] = []
    original_connect = DuckDBBackend.connect

    def tracking_connect(cls: type[DuckDBBackend], *args: object, **kwargs: object):
        calls.append(str(kwargs["memory_limit"]))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(DuckDBBackend, "connect", classmethod(tracking_connect))
    matrix, owner, preparation = benchmark._open_case(
        "duckdb-mtx-table",
        dataset,
        case_dir,
        memory_bytes=32 * 1024**2,
        duckdb_ingest_memory_bytes=64 * 1024**2,
        threads=1,
        resume=False,
    )
    try:
        assert np.asarray(matrix.mean(axis=0)).tolist() == [0.00005] * 3
        assert calls == ["67108864B", "33554432B"]
        assert preparation["duckdb_ingest_memory_budget_bytes"] == 64 * 1024**2
        assert preparation["duckdb_query_memory_budget_bytes"] == 32 * 1024**2
        assert preparation["duckdb_preserve_insertion_order"] is True
        assert preparation["duckdb_mtx_reader_mode"] == "assume-canonical-direct-v1"
    finally:
        owner.close()

    calls.clear()
    matrix, owner, preparation = benchmark._open_case(
        "duckdb-mtx-table",
        dataset,
        case_dir,
        memory_bytes=32 * 1024**2,
        duckdb_ingest_memory_bytes=128 * 1024**2,
        threads=1,
        resume=True,
    )
    try:
        assert np.asarray(matrix.mean(axis=0)).tolist() == [0.00005] * 3
        assert calls == ["33554432B"]
        assert preparation["preparation_reused"] is True
        assert preparation["preparation_sec"] == 0.0
    finally:
        owner.close()


def test_duckdb_partial_artifact_is_never_opened_or_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbnumpy import DuckDBBackend

    source = tmp_path / "matrix.mtx"
    source.write_text(
        "%%MatrixMarket matrix coordinate integer general\n20000 1 1\n1 1 1\n"
    )
    dataset = benchmark.Dataset(
        benchmark.Figure1GSpec(1, "test", 1, 0.00005),
        source,
        source.stat().st_size,
    )
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    database = case_dir / "matrix.duckdb"
    database.write_bytes(b"do not replace")

    def forbidden_connect(*args: object, **kwargs: object) -> None:
        pytest.fail("partial artifact must be rejected before DuckDB connects")

    monkeypatch.setattr(DuckDBBackend, "connect", forbidden_connect)
    with pytest.raises(FileExistsError, match="partial DuckDB artifact"):
        benchmark._open_case(
            "duckdb-mtx-table",
            dataset,
            case_dir,
            memory_bytes=32 * 1024**2,
            duckdb_ingest_memory_bytes=64 * 1024**2,
            threads=1,
            resume=True,
        )
    assert database.read_bytes() == b"do not replace"
