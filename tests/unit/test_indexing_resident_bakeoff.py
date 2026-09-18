from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "indexing_resident_bakeoff.py"
SPEC = importlib.util.spec_from_file_location("indexing_resident_bakeoff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def fixture(path: Path) -> None:
    i = np.repeat(np.arange(12, dtype=np.int64), 3)
    j = np.tile(np.array([0, 7, 19], dtype=np.int64), 12)
    x = np.arange(1, len(i) + 1, dtype=np.float64)
    pq.write_table(pa.table({"i": i, "j": j, "x": x}), path)


def test_execution_schedule_balances_complete_chronological_sequence() -> None:
    orders = [benchmark.balanced_execution_order(repeat) for repeat in range(8)]
    for position in range(len(benchmark.PATHS)):
        assert sorted(order[position] for order in orders) == sorted(
            benchmark.PATHS * 2
        )
    warmup = [
        benchmark.PATHS[position]
        for position in benchmark._BALANCED_WARMUP_ORDER  # noqa: SLF001
    ]
    chronological = warmup[-1:] + [path for order in orders for path in order]
    predecessor_counts: dict[tuple[str, str], int] = {}
    for pair in zip(chronological, chronological[1:], strict=False):
        predecessor_counts[pair] = predecessor_counts.get(pair, 0) + 1
    assert all(left != right for left, right in predecessor_counts)
    assert len(predecessor_counts) == 12
    assert set(predecessor_counts.values()) == {2, 3}


def test_fixed_controls_reject_an_unrelated_semantic_tree() -> None:
    from dbnumpy.ir import Source, StorageKind

    source = Source("s", "r", "c", (2, 3), "float64", StorageKind.SPARSE)
    with pytest.raises(ValueError, match=r"abs\(gather\)"):
        benchmark.direct_scalar_sql(source)


@pytest.mark.parametrize("backend", benchmark.BACKENDS)
@pytest.mark.parametrize("axis", [0, 1])
def test_resident_case_matches_all_paths_and_retains_samples(
    tmp_path: Path, backend: str, axis: int
) -> None:
    parquet = tmp_path / "canonical.parquet"
    fixture(parquet)
    case = benchmark.run_case(
        backend,
        parquet=parquet,
        shape=(12, 20),
        selector_size=5,
        repeats=2,
        seed=1,
        axis=axis,
    )

    assert [row["path"] for row in case["paths"]] == list(benchmark.PATHS)
    assert case["axis"] == axis
    assert case["execution_orders"] == [
        benchmark.balanced_execution_order(0),
        benchmark.balanced_execution_order(1),
    ]
    counts = {row["stored_gather_rows"] for row in case["paths"]}
    assert len(counts) == 1
    for row in case["paths"]:
        assert len(row["execute_collect"]["samples_ms"]) == 2
        assert len(row["scalar_samples"]) == 2
        assert row["scalar_samples"][0] > 0
        assert len(row["explain"]["sha256"]) == 64
        assert row["explain"]["text"]
    by_path = {row["path"]: row for row in case["paths"]}
    assert "COALESCE" in by_path["dbverse_ibis"]["sql"]
    assert "COALESCE" not in by_path["sqlalchemy_core"]["sql"]
    assert "COALESCE" not in by_path["direct_sql"]["sql"]


def test_document_records_provenance_and_is_json_serializable(tmp_path: Path) -> None:
    parquet = tmp_path / "canonical.parquet"
    fixture(parquet)
    args = benchmark.build_parser().parse_args(
        [
            "--parquet",
            str(parquet),
            "--rows",
            "12",
            "--cols",
            "20",
            "--selector-size",
            "4",
            "--repeats",
            "1",
            "--backends",
            "duckdb",
        ]
    )
    resolved = benchmark.validate_args(benchmark.build_parser(), args)
    document = benchmark.run_benchmark(args, resolved)

    encoded = json.dumps(document)
    assert "fixed_workload_generic_sql_probe" in encoded
    assert document["input"]["rows"] == 36
    assert len(document["input"]["sha256"]) == 64
    assert len(document["host"]["source_tree_sha256"]) == 64
    assert document["git"]["commit"]
    assert document["config"]["execution_cache_state"].startswith("warm")
    assert document["fixture_reference"]["expected_scalar"] > 0


def test_streaming_reference_is_independent_and_shape_checked(tmp_path: Path) -> None:
    parquet = tmp_path / "canonical.parquet"
    fixture(parquet)
    selector = np.array([0, 0, 1], dtype=np.int64)
    reference = benchmark.inspect_fixture(
        parquet, shape=(12, 20), selector=selector
    )

    values = np.arange(1, 37, dtype=np.float64).reshape(12, 3)
    expected = (np.abs(values[[0, 0, 1], :]) * 1.5).sum()
    np.testing.assert_allclose(reference["expected_scalar"], expected)
    assert reference["expected_stored_gather_rows"] == 9
    with pytest.raises(ValueError, match="exactly match"):
        benchmark.inspect_fixture(parquet, shape=(12, 21), selector=selector)

    null_parquet = tmp_path / "null.parquet"
    pq.write_table(
        pa.table(
            {
                "i": pa.array([0], type=pa.int64()),
                "j": pa.array([0], type=pa.int64()),
                "x": pa.array([None], type=pa.float64()),
            }
        ),
        null_parquet,
    )
    with pytest.raises(ValueError, match="NULL"):
        benchmark.inspect_fixture(
            null_parquet, shape=(1, 1), selector=np.array([0], dtype=np.int64)
        )

    nonfinite_parquet = tmp_path / "nonfinite.parquet"
    pq.write_table(
        pa.table({"i": [0], "j": [0], "x": [float("nan")]}),
        nonfinite_parquet,
    )
    with pytest.raises(ValueError, match="finite"):
        benchmark.inspect_fixture(
            nonfinite_parquet,
            shape=(1, 1),
            selector=np.array([0], dtype=np.int64),
        )

    column_selector = np.array([0, 7, 19], dtype=np.int64)
    column_reference = benchmark.inspect_fixture(
        parquet, shape=(12, 20), selector=column_selector, axis=1
    )
    assert column_reference["expected_stored_gather_rows"] == 36
    assert column_reference["axis"] == 1


def test_validation_refuses_missing_unsafe_and_overwrite(
    tmp_path: Path,
) -> None:
    parser = benchmark.build_parser()
    missing = parser.parse_args(["--parquet", str(tmp_path / "missing.parquet")])
    with pytest.raises(SystemExit):
        benchmark.validate_args(parser, missing)

    parquet = tmp_path / "canonical.parquet"
    fixture(parquet)
    unsafe = parser.parse_args(["--parquet", str(parquet), "--selector-size", "100001"])
    with pytest.raises(SystemExit):
        benchmark.validate_args(parser, unsafe)

    output = tmp_path / "result.json"
    output.write_text("existing")
    overwrite = parser.parse_args(["--parquet", str(parquet), "--output", str(output)])
    with pytest.raises(SystemExit):
        benchmark.validate_args(parser, overwrite)

    scan_unsafe = parser.parse_args(
        [
            "--parquet",
            str(parquet),
            "--rows",
            "12",
            "--cols",
            "20",
            "--max-estimated-scan-bytes",
            "1",
        ]
    )
    with pytest.raises(SystemExit):
        benchmark.validate_args(parser, scan_unsafe)
