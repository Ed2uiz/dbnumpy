from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "indexing_lowering_bakeoff.py"
SPEC = importlib.util.spec_from_file_location("indexing_lowering_bakeoff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_selector_patterns_are_bounded_deterministic_and_preserve_duplicates() -> None:
    ordered = benchmark.make_selector("ordered", 5, 8, seed=1)
    shuffled = benchmark.make_selector("shuffled", 5, 8, seed=1)
    repeated = benchmark.make_selector("repeated", 20, 8, seed=1)

    np.testing.assert_array_equal(ordered, np.arange(5, dtype=np.int64))
    np.testing.assert_array_equal(
        shuffled, benchmark.make_selector("shuffled", 5, 8, seed=1)
    )
    assert np.all((shuffled >= 0) & (shuffled < 8))
    assert len(np.unique(repeated)) < len(repeated)
    np.testing.assert_array_equal(benchmark.normalize_selector(ordered, 8), ordered)


def test_registered_relation_sql_is_bounded_by_names_not_selector_payload() -> None:
    from dbnumpy.ir import Source, StorageKind

    source = Source(
        "source",
        "source_rows",
        "source_cols",
        (3, 4),
        "float64",
        StorageKind.SPARSE,
    )
    expr = benchmark.build_gather(
        source, "selector", (2, 4), "output_rows", "output_cols"
    )
    direct = benchmark.direct_gather_sql(expr)
    statement = benchmark.sqlalchemy_statement(expr)
    sqlalchemy_sql = benchmark.compile_sqlalchemy(statement)

    assert "VALUES" not in direct.upper()
    assert "VALUES" not in sqlalchemy_sql.upper()
    assert "source_index" in direct and "output_index" in direct
    assert len(direct.encode()) < 512
    assert len(sqlalchemy_sql.encode()) < 512


def test_execution_order_balances_position_and_predecessor() -> None:
    calls = {name: 0 for name in benchmark.PATHS}
    call_order: list[str] = []

    def executor(name: str) -> pa.Table:
        calls[name] += 1
        call_order.append(name)
        return pa.table({"i": [0], "j": [0], "x": [1.0]})

    timings, orders = benchmark.rotated_execution(
        {
            name: (lambda path=name: executor(path))
            for name in benchmark.PATHS
        },
        repeats=8,
    )

    assert {order[0] for order in orders} == set(benchmark.PATHS)
    for position in range(len(benchmark.PATHS)):
        assert sorted(order[position] for order in orders) == sorted(
            benchmark.PATHS * 2
        )
    chronological = call_order[len(benchmark.PATHS) - 1 :]
    predecessor_counts: dict[tuple[str, str], int] = {}
    for pair in zip(chronological, chronological[1:], strict=False):
        predecessor_counts[pair] = predecessor_counts.get(pair, 0) + 1
    assert all(left != right for left, right in predecessor_counts)
    assert len(predecessor_counts) == 12
    assert set(predecessor_counts.values()) == {2, 3}
    assert all(calls[name] == 9 for name in benchmark.PATHS)
    assert all(len(timings[name].samples_ms) == 8 for name in benchmark.PATHS)


@pytest.mark.parametrize("backend_name", benchmark.BACKENDS)
def test_all_paths_return_exact_repeated_gather(backend_name: str) -> None:
    case = benchmark.run_case(
        backend_name,
        source_rows=12,
        source_cols=5,
        selector_size=7,
        pattern="repeated",
        storage="sparse",
        repeats=1,
        seed=7,
    )

    assert [row["path"] for row in case["paths"]] == list(benchmark.PATHS)
    assert all(row["correctness"] == "exact" for row in case["paths"])
    assert all(row["selector_size"] == 7 for row in case["paths"])
    assert case["execution_orders"] == [benchmark.balanced_execution_order(0)]
    sql_rows = [row for row in case["paths"] if row["sql_bytes"] is not None]
    assert all(row["sql_bytes"] < 1_000 and row["sql_sha256"] for row in sql_rows)


def test_document_is_json_serializable_and_records_fairness() -> None:
    args = benchmark.build_parser().parse_args(
        [
            "--backends",
            "duckdb",
            "--patterns",
            "ordered",
            "--selector-sizes",
            "3",
            "--source-rows",
            "8",
            "--source-cols",
            "4",
            "--repeats",
            "1",
            "--format",
            "json",
        ]
    )
    document = benchmark.run_benchmark(args)
    encoded = json.dumps(document)

    assert "generic_sql_subset_probe_only" in encoded
    assert document["config"]["serial"] is True
    assert "native_execution" in document["fairness"]


def test_cli_rejects_unsafe_or_ambiguous_inputs() -> None:
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--selector-sizes", "2,2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--backends", "unknown"])

    args = parser.parse_args(["--selector-sizes", "1000000", "--source-cols", "11"])
    with pytest.raises(SystemExit):
        benchmark.validate_args(parser, args)
    source_too_large = parser.parse_args(
        ["--source-rows", "2000001", "--source-cols", "1"]
    )
    with pytest.raises(SystemExit):
        benchmark.validate_args(parser, source_too_large)
