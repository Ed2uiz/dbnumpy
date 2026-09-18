"""Measure lowering growth for a repeatedly reused pointwise expression DAG.

The workload keeps its numerical value unchanged while doubling the number of
naive tree paths at every stage. It is deliberately tiny and serial: the goal
is compiler-shape evidence, not backend throughput.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from dbnumpy.backends import DataFusionBackend, DuckDBBackend
from dbnumpy.matrix import DBArray


def timed(callable_: Any, *, repeats: int) -> tuple[float, Any]:
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = callable_()
        durations.append(perf_counter() - start)
    return median(durations) * 1_000, result


def build_shared_dag(source: DBArray, depth: int) -> DBArray:
    result = source
    for _ in range(depth):
        result = (result + result) * 0.5
    return result


def benchmark_backend(backend_type: Any, *, repeats: int) -> list[dict[str, Any]]:
    backend = backend_type.connect(max_densify_cells=10_000)
    try:
        values = np.arange(64.0).reshape(8, 8) + 1.0
        source = backend.from_numpy(values)
        rows: list[dict[str, Any]] = []
        for depth in (1, 10, 25, 100):
            build_ms, plan = timed(
                lambda depth=depth: build_shared_dag(source, depth),
                repeats=repeats,
            )
            compile_ms, sql = timed(
                lambda plan=plan: backend.lowerer.compile_matrix(
                    plan._expr,  # noqa: SLF001 - compiler benchmark boundary
                    dialect=backend.dialect,
                ),
                repeats=repeats,
            )
            execute_ms, result = timed(plan.to_numpy, repeats=repeats)
            np.testing.assert_allclose(result, values)
            rows.append(
                {
                    "backend": backend.name,
                    "depth": depth,
                    "build_ms": build_ms,
                    "compile_ms": compile_ms,
                    "execute_ms": execute_ms,
                    "sql_bytes": len(sql.encode()),
                    "alias_stages": sql.count(' AS "__dbm_pointwise_value_'),
                    "source_references": sql.count('"dbm_0"'),
                }
            )
        return rows
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    result = {
        "versions": {
            package: version(package)
            for package in ("dbnumpy", "duckdb", "datafusion")
        },
        "repeats": args.repeats,
        "results": [
            *benchmark_backend(DuckDBBackend, repeats=args.repeats),
            *benchmark_backend(DataFusionBackend, repeats=args.repeats),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
