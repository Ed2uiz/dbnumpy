"""Measure semantic-plan construction, SQL lowering, and tiny-query execution.

This benchmark is intentionally small and serial. It isolates frontend/compiler
cost without creating memory pressure on a 16 GB development machine.
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
    return median(durations), result


def build_chain(source: DBArray, depth: int) -> DBArray:
    result = source
    for _ in range(depth):
        result = np.sqrt(result * 1.000001 + 0.000001)
    return result


def benchmark_backend(backend_type: Any, *, repeats: int) -> list[dict[str, Any]]:
    backend = backend_type.connect(max_densify_cells=100_000)
    try:
        source = backend.from_numpy(np.arange(256.0).reshape(16, 16) + 1.0)
        rows: list[dict[str, Any]] = []
        for depth in (1, 10, 100):
            build_seconds, plan = timed(
                lambda depth=depth: build_chain(source, depth), repeats=repeats
            )
            compile_seconds, sql = timed(
                lambda plan=plan: backend.lowerer.compile_matrix(
                    plan._expr,
                    dialect=backend.dialect,  # noqa: SLF001
                ),
                repeats=repeats,
            )
            execute_seconds, result = timed(plan.to_numpy, repeats=repeats)
            expected = np.arange(256.0).reshape(16, 16) + 1.0
            for _ in range(depth):
                expected = np.sqrt(expected * 1.000001 + 0.000001)
            np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-11)
            rows.append(
                {
                    "backend": backend.name,
                    "depth": depth,
                    "build_ms": build_seconds * 1_000,
                    "compile_ms": compile_seconds * 1_000,
                    "execute_ms": execute_seconds * 1_000,
                    "sql_bytes": len(sql.encode()),
                    "join_tokens": sql.upper().count(" JOIN "),
                    "source_scans": sql.count('FROM "dbm_0"'),
                }
            )
        return rows
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    result = {
        "versions": {
            package: version(package)
            for package in ("dbnumpy", "ibis-framework", "duckdb", "datafusion")
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
