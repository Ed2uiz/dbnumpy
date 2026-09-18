"""Measure deep pointwise pipelines composed around a relational transform."""

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


def chain(source: DBArray, depth: int = 100) -> DBArray:
    result = source
    for _ in range(depth):
        result = np.sqrt(result * 1.000001 + 0.000001)
    return result


def benchmark_backend(backend_type: Any, *, repeats: int) -> list[dict[str, Any]]:
    backend = backend_type.connect(max_densify_cells=10_000)
    try:
        values = np.arange(64.0).reshape(8, 8) + 1.0
        source = backend.from_numpy(values)
        cases = {
            "pointwise_then_transpose": (
                lambda: chain(source).T,
                lambda: chain_reference(values).T,
            ),
            "transpose_then_pointwise": (
                lambda: chain(source.T),
                lambda: chain_reference(values.T),
            ),
        }
        rows: list[dict[str, Any]] = []
        for case, (build, reference) in cases.items():
            build_ms, plan = timed(build, repeats=repeats)
            compile_ms, sql = timed(
                lambda plan=plan: backend.lowerer.compile_matrix(
                    plan._expr,  # noqa: SLF001 - compiler benchmark boundary
                    dialect=backend.dialect,
                ),
                repeats=repeats,
            )
            execute_ms, result = timed(plan.to_numpy, repeats=repeats)
            np.testing.assert_allclose(result, reference(), rtol=1e-11, atol=1e-11)
            rows.append(
                {
                    "backend": backend.name,
                    "case": case,
                    "build_ms": build_ms,
                    "compile_ms": compile_ms,
                    "execute_ms": execute_ms,
                    "sql_bytes": len(sql.encode()),
                    "source_references": sql.count('"dbm_0"'),
                }
            )
        return rows
    finally:
        backend.close()


def chain_reference(source: np.ndarray, depth: int = 100) -> np.ndarray:
    result = source
    for _ in range(depth):
        result = np.sqrt(result * 1.000001 + 0.000001)
    return result


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
