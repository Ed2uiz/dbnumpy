"""Measure compilation and execution of the expanded reduction surface.

The diagnostic is serial, deterministic, and bounded for a 16 GB development
host. It compares results with NumPy while recording cold lowerer time, direct
SQL execution time, query size, and accidental full-domain plan markers.
"""

from __future__ import annotations

import argparse
import json
import warnings
from functools import partial
from importlib.metadata import version
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from dbnumpy.backends import Backend, DataFusionBackend, DuckDBBackend
from dbnumpy.ir import ReductionOp

_CASES: tuple[tuple[str, ReductionOp, int | None, float, Any], ...] = (
    ("scalar_min", ReductionOp.MIN, None, 0.0, np.min),
    ("scalar_var", ReductionOp.VAR, None, 0.0, np.var),
    ("column_max", ReductionOp.MAX, 0, 0.0, np.max),
    ("row_any", ReductionOp.ANY, 1, 0.0, np.any),
    ("scalar_nanmean", ReductionOp.NANMEAN, None, 0.0, np.nanmean),
    ("column_nanvar", ReductionOp.NANVAR, 0, 0.5, np.nanvar),
    ("row_nanmax", ReductionOp.NANMAX, 1, 0.0, np.nanmax),
)


def timed(callable_: Any, *, repeats: int) -> tuple[float, Any]:
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = callable_()
        durations.append(perf_counter() - start)
    return median(durations) * 1_000, result


def _values(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    sparse_input: bool,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    values = rng.normal(size=shape)
    if sparse_input:
        values[rng.random(size=shape) >= 0.03] = 0.0
    values[0, 0] = np.nan
    values[-1, -1] = -0.0
    values[0, -1] = np.inf
    return values


def _method_result(
    matrix: Any,
    operation: ReductionOp,
    axis: int | None,
    ddof: float,
) -> Any:
    method = getattr(matrix, operation.value)
    if operation in {
        ReductionOp.VAR,
        ReductionOp.STD,
        ReductionOp.NANVAR,
        ReductionOp.NANSTD,
    }:
        return method(axis=axis, ddof=ddof)
    return method(axis=axis)


def _reference_result(
    function: Any,
    values: np.ndarray[Any, np.dtype[np.float64]],
    operation: ReductionOp,
    axis: int | None,
    ddof: float,
) -> Any:
    kwargs: dict[str, Any] = {"axis": axis}
    if operation in {
        ReductionOp.VAR,
        ReductionOp.STD,
        ReductionOp.NANVAR,
        ReductionOp.NANSTD,
    }:
        kwargs["ddof"] = ddof
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(values, **kwargs)


def _assert_reference(actual: Any, expected: Any) -> None:
    if np.asarray(expected).dtype == np.dtype(bool):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=3.0e-6,
            atol=3.0e-6,
            equal_nan=True,
        )


def run_backend(
    backend_type: type[Backend],
    *,
    shape: tuple[int, int],
    repeats: int,
    workload: str,
) -> list[dict[str, Any]]:
    backend = backend_type.connect(max_densify_cells=1_000_000)
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260711)
    try:
        for sparse_input in (False, True):
            values = _values(rng, shape, sparse_input=sparse_input)
            matrix = (
                backend.from_scipy(sparse.csr_array(values))
                if sparse_input
                else backend.from_numpy(values)
            )
            for label, operation, axis, ddof, reference in _CASES:
                compile_ms, compiled = timed(
                    partial(
                        backend.lowerer.compile_reduction,
                        matrix._expr,  # noqa: SLF001 - architecture diagnostic
                        operation,
                        axis,
                        dialect=backend.dialect,
                        ddof=ddof,
                    ),
                    repeats=repeats,
                )
                sql, _ = compiled
                execute_ms, _ = timed(
                    partial(backend._execute_sql, sql),  # noqa: SLF001
                    repeats=repeats,
                )
                actual = _method_result(matrix, operation, axis, ddof)
                expected = _reference_result(reference, values, operation, axis, ddof)
                _assert_reference(actual, expected)

                upper_sql = sql.upper()
                dimension_relations = (
                    matrix._expr.rows_relation.upper(),  # noqa: SLF001
                    matrix._expr.cols_relation.upper(),  # noqa: SLF001
                )
                dimension_domain_refs = sum(
                    upper_sql.count(marker) for marker in ("GENERATE_SERIES", "RANGE(")
                ) + sum(upper_sql.count(relation) for relation in dimension_relations)
                if dimension_domain_refs:
                    raise AssertionError(
                        f"raw {operation.value} unexpectedly referenced "
                        "dimension domains"
                    )
                results.append(
                    {
                        "axis": axis,
                        "backend": backend.name,
                        "compile_ms": compile_ms,
                        "ddof": ddof,
                        "execute_ms": execute_ms,
                        "dimension_domain_refs": dimension_domain_refs,
                        "plan": label,
                        "cross_join_count": upper_sql.count("CROSS JOIN"),
                        "sql_bytes": len(sql.encode()),
                        "storage": "sparse" if sparse_input else "dense",
                        "workload": workload,
                    }
                )
    finally:
        backend.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workload", choices=("micro", "moderate"), default="micro")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    shape = (64, 48) if args.workload == "micro" else (512, 384)
    result = {
        "repeats": args.repeats,
        "shape": shape,
        "versions": {
            package: version(package)
            for package in ("dbnumpy", "ibis-framework", "duckdb", "datafusion")
        },
        "results": [
            *run_backend(
                DuckDBBackend,
                shape=shape,
                repeats=args.repeats,
                workload=args.workload,
            ),
            *run_backend(
                DataFusionBackend,
                shape=shape,
                repeats=args.repeats,
                workload=args.workload,
            ),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
