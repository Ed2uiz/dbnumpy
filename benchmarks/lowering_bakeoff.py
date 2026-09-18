"""Compare Ibis/SQL relational plans with target-native construction.

The benchmark covers sparse matrix multiplication and a densifying reduction.
It is deliberately serial and small enough for a 16 GB development host.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from statistics import median
from time import perf_counter
from typing import Any

import datafusion.functions as df_functions
import numpy as np
from datafusion import col
from scipy import sparse

from dbnumpy.backends import DataFusionBackend, DuckDBBackend
from dbnumpy.ir import ReductionOp


def timed(callable_: Any, *, repeats: int) -> tuple[float, Any]:
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = callable_()
        durations.append(perf_counter() - start)
    return median(durations) * 1_000, result


def random_sparse(
    rng: np.random.Generator, shape: tuple[int, int], density: float
) -> np.ndarray:
    values = rng.normal(size=shape)
    values[rng.random(shape) >= density] = 0.0
    return values


def coordinates_to_dense(table: Any, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape)
    if table.num_rows:
        rows = table.column("i").to_numpy().astype(np.intp)
        cols = table.column("j").to_numpy().astype(np.intp)
        values = table.column("x").to_numpy()
        result[rows, cols] = values
    return result


def reduction_to_dense(table: Any, length: int) -> np.ndarray:
    result = np.zeros(length)
    if table.num_rows:
        index = table.column("index").to_numpy().astype(np.intp)
        result[index] = table.column("value").to_numpy()
    return result


def duckdb_native_matmul(backend: DuckDBBackend, left: Any, right: Any) -> Any:
    lhs = backend.connection.table(left._expr.relation).project(  # noqa: SLF001
        "i AS li, j AS k, x AS lx"
    )
    rhs = backend.connection.table(right._expr.relation).project(  # noqa: SLF001
        "i AS rk, j AS rj, x AS rx"
    )
    return (
        lhs.join(rhs, "k = rk")
        .aggregate("li, rj, sum(lx * rx) AS x")
        .project("li AS i, rj AS j, x")
    )


def datafusion_native_matmul(backend: DataFusionBackend, left: Any, right: Any) -> Any:
    lhs = backend.context.table(left._expr.relation).select(  # noqa: SLF001
        col("i").alias("li"), col("j").alias("k"), col("x").alias("lx")
    )
    rhs = backend.context.table(right._expr.relation).select(  # noqa: SLF001
        col("i").alias("rk"), col("j").alias("rj"), col("x").alias("rx")
    )
    return (
        lhs.join_on(rhs, col("k") == col("rk"))
        .aggregate(
            [col("li"), col("rj")],
            [df_functions.sum(col("lx") * col("rx")).alias("x")],
        )
        .select(col("li").alias("i"), col("rj").alias("j"), col("x"))
    )


def duckdb_native_densifying_reduction(backend: DuckDBBackend, matrix: Any) -> Any:
    source = matrix._expr  # noqa: SLF001
    rows = backend.connection.table(source.rows_relation).set_alias("rr")
    cols = backend.connection.table(source.cols_relation).set_alias("cc")
    values = backend.connection.table(source.relation).set_alias("vv")
    dense = (
        rows.cross(cols)
        .join(values, "rr.i = vv.i AND cc.j = vv.j", "left")
        .project("rr.i AS i, cc.j AS j, exp(coalesce(vv.x, 0.0)) AS x")
    )
    return dense.aggregate("j AS index, sum(x) AS value")


def datafusion_native_densifying_reduction(
    backend: DataFusionBackend, matrix: Any
) -> Any:
    pointwise = backend.native_pointwise_lowerer.lower(
        np.exp(matrix)._expr,
        context=backend.context,  # noqa: SLF001
    )
    return pointwise.aggregate(
        [col("j")], [df_functions.sum(col("x")).alias("value")]
    ).select(col("j").alias("index"), col("value"))


def run_backend(
    backend_type: Any, *, repeats: int, workload: str
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260710)
    if workload == "micro":
        left_shape, right_shape, density = (64, 80), (80, 48), 0.05
    else:
        left_shape, right_shape, density = (1_000, 800), (800, 600), 0.01
    left_values = random_sparse(rng, left_shape, density)
    right_values = random_sparse(rng, right_shape, density)
    left_reference = sparse.csr_array(left_values)
    right_reference = sparse.csr_array(right_values)
    backend = backend_type.connect(max_densify_cells=1_000_000)
    try:
        left = backend.from_scipy(left_reference, name="left")
        right = backend.from_scipy(right_reference, name="right")

        product = left @ right
        compile_ms, sql = timed(
            lambda: backend.lowerer.compile_matrix(
                product._expr,
                dialect=backend.dialect,  # noqa: SLF001
            ),
            repeats=repeats,
        )
        sql_execute_ms, product_table = timed(
            lambda: backend._execute_sql(sql),  # noqa: SLF001
            repeats=repeats,
        )
        product_result = coordinates_to_dense(product_table, product.shape)

        native_matmul_builder = (
            duckdb_native_matmul
            if isinstance(backend, DuckDBBackend)
            else datafusion_native_matmul
        )
        native_build_ms, native_product = timed(
            lambda: native_matmul_builder(backend, left, right), repeats=repeats
        )
        native_execute = native_product.to_arrow_table
        native_execute_ms, native_product_table = timed(native_execute, repeats=repeats)
        np.testing.assert_allclose(
            product_result, (left_reference @ right_reference).toarray()
        )
        np.testing.assert_allclose(
            coordinates_to_dense(native_product_table, product.shape), product_result
        )

        exp_expr = np.exp(left)._expr  # noqa: SLF001
        reduction_compile_ms, compiled = timed(
            lambda: backend.lowerer.compile_reduction(
                exp_expr,
                ReductionOp.SUM,
                0,
                dialect=backend.dialect,
            ),
            repeats=repeats,
        )
        reduction_sql, _ = compiled
        reduction_execute_ms, reduction_table = timed(
            lambda: backend._execute_sql(reduction_sql),  # noqa: SLF001
            repeats=repeats,
        )
        native_reduction_builder = (
            duckdb_native_densifying_reduction
            if isinstance(backend, DuckDBBackend)
            else datafusion_native_densifying_reduction
        )
        native_reduction_build_ms, native_reduction = timed(
            lambda: native_reduction_builder(backend, left), repeats=repeats
        )
        native_reduction_execute_ms, native_reduction_table = timed(
            native_reduction.to_arrow_table,
            repeats=repeats,
        )
        expected_reduction = np.exp(left_values).sum(axis=0)
        np.testing.assert_allclose(
            reduction_to_dense(reduction_table, left.shape[1]), expected_reduction
        )
        np.testing.assert_allclose(
            reduction_to_dense(native_reduction_table, left.shape[1]),
            expected_reduction,
        )

        return [
            {
                "backend": backend.name,
                "workload": workload,
                "plan": "sparse_matmul",
                "ibis_compile_ms": compile_ms,
                "ibis_sql_execute_ms": sql_execute_ms,
                "native_build_ms": native_build_ms,
                "native_execute_ms": native_execute_ms,
                "sql_bytes": len(sql.encode()),
            },
            {
                "backend": backend.name,
                "workload": workload,
                "plan": "densifying_colsum_exp",
                "ibis_compile_ms": reduction_compile_ms,
                "ibis_sql_execute_ms": reduction_execute_ms,
                "native_build_ms": native_reduction_build_ms,
                "native_execute_ms": native_reduction_execute_ms,
                "sql_bytes": len(reduction_sql.encode()),
            },
        ]
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workload", choices=("micro", "moderate"), default="micro")
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
            *run_backend(DuckDBBackend, repeats=args.repeats, workload=args.workload),
            *run_backend(
                DataFusionBackend, repeats=args.repeats, workload=args.workload
            ),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
