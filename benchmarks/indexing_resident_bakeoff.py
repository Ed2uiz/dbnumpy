"""Persistent-Parquet indexing and composed-reduction lowering bake-off.

The fixed workload is ``(abs(source[rows, :]) * 1.5).sum()`` over a sparse
canonical ``(i, j, x)`` Parquet relation. SQLAlchemy, direct SQL, and native
plans are deliberately narrow controls tied to that exact semantic tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from math import fsum
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import ibis
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sa
from sqlalchemy.engine.default import DefaultDialect

from dbnumpy.backends import DataFusionBackend, DuckDBBackend
from dbnumpy.indexing import _GatherIndices, _normalize_selector
from dbnumpy.ir import (
    BinaryOp,
    Gather,
    MatrixExpr,
    ReductionOp,
    ScalarBinary,
    Source,
    Unary,
    UnaryOp,
)

DEFAULT_PARQUET = Path(
    "/Volumes/WD/dbnumpy-benchmarks/0.2.1a0/20260711-figure1g-repro/"
    "canonical-all-v1/artifacts/30000/shared/canonical.parquet"
)
PATHS = ("dbverse_ibis", "sqlalchemy_core", "direct_sql", "backend_native")
BACKENDS = ("duckdb", "datafusion")
_BALANCED_PATH_ORDERS = (
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
    (3, 1, 2, 0),
    (3, 2, 0, 1),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
)
_BALANCED_WARMUP_ORDER = (1, 2, 3, 0)
MAX_SELECTOR_SIZE = 100_000
DEFAULT_MAX_STORED_GATHER_ROWS = 50_000_000
DEFAULT_MAX_INPUT_ROWS = 250_000_000
DEFAULT_MAX_ESTIMATED_SCAN_BYTES = 16 * 1024**3

FAIRNESS = {
    "fixed_workload": (
        "All controls are tied to (abs(Gather(Source)) * 1.5).sum(); they are not "
        "general replacement lowerers."
    ),
    "shared_relations": (
        "Within each backend case every path scans the same persistent Parquet "
        "registration and the same Arrow selector mapping relation."
    ),
    "warm_execution": (
        "Each path receives one unrecorded warm-up. Warm-up and measured order are "
        "chosen together so ordinal position and the complete chronological sequence "
        "of immediate cross-path predecessors remain balanced; cold storage-cache "
        "performance is not claimed."
    ),
    "native_asymmetry": (
        "Native plans are built once and skip SQL parsing during execution. SQL paths "
        "parse and plan their text on every measured call."
    ),
    "ibis_phase_overlap": (
        "The Ibis build diagnostic constructs its public relational expression. The "
        "reported actual DBVerse compile independently repeats lowering and serializes "
        "it, so build and compile must not be summed as disjoint phases."
    ),
    "sqlalchemy_scope": (
        "SQLAlchemy uses generic DefaultDialect SQL for this common subset; this is "
        "not evidence of a supported DuckDB or DataFusion dialect."
    ),
    "result": (
        "Only a scalar is collected in timed execution. Stored gather row counts are "
        "validated separately and are not part of execution timing."
    ),
    "fixture_scope": (
        "Controls omit broader DBVerse NULL/special-value semantics. The selected "
        "canonical fixture is independently streamed with PyArrow to require finite, "
        "nonnull, in-bounds values and to calculate the expected scalar and row count."
    ),
    "resources": (
        "The 1 GiB setting is an engine configuration, not a measured process-RSS or "
        "spill claim. Input rows, estimated repeated scan bytes, and actual selected "
        "stored rows are guarded before timed execution."
    ),
}


def samples(
    factory: Callable[[], Any], *, repeats: int, warmup: bool = True
) -> tuple[list[float], Any]:
    if warmup:
        factory()
    durations: list[float] = []
    value: Any = None
    for _ in range(repeats):
        start = perf_counter()
        value = factory()
        durations.append((perf_counter() - start) * 1_000)
    return durations, value


def summary(values: list[float]) -> dict[str, Any]:
    return {"median_ms": median(values), "samples_ms": values}


def balanced_execution_order(repeat: int) -> list[str]:
    """Balance position and chronological carryover in eight-repeat blocks."""

    cycle, row = divmod(repeat, len(_BALANCED_PATH_ORDERS))
    shift = cycle % len(PATHS)
    return [
        PATHS[(position + shift) % len(PATHS)]
        for position in _BALANCED_PATH_ORDERS[row]
    ]


def make_selector(size: int, axis_size: int, *, seed: int) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, axis_size, size=size, dtype=np.int64)


def normalize_selector(
    selector: np.ndarray[Any, Any], axis_size: int, *, axis: int = 0
) -> np.ndarray[Any, np.dtype[np.int64]]:
    normalized = _normalize_selector(selector, axis_size, axis=axis)
    if not isinstance(normalized, _GatherIndices):
        raise AssertionError("benchmark selector did not normalize to a gather")
    return normalized.values


def build_public_workload(
    matrix: Any, selector: np.ndarray[Any, Any], *, axis: int
) -> MatrixExpr:
    """Construct the semantic tree through the real public Python API."""

    gathered = matrix[selector, :] if axis == 0 else matrix[:, selector]
    result = abs(gathered) * 1.5
    return result._expr  # noqa: SLF001 - benchmark intentionally inspects the IR


def _fixed_gather(expr: MatrixExpr) -> Gather:
    if (
        not isinstance(expr, ScalarBinary)
        or expr.op is not BinaryOp.MULTIPLY
        or expr.reverse
        or expr.scalar != 1.5
        or not isinstance(expr.arg, Unary)
        or expr.arg.op is not UnaryOp.ABSOLUTE
        or not isinstance(expr.arg.arg, Gather)
    ):
        raise ValueError("control requires abs(gather) * 1.5")
    return expr.arg.arg


def _fixed_relations(expr: MatrixExpr) -> tuple[str, str, str]:
    gathered = _fixed_gather(expr)
    if (
        not isinstance(gathered.arg, Source)
        or not gathered.map_relation
    ):
        raise ValueError("control requires one axis mapping over a Source")
    coordinate = "i" if gathered.axis == 0 else "j"
    return gathered.arg.relation, gathered.map_relation, coordinate


def sqlalchemy_scalar(expr: MatrixExpr) -> Any:
    source_name, selector_name, coordinate = _fixed_relations(expr)
    source = sa.table(
        source_name,
        sa.column("i", sa.BigInteger()),
        sa.column("j", sa.BigInteger()),
        sa.column("x", sa.Float()),
    )
    selector = sa.table(
        selector_name,
        sa.column("source_index", sa.BigInteger()),
        sa.column("output_index", sa.BigInteger()),
    )
    joined = selector.join(
        source, selector.c.source_index == getattr(source.c, coordinate)
    )
    return sa.select(
        sa.func.sum(sa.func.abs(source.c.x) * 1.5).label("value")
    ).select_from(joined)


def sqlalchemy_count(expr: MatrixExpr) -> Any:
    source_name, selector_name, coordinate = _fixed_relations(expr)
    source = sa.table(source_name, sa.column(coordinate, sa.BigInteger()))
    selector = sa.table(selector_name, sa.column("source_index", sa.BigInteger()))
    return sa.select(sa.func.count().label("stored_rows")).select_from(
        selector.join(
            source, selector.c.source_index == getattr(source.c, coordinate)
        )
    )


def compile_sqlalchemy(statement: Any) -> str:
    return str(
        statement.compile(
            dialect=DefaultDialect(), compile_kwargs={"literal_binds": True}
        )
    )


def direct_scalar_sql(expr: MatrixExpr) -> str:
    source, selector, coordinate = _fixed_relations(expr)
    return (
        'SELECT sum(abs(s."x") * 1.5) AS "value" '
        f'FROM "{selector}" AS m INNER JOIN "{source}" AS s '
        f'ON m."source_index" = s."{coordinate}"'
    )


def direct_count_sql(expr: MatrixExpr) -> str:
    source, selector, coordinate = _fixed_relations(expr)
    return (
        'SELECT count(*) AS "stored_rows" '
        f'FROM "{selector}" AS m INNER JOIN "{source}" AS s '
        f'ON m."source_index" = s."{coordinate}"'
    )


def native_scalar_plan(backend: Any, expr: MatrixExpr) -> Any:
    source_name, selector_name, coordinate = _fixed_relations(expr)
    if isinstance(backend, DuckDBBackend):
        source = backend.connection.table(source_name).set_alias("s")
        selector = backend.connection.table(selector_name).set_alias("m")
        return selector.join(
            source, f"m.source_index = s.{coordinate}"
        ).aggregate(
            "sum(abs(s.x) * 1.5) AS value"
        )
    from datafusion import col
    from datafusion import functions as f

    source = backend.context.table(source_name)
    selector = backend.context.table(selector_name)
    joined = selector.join(
        source, left_on="source_index", right_on=coordinate, how="inner"
    )
    return joined.aggregate([], [f.sum(f.abs(col("x")) * 1.5).alias("value")])


def native_count_plan(backend: Any, expr: MatrixExpr) -> Any:
    source_name, selector_name, coordinate = _fixed_relations(expr)
    if isinstance(backend, DuckDBBackend):
        source = backend.connection.table(source_name).set_alias("s")
        selector = backend.connection.table(selector_name).set_alias("m")
        return selector.join(
            source, f"m.source_index = s.{coordinate}"
        ).aggregate(
            "count(*) AS stored_rows"
        )
    from datafusion import functions as f

    source = backend.context.table(source_name)
    selector = backend.context.table(selector_name)
    joined = selector.join(
        source, left_on="source_index", right_on=coordinate, how="inner"
    )
    return joined.aggregate([], [f.count().alias("stored_rows")])


def _scalar(table: pa.Table) -> float:
    if table.num_rows != 1 or table.num_columns != 1:
        raise AssertionError("scalar path must return exactly one value")
    value = table.column(0)[0].as_py()
    if value is None or not np.isfinite(value):
        raise AssertionError(f"expected a finite scalar, got {value!r}")
    return float(value)


def _count(table: pa.Table) -> int:
    if table.num_rows != 1 or table.num_columns != 1:
        raise AssertionError("count path must return exactly one value")
    return int(table.column(0)[0].as_py())


def _plan_record(kind: str, text: str) -> dict[str, str]:
    return {
        "kind": kind,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def sql_explain(backend: Any, sql: str) -> dict[str, str]:
    """Retain a backend EXPLAIN result without executing the workload."""

    table = backend._execute_sql(f"EXPLAIN {sql}")  # noqa: SLF001
    text = json.dumps(table.to_pylist(), sort_keys=True, separators=(",", ":"))
    return _plan_record("backend_sql_explain", text)


def native_explain(backend: Any, plan: Any) -> dict[str, str]:
    """Retain the native control's optimized logical explanation."""

    if isinstance(backend, DuckDBBackend):
        text = str(plan.explain())
    else:
        text = str(plan.optimized_logical_plan())
    return _plan_record("backend_native_optimized_logical_plan", text)


def _duck_source(backend: DuckDBBackend, path: Path, shape: tuple[int, int]) -> Any:
    path_literal = str(path).replace("'", "''")
    backend.connection.execute(
        'CREATE TEMP VIEW "resident_source" AS '
        "SELECT cast(i AS BIGINT) AS i, cast(j AS BIGINT) AS j, "
        f"cast(x AS DOUBLE) AS x FROM read_parquet('{path_literal}')",
    )
    return backend.from_relation("resident_source", shape=shape, storage="sparse")


def _open_backend(
    backend_name: str, path: Path, shape: tuple[int, int], selector_size: int
) -> tuple[Any, Any]:
    common = {
        "max_densify_cells": 1_000_000,
        "max_host_values": max(selector_size, 1),
    }
    if backend_name == "duckdb":
        backend = DuckDBBackend.connect(memory_limit="1GB", threads=2, **common)
        return backend, _duck_source(backend, path, shape)
    backend = DataFusionBackend.connect(
        memory_limit_bytes=1024**3, target_partitions=2, **common
    )
    return backend, backend.from_parquet(
        path, shape=shape, storage="sparse", name="resident_source"
    )


def _git_provenance(repo: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = command("git", "status", "--porcelain").splitlines()
        return {
            "commit": command("git", "rev-parse", "HEAD"),
            "dirty_entries": len(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_entries": None}


def _parquet_provenance(path: Path) -> dict[str, Any]:
    metadata = pq.ParquetFile(path).metadata
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "schema": str(pq.read_schema(path)),
    }


def inspect_fixture(
    path: Path,
    *,
    shape: tuple[int, int],
    selector: np.ndarray[Any, np.dtype[np.int64]],
    axis: int = 0,
) -> dict[str, Any]:
    """Stream fixture invariants and an engine-independent workload reference."""

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    try:
        i_type = schema.field("i").type
        j_type = schema.field("j").type
        x_type = schema.field("x").type
    except KeyError as exc:
        raise ValueError("fixture must contain i, j, x columns") from exc
    if not pa.types.is_integer(i_type) or not pa.types.is_integer(j_type):
        raise ValueError("fixture i and j columns must be integers")
    if not (pa.types.is_integer(x_type) or pa.types.is_floating(x_type)):
        raise ValueError("fixture x column must be numeric")
    if parquet.metadata.num_rows < 1:
        raise ValueError("fixture must contain at least one stored coordinate")

    weights = np.bincount(selector, minlength=shape[axis]).astype(
        np.int64, copy=False
    )
    scalar_parts: list[float] = []
    selected_stored_rows = 0
    observed_rows = 0
    min_i = shape[0]
    max_i = -1
    min_j = shape[1]
    max_j = -1
    min_x = float("inf")
    max_x = float("-inf")
    for batch in parquet.iter_batches(columns=["i", "j", "x"], batch_size=65_536):
        if any(batch.column(position).null_count for position in range(3)):
            raise ValueError("fixture coordinates and values must not contain NULL")
        i = batch.column(0).to_numpy().astype(np.int64, copy=False)
        j = batch.column(1).to_numpy().astype(np.int64, copy=False)
        x = batch.column(2).to_numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(x)):
            raise ValueError("fixture values must be finite for this fixed workload")
        if len(i) and (
            np.any(i < 0)
            or np.any(i >= shape[0])
            or np.any(j < 0)
            or np.any(j >= shape[1])
        ):
            raise ValueError(f"fixture coordinates exceed declared shape {shape}")
        observed_rows += len(i)
        if len(i):
            min_i = min(min_i, int(i.min()))
            max_i = max(max_i, int(i.max()))
            min_j = min(min_j, int(j.min()))
            max_j = max(max_j, int(j.max()))
            min_x = min(min_x, float(x.min()))
            max_x = max(max_x, float(x.max()))
            coordinate = i if axis == 0 else j
            multiplicity = weights[coordinate]
            selected_stored_rows += int(multiplicity.sum(dtype=np.int64))
            scalar_parts.append(float(np.sum(np.abs(x) * 1.5 * multiplicity)))
    if observed_rows != parquet.metadata.num_rows:
        raise AssertionError("streamed Parquet row count differs from metadata")
    if max_i != shape[0] - 1 or max_j != shape[1] - 1:
        raise ValueError(
            "declared shape must exactly match this canonical fixture's observed "
            f"coordinate domain; maxima are {(max_i, max_j)} for shape {shape}"
        )
    return {
        "method": "PyArrow streaming batches independent of query engines",
        "formula": "sum(abs(x) * 1.5 * selected_axis_multiplicity)",
        "axis": axis,
        "expected_scalar": fsum(scalar_parts),
        "expected_stored_gather_rows": selected_stored_rows,
        "observed_rows": observed_rows,
        "coordinate_min": [min_i, min_j],
        "coordinate_max": [max_i, max_j],
        "value_min": min_x,
        "value_max": max_x,
        "null_values": 0,
        "nonfinite_values": 0,
        "coordinate_uniqueness": "trusted canonical fixture; not rechecked here",
        "selector_sha256": hashlib.sha256(selector.tobytes()).hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(repo: Path) -> str:
    candidates = [repo / "pyproject.toml", Path(__file__).resolve()]
    candidates.extend(sorted((repo / "src" / "dbnumpy").rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(candidates)):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ibis_count_sql(backend: Any, gathered: Gather) -> str:
    table = backend.lowerer.lower_matrix(gathered).table
    return str(
        ibis.to_sql(table.aggregate(stored_rows=table.count()), dialect=backend.dialect)
    )


def run_case(
    backend_name: str,
    *,
    parquet: Path,
    shape: tuple[int, int],
    selector_size: int,
    repeats: int,
    seed: int,
    axis: int = 0,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    setup_start = perf_counter()
    backend, matrix = _open_backend(backend_name, parquet, shape, selector_size)
    source_setup_ms = (perf_counter() - setup_start) * 1_000
    try:
        selector_fixture = make_selector(selector_size, shape[axis], seed=seed)
        normalize_samples, selector = samples(
            lambda: normalize_selector(
                selector_fixture, shape[axis], axis=axis
            ),
            repeats=repeats,
        )
        selector_hash = hashlib.sha256(selector.tobytes()).hexdigest()
        if reference is None:
            reference = inspect_fixture(
                parquet, shape=shape, selector=selector, axis=axis
            )
        if reference["axis"] != axis:
            raise AssertionError("fixture reference used a different gather axis")
        if reference["selector_sha256"] != selector_hash:
            raise AssertionError("fixture reference used a different selector")
        start = perf_counter()
        selector_relation = backend._selector_relation(selector)  # noqa: SLF001
        selector_registration_ms = (perf_counter() - start) * 1_000
        start = perf_counter()
        output_shape = (
            (selector_size, shape[1])
            if axis == 0
            else (shape[0], selector_size)
        )
        rows_relation, cols_relation = backend.dimension_relations(output_shape)
        dimension_setup_ms = (perf_counter() - start) * 1_000

        public_api_samples, expr = samples(
            lambda: build_public_workload(matrix, selector, axis=axis),
            repeats=repeats,
        )
        gathered = _fixed_gather(expr)
        if gathered.axis != axis:
            raise AssertionError("public indexing gathered the wrong axis")
        if gathered.map_relation != selector_relation:
            raise AssertionError(
                "public indexing did not reuse the registered selector"
            )
        if (gathered.rows_relation, gathered.cols_relation) != (
            rows_relation,
            cols_relation,
        ):
            raise AssertionError("public indexing did not reuse output dimensions")

        ibis_build_samples, _ = samples(
            lambda: backend.lowerer.lower_reduction(expr, ReductionOp.SUM, None),
            repeats=repeats,
        )
        ibis_compile_samples, compiled = samples(
            lambda: backend.lowerer.compile_reduction(
                expr, ReductionOp.SUM, None, dialect=backend.dialect
            ),
            repeats=repeats,
        )
        ibis_sql = compiled[0]

        sa_build_samples, sa_statement = samples(
            lambda: sqlalchemy_scalar(expr), repeats=repeats
        )
        sa_compile_samples, sa_sql = samples(
            lambda: compile_sqlalchemy(sa_statement), repeats=repeats
        )
        direct_build_samples, direct_sql = samples(
            lambda: direct_scalar_sql(expr), repeats=repeats
        )
        native_build_samples, native_plan = samples(
            lambda: native_scalar_plan(backend, expr), repeats=repeats
        )

        explain_plans = {
            "dbverse_ibis": sql_explain(backend, ibis_sql),
            "sqlalchemy_core": sql_explain(backend, sa_sql),
            "direct_sql": sql_explain(backend, direct_sql),
            "backend_native": native_explain(backend, native_plan),
        }

        executors: dict[str, Callable[[], pa.Table]] = {
            "dbverse_ibis": lambda: backend._execute_sql(ibis_sql),  # noqa: SLF001
            "sqlalchemy_core": lambda: backend._execute_sql(sa_sql),  # noqa: SLF001
            "direct_sql": lambda: backend._execute_sql(direct_sql),  # noqa: SLF001
            "backend_native": native_plan.to_arrow_table,
        }
        for position in _BALANCED_WARMUP_ORDER:
            name = PATHS[position]
            executors[name]()
        execution_samples: dict[str, list[float]] = defaultdict(list)
        scalar_values: dict[str, list[float]] = defaultdict(list)
        execution_orders: list[list[str]] = []
        for repeat in range(repeats):
            order = balanced_execution_order(repeat)
            execution_orders.append(order)
            for name in order:
                start = perf_counter()
                table = executors[name]()
                execution_samples[name].append((perf_counter() - start) * 1_000)
                scalar_values[name].append(_scalar(table))

        count_tables = {
            "dbverse_ibis": backend._execute_sql(  # noqa: SLF001
                _ibis_count_sql(backend, gathered)
            ),
            "sqlalchemy_core": backend._execute_sql(  # noqa: SLF001
                compile_sqlalchemy(sqlalchemy_count(expr))
            ),
            "direct_sql": backend._execute_sql(  # noqa: SLF001
                direct_count_sql(expr)
            ),
            "backend_native": native_count_plan(backend, expr).to_arrow_table(),
        }
        counts = {name: _count(table) for name, table in count_tables.items()}
        if len(set(counts.values())) != 1:
            raise AssertionError(f"stored gather row counts differ: {counts}")
        expected_count = int(reference["expected_stored_gather_rows"])
        if set(counts.values()) != {expected_count}:
            raise AssertionError(
                f"stored gather row count differs from independent reference: "
                f"expected {expected_count}, got {counts}"
            )
        expected_scalar = float(reference["expected_scalar"])
        for values in scalar_values.values():
            np.testing.assert_allclose(
                values, expected_scalar, rtol=1e-11, atol=1e-8
            )

        build_data = {
            "dbverse_ibis": (
                ibis_build_samples,
                ibis_compile_samples,
                len(ibis_sql.encode()),
                ibis_sql,
            ),
            "sqlalchemy_core": (
                sa_build_samples,
                sa_compile_samples,
                len(sa_sql.encode()),
                sa_sql,
            ),
            "direct_sql": (
                direct_build_samples,
                None,
                len(direct_sql.encode()),
                direct_sql,
            ),
            "backend_native": (native_build_samples, None, None, None),
        }
        paths = []
        for name in PATHS:
            build_values, compile_values, sql_bytes, sql = build_data[name]
            paths.append(
                {
                    "path": name,
                    "support_status": {
                        "dbverse_ibis": "project_default_actual_reduction_sql",
                        "sqlalchemy_core": "fixed_workload_generic_sql_probe",
                        "direct_sql": "fixed_workload_handwritten_control",
                        "backend_native": "fixed_workload_backend_lower_bound",
                    }[name],
                    "build": summary(build_values),
                    "compile": None
                    if compile_values is None
                    else summary(compile_values),
                    "execute_collect": summary(execution_samples[name]),
                    "sql_bytes": sql_bytes,
                    "sql_sha256": None
                    if sql is None
                    else hashlib.sha256(sql.encode()).hexdigest(),
                    "sql": sql,
                    "explain": explain_plans[name],
                    "scalar_samples": scalar_values[name],
                    "stored_gather_rows": counts[name],
                }
            )
        return {
            "backend": backend_name,
            "axis": axis,
            "source_setup_ms": source_setup_ms,
            "selector_normalization": summary(normalize_samples),
            "selector_registration_ms": selector_registration_ms,
            "dimension_setup_ms": dimension_setup_ms,
            "public_api_semantic_build": summary(public_api_samples),
            "selector_sha256": selector_hash,
            "independent_expected_scalar": expected_scalar,
            "independent_expected_stored_rows": expected_count,
            "execution_orders": execution_orders,
            "paths": paths,
        }
    finally:
        backend.close()


def parse_backends(value: str) -> tuple[str, ...]:
    values = tuple(value.split(","))
    if (
        not values
        or len(set(values)) != len(values)
        or any(x not in BACKENDS for x in values)
    ):
        raise argparse.ArgumentTypeError(f"expected unique choices from {BACKENDS}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--cols", type=int, default=30_000)
    parser.add_argument("--axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--selector-size", type=int, default=1_024)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--backends", type=parse_backends, default=BACKENDS)
    parser.add_argument(
        "--max-stored-gather-rows",
        type=int,
        default=DEFAULT_MAX_STORED_GATHER_ROWS,
    )
    parser.add_argument("--max-input-rows", type=int, default=DEFAULT_MAX_INPUT_ROWS)
    parser.add_argument(
        "--max-estimated-scan-bytes",
        type=int,
        default=DEFAULT_MAX_ESTIMATED_SCAN_BYTES,
    )
    parser.add_argument("--output", type=Path)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    parquet = args.parquet.expanduser().resolve()
    if not parquet.is_file():
        parser.error(f"Parquet input does not exist: {parquet}")
    if args.rows < 1 or args.cols < 1:
        parser.error("shape dimensions must be positive")
    if not 1 <= args.selector_size <= MAX_SELECTOR_SIZE:
        parser.error(f"selector-size must be between 1 and {MAX_SELECTOR_SIZE:,}")
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be between 1 and 20")
    if args.max_stored_gather_rows < 1:
        parser.error("max-stored-gather-rows must be positive")
    if args.max_input_rows < 1:
        parser.error("max-input-rows must be positive")
    if args.max_estimated_scan_bytes < 1:
        parser.error("max-estimated-scan-bytes must be positive")
    physical_rows = pq.ParquetFile(parquet).metadata.num_rows
    if physical_rows > args.max_input_rows:
        parser.error(
            f"input has {physical_rows:,} rows, above "
            f"--max-input-rows={args.max_input_rows:,}"
        )
    axis_size = (args.rows, args.cols)[args.axis]
    estimated_rows = args.selector_size * physical_rows / axis_size
    if estimated_rows > args.max_stored_gather_rows:
        parser.error(
            f"uniform-density estimate of gathered stored rows "
            f"{estimated_rows:,.0f} exceeds "
            f"--max-stored-gather-rows={args.max_stored_gather_rows:,}"
        )
    estimated_file_passes = 2 + len(args.backends) * len(PATHS) * (args.repeats + 2)
    estimated_scan_bytes = parquet.stat().st_size * estimated_file_passes
    if estimated_scan_bytes > args.max_estimated_scan_bytes:
        parser.error(
            f"estimated repeated Parquet scan volume {estimated_scan_bytes:,} bytes "
            f"exceeds --max-estimated-scan-bytes="
            f"{args.max_estimated_scan_bytes:,}"
        )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            parser.error(f"refusing to overwrite existing output: {output}")
        if not output.parent.is_dir():
            parser.error(f"output parent does not exist: {output.parent}")
    return parquet


def run_benchmark(args: argparse.Namespace, parquet: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    physical_rows = pq.ParquetFile(parquet).metadata.num_rows
    shape = (args.rows, args.cols)
    selector = normalize_selector(
        make_selector(args.selector_size, shape[args.axis], seed=args.seed),
        shape[args.axis],
        axis=args.axis,
    )
    fixture_reference = inspect_fixture(
        parquet, shape=shape, selector=selector, axis=args.axis
    )
    actual_stored_rows = int(
        fixture_reference["expected_stored_gather_rows"]
    )
    if actual_stored_rows > args.max_stored_gather_rows:
        raise ValueError(
            f"actual selected stored rows {actual_stored_rows:,} exceed "
            f"--max-stored-gather-rows={args.max_stored_gather_rows:,}"
        )
    cases = [
        run_case(
            backend,
            parquet=parquet,
            shape=shape,
            selector_size=args.selector_size,
            repeats=args.repeats,
            seed=args.seed,
            axis=args.axis,
            reference=fixture_reference,
        )
        for backend in args.backends
    ]
    direct_scalars = {
        case["backend"]: median(
            next(p for p in case["paths"] if p["path"] == "direct_sql")[
                "scalar_samples"
            ]
        )
        for case in cases
    }
    if len(direct_scalars) == 2:
        np.testing.assert_allclose(
            direct_scalars["duckdb"],
            direct_scalars["datafusion"],
            rtol=1e-11,
            atol=1e-7,
        )
    return {
        "benchmark": "indexing_resident_bakeoff",
        "created_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "versions": {
            name: version(name)
            for name in (
                "dbnumpy",
                "ibis-framework",
                "sqlalchemy",
                "duckdb",
                "datafusion",
            )
        },
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "benchmark_script_sha256": _file_sha256(Path(__file__)),
            "source_tree_sha256": _source_tree_sha256(repo),
        },
        "git": _git_provenance(repo),
        "input": _parquet_provenance(parquet),
        "fixture_reference": fixture_reference,
        "config": {
            "shape": [args.rows, args.cols],
            "axis": args.axis,
            "selector_size": args.selector_size,
            "repeats": args.repeats,
            "seed": args.seed,
            "backends": list(args.backends),
            "serial": True,
            "configured_engine_memory_limit_bytes": 1024**3,
            "engine_threads_or_partitions": 2,
            "execution_cache_state": "warm_after_one_warmup_per_path",
            "path_execution_order": (
                "eight_run_position_and_chronological_predecessor_balance"
            ),
            "execution_order_balanced": args.repeats % 8 == 0,
            "execution_warmup_order": [
                PATHS[position] for position in _BALANCED_WARMUP_ORDER
            ],
            "max_stored_gather_rows": args.max_stored_gather_rows,
            "uniform_estimated_gathered_stored_rows": (
                args.selector_size * physical_rows / shape[args.axis]
            ),
            "actual_gathered_stored_rows": actual_stored_rows,
            "max_input_rows": args.max_input_rows,
            "max_estimated_scan_bytes": args.max_estimated_scan_bytes,
            "estimated_full_file_passes": (
                2 + len(args.backends) * len(PATHS) * (args.repeats + 2)
            ),
            "estimated_repeated_scan_bytes": (
                parquet.stat().st_size
                * (2 + len(args.backends) * len(PATHS) * (args.repeats + 2))
            ),
        },
        "fairness": FAIRNESS,
        "cases": cases,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    parquet = validate_args(parser, args)
    document = run_benchmark(args, parquet)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.expanduser().resolve().open("x", encoding="utf-8") as stream:
            stream.write(rendered)


if __name__ == "__main__":
    main()
