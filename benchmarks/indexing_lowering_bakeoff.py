"""Indexing-focused relational compiler bake-off.

This benchmark compares four ways to build the same one-axis gather over a
registered ``(source_index, output_index)`` relation.  It is deliberately
serial and small by default.  SQLAlchemy is a generic-Core portability probe,
not a claim that SQLAlchemy supports either target dialect.
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
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import ibis
import numpy as np
import pyarrow as pa
import sqlalchemy as sa
from sqlalchemy.engine.default import DefaultDialect

from dbnumpy.backends import DataFusionBackend, DuckDBBackend
from dbnumpy.indexing import _GatherIndices, _normalize_selector
from dbnumpy.ir import Gather, MatrixExpr, Source

PATHS = ("dbverse_ibis", "sqlalchemy_core", "direct_sql", "backend_native")
PATTERNS = ("ordered", "shuffled", "repeated")
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
MAX_SOURCE_CELLS = 2_000_000
MAX_OUTPUT_CELLS = 10_000_000

FAIRNESS = {
    "shared_inputs": (
        "Every path reads the same backend source and registered selector relation."
    ),
    "selector_setup": (
        "Selector normalization and Arrow registration are reported separately and "
        "excluded from every path's frontend and execution timing."
    ),
    "sql_execution": (
        "Ibis, SQLAlchemy, and direct-SQL execution includes target SQL parsing, "
        "planning, execution, and Arrow collection."
    ),
    "native_execution": (
        "Native execution skips SQL serialization and parsing, so it is a lower-bound "
        "control rather than an apples-to-apples translator comparison."
    ),
    "sqlalchemy_scope": (
        "SQLAlchemy uses its generic DefaultDialect for this standard SELECT/JOIN "
        "subset. Neither DuckDB nor DataFusion is an officially included SQLAlchemy "
        "dialect; success here is not general target support."
    ),
    "ordering": (
        "No path sorts relational rows. Correct order is represented by output_index; "
        "host reconstruction uses output coordinates. An eight-run schedule balances "
        "ordinal position and chronological cross-path carryover."
    ),
    "caches": (
        "Frontend timings rebuild expressions and bypass DBArray's compiled-plan "
        "cache. One unrecorded warm-up precedes medians."
    ),
    "resources": (
        "Cases run serially; each engine uses two partitions/threads and a 1 GiB "
        "engine memory limit."
    ),
}


@dataclass(frozen=True, slots=True)
class Timing:
    median_ms: float
    samples_ms: tuple[float, ...]
    value: Any


def timed(factory: Callable[[], Any], *, repeats: int, warmup: bool = True) -> Timing:
    if warmup:
        factory()
    samples: list[float] = []
    value: Any = None
    for _ in range(repeats):
        start = perf_counter()
        value = factory()
        samples.append((perf_counter() - start) * 1_000)
    return Timing(median(samples), tuple(samples), value)


def rotated_execution(
    executors: dict[str, Callable[[], pa.Table]], *, repeats: int
) -> tuple[dict[str, Timing], list[list[str]]]:
    """Warm each path, then balance sample position and predecessor."""

    for position in _BALANCED_WARMUP_ORDER:
        name = PATHS[position]
        executors[name]()
    durations: dict[str, list[float]] = defaultdict(list)
    values: dict[str, pa.Table] = {}
    orders: list[list[str]] = []
    for repeat in range(repeats):
        order = balanced_execution_order(repeat)
        orders.append(order)
        for name in order:
            start = perf_counter()
            values[name] = executors[name]()
            durations[name].append((perf_counter() - start) * 1_000)
    return (
        {
            name: Timing(median(durations[name]), tuple(durations[name]), values[name])
            for name in PATHS
        },
        orders,
    )


def balanced_execution_order(repeat: int) -> list[str]:
    """Balance position and chronological carryover in eight-repeat blocks."""

    cycle, row = divmod(repeat, len(_BALANCED_PATH_ORDERS))
    shift = cycle % len(PATHS)
    return [
        PATHS[(position + shift) % len(PATHS)]
        for position in _BALANCED_PATH_ORDERS[row]
    ]


def make_selector(
    pattern: str, size: int, source_rows: int, *, seed: int
) -> np.ndarray[Any, np.dtype[np.int64]]:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown selector pattern: {pattern}")
    if size < 0 or source_rows < 1:
        raise ValueError("selector size must be nonnegative and source_rows positive")
    if pattern == "ordered":
        return np.arange(size, dtype=np.int64) % source_rows
    rng = np.random.default_rng(seed)
    if pattern == "shuffled":
        blocks = [
            rng.permutation(source_rows)
            for _ in range((size + source_rows - 1) // source_rows)
        ]
        return (
            np.concatenate(blocks)[:size].astype(np.int64, copy=False)
            if blocks
            else np.empty(0, dtype=np.int64)
        )
    pool_size = min(source_rows, max(1, int(np.sqrt(max(size, 1)))))
    return rng.integers(0, pool_size, size=size, dtype=np.int64)


def build_gather(
    source: MatrixExpr,
    selector_relation: str,
    shape: tuple[int, int],
    rows_relation: str,
    cols_relation: str,
) -> Gather:
    return Gather(
        source,
        axis=0,
        map_relation=selector_relation,
        gather_shape=shape,
        rows_relation=rows_relation,
        cols_relation=cols_relation,
    )


def normalize_selector(
    selector: np.ndarray[Any, Any], source_rows: int
) -> np.ndarray[Any, np.dtype[np.int64]]:
    normalized = _normalize_selector(selector, source_rows, axis=0)
    if not isinstance(normalized, _GatherIndices):
        raise AssertionError("benchmark selector did not normalize to a gather")
    return normalized.values


def _gather_relations(expr: Gather) -> tuple[str, str]:
    if not isinstance(expr.arg, Source):
        raise ValueError("the bake-off control accepts a direct Source child only")
    if expr.axis != 0:
        raise ValueError("the bake-off control accepts one row mapping only")
    return expr.arg.relation, expr.map_relation


def sqlalchemy_statement(expr: Gather) -> Any:
    source_name, selector_name = _gather_relations(expr)
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
    return sa.select(
        selector.c.output_index.label("i"), source.c.j, source.c.x
    ).select_from(selector.join(source, selector.c.source_index == source.c.i))


def compile_sqlalchemy(statement: Any) -> str:
    return str(
        statement.compile(
            dialect=DefaultDialect(), compile_kwargs={"literal_binds": True}
        )
    )


def direct_gather_sql(expr: Gather) -> str:
    # Names originate in the backend's validated internal allocator.
    source_name, selector_name = _gather_relations(expr)
    return (
        f'SELECT m."output_index" AS "i", s."j", s."x" '
        f'FROM "{selector_name}" AS m INNER JOIN "{source_name}" AS s '
        'ON m."source_index" = s."i"'
    )


def _native_plan(backend: Any, expr: Gather) -> Any:
    source_name, selector_name = _gather_relations(expr)
    if isinstance(backend, DuckDBBackend):
        source = backend.connection.table(source_name).set_alias("s")
        selector = backend.connection.table(selector_name).set_alias("m")
        return selector.join(source, "m.source_index = s.i").project(
            "m.output_index AS i, s.j AS j, s.x AS x"
        )
    from datafusion import col

    source = backend.context.table(source_name)
    selector = backend.context.table(selector_name)
    return selector.join(
        source, left_on="source_index", right_on="i", how="inner"
    ).select(col("output_index").alias("i"), col("j"), col("x"))


def _native_collect(plan: Any) -> pa.Table:
    return plan.to_arrow_table()


def _assert_result(table: pa.Table, expected: np.ndarray[Any, Any]) -> tuple[int, str]:
    actual = np.zeros(expected.shape, dtype=np.float64)
    if table.num_rows:
        i = table.column("i").to_numpy().astype(np.intp, copy=False)
        j = table.column("j").to_numpy().astype(np.intp, copy=False)
        x = table.column("x").to_numpy()
        coordinates = np.stack((i, j), axis=1)
        if len(np.unique(coordinates, axis=0)) != table.num_rows:
            raise AssertionError("gather emitted duplicate output coordinates")
        actual[i, j] = x
    expected_rows = int(np.count_nonzero(expected))
    if table.num_rows != expected_rows:
        raise AssertionError(
            f"gather emitted {table.num_rows} rows; expected {expected_rows}"
        )
    np.testing.assert_array_equal(actual, expected)
    return table.num_rows, "exact"


def _path_row(
    *,
    backend: str,
    pattern: str,
    selector_size: int,
    path: str,
    support: str,
    lowering_build: Timing,
    compile_timing: Timing | None,
    execute_timing: Timing,
    sql: str | None,
    result_rows: int,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "pattern": pattern,
        "selector_size": selector_size,
        "path": path,
        "support_status": support,
        "lowering_build_median_ms": lowering_build.median_ms,
        "lowering_build_samples_ms": list(lowering_build.samples_ms),
        "sql_compile_median_ms": (
            None if compile_timing is None else compile_timing.median_ms
        ),
        "sql_compile_samples_ms": (
            None if compile_timing is None else list(compile_timing.samples_ms)
        ),
        "execute_collect_median_ms": execute_timing.median_ms,
        "execute_collect_samples_ms": list(execute_timing.samples_ms),
        "sql_bytes": None if sql is None else len(sql.encode()),
        "sql_sha256": None
        if sql is None
        else hashlib.sha256(sql.encode()).hexdigest(),
        "sql": sql,
        "result_rows": result_rows,
        "correctness": "exact",
    }


def _source_values(rows: int, cols: int, storage: str) -> np.ndarray[Any, Any]:
    values = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) + 1.0
    if storage == "sparse":
        values[(np.indices(values.shape).sum(axis=0) % 4) != 0] = 0.0
    return values


def run_case(
    backend_name: str,
    *,
    source_rows: int,
    source_cols: int,
    selector_size: int,
    pattern: str,
    storage: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    common = {
        "max_densify_cells": max(source_rows * source_cols, 1),
        "max_host_values": max(selector_size, 1),
        "max_selector_values": max(selector_size, 1),
    }
    if backend_name == "duckdb":
        backend = DuckDBBackend.connect(memory_limit="1GB", threads=2, **common)
    else:
        backend = DataFusionBackend.connect(
            memory_limit_bytes=1024**3, target_partitions=2, **common
        )
    try:
        values = _source_values(source_rows, source_cols, storage)
        if storage == "dense":
            matrix = backend.from_numpy(values, name="index_source")
        else:
            i, j = np.nonzero(values)
            matrix = backend.from_coo(
                i, j, values[i, j], shape=values.shape, name="index_source"
            )

        selector_fixture = make_selector(
            pattern, selector_size, source_rows, seed=seed
        )
        normalization = timed(
            lambda: normalize_selector(selector_fixture, source_rows),
            repeats=repeats,
        )
        selector = normalization.value
        registration = timed(
            lambda: backend._selector_relation(selector),  # noqa: SLF001
            repeats=1,
            warmup=False,
        )
        selector_name = registration.value
        output_shape = (selector_size, source_cols)
        dimensions = timed(
            lambda: backend.dimension_relations(output_shape),
            repeats=1,
            warmup=False,
        )
        rows_relation, cols_relation = dimensions.value
        ir_build = timed(
            lambda: build_gather(
                matrix._expr,  # noqa: SLF001 - benchmark intentionally isolates IR
                selector_name,
                output_shape,
                rows_relation,
                cols_relation,
            ),
            repeats=repeats,
        )
        expr = ir_build.value
        expected = values[selector, :]

        ibis_build = timed(
            lambda: backend.lowerer.lower_matrix(expr).table, repeats=repeats
        )
        ibis_compile = timed(
            lambda: str(ibis.to_sql(ibis_build.value, dialect=backend.dialect)),
            repeats=repeats,
        )

        sa_build = timed(
            lambda: sqlalchemy_statement(expr),
            repeats=repeats,
        )
        sa_compile = timed(lambda: compile_sqlalchemy(sa_build.value), repeats=repeats)

        direct_build = timed(
            lambda: direct_gather_sql(expr),
            repeats=repeats,
        )

        native_build = timed(
            lambda: _native_plan(backend, expr),
            repeats=repeats,
        )
        execution, execution_orders = rotated_execution(
            {
                "dbverse_ibis": lambda: backend._execute_sql(  # noqa: SLF001
                    ibis_compile.value
                ),
                "sqlalchemy_core": lambda: backend._execute_sql(  # noqa: SLF001
                    sa_compile.value
                ),
                "direct_sql": lambda: backend._execute_sql(  # noqa: SLF001
                    direct_build.value
                ),
                "backend_native": lambda: _native_collect(native_build.value),
            },
            repeats=repeats,
        )
        path_data = (
            (
                "dbverse_ibis",
                "project_default_supported_dialects",
                ibis_build,
                ibis_compile,
                ibis_compile.value,
            ),
            (
                "sqlalchemy_core",
                "generic_sql_subset_probe_only",
                sa_build,
                sa_compile,
                sa_compile.value,
            ),
            (
                "direct_sql",
                "narrow_handwritten_control",
                direct_build,
                None,
                direct_build.value,
            ),
            (
                "backend_native",
                "backend_specific_lower_bound_control",
                native_build,
                None,
                None,
            ),
        )
        results: list[dict[str, Any]] = []
        for name, support, build, compile_timing, sql in path_data:
            result_rows, _ = _assert_result(execution[name].value, expected)
            results.append(
                _path_row(
                    backend=backend_name,
                    pattern=pattern,
                    selector_size=selector_size,
                    path=name,
                    support=support,
                    lowering_build=build,
                    compile_timing=compile_timing,
                    execute_timing=execution[name],
                    sql=sql,
                    result_rows=result_rows,
                )
            )

        return {
            "backend": backend_name,
            "pattern": pattern,
            "selector_size": selector_size,
            "source_shape": [source_rows, source_cols],
            "storage": storage,
            "selector_normalization_median_ms": normalization.median_ms,
            "selector_normalization_samples_ms": list(normalization.samples_ms),
            "selector_registration_ms": registration.median_ms,
            "dimension_registration_ms": dimensions.median_ms,
            "semantic_ir_build_median_ms": ir_build.median_ms,
            "semantic_ir_build_samples_ms": list(ir_build.samples_ms),
            "execution_orders": execution_orders,
            "paths": results,
        }
    finally:
        backend.close()


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if (
        not result
        or any(item < 0 for item in result)
        or len(set(result)) != len(result)
    ):
        raise argparse.ArgumentTypeError("sizes must be unique nonnegative integers")
    return result


def parse_csv_choices(value: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(value.split(","))
    if (
        not result
        or len(set(result)) != len(result)
        or any(x not in choices for x in result)
    ):
        raise argparse.ArgumentTypeError(
            f"expected unique comma-separated choices from {choices}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", type=lambda x: parse_csv_choices(x, BACKENDS), default=BACKENDS
    )
    parser.add_argument(
        "--patterns", type=lambda x: parse_csv_choices(x, PATTERNS), default=PATTERNS
    )
    parser.add_argument("--selector-sizes", type=parse_csv_ints, default=(8, 256, 4096))
    parser.add_argument("--source-rows", type=int, default=8192)
    parser.add_argument("--source-cols", type=int, default=16)
    parser.add_argument("--storage", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--format", choices=("json", "console"), default="console")
    parser.add_argument(
        "--output",
        type=Path,
        help="write the rendered result to a new file instead of stdout",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.source_rows < 1 or args.source_cols < 1:
        parser.error("source dimensions must be positive")
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be between 1 and 100")
    source_cells = args.source_rows * args.source_cols
    if source_cells > MAX_SOURCE_CELLS:
        parser.error(
            f"source allocation is limited to {MAX_SOURCE_CELLS:,} logical cells"
        )
    max_output = max(args.selector_sizes) * args.source_cols
    if max_output > MAX_OUTPUT_CELLS:
        parser.error(
            f"default safety guard limits gathered output to "
            f"{MAX_OUTPUT_CELLS:,} cells"
        )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    cases = [
        run_case(
            backend,
            source_rows=args.source_rows,
            source_cols=args.source_cols,
            selector_size=size,
            pattern=pattern,
            storage=args.storage,
            repeats=args.repeats,
            seed=args.seed,
        )
        for backend in args.backends
        for pattern in args.patterns
        for size in args.selector_sizes
    ]
    return {
        "benchmark": "indexing_lowering_bakeoff",
        "completed_at": datetime.now(UTC).isoformat(),
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
        "config": {
            "backends": list(args.backends),
            "patterns": list(args.patterns),
            "selector_sizes": list(args.selector_sizes),
            "source_shape": [args.source_rows, args.source_cols],
            "storage": args.storage,
            "repeats": args.repeats,
            "serial": True,
            "engine_threads_or_partitions": 2,
            "engine_memory_limit_bytes": 1024**3,
            "maximum_source_cells": MAX_SOURCE_CELLS,
            "maximum_output_cells": MAX_OUTPUT_CELLS,
            "path_execution_order": (
                "eight_run_position_and_chronological_predecessor_balance"
            ),
            "execution_order_balanced": args.repeats % 8 == 0,
            "execution_warmup_order": [
                PATHS[position] for position in _BALANCED_WARMUP_ORDER
            ],
            "source_boundary": "host Arrow relation registered once per case",
            "command": [sys.executable, *sys.argv],
        },
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--short"),
            "benchmark_script_sha256": _file_sha256(Path(__file__)),
            "source_tree_sha256": _source_tree_sha256(repo),
        },
        "fairness": FAIRNESS,
        "cases": cases,
    }


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(repo: Path) -> str:
    """Fingerprint the local implementation even when the tree is uncommitted."""

    candidates = [repo / "pyproject.toml", Path(__file__).resolve()]
    candidates.extend(sorted((repo / "src" / "dbnumpy").rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(candidates)):
        relative = path.relative_to(repo).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def render_console(document: dict[str, Any]) -> str:
    lines = [
        "backend    pattern    size  path              build   compile  execute  bytes",
        "---------- ---------- ----- ----------------- ------- ------- -------- ------",
    ]
    for case in document["cases"]:
        for row in case["paths"]:
            compile_ms = row["sql_compile_median_ms"]
            lines.append(
                f"{row['backend']:<10} {row['pattern']:<10} {row['selector_size']:>5} "
                f"{row['path']:<17} {row['lowering_build_median_ms']:>9.3f} "
                f"{'—' if compile_ms is None else f'{compile_ms:.3f}':>11} "
                f"{row['execute_collect_median_ms']:>10.3f} "
                f"{'—' if row['sql_bytes'] is None else row['sql_bytes']:>9}"
            )
    lines.append("\nFairness notes:")
    lines.extend(f"- {value}" for value in document["fairness"].values())
    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    document = run_benchmark(args)
    rendered = (
        json.dumps(document, indent=2, sort_keys=True)
        if args.format == "json"
        else render_console(document)
    )
    if args.output is None:
        print(rendered)
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(
            f"output parent directory does not exist: {args.output.parent}"
        )
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
