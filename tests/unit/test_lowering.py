from __future__ import annotations

import pytest

from dbnumpy.ir import (
    BinaryOp,
    Broadcast,
    ElementwiseBinary,
    Gather,
    MatMul,
    ReductionOp,
    ScalarBinary,
    Slice,
    Source,
    StorageKind,
    Unary,
    UnaryOp,
)
from dbnumpy.lowering import IbisLowerer


def source(name: str, shape: tuple[int, int] = (3, 4)) -> Source:
    return Source(
        name,
        f"{name}_rows",
        f"{name}_cols",
        shape,
        "float64",
        StorageKind.SPARSE,
    )


def dense_source(name: str, shape: tuple[int, int] = (3, 4)) -> Source:
    return Source(
        name,
        f"{name}_rows",
        f"{name}_cols",
        shape,
        "float64",
        StorageKind.DENSE,
    )


def test_same_source_expression_compiles_without_join() -> None:
    matrix = source("matrix_a")
    expr = ElementwiseBinary(
        BinaryOp.MULTIPLY,
        ScalarBinary(BinaryOp.MULTIPLY, matrix, 2.0),
        ScalarBinary(BinaryOp.GREATER, matrix, 1.0),
    )
    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb")
    assert "JOIN" not in sql.upper()
    assert sql.count('FROM "matrix_a"') == 1
    assert "* 2.0" in sql


def test_shared_pointwise_dag_compiles_with_linear_alias_stages() -> None:
    matrix = source("matrix_a")
    expr = ScalarBinary(BinaryOp.ADD, matrix, 1.0)
    for _ in range(25):
        expr = ElementwiseBinary(BinaryOp.ADD, expr, expr)
    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb")
    assert len(sql) < 10_000
    assert sql.count('"matrix_a"') == 1
    assert sql.upper().count(' AS "__DBM_POINTWISE_VALUE_') == 25


def test_internal_names_are_deterministic_and_do_not_shadow_sources() -> None:
    matrix = dense_source("__dbm_internal_0_pointwise_base")
    branch = ScalarBinary(BinaryOp.ADD, matrix, 1.0)
    expr = ElementwiseBinary(BinaryOp.ADD, branch, branch)
    lowerer = IbisLowerer()

    first = lowerer.compile_matrix(expr, dialect="datafusion")
    second = lowerer.compile_matrix(expr, dialect="datafusion")
    assert first == second
    assert first.count('"__dbm_internal_0_pointwise_base"') == 1
    assert '"__dbm_internal_0_pointwise_base" AS (' not in first


def test_deep_pointwise_reduction_wraps_iterative_sql() -> None:
    matrix = dense_source("matrix_a")
    expr = matrix
    for _ in range(100):
        expr = Unary(
            UnaryOp.SQRT,
            ScalarBinary(
                BinaryOp.ADD,
                ScalarBinary(BinaryOp.MULTIPLY, expr, 1.000001),
                0.000001,
            ),
        )
    sql, _ = IbisLowerer().compile_reduction(
        expr,
        ReductionOp.SUM,
        None,
        dialect="duckdb",
    )
    assert 'WITH "__dbm_internal_0_matrix_reduction" AS (' in sql
    assert sql.count('FROM "matrix_a"') == 1
    assert "SUM(" in sql.upper()


def test_deep_pointwise_compile_is_not_limited_by_analysis_recursion() -> None:
    matrix = dense_source("matrix_a")
    expr = matrix
    for _ in range(500):
        expr = Unary(
            UnaryOp.SQRT,
            ScalarBinary(BinaryOp.ADD, expr, 0.000001),
        )

    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb")
    assert sql.count('FROM "matrix_a"') == 1
    assert len(sql) < 250_000


def test_nonzero_zero_image_compiles_an_explicit_grid() -> None:
    expr = Unary(UnaryOp.EXP, source("matrix_a"))
    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb").upper()
    assert "CROSS JOIN" in sql
    assert "LEFT OUTER JOIN" in sql
    assert "COALESCE" in sql


@pytest.mark.parametrize(
    ("operation", "expected_sql", "expects_grid"),
    [
        (UnaryOp.SIGN, "CASE WHEN ISNAN", False),
        (UnaryOp.TRUNC, "CEIL", False),
        (UnaryOp.ISNAN, "CAST(ISNAN", False),
        (UnaryOp.LOG2, "LOG2", True),
        (UnaryOp.LOG10, "LOG10", True),
        (UnaryOp.SINH, "SINH", False),
        (UnaryOp.COSH, "COSH", True),
        (UnaryOp.TANH, "TANH", False),
    ],
)
def test_extended_unary_lowering_tracks_zero_image(
    operation: UnaryOp, expected_sql: str, expects_grid: bool
) -> None:
    sql = (
        IbisLowerer()
        .compile_matrix(Unary(operation, source("matrix_a")), dialect="duckdb")
        .upper()
    )
    assert expected_sql in sql
    assert ("CROSS JOIN" in sql) is expects_grid


@pytest.mark.parametrize(
    ("operation", "expected_sql", "default"),
    [
        (ReductionOp.MIN, "MIN(", 0.0),
        (ReductionOp.MAX, "MAX(", 0.0),
        (ReductionOp.ANY, "BOOL_OR", False),
        (ReductionOp.ALL, "BOOL_AND", False),
        (ReductionOp.NANSUM, "FILTER", 0.0),
        (ReductionOp.NANMEAN, "ISNAN", 0.0),
        (ReductionOp.NANVAR, "SUM(", 0.0),
        (ReductionOp.NANSTD, "SQRT", 0.0),
        (ReductionOp.NANMIN, "MIN(", 0.0),
        (ReductionOp.NANMAX, "MAX(", 0.0),
    ],
)
def test_extended_reductions_compile_to_portable_aggregates(
    operation: ReductionOp, expected_sql: str, default: float | bool
) -> None:
    sql, lowered = IbisLowerer().compile_reduction(
        source("matrix_a"), operation, 0, dialect="datafusion", ddof=0
    )
    assert expected_sql in sql.upper()
    if operation in {
        ReductionOp.VAR,
        ReductionOp.STD,
        ReductionOp.NANVAR,
        ReductionOp.NANSTD,
    }:
        assert "VAR_POP" not in sql.upper()
    assert lowered.default_value == default


@pytest.mark.parametrize("dialect", ["duckdb", "datafusion"])
@pytest.mark.parametrize("operation", list(ReductionOp))
def test_raw_sparse_reductions_do_not_enumerate_dimension_relations(
    operation: ReductionOp, dialect: str
) -> None:
    sql, _ = IbisLowerer().compile_reduction(
        source("matrix_a"), operation, None, dialect=dialect, ddof=0
    )
    upper_sql = sql.upper()
    assert "GENERATE_SERIES" not in upper_sql
    assert "RANGE(" not in upper_sql
    assert '"MATRIX_A_ROWS"' not in upper_sql
    assert '"MATRIX_A_COLS"' not in upper_sql
    assert "CROSS JOIN" not in upper_sql


@pytest.mark.parametrize("dialect", ["duckdb", "datafusion"])
@pytest.mark.parametrize("operation", [ReductionOp.VAR, ReductionOp.NANVAR])
def test_variance_statistics_are_aliased_once_in_a_linear_plan(
    operation: ReductionOp,
    dialect: str,
) -> None:
    sql, _ = IbisLowerer().compile_reduction(
        source("matrix_a"),
        operation,
        None,
        dialect=dialect,
        ddof=0.5,
    )
    upper_sql = sql.upper()

    assert sql.count('FROM "matrix_a"') == 1
    assert upper_sql.count("OVER (") == 3
    assert upper_sql.count("COUNT(") == 3
    assert upper_sql.count("SUM(") == 1
    assert len(sql) < 14_000


def test_distinct_sparse_sources_compile_outer_join() -> None:
    expr = ElementwiseBinary(BinaryOp.ADD, source("matrix_a"), source("matrix_b"))
    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb").upper()
    assert "FULL OUTER JOIN" in sql
    assert "COALESCE" in sql


def test_matmul_remains_optimizer_visible_relational_algebra() -> None:
    expr = MatMul(source("matrix_a"), source("matrix_b", shape=(4, 2)))
    sql = IbisLowerer().compile_matrix(expr, dialect="datafusion").upper()
    assert "INNER JOIN" in sql
    assert "GROUP BY" in sql
    assert "SUM(" in sql


def test_distinct_dense_sources_do_not_construct_coordinate_grid() -> None:
    expr = ElementwiseBinary(
        BinaryOp.ADD,
        dense_source("matrix_a"),
        dense_source("matrix_b"),
    )
    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb").upper()
    assert "INNER JOIN" in sql
    assert "CROSS JOIN" not in sql
    assert "_ROWS" not in sql
    assert "_COLS" not in sql


def test_dense_sparse_elementwise_uses_dense_domain_without_grid() -> None:
    expr = ElementwiseBinary(
        BinaryOp.ADD,
        dense_source("matrix_a"),
        source("matrix_b"),
    )
    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb").upper()
    assert "LEFT OUTER JOIN" in sql
    assert "CROSS JOIN" not in sql
    assert "_ROWS" not in sql
    assert "_COLS" not in sql


@pytest.mark.parametrize("dialect", ["duckdb", "datafusion"])
def test_gather_is_one_readable_mapping_join_with_bounded_sql(dialect: str) -> None:
    matrix = source("matrix_a")
    expr = Gather(
        matrix,
        0,
        "selector_with_100000_rows",
        (100_000, 4),
        "gather_rows",
        "gather_cols",
    )
    sql = IbisLowerer().compile_matrix(expr, dialect=dialect)
    upper = sql.upper()

    assert upper.count("INNER JOIN") == 1
    assert sql.count('FROM "matrix_a"') == 1
    assert '"selector_with_100000_rows"' in sql
    assert "source_index" in sql and "output_index" in sql
    assert len(sql) < 2_000


@pytest.mark.parametrize("dialect", ["duckdb", "datafusion"])
def test_reverse_slice_is_affine_and_composes_with_pointwise(
    dialect: str,
) -> None:
    matrix = source("matrix_a")
    sliced = Slice(
        matrix,
        2,
        -1,
        -1,
        3,
        -1,
        -2,
        (3, 2),
        "slice_rows",
        "slice_cols",
    )
    expr = ScalarBinary(BinaryOp.ADD, sliced, 1.0)
    sql = IbisLowerer().compile_matrix(expr, dialect=dialect)
    upper = sql.upper()

    assert sql.count('FROM "matrix_a"') == 1
    assert "WHERE" in upper
    assert "<=" in sql and ">" in sql
    assert "%" in sql


def test_shared_source_cross_branch_postorder_is_dependency_safe() -> None:
    matrix = source("matrix_a")
    scalar = Slice(
        matrix,
        1,
        2,
        1,
        1,
        2,
        1,
        (1, 1),
        "scalar_rows",
        "scalar_cols",
    )
    broadcast = Broadcast(scalar, matrix.shape, "matrix_rows", "matrix_cols")
    expr = ElementwiseBinary(BinaryOp.SUBTRACT, broadcast, matrix)

    sql = IbisLowerer().compile_matrix(expr, dialect="duckdb")
    assert "JOIN" in sql.upper()
    assert sql.count('FROM "matrix_a"') == 2
