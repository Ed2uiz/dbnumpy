from __future__ import annotations

import json

import numpy as np
import pytest

from dbnumpy.ir import (
    BinaryOp,
    Broadcast,
    ElementwiseBinary,
    Gather,
    MatMul,
    ScalarBinary,
    Slice,
    Source,
    StorageKind,
    Unary,
    UnaryOp,
    has_sparse_source,
    implicit_value,
    pointwise_alias_nodes,
    relation_names,
    single_pointwise_source,
    storage_kind,
    to_dag,
    to_dict,
)


@pytest.fixture
def sparse_source() -> Source:
    return Source(
        "matrix_a",
        "matrix_a_rows",
        "matrix_a_cols",
        (3, 4),
        "float64",
        StorageKind.SPARSE,
    )


@pytest.mark.parametrize(
    ("expr_factory", "expected"),
    [
        (lambda x: ScalarBinary(BinaryOp.ADD, x, 0.0), StorageKind.SPARSE),
        (lambda x: ScalarBinary(BinaryOp.ADD, x, 1.0), StorageKind.DENSE),
        (lambda x: ScalarBinary(BinaryOp.MULTIPLY, x, 2.0), StorageKind.SPARSE),
        (lambda x: ScalarBinary(BinaryOp.TRUE_DIVIDE, x, 2.0), StorageKind.SPARSE),
        (lambda x: ScalarBinary(BinaryOp.TRUE_DIVIDE, x, 0.0), StorageKind.DENSE),
        (
            lambda x: ScalarBinary(BinaryOp.TRUE_DIVIDE, x, 1.0, reverse=True),
            StorageKind.DENSE,
        ),
        (lambda x: ScalarBinary(BinaryOp.EQUAL, x, 0.0), StorageKind.DENSE),
        (lambda x: ScalarBinary(BinaryOp.GREATER, x, 0.0), StorageKind.SPARSE),
        (lambda x: Unary(UnaryOp.SQRT, x), StorageKind.SPARSE),
        (lambda x: Unary(UnaryOp.EXP, x), StorageKind.DENSE),
        (lambda x: Unary(UnaryOp.COS, x), StorageKind.DENSE),
        (lambda x: Unary(UnaryOp.SIGN, x), StorageKind.SPARSE),
        (lambda x: Unary(UnaryOp.TRUNC, x), StorageKind.SPARSE),
        (lambda x: Unary(UnaryOp.ISNAN, x), StorageKind.SPARSE),
        (lambda x: Unary(UnaryOp.LOG2, x), StorageKind.DENSE),
        (lambda x: Unary(UnaryOp.LOG10, x), StorageKind.DENSE),
        (lambda x: Unary(UnaryOp.SINH, x), StorageKind.SPARSE),
        (lambda x: Unary(UnaryOp.COSH, x), StorageKind.DENSE),
        (lambda x: Unary(UnaryOp.TANH, x), StorageKind.SPARSE),
    ],
)
def test_zero_image_drives_storage(
    sparse_source: Source,
    expr_factory: object,
    expected: StorageKind,
) -> None:
    expr = expr_factory(sparse_source)  # type: ignore[operator]
    assert storage_kind(expr) is expected


def test_same_source_pointwise_expression_is_fusible(sparse_source: Source) -> None:
    expr = ElementwiseBinary(
        BinaryOp.MULTIPLY,
        ScalarBinary(BinaryOp.ADD, sparse_source, 2.0),
        ScalarBinary(BinaryOp.GREATER, sparse_source, 0.0),
    )
    assert single_pointwise_source(expr) == sparse_source


def test_shared_pointwise_dag_analysis_visits_each_node_once(
    sparse_source: Source,
) -> None:
    expr = ScalarBinary(BinaryOp.ADD, sparse_source, 1.0)
    shared: list[object] = []
    for _ in range(25):
        shared.append(expr)
        expr = ElementwiseBinary(BinaryOp.ADD, expr, expr)
    assert single_pointwise_source(expr) == sparse_source
    assert list(pointwise_alias_nodes(expr)) == shared


def test_deep_sparse_analysis_is_not_limited_by_python_recursion(
    sparse_source: Source,
) -> None:
    expr = sparse_source
    # Two nodes per stage intentionally exceeds the 8,192-entry analysis memo
    # so the cold traversal also exercises bounded-cache eviction.
    for _ in range(5_000):
        expr = Unary(
            UnaryOp.SQRT,
            ScalarBinary(BinaryOp.ADD, expr, 0.0),
        )

    assert implicit_value(expr) == 0.0
    assert storage_kind(expr) is StorageKind.SPARSE
    assert has_sparse_source(expr)


def test_dag_serialization_deduplicates_shared_branches(
    sparse_source: Source,
) -> None:
    branch = ScalarBinary(BinaryOp.ADD, sparse_source, float("nan"))
    expr = ElementwiseBinary(BinaryOp.MULTIPLY, branch, branch)
    plan = to_dag(expr)
    assert len(plan["nodes"]) == 3
    json.dumps(plan, allow_nan=False)
    branch_node = next(
        node for node in plan["nodes"] if node["node"] == "scalar_binary"
    )
    assert branch_node["scalar"] == {"special_float": "nan"}


def test_different_sources_are_not_fusible(sparse_source: Source) -> None:
    other = Source(
        "matrix_b",
        "matrix_b_rows",
        "matrix_b_cols",
        sparse_source.shape,
        "float64",
        StorageKind.SPARSE,
    )
    expr = ElementwiseBinary(BinaryOp.ADD, sparse_source, other)
    assert single_pointwise_source(expr) is None


def test_plan_is_plain_json_data(sparse_source: Source) -> None:
    expr = Unary(UnaryOp.SQRT, ScalarBinary(BinaryOp.ADD, sparse_source, 1.0))
    serialized = to_dict(expr)
    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["node"] == "unary"
    assert serialized["shape"] == [3, 4]


def test_extended_unary_plan_is_serializable_and_typed(sparse_source: Source) -> None:
    expr = Unary(UnaryOp.ISNAN, Unary(UnaryOp.LOG10, sparse_source))
    serialized = to_dict(expr)
    assert serialized["op"] == "isnan"
    assert serialized["arg"]["op"] == "log10"
    assert expr.dtype == "float64"
    json.dumps(to_dag(expr), allow_nan=False)


def test_nonconformable_elementwise_operands_fail(sparse_source: Source) -> None:
    other = Source(
        "matrix_b",
        "matrix_b_rows",
        "matrix_b_cols",
        (4, 3),
        "float64",
        StorageKind.SPARSE,
    )
    with pytest.raises(ValueError, match="identical shapes"):
        ElementwiseBinary(BinaryOp.ADD, sparse_source, other)


def test_matmul_shape_and_storage(sparse_source: Source) -> None:
    other = Source(
        "matrix_b",
        "matrix_b_rows",
        "matrix_b_cols",
        (4, 2),
        "float64",
        StorageKind.DENSE,
    )
    expr = MatMul(sparse_source, other)
    assert expr.shape == (3, 2)
    assert storage_kind(expr) is StorageKind.SPARSE
    assert implicit_value(expr) == 0.0


def test_dense_matmul_with_empty_inner_dimension_has_implicit_zeros() -> None:
    left = Source(
        "matrix_a",
        "matrix_a_rows",
        "matrix_a_cols",
        (2, 0),
        "float64",
        StorageKind.DENSE,
    )
    right = Source(
        "matrix_b",
        "matrix_b_rows",
        "matrix_b_cols",
        (0, 3),
        "float64",
        StorageKind.DENSE,
    )
    expr = MatMul(left, right)
    assert expr.shape == (2, 3)
    assert implicit_value(expr) == 0.0
    assert storage_kind(expr) is StorageKind.SPARSE


def test_nonconformable_matmul_fails(sparse_source: Source) -> None:
    with pytest.raises(ValueError, match="not conformable"):
        MatMul(sparse_source, sparse_source)


def test_comparison_dtype_is_numeric_for_r_parity(sparse_source: Source) -> None:
    expr = ScalarBinary(BinaryOp.GREATER, sparse_source, 0.0)
    assert expr.dtype == "float64"
    assert np.dtype(expr.dtype) == np.dtype(np.float64)


def test_broadcast_validates_singleton_dimensions(sparse_source: Source) -> None:
    broadcast = Broadcast(
        sparse_source,
        (3, 4),
        "target_rows",
        "target_cols",
    )
    assert broadcast.shape == (3, 4)
    assert storage_kind(broadcast) is StorageKind.SPARSE

    with pytest.raises(ValueError, match="cannot broadcast"):
        Broadcast(sparse_source, (2, 4), "target_rows", "target_cols")
    with pytest.raises(ValueError, match="nonnegative"):
        Broadcast(sparse_source, (-1, 4), "target_rows", "target_cols")


def test_slice_validates_normalized_bounds_and_declared_shape(
    sparse_source: Source,
) -> None:
    with pytest.raises(ValueError, match="does not match normalized bounds"):
        Slice(
            sparse_source,
            0,
            3,
            1,
            0,
            4,
            1,
            (99, 4),
            "rows",
            "cols",
        )
    with pytest.raises(ValueError, match="bounds are not normalized"):
        Slice(
            sparse_source,
            0,
            99,
            1,
            0,
            4,
            1,
            (3, 4),
            "rows",
            "cols",
        )


def test_signed_slice_and_gather_keep_callable_free_sparse_semantics(
    sparse_source: Source,
) -> None:
    sliced = Slice(
        sparse_source,
        2,
        -1,
        -1,
        0,
        4,
        1,
        (3, 4),
        "slice_rows",
        "slice_cols",
    )
    gathered = Gather(
        sliced,
        0,
        "row_selector",
        (5, 4),
        "gather_rows",
        "gather_cols",
    )

    assert gathered.shape == (5, 4)
    assert implicit_value(gathered) == 0.0
    assert storage_kind(gathered) is StorageKind.SPARSE
    assert has_sparse_source(gathered)
    assert single_pointwise_source(gathered) is None
    serialized = to_dict(gathered)
    assert serialized["node"] == "gather"
    assert serialized["axis"] == 0
    assert serialized["map_relation"] == "row_selector"
    assert relation_names(gathered) == {
        "matrix_a",
        "matrix_a_rows",
        "matrix_a_cols",
        "slice_rows",
        "slice_cols",
        "row_selector",
        "gather_rows",
        "gather_cols",
    }
    json.dumps(to_dag(gathered), allow_nan=False)


def test_gather_validates_axis_relation_and_preserved_shape(
    sparse_source: Source,
) -> None:
    with pytest.raises(ValueError, match="axis must be"):
        Gather(
            sparse_source,
            2,
            "selector",
            sparse_source.shape,
            "rows",
            "cols",
        )
    with pytest.raises(ValueError, match="axis must be"):
        Gather(
            sparse_source,
            True,
            "selector",
            sparse_source.shape,
            "rows",
            "cols",
        )
    with pytest.raises(ValueError, match="nonnegative"):
        Gather(
            sparse_source,
            0,
            "selector",
            (-1, sparse_source.shape[1]),
            "rows",
            "cols",
        )
    with pytest.raises(ValueError, match="unmapped axis"):
        Gather(
            sparse_source,
            0,
            "selector",
            (2, 5),
            "rows",
            "cols",
        )
