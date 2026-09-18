from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend
from dbnumpy.matrix import DBArray


@pytest.fixture
def values() -> np.ndarray:
    return np.array(
        [
            [0.0, 2.0, 0.0, -1.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0, 6.0],
        ]
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("database_op", "reference_op"),
    [
        (lambda x: x + 2.0, lambda x: x + 2.0),
        (lambda x: 2.0 - x, lambda x: 2.0 - x),
        (lambda x: x * 3.0, lambda x: x * 3.0),
        (lambda x: x / 2.0, lambda x: x / 2.0),
        (lambda x: x**2.0, lambda x: x**2.0),
        (lambda x: x > 1.0, lambda x: (x > 1.0).astype(float)),
        (lambda x: x == 0.0, lambda x: (x == 0.0).astype(float)),
        (lambda x: np.sqrt(abs(x)), lambda x: np.sqrt(abs(x))),
        (lambda x: np.exp(x), lambda x: np.exp(x)),
        (lambda x: np.sin(x), lambda x: np.sin(x)),
    ],
)
def test_scalar_and_unary_parity(
    backend: Backend,
    values: np.ndarray,
    database_op: Callable[[DBArray], DBArray],
    reference_op: Callable[[np.ndarray], np.ndarray],
) -> None:
    matrix = backend.from_scipy(sparse.csr_array(values))
    np.testing.assert_allclose(
        database_op(matrix).to_numpy(),
        reference_op(values),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parity
def test_same_source_branched_expression_parity(
    backend: Backend, values: np.ndarray
) -> None:
    matrix = backend.from_scipy(sparse.csr_array(values))
    actual = ((matrix * 2.0) * (matrix > 1.0)).to_numpy()
    expected = (values * 2.0) * (values > 1.0)
    np.testing.assert_allclose(actual, expected)
    assert "JOIN" not in ((matrix * 2.0) * (matrix > 1.0)).compile().upper()


@pytest.mark.parity
def test_reused_post_relational_branch_is_lowered_once(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    row = np.array([0.25, -0.5, 0.75])
    matrix = backend.from_numpy(values)
    branch = matrix.T + row
    result = (branch + branch) * 0.5

    np.testing.assert_allclose(result.to_numpy(), values.T + row)
    sql = result.compile()
    relations = [
        node["relation"] for node in result.plan()["nodes"] if node["node"] == "source"
    ]
    assert all(sql.count(f'"{relation}"') == 1 for relation in relations)


@pytest.mark.parity
def test_equivalent_transpose_branches_are_lowered_once(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    result = matrix.T + matrix.T

    np.testing.assert_allclose(result.to_numpy(), values.T + values.T)
    assert result.compile().count('"dbm_0"') == 1


@pytest.mark.parity
def test_distinct_matrix_elementwise_parity(backend: Backend) -> None:
    left_values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    right_values = np.array([[1.0, 0.0, 5.0], [0.0, 7.0, 4.0]])
    left = backend.from_scipy(sparse.csr_array(left_values))
    right = backend.from_scipy(sparse.csr_array(right_values))

    np.testing.assert_allclose((left + right).to_numpy(), left_values + right_values)
    np.testing.assert_allclose((left * right).to_numpy(), left_values * right_values)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_reduction_parity(
    backend: Backend, values: np.ndarray, sparse_input: bool
) -> None:
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    for axis in (None, 0, 1):
        np.testing.assert_allclose(matrix.sum(axis=axis), np.sum(values, axis=axis))
        np.testing.assert_allclose(matrix.mean(axis=axis), np.mean(values, axis=axis))
        np.testing.assert_allclose(matrix.var(axis=axis), np.var(values, axis=axis))
        np.testing.assert_allclose(matrix.std(axis=axis), np.std(values, axis=axis))
        np.testing.assert_allclose(
            matrix.var(axis=axis, ddof=1), np.var(values, axis=axis, ddof=1)
        )
        np.testing.assert_allclose(
            matrix.std(axis=axis, ddof=1), np.std(values, axis=axis, ddof=1)
        )


@pytest.mark.parity
def test_transpose_and_numpy_dispatch(backend: Backend, values: np.ndarray) -> None:
    matrix = backend.from_scipy(sparse.csr_array(values))
    np.testing.assert_array_equal(np.transpose(matrix).to_numpy(), values.T)
    assert np.transpose(matrix, axes=(0, 1)) is matrix
    assert np.transpose(matrix, axes=(-2, -1)) is matrix
    np.testing.assert_array_equal(
        np.transpose(matrix, axes=(-1, -2)).to_numpy(), values.T
    )
    np.testing.assert_allclose(np.sum(matrix, axis=0), np.sum(values, axis=0))
    np.testing.assert_allclose(np.mean(matrix, axis=1), np.mean(values, axis=1))


@pytest.mark.parity
def test_sparse_matrix_multiplication(backend: Backend) -> None:
    left_values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    right_values = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    left = backend.from_scipy(sparse.csr_array(left_values))
    right = backend.from_scipy(sparse.csr_array(right_values))

    result = left @ right
    np.testing.assert_allclose(result.to_numpy(), left_values @ right_values)
    sql = result.compile().upper()
    assert "INNER JOIN" in sql
    assert "GROUP BY" in sql


@pytest.mark.parity
def test_densification_after_transpose_uses_transposed_domains(
    backend: Backend,
) -> None:
    values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    result = matrix.T + 1.0
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result.to_numpy(), values.T + 1.0)


@pytest.mark.parity
def test_densification_after_matmul_uses_result_domains(backend: Backend) -> None:
    left_values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    right_values = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    left = backend.from_scipy(sparse.csr_array(left_values))
    right = backend.from_scipy(sparse.csr_array(right_values))
    result = (left @ right) + 1.0
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.to_numpy(), left_values @ right_values + 1.0)


@pytest.mark.parity
def test_numpy_row_vector_broadcasting(backend: Backend, values: np.ndarray) -> None:
    matrix = backend.from_scipy(sparse.csr_array(values))
    vector = np.array([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose((matrix + vector).to_numpy(), values + vector)
    np.testing.assert_allclose((vector - matrix).to_numpy(), vector - values)
    np.testing.assert_allclose(np.add(matrix, vector).to_numpy(), values + vector)


@pytest.mark.parity
def test_zero_dimensional_array_operand_behaves_as_scalar(backend: Backend) -> None:
    values = np.arange(6.0).reshape(2, 3)
    matrix = backend.from_numpy(values)
    scalar = np.array(2.5)

    np.testing.assert_allclose((matrix + scalar).to_numpy(), values + scalar)
    np.testing.assert_allclose((scalar - matrix).to_numpy(), scalar - values)


@pytest.mark.parity
def test_two_dimensional_singleton_broadcasting(backend: Backend) -> None:
    column_values = np.array([[1.0], [2.0], [3.0]])
    row_values = np.array([[10.0, 20.0, 30.0, 40.0]])
    column = backend.from_scipy(sparse.csr_array(column_values))
    row = backend.from_scipy(sparse.csr_array(row_values))

    result = column + row
    assert result.shape == (3, 4)
    np.testing.assert_allclose(result.to_numpy(), column_values + row_values)


@pytest.mark.parity
def test_nonbroadcastable_shapes_fail_early(backend: Backend) -> None:
    left = backend.from_numpy(np.ones((2, 3)))
    right = backend.from_numpy(np.ones((4, 2)))
    with pytest.raises(ValueError, match="not broadcastable"):
        _ = left + right


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_positive_step_slice_parity(backend: Backend, sparse_input: bool) -> None:
    values = np.arange(30.0).reshape(5, 6)
    values[values % 3 == 0] = 0.0
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )

    result = matrix[1:5:2, 0:6:2]
    assert result.shape == values[1:5:2, 0:6:2].shape
    np.testing.assert_array_equal(result.to_numpy(), values[1:5:2, 0:6:2])


@pytest.mark.parity
def test_slice_then_densify_uses_sliced_domains(backend: Backend) -> None:
    values = np.array(
        [
            [0.0, 1.0, 0.0, 2.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0, 6.0],
        ]
    )
    matrix = backend.from_scipy(sparse.csr_array(values))
    result = matrix[1:, 1:4:2] + 1.0
    expected = values[1:, 1:4:2] + 1.0
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parity
def test_empty_slice_preserves_two_dimensional_shape(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    result = matrix[3:3, 1:4]
    assert result.shape == (0, 3)
    assert result.to_numpy().shape == (0, 3)


@pytest.mark.parity
def test_rank_expanding_and_paired_advanced_indexing_are_explicitly_deferred(
    backend: Backend,
) -> None:
    matrix = backend.from_numpy(np.arange(12.0).reshape(3, 4))
    with pytest.raises(NotImplementedError, match="newaxis"):
        _ = matrix[None, :]
    with pytest.raises(NotImplementedError, match="paired semantics"):
        _ = matrix[[2, 0], [3, 1]]


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nan_comparisons_and_reductions_follow_numpy(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[1.0, np.nan], [0.0, 3.0]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )

    np.testing.assert_array_equal((matrix == np.nan).to_numpy(), values == np.nan)
    np.testing.assert_array_equal((matrix != np.nan).to_numpy(), values != np.nan)
    np.testing.assert_array_equal((matrix > 2.0).to_numpy(), values > 2.0)
    assert np.isnan(matrix.sum())
    assert np.isnan(matrix.mean())
    assert np.isnan(matrix.var())
    assert np.isnan(matrix.std())


@pytest.mark.parity
def test_nan_matrix_comparison_follows_numpy(backend: Backend) -> None:
    left_values = np.array([[np.nan, 1.0], [0.0, 2.0]])
    right_values = np.array([[np.nan, 2.0], [0.0, np.nan]])
    left = backend.from_numpy(left_values)
    right = backend.from_numpy(right_values)
    np.testing.assert_array_equal(
        (left == right).to_numpy(), (left_values == right_values).astype(float)
    )
    np.testing.assert_array_equal(
        (left != right).to_numpy(), (left_values != right_values).astype(float)
    )


@pytest.mark.parity
def test_variance_is_stable_for_large_offset_dense_values(backend: Backend) -> None:
    values = 1.0e12 + np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
    )
    matrix = backend.from_numpy(values)
    for axis in (None, 0, 1):
        np.testing.assert_allclose(
            matrix.var(axis=axis),
            np.var(values, axis=axis),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            matrix.std(axis=axis, ddof=1),
            np.std(values, axis=axis, ddof=1),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parity
def test_array_protocol_and_matrix_container_conventions(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    assert len(matrix) == len(values)
    assert matrix.ndim == values.ndim
    assert matrix.size == values.size
    assert matrix.dtype == values.astype(float).dtype
    assert "lazy=True" in repr(matrix)
    np.testing.assert_array_equal(np.asarray(matrix), values)
    np.testing.assert_array_equal(np.asarray(matrix, dtype=np.float32), values)
    with pytest.raises(ValueError, match=r"truth value.*ambiguous"):
        bool(matrix)


@pytest.mark.parity
def test_numpy_matmul_and_keepdims_dispatch(backend: Backend) -> None:
    left_values = np.arange(6.0).reshape(2, 3)
    right_values = np.arange(12.0).reshape(3, 4)
    left = backend.from_numpy(left_values)
    right = backend.from_numpy(right_values)
    np.testing.assert_allclose(
        np.matmul(left, right).to_numpy(), np.matmul(left_values, right_values)
    )
    np.testing.assert_allclose(
        np.sum(left, axis=0, keepdims=True),
        np.sum(left_values, axis=0, keepdims=True),
    )
    np.testing.assert_allclose(
        np.mean(left, keepdims=True), np.mean(left_values, keepdims=True)
    )
    for axis in (-2, -1):
        np.testing.assert_allclose(
            np.sum(left, axis=axis), np.sum(left_values, axis=axis)
        )
        np.testing.assert_allclose(
            np.var(left, axis=axis, keepdims=True),
            np.var(left_values, axis=axis, keepdims=True),
        )


@pytest.mark.parity
def test_numpy_matmul_operands_upload_in_both_directions(backend: Backend) -> None:
    left_values = np.arange(6.0).reshape(2, 3)
    middle_values = np.arange(12.0).reshape(3, 4)
    right_values = np.arange(8.0).reshape(4, 2)
    middle = backend.from_scipy(sparse.csr_array(middle_values))

    np.testing.assert_allclose(
        (left_values @ middle).to_numpy(), left_values @ middle_values
    )
    np.testing.assert_allclose(
        (middle @ right_values).to_numpy(), middle_values @ right_values
    )
    np.testing.assert_allclose(
        np.matmul(left_values, middle).to_numpy(),
        np.matmul(left_values, middle_values),
    )
    with pytest.raises(ValueError, match="must be two-dimensional"):
        _ = middle @ np.ones(4)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("database_op", "reference_op"),
    [
        (lambda x: -x, lambda x: -x),
        (lambda x: abs(x), lambda x: abs(x)),
        (lambda x: np.expm1(x), lambda x: np.expm1(x)),
        (lambda x: np.log(x + 1.0), lambda x: np.log(x + 1.0)),
        (lambda x: np.log1p(x), lambda x: np.log1p(x)),
        (lambda x: np.cos(x), lambda x: np.cos(x)),
        (lambda x: np.tan(x), lambda x: np.tan(x)),
        (lambda x: np.floor(x), lambda x: np.floor(x)),
        (lambda x: np.ceil(x), lambda x: np.ceil(x)),
        (lambda x: 2.0**x, lambda x: 2.0**x),
        (lambda x: x >= 1.0, lambda x: (x >= 1.0).astype(float)),
        (lambda x: x < 1.0, lambda x: (x < 1.0).astype(float)),
        (lambda x: x <= 1.0, lambda x: (x <= 1.0).astype(float)),
        (lambda x: x != 1.0, lambda x: (x != 1.0).astype(float)),
    ],
)
def test_extended_ufunc_and_reverse_operator_parity(
    backend: Backend,
    database_op: Callable[[DBArray], DBArray],
    reference_op: Callable[[np.ndarray], np.ndarray],
) -> None:
    values = np.array([[0.0, 0.5, 2.0], [3.25, 0.0, 4.75]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    np.testing.assert_allclose(
        database_op(matrix).to_numpy(),
        reference_op(values),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parity
def test_reverse_division_matches_numpy_inf_nan_semantics(backend: Backend) -> None:
    values = np.array([[0.0, 2.0], [-4.0, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = 1.0 / values
    np.testing.assert_allclose((1.0 / matrix).to_numpy(), expected, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_real_domain_edges_follow_numpy(backend: Backend, sparse_input: bool) -> None:
    values = np.array([[-4.0, -1.0, 0.0, 1.0, 4.0]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    operations = [
        (lambda x: np.sqrt(x), lambda x: np.sqrt(x)),
        (lambda x: np.log(x), lambda x: np.log(x)),
        (lambda x: np.log1p(x), lambda x: np.log1p(x)),
        (lambda x: x**-1.0, lambda x: x**-1.0),
        (lambda x: 0.0**x, lambda x: 0.0**x),
    ]
    with np.errstate(all="ignore"):
        for database_input, reference_input in (
            (matrix, values),
            (matrix.T, values.T),
        ):
            for database_op, reference_op in operations:
                np.testing.assert_allclose(
                    database_op(database_input).to_numpy(),
                    reference_op(reference_input),
                    equal_nan=True,
                )


@pytest.mark.parity
def test_signed_zero_negative_powers_follow_numpy(backend: Backend) -> None:
    values = np.array([[-0.0, -0.0, 0.0, 0.0]])
    matrix = backend.from_numpy(values)
    with np.errstate(all="ignore"):
        for exponent in (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0):
            np.testing.assert_array_equal(
                (matrix**exponent).to_numpy(), values**exponent
            )
        exponents = np.array([[-3.0, 0.5, -3.0, 0.5]])
        exponent_matrix = backend.from_numpy(exponents)
        np.testing.assert_array_equal(
            (matrix**exponent_matrix).to_numpy(), values**exponents
        )


@pytest.mark.parity
def test_signed_zero_comparisons_follow_numpy(backend: Backend) -> None:
    left_values = np.array([[-0.0, 0.0]])
    right_values = np.array([[0.0, -0.0]])
    left = backend.from_numpy(left_values)
    right = backend.from_numpy(right_values)
    operations = [
        lambda x, y: x == y,
        lambda x, y: x != y,
        lambda x, y: x > y,
        lambda x, y: x >= y,
        lambda x, y: x < y,
        lambda x, y: x <= y,
    ]
    for operation in operations:
        np.testing.assert_array_equal(
            operation(left, 0.0).to_numpy(), operation(left_values, 0.0)
        )
        np.testing.assert_array_equal(
            operation(left, right).to_numpy(),
            operation(left_values, right_values),
        )


@pytest.mark.parity
def test_signed_zero_unary_functions_follow_numpy(backend: Backend) -> None:
    values = np.array([[-0.0, 0.0]])
    matrix = backend.from_numpy(values)
    with np.errstate(all="ignore"):
        for operation in (
            np.negative,
            np.absolute,
            np.sqrt,
            np.exp,
            np.expm1,
            np.log,
            np.log1p,
            np.sin,
            np.cos,
            np.tan,
            np.floor,
            np.ceil,
        ):
            actual = operation(matrix).to_numpy()
            expected = operation(values)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_small_exponential_and_logarithm_values_retain_precision(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[-1.0e-12, -1.0e-16, 0.0, 1.0e-16, 1.0e-12]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    for database_input, reference_input in ((matrix, values), (matrix.T, values.T)):
        np.testing.assert_allclose(
            np.expm1(database_input).to_numpy(),
            np.expm1(reference_input),
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.log1p(database_input).to_numpy(),
            np.log1p(reference_input),
            rtol=1e-14,
            atol=0.0,
        )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nonfinite_values_across_supported_unary_functions(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    operations = [
        np.negative,
        np.absolute,
        np.sqrt,
        np.exp,
        np.expm1,
        np.log,
        np.log1p,
        np.sin,
        np.cos,
        np.tan,
        np.floor,
        np.ceil,
    ]
    with np.errstate(all="ignore"):
        for database_input, reference_input in (
            (matrix, values),
            (matrix.T, values.T),
        ):
            for operation in operations:
                np.testing.assert_allclose(
                    operation(database_input).to_numpy(),
                    operation(reference_input),
                    rtol=1e-12,
                    equal_nan=True,
                )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)
def test_nonfinite_values_across_supported_binary_functions(
    backend: Backend, left_sparse: bool, right_sparse: bool
) -> None:
    left_values = np.array([[-np.inf, -1.0, 0.0], [1.0, np.inf, np.nan]])
    right_values = np.array([[0.0, 2.0, -1.0], [0.0, np.inf, np.nan]])
    left = (
        backend.from_scipy(sparse.csr_array(left_values))
        if left_sparse
        else backend.from_numpy(left_values)
    )
    right = (
        backend.from_scipy(sparse.csr_array(right_values))
        if right_sparse
        else backend.from_numpy(right_values)
    )
    operations = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x**y,
        lambda x, y: y**x,
        lambda x, y: x > y,
        lambda x, y: x >= y,
        lambda x, y: x < y,
        lambda x, y: x <= y,
        lambda x, y: x == y,
        lambda x, y: x != y,
    ]
    with np.errstate(all="ignore"):
        for operation in operations:
            np.testing.assert_allclose(
                operation(left, right).to_numpy(),
                operation(left_values, right_values),
                equal_nan=True,
            )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)
def test_nonfinite_matmul_follows_numpy_or_scipy_storage_semantics(
    backend: Backend, left_sparse: bool, right_sparse: bool
) -> None:
    left_values = np.array([[0.0, np.inf]])
    right_values = np.array([[1.0], [0.0]])
    left_reference = sparse.csr_array(left_values) if left_sparse else left_values
    right_reference = sparse.csr_array(right_values) if right_sparse else right_values
    left = (
        backend.from_scipy(left_reference)
        if left_sparse
        else backend.from_numpy(left_reference)
    )
    right = (
        backend.from_scipy(right_reference)
        if right_sparse
        else backend.from_numpy(right_reference)
    )

    expected = left_reference @ right_reference
    if sparse.issparse(expected):
        expected = expected.toarray()
    np.testing.assert_allclose((left @ right).to_numpy(), expected, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nonfinite_reductions_follow_numpy(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[np.inf, 1.0, 0.0], [2.0, -np.inf, 3.0]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    with np.errstate(all="ignore"):
        for axis in (None, 0, 1):
            np.testing.assert_allclose(
                matrix.sum(axis=axis), np.sum(values, axis=axis), equal_nan=True
            )
            np.testing.assert_allclose(
                matrix.mean(axis=axis), np.mean(values, axis=axis), equal_nan=True
            )
            np.testing.assert_allclose(
                matrix.var(axis=axis), np.var(values, axis=axis), equal_nan=True
            )
            np.testing.assert_allclose(
                matrix.std(axis=axis), np.std(values, axis=axis), equal_nan=True
            )


@pytest.mark.parity
def test_all_zero_sparse_matrix_uses_shape_for_implicit_values(
    backend: Backend,
) -> None:
    values = np.zeros((3, 4))
    matrix = backend.from_scipy(sparse.csr_array(values))
    for axis in (None, 0, 1):
        np.testing.assert_array_equal(matrix.sum(axis=axis), values.sum(axis=axis))
        np.testing.assert_array_equal(matrix.mean(axis=axis), values.mean(axis=axis))
        np.testing.assert_array_equal(matrix.var(axis=axis), values.var(axis=axis))
        np.testing.assert_array_equal(
            matrix.std(axis=axis, ddof=1), values.std(axis=axis, ddof=1)
        )
    np.testing.assert_array_equal(np.exp(matrix).to_numpy(), np.exp(values))

    right_values = np.zeros((4, 2))
    right = backend.from_scipy(sparse.csr_array(right_values))
    np.testing.assert_array_equal((matrix @ right).to_numpy(), values @ right_values)
