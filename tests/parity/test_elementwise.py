"""Elementwise operations match NumPy, including IEEE edge values."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from dbnumpy import DensificationError, StorageKind
from dbnumpy.backends import Backend
from dbnumpy.lowering.ibis import _relational_sql

pytestmark = pytest.mark.parity

EDGE_VALUES = [
    [-np.inf, -1000.0, -2.0, -1.0, -0.5, -1e-16, -0.0, 0.0],
    [1e-12, 0.5, 1.0, 2.0, 16.0, 1000.0, np.inf, np.nan],
]

# ufunc, matrix method, alias
UNARY = [
    (np.negative, None, None),
    (np.absolute, None, None),
    (np.sqrt, "sqrt", None),
    (np.exp, "exp", None),
    (np.expm1, "expm1", None),
    (np.log, "log", None),
    (np.log1p, "log1p", None),
    (np.log2, "log2", None),
    (np.log10, "log10", None),
    (np.sin, "sin", None),
    (np.cos, "cos", None),
    (np.tan, "tan", None),
    (np.arcsin, "arcsin", "asin"),
    (np.arccos, "arccos", "acos"),
    (np.arctan, "arctan", "atan"),
    (np.sinh, "sinh", None),
    (np.cosh, "cosh", None),
    (np.tanh, "tanh", None),
    (np.floor, "floor", None),
    (np.ceil, "ceil", None),
    (np.trunc, "trunc", None),
    (np.sign, "sign", None),
    (np.isnan, "isnan", None),
]

# NumPy ufunc and the matching Python operator (which also covers x.__rsub__ etc.)
BINARY = [
    (np.add, operator.add),
    (np.subtract, operator.sub),
    (np.multiply, operator.mul),
    (np.true_divide, operator.truediv),
    (np.power, operator.pow),
    (np.greater, operator.gt),
    (np.greater_equal, operator.ge),
    (np.less, operator.lt),
    (np.less_equal, operator.le),
    (np.equal, operator.eq),
    (np.not_equal, operator.ne),
]


def numpy_result(operation: Any, *args: Any) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.asarray(operation(*args), dtype=np.float64)


def query_paths(backend: Backend, expression: Any) -> list[np.ndarray]:
    """Collect an expression through every query builder that can produce it."""
    results = [expression.to_numpy()]
    for sql in (
        expression.compile(),
        _relational_sql(
            backend.lowerer.lower_matrix(expression._expr).table,
            dialect=backend.dialect,
        ),
    ):
        table = backend._execute_sql(sql)
        values = np.zeros(expression.shape)
        values[table["i"].to_numpy(), table["j"].to_numpy()] = table["x"].to_numpy()
        results.append(values)
    return results


def assert_matches(actual: Any, expected: Any, *, signed: bool, msg: str) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-12, atol=0.0, equal_nan=True, err_msg=msg
    )
    if signed:
        zeros = expected == 0.0
        np.testing.assert_array_equal(
            np.signbit(actual[zeros]), np.signbit(expected[zeros]), err_msg=msg
        )


@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    ("operation", "method", "alias"), UNARY, ids=[op[0].__name__ for op in UNARY]
)
def test_unary(
    backend: Backend,
    upload: Any,
    sparse_input: bool,
    operation: Any,
    method: Any,
    alias: Any,
) -> None:
    matrix, values = upload(EDGE_VALUES, sparse_input=sparse_input)
    expected = numpy_result(operation, values)
    result = operation(matrix)

    for actual in query_paths(backend, result):
        assert_matches(actual, expected, signed=not sparse_input, msg="query path")
    assert_matches(
        operation(matrix.T[1:, :]).to_numpy(),
        numpy_result(operation, values.T[1:, :]),
        signed=not sparse_input,
        msg="after a relational step",
    )
    for name in filter(None, (method, alias)):
        np.testing.assert_array_equal(
            getattr(matrix, name)().to_numpy(), result.to_numpy()
        )
    if sparse_input:
        stays_sparse = numpy_result(operation, 0.0) == 0.0
        assert result.storage is (
            StorageKind.SPARSE if stays_sparse else StorageKind.DENSE
        )


# 0.5 is avoided as an exponent: NumPy evaluates x ** 0.5 as sqrt(x), which
# differs from power() at -0.0 and -inf.
LEFT = [
    [-np.inf, -2.0, -1.0, -0.0, 0.0, 0.0, 0.25],
    [1.0, 2.0, np.inf, np.nan, 0.0, -0.0, 3.0],
]
RIGHT = [
    [0.0, 2.0, -1.0, -3.0, -0.0, 0.25, 0.0],
    [0.0, -0.5, np.inf, np.nan, -1.0, 2.0, np.nan],
]
SCALARS = [-1.0, -0.0, 0.0, 0.25, 2.0, np.inf, np.nan]


@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)
def test_binary_between_matrices(
    backend: Backend, upload: Any, left_sparse: bool, right_sparse: bool
) -> None:
    left, left_values = upload(LEFT, sparse_input=left_sparse)
    right, right_values = upload(RIGHT, sparse_input=right_sparse)
    signed = not (left_sparse or right_sparse)
    for operation, _ in BINARY:
        expected = numpy_result(operation, left_values, right_values)
        for actual in query_paths(backend, operation(left, right)):
            assert_matches(actual, expected, signed=signed, msg=operation.__name__)


@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_binary_with_scalars(backend: Backend, upload: Any, sparse_input: bool) -> None:
    matrix, values = upload(LEFT, sparse_input=sparse_input)
    for ufunc, python_operator in BINARY:
        for scalar in SCALARS:
            for actual, expected in (
                (python_operator(matrix, scalar), numpy_result(ufunc, values, scalar)),
                (python_operator(scalar, matrix), numpy_result(ufunc, scalar, values)),
            ):
                for result in query_paths(backend, actual):
                    assert_matches(
                        result,
                        expected,
                        signed=not sparse_input,
                        msg=f"{ufunc.__name__} with {scalar}",
                    )


@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_broadcasting(backend: Backend, upload: Any, sparse_input: bool) -> None:
    matrix, values = upload(
        [[0.0, 2.0, 0.0], [3.0, 0.0, -1.0]], sparse_input=sparse_input
    )
    row, column = np.array([1.0, 2.0, 3.0]), np.array([[10.0], [20.0]])
    cases = [
        (matrix + row, values + row),
        (row - matrix, row - values),
        (matrix * column, values * column),
        (matrix + np.array(2.5), values + 2.5),
        (matrix + matrix[0, :], values + values[0, :]),
        (matrix[:, 1] + matrix[1, :1], values[:, 1] + values[1, :1]),
        (matrix[:, :1] + matrix[:1, :], values[:, :1] + values[:1, :]),
    ]
    for actual, expected in cases:
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual.to_numpy(), expected)
    with pytest.raises(ValueError, match="not broadcastable"):
        matrix + np.ones((4, 2))
    with pytest.raises(ValueError, match="equal shapes"):
        matrix[0, :] + matrix[:, 0]
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        matrix + np.ones((1, 2, 3))


def test_huge_sparse_broadcast_stays_lazy(backend: Backend) -> None:
    result = backend.from_coo([], [], [], shape=(100_000, 1)) + backend.from_coo(
        [], [], [], shape=(1, 100_000)
    )
    assert result.sum() == 0.0
    assert result.to_scipy().nnz == 0
    with pytest.raises(DensificationError, match=r"to_numpy\(\).*10,000,000,000"):
        result.to_numpy()


@pytest.mark.parametrize("factor_shape", [(1, 11), (11, 1), (1, 1)])
def test_finite_factors_keep_sparse_scaling_sparse(
    backend: Backend, factor_shape: tuple[int, int]
) -> None:
    backend.max_densify_cells = 100
    matrix = backend.from_coo([0, 10], [1, 10], [2.0, 4.0], shape=(11, 11))
    factors = np.arange(np.prod(factor_shape), dtype=float).reshape(factor_shape) + 2
    expected = matrix.to_scipy().toarray() * factors
    for scaled in (matrix * factors, factors * matrix, matrix * matrix[0, 1]):
        assert scaled.storage is StorageKind.SPARSE
        assert "dbm_dim" not in scaled.compile()
    np.testing.assert_array_equal((matrix * factors).to_scipy().toarray(), expected)
    np.testing.assert_array_equal((factors * matrix).T.to_scipy().toarray(), expected.T)


def test_nonfinite_or_unproven_factors_densify(backend: Backend) -> None:
    values = np.array([[0.0, 2.0], [np.inf, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    for factors in (np.array([np.inf, 2.0]), np.array([np.nan, 2.0])):
        np.testing.assert_allclose(
            (matrix * factors).to_numpy(),
            numpy_result(np.multiply, values, factors),
            equal_nan=True,
        )
    factors = backend.from_coo([0], [1], [2.0], shape=(1, 2))
    np.testing.assert_allclose(
        (matrix * factors).to_numpy(),
        numpy_result(np.multiply, values, factors.to_numpy()),
        equal_nan=True,
    )
    backend.max_densify_cells = 3
    with pytest.raises(DensificationError):
        matrix * np.array([np.inf, 2.0])
    with pytest.raises(DensificationError):
        matrix * (backend.from_numpy(np.full((1, 2), 1e308)) * 2)
