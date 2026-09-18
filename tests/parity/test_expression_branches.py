"""Branches must keep their meaning when inputs are shared or reordered."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend
from dbnumpy.lowering.ibis import _relational_sql


def assert_values(actual: object, expected: object) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual), mask(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize(
    "operation", ["sqrt", "log", "log1p", "expm1", "sin", "arcsin", "negative"]
)
def test_query_builders_agree_on_nonfinite_values(
    backend: Backend, operation: str
) -> None:
    values = np.array([[-np.inf, -1.0, -0.0, 0.0, 1e-12, 1.0, np.inf, np.nan]])
    matrix = backend.from_numpy(values)
    with np.errstate(all="ignore"):
        expression = getattr(np, operation)(matrix)
        expected = getattr(np, operation)(values)
    sql_paths = (
        expression.compile(),
        _relational_sql(
            backend.lowerer.lower_matrix(expression._expr).table,
            dialect=backend.dialect,
        ),
    )
    for sql in sql_paths:
        table = backend._execute_sql(sql)
        actual = np.zeros(values.shape)
        actual[table["i"].to_numpy(), table["j"].to_numpy()] = table["x"].to_numpy()
        assert_values(actual, expected)
    assert_values(expression.to_numpy(), expected)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True])
def test_staged_indexing_with_repeated_positions_and_shared_branches(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[0.0, np.nan, -np.inf], [1.0, np.inf, -1.0], [0.0, 2.0, 0.0]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    for _ in range(12):
        matrix = matrix[[2, 0, 2], :][:, ::-1].T
        values = values[[2, 0, 2], :][:, ::-1].T
    with np.errstate(all="ignore"):
        result, expected = (matrix + 1) + matrix, (values + 1) + values
        assert_values(result.to_numpy(), expected)
        assert_values(result.compute().to_numpy(), expected)
        assert_values(result.sum(axis=0), expected.sum(axis=0))


@pytest.mark.parity
@pytest.mark.parametrize("transform", ["pointwise", "transpose", "gather"])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_shared_ancestor_in_either_operand_order(
    backend: Backend, transform: str, sparse_input: bool
) -> None:
    values = np.array([[0.0, 1.0], [2.0, 0.0]])
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    if transform == "pointwise":
        matrix, values = np.sqrt(matrix + 1), np.sqrt(values + 1)
    elif transform == "transpose":
        matrix, values = matrix.T, values.T
    else:
        matrix, values = matrix[[1, 0, 1], :], values[[1, 0, 1], :]

    for result in ((matrix + 1) + matrix, matrix + (matrix + 1)):
        expected = (values + 1) + values
        assert result.compile()
        assert_values(result.to_numpy(), expected)
        assert_values(result.sum(axis=0), expected.sum(axis=0))
        assert_values(result.to_scipy().toarray(), expected)
        assert_values(result.compute().to_numpy(), expected)
        assert result.explain()


@pytest.mark.parity
@pytest.mark.parametrize("seed", range(12))
def test_generated_branches_match_numpy(backend: Backend, seed: int) -> None:
    """Retain earlier arrays so new branches can overlap whole subqueries."""
    rng = np.random.default_rng(20260915 + seed)
    values = rng.choice(
        [0.0, 0.0, 0.0, -1.0, 1.0, 2.0, 1e-10, np.nan, np.inf, -np.inf],
        size=(4, 5),
    )
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if seed % 2
        else backend.from_numpy(values)
    )
    pool = [(matrix, values)]
    operations: list[object] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(18):
            matrix, values = pool[-1]
            choice = int(rng.integers(0, 5))
            if choice == 0:
                op = str(
                    rng.choice(["sin", "sqrt", "log1p", "absolute", "negative", "cos"])
                )
                matrix, values = getattr(np, op)(matrix), getattr(np, op)(values)
                operations.append(op)
            elif choice == 1:
                op = str(rng.choice(["add", "multiply", "true_divide", "power"]))
                scalar = float(rng.choice([0.0, 0.5, -1.0, 2.0]))
                matrix = getattr(np, op)(matrix, scalar)
                values = getattr(np, op)(values, scalar)
                operations.append((op, scalar))
            elif choice == 2:
                indices = rng.integers(
                    0, values.shape[0], size=values.shape[0]
                ).tolist()
                matrix, values = matrix[indices, :], values[indices, :]
                operations.append(("gather", indices))
            elif choice == 3:
                matrix, values = matrix.T, values.T
                operations.append("transpose")
            else:
                matches = [
                    j for j, (_, ref) in enumerate(pool) if ref.shape == values.shape
                ]
                index = int(rng.choice(matches))
                right, reference = pool[index]
                op = str(rng.choice(["add", "multiply", "subtract"]))
                matrix = getattr(np, op)(matrix, right)
                values = getattr(np, op)(values, reference)
                operations.append((op, "pool", index))
            pool.append((matrix, values))
        try:
            assert_values(matrix.to_numpy(), values)
            assert_values(matrix.sum(axis=0), values.sum(axis=0))
            assert_values(matrix.compute().to_numpy(), values)
        except Exception as error:
            error.add_note(f"seed={20260915 + seed}; operations={operations!r}")
            raise
