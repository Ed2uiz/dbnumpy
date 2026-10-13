"""Matrix multiplication matches NumPy, or SciPy when an operand is sparse."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from dbnumpy import DensificationError
from dbnumpy.backends import Backend

pytestmark = pytest.mark.parity

COMBOS = pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)


@COMBOS
def test_matmul(upload: Any, left_sparse: bool, right_sparse: bool) -> None:
    rng = np.random.default_rng(101)
    left_values = rng.integers(-3, 4, size=(5, 4)).astype(float)
    right_values = rng.integers(-3, 4, size=(4, 6)).astype(float)
    left_values[rng.random(left_values.shape) < 0.5] = 0.0
    right_values[rng.random(right_values.shape) < 0.5] = 0.0
    left, _ = upload(left_values, sparse_input=left_sparse)
    right, _ = upload(right_values, sparse_input=right_sparse)

    product, expected = left @ right, left_values @ right_values
    np.testing.assert_allclose(product.to_numpy(), expected)
    np.testing.assert_allclose((product + 1.0).to_numpy(), expected + 1.0)
    np.testing.assert_allclose(product.T[1:6:2, ::2].to_numpy(), expected.T[1:6:2, ::2])
    np.testing.assert_allclose(product.var(axis=0), expected.var(axis=0))
    np.testing.assert_allclose(
        (left[[4, 0, 4], :] @ right).to_numpy(), expected[[4, 0, 4], :]
    )

    empty = (
        upload(np.empty((2, 0)), sparse_input=left_sparse)[0]
        @ upload(np.empty((0, 3)), sparse_input=right_sparse)[0]
    )
    np.testing.assert_array_equal((empty + 1.0).to_numpy(), np.ones((2, 3)))


@COMBOS
def test_nonfinite_products_follow_numpy_or_scipy(
    upload: Any, left_sparse: bool, right_sparse: bool
) -> None:
    # A stored inf times a missing sparse zero is skipped, as in SciPy.
    left_values, right_values = np.array([[0.0, np.inf]]), np.array([[1.0], [0.0]])
    left, _ = upload(left_values, sparse_input=left_sparse)
    right, _ = upload(right_values, sparse_input=right_sparse)
    left_ref = sparse.csr_array(left_values) if left_sparse else left_values
    right_ref = sparse.csr_array(right_values) if right_sparse else right_values
    with np.errstate(invalid="ignore"):
        expected = left_ref @ right_ref
    expected = expected.toarray() if sparse.issparse(expected) else expected
    np.testing.assert_allclose((left @ right).to_numpy(), expected, equal_nan=True)


def test_host_operands_and_numpy_dispatch(backend: Backend) -> None:
    left, middle, right = (
        np.arange(n * m, dtype=float).reshape(n, m) for n, m in ((2, 3), (3, 4), (4, 2))
    )
    matrix = backend.from_scipy(sparse.csr_array(middle))
    np.testing.assert_allclose((left @ matrix).to_numpy(), left @ middle)
    np.testing.assert_allclose((matrix @ right).to_numpy(), middle @ right)
    np.testing.assert_allclose(np.matmul(left, matrix).to_numpy(), left @ middle)
    np.testing.assert_allclose(np.matmul(matrix, right).to_numpy(), middle @ right)
    with pytest.raises(ValueError, match="must be two-dimensional"):
        matrix @ np.ones(4)


def test_zero_products_stay_lazy_and_canonical(backend: Backend) -> None:
    product = backend.from_numpy(np.empty((100_000, 0))) @ backend.from_numpy(
        np.empty((0, 100_000))
    )
    assert product.sum() == 0.0
    with pytest.raises(DensificationError, match=r"to_numpy\(\).*10,000,000,000"):
        product.to_numpy()

    # 0 * inf is nan in a dense product but a skipped term in to_scipy().
    zero = backend.from_coo([0], [0], [1.0], shape=(2, 2)) * 0
    infinite = backend.from_coo([0], [0], [np.inf], shape=(2, 2))
    assert np.isnan((zero @ infinite).to_numpy()[0, 0])
    assert (zero.to_scipy() @ infinite.to_scipy()).toarray()[0, 0] == 0
