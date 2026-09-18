"""Small deterministic differential cases spanning complete semantic plans."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from dbnumpy import DensificationError
from dbnumpy.backends import Backend
from dbnumpy.matrix import DBArray


def _upload(
    backend: Backend, values: np.ndarray, *, sparse_input: bool
) -> DBArray[np.float64]:
    if sparse_input:
        return backend.from_scipy(sparse.csr_array(values))
    return backend.from_numpy(values)


@pytest.mark.parity
@pytest.mark.parametrize("seed", [7, 2026, 91_337])
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_seeded_same_source_pointwise_plan(
    backend: Backend, seed: int, sparse_input: bool
) -> None:
    rng = np.random.default_rng(seed)
    values = rng.integers(-5, 6, size=(4, 5)).astype(np.float64)
    values[rng.random(values.shape) < 0.55] = 0.0
    matrix = _upload(backend, values, sparse_input=sparse_input)

    actual = np.sqrt(abs(np.sin(matrix * 0.25) + np.expm1(matrix * 0.1)) + 0.5)
    expected = np.sqrt(np.abs(np.sin(values * 0.25) + np.expm1(values * 0.1)) + 0.5)

    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=1e-12)
    plan = actual.plan()
    source = next(node for node in plan["nodes"] if node["node"] == "source")
    # A sparse plan may join its dimension domains to enumerate the +0.5
    # result, but the branched value expression should still scan its one
    # physical value source only once.
    assert actual.compile().count(f'"{source["relation"]}"') == 1


@pytest.mark.parity
@pytest.mark.parametrize("seed", [314, 2_718])
@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)
def test_seeded_distinct_source_elementwise_plans(
    backend: Backend, seed: int, left_sparse: bool, right_sparse: bool
) -> None:
    rng = np.random.default_rng(seed)
    left_values = rng.integers(-4, 5, size=(3, 4)).astype(np.float64)
    right_values = rng.integers(-4, 5, size=(3, 4)).astype(np.float64)
    left_values[rng.random(left_values.shape) < 0.45] = 0.0
    right_values[rng.random(right_values.shape) < 0.45] = 0.0
    left = _upload(backend, left_values, sparse_input=left_sparse)
    right = _upload(backend, right_values, sparse_input=right_sparse)

    cases = [
        (left + right, left_values + right_values),
        (left - right, left_values - right_values),
        (left * right, left_values * right_values),
        (left == right, left_values == right_values),
        (left >= right, left_values >= right_values),
        (
            np.exp((left + right) * 0.1),
            np.exp((left_values + right_values) * 0.1),
        ),
        (
            (left + 1.0) / (abs(right) + 1.0),
            (left_values + 1.0) / (np.abs(right_values) + 1.0),
        ),
    ]
    for actual, expected in cases:
        np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize("seed", [101, 808])
@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)
def test_seeded_matmul_transform_and_reductions(
    backend: Backend, seed: int, left_sparse: bool, right_sparse: bool
) -> None:
    rng = np.random.default_rng(seed)
    left_values = rng.integers(-3, 4, size=(5, 4)).astype(np.float64)
    right_values = rng.integers(-3, 4, size=(4, 6)).astype(np.float64)
    left_values[rng.random(left_values.shape) < 0.5] = 0.0
    right_values[rng.random(right_values.shape) < 0.5] = 0.0
    left = _upload(backend, left_values, sparse_input=left_sparse)
    right = _upload(backend, right_values, sparse_input=right_sparse)

    product = left @ right
    expected_product = left_values @ right_values
    transformed = product.T[1:6:2, 0:5:2] + 0.25
    expected_transformed = expected_product.T[1:6:2, 0:5:2] + 0.25
    nonlinear = np.tanh(product)

    np.testing.assert_allclose(transformed.to_numpy(), expected_transformed)
    np.testing.assert_allclose(nonlinear.to_numpy(), np.tanh(expected_product))
    np.testing.assert_allclose(product.sum(axis=0), expected_product.sum(axis=0))
    np.testing.assert_allclose(product.mean(axis=1), expected_product.mean(axis=1))
    np.testing.assert_allclose(product.var(), expected_product.var(), rtol=1e-12)
    np.testing.assert_allclose(product.min(axis=0), expected_product.min(axis=0))
    np.testing.assert_allclose(product.max(axis=1), expected_product.max(axis=1))


@pytest.mark.parity
@pytest.mark.parametrize("seed", range(8))
def test_seeded_cross_shape_composition(backend: Backend, seed: int) -> None:
    rng = np.random.default_rng(20_260_800 + seed)
    rows = int(rng.integers(1, 6))
    inner = int(rng.integers(1, 6))
    cols = int(rng.integers(1, 6))
    left_values = rng.normal(size=(rows, inner))
    peer_values = rng.normal(size=(rows, inner))
    right_values = rng.normal(size=(inner, cols))
    for values in (left_values, peer_values, right_values):
        values[rng.random(values.shape) < 0.45] = 0.0

    left = _upload(backend, left_values, sparse_input=seed % 2 == 1)
    peer = _upload(backend, peer_values, sparse_input=seed % 3 != 0)
    right = _upload(backend, right_values, sparse_input=seed % 2 == 0)

    expression = np.log1p(abs(left - peer) * 0.1) + (left >= peer)
    expected = np.log1p(np.abs(left_values - peer_values) * 0.1) + (
        left_values >= peer_values
    )
    np.testing.assert_allclose(expression.to_numpy(), expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(expression.sum(axis=0), expected.sum(axis=0))

    product = left @ right
    np.testing.assert_allclose(
        product.to_numpy(), left_values @ right_values, rtol=1e-11, atol=1e-11
    )

    row = rng.normal(size=(1, inner))
    column = rng.normal(size=(rows, 1))
    np.testing.assert_allclose(
        (left + row + column).to_numpy(),
        left_values + row + column,
        rtol=1e-11,
        atol=1e-11,
    )

    row_step = int(rng.integers(1, 4))
    col_step = int(rng.integers(1, 4))
    np.testing.assert_allclose(
        expression[::row_step, ::col_step].to_numpy(),
        expected[::row_step, ::col_step],
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["dense-dense", "dense-sparse", "sparse-dense", "sparse-sparse"],
)
def test_empty_inner_dimension_matmul_retains_implicit_zeros(
    backend: Backend, left_sparse: bool, right_sparse: bool
) -> None:
    left_values = np.empty((2, 0), dtype=np.float64)
    right_values = np.empty((0, 3), dtype=np.float64)
    left = _upload(backend, left_values, sparse_input=left_sparse)
    right = _upload(backend, right_values, sparse_input=right_sparse)

    product = left @ right
    expected = left_values @ right_values
    assert product.shape == (2, 3)
    np.testing.assert_array_equal(product.to_numpy(), expected)
    np.testing.assert_array_equal((product + 1.0).to_numpy(), expected + 1.0)


@pytest.mark.parity
def test_large_empty_inner_dimension_product_remains_lazy(backend: Backend) -> None:
    left = backend.from_numpy(np.empty((100_000, 0)))
    right = backend.from_numpy(np.empty((0, 100_000)))
    product = left @ right

    assert product.shape == (100_000, 100_000)
    assert product.sum() == 0.0
    with pytest.raises(DensificationError, match=r"to_numpy\(\).*10,000,000,000"):
        product.to_numpy()


@pytest.mark.parity
def test_large_all_zero_sparse_broadcast_remains_lazy(backend: Backend) -> None:
    column = backend.from_coo([], [], [], shape=(100_000, 1))
    row = backend.from_coo([], [], [], shape=(1, 100_000))
    result = column + row

    assert result.shape == (100_000, 100_000)
    assert result.sum() == 0.0
    host_sparse = result.to_scipy()
    assert host_sparse.shape == result.shape
    assert host_sparse.nnz == 0
    with pytest.raises(DensificationError, match=r"to_numpy\(\).*10,000,000,000"):
        result.to_numpy()
