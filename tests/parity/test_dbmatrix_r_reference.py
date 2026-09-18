"""Behavioral cases ported from the local dbmatrix-r test suite.

The assertions use Python-native indexing and NumPy missing-value conventions,
but retain R's tested dense/sparse arithmetic, summaries, and fusion behaviors.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend


@pytest.mark.parity
def test_r_scalar_densification_cases(backend: Backend) -> None:
    values = np.array([[0.0, 2.0], [3.0, 0.0]])
    matrix = backend.from_scipy(sparse.csc_array(values))

    np.testing.assert_array_equal((matrix + 0.0).to_numpy(), values + 0.0)
    np.testing.assert_array_equal((matrix * 0.0).to_numpy(), values * 0.0)
    np.testing.assert_array_equal((matrix * 2.0).to_numpy(), values * 2.0)
    np.testing.assert_array_equal((matrix + 1.0).to_numpy(), values + 1.0)
    np.testing.assert_array_equal((1.0 + matrix).to_numpy(), 1.0 + values)


@pytest.mark.parity
def test_r_row_column_sample_statistics(backend: Backend) -> None:
    values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0]])
    for matrix in (
        backend.from_numpy(values),
        backend.from_scipy(sparse.csr_array(values)),
    ):
        np.testing.assert_allclose(matrix.sum(axis=0), values.sum(axis=0))
        np.testing.assert_allclose(matrix.sum(axis=1), values.sum(axis=1))
        np.testing.assert_allclose(matrix.mean(axis=0), values.mean(axis=0))
        np.testing.assert_allclose(matrix.mean(axis=1), values.mean(axis=1))
        np.testing.assert_allclose(
            matrix.var(axis=0, ddof=1), values.var(axis=0, ddof=1)
        )
        np.testing.assert_allclose(
            matrix.var(axis=1, ddof=1), values.var(axis=1, ddof=1)
        )
        np.testing.assert_allclose(
            matrix.std(axis=0, ddof=1), values.std(axis=0, ddof=1)
        )
        np.testing.assert_allclose(
            matrix.std(axis=1, ddof=1), values.std(axis=1, ddof=1)
        )


@pytest.mark.parity
def test_known_r_sparse_zero_image_gaps_are_corrected(backend: Backend) -> None:
    values = np.array([[0.0, 2.0], [-1.0, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))

    # dbmatrix-r currently transforms only stored rows for these operations.
    np.testing.assert_array_equal((matrix == 0.0).to_numpy(), (values == 0.0))
    np.testing.assert_allclose(np.exp(matrix).to_numpy(), np.exp(values))
    np.testing.assert_allclose(np.cos(matrix).to_numpy(), np.cos(values))


@pytest.mark.parity
def test_r_math_and_summary_extension_on_finite_values(backend: Backend) -> None:
    """Port the finite subset of R's Math and Summary S4 group intent."""

    values = np.array([[0.0, -2.75, 0.0], [1.25, 4.0, -3.0]])
    for matrix in (
        backend.from_numpy(values),
        backend.from_scipy(sparse.csr_array(values)),
    ):
        np.testing.assert_array_equal(np.sign(matrix).to_numpy(), np.sign(values))
        np.testing.assert_array_equal(np.trunc(matrix).to_numpy(), np.trunc(values))
        assert matrix.min() == np.min(values)
        assert matrix.max() == np.max(values)
        assert matrix.any() == np.any(values)
        assert matrix.all() == np.all(values)
