from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from dbnumpy import DBArray, DBScalar, DBVector
from dbnumpy.backends import Backend


def _materialize(result: DBArray[Any] | DBVector | DBScalar) -> Any:
    return result.item() if isinstance(result, DBScalar) else result.to_numpy()


def _assert_matches_numpy(
    result: DBArray[Any] | DBVector | DBScalar, expected: Any
) -> None:
    actual = _materialize(result)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert result.shape == np.shape(expected)


def _to_dbnumpy(
    backend: Backend,
    values: np.ndarray[Any, np.dtype[np.float64]],
    *,
    sparse_input: bool,
) -> DBArray[Any]:
    if sparse_input:
        return backend.from_scipy(sparse.csr_array(values))
    return backend.from_numpy(values)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_seeded_indexing_compositions_match_numpy(
    backend: Backend, sparse_input: bool
) -> None:
    """Exercise representative indexing laws without an expensive fuzz matrix."""
    for seed in (9101, 47017):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=(5, 6))
        values[rng.random(values.shape) < 0.45] = 0.0
        matrix = _to_dbnumpy(backend, values, sparse_input=sparse_input)

        row_indices = np.array([-1, 1, -1, 0], dtype=np.int64)
        column_mask = np.array([True, False, True, False, False, True])

        cases: tuple[tuple[Any, Any], ...] = (
            (slice(None, None, -2), slice(5, None, -2)),
            (row_indices, slice(None)),
            (slice(None), column_mask),
            (row_indices, 2),
            (-2, column_mask),
        )
        for key in cases:
            _assert_matches_numpy(matrix[key], values[key])

        cartesian = matrix[row_indices, :][:, [5, 0, 5]]
        expected_cartesian = values[row_indices, :][:, [5, 0, 5]]
        _assert_matches_numpy(cartesian, expected_cartesian)

        scalar = matrix[-2, 2]
        vector = matrix[-1, :]
        assert isinstance(scalar, DBScalar)
        assert isinstance(vector, DBVector)
        assert isinstance(matrix[::-2, 1::2], DBArray)
        _assert_matches_numpy(scalar + vector, values[-2, 2] + values[-1, :])
        _assert_matches_numpy(vector - scalar, values[-1, :] - values[-2, 2])
        _assert_matches_numpy(matrix + scalar, values + values[-2, 2])
        _assert_matches_numpy(
            matrix + (vector + scalar), values + (values[-1, :] + values[-2, 2])
        )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_indexing_empty_axes_matches_numpy(
    backend: Backend, sparse_input: bool
) -> None:
    """Keep zero-sized domains in the property sample for both storage modes."""
    for values in (
        np.empty((0, 4), dtype=np.float64),
        np.empty((3, 0), dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
    ):
        matrix = _to_dbnumpy(backend, values, sparse_input=sparse_input)
        selectors: tuple[tuple[Any, Any], ...] = (
            (slice(None, None, -1), slice(None, None, -1)),
            (np.arange(values.shape[0], dtype=np.int64), slice(None)),
            (slice(None), np.zeros(values.shape[1], dtype=bool)),
        )
        for key in selectors:
            _assert_matches_numpy(matrix[key], values[key])

    empty_vector = _to_dbnumpy(
        backend, np.empty((3, 0), dtype=np.float64), sparse_input=sparse_input
    )[1, :]
    assert isinstance(empty_vector, DBVector)
    _assert_matches_numpy(empty_vector, np.empty((0,), dtype=np.float64))
