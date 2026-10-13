"""Collecting results into NumPy and SciPy is bounded, owned and validated."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from scipy import sparse

from dbnumpy import DensificationError
from dbnumpy.backends import Backend
from dbnumpy.backends.base import _copy_arrow_column


def test_to_scipy_is_bounded_and_owned(backend: Backend) -> None:
    matrix = backend.from_scipy(sparse.csr_array(np.diag([2.0, 3.0, 4.0])))
    backend.max_sparse_host_values = 2
    with pytest.raises(DensificationError, match="max_sparse_host_values=2"):
        matrix.to_scipy()
    backend.max_sparse_host_values = 3
    for format in ("coo", "csr", "csc"):
        exported = matrix.to_scipy(format=format)
        exported.data[:] = -10.0
        np.testing.assert_array_equal(matrix.to_numpy(), np.diag([2.0, 3.0, 4.0]))


def test_arrow_chunks_are_copied_across_boundaries() -> None:
    column = pa.chunked_array([[1.0, np.nan], [], [np.inf, -np.inf, 0.0]])
    copied = _copy_arrow_column(column, np.float64)
    np.testing.assert_array_equal(copied, [1.0, np.nan, np.inf, -np.inf, 0.0])
    copied[:] = 7.0
    assert column.to_pylist()[0] == 1.0


@pytest.mark.parametrize("i", [-1, 2])
def test_collection_rejects_out_of_bounds_coordinates(backend: Backend, i: int) -> None:
    backend._register_arrow("outside", pa.table({"i": [i], "j": [0], "x": [7.0]}))
    matrix = backend.from_relation("outside", shape=(2, 1), storage="sparse")
    for collect in (
        lambda: matrix.to_numpy(),
        lambda: matrix.sum(axis=1),
        lambda: matrix[:, 0].to_numpy(),
    ):
        with pytest.raises(ValueError, match="outside the declared matrix shape"):
            collect()


def test_to_numpy_rejects_duplicate_coordinates(backend: Backend) -> None:
    backend._register_arrow(
        "duplicated", pa.table({"i": [0, 0], "j": [0, 0], "x": [1.0, 5.0]})
    )
    matrix = backend.from_relation("duplicated", shape=(2, 2), storage="sparse")
    with pytest.raises(ValueError, match="duplicate matrix coordinates"):
        matrix.to_numpy()
