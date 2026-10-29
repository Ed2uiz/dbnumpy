"""Sparse scaling must not retain the coordinate matrix in a hash table."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dbnumpy import DataFusionBackend


@pytest.mark.parametrize("axis", [0, 1])
def test_external_sparse_scaling_with_small_memory_pool(tmp_path, axis):
    rows, cols = 2048, 256
    path = tmp_path / "counts.parquet"
    pq.write_table(
        pa.table(
            {
                "i": np.repeat(np.arange(rows), cols),
                "j": np.tile(np.arange(cols), rows),
                "x": np.full(rows * cols, 2.0),
            }
        ),
        path,
    )
    with DataFusionBackend.connect(
        target_partitions=1,
        memory_limit_bytes=16 * 1024**2,
        temp_directory=tmp_path,
    ) as backend:
        matrix = backend.from_parquet(path, shape=(rows, cols))
        factors = np.linspace(0.0, 2.0, rows if axis == 0 else cols)
        factors[::3] = 0.0
        vector = factors[:, None] if axis == 0 else factors[None, :]
        expected = np.broadcast_to(2.0 * vector, (rows, cols))
        scaled = matrix * vector
        np.testing.assert_allclose(scaled.sum(), expected.sum())
        np.testing.assert_allclose(scaled.sum(axis=0), expected.sum(axis=0))
        np.testing.assert_allclose(scaled.sum(axis=1), expected.sum(axis=1))
        np.testing.assert_allclose((vector * matrix).sum(), expected.sum())


@pytest.mark.parametrize("axis", [0, 1])
def test_sparse_scaling_preserves_unmatched_coordinates(backend, axis):
    from scipy import sparse

    values = np.array([[np.nan, 2.0, 0.0], [np.inf, -3.0, 4.0], [5.0, 0.0, 6.0]])
    factors = np.array([0.0, 2.0, 0.0])
    vector = factors[:, None] if axis == 0 else factors[None, :]
    matrix = backend.from_scipy(sparse.csr_array(values))
    # Sparse factors omit zero coordinates: the outer join must retain the
    # matching matrix entries, including NaN and Inf multiplied by zero.
    factor_matrix = backend.from_scipy(sparse.csr_array(vector))
    with np.errstate(invalid="ignore"):
        expected = values * vector
    for result in (matrix * factor_matrix, factor_matrix * matrix):
        np.testing.assert_array_equal(result.to_numpy(), expected)
        np.testing.assert_allclose(result.sum(axis=0), expected.sum(axis=0))
