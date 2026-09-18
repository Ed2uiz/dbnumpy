"""Sparse arithmetic, broadcasting, and products used by the arithmetic guide."""

import numpy as np
from scipy import sparse

from dbnumpy import DBDenseArray, DBSparseArray, DuckDBBackend

reference = np.array(
    [
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0],
    ]
)

with DuckDBBackend.connect() as backend:
    matrix = backend.from_scipy(sparse.csr_array(reference))
    assert isinstance(matrix, DBSparseArray)

    scaled = matrix * 100.0
    shifted = matrix + 1.0
    hadamard = matrix * matrix
    product = matrix @ matrix.T
    broadcast = matrix + np.array([10.0, 20.0, 30.0])

    assert isinstance(scaled, DBSparseArray)
    assert isinstance(shifted, DBDenseArray)

    np.testing.assert_allclose(scaled.to_numpy(), reference * 100.0)
    np.testing.assert_allclose(shifted.to_numpy(), reference + 1.0)
    np.testing.assert_allclose(hadamard.to_numpy(), reference * reference)
    np.testing.assert_allclose(product.to_numpy(), reference @ reference.T)
    np.testing.assert_allclose(
        broadcast.to_numpy(), reference + np.array([10.0, 20.0, 30.0])
    )

    print(product)
    print(product.to_numpy())
