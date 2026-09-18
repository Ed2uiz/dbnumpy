"""Minimal dense database-matrix workflow used by the overview guide."""

import numpy as np

from dbnumpy import DBDenseArray, DuckDBBackend

reference = np.arange(1.0, 10.0).reshape(3, 3)

with DuckDBBackend.connect() as backend:
    matrix = backend.from_numpy(reference, name="overview_matrix")
    assert isinstance(matrix, DBDenseArray)
    assert matrix.shape == (3, 3)

    transformed = np.sqrt(matrix + 1.0)
    assert transformed.shape == reference.shape

    result = transformed.to_numpy()
    np.testing.assert_allclose(result, np.sqrt(reference + 1.0))

    print(matrix)
    print(result)
