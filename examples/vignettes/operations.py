"""Transpose, reductions, slicing, and execution boundaries for the guide."""

import numpy as np

from dbnumpy import DuckDBBackend

reference = np.arange(1.0, 13.0).reshape(3, 4)

with DuckDBBackend.connect() as backend:
    matrix = backend.from_numpy(reference)

    transposed = matrix.T
    subset = matrix[1:, ::2]
    transformed = np.log1p(matrix)

    np.testing.assert_allclose(transposed.to_numpy(), reference.T)
    np.testing.assert_allclose(subset.to_numpy(), reference[1:, ::2])
    np.testing.assert_allclose(transformed.to_numpy(), np.log1p(reference))

    np.testing.assert_allclose(matrix.sum(axis=0), reference.sum(axis=0))
    np.testing.assert_allclose(matrix.sum(axis=1), reference.sum(axis=1))
    np.testing.assert_allclose(matrix.mean(axis=0), reference.mean(axis=0))
    np.testing.assert_allclose(
        matrix.var(axis=1, ddof=1), reference.var(axis=1, ddof=1)
    )
    np.testing.assert_allclose(
        matrix.std(axis=0, ddof=1), reference.std(axis=0, ddof=1)
    )

    materialized = transformed.compute(name="operations_result")
    np.testing.assert_allclose(materialized.to_numpy(), np.log1p(reference))

    print("shape:", matrix.shape)
    print("column means:", matrix.mean(axis=0))
    print("SQL starts with:", transformed.compile().splitlines()[0])
