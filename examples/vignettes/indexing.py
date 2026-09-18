"""Lazy NumPy-shaped basic and one-axis advanced indexing."""

import numpy as np

from dbnumpy import DBScalar, DBVector, DuckDBBackend

reference = np.arange(20.0).reshape(4, 5)

with DuckDBBackend.connect() as backend:
    matrix = backend.from_numpy(reference)

    row = matrix[1, :]
    column = matrix[:, -1]
    scalar = matrix[2, 3]
    reversed_matrix = matrix[::-1, ::-2]
    gathered = matrix[[3, 1, 3], 1:5:2]
    cartesian = matrix[[3, 1], :][:, [4, 0]]

    assert isinstance(row, DBVector)
    assert isinstance(column, DBVector)
    assert isinstance(scalar, DBScalar)
    assert row.shape == (5,)
    assert column.shape == (4,)
    assert scalar.shape == ()

    np.testing.assert_array_equal(row.to_numpy(), reference[1, :])
    np.testing.assert_array_equal(column.to_numpy(), reference[:, -1])
    assert scalar.item() == reference[2, 3]
    np.testing.assert_array_equal(
        reversed_matrix.to_numpy(), reference[::-1, ::-2]
    )
    np.testing.assert_array_equal(
        gathered.to_numpy(), reference[[3, 1, 3], 1:5:2]
    )
    np.testing.assert_array_equal(
        cartesian.to_numpy(), reference[[3, 1], :][:, [4, 0]]
    )

    # Construction and compilation are lazy; only collection above executes.
    print("vector semantic plan rank:", row.plan()["result"]["rank"])
    print("gather SQL contains join:", "JOIN" in gathered.compile().upper())
