"""Extended lazy math and eager reductions for the operations guide."""

import numpy as np
from scipy import sparse

from dbnumpy import DuckDBBackend

reference = np.array(
    [
        [0.0, np.nan, -2.75, 4.0],
        [0.0, 5.0, 0.0, 16.0],
    ]
)

with DuckDBBackend.connect() as backend:
    matrix = backend.from_scipy(sparse.csr_array(reference))

    np.testing.assert_allclose(
        np.sign(matrix).to_numpy(), np.sign(reference), equal_nan=True
    )
    with np.errstate(all="ignore"):
        np.testing.assert_allclose(
            np.log2(matrix).to_numpy(), np.log2(reference), equal_nan=True
        )
    np.testing.assert_array_equal(
        np.isnan(matrix).to_numpy(), np.isnan(reference).astype(np.float64)
    )
    np.testing.assert_allclose(
        np.arctan(matrix).to_numpy(), np.arctan(reference), equal_nan=True
    )
    with np.errstate(all="ignore"):
        np.testing.assert_allclose(
            matrix.arccos().to_numpy(), np.arccos(reference), equal_nan=True
        )

    np.testing.assert_allclose(
        matrix.nanmean(axis=0, keepdims=True),
        np.nanmean(reference, axis=0, keepdims=True),
    )
    np.testing.assert_allclose(
        np.nanstd(matrix, axis=1, ddof=1),
        np.nanstd(reference, axis=1, ddof=1),
    )
    np.testing.assert_array_equal(matrix.any(axis=0), np.any(reference, axis=0))
    assert np.asarray(matrix.any(axis=0)).dtype == np.dtype(bool)

    tiny = backend.from_numpy(np.array([[-1.0e-200, 1.0e-200]]))
    np.testing.assert_allclose(tiny.std(), 1.0e-200)

    print("NaN-skipping column means:", matrix.nanmean(axis=0))
