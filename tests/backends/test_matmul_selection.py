import numpy as np


def test_matmul_after_repeated_row_selection(backend):
    values = np.arange(144, dtype=float).reshape(12, 12)
    x = backend.from_numpy(values)
    indices = [11, 0, 11, 2, 5, 3]
    result = x[indices, :] @ x.T
    np.testing.assert_allclose(result.to_numpy(), values[indices, :] @ values.T)
