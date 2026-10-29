import numpy as np
import pytest

from dbnumpy import PolarsBackend
from dbnumpy.lowering.polars_sql import to_polars_sql


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_zero_scalars_clear_negative_zero(engine: str) -> None:
    # Polars folds x + 0.0 to x, which keeps -0.0; NumPy gives 0.0.
    values = np.array([[-0.0, 0.0, 1.0]])
    with PolarsBackend.connect(engine=engine) as backend:
        x = backend.from_numpy(values)
        for actual, expected in (
            (x + 0.0, values + 0.0),
            (0.0 + x, 0.0 + values),
            (x - -0.0, values - -0.0),
            (0.0 - x, 0.0 - values),
        ):
            np.testing.assert_array_equal(
                np.signbit(actual.to_numpy()), np.signbit(expected)
            )


def test_ibis_negative_zero_is_recognized() -> None:
    sql = to_polars_sql("SELECT x - (-(1.0) / CAST('inf' AS DOUBLE)) AS x FROM t")
    assert "CASE WHEN" in sql
