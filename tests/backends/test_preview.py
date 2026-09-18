from unittest.mock import patch

import numpy as np
import pytest

from dbnumpy import DensificationError


@pytest.mark.parametrize(
    "shape", [(0, 6), (6, 0), (1, 1), (4, 4), (5, 5), (3, 12), (12, 3), (12, 12)]
)
def test_preview_matches_numpy(backend, capsys, shape):
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    x = backend.from_numpy(values)
    x.show(edgeitems=2)
    assert capsys.readouterr().out == (
        repr(x) + "\n" + np.array2string(values, edgeitems=2, threshold=0) + "\n"
    )


def test_preview_expressions(backend, capsys):
    values = np.arange(144, dtype=float).reshape(12, 12)
    x = backend.from_numpy(values)
    result = np.sqrt(x + np.arange(12)).T
    expected = np.sqrt(values + np.arange(12)).T
    result.show(edgeitems=2)
    assert capsys.readouterr().out.split("\n", 1)[1] == (
        np.array2string(expected, edgeitems=2, threshold=0) + "\n"
    )


def test_huge_sparse_preview_is_bounded(backend, capsys):
    size = 10**9
    x = backend.from_coo(
        [0, size - 1, 500], [0, size - 1, 500], [7.0, 9.0, 11.0], shape=(size, size)
    )
    original = x._expr
    with (
        patch.object(backend, "collect_matrix", side_effect=AssertionError),
        patch.object(backend, "_execute_sql", wraps=backend._execute_sql) as execute,
    ):
        preview = backend._collect_preview(x._expr, edgeitems=2)
        assert execute.call_count == 1
        sql = execute.call_args.args[0]
        assert "WHERE" in sql and "LIMIT 17" in sql
        assert '"i" >= 999999998' in sql
        assert '"j" >= 999999998' in sql
        expected = np.zeros((4, 4))
        expected[0, 0], expected[-1, -1] = 7, 9
        np.testing.assert_array_equal(preview, expected)
        x.show(edgeitems=2)
    assert x._expr is original
    assert "..." in capsys.readouterr().out


@pytest.mark.parametrize(
    "edgeitems,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_invalid_preview_size(backend, edgeitems, error):
    x = backend.from_numpy(np.ones((2, 2)))
    with pytest.raises(error):
        x.show(edgeitems=edgeitems)


def test_repr_does_not_execute(backend, capsys):
    x = backend.from_numpy(np.ones((2, 2)))
    with patch.object(backend, "_execute_sql", side_effect=AssertionError):
        print(x)
    assert "lazy=True" in capsys.readouterr().out


def test_special_values(backend):
    values = np.array([[np.nan, np.inf], [-np.inf, -0.0]])
    x = backend.from_numpy(values)
    actual = backend._collect_preview(x._expr, edgeitems=np.int64(2))
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(values))


def test_preview_limits_and_closed_backend(backend):
    x = backend.from_numpy(np.ones((12, 12)))
    backend.max_host_values = 15
    with pytest.raises(DensificationError, match="max_host_values"):
        x.show(edgeitems=2)
    backend.max_host_values = 1000
    backend.max_densify_cells = 15
    with pytest.raises(DensificationError, match="show"):
        x.show(edgeitems=2)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        x.show()


def test_empty_preview_skips_query(backend, capsys):
    x = backend.from_coo([], [], [], shape=(0, 10**9))
    with patch.object(backend, "_execute_sql", side_effect=AssertionError):
        x.show(edgeitems=10**9)
    assert capsys.readouterr().out.endswith("[]\n")


def test_preview_repeated_gathers_and_product(backend):
    values = np.arange(144, dtype=float).reshape(12, 12)
    x = backend.from_numpy(values)
    indices = [11, 0, 11, 2, 5, 3]
    result = x[indices, :] @ x.T
    expected = (values[indices, :] @ values.T)[np.ix_([0, 1, 4, 5], [0, 1, 10, 11])]
    actual = backend._collect_preview(result._expr, edgeitems=2)
    np.testing.assert_allclose(actual, expected)
