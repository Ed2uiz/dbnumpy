from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datafusion import DataFrame, SessionContext

from dbnumpy.backends import DataFusionBackend


@pytest.mark.parametrize("storage", ["dense", "sparse"])
@pytest.mark.parametrize("borrowed_context", [False, True])
@pytest.mark.parametrize("row_group_size", [1, 2])
@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
def test_external_parquet_nan_statistics_preserve_values(
    tmp_path, storage, borrowed_context, row_group_size, dtype
):
    path = tmp_path / "nan and finite.parquet"
    table = pa.table(
        {"i": [0, 0], "j": [0, 1], "x": pa.array([np.nan, 2.0], type=dtype)}
    )
    pq.write_table(table, path, row_group_size=row_group_size, write_page_index=True)
    original = path.read_bytes()
    context = SessionContext() if borrowed_context else None
    backend = DataFusionBackend(context) if context else DataFusionBackend.connect()
    with backend:
        # Import must only build a scan, never load or rewrite the input values.
        with (
            patch.object(DataFrame, "collect", side_effect=AssertionError("collect")),
            patch.object(
                DataFrame, "to_arrow_table", side_effect=AssertionError("read")
            ),
            patch.object(pq, "read_table", side_effect=AssertionError("read")),
        ):
            matrix = backend.from_parquet(path, shape=(1, 2), storage=storage)
        expected = np.array([[np.nan, 2.0]])
        np.testing.assert_array_equal(matrix.to_numpy(), expected)
        np.testing.assert_array_equal(np.isnan(matrix).to_numpy(), np.isnan(expected))
        # Pointwise execution and SQL execution must both preserve the NaN.
        np.testing.assert_array_equal((matrix * 3).to_numpy(), expected * 3)
        np.testing.assert_array_equal((matrix.T * 3).to_numpy(), expected.T * 3)
        np.testing.assert_array_equal(
            matrix[:, [1, 0, 1]].to_numpy(), expected[:, [1, 0, 1]]
        )
        assert np.isnan(matrix.sum())
        np.testing.assert_array_equal(matrix.mean(axis=0), expected.mean(axis=0))
        assert np.nansum(matrix) == 2.0
        np.testing.assert_array_equal(matrix.compute().to_numpy(), expected)
    assert path.read_bytes() == original
    if context is not None:
        # Import must not change settings on a context owned by the caller.
        setting = context.sql("SHOW datafusion.execution.collect_statistics")
        assert setting.to_arrow_table().column(1).to_pylist() == ["true"]


@pytest.mark.parametrize("split_files", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [np.nan, np.nan],
        [-0.0, 0.0],
        [-0.0, -0.0],
        [np.inf, np.nan],
        [-np.inf, 2.0],
        [None, 2.0],
        [None, None],
        [np.nextafter(0.0, 1.0), 0.0],
    ],
)
def test_external_parquet_special_values(tmp_path, split_files, values):
    table = pa.table({"i": [0, 0], "j": [0, 1], "x": pa.array(values, pa.float64())})
    tables = [table.slice(0, 1), table.slice(1, 1)] if split_files else [table]
    paths = []
    for index, part in enumerate(tables):
        path = tmp_path / f"part-{index}.parquet"
        pq.write_table(part, path, row_group_size=1, write_page_index=True)
        paths.append(path)
    expected = np.array([values], dtype=np.float64)
    with DataFusionBackend.connect() as backend:
        matrix = backend.from_parquet(paths, shape=(1, 2), storage="dense")
        for result in (matrix, matrix.T.T, matrix.compute()):
            actual = result.to_numpy()
            np.testing.assert_array_equal(actual, expected)
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(
                np.signbit(actual[finite]), np.signbit(expected[finite])
            )
            np.testing.assert_array_equal(
                np.isnan(result).to_numpy(), np.isnan(expected)
            )


@pytest.mark.parametrize("seed", range(6))
def test_external_parquet_shuffled_sparse_workflow(tmp_path, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(8, 7))
    values[rng.random(values.shape) < 0.65] = 0.0
    values[0, :] = 0.0
    values[:, 0] = 0.0
    values[2, 3] = np.nan
    i, j = np.nonzero(values)
    order = rng.permutation(len(i))
    table = pa.table({"i": i[order], "j": j[order], "x": values[i[order], j[order]]})
    paths = []
    for index, indices in enumerate(np.array_split(np.arange(len(i)), 3)):
        path = tmp_path / f"part-{index}.parquet"
        pq.write_table(table.take(indices), path, row_group_size=2)
        paths.append(path)
    with DataFusionBackend.connect(target_partitions=1 + seed % 2) as backend:
        matrix = backend.from_parquet(paths, shape=values.shape)
        result = np.log1p(abs(matrix)) * np.arange(1, 8)
        expected = np.log1p(abs(values)) * np.arange(1, 8)
        for expression, reference in (
            (matrix, values),
            (result, expected),
            (result[[7, 2, 0, 2], :].T, expected[[7, 2, 0, 2], :].T),
            (result.compute(), expected),
        ):
            np.testing.assert_allclose(
                expression.to_numpy(), reference, rtol=1e-12, atol=1e-12
            )
            for axis in (None, 0, 1):
                for operation in (
                    np.sum,
                    np.mean,
                    np.var,
                    np.nansum,
                    np.nanmean,
                    np.nanvar,
                ):
                    np.testing.assert_allclose(
                        operation(expression, axis=axis),
                        operation(reference, axis=axis),
                        rtol=1e-12,
                        atol=1e-12,
                    )


def _write_coordinates(
    path: Path,
    *,
    i: list[int],
    j: list[int],
    x: list[float],
) -> None:
    pq.write_table(pa.table({"i": i, "j": j, "x": x}), path)


@pytest.mark.backend
def test_parquet_files_remain_lazy_and_include_structural_zeros(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_coordinates(first, i=[0, 2], j=[0, 1], x=[3.0, 6.0])
    _write_coordinates(second, i=[1], j=[2], x=[9.0])
    backend = DataFusionBackend.connect(memory_limit_bytes=64 * 1024**2)
    try:
        with patch.object(
            backend, "_register_arrow", side_effect=AssertionError("Arrow upload used")
        ):
            matrix = backend.from_parquet(
                [first, second], shape=(3, 3), name="external_coordinates"
            )
            np.testing.assert_allclose(matrix.mean(axis=0), [1.0, 2.0, 3.0])
            np.testing.assert_allclose(matrix.mean(axis=1), [1.0, 3.0, 2.0])

        plan = matrix.explain().lower()
        assert "parquet" in plan
        assert str(first) in plan or first.name in plan
        assert str(second) in plan or second.name in plan
    finally:
        backend.close()


@pytest.mark.backend
def test_parquet_schema_is_validated_before_registration(tmp_path: Path) -> None:
    missing = tmp_path / "missing_column.parquet"
    strings = tmp_path / "strings.parquet"
    pq.write_table(pa.table({"i": [0], "j": [0]}), missing)
    pq.write_table(pa.table({"i": ["0"], "j": [0], "x": [1.0]}), strings)
    backend = DataFusionBackend.connect()
    try:
        with pytest.raises(ValueError, match="numeric i, j, x"):
            backend.from_parquet(missing, shape=(1, 1))
        with pytest.raises(ValueError, match="integer i/j and numeric x"):
            backend.from_parquet(strings, shape=(1, 1))
    finally:
        backend.close()


@pytest.mark.backend
def test_parquet_paths_and_relation_names_are_validated(tmp_path: Path) -> None:
    coordinates = tmp_path / "coordinates.parquet"
    _write_coordinates(coordinates, i=[0], j=[0], x=[1.0])
    backend = DataFusionBackend.connect()
    try:
        with pytest.raises(ValueError, match="at least one"):
            backend.from_parquet([], shape=(1, 1))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            backend.from_parquet(tmp_path / "absent.parquet", shape=(1, 1))
        with pytest.raises(ValueError, match="regular file"):
            backend.from_parquet(tmp_path, shape=(1, 1))
        with pytest.raises(ValueError, match="must be unique"):
            backend.from_parquet([coordinates, coordinates], shape=(1, 1))
        with pytest.raises(ValueError, match="relation names"):
            backend.from_parquet(coordinates, shape=(1, 1), name="unsafe-name")
    finally:
        backend.close()


@pytest.mark.backend
def test_parquet_numeric_columns_are_normalized(tmp_path: Path) -> None:
    coordinates = tmp_path / "narrow.parquet"
    pq.write_table(
        pa.table(
            {
                "i": pa.array([0, 1], type=pa.int32()),
                "j": pa.array([1, 0], type=pa.uint32()),
                "x": pa.array([2, 4], type=pa.int16()),
            }
        ),
        coordinates,
    )
    backend = DataFusionBackend.connect()
    try:
        matrix = backend.from_parquet(coordinates, shape=(2, 2), storage="sparse")
        np.testing.assert_allclose(matrix.to_numpy(), [[0.0, 2.0], [4.0, 0.0]])
        assert matrix.dtype == np.dtype(np.float64)
    finally:
        backend.close()
