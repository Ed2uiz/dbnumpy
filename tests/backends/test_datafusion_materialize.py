from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datafusion import DataFrame, SessionContext
from scipy import sparse

from dbnumpy import DataFusionBackend
from dbnumpy.ir import Source, single_pointwise_source


@pytest.mark.parametrize("storage", ["dense", "sparse"])
@pytest.mark.parametrize("path", ["native", "sql"])
def test_compute_streams_to_files_and_remains_composable(tmp_path, storage, path):
    values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        matrix = (
            backend.from_numpy(values)
            if storage == "dense"
            else backend.from_scipy(sparse.csr_array(values))
        )
        expression = matrix * 2 if path == "native" else matrix.T * 2
        assert (single_pointwise_source(expression._expr) is not None) == (
            path == "native"
        )
        expected = values * 2 if path == "native" else values.T * 2
        with (
            patch.object(DataFrame, "cache", side_effect=AssertionError("cache")),
            patch.object(DataFrame, "collect", side_effect=AssertionError("collect")),
            patch.object(
                DataFrame, "to_arrow_table", side_effect=AssertionError("collect")
            ),
        ):
            computed = expression.compute()
        assert isinstance(computed._expr, Source)
        assert computed.storage == matrix.storage
        directories = [Path(d.name) for d in backend._materialized_files.values()]
        assert len(directories) == 1
        assert list(directories[0].glob("*.parquet"))
        assert "PARQUET" in computed.explain().upper()
        np.testing.assert_array_equal(computed.to_numpy(), expected)
        np.testing.assert_allclose(computed.mean(axis=0), expected.mean(axis=0))
        np.testing.assert_array_equal(
            computed[[1, 0, 1], :].T.to_numpy(), expected[[1, 0, 1], :].T
        )
        second = (computed * 3).compute()
        np.testing.assert_array_equal(second.to_numpy(), expected * 3)
        assert len(backend._materialized_files) == 2
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        computed.to_numpy()


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (3, 4)])
@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_empty_and_all_zero_results_preserve_shape(tmp_path, shape, storage):
    values = np.zeros(shape)
    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        matrix = (
            backend.from_numpy(values)
            if storage == "dense"
            else backend.from_scipy(sparse.csr_array(values))
        )
        result = (matrix * 2).compute()
        assert result.shape == shape
        assert result.storage == matrix.storage
        np.testing.assert_array_equal(result.to_numpy(), values)
        np.testing.assert_array_equal(result.to_scipy().toarray(), values)
        assert result.sum() == 0
    assert not list(tmp_path.iterdir())


def test_special_values_survive_parquet_and_later_operations(tmp_path):
    values = np.array([[-0.0, 0.0, np.nan], [np.inf, -np.inf, np.nextafter(0.0, 1)]])
    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        result = backend.from_numpy(values).compute()
        actual = result.to_numpy()
        np.testing.assert_array_equal(actual, values)
        np.testing.assert_array_equal(np.signbit(actual[0, :2]), [True, False])
        np.testing.assert_array_equal(np.isnan(result).to_numpy(), np.isnan(values))
        np.testing.assert_array_equal((result.T * 2).to_numpy(), values.T * 2)


@pytest.mark.parametrize(
    "values", [[[np.nan, 2.0]], [[-0.0, 0.0]], [[np.nan, np.nan]], [[-0.0, -0.0]]]
)
@pytest.mark.parametrize("transpose", [False, True])
def test_value_statistics_do_not_replace_nan_or_signed_zero(
    tmp_path, values, transpose
):
    expected = np.array(values)
    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        matrix = backend.from_numpy(expected)
        if transpose:
            matrix, expected = matrix.T, expected.T
        result = matrix.compute()
        actual = result.to_numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        np.testing.assert_array_equal(np.isnan(result).to_numpy(), np.isnan(expected))


@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
def test_failed_write_removes_partial_files_and_releases_name(tmp_path, error):
    def fail_write(self, path, *args, **kwargs):
        (Path(path) / "partial.parquet").write_bytes(b"partial")
        raise error

    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        matrix = backend.from_numpy(np.eye(2))
        owned = backend._owned_relations.copy()
        existing = set(tmp_path.iterdir())
        with patch.object(DataFrame, "write_parquet", new=fail_write):
            with pytest.raises(type(error)):
                matrix.compute(name="retryable")
        assert set(tmp_path.iterdir()) == existing
        assert backend._owned_relations == owned
        assert not backend._materialized_files
        assert not backend.context.table_exist("retryable")
        assert matrix.compute(name="retryable").sum() == 2


@pytest.mark.parametrize("stage", ["read_parquet", "register_table", "dimension"])
def test_failed_registration_cleans_files_and_preserves_existing_results(
    tmp_path, stage
):
    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
        previous = matrix.compute(name="previous")
        existing = set(tmp_path.iterdir())
        owned = backend._owned_relations.copy()
        target = backend if stage == "dimension" else backend.context
        method = "_register_dimension" if stage == "dimension" else stage
        with patch.object(target, method, side_effect=RuntimeError("injected")):
            with pytest.raises(RuntimeError, match="injected"):
                matrix.T.compute(name="retryable")
        assert set(tmp_path.iterdir()) == existing
        assert backend._owned_relations == owned
        assert not backend.context.table_exist("retryable")
        assert previous.sum() == 15
        assert matrix.T.compute(name="retryable").sum() == 15


def test_close_preserves_borrowed_files_and_caller_directory(tmp_path):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"i": [0, 1], "j": [1, 0], "x": [2.0, 3.0]}), path)
    context = SessionContext()
    backend = DataFusionBackend(context, temp_directory=tmp_path)
    matrix = backend.from_parquet(path, shape=(2, 2))
    result = (matrix * 2).compute(name="stored")
    assert result.sum() == 10
    backend.close()
    backend.close()
    assert set(tmp_path.iterdir()) == {path}
    assert not context.table_exist("stored")
    assert pq.read_table(path).num_rows == 2


def test_compute_does_not_replace_a_borrowed_table(tmp_path):
    context = SessionContext()
    context.register_record_batches("borrowed", [pa.table({"value": [7]}).to_batches()])
    with DataFusionBackend(context, temp_directory=tmp_path) as backend:
        matrix = backend.from_numpy(np.eye(2))
        with pytest.raises(ValueError, match="already exists"):
            matrix.compute(name="borrowed")
        assert not list(tmp_path.iterdir())
        assert (
            context.table("borrowed").to_arrow_table().column("value")[0].as_py() == 7
        )


def test_default_temp_files_are_removed_on_close():
    with DataFusionBackend.connect() as backend:
        backend.from_numpy(np.eye(2)).compute()
        directory = Path(next(iter(backend._materialized_files.values())).name)
        assert directory.is_dir()
    assert not directory.exists()


def test_computed_result_is_independent_of_the_input_file(tmp_path):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"i": [0, 1], "j": [1, 0], "x": [2.0, 3.0]}), path)
    with DataFusionBackend.connect(temp_directory=tmp_path) as backend:
        source = backend.from_parquet(path, shape=(2, 2))
        result = (source * 2).compute()
        path.unlink()
        np.testing.assert_array_equal(result.to_numpy(), [[0, 4], [6, 0]])
        assert (result + result).sum() == 20


def test_null_values_and_a_temp_path_with_spaces(tmp_path):
    directory = tmp_path / "temporary results"
    directory.mkdir()
    context = SessionContext()
    context.register_record_batches(
        "nullable",
        [pa.table({"i": [0, 0], "j": [0, 1], "x": pa.array([None, 2.0])}).to_batches()],
    )
    with DataFusionBackend(context, temp_directory=directory) as backend:
        source = backend.from_relation("nullable", shape=(1, 2), storage="dense")
        result = source.compute()
        np.testing.assert_array_equal(result.to_numpy(), [[np.nan, 2.0]])
        assert result.nanmean() == 2
    assert not list(directory.iterdir())


def test_temp_directory_must_exist_and_be_writable(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        DataFusionBackend.connect(temp_directory=tmp_path / "missing")
    with patch("dbnumpy.backends.datafusion.os.access", return_value=False):
        with pytest.raises(ValueError, match="not writable"):
            DataFusionBackend.connect(temp_directory=tmp_path)
