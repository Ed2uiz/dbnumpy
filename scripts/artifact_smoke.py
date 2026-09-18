"""Exercise an installed dbnumpy artifact against both backend adapters."""

from __future__ import annotations

from importlib import resources
from importlib.metadata import distribution, version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import dbnumpy
from dbnumpy import DataFusionBackend, DuckDBBackend


def main() -> None:
    assert dbnumpy.__version__ == version("dbnumpy")
    assert resources.files("dbnumpy").joinpath("py.typed").is_file()
    package = distribution("dbnumpy")
    assert package.metadata["License-Expression"] == "MIT"
    licenses = [
        path for path in package.files or () if str(path).endswith("/licenses/LICENSE")
    ]
    assert len(licenses) == 1
    assert "MIT License" in package.locate_file(licenses[0]).read_text()
    assert issubclass(dbnumpy.DensificationError, dbnumpy.DBArrayError)

    values = np.array([[0.0, 0.5], [1.0, np.nan]])
    for backend_type in (DuckDBBackend, DataFusionBackend):
        with backend_type.connect() as backend:
            matrix = backend.from_numpy(values)
            assert isinstance(matrix, dbnumpy.DBArray)
            assert isinstance(matrix, dbnumpy.DBDenseArray)
            np.testing.assert_allclose(
                np.arcsin(matrix).to_numpy(),
                np.arcsin(values),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                matrix.nanvar(axis=0),
                np.nanvar(values, axis=0),
            )
            copied = np.array([[1.0]])
            owned = backend.from_numpy(copied)
            copied[:] = 999
            assert owned[0, 0].item() == 1.0

            position = 2**63 - 2
            large = backend.from_coo([position], [0], [7.0], shape=(position + 1, 1))
            assert large[::2, :][position // 2, 0].item() == 7.0

            backend.max_densify_cells = 100
            sparse_matrix = backend.from_coo(
                [0, 10], [0, 10], [2.0, 4.0], shape=(11, 11)
            )
            assert isinstance(sparse_matrix, dbnumpy.DBSparseArray)
            assert (sparse_matrix * np.ones(11)).sum() == 6.0

    # Exercise external file input in the installed artifact, including statistics
    # that must not turn [NaN, 2] into [2, 2] in DataFusion 54.
    with TemporaryDirectory() as directory, DataFusionBackend.connect() as backend:
        path = Path(directory) / "input.parquet"
        pq.write_table(pa.table({"i": [0, 0], "j": [0, 1], "x": [np.nan, 2.0]}), path)
        matrix = backend.from_parquet(path, shape=(1, 2), storage="dense")
        np.testing.assert_array_equal(matrix.to_numpy(), [[np.nan, 2.0]])
        np.testing.assert_array_equal(matrix.compute().to_numpy(), [[np.nan, 2.0]])


if __name__ == "__main__":
    main()
