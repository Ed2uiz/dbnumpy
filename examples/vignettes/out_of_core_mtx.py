"""Engine-native Matrix Market ingestion with a file-backed DuckDB database."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dbnumpy import DuckDBBackend

MTX = """%%MatrixMarket matrix coordinate real general
% four stored values in a 3 x 4 sparse matrix
3 4 4
1 1 3.0
2 3 6.0
3 1 9.0
3 4 12.0
"""


with TemporaryDirectory() as directory:
    root = Path(directory)
    mtx_path = root / "example.mtx"
    database_path = root / "matrix.duckdb"
    spill_path = root / "spill"
    spill_path.mkdir()
    mtx_path.write_text(MTX, encoding="utf-8")

    with DuckDBBackend.connect(
        database=database_path,
        memory_limit="256MB",
        temp_directory=spill_path,
    ) as backend:
        matrix = backend.from_mtx(
            mtx_path,
            name="example_values",
            temporary=False,
            overwrite=False,
        )
        np.testing.assert_allclose(matrix.mean(axis=0), [4.0, 0.0, 2.0, 4.0])

    # A persistent value table can be reopened and wrapped without an MTX
    # reparse or a host SciPy/Arrow upload. Shape remains explicit metadata.
    with DuckDBBackend.connect(
        database=database_path,
        memory_limit="256MB",
        temp_directory=spill_path,
    ) as backend:
        reopened = backend.from_relation(
            "example_values", shape=(3, 4), storage="sparse"
        )
        np.testing.assert_allclose(reopened.mean(axis=0), [4.0, 0.0, 2.0, 4.0])

        print("column means:", reopened.mean(axis=0))
        print("database:", database_path)
