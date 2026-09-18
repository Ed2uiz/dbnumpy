from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from dbnumpy import DuckDBBackend


class _TracingConnection:
    def __init__(self, connection: object) -> None:
        self.connection = connection
        self.sql: list[str] = []

    def execute(self, sql: str, *args: object, **kwargs: object) -> object:
        self.sql.append(sql)
        return self.connection.execute(sql, *args, **kwargs)  # type: ignore[attr-defined, no-any-return]

    def __getattr__(self, name: str) -> object:
        return getattr(self.connection, name)


def _write_mtx(path: Path, body: str) -> Path:
    text = "%%MatrixMarket matrix coordinate real general\n" + body
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="ascii") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="ascii")
    return path


@pytest.mark.backend
@pytest.mark.parametrize("suffix", [".mtx", ".mtx.gz"])
def test_native_ingestion_canonicalizes_and_preserves_sparse_means(
    tmp_path: Path, suffix: str
) -> None:
    path = _write_mtx(
        tmp_path / f"values{suffix}",
        "% comments and irregular whitespace are accepted\n"
        "3 4 6\n"
        "1 1 2\n"
        "1   1 3\n"
        "2 2 0\n"
        "3 2 -4\n"
        "3 2 4\n"
        "2 4 9\n",
    )
    backend = DuckDBBackend.connect(max_densify_cells=100)
    matrix = backend.from_mtx(path, name="native_values")

    assert matrix.shape == (3, 4)
    np.testing.assert_array_equal(
        matrix.to_numpy(),
        np.array([[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0], [0, 0, 0, 0]]),
    )
    np.testing.assert_allclose(matrix.mean(axis=0), [5 / 3, 0, 0, 3])
    rows = backend.connection.execute(
        'SELECT i, j, x FROM "native_values" ORDER BY i, j'
    ).fetchall()
    assert rows == [(0, 0, 5.0), (1, 3, 9.0)]
    assert "native_values" not in backend._arrow_objects


@pytest.mark.backend
@pytest.mark.parametrize("assume_canonical", [False, True])
@pytest.mark.parametrize("suffix", [".mtx", ".mtx.gz"])
def test_native_ingestion_uses_one_destination_ctas_and_no_stage_table(
    tmp_path: Path, assume_canonical: bool, suffix: str
) -> None:
    path = _write_mtx(tmp_path / f"values{suffix}", "2 2 2\n1 1 2\n2 2 3\n")
    backend = DuckDBBackend.connect()
    trace = _TracingConnection(backend.connection)
    backend.connection = trace

    backend.from_mtx(
        path, name="values", assume_canonical=assume_canonical
    )

    statements = "\n".join(trace.sql)
    assert "dbm_mtx_stage" not in statements
    destination_ctas = [
        sql
        for sql in trace.sql
        if 'CREATE TABLE "values" AS' in sql
    ]
    assert len(destination_ctas) == 1
    if assume_canonical:
        assert "columns={'raw_i': 'VARCHAR'" in destination_ctas[0]
        assert "regexp_split_to_array" not in destination_ctas[0]


@pytest.mark.backend
def test_assume_canonical_preserves_duplicate_rows_without_reduction(
    tmp_path: Path,
) -> None:
    path = _write_mtx(tmp_path / "duplicates.mtx", "2 2 2\n1 1 2\n1 1 3\n")
    backend = DuckDBBackend.connect()

    backend.from_mtx(path, name="values", assume_canonical=True)

    assert backend.connection.execute(
        'SELECT i, j, x FROM "values" ORDER BY x'
    ).fetchall() == [(0, 0, 2.0), (0, 0, 3.0)]


@pytest.mark.backend
def test_assume_canonical_rejects_symmetric_input(tmp_path: Path) -> None:
    path = tmp_path / "symmetric.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 2 4\n",
        encoding="ascii",
    )
    backend = DuckDBBackend.connect()

    with pytest.raises(ValueError, match="only supports general"):
        backend.from_mtx(path, assume_canonical=True)


@pytest.mark.backend
def test_assume_canonical_validates_declared_row_count(tmp_path: Path) -> None:
    path = _write_mtx(tmp_path / "truncated.mtx", "2 2 2\n1 1 4\n")
    backend = DuckDBBackend.connect()

    with pytest.raises(ValueError, match="row count"):
        backend.from_mtx(path, name="values", assume_canonical=True)
    assert backend._relation_kind("values") is None


@pytest.mark.backend
def test_native_ingestion_supports_an_empty_sparse_body(tmp_path: Path) -> None:
    path = _write_mtx(tmp_path / "empty.mtx", "3 4 0\n")

    with DuckDBBackend.connect(max_densify_cells=100) as backend:
        matrix = backend.from_mtx(path, name="empty_values", temporary=True)
        assert matrix.shape == (3, 4)
        assert matrix.sum() == 0.0
        np.testing.assert_array_equal(matrix.mean(axis=0), np.zeros(4))
        assert backend.connection.execute(
            'SELECT COUNT(*) FROM "empty_values"'
        ).fetchone() == (0,)


@pytest.mark.backend
def test_safe_ingestion_rejects_missing_nonempty_body(tmp_path: Path) -> None:
    path = _write_mtx(tmp_path / "missing.mtx", "2 2 1\n")
    backend = DuckDBBackend.connect()

    with pytest.raises(ValueError, match="row count"):
        backend.from_mtx(path, name="values")
    assert backend._relation_kind("values") is None


@pytest.mark.backend
def test_persistent_ingestion_survives_reopen_and_wrap(tmp_path: Path) -> None:
    path = _write_mtx(tmp_path / "values.mtx", "2 2 1\n2 1 6\n")
    database = tmp_path / "matrix.duckdb"

    first = DuckDBBackend.connect(database)
    first.from_mtx(path, name="persisted")
    first.close()

    second = DuckDBBackend.connect(database)
    matrix = second.from_relation("persisted", shape=(2, 2), storage="sparse")
    np.testing.assert_array_equal(matrix.to_numpy(), [[0, 0], [6, 0]])
    second.close()


@pytest.mark.backend
def test_temporary_ingestion_does_not_survive_reopen(tmp_path: Path) -> None:
    path = _write_mtx(tmp_path / "values.mtx", "1 1 1\n1 1 2\n")
    database = tmp_path / "matrix.duckdb"
    first = DuckDBBackend.connect(database)
    first.from_mtx(path, name="temporary_values", temporary=True)
    first.close()

    second = DuckDBBackend.connect(database)
    with pytest.raises(ValueError, match="must exist"):
        second.from_relation("temporary_values", shape=(1, 1), storage="sparse")


@pytest.mark.backend
def test_overwrite_is_explicit_and_transactional(tmp_path: Path) -> None:
    first_path = _write_mtx(tmp_path / "first.mtx", "1 1 1\n1 1 2\n")
    second_path = _write_mtx(tmp_path / "second.mtx", "1 1 1\n1 1 7\n")
    invalid_path = _write_mtx(tmp_path / "invalid.mtx", "1 1 1\n2 1 9\n")
    backend = DuckDBBackend.connect()
    backend.from_mtx(first_path, name="values")

    with pytest.raises(ValueError, match="already exists"):
        backend.from_mtx(second_path, name="values")
    with pytest.raises(ValueError, match="out-of-bounds"):
        backend.from_mtx(invalid_path, name="values", overwrite=True)
    assert backend.connection.execute('SELECT x FROM "values"').fetchone() == (2.0,)

    replaced = backend.from_mtx(second_path, name="values", overwrite=True)
    np.testing.assert_array_equal(replaced.to_numpy(), [[7]])


@pytest.mark.backend
def test_assume_canonical_overwrite_is_transactional(tmp_path: Path) -> None:
    first_path = _write_mtx(tmp_path / "first.mtx", "1 1 1\n1 1 2\n")
    invalid_path = _write_mtx(tmp_path / "invalid.mtx", "1 1 1\n2 1 9\n")
    backend = DuckDBBackend.connect()
    backend.from_mtx(first_path, name="values", assume_canonical=True)

    with pytest.raises(ValueError, match="out-of-bounds"):
        backend.from_mtx(
            invalid_path,
            name="values",
            overwrite=True,
            assume_canonical=True,
        )

    assert backend.connection.execute('SELECT x FROM "values"').fetchone() == (2.0,)


@pytest.mark.backend
def test_overwrite_preserves_immutable_copied_inputs(tmp_path: Path) -> None:
    path = _write_mtx(tmp_path / "replacement.mtx", "1 1 1\n1 1 7\n")
    backend = DuckDBBackend.connect()
    matrix = backend.from_coo([0], [0], [2.0], shape=(1, 1), name="values")
    assert "values" in backend._arrow_objects

    with pytest.raises(ValueError, match="cannot overwrite copied input"):
        backend.from_mtx(path, name="values", overwrite=True)

    assert "values" in backend._arrow_objects
    np.testing.assert_array_equal(matrix.to_numpy(), [[2]])


@pytest.mark.backend
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("2 2 2\n1 1 1\n", "row count"),
        ("2 2 1\n3 1 1\n", "out-of-bounds"),
        ("2 2 1\n1 1 nope\n", "malformed"),
        ("2 2 1\n1 1 2 extra\n", "malformed"),
    ],
)
def test_rejects_malformed_bodies(tmp_path: Path, body: str, message: str) -> None:
    path = _write_mtx(tmp_path / "invalid.mtx", body)
    backend = DuckDBBackend.connect()
    with pytest.raises(ValueError, match=message):
        backend.from_mtx(path, name="invalid_values")
    assert backend._relation_kind("invalid_values") is None


@pytest.mark.backend
def test_symmetric_and_pattern_inputs(tmp_path: Path) -> None:
    path = tmp_path / "pattern.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "3 3 2\n"
        "1 1\n"
        "3 1\n",
        encoding="ascii",
    )
    backend = DuckDBBackend.connect(max_densify_cells=100)
    matrix = backend.from_mtx(path)
    np.testing.assert_array_equal(
        matrix.to_numpy(), [[1, 0, 1], [0, 0, 0], [1, 0, 0]]
    )


@pytest.mark.backend
def test_temp_directory_validation_and_configuration(tmp_path: Path) -> None:
    spill = tmp_path / "spill"
    spill.mkdir()
    backend = DuckDBBackend.connect(temp_directory=spill)
    configured = backend.connection.execute(
        "SELECT current_setting('temp_directory')"
    ).fetchone()[0]
    assert Path(configured) == spill
    backend.close()

    with pytest.raises(ValueError, match="existing directory"):
        DuckDBBackend.connect(temp_directory=tmp_path / "missing")
