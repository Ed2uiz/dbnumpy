"""DuckDB execution adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from dbnumpy.backends.base import Backend
from dbnumpy.io import MatrixMarketHeader, read_matrix_market_header

if TYPE_CHECKING:
    from dbnumpy.matrix import DBArray


class DuckDBBackend(Backend):
    dialect = "duckdb"
    native_matrix_market_ingestion = True

    def __init__(
        self,
        connection: Any,
        *,
        max_densify_cells: int = 5_000_000,
        max_host_values: int = 50_000_000,
        max_sparse_host_values: int = 5_000_000,
        max_selector_values: int = 5_000_000,
        max_selector_relations: int = 256,
    ) -> None:
        super().__init__(
            max_densify_cells=max_densify_cells,
            max_host_values=max_host_values,
            max_sparse_host_values=max_sparse_host_values,
            max_selector_values=max_selector_values,
            max_selector_relations=max_selector_relations,
        )
        self.connection = connection
        self._arrow_objects: dict[str, pa.Table] = {}

    @classmethod
    def connect(
        cls,
        database: str | Path = ":memory:",
        *,
        memory_limit: str = "2GB",
        threads: int = 2,
        temp_directory: str | Path | None = None,
        max_densify_cells: int = 5_000_000,
        max_host_values: int = 50_000_000,
        max_sparse_host_values: int = 5_000_000,
        max_selector_values: int = 5_000_000,
        max_selector_relations: int = 256,
    ) -> DuckDBBackend:
        if threads < 1:
            raise ValueError("threads must be positive")
        if max_densify_cells < 0:
            raise ValueError("max_densify_cells must be nonnegative")
        if max_sparse_host_values < 0:
            raise ValueError("max_sparse_host_values must be nonnegative")
        if max_host_values < 0:
            raise ValueError("max_host_values must be nonnegative")
        if max_selector_values < 0:
            raise ValueError("max_selector_values must be nonnegative")
        if max_selector_relations < 0:
            raise ValueError("max_selector_relations must be nonnegative")
        resolved_temp_directory: Path | None = None
        if temp_directory is not None:
            resolved_temp_directory = Path(temp_directory).expanduser().resolve()
            if not resolved_temp_directory.is_dir():
                raise ValueError(
                    "temp_directory must name an existing directory; got "
                    f"{resolved_temp_directory}"
                )
            if not os.access(resolved_temp_directory, os.W_OK | os.X_OK):
                raise ValueError(
                    f"temp_directory is not writable: {resolved_temp_directory}"
                )
        import duckdb

        config: dict[str, Any] = {"memory_limit": memory_limit, "threads": threads}
        if resolved_temp_directory is not None:
            config["temp_directory"] = str(resolved_temp_directory)
        connection = duckdb.connect(
            str(database),
            config=config,
        )
        return cls(
            connection,
            max_densify_cells=max_densify_cells,
            max_host_values=max_host_values,
            max_sparse_host_values=max_sparse_host_values,
            max_selector_values=max_selector_values,
            max_selector_relations=max_selector_relations,
        )

    def from_mtx(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        temporary: bool = False,
        overwrite: bool = False,
        assume_canonical: bool = False,
    ) -> DBArray[np.float64]:
        """Ingest Matrix Market coordinates directly into a DuckDB-owned table.

        Python reads only the header. By default DuckDB validates and canonicalizes
        the body while creating the destination: coordinates become zero-based
        int64 values, duplicates are summed, symmetry is expanded, and resulting
        zeros are removed.

        ``assume_canonical=True`` is a scalable fast path for trusted ``general``
        coordinate files whose positions are already unique. It streams rows
        directly into the destination without duplicate reduction or sorting.
        Supplying duplicate coordinates in this mode intentionally preserves them.
        """

        self._check_open()
        source = Path(path).expanduser().resolve()
        header = read_matrix_market_header(source)
        if assume_canonical and header.symmetry != "general":
            raise ValueError(
                "assume_canonical=True only supports general Matrix Market input"
            )
        relation = self._mtx_relation_name(name)
        if relation in self._arrow_objects:
            raise ValueError(
                f"cannot overwrite copied input or selector relation {relation!r}; "
                "choose a new destination name"
            )

        relation_kind = self._relation_kind(relation)
        if relation_kind is not None and not overwrite:
            raise ValueError(
                f"relation {relation!r} already exists; pass overwrite=True "
                "to replace it"
            )

        transaction_started = False
        try:
            self.connection.execute("BEGIN TRANSACTION")
            transaction_started = True
            if relation_kind is not None:
                self.connection.execute(
                    f'DROP {relation_kind} "{relation}"'
                )
            persistence = "TEMP " if temporary else ""
            load_sql = (
                self._direct_mtx_sql(header)
                if assume_canonical
                else self._canonical_mtx_sql(header)
            )
            self.connection.execute(
                f'CREATE {persistence}TABLE "{relation}" AS {load_sql}',
                [str(source)],
            )
            if assume_canonical:
                self._validate_direct_mtx(relation, header)
            elif header.entries > 0 and self._relation_is_empty(relation):
                # A nonempty input may legitimately canonicalize to an empty
                # matrix. Only this exceptional result needs a second source
                # count to distinguish cancellation from a truncated body.
                self._validate_empty_safe_mtx(source, header)
            self.connection.execute("COMMIT")
        except Exception as exc:
            if transaction_started:
                try:
                    self.connection.execute("ROLLBACK")
                except Exception:
                    pass
            if isinstance(exc, (ValueError, OverflowError)):
                raise
            raise ValueError(f"invalid Matrix Market data in {source}: {exc}") from exc

        # Replacement is transactional. Existing matrix objects with this source
        # name see the replacement after commit, matching DuckDB relation semantics.
        self._arrow_objects.pop(relation, None)
        self._owned_relations.discard(relation)
        return self.from_relation(
            relation, shape=header.shape, storage="sparse"
        )

    def _mtx_relation_name(self, requested: str | None) -> str:
        if requested is not None:
            self._validate_name(requested)
            return requested
        generated = self._relation_name(None)
        self._owned_relations.discard(generated)
        return generated

    def _relation_kind(self, name: str) -> str | None:
        table = self.connection.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = ? LIMIT 1", [name]
        ).fetchone()
        if table is not None:
            return "TABLE"
        view = self.connection.execute(
            "SELECT 1 FROM duckdb_views() WHERE view_name = ? LIMIT 1", [name]
        ).fetchone()
        return "VIEW" if view is not None else None

    @staticmethod
    def _raw_mtx_sql(header: MatrixMarketHeader) -> str:
        value_sql = (
            "1.0"
            if header.field == "pattern"
            else "try_cast(parts[3] AS DOUBLE)"
        )
        if header.field == "integer":
            value_sql = (
                "CASE WHEN regexp_full_match(parts[3], '[+-]?[0-9]+') "
                "THEN try_cast(parts[3] AS DOUBLE) END"
            )
        # A delimiter forbidden by the Matrix Market grammar makes each physical
        # data row one VARCHAR. `comment` skips both the banner and `%` comments;
        # skip=1 then skips the first non-comment row, i.e. the size declaration.
        return f'''
            SELECT
                CASE WHEN regexp_full_match(parts[1], '[+-]?[0-9]+')
                     THEN try_cast(parts[1] AS BIGINT) END AS raw_i,
                CASE WHEN regexp_full_match(parts[2], '[+-]?[0-9]+')
                     THEN try_cast(parts[2] AS BIGINT) END AS raw_j,
                {value_sql} AS raw_x,
                len(parts) AS field_count
            FROM (
                SELECT line, regexp_split_to_array(trim(line), '\\s+') AS parts
                FROM read_csv(
                    ?, delim='|', header=false, auto_detect=false,
                    columns={{'line': 'VARCHAR'}}, comment='%', skip=1
                )
                WHERE trim(line) <> ''
            )
        '''

    def _validate_direct_mtx(
        self, relation: str, header: MatrixMarketHeader
    ) -> None:
        actual_entries = self.connection.execute(
            f'SELECT count(*) FROM "{relation}"'
        ).fetchone()[0]
        if actual_entries != header.entries:
            raise ValueError(
                "Matrix Market body row count does not match its header: "
                f"expected {header.entries}, found {actual_entries}"
            )
        invalid = self.connection.execute(
            f'''
            SELECT count(*) FROM "{relation}"
            WHERE i IS NULL OR j IS NULL OR x IS NULL
               OR i < 0 OR i >= ? OR j < 0 OR j >= ?
            ''',
            [header.rows, header.columns],
        ).fetchone()[0]
        if invalid:
            raise ValueError(
                f"Matrix Market body contains {invalid} malformed or "
                "out-of-bounds row(s)"
            )

    def _relation_is_empty(self, relation: str) -> bool:
        return (
            self.connection.execute(
                f'SELECT NOT EXISTS (SELECT 1 FROM "{relation}" LIMIT 1)'
            ).fetchone()[0]
            is True
        )

    def _validate_empty_safe_mtx(
        self, source: Path, header: MatrixMarketHeader
    ) -> None:
        raw = self._raw_mtx_sql(header)
        actual_entries = self.connection.execute(
            f"SELECT count(*) FROM ({raw})", [str(source)]
        ).fetchone()[0]
        if actual_entries != header.entries:
            raise ValueError(
                "Matrix Market body row count does not match its header: "
                f"expected {header.entries}, found {actual_entries}"
            )

    @classmethod
    def _canonical_mtx_sql(cls, header: MatrixMarketHeader) -> str:
        raw = cls._raw_mtx_sql(header)
        expected_fields = 2 if header.field == "pattern" else 3
        count_error = (
            "'Matrix Market body row count does not match its header: expected "
            f"{header.entries}, found ' || cast(actual_entries AS VARCHAR)"
        )
        skew_check = ""
        if header.symmetry == "skew-symmetric":
            skew_check = (
                "WHEN raw_i = raw_j AND raw_x <> 0.0 THEN "
                "error('skew-symmetric Matrix Market diagonal entries must be zero') "
            )
        original = "SELECT raw_i - 1 AS i, raw_j - 1 AS j, raw_x AS x FROM checked"
        if header.symmetry == "general":
            expanded = original
        else:
            mirror_x = (
                "-raw_x" if header.symmetry == "skew-symmetric" else "raw_x"
            )
            expanded = f'''
                SELECT
                    CASE WHEN mirror = 0 THEN raw_i - 1 ELSE raw_j - 1 END AS i,
                    CASE WHEN mirror = 0 THEN raw_j - 1 ELSE raw_i - 1 END AS j,
                    CASE WHEN mirror = 0 THEN raw_x ELSE {mirror_x} END AS x
                FROM checked
                CROSS JOIN LATERAL range(
                    CASE WHEN raw_i = raw_j THEN 1 ELSE 2 END
                ) AS copies(mirror)
            '''
        return f'''
            WITH raw AS (
                SELECT *, count(*) OVER () AS actual_entries
                FROM ({raw})
            ),
            checked AS (
                SELECT
                    cast(CASE
                        WHEN actual_entries <> {header.entries}
                            THEN error({count_error})
                        WHEN field_count <> {expected_fields}
                          OR raw_i IS NULL OR raw_j IS NULL OR raw_x IS NULL
                            THEN error('Matrix Market body contains malformed row(s)')
                        WHEN raw_i < 1 OR raw_i > {header.rows}
                          OR raw_j < 1 OR raw_j > {header.columns}
                            THEN error(
                                'Matrix Market body contains out-of-bounds row(s)'
                            )
                        {skew_check}
                        ELSE raw_i END AS BIGINT) AS raw_i,
                    raw_j, raw_x
                FROM raw
            )
            SELECT cast(i AS BIGINT) AS i, cast(j AS BIGINT) AS j,
                   cast(sum(x) AS DOUBLE) AS x
            FROM ({expanded})
            GROUP BY i, j
            HAVING sum(x) <> 0.0
        '''

    @staticmethod
    def _direct_mtx_sql(header: MatrixMarketHeader) -> str:
        columns = (
            "{'raw_i': 'VARCHAR', 'raw_j': 'VARCHAR'}"
            if header.field == "pattern"
            else "{'raw_i': 'VARCHAR', 'raw_j': 'VARCHAR', 'raw_x': 'VARCHAR'}"
        )
        value_sql = (
            "1.0" if header.field == "pattern" else "try_cast(raw_x AS DOUBLE)"
        )
        if header.field == "integer":
            value_sql = (
                "CASE WHEN regexp_full_match(raw_x, '[+-]?[0-9]+') "
                "THEN try_cast(raw_x AS DOUBLE) END"
            )
        return f'''
            SELECT
                CASE WHEN regexp_full_match(raw_i, '[+-]?[0-9]+')
                     THEN try_cast(raw_i AS BIGINT) - 1 END AS i,
                CASE WHEN regexp_full_match(raw_j, '[+-]?[0-9]+')
                     THEN try_cast(raw_j AS BIGINT) - 1 END AS j,
                cast({value_sql} AS DOUBLE) AS x
            FROM read_csv(
                ?, delim=' ', header=false, auto_detect=false,
                columns={columns}, comment='%', skip=1,
                ignore_errors=false
            )
        '''

    def _register_arrow(self, name: str, table: pa.Table) -> None:
        self._validate_name(name)
        self.connection.register(name, table)
        self._arrow_objects[name] = table
        self._record_created(name)

    def _unregister_relation(self, name: str) -> None:
        if name in self._arrow_objects:
            self.connection.unregister(name)
            self._arrow_objects.pop(name, None)
        else:
            kind = self._relation_kind(name)
            if kind is not None:
                self.connection.execute(f'DROP {kind} "{name}"')

    def _register_dimension(self, name: str, *, column: str, size: int) -> None:
        self._validate_name(name)
        self._validate_name(column)
        self.connection.execute(
            f'CREATE OR REPLACE TEMP VIEW "{name}" AS '
            f'SELECT range AS "{column}" FROM range({size})'
        )

        self._record_created(name)

    def _execute_sql(self, sql: str) -> pa.Table:
        return self.connection.execute(sql).to_arrow_table()

    def _explain_sql(self, sql: str) -> str:
        rows = self.connection.execute(f"EXPLAIN {sql}").fetchall()
        return "\n".join(str(row[-1]) for row in rows)

    def _materialize_sql(self, name: str, sql: str) -> None:
        self._validate_name(name)
        self.connection.execute(f'CREATE TEMP TABLE "{name}" AS {sql}')

        self._record_created(name)

    def _close_impl(self) -> None:
        self.connection.close()
        self._arrow_objects.clear()
