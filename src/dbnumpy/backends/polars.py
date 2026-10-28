"""Experimental Polars execution adapter."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dbnumpy.backends.base import Backend
from dbnumpy.backends.datafusion import DataFusionBackend, ParquetPath
from dbnumpy.ir import Source, StorageKind
from dbnumpy.lowering.polars_sql import to_polars_sql

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dbnumpy.matrix import DBArray

_ENGINES = frozenset({"streaming", "in-memory", "auto"})


class PolarsBackend(Backend):
    """Run dbnumpy's DuckDB-dialect SQL through a Polars SQLContext.

    Polars has no query memory limit. The default streaming engine keeps peak
    memory low, but nothing enforces a budget as DuckDB and DataFusion do.
    """

    # The SQL is generated for DuckDB, then rewritten by to_polars_sql().
    dialect = "duckdb"
    external_parquet_scan = True

    def __init__(
        self,
        *,
        engine: str = "streaming",
        temp_directory: str | Path | None = None,
        max_densify_cells: int = 5_000_000,
        max_host_values: int = 50_000_000,
        max_sparse_host_values: int = 5_000_000,
        max_selector_values: int = 5_000_000,
        max_selector_relations: int = 256,
    ) -> None:
        if engine not in _ENGINES:
            raise ValueError(f"engine must be one of {sorted(_ENGINES)}")
        import polars as pl

        super().__init__(
            max_densify_cells=max_densify_cells,
            max_host_values=max_host_values,
            max_sparse_host_values=max_sparse_host_values,
            max_selector_values=max_selector_values,
            max_selector_relations=max_selector_relations,
        )
        self.engine = engine
        self.context: Any = pl.SQLContext()
        self._temp_directory = DataFusionBackend._resolve_temp_directory(  # noqa: SLF001
            temp_directory
        )
        self._materialized_files: dict[str, TemporaryDirectory[str]] = {}

    @classmethod
    def connect(cls, **options: Any) -> PolarsBackend:
        """Connect with the streaming engine and disk-backed computed results.

        ``engine`` may also be ``"in-memory"`` or ``"auto"``. ``temp_directory``
        holds the private Parquet directories created by compute().
        """
        return cls(**options)

    def from_parquet(
        self,
        paths: ParquetPath | Sequence[ParquetPath],
        *,
        shape: tuple[int, int],
        storage: StorageKind | str = StorageKind.SPARSE,
        name: str | None = None,
    ) -> DBArray[np.float64]:
        """Lazily scan canonical coordinate Parquet files as a matrix.

        The contract matches :meth:`DataFusionBackend.from_parquet`.
        """
        import polars as pl

        with self._registration_scope():
            self._check_open()
            rows, cols = self._validate_shape(shape)
            kind = StorageKind(storage)
            parquet_paths = DataFusionBackend._validate_parquet_paths(paths)  # noqa: SLF001
            for path in parquet_paths:
                DataFusionBackend._validate_coordinate_schema(  # noqa: SLF001
                    pq.read_schema(path), path=path
                )
            frame = pl.scan_parquet(list(parquet_paths)).select(
                pl.col("i").cast(pl.Int64),
                pl.col("j").cast(pl.Int64),
                pl.col("x").cast(pl.Float64),
            )
            relation = self._relation_name(name)
            self.context.register(relation, frame)
            self._record_created(relation)
            rows_relation, cols_relation = self.dimension_relations((rows, cols))
            return self._matrix(
                Source(
                    relation,
                    rows_relation,
                    cols_relation,
                    (rows, cols),
                    "float64",
                    kind,
                )
            )

    def _query(self, sql: str) -> Any:
        return self.context.execute(to_polars_sql(sql), eager=False)

    def _register_arrow(self, name: str, table: pa.Table) -> None:
        import polars as pl

        self._validate_name(name)
        self.context.register(name, pl.DataFrame(table).lazy())
        self._record_created(name)

    def _unregister_relation(self, name: str) -> None:
        self.context.unregister(name)
        directory = self._materialized_files.pop(name, None)
        if directory is not None:
            directory.cleanup()

    def _register_dimension(self, name: str, *, column: str, size: int) -> None:
        import polars as pl

        self._validate_name(name)
        self._validate_name(column)
        frame = pl.LazyFrame().select(
            pl.int_range(0, size, dtype=pl.Int64).alias(column)
        )
        self.context.register(name, frame)
        self._record_created(name)

    def _execute_sql(self, sql: str) -> pa.Table:
        return self._query(sql).collect(engine=self.engine).to_arrow()

    def _explain_sql(self, sql: str) -> str:
        return str(self._query(sql).explain(engine=self.engine))

    def _materialize_sql(self, name: str, sql: str) -> None:
        """Stream a result into Parquet, then register a scan of it."""
        import polars as pl

        self._validate_name(name)
        if name in self.context.tables():
            raise ValueError(f"relation {name!r} already exists in Polars")
        directory = TemporaryDirectory(
            prefix="dbnumpy-compute-", dir=self._temp_directory
        )
        try:
            target = Path(directory.name) / "part.parquet"
            # Parquet statistics omit NaNs; do not let scans trust them.
            self._query(sql).sink_parquet(target, statistics=False)
            self.context.register(name, pl.scan_parquet(target))
        except BaseException:
            directory.cleanup()
            raise
        self._materialized_files[name] = directory
        self._record_created(name)

    def _close_impl(self) -> None:
        for name in tuple(self.context.tables()):
            self._unregister_relation(name)
        self.context = None
