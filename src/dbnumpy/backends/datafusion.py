"""Apache DataFusion execution adapter."""

from __future__ import annotations

import os
from collections import OrderedDict
from os import PathLike, fspath
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from dbnumpy.backends.base import Backend
from dbnumpy.ir import MatrixExpr, Source, StorageKind, single_pointwise_source
from dbnumpy.lowering.datafusion_native import DataFusionPointwiseLowerer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dbnumpy.matrix import DBArray

type ParquetPath = str | PathLike[str]


class DataFusionBackend(Backend):
    dialect = "datafusion"
    native_pointwise_execution = True
    external_parquet_scan = True

    def __init__(
        self,
        context: Any,
        *,
        temp_directory: str | Path | None = None,
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
        self.context = context
        self.native_pointwise_lowerer = DataFusionPointwiseLowerer()
        self._native_plan_cache: OrderedDict[int, tuple[MatrixExpr, Any]] = (
            OrderedDict()
        )
        self._external_parquet_relations: set[str] = set()
        self._temp_directory = self._resolve_temp_directory(temp_directory)
        self._materialized_files: dict[str, TemporaryDirectory[str]] = {}

    @classmethod
    def connect(
        cls,
        *,
        target_partitions: int = 2,
        memory_limit_bytes: int = 2 * 1024**3,
        temp_directory: str | Path | None = None,
        sql_parser_recursion_limit: int = 512,
        max_densify_cells: int = 5_000_000,
        max_host_values: int = 50_000_000,
        max_sparse_host_values: int = 5_000_000,
        max_selector_values: int = 5_000_000,
        max_selector_relations: int = 256,
    ) -> DataFusionBackend:
        """Connect with bounded query memory and disk-backed computed results.

        ``temp_directory`` must be an existing writable directory. It holds
        engine spill files and private Parquet directories created by compute().
        If omitted, the system temporary directory is used. Close the backend
        to release computed results; caller-owned input files are never removed.
        """
        if target_partitions < 1:
            raise ValueError("target_partitions must be positive")
        if memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        if sql_parser_recursion_limit < 1:
            raise ValueError("sql_parser_recursion_limit must be positive")
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
        from datafusion import RuntimeEnvBuilder, SessionConfig, SessionContext

        resolved_temp_directory = cls._resolve_temp_directory(temp_directory)
        config = (
            SessionConfig()
            .with_target_partitions(target_partitions)
            .set(
                "datafusion.sql_parser.recursion_limit",
                str(sql_parser_recursion_limit),
            )
        )
        runtime = (
            RuntimeEnvBuilder()
            .with_disk_manager_os()
            .with_fair_spill_pool(memory_limit_bytes)
        )
        if resolved_temp_directory is not None:
            runtime = runtime.with_temp_file_path(resolved_temp_directory)
        context = SessionContext(config=config, runtime=runtime)
        return cls(
            context,
            temp_directory=resolved_temp_directory,
            max_densify_cells=max_densify_cells,
            max_host_values=max_host_values,
            max_sparse_host_values=max_sparse_host_values,
            max_selector_values=max_selector_values,
            max_selector_relations=max_selector_relations,
        )

    @staticmethod
    def _resolve_temp_directory(directory: str | Path | None) -> Path | None:
        if directory is None:
            return None
        resolved = Path(directory).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"temp_directory must name an existing directory; got {resolved}"
            )
        if not os.access(resolved, os.W_OK | os.X_OK):
            raise ValueError(f"temp_directory is not writable: {resolved}")
        return resolved

    def from_parquet(
        self,
        paths: ParquetPath | Sequence[ParquetPath],
        *,
        shape: tuple[int, int],
        storage: StorageKind | str = StorageKind.SPARSE,
        name: str | None = None,
    ) -> DBArray[np.float64]:
        """Lazily wrap canonical coordinate Parquet files as a matrix.

        Every file must contain numeric ``(i, j, x)`` columns, with integer
        zero-based ``i``/``j`` coordinates. As with :meth:`from_relation`, the
        caller is responsible for ensuring coordinates are in bounds, unique,
        and consistent with ``storage``. No PyArrow table or record-batch list
        is constructed: DataFusion keeps one or more external Parquet scans in
        the lazy plan and casts the coordinate relation to int64/int64/float64.

        Registration persists only in this backend's current SessionContext.
        The source files must remain present and unchanged until lazy work has
        finished; closing the backend discards the registration metadata.
        """

        with self._registration_scope():
            self._check_open()
            rows, cols = self._validate_shape(shape)
            kind = StorageKind(storage)
            parquet_paths = self._validate_parquet_paths(paths)

            from datafusion import SessionConfig, SessionContext, col

            # DataFusion 54 replaces columns with min == max by a constant.
            # Parquet statistics exclude NaNs, so [NaN, 2] can become [2, 2].
            # Define scans without file statistics in a separate context, then
            # register their plans below. Execution still uses self.context and
            # its runtime limits; caller-owned context settings remain unchanged.
            reader = SessionContext(
                SessionConfig().set("datafusion.execution.collect_statistics", "false")
            )

            frames = []
            for path in parquet_paths:
                frame = reader.read_parquet(path)
                self._validate_coordinate_schema(frame.schema(), path=path)
                frames.append(
                    frame.select(
                        col("i").cast(pa.int64()).alias("i"),
                        col("j").cast(pa.int64()).alias("j"),
                        col("x").cast(pa.float64()).alias("x"),
                    )
                )
            frame = frames[0]
            for other in frames[1:]:
                frame = frame.union(other)

            relation = self._relation_name(name)
            try:
                self.context.register_table(relation, frame)
            except Exception:
                self._owned_relations.discard(relation)
                raise
            self._record_created(relation)
            self._external_parquet_relations.add(relation)
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

    @staticmethod
    def _validate_parquet_paths(
        paths: ParquetPath | Sequence[ParquetPath],
    ) -> tuple[str, ...]:
        candidates: tuple[ParquetPath, ...]
        if isinstance(paths, (str, PathLike)):
            candidates = (paths,)
        else:
            candidates = tuple(paths)
        if not candidates:
            raise ValueError("at least one Parquet file path is required")

        validated = []
        for candidate in candidates:
            try:
                raw_path = fspath(candidate)
            except TypeError as exc:
                raise TypeError(
                    "Parquet paths must be strings or path-like objects"
                ) from exc
            if not isinstance(raw_path, str):
                raise TypeError("Parquet paths must resolve to strings, not bytes")
            if not raw_path or "\x00" in raw_path:
                raise ValueError("Parquet file paths must be nonempty and valid")
            path = Path(raw_path).expanduser()
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Parquet file does not exist: {path}") from exc
            if not resolved.is_file():
                raise ValueError(f"Parquet path must identify a regular file: {path}")
            normalized = str(resolved)
            if normalized in validated:
                raise ValueError(f"Parquet file paths must be unique: {path}")
            validated.append(normalized)
        return tuple(validated)

    @staticmethod
    def _validate_coordinate_schema(schema: pa.Schema, *, path: str) -> None:
        try:
            i_type = schema.field("i").type
            j_type = schema.field("j").type
            x_type = schema.field("x").type
        except KeyError as exc:
            raise ValueError(
                f"Parquet file {path!r} must expose numeric i, j, x columns"
            ) from exc
        if not (
            pa.types.is_integer(i_type)
            and pa.types.is_integer(j_type)
            and (pa.types.is_integer(x_type) or pa.types.is_floating(x_type))
        ):
            raise ValueError(
                f"Parquet file {path!r} must expose integer i/j and numeric x columns"
            )

    def _register_arrow(self, name: str, table: pa.Table) -> None:
        self._validate_name(name)
        batches = table.to_batches(max_chunksize=65_536)
        if not batches:
            batches = [
                pa.RecordBatch.from_arrays(
                    [pa.array([], type=field.type) for field in table.schema],
                    schema=table.schema,
                )
            ]
        self.context.register_record_batches(name, [batches])
        self._record_created(name)

    def _unregister_relation(self, name: str) -> None:
        self.context.deregister_table(name)
        self._external_parquet_relations.discard(name)
        directory = self._materialized_files.get(name)
        if directory is not None:
            directory.cleanup()
            del self._materialized_files[name]

    def _register_dimension(self, name: str, *, column: str, size: int) -> None:
        self._validate_name(name)
        self._validate_name(column)
        frame = self.context.sql(
            f'SELECT value AS "{column}" FROM generate_series(0, {size - 1})'
        )
        self.context.register_table(name, frame)
        self._record_created(name)

    def _execute_sql(self, sql: str) -> pa.Table:
        return self.context.sql(sql).to_arrow_table()

    def _matrix_frame(self, expr: MatrixExpr) -> Any:
        """Choose one execution path for collection, storage, and inspection."""
        if single_pointwise_source(expr) is not None:
            return self._native_pointwise_plan(expr)
        return self.context.sql(self.compile(expr))

    def _execute_matrix_expr(self, expr: MatrixExpr) -> pa.Table:
        return self._matrix_frame(expr).to_arrow_table()

    def _explain_sql(self, sql: str) -> str:
        table = self.context.sql(f"EXPLAIN {sql}").to_arrow_table()
        return "\n".join(
            " | ".join(str(value) for value in row.values())
            for row in table.to_pylist()
        )

    def _explain_expr(self, expr: MatrixExpr) -> str:
        frame = self._matrix_frame(expr)
        return f"{frame.optimized_logical_plan()}\n{frame.execution_plan()}"

    def _materialize_sql(self, name: str, sql: str) -> None:
        self._materialize_frame(name, self.context.sql(sql))

    def _materialize_expr(self, name: str, expr: MatrixExpr) -> None:
        self._materialize_frame(name, self._matrix_frame(expr))

    def _materialize_frame(self, name: str, frame: Any) -> None:
        """Stream a result into engine-written Parquet, then register its scan.

        DataFusion's cache() holds every output batch in RAM. Its streaming
        file writer bounds the output buffering without a Python collection or
        a second query execution. Supplying the schema also supports empty
        outputs, for which the writer may produce no data files.
        """
        from datafusion import ParquetColumnOptions, ParquetWriterOptions

        self._validate_name(name)
        if self.context.table_exist(name):
            raise ValueError(f"relation {name!r} already exists in DataFusion")
        directory = TemporaryDirectory(
            prefix="dbnumpy-compute-", dir=self._temp_directory
        )
        try:
            # Parquet min/max statistics omit NaNs and cannot distinguish signed
            # zeros. DataFusion 54 can use them to replace a value column with a
            # constant. Keep coordinate statistics, but require reading x values.
            options = ParquetWriterOptions(
                column_specific_options={
                    "x": ParquetColumnOptions(statistics_enabled="none")
                }
            )
            frame.write_parquet(directory.name, compression=options)
            stored = self.context.read_parquet(directory.name, schema=frame.schema())
            self.context.register_table(name, stored)
        except BaseException:
            directory.cleanup()
            raise
        self._materialized_files[name] = directory
        self._external_parquet_relations.add(name)
        self._record_created(name)

    def _native_pointwise_plan(self, expr: MatrixExpr) -> Any:
        key = id(expr)
        cached = self._native_plan_cache.get(key)
        if cached is not None and cached[0] is expr:
            self._native_plan_cache.move_to_end(key)
            return cached[1]
        frame = self.native_pointwise_lowerer.lower(expr, context=self.context)
        self._native_plan_cache[key] = (expr, frame)
        self._native_plan_cache.move_to_end(key)
        if len(self._native_plan_cache) > self._compile_cache_limit:
            self._native_plan_cache.popitem(last=False)
        return frame

    def _close_impl(self) -> None:
        # SessionContext currently has no explicit close API. Dropping the last
        # reference releases registered providers and runtime resources.
        self._native_plan_cache.clear()
        for name in tuple(self._materialized_files):
            self._unregister_relation(name)
        self._external_parquet_relations.clear()
        self.context = None
