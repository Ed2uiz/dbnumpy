"""Backend execution contract shared by DuckDB and DataFusion."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
from itertools import count
from operator import index
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pyarrow as pa

from dbnumpy.backends.capabilities import (
    V1_MATRIX_NODES,
    V1_REDUCTIONS,
    BackendCapabilities,
)
from dbnumpy.exceptions import DensificationError, UnsupportedOperationError
from dbnumpy.ir import (
    BOOLEAN_REDUCTION_OPS,
    MatrixExpr,
    ReductionOp,
    Source,
    StorageKind,
    expression_depth,
    has_sparse_source,
    storage_kind,
)
from dbnumpy.lowering import IbisLowerer, LoweredReduction

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dbnumpy.matrix import DBArray

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _copy_arrow_column[ScalarT: np.generic](
    column: pa.ChunkedArray, dtype: type[ScalarT]
) -> np.ndarray[Any, np.dtype[ScalarT]]:
    """Copy each Arrow chunk once without joining the input buffers first."""
    output = np.empty(len(column), dtype=dtype)
    offset = 0
    for chunk in column.chunks:
        stop = offset + len(chunk)
        output[offset:stop] = chunk.to_numpy(zero_copy_only=False)
        offset = stop
    return output


class Backend(ABC):
    """Resource-owning analytical backend.

    A matrix keeps this object by identity, which makes accidental cross-engine
    expression trees an explicit error instead of an implicit data transfer.
    """

    dialect: str
    native_pointwise_execution = False
    native_matrix_market_ingestion = False
    external_parquet_scan = False

    def __init__(
        self,
        *,
        max_densify_cells: int = 5_000_000,
        max_host_values: int = 50_000_000,
        max_sparse_host_values: int = 5_000_000,
        max_selector_values: int = 5_000_000,
        max_selector_relations: int = 256,
        lowerer: IbisLowerer | None = None,
    ) -> None:
        if max_densify_cells < 0:
            raise ValueError("max_densify_cells must be nonnegative")
        if max_host_values < 0:
            raise ValueError("max_host_values must be nonnegative")
        if max_sparse_host_values < 0:
            raise ValueError("max_sparse_host_values must be nonnegative")
        if max_selector_values < 0:
            raise ValueError("max_selector_values must be nonnegative")
        if max_selector_relations < 0:
            raise ValueError("max_selector_relations must be nonnegative")
        self.max_densify_cells = max_densify_cells
        self.max_host_values = max_host_values
        self.max_sparse_host_values = max_sparse_host_values
        self.max_selector_values = max_selector_values
        self.max_selector_relations = max_selector_relations
        self.lowerer = lowerer or IbisLowerer()
        self.identity = uuid4().hex
        self._names = count()
        self._owned_relations: set[str] = set()
        self._created_relations: set[str] = set()
        self._registration_journal: list[tuple[str, Any]] = []
        self._registration_depth = 0
        self._dimension_cache: dict[tuple[int, int], tuple[str, str]] = {}
        self._selector_cache: dict[tuple[int, bytes], str] = {}
        self._selector_values_registered = 0
        self._compile_cache: OrderedDict[int, tuple[MatrixExpr, str]] = OrderedDict()
        self._reduction_cache: OrderedDict[
            tuple[int, ReductionOp, int | None, float],
            tuple[MatrixExpr, str, LoweredReduction],
        ] = OrderedDict()
        self._compile_cache_limit = 256
        self._closed = False

    @property
    def name(self) -> str:
        return type(self).__name__.removesuffix("Backend").lower()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name=self.name,
            sql_dialect=self.dialect,
            semantic_nodes=V1_MATRIX_NODES,
            reductions=V1_REDUCTIONS,
            native_pointwise_execution=self.native_pointwise_execution,
            native_matrix_market_ingestion=self.native_matrix_market_ingestion,
            external_parquet_scan=self.external_parquet_scan,
        )

    def from_numpy(
        self,
        values: np.ndarray[Any, Any] | Sequence[Sequence[float]],
        *,
        name: str | None = None,
    ) -> DBArray[np.float64]:
        """Copy a two-dimensional array into a dense coordinate relation."""

        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError(f"expected a two-dimensional array, got ndim={array.ndim}")
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(f"dense matrix values must be numeric, got {array.dtype}")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError(
                "complex matrix values are not supported by the V1 float64 IR"
            )
        rows, cols = array.shape
        self._guard_dense_input(rows * cols)
        if rows and cols:
            i = np.repeat(np.arange(rows, dtype=np.int64), cols)
            j = np.tile(np.arange(cols, dtype=np.int64), rows)
        else:
            i = np.array([], dtype=np.int64)
            j = np.array([], dtype=np.int64)
        # Arrow can borrow NumPy buffers. Own the values before registration so
        # later writes to an input array/view cannot change a lazy matrix.
        x = np.array(array, dtype=np.float64, order="C", copy=True).reshape(-1)
        return self._from_coordinate_arrays(
            i,
            j,
            x,
            shape=(rows, cols),
            storage=StorageKind.DENSE,
            name=name,
        )

    def from_coo(
        self,
        row: Sequence[int] | np.ndarray[Any, Any],
        col: Sequence[int] | np.ndarray[Any, Any],
        data: Sequence[float] | np.ndarray[Any, Any],
        *,
        shape: tuple[int, int],
        name: str | None = None,
    ) -> DBArray[np.float64]:
        """Copy canonical COO coordinates into a sparse relation."""

        raw_i = np.asarray(row)
        raw_j = np.asarray(col)
        raw_x = np.asarray(data)
        if raw_i.ndim != 1 or raw_j.ndim != 1 or raw_x.ndim != 1:
            raise ValueError("COO row, col, and data inputs must be one-dimensional")
        for label, coordinates in (("row", raw_i), ("column", raw_j)):
            if coordinates.size and (
                not np.issubdtype(coordinates.dtype, np.integer)
                or np.issubdtype(coordinates.dtype, np.bool_)
            ):
                raise TypeError(f"COO {label} indices must have an integer dtype")
        if not (
            np.issubdtype(raw_x.dtype, np.number)
            or np.issubdtype(raw_x.dtype, np.bool_)
        ):
            raise TypeError(f"COO data must be numeric, got {raw_x.dtype}")
        if np.issubdtype(raw_x.dtype, np.complexfloating):
            raise TypeError("complex COO data are not supported by the V1 float64 IR")
        i = raw_i.astype(np.int64, copy=False)
        j = raw_j.astype(np.int64, copy=False)
        x = raw_x.astype(np.float64, copy=False)
        if not (len(i) == len(j) == len(x)):
            raise ValueError("COO row, col, and data lengths must match")
        rows, cols = self._validate_shape(shape)
        if len(i) and (i.min() < 0 or i.max() >= rows):
            raise IndexError("COO row index is outside the declared shape")
        if len(j) and (j.min() < 0 or j.max() >= cols):
            raise IndexError("COO column index is outside the declared shape")

        # Canonical sparse storage omits explicit zeros. Duplicate coordinates
        # are intentionally rejected here; from_scipy() canonicalizes them.
        nonzero = x != 0.0
        i, j, x = i[nonzero], j[nonzero], x[nonzero]
        if len(i):
            coordinate_pairs = np.stack((i, j), axis=1)
            if len(np.unique(coordinate_pairs, axis=0)) != len(i):
                raise ValueError("COO coordinates must be unique")

        return self._from_coordinate_arrays(
            i,
            j,
            x,
            shape=(rows, cols),
            storage=StorageKind.SPARSE,
            name=name,
        )

    def from_scipy(
        self, values: Any, *, name: str | None = None
    ) -> DBArray[np.float64]:
        """Copy a SciPy sparse array or legacy sparse matrix into the backend."""

        try:
            import scipy.sparse as sp
        except ImportError as exc:  # pragma: no cover - exercised without extra
            msg = "from_scipy() requires the optional scipy dependency"
            raise ImportError(msg) from exc
        if not sp.issparse(values):
            raise TypeError("from_scipy() expects a SciPy sparse object")
        coo = values.tocoo(copy=True)
        coo.sum_duplicates()
        coo.eliminate_zeros()
        return self.from_coo(
            coo.row,
            coo.col,
            coo.data,
            shape=coo.shape,
            name=name,
        )

    def from_relation(
        self,
        relation: str,
        *,
        shape: tuple[int, int],
        storage: StorageKind | str,
    ) -> DBArray[np.float64]:
        """Wrap an existing ``(i, j, x)`` relation without copying its values."""

        with self._registration_scope():
            self._check_open()
            self._validate_name(relation)
            if relation in self._owned_relations:
                raise ValueError(
                    f"relation {relation!r} is already owned by this backend"
                )
            try:
                schema = self._execute_sql(
                    f'SELECT "i", "j", "x" FROM "{relation}" LIMIT 0'
                )
            except Exception as exc:
                raise ValueError(
                    f"relation {relation!r} must exist and expose "
                    "numeric i, j, x columns"
                ) from exc
            i_type = schema.schema.field("i").type
            j_type = schema.schema.field("j").type
            x_type = schema.schema.field("x").type
            if not (
                pa.types.is_integer(i_type)
                and pa.types.is_integer(j_type)
                and (pa.types.is_integer(x_type) or pa.types.is_floating(x_type))
            ):
                raise ValueError(
                    f"relation {relation!r} must expose integer i/j "
                    "and numeric x columns"
                )
            rows, cols = self._validate_shape(shape)
            kind = StorageKind(storage)
            # Reserve the external name before generating dimension relations so a
            # user relation that resembles our internal prefix cannot be replaced.
            self._reserve_relation(relation)
            rows_relation, cols_relation = self.dimension_relations((rows, cols))
            source = Source(
                relation,
                rows_relation,
                cols_relation,
                (rows, cols),
                "float64",
                kind,
            )
            return self._matrix(source)

    def compile(self, expr: MatrixExpr) -> str:
        self._check_open()
        self.guard_densification(expr)
        key = id(expr)
        cached = self._compile_cache.get(key)
        if cached is not None and cached[0] is expr:
            self._compile_cache.move_to_end(key)
            return cached[1]
        sql = self.lowerer.compile_matrix(expr, dialect=self.dialect)
        self._compile_cache[key] = (expr, sql)
        self._compile_cache.move_to_end(key)
        if len(self._compile_cache) > self._compile_cache_limit:
            self._compile_cache.popitem(last=False)
        return sql

    def collect_matrix(self, expr: MatrixExpr) -> np.ndarray[Any, np.dtype[np.float64]]:
        self._check_open()
        self.guard_densification(expr)
        self.guard_host_collection(expr, sparse_output=False)
        table = self._execute_matrix_expr(expr)
        output = np.zeros(expr.shape, dtype=np.float64)
        if table.num_rows:
            i = table.column("i").to_numpy(zero_copy_only=False).astype(np.intp)
            j = table.column("j").to_numpy(zero_copy_only=False).astype(np.intp)
            x = table.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
            output[i, j] = x
        return output

    def collect_coordinates(
        self, expr: MatrixExpr
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.int64]],
        np.ndarray[Any, np.dtype[np.int64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        """Execute a plan and return canonical nonzero COO arrays."""

        self._check_open()
        self.guard_densification(expr)
        self.guard_host_collection(expr, sparse_output=True)
        # Bound the Arrow result itself, rather than counting only after an
        # unbounded collect. Stored explicit zeros also consume this budget.
        sql = self.compile(expr)
        table = self._execute_sql(
            f'SELECT * FROM ({sql}) AS "__dbm_sparse_export" '
            f"LIMIT {self.max_sparse_host_values + 1}"
        )
        if table.num_rows > self.max_sparse_host_values:
            raise DensificationError(
                "sparse export exceeds "
                f"max_sparse_host_values={self.max_sparse_host_values:,}; "
                "keep the result database-resident or explicitly raise the limit"
            )
        if not table.num_rows:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )
        # Copy chunks directly into owned, writable arrays. Converting a whole
        # ChunkedArray first can allocate a concatenation buffer as well as the
        # subsequent dtype copy, while retaining the complete Arrow result.
        i = _copy_arrow_column(table.column("i"), np.int64)
        j = _copy_arrow_column(table.column("j"), np.int64)
        x = _copy_arrow_column(table.column("x"), np.float64)
        del table
        nonzero = x != 0.0
        if nonzero.all():
            return i, j, x
        return i[nonzero], j[nonzero], x[nonzero]

    def collect_vector(
        self, expr: MatrixExpr, *, varying_axis: int
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Collect a singleton-backed lazy vector under the 1D host guard."""

        self._check_open()
        self.guard_densification(expr)
        if varying_axis not in (0, 1):
            raise ValueError("a vector's varying axis must be 0 or 1")
        fixed_axis = 1 - varying_axis
        if expr.shape[fixed_axis] != 1:
            raise ValueError(
                "a lazy vector must use a singleton internal matrix dimension"
            )
        length = expr.shape[varying_axis]
        if length > self.max_host_values:
            raise DensificationError(
                f"vector collection would return {length:,} host values, above "
                f"max_host_values={self.max_host_values:,}"
            )
        table = self._execute_matrix_expr(expr)
        output = np.zeros(length, dtype=np.float64)
        if table.num_rows:
            coordinate = "i" if varying_axis == 0 else "j"
            positions = (
                table.column(coordinate).to_numpy(zero_copy_only=False).astype(np.intp)
            )
            values = table.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
            output[positions] = values
        return output

    def collect_scalar(self, expr: MatrixExpr) -> float:
        """Collect a singleton matrix expression as one Python float."""

        self._check_open()
        self.guard_densification(expr)
        if expr.shape != (1, 1):
            raise ValueError("a lazy scalar must use a (1, 1) internal matrix")
        table = self._execute_matrix_expr(expr)
        if not table.num_rows:
            return 0.0
        value = table.column("x")[0].as_py()
        return float("nan") if value is None else float(value)

    def collect_reduction(
        self,
        expr: MatrixExpr,
        op: ReductionOp,
        axis: int | None,
        *,
        ddof: float = 0,
    ) -> float | bool | np.ndarray[Any, Any]:
        self._check_open()
        self.guard_densification(expr)
        if axis is not None:
            output_length = expr.shape[1] if axis == 0 else expr.shape[0]
            if output_length > self.max_host_values:
                raise DensificationError(
                    f"axis reduction would collect {output_length:,} host values, "
                    f"above max_host_values={self.max_host_values:,}"
                )
        sql, lowered = self._compile_reduction(expr, op, axis, ddof=ddof)
        table = self._execute_sql(sql)
        if axis is None:
            if not table.num_rows:
                return lowered.default_value
            value = table.column("value")[0].as_py()
            if value is None:
                return lowered.default_value
            return bool(value) if op in BOOLEAN_REDUCTION_OPS else float(value)

        assert lowered.output_length is not None
        dtype = np.bool_ if op in BOOLEAN_REDUCTION_OPS else np.float64
        output: np.ndarray[Any, Any] = np.full(
            lowered.output_length,
            lowered.default_value,
            dtype=dtype,
        )
        if table.num_rows:
            index = table.column("index").to_numpy(zero_copy_only=False).astype(np.intp)
            values = table.column("value").to_numpy(zero_copy_only=False)
            output[index] = values
        return output

    def explain(self, expr: MatrixExpr) -> str:
        self._check_open()
        self.guard_densification(expr)
        return self._explain_expr(expr)

    def materialize(
        self, expr: MatrixExpr, *, name: str | None = None
    ) -> DBArray[np.float64]:
        """Materialize a lazy expression and return a new source-backed matrix."""

        with self._registration_scope():
            relation = self._relation_name(name)
            self._materialize_expr(relation, expr)
            rows_relation, cols_relation = self.dimension_relations(expr.shape)
            source = Source(
                relation,
                rows_relation,
                cols_relation,
                expr.shape,
                expr.dtype,
                storage_kind(expr),
            )
            return self._matrix(source)

    def guard_densification(self, expr: MatrixExpr) -> None:
        """Reject an unsafe sparse-to-dense plan before SQL is generated."""

        if expression_depth(expr) > 768:
            raise UnsupportedOperationError(
                "expression exceeds the supported depth of 768 operations; "
                "use compute() on an intermediate result before composing more work"
            )
        if storage_kind(expr) is not StorageKind.DENSE:
            return
        if not has_sparse_source(expr):
            return
        cells = expr.shape[0] * expr.shape[1]
        if cells > self.max_densify_cells:
            msg = (
                f"operation would enumerate {cells:,} cells, above this backend's "
                f"max_densify_cells={self.max_densify_cells:,}; explicitly raise "
                "the limit only after checking memory and spill capacity"
            )
            raise DensificationError(msg)

    def guard_host_collection(self, expr: MatrixExpr, *, sparse_output: bool) -> None:
        """Bound predictable host materialization without blocking lazy work."""

        if sparse_output and storage_kind(expr) is StorageKind.SPARSE:
            return
        cells = expr.shape[0] * expr.shape[1]
        if cells > self.max_densify_cells:
            target = "to_scipy()" if sparse_output else "to_numpy()"
            raise DensificationError(
                f"{target} would materialize up to {cells:,} host values, above "
                f"max_densify_cells={self.max_densify_cells:,}; keep the result "
                "database-resident or explicitly raise the limit"
            )

    def guard_shape_expansion(self, cells: int, *, operation: str) -> None:
        """Reject an operation whose expanded output domain is predictably large."""

        if cells > self.max_densify_cells:
            raise DensificationError(
                f"{operation} would produce a {cells:,}-cell output domain, above "
                f"max_densify_cells={self.max_densify_cells:,}"
            )

    def close(self) -> None:
        if not self._closed:
            self._close_impl()
            self._compile_cache.clear()
            self._reduction_cache.clear()
            self._dimension_cache.clear()
            self._selector_cache.clear()
            self._selector_values_registered = 0
            self._owned_relations.clear()
            self._created_relations.clear()
            self._closed = True

    def __enter__(self) -> Backend:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _from_coordinate_arrays(
        self,
        i: np.ndarray[Any, Any],
        j: np.ndarray[Any, Any],
        x: np.ndarray[Any, Any],
        *,
        shape: tuple[int, int],
        storage: StorageKind,
        name: str | None,
    ) -> DBArray[np.float64]:
        with self._registration_scope():
            self._check_open()
            relation = self._relation_name(name)
            table = pa.table(
                {
                    "i": pa.array(i, type=pa.int64()),
                    "j": pa.array(j, type=pa.int64()),
                    "x": pa.array(x, type=pa.float64()),
                }
            )
            self._register_arrow(relation, table)
            rows_relation, cols_relation = self.dimension_relations(shape)
            source = Source(
                relation,
                rows_relation,
                cols_relation,
                shape,
                "float64",
                storage,
                finite_values=bool(np.isfinite(x).all()),
            )
            return self._matrix(source)

    def dimension_relations(self, shape: tuple[int, int]) -> tuple[str, str]:
        """Return shared row/column domain relations for a logical shape."""

        with self._registration_scope():
            rows, cols = self._validate_shape(shape)
            cached = self._dimension_cache.get((rows, cols))
            if cached is not None:
                return cached
            while True:
                suffix = next(self._names)
                rows_relation = f"dbm_dim_{rows}_{cols}_rows_{suffix}"
                cols_relation = f"dbm_dim_{rows}_{cols}_cols_{suffix}"
                if not {rows_relation, cols_relation} & self._owned_relations:
                    self._reserve_relation(rows_relation)
                    self._reserve_relation(cols_relation)
                    break
            self._register_dimension(rows_relation, column="i", size=rows)
            self._register_dimension(cols_relation, column="j", size=cols)
            result = (rows_relation, cols_relation)
            self._dimension_cache[(rows, cols)] = result
            self._record_registration("dimension", (rows, cols))
            return result

    def _selector_relation(self, indices: np.ndarray[Any, Any]) -> str:
        """Register a bounded output-to-source index mapping for lazy gather."""

        with self._registration_scope():
            self._check_open()
            normalized = np.asarray(indices)
            if normalized.ndim != 1 or not np.issubdtype(normalized.dtype, np.integer):
                raise TypeError(
                    "selector relation indices must be one-dimensional integers"
                )
            normalized = np.ascontiguousarray(normalized, dtype=np.int64)
            length = len(normalized)
            digest = sha256(memoryview(normalized).cast("B")).digest()
            cache_key = (length, digest)
            cached = self._selector_cache.get(cache_key)
            if cached is not None:
                return cached
            if len(self._selector_cache) >= self.max_selector_relations:
                raise DensificationError(
                    "backend selector-relation budget is exhausted at "
                    f"max_selector_relations={self.max_selector_relations:,}; "
                    "close the backend to release lazy selector resources"
                )
            next_total = self._selector_values_registered + length
            if next_total > self.max_selector_values:
                raise DensificationError(
                    f"selector mappings would retain {next_total:,} values, above "
                    f"max_selector_values={self.max_selector_values:,}; close the "
                    "backend to release lazy selector resources"
                )
            while True:
                relation = f"dbm_selector_{next(self._names)}"
                if relation not in self._owned_relations:
                    self._reserve_relation(relation)
                    break
            table = pa.table(
                {
                    "source_index": pa.array(normalized, type=pa.int64()),
                    "output_index": pa.array(np.arange(length), type=pa.int64()),
                }
            )
            self._register_arrow(relation, table)
            self._selector_cache[cache_key] = relation
            self._record_registration("selector", cache_key)
            self._selector_values_registered = next_total
            return relation

    def _relation_name(self, requested: str | None) -> str:
        if requested is None:
            while True:
                candidate = f"dbm_{next(self._names)}"
                if candidate not in self._owned_relations:
                    self._reserve_relation(candidate)
                    return candidate
        self._validate_name(requested)
        if requested in self._owned_relations:
            raise ValueError(f"relation {requested!r} is already owned by this backend")
        self._reserve_relation(requested)
        return requested

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(
                f"relation names must match [A-Za-z_][A-Za-z0-9_]*; got {name!r}"
            )

    @staticmethod
    def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError("matrix shape must contain exactly two dimensions")
        if isinstance(shape[0], (bool, np.bool_)) or isinstance(
            shape[1], (bool, np.bool_)
        ):
            raise TypeError("matrix dimensions must be integers, not booleans")
        try:
            rows, cols = index(shape[0]), index(shape[1])
        except TypeError as exc:
            raise TypeError(f"matrix dimensions must be integers; got {shape}") from exc
        if rows < 0 or cols < 0:
            raise ValueError(f"matrix dimensions must be nonnegative; got {shape}")
        max_dimension = np.iinfo(np.int64).max
        if rows > max_dimension or cols > max_dimension:
            raise OverflowError(
                "matrix dimensions must fit the signed int64 coordinate contract"
            )
        return rows, cols

    def _guard_dense_input(self, cells: int) -> None:
        # Dense ingestion is explicit rather than an accidental operation, but
        # the same cap prevents a surprising host-side COO expansion.
        if cells > self.max_densify_cells:
            raise DensificationError(
                f"dense ingestion would create {cells:,} coordinate rows, above "
                f"max_densify_cells={self.max_densify_cells:,}"
            )

    def _matrix(self, expr: MatrixExpr) -> DBArray[np.float64]:
        from dbnumpy.matrix import matrix_from_expr

        return matrix_from_expr(self, expr)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _record_registration(self, kind: str, key: Any) -> None:
        if self._registration_depth:
            self._registration_journal.append((kind, key))

    def _reserve_relation(self, name: str) -> None:
        if name not in self._owned_relations:
            self._owned_relations.add(name)
            self._record_registration("owned", name)

    def _record_created(self, name: str) -> None:
        self._created_relations.add(name)
        self._record_registration("created", name)

    @contextmanager
    def _registration_scope(self) -> Iterator[None]:
        """Journal new resources without copying a growing session catalog."""

        self._check_open()
        start = len(self._registration_journal)
        selector_values = self._selector_values_registered
        self._registration_depth += 1
        try:
            yield
        except BaseException:
            for kind, key in reversed(self._registration_journal[start:]):
                if kind == "created":
                    self._unregister_relation(key)
                    self._created_relations.discard(key)
                elif kind == "owned":
                    self._owned_relations.discard(key)
                elif kind == "dimension":
                    self._dimension_cache.pop(key, None)
                elif kind == "selector":
                    self._selector_cache.pop(key, None)
            del self._registration_journal[start:]
            self._selector_values_registered = selector_values
            raise
        finally:
            self._registration_depth -= 1
            if not self._registration_depth:
                self._registration_journal.clear()

    def _compile_reduction(
        self,
        expr: MatrixExpr,
        op: ReductionOp,
        axis: int | None,
        *,
        ddof: float,
    ) -> tuple[str, LoweredReduction]:
        key = (id(expr), op, axis, ddof)
        cached = self._reduction_cache.get(key)
        if cached is not None and cached[0] is expr:
            self._reduction_cache.move_to_end(key)
            return cached[1], cached[2]
        sql, lowered = self.lowerer.compile_reduction(
            expr,
            op,
            axis,
            dialect=self.dialect,
            ddof=ddof,
        )
        self._reduction_cache[key] = (expr, sql, lowered)
        self._reduction_cache.move_to_end(key)
        if len(self._reduction_cache) > self._compile_cache_limit:
            self._reduction_cache.popitem(last=False)
        return sql, lowered

    def _execute_matrix_expr(self, expr: MatrixExpr) -> pa.Table:
        return self._execute_sql(self.compile(expr))

    def _explain_expr(self, expr: MatrixExpr) -> str:
        return self._explain_sql(self.compile(expr))

    def _materialize_expr(self, name: str, expr: MatrixExpr) -> None:
        self._materialize_sql(name, self.compile(expr))

    @abstractmethod
    def _register_arrow(self, name: str, table: pa.Table) -> None: ...

    def _unregister_relation(self, name: str) -> None:
        """Adapters must release registrations they record as created."""

        raise NotImplementedError

    @abstractmethod
    def _register_dimension(self, name: str, *, column: str, size: int) -> None: ...

    @abstractmethod
    def _execute_sql(self, sql: str) -> pa.Table: ...

    @abstractmethod
    def _explain_sql(self, sql: str) -> str: ...

    @abstractmethod
    def _materialize_sql(self, name: str, sql: str) -> None: ...

    @abstractmethod
    def _close_impl(self) -> None: ...
