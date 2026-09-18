"""NumPy-shaped lazy indexing over the canonical two-dimensional matrix IR."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from dbnumpy.exceptions import (
    BackendMismatchError,
    DensificationError,
    UnsupportedOperationError,
)
from dbnumpy.ir import (
    BinaryOp,
    Broadcast,
    ElementwiseBinary,
    Gather,
    MatrixExpr,
    ScalarBinary,
    Slice,
    StorageKind,
    Transpose,
    Unary,
    UnaryOp,
    storage_kind,
    to_dag,
)

if TYPE_CHECKING:
    from dbnumpy.backends.base import Backend
    from dbnumpy.matrix import DBArray

type VectorLayout = Literal["row-singleton", "column-singleton"]


@dataclass(frozen=True, slots=True)
class _At:
    position: int


@dataclass(frozen=True, slots=True)
class _Range:
    start: int
    stop: int
    step: int
    length: int


@dataclass(frozen=True, slots=True)
class _GatherIndices:
    values: np.ndarray[Any, np.dtype[np.int64]]


type _Selector = _At | _Range | _GatherIndices


class _LazyRankedResult:
    """Shared explicit execution surface for rank wrappers over a 2D plan."""

    __array_priority__ = 1000.0
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, backend: Backend, expr: MatrixExpr) -> None:
        self._backend = backend
        self._expr = expr

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._expr.dtype)

    @property
    def storage(self) -> StorageKind:
        return storage_kind(self._expr)

    def compile(self) -> str:
        return self.backend.compile(self._expr)

    def explain(self) -> str:
        return self.backend.explain(self._expr)

    def _as_matrix(self) -> DBArray[Any]:
        from dbnumpy.matrix import matrix_from_expr

        return matrix_from_expr(self.backend, self._expr)

    def _unary(self, operation: UnaryOp) -> Self:
        raise NotImplementedError

    def _binary(self, other: Any, operation: BinaryOp, *, reverse: bool = False) -> Any:
        raise NotImplementedError

    def sqrt(self) -> Self:
        return self._unary(UnaryOp.SQRT)

    def exp(self) -> Self:
        return self._unary(UnaryOp.EXP)

    def expm1(self) -> Self:
        return self._unary(UnaryOp.EXPM1)

    def log(self) -> Self:
        return self._unary(UnaryOp.LOG)

    def log1p(self) -> Self:
        return self._unary(UnaryOp.LOG1P)

    def sin(self) -> Self:
        return self._unary(UnaryOp.SIN)

    def cos(self) -> Self:
        return self._unary(UnaryOp.COS)

    def tan(self) -> Self:
        return self._unary(UnaryOp.TAN)

    def floor(self) -> Self:
        return self._unary(UnaryOp.FLOOR)

    def ceil(self) -> Self:
        return self._unary(UnaryOp.CEIL)

    def sign(self) -> Self:
        return self._unary(UnaryOp.SIGN)

    def trunc(self) -> Self:
        return self._unary(UnaryOp.TRUNC)

    def isnan(self) -> Self:
        return self._unary(UnaryOp.ISNAN)

    def log2(self) -> Self:
        return self._unary(UnaryOp.LOG2)

    def log10(self) -> Self:
        return self._unary(UnaryOp.LOG10)

    def sinh(self) -> Self:
        return self._unary(UnaryOp.SINH)

    def cosh(self) -> Self:
        return self._unary(UnaryOp.COSH)

    def tanh(self) -> Self:
        return self._unary(UnaryOp.TANH)

    def arcsin(self) -> Self:
        return self._unary(UnaryOp.ARCSIN)

    def arccos(self) -> Self:
        return self._unary(UnaryOp.ARCCOS)

    def arctan(self) -> Self:
        return self._unary(UnaryOp.ARCTAN)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if any(key != "out" for key in kwargs):
            return NotImplemented
        unary = {
            np.negative: UnaryOp.NEGATIVE,
            np.absolute: UnaryOp.ABSOLUTE,
            np.sqrt: UnaryOp.SQRT,
            np.exp: UnaryOp.EXP,
            np.expm1: UnaryOp.EXPM1,
            np.log: UnaryOp.LOG,
            np.log1p: UnaryOp.LOG1P,
            np.sin: UnaryOp.SIN,
            np.cos: UnaryOp.COS,
            np.tan: UnaryOp.TAN,
            np.floor: UnaryOp.FLOOR,
            np.ceil: UnaryOp.CEIL,
            np.sign: UnaryOp.SIGN,
            np.trunc: UnaryOp.TRUNC,
            np.isnan: UnaryOp.ISNAN,
            np.log2: UnaryOp.LOG2,
            np.log10: UnaryOp.LOG10,
            np.sinh: UnaryOp.SINH,
            np.cosh: UnaryOp.COSH,
            np.tanh: UnaryOp.TANH,
            np.arcsin: UnaryOp.ARCSIN,
            np.arccos: UnaryOp.ARCCOS,
            np.arctan: UnaryOp.ARCTAN,
        }
        if ufunc in unary and len(inputs) == 1:
            return self._unary(unary[ufunc])
        binary = {
            np.add: BinaryOp.ADD,
            np.subtract: BinaryOp.SUBTRACT,
            np.multiply: BinaryOp.MULTIPLY,
            np.true_divide: BinaryOp.TRUE_DIVIDE,
            np.power: BinaryOp.POWER,
            np.greater: BinaryOp.GREATER,
            np.greater_equal: BinaryOp.GREATER_EQUAL,
            np.less: BinaryOp.LESS,
            np.less_equal: BinaryOp.LESS_EQUAL,
            np.equal: BinaryOp.EQUAL,
            np.not_equal: BinaryOp.NOT_EQUAL,
        }
        if ufunc in binary and len(inputs) == 2:
            left, right = inputs
            if left is self:
                return self._binary(right, binary[ufunc])
            if right is self:
                return self._binary(left, binary[ufunc], reverse=True)
        return NotImplemented

    def __array_function__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not all(
            issubclass(operand_type, (_LazyRankedResult, np.ndarray))
            for operand_type in types
        ):
            return NotImplemented
        if args[0] is not self:
            return NotImplemented
        if func in {np.sum, np.mean} and isinstance(self, DBVector):
            method = self.sum if func is np.sum else self.mean
            return method(*args[1:], **kwargs)
        if func is not np.transpose:
            return NotImplemented
        axes = kwargs.get("axes", args[1] if len(args) > 1 else None)
        expected = tuple(range(self.ndim))
        if axes is None:
            return self.transpose()
        try:
            raw_axes = tuple(index(axis) for axis in axes)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"axes must be {expected} for this lazy result") from exc
        if len(raw_axes) != self.ndim:
            raise ValueError(f"axes must be {expected} for this lazy result")
        if self.ndim == 0:
            return self.transpose()
        for axis in raw_axes:
            if axis < -self.ndim or axis >= self.ndim:
                raise np.exceptions.AxisError(axis, ndim=self.ndim)
        normalized = tuple(axis % self.ndim for axis in raw_axes)
        if normalized != expected:
            raise ValueError(f"axes must be {expected} for this lazy result")
        return self.transpose()

    def __neg__(self) -> Self:
        return self._unary(UnaryOp.NEGATIVE)

    def __abs__(self) -> Self:
        return self._unary(UnaryOp.ABSOLUTE)

    def __add__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.ADD)

    def __radd__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.ADD, reverse=True)

    def __sub__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.SUBTRACT)

    def __rsub__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.SUBTRACT, reverse=True)

    def __mul__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.MULTIPLY)

    def __rmul__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.MULTIPLY, reverse=True)

    def __truediv__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.TRUE_DIVIDE)

    def __rtruediv__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.TRUE_DIVIDE, reverse=True)

    def __pow__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.POWER)

    def __rpow__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.POWER, reverse=True)

    def __gt__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.GREATER)

    def __ge__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.GREATER_EQUAL)

    def __lt__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.LESS)

    def __le__(self, other: Any) -> Any:
        return self._binary(other, BinaryOp.LESS_EQUAL)

    def __eq__(self, other: object) -> Any:
        return self._binary(other, BinaryOp.EQUAL)

    def __ne__(self, other: object) -> Any:
        return self._binary(other, BinaryOp.NOT_EQUAL)

    def __bool__(self) -> bool:
        raise ValueError(
            f"the truth value of a {type(self).__name__} is lazy; collect it explicitly"
        )


class DBVector(_LazyRankedResult):
    """A lazy one-dimensional result backed by a singleton 2D matrix plan.

    ``layout`` is an internal physical invariant, not vector orientation.
    NumPy vectors have no row/column orientation, so it never changes public
    one-dimensional semantics.
    """

    def __init__(
        self, backend: Backend, expr: MatrixExpr, *, layout: VectorLayout
    ) -> None:
        super().__init__(backend, expr)
        varying_axis = 1 if layout == "row-singleton" else 0
        fixed_axis = 1 - varying_axis
        if expr.shape[fixed_axis] != 1:
            raise ValueError("DBVector requires one singleton internal dimension")
        self._layout = layout
        self._varying_axis = varying_axis

    @property
    def shape(self) -> tuple[int]:
        return (self._expr.shape[self._varying_axis],)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return self.shape[0]

    @property
    def T(self) -> DBVector:  # noqa: N802 - NumPy-compatible spelling
        return self

    def __len__(self) -> int:
        return self.size

    def plan(self) -> dict[str, Any]:
        return {
            "dbverse_ir_version": 1,
            "result": {
                "rank": 1,
                "shape": [self.size],
                "internal_layout": self._layout,
            },
            **to_dag(self._expr),
        }

    def compute(self, *, name: str | None = None) -> DBVector:
        materialized = self.backend.materialize(self._expr, name=name)
        return DBVector(self.backend, materialized._expr, layout=self._layout)

    def to_numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.backend.collect_vector(self._expr, varying_axis=self._varying_axis)

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray[Any, Any]:
        result = self.to_numpy()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy is True:
            result = result.copy()
        return result

    def __getitem__(self, key: Any) -> DBVector | DBScalar:
        vector_key = _split_vector_key(key)
        matrix_key = (0, vector_key) if self._varying_axis == 1 else (vector_key, 0)
        result = index_matrix(self._as_matrix(), matrix_key)
        if not isinstance(result, (DBVector, DBScalar)):
            raise AssertionError("one-dimensional indexing changed to matrix rank")
        return result

    def transpose(self) -> DBVector:
        return self

    def sum(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        _validate_vector_axis(axis)
        result = self._as_matrix().sum(
            axis=None,
            dtype=dtype,
            out=out,
            keepdims=False,
            initial=initial,
            where=where,
        )
        numeric = float(result)
        return np.asarray([numeric], dtype=np.float64) if keepdims else numeric

    def mean(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        _validate_vector_axis(axis)
        result = self._as_matrix().mean(
            axis=None,
            dtype=dtype,
            out=out,
            keepdims=False,
            where=where,
        )
        numeric = float(result)
        return np.asarray([numeric], dtype=np.float64) if keepdims else numeric

    def _unary(self, operation: UnaryOp) -> DBVector:
        expr = Unary(operation, self._expr)
        self.backend.guard_densification(expr)
        return DBVector(self.backend, expr, layout=self._layout)

    def _binary(self, other: Any, operation: BinaryOp, *, reverse: bool = False) -> Any:
        with self.backend._registration_scope():
            from dbnumpy.matrix import DBArray

            if isinstance(other, DBVector):
                if self.backend is not other.backend:
                    raise BackendMismatchError(
                        "vector operands belong to different backend instances"
                    )
                if self.shape != other.shape:
                    raise ValueError(
                        f"vector operands must have equal shapes; got {self.shape} and "
                        f"{other.shape}"
                    )
                other_expr = other._expr
                if self._layout != other._layout:
                    other_expr = Transpose(other_expr)
                left, right = (
                    (other_expr, self._expr) if reverse else (self._expr, other_expr)
                )
                expr: MatrixExpr = ElementwiseBinary(operation, left, right)
            elif isinstance(other, DBScalar):
                if self.backend is not other.backend:
                    raise BackendMismatchError(
                        "vector and scalar belong to different backend instances"
                    )
                result = self._as_matrix()._binary(other, operation, reverse=reverse)
                return DBVector(self.backend, result._expr, layout=self._layout)
            elif isinstance(other, DBArray):
                return other._binary(self, operation, reverse=not reverse)
            elif isinstance(other, (np.ndarray, list, tuple)):
                array = np.asarray(other)
                if array.ndim == 0:
                    return self._binary(array.item(), operation, reverse=reverse)
                if array.ndim == 1:
                    if len(array) not in (1, self.size):
                        raise ValueError(
                            f"vector operands are not broadcastable: {self.shape} and "
                            f"{array.shape}"
                        )
                    shaped = (
                        array.reshape(1, -1)
                        if self._layout == "row-singleton"
                        else array.reshape(-1, 1)
                    )
                    other_expr = self.backend.from_numpy(shaped)._expr
                    left, right = (
                        (other_expr, self._expr)
                        if reverse
                        else (self._expr, other_expr)
                    )
                    left, right = self._as_matrix()._broadcast_pair(
                        left, right, op=operation
                    )
                    expr = ElementwiseBinary(operation, left, right)
                elif array.ndim == 2:
                    uploaded = self.backend.from_numpy(array)
                    return uploaded._binary(self, operation, reverse=not reverse)
                else:
                    raise ValueError("vector operands may have at most two dimensions")
            elif np.isscalar(other) and not isinstance(other, (str, bytes, complex)):
                scalar = (
                    bool(other) if isinstance(other, (bool, np.bool_)) else float(other)
                )
                expr = ScalarBinary(operation, self._expr, scalar, reverse=reverse)
            else:
                raise TypeError(
                    f"unsupported operand type for DBVector: {type(other).__name__}"
                )
            self.backend.guard_densification(expr)
            return DBVector(self.backend, expr, layout=self._layout)

    def _row_expr(self) -> MatrixExpr:
        return self._expr if self._layout == "row-singleton" else Transpose(self._expr)

    def __repr__(self) -> str:
        return (
            f"DBVector(shape={self.shape}, dtype={self.dtype}, "
            f"backend={self.backend.name!r}, lazy=True)"
        )


class DBScalar(_LazyRankedResult):
    """A lazy scalar result backed by a singleton matrix expression."""

    def __init__(self, backend: Backend, expr: MatrixExpr) -> None:
        super().__init__(backend, expr)
        if expr.shape != (1, 1):
            raise ValueError("DBScalar requires a (1, 1) internal matrix")

    @property
    def shape(self) -> tuple[()]:
        return ()

    @property
    def ndim(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def T(self) -> DBScalar:  # noqa: N802 - NumPy-compatible spelling
        return self

    def transpose(self) -> DBScalar:
        return self

    def plan(self) -> dict[str, Any]:
        return {
            "dbverse_ir_version": 1,
            "result": {"rank": 0, "shape": []},
            **to_dag(self._expr),
        }

    def compute(self, *, name: str | None = None) -> DBScalar:
        materialized = self.backend.materialize(self._expr, name=name)
        return DBScalar(self.backend, materialized._expr)

    def item(self) -> float:
        return self.backend.collect_scalar(self._expr)

    def to_numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.asarray(self.item(), dtype=np.float64).reshape(())

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray[Any, Any]:
        result = self.to_numpy()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy is True:
            result = result.copy()
        return result

    def _unary(self, operation: UnaryOp) -> DBScalar:
        expr = Unary(operation, self._expr)
        self.backend.guard_densification(expr)
        return DBScalar(self.backend, expr)

    def _binary(self, other: Any, operation: BinaryOp, *, reverse: bool = False) -> Any:
        with self.backend._registration_scope():
            from dbnumpy.matrix import DBArray

            if isinstance(other, DBScalar):
                if self.backend is not other.backend:
                    raise BackendMismatchError(
                        "scalar operands belong to different backend instances"
                    )
                left, right = (
                    (other._expr, self._expr) if reverse else (self._expr, other._expr)
                )
                expr: MatrixExpr = ElementwiseBinary(operation, left, right)
            elif isinstance(other, DBVector):
                return other._binary(self, operation, reverse=not reverse)
            elif isinstance(other, DBArray):
                return other._binary(self, operation, reverse=not reverse)
            elif isinstance(other, (np.ndarray, list, tuple)):
                array = np.asarray(other)
                if array.ndim == 0:
                    return self._binary(array.item(), operation, reverse=reverse)
                if array.ndim == 1:
                    uploaded = self.backend.from_numpy(array.reshape(1, -1))
                    vector = DBVector(
                        self.backend, uploaded._expr, layout="row-singleton"
                    )
                    return self._binary(vector, operation, reverse=reverse)
                if array.ndim == 2:
                    uploaded = self.backend.from_numpy(array)
                    return self._binary(uploaded, operation, reverse=reverse)
                raise ValueError("scalar operands may have at most two dimensions")
            elif np.isscalar(other) and not isinstance(other, (str, bytes, complex)):
                scalar = (
                    bool(other) if isinstance(other, (bool, np.bool_)) else float(other)
                )
                expr = ScalarBinary(operation, self._expr, scalar, reverse=reverse)
            else:
                raise TypeError(
                    f"unsupported operand type for DBScalar: {type(other).__name__}"
                )
            self.backend.guard_densification(expr)
            return DBScalar(self.backend, expr)

    def __repr__(self) -> str:
        return (
            f"DBScalar(shape=(), dtype={self.dtype}, "
            f"backend={self.backend.name!r}, lazy=True)"
        )


def index_matrix(
    matrix: DBArray[Any], key: Any
) -> DBArray[Any] | DBVector | DBScalar:
    """Build a lazy, NumPy-shaped basic or one-axis advanced index plan."""

    with matrix.backend._registration_scope():
        rows_key, cols_key = _split_key(key)
        rows = _normalize_selector(
            rows_key,
            matrix.shape[0],
            axis=0,
            max_values=matrix.backend.max_selector_values,
        )
        cols = _normalize_selector(
            cols_key,
            matrix.shape[1],
            axis=1,
            max_values=matrix.backend.max_selector_values,
        )
        if isinstance(rows, _GatherIndices) and isinstance(cols, _GatherIndices):
            raise UnsupportedOperationError(
                "simultaneous advanced row and column selectors have NumPy paired "
                "semantics and are deferred; use matrix[rows, :][:, cols] for an "
                "explicit Cartesian selection"
            )

        row_range = _range_for_plan(rows, matrix.shape[0])
        col_range = _range_for_plan(cols, matrix.shape[1])
        base_shape = (row_range.length, col_range.length)
        expr: MatrixExpr = matrix._expr
        if not (
            _is_identity_range(row_range, matrix.shape[0])
            and _is_identity_range(col_range, matrix.shape[1])
        ):
            base_rows, base_cols = matrix.backend.dimension_relations(base_shape)
            expr = Slice(
                expr,
                row_range.start,
                row_range.stop,
                row_range.step,
                col_range.start,
                col_range.stop,
                col_range.step,
                base_shape,
                base_rows,
                base_cols,
            )

        advanced = rows if isinstance(rows, _GatherIndices) else cols
        if isinstance(advanced, _GatherIndices):
            relation = matrix.backend._selector_relation(advanced.values)
            gather_axis = 0 if isinstance(rows, _GatherIndices) else 1
            gather_shape = (
                len(rows.values) if isinstance(rows, _GatherIndices) else base_shape[0],
                len(cols.values) if isinstance(cols, _GatherIndices) else base_shape[1],
            )
            gather_rows, gather_cols = matrix.backend.dimension_relations(gather_shape)
            expr = Gather(
                expr,
                gather_axis,
                relation,
                gather_shape,
                gather_rows,
                gather_cols,
            )

        scalar_axes = (isinstance(rows, _At), isinstance(cols, _At))
        if scalar_axes == (True, True):
            matrix.backend.guard_densification(expr)
            return DBScalar(matrix.backend, expr)
        if scalar_axes[0] != scalar_axes[1]:
            layout: VectorLayout = (
                "row-singleton" if scalar_axes[0] else "column-singleton"
            )
            matrix.backend.guard_densification(expr)
            return DBVector(matrix.backend, expr, layout=layout)

        from dbnumpy.matrix import matrix_from_expr

        return matrix_from_expr(matrix.backend, expr)


def _split_key(key: Any) -> tuple[Any, Any]:
    if key is Ellipsis:
        return slice(None), slice(None)
    if not isinstance(key, tuple):
        return key, slice(None)
    ellipses = sum(part is Ellipsis for part in key)
    if ellipses > 1:
        raise IndexError("an index can only have a single ellipsis")
    if ellipses:
        consuming = len(key) - 1
        if consuming > 2:
            raise IndexError("DBArray indexing expects at most two dimensions")
        position = next(i for i, part in enumerate(key) if part is Ellipsis)
        missing = 2 - (len(key) - 1)
        key = key[:position] + (slice(None),) * missing + key[position + 1 :]
    elif len(key) > 2:
        raise IndexError("DBArray indexing expects at most two dimensions")
    if len(key) == 0:
        return slice(None), slice(None)
    if len(key) == 1:
        return key[0], slice(None)
    return key[0], key[1]


def _split_vector_key(key: Any) -> Any:
    if key is Ellipsis:
        return slice(None)
    if not isinstance(key, tuple):
        return key
    ellipses = sum(part is Ellipsis for part in key)
    if ellipses > 1:
        raise IndexError("an index can only have a single ellipsis")
    if ellipses:
        consuming = len(key) - 1
        if consuming > 1:
            raise IndexError("too many indices for DBVector")
        position = next(i for i, part in enumerate(key) if part is Ellipsis)
        missing = 1 - consuming
        key = key[:position] + (slice(None),) * missing + key[position + 1 :]
    elif len(key) > 1:
        raise IndexError("too many indices for DBVector")
    if not key:
        return slice(None)
    return key[0]


def _normalize_selector(
    value: Any, size: int, *, axis: int, max_values: int | None = None
) -> _Selector:
    if value is None:
        raise UnsupportedOperationError(
            "newaxis/None indexing would exceed the current two-dimensional model"
        )
    if isinstance(value, (bool, np.bool_)):
        raise UnsupportedOperationError(
            "Boolean scalar indexing changes rank and is not supported"
        )
    if isinstance(value, slice):
        start, stop, step = value.indices(size)
        return _Range(start, stop, step, len(range(start, stop, step)))
    try:
        position = index(value)
    except TypeError:
        position = None
    if position is not None:
        if position < 0:
            position += size
        if position < 0 or position >= size:
            raise IndexError(
                f"index {index(value)} is out of bounds for axis {axis} "
                f"with size {size}"
            )
        return _At(position)

    if isinstance(value, (list, tuple)) and not value:
        return _GatherIndices(np.array([], dtype=np.int64))
    array = np.asarray(value)
    if array.ndim == 0:
        raise IndexError(
            "only integers, slices, ellipsis, and integer or Boolean arrays are valid"
        )
    if array.ndim != 1:
        raise UnsupportedOperationError(
            "advanced selectors must be one-dimensional in this indexing tranche"
        )
    if np.issubdtype(array.dtype, np.bool_):
        if len(array) not in (0, size):
            raise IndexError(
                f"Boolean index has length {len(array)} for axis {axis} "
                f"with size {size}"
            )
        positions = np.flatnonzero(array).astype(np.int64, copy=False)
        _guard_selector_length(len(positions), max_values)
        return _GatherIndices(positions)
    if not np.issubdtype(array.dtype, np.integer):
        raise IndexError("advanced indices must have an integer or Boolean dtype")
    _guard_selector_length(len(array), max_values)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if len(array) and np.any(array >= size):
            raise IndexError(
                f"advanced index is out of bounds for axis {axis} with size {size}"
            )
        normalized = array.astype(np.int64, copy=False)
    else:
        if len(array) and (np.any(array < -size) or np.any(array >= size)):
            raise IndexError(
                f"advanced index is out of bounds for axis {axis} with size {size}"
            )
        normalized = array.astype(np.int64, copy=False)
        normalized = np.where(normalized < 0, normalized + size, normalized)
    return _GatherIndices(normalized)


def _range_for_plan(selector: _Selector, size: int) -> _Range:
    if isinstance(selector, _Range):
        return selector
    if isinstance(selector, _At):
        return _Range(selector.position, selector.position + 1, 1, 1)
    return _Range(0, size, 1, size)


def _is_identity_range(value: _Range, size: int) -> bool:
    return value == _Range(0, size, 1, size)


def _broadcast_to(
    backend: Backend, expr: MatrixExpr, target_shape: tuple[int, int]
) -> MatrixExpr:
    if expr.shape == target_shape:
        return expr
    backend.guard_shape_expansion(
        target_shape[0] * target_shape[1], operation="lazy scalar broadcasting"
    )
    rows_relation, cols_relation = backend.dimension_relations(target_shape)
    return Broadcast(expr, target_shape, rows_relation, cols_relation)


def _validate_vector_axis(axis: int | None) -> None:
    if axis is None:
        return
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer, not a boolean")
    try:
        normalized = index(axis)
    except TypeError as exc:
        raise TypeError(f"axis must be an integer; got {axis!r}") from exc
    if normalized not in (0, -1):
        raise np.exceptions.AxisError(normalized, ndim=1)


def _guard_selector_length(length: int, maximum: int | None) -> None:
    if maximum is not None and length > maximum:
        raise DensificationError(
            f"selector contains {length:,} positions, above "
            f"max_selector_values={maximum:,}"
        )
