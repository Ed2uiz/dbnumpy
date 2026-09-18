"""Python-native lazy matrix objects."""

from __future__ import annotations

from numbers import Real
from operator import index
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from dbnumpy.exceptions import BackendMismatchError, UnsupportedOperationError
from dbnumpy.ir import (
    BinaryOp,
    Broadcast,
    ElementwiseBinary,
    MatMul,
    MatrixExpr,
    ReductionOp,
    Scalar,
    ScalarBinary,
    Slice,
    StorageKind,
    Transpose,
    Unary,
    UnaryOp,
    known_finite,
    storage_kind,
    to_dag,
)

if TYPE_CHECKING:
    from dbnumpy.backends.base import Backend
    from dbnumpy.indexing import DBScalar, DBVector


class DBArray[ScalarT: np.generic]:
    """Base class for a lazy two-dimensional database-backed array.

    The object is intentionally not an ``ndarray`` subclass: it has no resident
    strided buffer. Selected NumPy protocols preserve familiar syntax while
    explicit collection remains visible through ``to_numpy()``.
    """

    __array_priority__: ClassVar[float] = 1000.0
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, backend: Backend, expr: MatrixExpr) -> None:
        self._backend = backend
        self._expr = expr

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def shape(self) -> tuple[int, int]:
        return self._expr.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._expr.dtype)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def storage(self) -> StorageKind:
        return storage_kind(self._expr)

    @property
    def T(self) -> DBArray[ScalarT]:  # noqa: N802 - NumPy-compatible spelling
        return self.transpose()

    def __len__(self) -> int:
        return self.shape[0]

    def __bool__(self) -> bool:
        raise ValueError(
            "the truth value of a DBArray is ambiguous; use a reduction explicitly"
        )

    def transpose(self) -> DBArray[ScalarT]:
        return matrix_from_expr(self.backend, Transpose(self._expr))

    def take(
        self,
        indices: Any,
        axis: int | None = None,
        out: Any = None,
        mode: str = "raise",
    ) -> DBArray[ScalarT] | DBVector:
        """Lazily gather one matrix axis using NumPy ``take`` semantics."""

        if axis is None:
            raise UnsupportedOperationError(
                "axis=None would flatten the matrix and requires point-gather semantics"
            )
        if out is not None:
            raise UnsupportedOperationError("out= is not supported for lazy matrices")
        if mode != "raise":
            raise UnsupportedOperationError(
                "only mode='raise' is supported for lazy matrix take"
            )
        normalized_axis = self._normalize_axis(axis)
        if normalized_axis is None:  # pragma: no cover - excluded above
            raise AssertionError("take axis unexpectedly normalized to None")
        key = (indices, slice(None)) if normalized_axis == 0 else (slice(None), indices)
        result = self[key]
        return cast("DBArray[ScalarT] | DBVector", result)

    def __getitem__(self, key: Any) -> DBArray[ScalarT] | DBVector | DBScalar:
        from dbnumpy.indexing import index_matrix

        return index_matrix(self, key)

    def compile(self) -> str:
        """Compile the lazy semantic plan without executing it."""

        return self.backend.compile(self._expr)

    def explain(self) -> str:
        """Return the target engine's plan for this expression."""

        return self.backend.explain(self._expr)

    def plan(self) -> dict[str, Any]:
        """Return the backend-independent semantic plan as plain data."""

        return {"dbverse_ir_version": 1, **to_dag(self._expr)}

    def compute(self, *, name: str | None = None) -> DBArray[np.float64]:
        """Materialize this plan inside its current analytical backend."""

        return self.backend.materialize(self._expr, name=name)

    def to_numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Execute and collect the complete matrix into host memory."""

        return self.backend.collect_matrix(self._expr)

    def toarray(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """SciPy-compatible alias for ``to_numpy()``."""

        return self.to_numpy()

    def show(self, *, edgeitems: int = 5) -> None:
        """Print the first and last entries along each axis.

        Runs a coordinate-filtered query and collects at most ``(2*edgeitems)**2``
        values. The array stays lazy. Database work may exceed the preview size
        for expressions whose requested values depend on larger inputs.
        """
        values = self.backend._collect_preview(self._expr, edgeitems=edgeitems)
        if not values.size:
            print(repr(self))
            print("[]")
            return
        edgeitems = index(edgeitems)
        # Insert a hidden middle row/column where NumPy should print an ellipsis.
        # Never allocate a display buffer with the original matrix dimensions.
        shape = tuple(min(size, 2 * edgeitems + 1) for size in self.shape)
        display = np.zeros(shape, dtype=np.float64)
        positions = [
            np.concatenate((np.arange(edgeitems), np.arange(edgeitems + 1, size)))
            if size > 2 * edgeitems
            else np.arange(size)
            for size in shape
        ]
        display[np.ix_(*positions)] = values
        print(repr(self))
        print(np.array2string(display, edgeitems=edgeitems, threshold=0))

    def to_scipy(self, *, format: str = "csr") -> Any:
        """Execute and collect as a SciPy sparse array."""

        try:
            import scipy.sparse as sp
        except ImportError as exc:  # pragma: no cover - exercised without extra
            msg = "to_scipy() requires the optional scipy dependency"
            raise ImportError(msg) from exc
        i, j, x = self.backend.collect_coordinates(self._expr)
        result = sp.coo_array((x, (i, j)), shape=self.shape)
        return result.asformat(format)

    def sum(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a NumPy-compatible sum over the requested axis."""

        self._validate_reduction_options(dtype, out, initial, where)
        return self._reduce(ReductionOp.SUM, axis, keepdims=keepdims)

    def mean(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a NumPy-compatible arithmetic mean over an axis."""

        self._validate_reduction_options(dtype, out, None, where)
        return self._reduce(ReductionOp.MEAN, axis, keepdims=keepdims)

    def var(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        ddof: float = 0,
        keepdims: bool = False,
        *,
        where: Any = True,
        mean: Any = None,
        correction: Any = None,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a variance reduction, including fractional ``ddof``."""

        self._validate_reduction_options(dtype, out, None, where)
        ddof = self._resolve_ddof(ddof, mean=mean, correction=correction)
        return self._reduce(ReductionOp.VAR, axis, ddof=ddof, keepdims=keepdims)

    def std(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        ddof: float = 0,
        keepdims: bool = False,
        *,
        where: Any = True,
        mean: Any = None,
        correction: Any = None,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a standard-deviation reduction over an axis."""

        self._validate_reduction_options(dtype, out, None, where)
        ddof = self._resolve_ddof(ddof, mean=mean, correction=correction)
        return self._reduce(ReductionOp.STD, axis, ddof=ddof, keepdims=keepdims)

    def min(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a minimum that includes implicit sparse zeros."""

        self._validate_reduction_options(None, out, initial, where)
        return self._reduce(ReductionOp.MIN, axis, keepdims=keepdims)

    def max(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a maximum that includes implicit sparse zeros."""

        self._validate_reduction_options(None, out, initial, where)
        return self._reduce(ReductionOp.MAX, axis, keepdims=keepdims)

    def any(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = True,
    ) -> bool | np.ndarray[Any, np.dtype[np.bool_]]:
        """Return whether any logical value is truthy using NumPy semantics."""

        self._validate_reduction_options(None, out, None, where)
        result = self._reduce(ReductionOp.ANY, axis, keepdims=keepdims)
        return cast(
            bool | np.ndarray[Any, np.dtype[np.bool_]],
            result,
        )

    def all(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = True,
    ) -> bool | np.ndarray[Any, np.dtype[np.bool_]]:
        """Return whether every logical value is truthy using NumPy semantics."""

        self._validate_reduction_options(None, out, None, where)
        result = self._reduce(ReductionOp.ALL, axis, keepdims=keepdims)
        return cast(
            bool | np.ndarray[Any, np.dtype[np.bool_]],
            result,
        )

    def nansum(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a sum that skips stored NaNs but retains sparse zeros."""

        self._validate_reduction_options(dtype, out, initial, where)
        return self._reduce(ReductionOp.NANSUM, axis, keepdims=keepdims)

    def nanmean(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a mean that skips stored NaNs but retains sparse zeros."""

        self._validate_reduction_options(dtype, out, None, where)
        return self._reduce(ReductionOp.NANMEAN, axis, keepdims=keepdims)

    def nanvar(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        ddof: float = 0,
        keepdims: bool = False,
        *,
        where: Any = True,
        mean: Any = None,
        correction: Any = None,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a NaN-skipping variance, including fractional ``ddof``."""

        self._validate_reduction_options(dtype, out, None, where)
        ddof = self._resolve_ddof(ddof, mean=mean, correction=correction)
        return self._reduce(ReductionOp.NANVAR, axis, ddof=ddof, keepdims=keepdims)

    def nanstd(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
        ddof: float = 0,
        keepdims: bool = False,
        *,
        where: Any = True,
        mean: Any = None,
        correction: Any = None,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a NaN-skipping standard deviation over an axis."""

        self._validate_reduction_options(dtype, out, None, where)
        ddof = self._resolve_ddof(ddof, mean=mean, correction=correction)
        return self._reduce(ReductionOp.NANSTD, axis, ddof=ddof, keepdims=keepdims)

    def nanmin(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a NaN-skipping minimum including implicit sparse zeros."""

        self._validate_reduction_options(None, out, initial, where)
        return self._reduce(ReductionOp.NANMIN, axis, keepdims=keepdims)

    def nanmax(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Execute a NaN-skipping maximum including implicit sparse zeros."""

        self._validate_reduction_options(None, out, initial, where)
        return self._reduce(ReductionOp.NANMAX, axis, keepdims=keepdims)

    def sqrt(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.SQRT)

    def exp(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.EXP)

    def expm1(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.EXPM1)

    def log(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.LOG)

    def log1p(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.LOG1P)

    def sin(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.SIN)

    def cos(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.COS)

    def tan(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.TAN)

    def floor(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.FLOOR)

    def ceil(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.CEIL)

    def sign(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise NumPy sign expression."""

        return self._unary(UnaryOp.SIGN)

    def trunc(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise truncation toward zero."""

        return self._unary(UnaryOp.TRUNC)

    def isnan(self) -> DBArray[np.float64]:
        """Return a lazy float64 0/1 matrix identifying stored NaNs."""

        return cast(DBArray[np.float64], self._unary(UnaryOp.ISNAN))

    def log2(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise base-2 logarithm."""

        return self._unary(UnaryOp.LOG2)

    def log10(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise base-10 logarithm."""

        return self._unary(UnaryOp.LOG10)

    def sinh(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise hyperbolic sine."""

        return self._unary(UnaryOp.SINH)

    def cosh(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise hyperbolic cosine."""

        return self._unary(UnaryOp.COSH)

    def tanh(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise hyperbolic tangent."""

        return self._unary(UnaryOp.TANH)

    def arcsin(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise inverse sine."""

        return self._unary(UnaryOp.ARCSIN)

    def asin(self) -> DBArray[ScalarT]:
        """Alias for `arcsin()`, matching R's function spelling."""

        return self.arcsin()

    def arccos(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise inverse cosine."""

        return self._unary(UnaryOp.ARCCOS)

    def acos(self) -> DBArray[ScalarT]:
        """Alias for `arccos()`, matching R's function spelling."""

        return self.arccos()

    def arctan(self) -> DBArray[ScalarT]:
        """Return a lazy elementwise inverse tangent."""

        return self._unary(UnaryOp.ARCTAN)

    def atan(self) -> DBArray[ScalarT]:
        """Alias for `arctan()`, matching R's function spelling."""

        return self.arctan()

    def __array__(
        self,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, Any]:
        result = self.to_numpy()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy is True:
            result = result.copy()
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
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
            if isinstance(left, DBArray):
                return left._binary(right, binary[ufunc])
            if isinstance(right, DBArray):
                return right._binary(left, binary[ufunc], reverse=True)
        if ufunc is np.matmul and len(inputs) == 2:
            left, right = inputs
            if isinstance(left, DBArray):
                return left.__matmul__(right)
            if isinstance(right, DBArray):
                return right.__rmatmul__(left)
        return NotImplemented

    def __array_function__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not all(
            issubclass(operand_type, (DBArray, np.ndarray)) for operand_type in types
        ):
            return NotImplemented
        dispatch = {
            np.sum: "sum",
            np.mean: "mean",
            np.var: "var",
            np.std: "std",
            np.min: "min",
            np.amin: "min",
            np.max: "max",
            np.amax: "max",
            np.any: "any",
            np.all: "all",
            np.nansum: "nansum",
            np.nanmean: "nanmean",
            np.nanvar: "nanvar",
            np.nanstd: "nanstd",
            np.nanmin: "nanmin",
            np.nanmax: "nanmax",
            np.transpose: "transpose",
            np.take: "take",
        }
        name = dispatch.get(func)
        if name is None:
            return NotImplemented
        matrix = args[0]
        if not isinstance(matrix, DBArray):
            return NotImplemented
        if name == "transpose":
            axes = kwargs.get("axes", args[1] if len(args) > 1 else None)
            if axes is None:
                return matrix.transpose()
            try:
                raw_axes = tuple(index(axis) for axis in axes)
            except (TypeError, ValueError) as exc:
                raise ValueError("axes must be a permutation of (0, 1)") from exc
            if len(raw_axes) != 2 or any(axis < -2 or axis >= 2 for axis in raw_axes):
                raise ValueError("axes must be a permutation of (0, 1)")
            normalized_axes = tuple(axis % 2 for axis in raw_axes)
            if set(normalized_axes) != {0, 1}:
                raise ValueError("axes must be a permutation of (0, 1)")
            return matrix if normalized_axes == (0, 1) else matrix.transpose()
        return getattr(matrix, name)(*args[1:], **kwargs)

    def __neg__(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.NEGATIVE)

    def __abs__(self) -> DBArray[ScalarT]:
        return self._unary(UnaryOp.ABSOLUTE)

    def __add__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.ADD)

    def __radd__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.ADD, reverse=True)

    def __sub__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.SUBTRACT)

    def __rsub__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.SUBTRACT, reverse=True)

    def __mul__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.MULTIPLY)

    def __rmul__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.MULTIPLY, reverse=True)

    def __truediv__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.TRUE_DIVIDE)

    def __rtruediv__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.TRUE_DIVIDE, reverse=True)

    def __pow__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.POWER)

    def __rpow__(self, other: Any) -> DBArray[Any]:
        return self._binary(other, BinaryOp.POWER, reverse=True)

    def __gt__(self, other: Any) -> DBArray[np.float64]:
        return self._binary(other, BinaryOp.GREATER)

    def __ge__(self, other: Any) -> DBArray[np.float64]:
        return self._binary(other, BinaryOp.GREATER_EQUAL)

    def __lt__(self, other: Any) -> DBArray[np.float64]:
        return self._binary(other, BinaryOp.LESS)

    def __le__(self, other: Any) -> DBArray[np.float64]:
        return self._binary(other, BinaryOp.LESS_EQUAL)

    def __eq__(self, other: object) -> DBArray[np.float64]:  # type: ignore[override]
        return self._binary(other, BinaryOp.EQUAL)

    def __ne__(self, other: object) -> DBArray[np.float64]:  # type: ignore[override]
        return self._binary(other, BinaryOp.NOT_EQUAL)

    def __matmul__(self, other: Any) -> DBArray[Any]:
        with self.backend._registration_scope():
            if isinstance(other, (np.ndarray, list, tuple)):
                other = self._upload_matmul_operand(other)
            if not isinstance(other, DBArray):
                return NotImplemented
            self._require_same_backend(other)
            expr = MatMul(self._expr, other._expr)
            if storage_kind(expr) is StorageKind.DENSE:
                self.backend.guard_shape_expansion(
                    expr.shape[0] * expr.shape[1],
                    operation="dense matrix multiplication",
                )
            return matrix_from_expr(self.backend, expr)

    def __rmatmul__(self, other: Any) -> DBArray[Any]:
        if isinstance(other, DBArray):
            return other.__matmul__(self)
        if isinstance(other, (np.ndarray, list, tuple)):
            return self._upload_matmul_operand(other).__matmul__(self)
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"backend={self.backend.name!r}, lazy=True)"
        )

    def _unary(self, op: UnaryOp) -> DBArray[ScalarT]:
        return matrix_from_expr(self.backend, Unary(op, self._expr))

    def _binary(
        self,
        other: Any,
        op: BinaryOp,
        *,
        reverse: bool = False,
    ) -> DBArray[Any]:
        with self.backend._registration_scope():
            from dbnumpy.indexing import DBScalar, DBVector

            expr: MatrixExpr
            if isinstance(other, DBScalar):
                if self.backend is not other.backend:
                    raise BackendMismatchError(
                        "matrix and scalar belong to different backend instances"
                    )
                left, right = (
                    (other._expr, self._expr) if reverse else (self._expr, other._expr)
                )
                left, right = self._broadcast_pair(left, right, op=op)
                expr = ElementwiseBinary(op, left, right)
            elif isinstance(other, DBVector):
                if self.backend is not other.backend:
                    raise BackendMismatchError(
                        "matrix and vector belong to different backend instances"
                    )
                vector_expr = other._row_expr()
                left, right = (
                    (vector_expr, self._expr) if reverse else (self._expr, vector_expr)
                )
                left, right = self._broadcast_pair(left, right, op=op)
                expr = ElementwiseBinary(op, left, right)
            elif isinstance(other, DBArray):
                self._require_same_backend(other)
                if reverse:
                    left, right = other._expr, self._expr
                else:
                    left, right = self._expr, other._expr
                left, right = self._broadcast_pair(left, right, op=op)
                expr = ElementwiseBinary(op, left, right)
            elif isinstance(other, (np.ndarray, list, tuple)):
                array = np.asarray(other)
                if array.ndim == 0:
                    return self._binary(array.item(), op, reverse=reverse)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                if array.ndim != 2:
                    raise ValueError(
                        "array operands must be one- or two-dimensional "
                        "for broadcasting"
                    )
                uploaded = self.backend.from_numpy(array)
                return self._binary(uploaded, op, reverse=reverse)
            elif np.isscalar(other) and not isinstance(other, (str, bytes, complex)):
                scalar: Scalar
                if isinstance(other, (bool, np.bool_)):
                    scalar = bool(other)
                else:
                    scalar = float(other)
                expr = ScalarBinary(op, self._expr, scalar, reverse=reverse)
            else:
                msg = f"unsupported operand type for DBArray: {type(other).__name__}"
                raise TypeError(msg)
            return matrix_from_expr(self.backend, expr)

    def _broadcast_pair(
        self, left: MatrixExpr, right: MatrixExpr, *, op: BinaryOp
    ) -> tuple[MatrixExpr, MatrixExpr]:
        if left.shape == right.shape:
            return left, right
        try:
            target = np.broadcast_shapes(left.shape, right.shape)
        except ValueError as exc:
            raise ValueError(
                f"matrix shapes {left.shape} and {right.shape} are not broadcastable"
            ) from exc
        if len(target) != 2:
            raise ValueError("DBArray broadcasting must produce two dimensions")
        target_shape = (int(target[0]), int(target[1]))
        expands_dense_operand = (
            left.shape != target_shape and storage_kind(left) is StorageKind.DENSE
        ) or (right.shape != target_shape and storage_kind(right) is StorageKind.DENSE)
        sparse_scaling = op is BinaryOp.MULTIPLY and any(
            values.shape == target_shape
            and storage_kind(values) is StorageKind.SPARSE
            and known_finite(factors)
            for values, factors in ((left, right), (right, left))
        )
        if expands_dense_operand and not sparse_scaling:
            self.backend.guard_shape_expansion(
                target_shape[0] * target_shape[1], operation="matrix broadcasting"
            )
        rows_relation, cols_relation = self.backend.dimension_relations(target_shape)
        if left.shape != target_shape:
            left = Broadcast(left, target_shape, rows_relation, cols_relation)
        if right.shape != target_shape:
            right = Broadcast(right, target_shape, rows_relation, cols_relation)
        return left, right

    def _upload_matmul_operand(self, other: Any) -> DBArray[np.float64]:
        array = np.asarray(other)
        if array.ndim != 2:
            raise ValueError(
                "matrix multiplication operands must be two-dimensional; "
                "lazy vector result types are not implemented yet"
            )
        return self.backend.from_numpy(array)

    def _slice(self, rows: slice, cols: slice) -> DBArray[ScalarT]:
        row_start, row_stop, row_step = rows.indices(self.shape[0])
        col_start, col_stop, col_step = cols.indices(self.shape[1])
        row_count = len(range(row_start, row_stop, row_step))
        col_count = len(range(col_start, col_stop, col_step))
        shape = (row_count, col_count)
        rows_relation, cols_relation = self.backend.dimension_relations(shape)
        expr = Slice(
            self._expr,
            row_start,
            row_stop,
            row_step,
            col_start,
            col_stop,
            col_step,
            shape,
            rows_relation,
            cols_relation,
        )
        return matrix_from_expr(self.backend, expr)

    def _require_same_backend(self, other: DBArray[Any]) -> None:
        if self.backend is not other.backend:
            msg = (
                "matrix operands belong to different backend instances; explicit "
                "transfer is required before combining them"
            )
            raise BackendMismatchError(msg)

    def _reduce(
        self,
        op: ReductionOp,
        axis: int | None,
        *,
        ddof: float = 0,
        keepdims: bool = False,
    ) -> float | bool | np.ndarray[Any, Any]:
        keep_dimensions = self._normalize_keepdims(keepdims)
        axis = self._normalize_axis(axis)
        if op in {
            ReductionOp.MIN,
            ReductionOp.MAX,
            ReductionOp.NANMIN,
            ReductionOp.NANMAX,
        }:
            reduced_length = self.size if axis is None else self.shape[axis]
            if reduced_length == 0:
                raise ValueError(
                    "zero-size array to reduction operation which has no identity"
                )
        result = self.backend.collect_reduction(self._expr, op, axis, ddof=ddof)
        return self._keepdims(result, axis) if keep_dimensions else result

    @staticmethod
    def _normalize_axis(axis: int | None) -> int | None:
        if axis is None:
            return None
        if isinstance(axis, (bool, np.bool_)):
            raise TypeError("axis must be an integer, not a boolean")
        try:
            normalized = index(axis)
        except TypeError as exc:
            raise TypeError(f"axis must be an integer; got {axis!r}") from exc
        if normalized in (-2, -1):
            normalized %= 2
        if normalized not in (0, 1):
            raise np.exceptions.AxisError(normalized, ndim=2)
        return normalized

    @staticmethod
    def _normalize_keepdims(keepdims: bool) -> bool:
        try:
            return index(keepdims) != 0
        except TypeError as exc:
            raise TypeError(f"keepdims must be an integer; got {keepdims!r}") from exc

    @staticmethod
    def _resolve_ddof(ddof: Any, *, mean: Any, correction: Any) -> float | int:
        if mean is not None:
            raise UnsupportedOperationError("precomputed mean is not supported yet")
        if correction is not None:
            if ddof != 0:
                raise ValueError("ddof and correction cannot both be supplied")
            ddof = correction
        if not isinstance(ddof, (Real, np.bool_)):
            raise TypeError(f"ddof must be a real number; got {ddof!r}")
        if isinstance(ddof, (bool, np.bool_, int, np.integer)):
            normalized: float | int = int(ddof)
        else:
            normalized = float(ddof)
        if isinstance(normalized, float) and not np.isfinite(normalized):
            raise ValueError(f"ddof must be finite; got {normalized}")
        if normalized < 0:
            raise ValueError(f"ddof must be nonnegative; got {normalized}")
        return normalized

    @staticmethod
    def _validate_reduction_options(
        dtype: Any,
        out: Any,
        initial: Any,
        where: Any,
    ) -> None:
        if dtype is not None and np.dtype(dtype) != np.dtype("float64"):
            raise UnsupportedOperationError("only float64 reductions are supported")
        if out is not None:
            raise UnsupportedOperationError("out= is not supported for lazy matrices")
        if initial is not None:
            raise UnsupportedOperationError("initial= is not supported yet")
        if not DBArray._where_selects_all(where):
            raise UnsupportedOperationError("where= is not supported yet")

    @staticmethod
    def _where_selects_all(where: Any) -> bool:
        if isinstance(where, np.ndarray):
            return (
                where.ndim == 0 and where.dtype == np.dtype(bool) and bool(where.item())
            )
        if np.isscalar(where):
            try:
                return bool(where)
            except (TypeError, ValueError):
                return False
        return False

    @staticmethod
    def _keepdims(
        result: float | bool | np.ndarray[Any, Any], axis: int | None
    ) -> np.ndarray[Any, Any]:
        array = np.asarray(result)
        if axis is None:
            return array.reshape(1, 1)
        return np.expand_dims(array, axis=axis)


class DBDenseArray[ScalarT: np.generic](DBArray[ScalarT]):
    """A matrix whose relational plan enumerates every coordinate."""


class DBSparseArray[ScalarT: np.generic](DBArray[ScalarT]):
    """A matrix whose absent relational coordinates have the value zero."""


def matrix_from_expr(backend: Backend, expr: MatrixExpr) -> DBArray[Any]:
    backend.guard_densification(expr)
    cls = DBSparseArray if storage_kind(expr) is StorageKind.SPARSE else DBDenseArray
    return cls(backend, expr)
