"""Backend-neutral semantic nodes for two-dimensional matrix computation.

The nodes in this module deliberately contain no Ibis, DuckDB, DataFusion, or
Python-callable objects.  They are immutable and use named operations so the
same semantic plan can later be serialized or lowered by another language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from operator import index
from typing import Any

import numpy as np

type Shape = tuple[int, int]
type Scalar = bool | int | float


def _validate_shape(shape: Shape, *, node: str) -> None:
    if len(shape) != 2:
        raise ValueError(f"{node} shape must contain exactly two dimensions")
    for dimension in shape:
        if isinstance(dimension, (bool, np.bool_)):
            raise TypeError(f"{node} dimensions must be integers, not booleans")
        try:
            value = index(dimension)
        except TypeError as exc:
            raise TypeError(f"{node} dimensions must be integers") from exc
        if value < 0:
            raise ValueError(f"{node} dimensions must be nonnegative; got {shape}")


def _normalized_slice_length(
    start: int, stop: int, step: int, size: int, *, axis: str
) -> int:
    values = {"start": start, "stop": stop, "step": step}
    normalized: dict[str, int] = {}
    for label, value in values.items():
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"slice {axis} {label} must be an integer, not a boolean")
        try:
            normalized[label] = index(value)
        except TypeError as exc:
            raise TypeError(f"slice {axis} {label} must be an integer") from exc
    start = normalized["start"]
    stop = normalized["stop"]
    step = normalized["step"]
    if step == 0:
        raise ValueError("matrix slice steps must be nonzero")
    if step > 0:
        in_bounds = 0 <= start <= size and 0 <= stop <= size
    else:
        in_bounds = -1 <= start < size and -1 <= stop < size
    if not in_bounds:
        raise ValueError(
            f"slice {axis} bounds are not normalized for axis length {size}"
        )
    return len(range(start, stop, step))


class StorageKind(StrEnum):
    """Logical treatment of coordinates absent from the value relation."""

    DENSE = "dense"
    SPARSE = "sparse"


class UnaryOp(StrEnum):
    NEGATIVE = "negative"
    ABSOLUTE = "absolute"
    SQRT = "sqrt"
    EXP = "exp"
    EXPM1 = "expm1"
    LOG = "log"
    LOG1P = "log1p"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    FLOOR = "floor"
    CEIL = "ceil"
    SIGN = "sign"
    TRUNC = "trunc"
    ISNAN = "isnan"
    LOG2 = "log2"
    LOG10 = "log10"
    SINH = "sinh"
    COSH = "cosh"
    TANH = "tanh"
    ARCSIN = "arcsin"
    ARCCOS = "arccos"
    ARCTAN = "arctan"


class BinaryOp(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    TRUE_DIVIDE = "true_divide"
    POWER = "power"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"


class ReductionOp(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    VAR = "var"
    STD = "std"
    MIN = "min"
    MAX = "max"
    ANY = "any"
    ALL = "all"
    NANSUM = "nansum"
    NANMEAN = "nanmean"
    NANVAR = "nanvar"
    NANSTD = "nanstd"
    NANMIN = "nanmin"
    NANMAX = "nanmax"


BOOLEAN_REDUCTION_OPS = frozenset({ReductionOp.ANY, ReductionOp.ALL})


class MatrixExpr(ABC):
    """Base class for immutable semantic matrix expressions."""

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """The two-dimensional output shape."""

    @property
    @abstractmethod
    def dtype(self) -> str:
        """A NumPy-compatible logical dtype string."""


@dataclass(frozen=True, slots=True)
class Source(MatrixExpr):
    """A matrix stored as a coordinate relation and two dimension relations."""

    relation: str
    rows_relation: str
    cols_relation: str
    source_shape: Shape
    source_dtype: str
    storage: StorageKind
    finite_values: bool = False

    @property
    def shape(self) -> Shape:
        return self.source_shape

    @property
    def dtype(self) -> str:
        return self.source_dtype


@dataclass(frozen=True, slots=True)
class Unary(MatrixExpr):
    op: UnaryOp
    arg: MatrixExpr

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class ScalarBinary(MatrixExpr):
    op: BinaryOp
    arg: MatrixExpr
    scalar: Scalar
    reverse: bool = False

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class ElementwiseBinary(MatrixExpr):
    op: BinaryOp
    left: MatrixExpr
    right: MatrixExpr

    def __post_init__(self) -> None:
        if self.left.shape != self.right.shape:
            msg = (
                "elementwise matrix operands must have identical shapes; "
                f"got {self.left.shape} and {self.right.shape}"
            )
            raise ValueError(msg)

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class Transpose(MatrixExpr):
    arg: MatrixExpr

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class Broadcast(MatrixExpr):
    """Replicate singleton dimensions to a target two-dimensional shape."""

    arg: MatrixExpr
    target_shape: Shape
    rows_relation: str
    cols_relation: str

    def __post_init__(self) -> None:
        _validate_shape(self.target_shape, node="broadcast")
        for source_size, target_size in zip(
            self.arg.shape, self.target_shape, strict=True
        ):
            if source_size not in (1, target_size):
                raise ValueError(
                    f"cannot broadcast shape {self.arg.shape} to {self.target_shape}"
                )

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class Slice(MatrixExpr):
    """A normalized nonzero-step, two-dimensional matrix slice."""

    arg: MatrixExpr
    row_start: int
    row_stop: int
    row_step: int
    col_start: int
    col_stop: int
    col_step: int
    slice_shape: Shape
    rows_relation: str
    cols_relation: str

    def __post_init__(self) -> None:
        _validate_shape(self.slice_shape, node="slice")
        expected = (
            _normalized_slice_length(
                self.row_start,
                self.row_stop,
                self.row_step,
                self.arg.shape[0],
                axis="row",
            ),
            _normalized_slice_length(
                self.col_start,
                self.col_stop,
                self.col_step,
                self.arg.shape[1],
                axis="column",
            ),
        )
        if self.slice_shape != expected:
            raise ValueError(
                f"slice shape {self.slice_shape} does not match normalized "
                f"bounds; expected {expected}"
            )

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class Gather(MatrixExpr):
    """Select and reorder coordinates through bounded mapping relations.

    The selected axis has a two-column relation named by ``map_relation`` with
    ``source_index`` and contiguous ``output_index`` int64 columns.  A missing
    mapping means that axis is preserved.  Keeping selector payloads in
    backend-owned relations makes SQL and serialized plans independent of the
    number of selected positions.
    """

    arg: MatrixExpr
    axis: int
    map_relation: str
    gather_shape: Shape
    rows_relation: str
    cols_relation: str

    def __post_init__(self) -> None:
        _validate_shape(self.gather_shape, node="gather")
        if isinstance(self.axis, (bool, np.bool_)):
            raise ValueError("gather axis must be 0 or 1")
        try:
            axis = index(self.axis)
        except TypeError as exc:
            raise ValueError("gather axis must be 0 or 1") from exc
        if axis not in (0, 1):
            raise ValueError("gather axis must be 0 or 1")
        if not self.map_relation:
            raise ValueError("gather map relation must not be empty")
        preserved_axis = 1 - axis
        if self.gather_shape[preserved_axis] != self.arg.shape[preserved_axis]:
            raise ValueError(
                "gather must retain the unmapped axis length; "
                f"got {self.gather_shape} for input {self.arg.shape}"
            )

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


@dataclass(frozen=True, slots=True)
class MatMul(MatrixExpr):
    left: MatrixExpr
    right: MatrixExpr

    def __post_init__(self) -> None:
        if self.left.shape[1] != self.right.shape[0]:
            msg = (
                "matrix multiplication operands are not conformable; "
                f"got {self.left.shape} and {self.right.shape}"
            )
            raise ValueError(msg)

    @property
    def shape(self) -> Shape:
        return _metadata(self)[0]

    @property
    def dtype(self) -> str:
        return _metadata(self)[1]


COMPARISON_OPS = frozenset(
    {
        BinaryOp.GREATER,
        BinaryOp.GREATER_EQUAL,
        BinaryOp.LESS,
        BinaryOp.LESS_EQUAL,
        BinaryOp.EQUAL,
        BinaryOp.NOT_EQUAL,
    }
)

_ANALYSIS_CACHE_LIMIT = 8_192
_IMPLICIT_CACHE: OrderedDict[int, tuple[MatrixExpr, float | None]] = OrderedDict()
_SPARSE_SOURCE_CACHE: OrderedDict[int, tuple[MatrixExpr, bool]] = OrderedDict()


_METADATA_CACHE: OrderedDict[int, tuple[MatrixExpr, Shape, str, int]] = OrderedDict()


def expression_depth(expr: MatrixExpr) -> int:
    return _metadata(expr)[2]


def _metadata(expr: MatrixExpr) -> tuple[Shape, str, int]:
    """Compute shape/dtype/depth without recursive property access."""

    computed: dict[int, tuple[Shape, str, int]] = {}
    stack = [(expr, False)]
    while stack:
        node, visited = stack.pop()
        key = id(node)
        if key in computed:
            continue
        cached = _METADATA_CACHE.get(key)
        if cached is not None and cached[0] is node:
            _METADATA_CACHE.move_to_end(key)
            computed[key] = cached[1:]
            continue
        inputs = children(node)
        if inputs and not visited:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(inputs))
            continue
        if isinstance(node, Source):
            shape, dtype, depth = node.source_shape, node.source_dtype, 0
        else:
            shape, dtype, _ = computed[id(inputs[0])]
            depth = 1 + max(computed[id(child)][2] for child in inputs)
            if isinstance(node, Transpose):
                shape = (shape[1], shape[0])
            elif isinstance(node, Broadcast):
                shape = node.target_shape
            elif isinstance(node, Slice):
                shape = node.slice_shape
            elif isinstance(node, Gather):
                shape = node.gather_shape
            elif isinstance(node, (ElementwiseBinary, MatMul)):
                right_shape, right_dtype, _ = computed[id(node.right)]
                dtype = str(np.result_type(dtype, right_dtype))
                if isinstance(node, MatMul):
                    shape = (shape[0], right_shape[1])
            if (
                isinstance(node, (ScalarBinary, ElementwiseBinary))
                and node.op in COMPARISON_OPS
            ) or (isinstance(node, Unary) and node.op is UnaryOp.ISNAN):
                dtype = "float64"
        computed[key] = (shape, dtype, depth)
        _METADATA_CACHE[key] = (node, shape, dtype, depth)
        if len(_METADATA_CACHE) > _ANALYSIS_CACHE_LIMIT:
            _METADATA_CACHE.popitem(last=False)
    return computed[id(expr)]


def sources(expr: MatrixExpr) -> frozenset[Source]:
    """Return the distinct physical sources referenced by *expr*."""

    result: set[Source] = set()
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        key = id(node)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(node, Source):
            result.add(node)
        elif isinstance(
            node, (Unary, ScalarBinary, Transpose, Broadcast, Slice, Gather)
        ):
            stack.append(node.arg)
        elif isinstance(node, (ElementwiseBinary, MatMul)):
            stack.extend((node.left, node.right))
        else:  # pragma: no cover - sealed semantic node family
            raise AssertionError(f"unknown matrix expression: {type(node)!r}")
    return frozenset(result)


def relation_names(expr: MatrixExpr) -> frozenset[str]:
    """Return every backend relation referenced by a semantic plan."""

    result: set[str] = set()
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        key = id(node)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(node, Source):
            result.update((node.relation, node.rows_relation, node.cols_relation))
        elif isinstance(node, (Broadcast, Slice)):
            result.update((node.rows_relation, node.cols_relation))
        elif isinstance(node, Gather):
            result.update((node.map_relation, node.rows_relation, node.cols_relation))
        stack.extend(children(node))
    return frozenset(result)


def is_pointwise(expr: MatrixExpr) -> bool:
    """Whether an expression preserves each source coordinate independently."""

    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        key = id(node)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(node, Source):
            continue
        if isinstance(node, (Unary, ScalarBinary)):
            stack.append(node.arg)
            continue
        if isinstance(node, ElementwiseBinary):
            stack.extend((node.left, node.right))
            continue
        return False
    return True


def single_pointwise_source(expr: MatrixExpr) -> Source | None:
    """Return the source when a complete pointwise tree can avoid self-joins."""

    seen: set[int] = set()
    stack = [expr]
    source: Source | None = None
    while stack:
        node = stack.pop()
        key = id(node)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(node, Source):
            if source is None:
                source = node
            elif source != node:
                return None
        elif isinstance(node, (Unary, ScalarBinary)):
            stack.append(node.arg)
        elif isinstance(node, ElementwiseBinary):
            stack.extend((node.left, node.right))
        else:
            return None
    return source


def pointwise_alias_nodes(expr: MatrixExpr) -> tuple[MatrixExpr, ...]:
    """Return pointwise nodes that need one lowering alias, in dependency order.

    A semantic DAG can be small even when recursively substituting each child
    would create exponentially large SQL or backend expression trees.  The
    same problem occurs when a domain-safe operation template references its
    child in both a condition and a value branch. The lowering hot paths use
    this list to introduce one alias stage per reused value. Sources are
    excluded because every occurrence can reference the same short physical
    value column directly.
    """

    if not is_pointwise(expr):
        return ()

    uses: dict[int, int] = {}
    ordered = list(postorder(expr))
    for node in ordered:
        duplicates_children = (
            isinstance(node, Unary)
            and node.op
            in {
                UnaryOp.SQRT,
                UnaryOp.EXPM1,
                UnaryOp.LOG,
                UnaryOp.LOG1P,
                UnaryOp.SIN,
                UnaryOp.COS,
                UnaryOp.TAN,
                UnaryOp.SIGN,
                UnaryOp.TRUNC,
                UnaryOp.LOG2,
                UnaryOp.LOG10,
                UnaryOp.ARCSIN,
                UnaryOp.ARCCOS,
            }
        ) or (
            isinstance(node, (ScalarBinary, ElementwiseBinary))
            and node.op in COMPARISON_OPS | {BinaryOp.POWER}
        )
        for child in children(node):
            key = id(child)
            uses[key] = uses.get(key, 0) + (2 if duplicates_children else 1)

    return tuple(
        node
        for node in ordered
        if not isinstance(node, Source) and uses.get(id(node), 0) > 1
    )


def _apply_unary(op: UnaryOp, value: float) -> float:
    functions = {
        UnaryOp.NEGATIVE: np.negative,
        UnaryOp.ABSOLUTE: np.absolute,
        UnaryOp.SQRT: np.sqrt,
        UnaryOp.EXP: np.exp,
        UnaryOp.EXPM1: np.expm1,
        UnaryOp.LOG: np.log,
        UnaryOp.LOG1P: np.log1p,
        UnaryOp.SIN: np.sin,
        UnaryOp.COS: np.cos,
        UnaryOp.TAN: np.tan,
        UnaryOp.FLOOR: np.floor,
        UnaryOp.CEIL: np.ceil,
        UnaryOp.SIGN: np.sign,
        UnaryOp.TRUNC: np.trunc,
        UnaryOp.ISNAN: np.isnan,
        UnaryOp.LOG2: np.log2,
        UnaryOp.LOG10: np.log10,
        UnaryOp.SINH: np.sinh,
        UnaryOp.COSH: np.cosh,
        UnaryOp.TANH: np.tanh,
        UnaryOp.ARCSIN: np.arcsin,
        UnaryOp.ARCCOS: np.arccos,
        UnaryOp.ARCTAN: np.arctan,
    }
    with np.errstate(all="ignore"):
        return float(functions[op](value))


def _apply_binary(op: BinaryOp, left: float, right: float) -> float:
    functions = {
        BinaryOp.ADD: np.add,
        BinaryOp.SUBTRACT: np.subtract,
        BinaryOp.MULTIPLY: np.multiply,
        BinaryOp.TRUE_DIVIDE: np.true_divide,
        BinaryOp.POWER: np.power,
        BinaryOp.GREATER: np.greater,
        BinaryOp.GREATER_EQUAL: np.greater_equal,
        BinaryOp.LESS: np.less,
        BinaryOp.LESS_EQUAL: np.less_equal,
        BinaryOp.EQUAL: np.equal,
        BinaryOp.NOT_EQUAL: np.not_equal,
    }
    with np.errstate(all="ignore"):
        return float(functions[op](left, right))


def implicit_value(expr: MatrixExpr) -> float | None:
    """Value at an absent coordinate, or ``None`` for an enumerated relation.

    This generalized zero-image rule replaces the operation-specific sparse
    conditionals in dbmatrix-r.  A nonzero, infinite, or NaN implicit result
    means a sparse input must be enumerated before the operation is correct.
    """

    key = id(expr)
    cached = _IMPLICIT_CACHE.get(key)
    if cached is not None and cached[0] is expr:
        _IMPLICIT_CACHE.move_to_end(key)
        return cached[1]

    # Frontend construction commonly adds one node to an already-analyzed
    # immutable plan. Keep that path constant-time; reserve the iterative walk
    # for genuinely cold subgraphs.
    cached_children: dict[int, float | None] = {}
    for child in children(expr):
        child_key = id(child)
        child_cached = _IMPLICIT_CACHE.get(child_key)
        if child_cached is None or child_cached[0] is not child:
            break
        _IMPLICIT_CACHE.move_to_end(child_key)
        cached_children[child_key] = child_cached[1]
    else:
        value = _compute_implicit_value(expr, cached_children)
        _remember_implicit(expr, value)
        return value

    computed: dict[int, float | None] = {}
    discovered: set[int] = set()
    stack: list[tuple[MatrixExpr, bool]] = [(expr, False)]
    while stack:
        node, visited = stack.pop()
        node_key = id(node)
        if node_key in computed:
            continue

        node_cached = _IMPLICIT_CACHE.get(node_key)
        if node_cached is not None and node_cached[0] is node:
            _IMPLICIT_CACHE.move_to_end(node_key)
            computed[node_key] = node_cached[1]
            continue

        if visited:
            value = _compute_implicit_value(node, computed)
            computed[node_key] = value
            _remember_implicit(node, value)
            continue

        if node_key in discovered:
            continue
        discovered.add(node_key)
        stack.append((node, True))
        for child in reversed(children(node)):
            if id(child) not in computed:
                stack.append((child, False))

    return computed[key]


def _remember_implicit(expr: MatrixExpr, value: float | None) -> None:
    key = id(expr)
    _IMPLICIT_CACHE[key] = (expr, value)
    _IMPLICIT_CACHE.move_to_end(key)
    if len(_IMPLICIT_CACHE) > _ANALYSIS_CACHE_LIMIT:
        _IMPLICIT_CACHE.popitem(last=False)


def _compute_implicit_value(
    expr: MatrixExpr, computed: dict[int, float | None]
) -> float | None:
    if isinstance(expr, Source):
        return 0.0 if expr.storage is StorageKind.SPARSE else None
    if isinstance(expr, Unary):
        value = computed[id(expr.arg)]
        return None if value is None else _apply_unary(expr.op, value)
    if isinstance(expr, ScalarBinary):
        value = computed[id(expr.arg)]
        if value is None:
            return None
        scalar = float(expr.scalar)
        left, right = (scalar, value) if expr.reverse else (value, scalar)
        return _apply_binary(expr.op, left, right)
    if isinstance(expr, ElementwiseBinary):
        left_fill = computed[id(expr.left)]
        right_fill = computed[id(expr.right)]
        if sparse_scale_operands(expr) is not None:
            return 0.0
        if left_fill is None or right_fill is None:
            return None
        return _apply_binary(expr.op, left_fill, right_fill)
    if isinstance(expr, (Transpose, Broadcast, Slice, Gather)):
        return computed[id(expr.arg)]
    if isinstance(expr, MatMul):
        # A product with an empty contraction dimension has a shaped all-zero
        # result even when both inputs use dense physical storage.  The join
        # emits no rows, so downstream pointwise operations must treat those
        # output coordinates as implicit zeros.
        if expr.left.shape[1] == 0:
            return 0.0
        left_storage = _storage_from_implicit(computed[id(expr.left)])
        right_storage = _storage_from_implicit(computed[id(expr.right)])
        if left_storage is StorageKind.DENSE and right_storage is StorageKind.DENSE:
            return None
        return 0.0
    raise AssertionError(f"unknown matrix expression: {type(expr)!r}")


def known_finite(expr: MatrixExpr) -> bool:
    """Conservative proof from owned input buffers, never a hidden query.

    Arithmetic can overflow, so only coordinate transformations inherit the
    proof. External relations are mutable and must not claim this property.
    """

    while isinstance(expr, (Transpose, Broadcast, Slice, Gather)):
        expr = expr.arg
    return isinstance(expr, Source) and expr.finite_values


def sparse_scale_operands(
    expr: ElementwiseBinary,
) -> tuple[MatrixExpr, Broadcast] | None:
    """Find sparse coordinates multiplied by a proven-finite broadcast."""

    if expr.op is not BinaryOp.MULTIPLY:
        return None
    for values, factors in ((expr.left, expr.right), (expr.right, expr.left)):
        if (
            isinstance(factors, Broadcast)
            and known_finite(factors.arg)
            and storage_kind(values) is StorageKind.SPARSE
        ):
            return values, factors
    return None


def has_sparse_source(expr: MatrixExpr) -> bool:
    """Return whether an expression references any sparse physical source."""

    key = id(expr)
    cached = _SPARSE_SOURCE_CACHE.get(key)
    if cached is not None and cached[0] is expr:
        _SPARSE_SOURCE_CACHE.move_to_end(key)
        return cached[1]

    cached_children: dict[int, bool] = {}
    for child in children(expr):
        child_key = id(child)
        child_cached = _SPARSE_SOURCE_CACHE.get(child_key)
        if child_cached is None or child_cached[0] is not child:
            break
        _SPARSE_SOURCE_CACHE.move_to_end(child_key)
        cached_children[child_key] = child_cached[1]
    else:
        result = _compute_has_sparse_source(expr, cached_children)
        _remember_sparse_source(expr, result)
        return result

    computed: dict[int, bool] = {}
    discovered: set[int] = set()
    stack: list[tuple[MatrixExpr, bool]] = [(expr, False)]
    while stack:
        node, visited = stack.pop()
        node_key = id(node)
        if node_key in computed:
            continue

        node_cached = _SPARSE_SOURCE_CACHE.get(node_key)
        if node_cached is not None and node_cached[0] is node:
            _SPARSE_SOURCE_CACHE.move_to_end(node_key)
            computed[node_key] = node_cached[1]
            continue

        if visited:
            result = _compute_has_sparse_source(node, computed)
            computed[node_key] = result
            _remember_sparse_source(node, result)
            continue

        if node_key in discovered:
            continue
        discovered.add(node_key)
        stack.append((node, True))
        for child in reversed(children(node)):
            if id(child) not in computed:
                stack.append((child, False))

    return computed[key]


def _compute_has_sparse_source(expr: MatrixExpr, computed: dict[int, bool]) -> bool:
    if isinstance(expr, Source):
        return expr.storage is StorageKind.SPARSE
    if isinstance(expr, (Unary, ScalarBinary, Transpose, Broadcast, Slice, Gather)):
        return computed[id(expr.arg)]
    if isinstance(expr, (ElementwiseBinary, MatMul)):
        return computed[id(expr.left)] or computed[id(expr.right)]
    raise AssertionError(f"unknown matrix expression: {type(expr)!r}")


def _remember_sparse_source(expr: MatrixExpr, value: bool) -> None:
    key = id(expr)
    _SPARSE_SOURCE_CACHE[key] = (expr, value)
    _SPARSE_SOURCE_CACHE.move_to_end(key)
    if len(_SPARSE_SOURCE_CACHE) > _ANALYSIS_CACHE_LIMIT:
        _SPARSE_SOURCE_CACHE.popitem(last=False)


def storage_kind(expr: MatrixExpr) -> StorageKind:
    """Infer whether the result relation may omit zero-valued coordinates."""

    return _storage_from_implicit(implicit_value(expr))


def _storage_from_implicit(value: float | None) -> StorageKind:
    if value is not None and np.isfinite(value) and value == 0.0:
        return StorageKind.SPARSE
    return StorageKind.DENSE


def to_dict(expr: MatrixExpr) -> dict[str, Any]:
    """Return a versionable, callable-free representation of a semantic plan."""

    common: dict[str, Any] = {"shape": list(expr.shape), "dtype": expr.dtype}
    if isinstance(expr, Source):
        return {
            "node": "source",
            **common,
            "relation": expr.relation,
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "finite_values": expr.finite_values,
            "storage": expr.storage.value,
        }
    if isinstance(expr, Unary):
        return {
            "node": "unary",
            **common,
            "op": expr.op.value,
            "arg": to_dict(expr.arg),
        }
    if isinstance(expr, ScalarBinary):
        return {
            "node": "scalar_binary",
            **common,
            "op": expr.op.value,
            "scalar": expr.scalar,
            "reverse": expr.reverse,
            "arg": to_dict(expr.arg),
        }
    if isinstance(expr, ElementwiseBinary):
        return {
            "node": "elementwise_binary",
            **common,
            "op": expr.op.value,
            "left": to_dict(expr.left),
            "right": to_dict(expr.right),
        }
    if isinstance(expr, Transpose):
        return {"node": "transpose", **common, "arg": to_dict(expr.arg)}
    if isinstance(expr, Broadcast):
        return {
            "node": "broadcast",
            **common,
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "arg": to_dict(expr.arg),
        }
    if isinstance(expr, Slice):
        return {
            "node": "slice",
            **common,
            "rows": [expr.row_start, expr.row_stop, expr.row_step],
            "cols": [expr.col_start, expr.col_stop, expr.col_step],
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "arg": to_dict(expr.arg),
        }
    if isinstance(expr, Gather):
        return {
            "node": "gather",
            **common,
            "axis": expr.axis,
            "map_relation": expr.map_relation,
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "arg": to_dict(expr.arg),
        }
    if isinstance(expr, MatMul):
        return {
            "node": "matmul",
            **common,
            "left": to_dict(expr.left),
            "right": to_dict(expr.right),
        }
    raise AssertionError(f"unknown matrix expression: {type(expr)!r}")


def to_dag(expr: MatrixExpr) -> dict[str, Any]:
    """Serialize an expression DAG once per identity using strict JSON values."""

    ordered = list(postorder(expr))

    identifiers = {id(node): f"n{index}" for index, node in enumerate(ordered)}
    nodes = [_dag_node(node, identifiers) for node in ordered]
    return {"root": identifiers[id(expr)], "nodes": nodes}


def children(expr: MatrixExpr) -> tuple[MatrixExpr, ...]:
    if isinstance(expr, Source):
        return ()
    if isinstance(expr, (Unary, ScalarBinary, Transpose, Broadcast, Slice, Gather)):
        return (expr.arg,)
    if isinstance(expr, (ElementwiseBinary, MatMul)):
        return (expr.left, expr.right)
    raise AssertionError(f"unknown matrix expression: {type(expr)!r}")


def postorder(
    expr: MatrixExpr,
    *,
    stop: Callable[[MatrixExpr], bool] | None = None,
) -> Iterator[MatrixExpr]:
    """Visit each operation once, after its inputs, without Python recursion.

    A queued input is not a finished input: another branch may need it first.
    Stop points are yielded without visiting their inputs, for compiled subplans.
    """
    finished: set[int] = set()
    stack = [(expr, False)]
    while stack:
        node, visited = stack.pop()
        key = id(node)
        if key in finished:
            continue
        if visited:
            finished.add(key)
            yield node
            continue
        stack.append((node, True))
        if stop is None or not stop(node):
            stack.extend((child, False) for child in reversed(children(node)))


def _dag_node(expr: MatrixExpr, identifiers: dict[int, str]) -> dict[str, Any]:
    common: dict[str, Any] = {
        "id": identifiers[id(expr)],
        "shape": list(expr.shape),
        "dtype": expr.dtype,
        "storage": storage_kind(expr).value,
    }
    if isinstance(expr, Source):
        return {
            **common,
            "node": "source",
            "relation": expr.relation,
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
        }
    if isinstance(expr, Unary):
        return {
            **common,
            "node": "unary",
            "op": expr.op.value,
            "arg": identifiers[id(expr.arg)],
        }
    if isinstance(expr, ScalarBinary):
        return {
            **common,
            "node": "scalar_binary",
            "op": expr.op.value,
            "scalar": _json_scalar(expr.scalar),
            "reverse": expr.reverse,
            "arg": identifiers[id(expr.arg)],
        }
    if isinstance(expr, ElementwiseBinary):
        return {
            **common,
            "node": "elementwise_binary",
            "op": expr.op.value,
            "left": identifiers[id(expr.left)],
            "right": identifiers[id(expr.right)],
        }
    if isinstance(expr, Transpose):
        return {
            **common,
            "node": "transpose",
            "arg": identifiers[id(expr.arg)],
        }
    if isinstance(expr, Broadcast):
        return {
            **common,
            "node": "broadcast",
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "arg": identifiers[id(expr.arg)],
        }
    if isinstance(expr, Slice):
        return {
            **common,
            "node": "slice",
            "rows": [expr.row_start, expr.row_stop, expr.row_step],
            "cols": [expr.col_start, expr.col_stop, expr.col_step],
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "arg": identifiers[id(expr.arg)],
        }
    if isinstance(expr, Gather):
        return {
            **common,
            "node": "gather",
            "axis": expr.axis,
            "map_relation": expr.map_relation,
            "rows_relation": expr.rows_relation,
            "cols_relation": expr.cols_relation,
            "arg": identifiers[id(expr.arg)],
        }
    if isinstance(expr, MatMul):
        return {
            **common,
            "node": "matmul",
            "left": identifiers[id(expr.left)],
            "right": identifiers[id(expr.right)],
        }
    raise AssertionError(f"unknown matrix expression: {type(expr)!r}")


def _json_scalar(value: Scalar) -> Scalar | dict[str, str]:
    if isinstance(value, (bool, int)):
        return value
    numeric = float(value)
    if np.isnan(numeric):
        return {"special_float": "nan"}
    if np.isposinf(numeric):
        return {"special_float": "+inf"}
    if np.isneginf(numeric):
        return {"special_float": "-inf"}
    return numeric
