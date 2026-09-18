"""Lower DBVerse matrix semantics through the public Ibis expression API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar, cast

import ibis
import ibis.expr.types as ir
import sqlglot
from sqlglot import expressions as sge

from dbnumpy.ir import (
    COMPARISON_OPS,
    BinaryOp,
    Broadcast,
    ElementwiseBinary,
    Gather,
    MatMul,
    MatrixExpr,
    ReductionOp,
    ScalarBinary,
    Slice,
    Source,
    StorageKind,
    Transpose,
    Unary,
    UnaryOp,
    expression_depth,
    implicit_value,
    known_finite,
    postorder,
    relation_names,
    single_pointwise_source,
    sparse_scale_operands,
    storage_kind,
)
from dbnumpy.ir import (
    children as expression_children,
)
from dbnumpy.lowering.pointwise import pointwise_value
from dbnumpy.lowering.pointwise_sql import PointwiseSQLCompiler

type Axis = int | None

_VARIANCE_REDUCTIONS = frozenset(
    {
        ReductionOp.VAR,
        ReductionOp.STD,
        ReductionOp.NANVAR,
        ReductionOp.NANSTD,
    }
)


@dataclass(slots=True)
class _NameAllocator:
    """Allocate deterministic, collision-free identifiers for one SQL plan."""

    forbidden: set[str]
    next_id: int = 0
    compiled: dict[tuple[int, str], tuple[MatrixExpr, str]] = field(
        default_factory=dict
    )

    @classmethod
    def for_expr(cls, expr: MatrixExpr) -> _NameAllocator:
        return cls(set(relation_names(expr)))

    def name(self, label: str) -> str:
        while True:
            candidate = f"__dbm_internal_{self.next_id}_{label}"
            self.next_id += 1
            if candidate not in self.forbidden:
                self.forbidden.add(candidate)
                return candidate

    def prefix(self, label: str) -> str:
        while True:
            candidate = f"__dbm_internal_{self.next_id}_{label}"
            self.next_id += 1
            if not any(name.startswith(candidate) for name in self.forbidden):
                self.forbidden.add(candidate)
                return candidate


@ibis.udf.scalar.builtin(name="floor")  # type: ignore[untyped-decorator]
def _floor_float64(value: float) -> float:
    """Declare the engine builtin without Ibis's lossy int64 result cast."""

    raise NotImplementedError


@ibis.udf.scalar.builtin(name="__dbm_integer_quotient")  # type: ignore[untyped-decorator]
def _integer_quotient(left: int, right: int) -> int:
    """Compiler marker for exact integer division; never a runtime UDF."""

    raise NotImplementedError


def _float_literal(value: float) -> ir.Value:
    # Ibis's DataFusion literal writer loses the sign of negative infinity.
    return (
        -ibis.literal(float("inf"))
        if value == float("-inf")
        else ibis.literal(value, type="float64")
    )


def _relational_sql(table: ir.Table, *, dialect: str) -> str:
    sql = cast(str, ibis.to_sql(table, dialect=dialect))
    if "__DBM_INTEGER_QUOTIENT(" not in sql.upper():
        return sql

    def integer_division(node: sge.Expression) -> sge.Expression:
        if (
            isinstance(node, sge.Anonymous)
            and node.name.lower() == "__dbm_integer_quotient"
        ):
            left, right = node.expressions
            # DuckDB / always produces a double, whereas DataFusion / on
            # integer operands is integer division. Keep both operands int64.
            cls = sge.IntDiv if dialect == "duckdb" else sge.Div
            return cls(
                this=sge.Paren(this=left.copy()),
                expression=sge.Paren(this=right.copy()),
                typed=True,
            )
        return node

    # This narrow rewrite uses SQLGlot's public AST API and only touches our
    # marker. Deep pointwise SQL remains in separately compiled CTE bindings.
    # ibis.to_sql() above registers its DataFusion SQLGlot dialect. Reuse it
    # so unrelated functions (for example isnan) keep the engine's spelling.
    return (
        sqlglot.parse_one(sql, read=dialect)
        .transform(integer_division)
        .sql(dialect=dialect)
    )


@ibis.udf.scalar.builtin(name="ceil")  # type: ignore[untyped-decorator]
def _ceil_float64(value: float) -> float:
    """Declare the engine builtin without Ibis's lossy int64 result cast."""

    raise NotImplementedError


@ibis.udf.scalar.builtin(name="sinh")  # type: ignore[untyped-decorator]
def _sinh_float64(value: float) -> float:
    """Declare the shared DuckDB/DataFusion hyperbolic sine builtin."""

    raise NotImplementedError


@ibis.udf.scalar.builtin(name="cosh")  # type: ignore[untyped-decorator]
def _cosh_float64(value: float) -> float:
    """Declare the shared DuckDB/DataFusion hyperbolic cosine builtin."""

    raise NotImplementedError


@ibis.udf.scalar.builtin(name="tanh")  # type: ignore[untyped-decorator]
def _tanh_float64(value: float) -> float:
    """Declare the shared DuckDB/DataFusion hyperbolic tangent builtin."""

    raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LoweredMatrix:
    table: ir.Table
    storage: StorageKind


@dataclass(frozen=True, slots=True)
class LoweredReduction:
    table: ir.Table
    axis: Axis
    output_length: int | None
    default_value: float | bool


class IbisLowerer:
    """Translate semantic nodes to ordinary relational Ibis expressions.

    Only documented table/value operations are used here.  DBVerse does not
    subclass or inspect Ibis's internal operation nodes.
    """

    _schema: ClassVar[dict[str, str]] = {
        "i": "int64",
        "j": "int64",
        "x": "float64",
    }

    def __init__(
        self, *, pointwise_compiler: PointwiseSQLCompiler | None = None
    ) -> None:
        self.pointwise_compiler = pointwise_compiler or PointwiseSQLCompiler()

    def lower_matrix(self, expr: MatrixExpr) -> LoweredMatrix:
        values: dict[int, LoweredMatrix] = {}
        names = _NameAllocator.for_expr(expr)
        stage_relations = expression_depth(expr) >= 32

        def lower(node: MatrixExpr) -> LoweredMatrix:
            return values[id(node)]

        for node in postorder(
            expr, stop=lambda node: single_pointwise_source(node) is not None
        ):
            result = self._lower_node(node, lower)
            if stage_relations and isinstance(
                node, (Gather, Slice, Transpose, Broadcast, ElementwiseBinary)
            ):
                result = LoweredMatrix(
                    result.table.alias(names.name("matrix_step")), result.storage
                )
            values[id(node)] = result
        return values[id(expr)]

    def _lower_node(
        self, expr: MatrixExpr, lower: Callable[[MatrixExpr], LoweredMatrix]
    ) -> LoweredMatrix:
        source = single_pointwise_source(expr)
        if source is not None:
            return self._lower_single_source_pointwise(expr, source)

        if isinstance(expr, Source):
            return LoweredMatrix(self._source(expr), expr.storage)

        if isinstance(expr, Unary):
            child = lower(expr.arg)
            table = self._enumerate_if_needed(child, expr)
            value = self._unary(expr.op, table.x)
            return LoweredMatrix(table.select("i", "j", x=value), storage_kind(expr))

        if isinstance(expr, ScalarBinary):
            child = lower(expr.arg)
            table = self._enumerate_if_needed(child, expr)
            scalar = _float_literal(expr.scalar)
            left, right = (scalar, table.x) if expr.reverse else (table.x, scalar)
            value = self._binary(expr.op, left, right)
            return LoweredMatrix(table.select("i", "j", x=value), storage_kind(expr))

        if isinstance(expr, ElementwiseBinary):
            return self._lower_elementwise(expr, lower)

        if isinstance(expr, Transpose):
            child = lower(expr.arg)
            return LoweredMatrix(
                child.table.select(i=child.table.j, j=child.table.i, x=child.table.x),
                child.storage,
            )

        if isinstance(expr, Broadcast):
            return self._lower_broadcast(expr, lower)

        if isinstance(expr, Slice):
            return self._lower_slice(expr, lower)

        if isinstance(expr, Gather):
            return self._lower_gather(expr, lower)

        if isinstance(expr, MatMul):
            return self._lower_matmul(expr, lower)

        raise AssertionError(f"unknown matrix expression: {type(expr)!r}")

    def lower_reduction(
        self,
        expr: MatrixExpr,
        op: ReductionOp,
        axis: Axis,
        *,
        ddof: float = 0,
    ) -> LoweredReduction:
        if axis not in (None, 0, 1):
            raise ValueError(f"axis must be None, 0, or 1; got {axis!r}")
        if ddof < 0:
            raise ValueError(f"ddof must be nonnegative; got {ddof}")

        table = self.lower_matrix(expr).table
        return self._lower_reduction_table(expr, table, op, axis, ddof=ddof)

    def _lower_reduction_table(
        self,
        expr: MatrixExpr,
        table: ir.Table,
        op: ReductionOp,
        axis: Axis,
        *,
        ddof: float,
    ) -> LoweredReduction:
        if op in _VARIANCE_REDUCTIONS:
            return self._lower_variance_reduction_table(
                expr, table, op, axis, ddof=ddof
            )
        if axis is None:
            count = expr.shape[0] * expr.shape[1]
            value = self._reduction_value(table.x, op, count=count, ddof=ddof)
            reduced = table.aggregate(value=value)
            return LoweredReduction(reduced, None, None, self._default(op, count, ddof))

        index_name = "j" if axis == 0 else "i"
        count = expr.shape[0] if axis == 0 else expr.shape[1]
        index = table[index_name]
        value = self._reduction_value(table.x, op, count=count, ddof=ddof)
        reduced = table.group_by(index=index).aggregate(value=value).order_by("index")
        output_length = expr.shape[1] if axis == 0 else expr.shape[0]
        return LoweredReduction(
            reduced,
            axis,
            output_length,
            self._default(op, count, ddof),
        )

    def _lower_variance_reduction_table(
        self,
        expr: MatrixExpr,
        table: ir.Table,
        op: ReductionOp,
        axis: Axis,
        *,
        ddof: float,
    ) -> LoweredReduction:
        """Lower variance through anchored, normalized centered-squares passes.

        Subtracting a finite anchor before scaling retains small spreads around
        large offsets. If that subtraction overflows, the equivalent
        difference of two bounded normalized values is used instead. The final
        statistics projection merges structural sparse zeros analytically.
        """

        index_name = None if axis is None else ("j" if axis == 0 else "i")
        count = (
            expr.shape[0] * expr.shape[1]
            if axis is None
            else (expr.shape[0] if axis == 0 else expr.shape[1])
        )
        logical_count = ibis.literal(float(count), type="float64")
        denominator_base = float(
            Decimal(count)
            - (Decimal(ddof) if isinstance(ddof, int) else Decimal.from_float(ddof))
        )
        infinity = ibis.literal(float("inf"), type="float64")
        finite = ~table.x.isnan() & (table.x != infinity) & (table.x != -infinity)
        scale_window = (
            ibis.window()
            if index_name is None
            else ibis.window(group_by=table[index_name])
        )
        scaled = table.select(
            **({} if index_name is None else {"index": table[index_name]}),
            x=table.x,
            scale=table.x.abs().max(where=finite).over(scale_window).fill_null(0.0),
            anchor=table.x.min(where=finite).over(scale_window).fill_null(0.0),
        )
        safe_scale = (scaled.scale == 0.0).ifelse(1.0, scaled.scale)
        raw_delta = scaled.x - scaled.anchor
        finite_raw_delta = (
            ~raw_delta.isnan() & (raw_delta != infinity) & (raw_delta != -infinity)
        )
        normalized_delta = finite_raw_delta.ifelse(
            raw_delta / safe_scale,
            scaled.x / safe_scale - scaled.anchor / safe_scale,
        )
        normalized = scaled.select(
            **({} if index_name is None else {"index": scaled.index}),
            x=normalized_delta,
            scale=scaled.scale,
            anchor_scaled=scaled.anchor / safe_scale,
        )
        normalized_finite = (
            ~normalized.x.isnan()
            & (normalized.x != infinity)
            & (normalized.x != -infinity)
        )
        mean_window = (
            ibis.window()
            if index_name is None
            else ibis.window(group_by=normalized.index)
        )
        with_mean = normalized.select(
            **({} if index_name is None else {"index": normalized.index}),
            x=normalized.x,
            scale=normalized.scale,
            anchor_scaled=normalized.anchor_scaled,
            mean_delta=(
                normalized.x.mean(where=normalized_finite)
                .over(mean_window)
                .fill_null(0.0)
            ),
        )
        prepared = with_mean.select(
            **({} if index_name is None else {"index": with_mean.index}),
            x=with_mean.x,
            mean_delta=with_mean.mean_delta,
            explicit_mean_scaled=with_mean.anchor_scaled + with_mean.mean_delta,
            scale=with_mean.scale,
        )
        statistic_values = self._variance_statistics(
            prepared.x,
            mean_delta_value=prepared.mean_delta,
            explicit_mean_scaled_value=prepared.explicit_mean_scaled,
            scale_value=prepared.scale,
        )
        if index_name is None:
            statistics = prepared.aggregate(**statistic_values)
            reduced = statistics.select(
                value=self._variance_from_statistics(
                    statistics,
                    count=logical_count,
                    denominator_base=denominator_base,
                    skip_nan=op in {ReductionOp.NANVAR, ReductionOp.NANSTD},
                    root=op in {ReductionOp.STD, ReductionOp.NANSTD},
                )
            )
            return LoweredReduction(reduced, None, None, self._default(op, count, ddof))

        statistics = prepared.group_by(index=prepared.index).aggregate(
            **statistic_values
        )
        reduced = statistics.select(
            index=statistics.index,
            value=self._variance_from_statistics(
                statistics,
                count=logical_count,
                denominator_base=denominator_base,
                skip_nan=op in {ReductionOp.NANVAR, ReductionOp.NANSTD},
                root=op in {ReductionOp.STD, ReductionOp.NANSTD},
            ),
        ).order_by("index")
        output_length = expr.shape[1] if axis == 0 else expr.shape[0]
        return LoweredReduction(
            reduced,
            axis,
            output_length,
            self._default(op, count, ddof),
        )

    def compile_matrix(self, expr: MatrixExpr, *, dialect: str) -> str:
        return self._compile_matrix(
            expr,
            dialect=dialect,
            names=_NameAllocator.for_expr(expr),
        )

    def _compile_matrix(
        self,
        expr: MatrixExpr,
        *,
        dialect: str,
        names: _NameAllocator,
    ) -> str:
        key = (id(expr), dialect)
        cached = names.compiled.get(key)
        if cached is not None and cached[0] is expr:
            return cached[1]
        sql = self._compile_matrix_uncached(expr, dialect=dialect, names=names)
        names.compiled[key] = (expr, sql)
        return sql

    def _compile_matrix_uncached(
        self,
        expr: MatrixExpr,
        *,
        dialect: str,
        names: _NameAllocator,
    ) -> str:
        if single_pointwise_source(expr) is not None:
            return self.pointwise_compiler.compile(
                expr,
                dialect=dialect,
                name_prefix=names.prefix("pointwise"),
            )
        local_base = self._local_pointwise_base(expr)
        if local_base is not expr:
            return self._compile_local_pointwise(
                expr,
                local_base,
                dialect=dialect,
                names=names,
            )
        rewritten, bindings = self._extract_pointwise_children(
            expr,
            dialect=dialect,
            names=names,
        )
        body = _relational_sql(self.lower_matrix(rewritten).table, dialect=dialect)
        if not bindings:
            return body
        bindings = self._reachable_bindings(bindings, body)
        ctes = ", ".join(f'"{name}" AS ({sql})' for name, sql in bindings)
        result_name = names.name("relational_result")
        return f'WITH {ctes} SELECT * FROM ({body}) AS "{result_name}"'

    @staticmethod
    def _reachable_bindings(
        bindings: list[tuple[str, str]], body: str
    ) -> list[tuple[str, str]]:
        """Discard extracted CTEs superseded by a compiled parent fragment."""

        names = {name for name, _ in bindings}
        needed = {name for name in names if f'"{name}"' in body}
        while True:
            expanded = set(needed)
            for name, sql in bindings:
                if name in needed:
                    expanded.update(
                        candidate for candidate in names if f'"{candidate}"' in sql
                    )
            if expanded == needed:
                break
            needed = expanded
        return [(name, sql) for name, sql in bindings if name in needed]

    def compile_reduction(
        self,
        expr: MatrixExpr,
        op: ReductionOp,
        axis: Axis,
        *,
        dialect: str,
        ddof: float = 0,
    ) -> tuple[str, LoweredReduction]:
        if axis not in (None, 0, 1):
            raise ValueError(f"axis must be None, 0, or 1; got {axis!r}")
        if ddof < 0:
            raise ValueError(f"ddof must be nonnegative; got {ddof}")
        names = _NameAllocator.for_expr(expr)
        cte_name = names.name("matrix_reduction")
        table = ibis.table(self._schema, name=cte_name)
        lowered = self._lower_reduction_table(
            expr,
            table,
            op,
            axis,
            ddof=ddof,
        )
        reduction_sql = cast(str, ibis.to_sql(lowered.table, dialect=dialect))
        matrix_sql = self._compile_matrix(expr, dialect=dialect, names=names)
        sql = f'WITH "{cte_name}" AS ({matrix_sql}) {reduction_sql}'
        return sql, lowered

    def _compile_local_pointwise(
        self,
        expr: MatrixExpr,
        base: MatrixExpr,
        *,
        dialect: str,
        names: _NameAllocator,
    ) -> str:
        base_name = names.name("local_pointwise_base")
        rows_name = names.name("local_pointwise_rows")
        cols_name = names.name("local_pointwise_cols")
        synthetic = Source(
            base_name,
            rows_name,
            cols_name,
            base.shape,
            base.dtype,
            storage_kind(base),
            finite_values=known_finite(base),
        )
        rewritten = self._replace_subexpression(expr, base, synthetic)
        pointwise_sql = self.pointwise_compiler.compile(
            rewritten,
            dialect=dialect,
            name_prefix=names.prefix("pointwise"),
        )
        bindings = [
            (base_name, self._compile_matrix(base, dialect=dialect, names=names))
        ]
        if (
            storage_kind(base) is StorageKind.SPARSE
            and storage_kind(expr) is StorageKind.DENSE
        ):
            rows, cols = self._dimension_axes(base)
            bindings.extend(
                [
                    (rows_name, cast(str, ibis.to_sql(rows, dialect=dialect))),
                    (cols_name, cast(str, ibis.to_sql(cols, dialect=dialect))),
                ]
            )
        ctes = ", ".join(f'"{name}" AS ({sql})' for name, sql in bindings)
        result_name = names.name("local_pointwise_result")
        return f'WITH {ctes} SELECT * FROM ({pointwise_sql}) AS "{result_name}"'

    def _local_pointwise_base(self, expr: MatrixExpr) -> MatrixExpr:
        bases: dict[int, MatrixExpr] = {}
        for node in postorder(expr):
            if isinstance(node, Source):
                bases[id(node)] = node
            elif isinstance(node, (Unary, ScalarBinary)):
                bases[id(node)] = bases[id(node.arg)]
            elif isinstance(node, ElementwiseBinary):
                left_base = bases[id(node.left)]
                right_base = bases[id(node.right)]
                bases[id(node)] = (
                    left_base if self._same_expr(left_base, right_base) else node
                )
            else:
                bases[id(node)] = node
        return bases[id(expr)]

    def _replace_subexpression(
        self,
        expr: MatrixExpr,
        target: MatrixExpr,
        replacement: MatrixExpr,
    ) -> MatrixExpr:
        transformed: dict[int, MatrixExpr] = {}
        for node in postorder(expr, stop=lambda node: self._same_expr(node, target)):
            if self._same_expr(node, target):
                transformed[id(node)] = replacement
            else:
                transformed[id(node)] = self._replace_children(
                    node,
                    tuple(
                        transformed[id(child)] for child in expression_children(node)
                    ),
                )
        return transformed[id(expr)]

    def _extract_pointwise_children(
        self,
        expr: MatrixExpr,
        *,
        dialect: str,
        names: _NameAllocator,
    ) -> tuple[MatrixExpr, list[tuple[str, str]]]:
        """Replace pointwise children of relational nodes with compiled CTEs."""

        local_bases: dict[int, MatrixExpr] = {}
        transformed: dict[int, MatrixExpr] = {}
        replacements: dict[int, Source] = {}
        bindings: list[tuple[str, str]] = []

        def replacement(node: MatrixExpr) -> Source:
            key = id(node)
            existing = replacements.get(key)
            if existing is not None:
                return existing
            name = names.name("pointwise_subplan_values")
            rows_name = names.name("pointwise_subplan_rows")
            cols_name = names.name("pointwise_subplan_cols")
            if storage_kind(node) is StorageKind.SPARSE:
                dimension_base = local_bases[key]
                rows, cols = self._dimension_axes(dimension_base)
                bindings.extend(
                    [
                        (rows_name, cast(str, ibis.to_sql(rows, dialect=dialect))),
                        (cols_name, cast(str, ibis.to_sql(cols, dialect=dialect))),
                    ]
                )
            bindings.append(
                (name, self._compile_matrix(node, dialect=dialect, names=names))
            )
            result = Source(
                name,
                rows_name,
                cols_name,
                node.shape,
                node.dtype,
                storage_kind(node),
                finite_values=known_finite(node),
            )
            replacements[key] = result
            return result

        for node in postorder(expr):
            key = id(node)
            if isinstance(node, Source):
                local_bases[key] = node
                transformed[key] = node
                continue

            children = expression_children(node)
            if isinstance(node, (Unary, ScalarBinary)):
                local_bases[key] = local_bases[id(node.arg)]
            elif isinstance(node, ElementwiseBinary):
                left_base = local_bases[id(node.left)]
                right_base = local_bases[id(node.right)]
                local_bases[key] = (
                    left_base if self._same_expr(left_base, right_base) else node
                )
            else:
                local_bases[key] = node

            parent_is_pipeline = local_bases[key] is not node
            rewritten_children = tuple(
                replacement(child)
                if (
                    not parent_is_pipeline
                    and not isinstance(child, Source)
                    and local_bases[id(child)] is not child
                )
                else transformed[id(child)]
                for child in children
            )
            transformed[key] = self._replace_children(node, rewritten_children)

        return transformed[id(expr)], bindings

    @staticmethod
    def _replace_children(
        expr: MatrixExpr, children: tuple[MatrixExpr, ...]
    ) -> MatrixExpr:
        if isinstance(expr, Source):
            return expr
        if isinstance(expr, Unary):
            return Unary(expr.op, children[0])
        if isinstance(expr, ScalarBinary):
            return ScalarBinary(expr.op, children[0], expr.scalar, expr.reverse)
        if isinstance(expr, ElementwiseBinary):
            return ElementwiseBinary(expr.op, children[0], children[1])
        if isinstance(expr, Transpose):
            return Transpose(children[0])
        if isinstance(expr, Broadcast):
            return Broadcast(
                children[0], expr.target_shape, expr.rows_relation, expr.cols_relation
            )
        if isinstance(expr, Slice):
            return Slice(
                children[0],
                expr.row_start,
                expr.row_stop,
                expr.row_step,
                expr.col_start,
                expr.col_stop,
                expr.col_step,
                expr.slice_shape,
                expr.rows_relation,
                expr.cols_relation,
            )
        if isinstance(expr, Gather):
            return Gather(
                children[0],
                expr.axis,
                expr.map_relation,
                expr.gather_shape,
                expr.rows_relation,
                expr.cols_relation,
            )
        if isinstance(expr, MatMul):
            return MatMul(children[0], children[1])
        raise AssertionError(f"unknown matrix expression: {type(expr)!r}")

    def _source(self, source: Source) -> ir.Table:
        table = ibis.table(self._schema, name=source.relation)
        return table.select(
            table.i,
            table.j,
            x=table.x.fill_null(ibis.literal(float("nan"), type="float64")),
        )

    def _dimension_grid(self, expr: MatrixExpr) -> ir.Table:
        rows, cols = self._dimension_axes(expr)
        return rows.cross_join(cols).select(rows.i, cols.j)

    def _dimension_axes(self, expr: MatrixExpr) -> tuple[ir.Table, ir.Table]:
        if isinstance(expr, Source):
            rows = ibis.table({"i": "int64"}, name=expr.rows_relation)
            cols = ibis.table({"j": "int64"}, name=expr.cols_relation)
            return rows, cols
        if isinstance(expr, (Unary, ScalarBinary)):
            return self._dimension_axes(expr.arg)
        if isinstance(expr, ElementwiseBinary):
            return self._dimension_axes(expr.left)
        if isinstance(expr, Transpose):
            child_rows, child_cols = self._dimension_axes(expr.arg)
            rows = child_cols.select(i=child_cols.j)
            cols = child_rows.select(j=child_rows.i)
            return rows, cols
        if isinstance(expr, Broadcast):
            rows = ibis.table({"i": "int64"}, name=expr.rows_relation)
            cols = ibis.table({"j": "int64"}, name=expr.cols_relation)
            return rows, cols
        if isinstance(expr, Slice):
            rows = ibis.table({"i": "int64"}, name=expr.rows_relation)
            cols = ibis.table({"j": "int64"}, name=expr.cols_relation)
            return rows, cols
        if isinstance(expr, Gather):
            rows = ibis.table({"i": "int64"}, name=expr.rows_relation)
            cols = ibis.table({"j": "int64"}, name=expr.cols_relation)
            return rows, cols
        if isinstance(expr, MatMul):
            rows, _ = self._dimension_axes(expr.left)
            _, cols = self._dimension_axes(expr.right)
            return rows, cols
        raise AssertionError(f"unknown matrix expression: {type(expr)!r}")

    def _lower_single_source_pointwise(
        self, expr: MatrixExpr, source: Source
    ) -> LoweredMatrix:
        values = self._source(source)
        result_storage = storage_kind(expr)

        if source.storage is StorageKind.SPARSE and result_storage is StorageKind.DENSE:
            grid = self._dimension_grid(source)
            joined = grid.left_join(
                values,
                [grid.i == values.i, grid.j == values.j],
            )
            base = values.x.fill_null(0.0)
            value = self._pointwise_value(expr, source, base)
            table = joined.select(grid.i, grid.j, x=value)
        else:
            value = self._pointwise_value(expr, source, values.x)
            table = values.select(values.i, values.j, x=value)

        return LoweredMatrix(table, result_storage)

    def _pointwise_value(
        self,
        expr: MatrixExpr,
        source: Source,
        value: ir.Value,
    ) -> ir.Value:
        return pointwise_value(
            expr,
            source,
            base=value,
            literal=_float_literal,
            unary=self._unary,
            binary=self._binary,
        )

    def _enumerate_if_needed(
        self, child: LoweredMatrix, result: MatrixExpr
    ) -> ir.Table:
        result_storage = storage_kind(result)
        if (
            child.storage is not StorageKind.SPARSE
            or result_storage is StorageKind.SPARSE
        ):
            return child.table
        grid = self._dimension_grid(result)
        values = child.table
        joined = grid.left_join(
            values,
            [grid.i == values.i, grid.j == values.j],
        )
        fill = implicit_value(self._child_expr(result))
        if fill is None:
            raise AssertionError("dense child unexpectedly required enumeration")
        return joined.select(grid.i, grid.j, x=values.x.fill_null(fill))

    def _lower_elementwise(
        self, expr: ElementwiseBinary, lower: Callable[[MatrixExpr], LoweredMatrix]
    ) -> LoweredMatrix:
        scaling = sparse_scale_operands(expr)
        if scaling is not None:
            values_expr, factors_expr = scaling
            values = lower(values_expr).table
            factors = lower(factors_expr.arg).table
            predicates = []
            if factors_expr.arg.shape[0] != 1:
                predicates.append(values.i == factors.i)
            if factors_expr.arg.shape[1] != 1:
                predicates.append(values.j == factors.j)
            # Missing sparse factors are zero. The finite proof is necessary:
            # multiplying an absent matrix coordinate by Inf/NaN is not zero.
            if not predicates:
                factors = factors.aggregate(x=factors.x.sum().fill_null(0.0))
                joined = values.cross_join(factors)
            else:
                joined = values.left_join(factors, predicates)
            factor_value = factors.x.fill_null(0.0)
            left, right = (
                (values.x, factor_value)
                if values_expr is expr.left
                else (factor_value, values.x)
            )
            return LoweredMatrix(
                joined.select(values.i, values.j, x=self._binary(expr.op, left, right)),
                StorageKind.SPARSE,
            )
        if self._same_expr(expr.left, expr.right):
            child = lower(expr.left)
            table = self._enumerate_if_needed(child, expr)
            value = self._binary(expr.op, table.x, table.x)
            return LoweredMatrix(
                table.select(table.i, table.j, x=value), storage_kind(expr)
            )

        left = lower(expr.left)
        right = lower(expr.right)
        # Overlapping branches can contain the same relations. Give the right
        # operand its own reference so Ibis binds each join predicate correctly.
        left = LoweredMatrix(left.table.view(), left.storage)
        right = LoweredMatrix(right.table.view(), right.storage)
        result_storage = storage_kind(expr)

        if result_storage is StorageKind.DENSE:
            left_table = left.table
            right_table = right.table
            if left.storage is StorageKind.DENSE and right.storage is StorageKind.DENSE:
                joined = left_table.inner_join(
                    right_table,
                    [
                        left_table.i == right_table.i,
                        left_table.j == right_table.j,
                    ],
                )
                value = self._binary(expr.op, left_table.x, right_table.x)
                table = joined.select(left_table.i, left_table.j, x=value)
                return LoweredMatrix(table, result_storage)

            if left.storage is StorageKind.DENSE:
                joined = left_table.left_join(
                    right_table,
                    [
                        left_table.i == right_table.i,
                        left_table.j == right_table.j,
                    ],
                )
                right_value = self._filled(right_table.x, expr.right)
                value = self._binary(expr.op, left_table.x, right_value)
                table = joined.select(left_table.i, left_table.j, x=value)
                return LoweredMatrix(table, result_storage)

            if right.storage is StorageKind.DENSE:
                joined = right_table.left_join(
                    left_table,
                    [
                        right_table.i == left_table.i,
                        right_table.j == left_table.j,
                    ],
                )
                left_value = self._filled(left_table.x, expr.left)
                value = self._binary(expr.op, left_value, right_table.x)
                table = joined.select(right_table.i, right_table.j, x=value)
                return LoweredMatrix(table, result_storage)

            grid = self._dimension_grid(expr)
            with_left = grid.left_join(
                left_table,
                [grid.i == left_table.i, grid.j == left_table.j],
            ).select(grid.i, grid.j, left_x=left_table.x)
            joined = with_left.left_join(
                right_table,
                [
                    with_left.i == right_table.i,
                    with_left.j == right_table.j,
                ],
            )
            left_value = self._filled(with_left.left_x, expr.left)
            right_value = self._filled(right_table.x, expr.right)
            value = self._binary(expr.op, left_value, right_value)
            table = joined.select(with_left.i, with_left.j, x=value)
            return LoweredMatrix(table, result_storage)

        left_table = left.table
        right_table = right.table
        if left.storage is StorageKind.DENSE:
            joined = left_table.left_join(
                right_table,
                [left_table.i == right_table.i, left_table.j == right_table.j],
            )
            index_i, index_j = left_table.i, left_table.j
        elif right.storage is StorageKind.DENSE:
            joined = right_table.left_join(
                left_table,
                [right_table.i == left_table.i, right_table.j == left_table.j],
            )
            index_i, index_j = right_table.i, right_table.j
        else:
            joined = left_table.outer_join(
                right_table,
                [left_table.i == right_table.i, left_table.j == right_table.j],
            )
            index_i = ibis.coalesce(left_table.i, right_table.i)
            index_j = ibis.coalesce(left_table.j, right_table.j)

        left_value = self._filled(left_table.x, expr.left)
        right_value = self._filled(right_table.x, expr.right)
        value = self._binary(expr.op, left_value, right_value)
        return LoweredMatrix(
            joined.select(i=index_i, j=index_j, x=value), result_storage
        )

    def _lower_matmul(
        self, expr: MatMul, lower: Callable[[MatrixExpr], LoweredMatrix]
    ) -> LoweredMatrix:
        left = lower(expr.left).table
        right = lower(expr.right).table.view()
        joined = left.inner_join(right, left.j == right.i)
        products = joined.select(i=left.i, j=right.j, product=left.x * right.x)
        result = products.group_by("i", "j").aggregate(x=products.product.sum())
        return LoweredMatrix(result, storage_kind(expr))

    def _lower_broadcast(
        self, expr: Broadcast, lower: Callable[[MatrixExpr], LoweredMatrix]
    ) -> LoweredMatrix:
        child = lower(expr.arg)
        expand_rows = expr.arg.shape[0] == 1 and expr.shape[0] != 1
        expand_cols = expr.arg.shape[1] == 1 and expr.shape[1] != 1
        rows, cols = self._dimension_axes(expr)

        if expand_rows and expand_cols:
            grid = rows.cross_join(cols).select(rows.i, cols.j)
            joined = grid.cross_join(child.table)
            table = joined.select(grid.i, grid.j, x=child.table.x)
        elif expand_rows:
            joined = rows.cross_join(child.table)
            table = joined.select(rows.i, child.table.j, x=child.table.x)
        elif expand_cols:
            joined = child.table.cross_join(cols)
            table = joined.select(child.table.i, cols.j, x=child.table.x)
        else:
            table = child.table
        return LoweredMatrix(table, child.storage)

    def _lower_slice(
        self, expr: Slice, lower: Callable[[MatrixExpr], LoweredMatrix]
    ) -> LoweredMatrix:
        child = lower(expr.arg)
        table = child.table
        row_offset, row_predicate = self._slice_axis(
            table.i, expr.row_start, expr.row_stop, expr.row_step
        )
        col_offset, col_predicate = self._slice_axis(
            table.j, expr.col_start, expr.col_stop, expr.col_step
        )
        predicate = row_predicate & col_predicate
        sliced = table.filter(predicate).select(
            i=(
                ibis.literal(0, type="int64")
                if abs(expr.row_step) >= expr.arg.shape[0]
                else _integer_quotient(row_offset, ibis.literal(abs(expr.row_step)))
            ),
            j=(
                ibis.literal(0, type="int64")
                if abs(expr.col_step) >= expr.arg.shape[1]
                else _integer_quotient(col_offset, ibis.literal(abs(expr.col_step)))
            ),
            x=table.x,
        )
        return LoweredMatrix(sliced, child.storage)

    @staticmethod
    def _slice_axis(
        coordinate: ir.Value, start: int, stop: int, step: int
    ) -> tuple[ir.Value, ir.BooleanValue]:
        if abs(step) > 2**63 - 1:
            nonempty = start < stop if step > 0 else start > stop
            return ibis.literal(0, type="int64"), (
                (coordinate == start) & ibis.literal(nonempty)
            )
        if step > 0:
            offset = coordinate - start
            predicate = (
                (coordinate >= start) & (coordinate < stop) & ((offset % step) == 0)
            )
            return offset, predicate
        offset = start - coordinate
        magnitude = -step
        predicate = (
            (coordinate <= start) & (coordinate > stop) & ((offset % magnitude) == 0)
        )
        return offset, predicate

    def _lower_gather(
        self, expr: Gather, lower: Callable[[MatrixExpr], LoweredMatrix]
    ) -> LoweredMatrix:
        child = lower(expr.arg)
        table = child.table
        mapping = ibis.table(
            {"source_index": "int64", "output_index": "int64"},
            name=expr.map_relation,
        )
        coordinate = table.i if expr.axis == 0 else table.j
        joined = mapping.inner_join(table, mapping.source_index == coordinate)
        if expr.axis == 0:
            table = joined.select(i=mapping.output_index, j=table.j, x=table.x)
        else:
            table = joined.select(i=table.i, j=mapping.output_index, x=table.x)
        return LoweredMatrix(table, child.storage)

    @staticmethod
    def _unary(op: UnaryOp, value: ir.Value) -> ir.Value:
        if op is UnaryOp.NEGATIVE:
            return -value
        if op is UnaryOp.ABSOLUTE:
            return value.abs()
        if op is UnaryOp.SQRT:
            negative_nonzero = (value < 0.0) & ~IbisLowerer._is_zero(value)
            return negative_nonzero.ifelse(
                ibis.literal(float("nan"), type="float64"), value.sqrt()
            )
        if op is UnaryOp.EXP:
            return value.exp()
        if op is UnaryOp.EXPM1:
            series = value * (
                1.0
                + value
                * (
                    0.5
                    + value * (1.0 / 6.0 + value * (1.0 / 24.0 + value * (1.0 / 120.0)))
                )
            )
            return (value.abs() < 1.0e-5).ifelse(series, value.exp() - 1.0)
        if op is UnaryOp.LOG:
            is_zero = IbisLowerer._is_zero(value)
            negative_nonzero = (value < 0.0) & ~is_zero
            return negative_nonzero.ifelse(
                ibis.literal(float("nan"), type="float64"),
                is_zero.ifelse(-ibis.literal(float("inf"), type="float64"), value.ln()),
            )
        if op is UnaryOp.LOG1P:
            series = value * (
                1.0
                + value * (-0.5 + value * (1.0 / 3.0 + value * (-0.25 + value * 0.2)))
            )
            return (value < -1.0).ifelse(
                ibis.literal(float("nan"), type="float64"),
                (value == -1.0).ifelse(
                    -ibis.literal(float("inf"), type="float64"),
                    (value.abs() < 1.0e-4).ifelse(series, (value + 1.0).ln()),
                ),
            )
        if op is UnaryOp.SIN:
            infinity = ibis.literal(float("inf"), type="float64")
            return ((value == infinity) | (value == -infinity)).ifelse(
                ibis.literal(float("nan"), type="float64"), value.sin()
            )
        if op is UnaryOp.COS:
            infinity = ibis.literal(float("inf"), type="float64")
            return ((value == infinity) | (value == -infinity)).ifelse(
                ibis.literal(float("nan"), type="float64"), value.cos()
            )
        if op is UnaryOp.TAN:
            infinity = ibis.literal(float("inf"), type="float64")
            return ((value == infinity) | (value == -infinity)).ifelse(
                ibis.literal(float("nan"), type="float64"), value.tan()
            )
        if op is UnaryOp.FLOOR:
            return _floor_float64(value)
        if op is UnaryOp.CEIL:
            return _ceil_float64(value)
        if op is UnaryOp.SIGN:
            return value.isnan().ifelse(
                value,
                IbisLowerer._is_zero(value).ifelse(
                    ibis.literal(0.0),
                    (value > 0.0).ifelse(ibis.literal(1.0), ibis.literal(-1.0)),
                ),
            )
        if op is UnaryOp.TRUNC:
            keep_input = value.isnan() | IbisLowerer._is_zero(value)
            truncated = (value < 0.0).ifelse(
                _ceil_float64(value), _floor_float64(value)
            )
            return keep_input.ifelse(value, truncated)
        if op is UnaryOp.ISNAN:
            return value.isnan().cast("float64")
        if op in {UnaryOp.LOG2, UnaryOp.LOG10}:
            is_zero = IbisLowerer._is_zero(value)
            negative_nonzero = (value < 0.0) & ~is_zero
            base = 2 if op is UnaryOp.LOG2 else 10
            logarithm = value.log(base)
            return value.isnan().ifelse(
                value,
                negative_nonzero.ifelse(
                    ibis.literal(float("nan"), type="float64"),
                    is_zero.ifelse(
                        -ibis.literal(float("inf"), type="float64"), logarithm
                    ),
                ),
            )
        if op is UnaryOp.SINH:
            return _sinh_float64(value)
        if op is UnaryOp.COSH:
            return _cosh_float64(value)
        if op is UnaryOp.TANH:
            return _tanh_float64(value)
        if op in {UnaryOp.ARCSIN, UnaryOp.ARCCOS}:
            function = value.asin if op is UnaryOp.ARCSIN else value.acos
            outside_domain = (value < -1.0) | (value > 1.0)
            return outside_domain.ifelse(
                ibis.literal(float("nan"), type="float64"), function()
            )
        if op is UnaryOp.ARCTAN:
            return value.atan()
        raise AssertionError(f"unsupported unary operation: {op}")

    @staticmethod
    def _binary(op: BinaryOp, left: ir.Value, right: ir.Value) -> ir.Value:
        if op is BinaryOp.ADD:
            result = left + right
        elif op is BinaryOp.SUBTRACT:
            result = left - right
        elif op is BinaryOp.MULTIPLY:
            result = left * right
        elif op is BinaryOp.TRUE_DIVIDE:
            result = left / right
        elif op is BinaryOp.POWER:
            negative_infinity = -ibis.literal(float("inf"), type="float64")
            is_zero = (left == 0.0) | ((1.0 / left) == negative_infinity)
            unsafe = is_zero & (right < 0.0)
            square_root_zero = is_zero & (right == 0.5)
            negative_zero = (1.0 / left) < 0.0
            odd_integer = (right == _floor_float64(right)) & (
                (right % 2.0).abs() == 1.0
            )
            signed_infinity = (negative_zero & odd_integer).ifelse(
                negative_infinity,
                ibis.literal(float("inf"), type="float64"),
            )
            safe_left = unsafe.ifelse(ibis.literal(1.0), left)
            result = unsafe.ifelse(
                signed_infinity,
                square_root_zero.ifelse(left, safe_left**right),
            )
        elif op is BinaryOp.GREATER:
            zero_pair = IbisLowerer._zero_pair(left, right)
            result = ~(left.isnan() | right.isnan() | zero_pair) & (left > right)
        elif op is BinaryOp.GREATER_EQUAL:
            zero_pair = IbisLowerer._zero_pair(left, right)
            result = ~(left.isnan() | right.isnan()) & (zero_pair | (left >= right))
        elif op is BinaryOp.LESS:
            zero_pair = IbisLowerer._zero_pair(left, right)
            result = ~(left.isnan() | right.isnan() | zero_pair) & (left < right)
        elif op is BinaryOp.LESS_EQUAL:
            zero_pair = IbisLowerer._zero_pair(left, right)
            result = ~(left.isnan() | right.isnan()) & (zero_pair | (left <= right))
        elif op is BinaryOp.EQUAL:
            zero_pair = IbisLowerer._zero_pair(left, right)
            result = ~(left.isnan() | right.isnan()) & (zero_pair | (left == right))
        elif op is BinaryOp.NOT_EQUAL:
            zero_pair = IbisLowerer._zero_pair(left, right)
            result = left.isnan() | right.isnan() | (~zero_pair & (left != right))
        else:
            raise AssertionError(f"unsupported binary operation: {op}")
        return result.cast("float64") if op in COMPARISON_OPS else result

    @staticmethod
    def _zero_pair(left: ir.Value, right: ir.Value) -> ir.Value:
        return IbisLowerer._is_zero(left) & IbisLowerer._is_zero(right)

    @staticmethod
    def _is_zero(value: ir.Value) -> ir.Value:
        negative_infinity = -ibis.literal(float("inf"), type="float64")
        return (value == 0.0) | ((1.0 / value) == negative_infinity)

    @staticmethod
    def _filled(value: ir.Value, expr: MatrixExpr) -> ir.Value:
        fill = implicit_value(expr)
        return value if fill is None else value.fill_null(fill)

    @staticmethod
    def _child_expr(expr: MatrixExpr) -> MatrixExpr:
        if isinstance(expr, (Unary, ScalarBinary, Transpose)):
            return expr.arg
        if isinstance(expr, ElementwiseBinary) and IbisLowerer._same_expr(
            expr.left, expr.right
        ):
            return expr.left
        raise AssertionError(f"expression has no single child: {type(expr)!r}")

    @staticmethod
    def _same_expr(left: MatrixExpr, right: MatrixExpr) -> bool:
        return left is right or left == right

    @staticmethod
    def _reduction_value(
        value: ir.Value,
        op: ReductionOp,
        *,
        count: int,
        ddof: float,
    ) -> ir.Scalar:
        # A valid pair of int64 dimensions can have a product larger than an
        # engine int64. Reductions only need that logical count in floating
        # arithmetic, whose range covers the complete shape contract.
        logical_count = ibis.literal(float(count), type="float64")
        total = value.sum()
        if op is ReductionOp.SUM:
            return total
        if op is ReductionOp.MEAN:
            return total / logical_count
        if op is ReductionOp.NANSUM:
            return value.sum(where=~value.isnan()).fill_null(0.0)
        if op is ReductionOp.NANMEAN:
            nan_count = value.count(where=value.isnan()).cast("float64")
            effective_count = logical_count - nan_count
            nan_total = value.sum(where=~value.isnan()).fill_null(0.0)
            return (effective_count <= 0).ifelse(
                ibis.literal(float("nan"), type="float64"),
                nan_total / effective_count,
            )
        if op in {ReductionOp.MIN, ReductionOp.MAX}:
            return IbisLowerer._extreme(value, op, count=logical_count, skip_nan=False)
        if op in {ReductionOp.NANMIN, ReductionOp.NANMAX}:
            return IbisLowerer._extreme(value, op, count=logical_count, skip_nan=True)
        if op is ReductionOp.ANY:
            truthy = value.isnan() | ~IbisLowerer._is_zero(value)
            return truthy.any().fill_null(False)
        if op is ReductionOp.ALL:
            explicit_count = value.count().cast("float64")
            has_implicit_zero = explicit_count < logical_count
            truthy = value.isnan() | ~IbisLowerer._is_zero(value)
            return has_implicit_zero.ifelse(False, truthy.all().fill_null(True))
        raise AssertionError(f"unsupported reduction: {op}")

    @staticmethod
    def _extreme(
        value: ir.Value,
        op: ReductionOp,
        *,
        count: ir.Scalar,
        skip_nan: bool,
    ) -> ir.Scalar:
        is_min = op in {ReductionOp.MIN, ReductionOp.NANMIN}
        explicit_count = value.count().cast("float64")
        has_implicit_zero = explicit_count < count
        valid = ~value.isnan()
        valid_count = value.count(where=valid)
        zero = ibis.literal(0.0, type="float64")
        nan = ibis.literal(float("nan"), type="float64")
        infinity = ibis.literal(float("inf"), type="float64")
        negative_infinity = -infinity
        # Ibis considers +0.0 and -0.0 literal payloads equal when interning.
        # Dividing a negative finite value by infinity keeps this expression
        # distinct and also forces DuckDB to evaluate a floating-point -0.0.
        negative_zero = -ibis.literal(1.0, type="float64") / infinity
        is_positive_infinity = value == infinity
        is_negative_infinity = value == negative_infinity
        finite = valid & ~is_positive_infinity & ~is_negative_infinity
        finite_count = value.count(where=finite)
        finite_candidate = (
            value.min(where=finite) if is_min else value.max(where=finite)
        )
        has_positive_infinity = is_positive_infinity.any().fill_null(False)
        has_negative_infinity = is_negative_infinity.any().fill_null(False)
        if is_min:
            candidate = has_negative_infinity.ifelse(
                negative_infinity,
                (finite_count > 0).ifelse(
                    finite_candidate,
                    has_positive_infinity.ifelse(infinity, nan),
                ),
            )
        else:
            candidate = has_positive_infinity.ifelse(
                infinity,
                (finite_count > 0).ifelse(
                    finite_candidate,
                    has_negative_infinity.ifelse(negative_infinity, nan),
                ),
            )
        folded = (valid_count == 0).ifelse(
            zero,
            ((candidate < 0.0) if is_min else (candidate > 0.0)).ifelse(
                candidate, zero
            ),
        )
        result = has_implicit_zero.ifelse(
            folded,
            (valid_count == 0).ifelse(nan, candidate),
        )
        is_zero = IbisLowerer._is_zero(value)
        is_negative_zero = is_zero & ((1.0 / value) == negative_infinity)
        has_negative_zero = is_negative_zero.any().fill_null(False)
        has_positive_zero = (is_zero & ~is_negative_zero).any().fill_null(
            False
        ) | has_implicit_zero
        signed_zero = (
            has_negative_zero.ifelse(negative_zero, zero)
            if is_min
            else has_positive_zero.ifelse(zero, negative_zero)
        )
        result = IbisLowerer._is_zero(result).ifelse(signed_zero, result)
        if skip_nan:
            return result
        has_nan = value.isnan().any().fill_null(False)
        return has_nan.ifelse(nan, result)

    @staticmethod
    def _variance_statistics(
        value: ir.Value,
        *,
        mean_delta_value: ir.Value,
        explicit_mean_scaled_value: ir.Value,
        scale_value: ir.Value,
    ) -> dict[str, ir.Scalar]:
        """Name each aggregate once so the final formula stays a linear plan."""

        infinity = ibis.literal(float("inf"), type="float64")
        is_nan = value.isnan()
        is_infinite = (value == infinity) | (value == -infinity)
        finite = ~is_nan & ~is_infinite
        centered = value - mean_delta_value
        return {
            "explicit_count": value.count().cast("float64"),
            "finite_count": value.count(where=finite).cast("float64"),
            "nan_count": value.count(where=is_nan).cast("float64"),
            "explicit_mean_scaled": explicit_mean_scaled_value.max().fill_null(0.0),
            "explicit_m2": (centered * centered).sum(where=finite).fill_null(0.0),
            "scale": scale_value.max().fill_null(0.0),
            "has_infinite": is_infinite.any().fill_null(False),
            "has_nan": is_nan.any().fill_null(False),
        }

    @staticmethod
    def _variance_from_statistics(
        statistics: ir.Table,
        *,
        count: ir.Scalar,
        denominator_base: float,
        skip_nan: bool,
        root: bool,
    ) -> ir.Value:
        nan = ibis.literal(float("nan"), type="float64")
        infinity = ibis.literal(float("inf"), type="float64")
        implicit_count = count - statistics.explicit_count
        effective_count = count - statistics.nan_count if skip_nan else count
        base = ibis.literal(denominator_base, type="float64")
        denominator = base - statistics.nan_count if skip_nan else base
        safe_count = (effective_count <= 0).ifelse(1.0, effective_count)
        safe_denominator = (denominator <= 0).ifelse(1.0, denominator)
        explicit_variance = statistics.explicit_m2 / safe_denominator
        between_scale = (
            (statistics.finite_count / safe_count) * (implicit_count / safe_denominator)
        ).sqrt()
        scaled_mean = statistics.explicit_mean_scaled * between_scale
        between_variance = scaled_mean * scaled_mean
        normalized_variance = explicit_variance + between_variance
        nonnegative_variance = (normalized_variance < 0.0).ifelse(
            0.0, normalized_variance
        )
        standard_deviation = statistics.scale * nonnegative_variance.sqrt()
        result = standard_deviation if root else standard_deviation * standard_deviation
        invalid_value = statistics.has_infinite
        if not skip_nan:
            invalid_value = invalid_value | statistics.has_nan
        zero_division_result = (
            nan if skip_nan else (nonnegative_variance > 0.0).ifelse(infinity, nan)
        )
        return invalid_value.ifelse(
            nan,
            (denominator <= 0).ifelse(zero_division_result, result),
        )

    @staticmethod
    def _default(op: ReductionOp, count: int, ddof: float) -> float | bool:
        if op in {ReductionOp.SUM, ReductionOp.NANSUM}:
            return 0.0
        if op in {ReductionOp.MEAN, ReductionOp.NANMEAN}:
            return 0.0 if count else float("nan")
        if op in {
            ReductionOp.MIN,
            ReductionOp.MAX,
            ReductionOp.NANMIN,
            ReductionOp.NANMAX,
        }:
            return 0.0 if count else float("nan")
        if op is ReductionOp.ANY:
            return False
        if op is ReductionOp.ALL:
            return False if count else True
        return 0.0 if count > ddof else float("nan")
