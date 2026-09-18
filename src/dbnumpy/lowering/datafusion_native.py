"""Native DataFusion lowering for the single-source pointwise hot path."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from dbnumpy.ir import (
    COMPARISON_OPS,
    BinaryOp,
    MatrixExpr,
    Source,
    StorageKind,
    UnaryOp,
    pointwise_alias_nodes,
    single_pointwise_source,
    storage_kind,
)
from dbnumpy.lowering.pointwise import pointwise_value


class DataFusionPointwiseLowerer:
    """Build a DataFusion expression without routing through its SQL parser."""

    def lower(self, expr: MatrixExpr, *, context: Any) -> Any:
        from datafusion import col

        source = single_pointwise_source(expr)
        if source is None:
            raise ValueError("native pointwise lowering requires exactly one source")

        if (
            source.storage is StorageKind.SPARSE
            and storage_kind(expr) is StorageKind.DENSE
        ):
            base_sql = (
                "SELECT r.i, c.j, CASE WHEN v.i IS NULL THEN 0.0 "
                "ELSE COALESCE(v.x, CAST('NaN' AS DOUBLE)) END AS x "
                f'FROM "{source.rows_relation}" AS r '
                f'CROSS JOIN "{source.cols_relation}" AS c '
                f'LEFT OUTER JOIN "{source.relation}" AS v '
                "ON r.i = v.i AND c.j = v.j"
            )
            frame = context.sql(base_sql)
        else:
            import datafusion.functions as f
            from datafusion import lit

            frame = context.table(source.relation).select(
                col("i"),
                col("j"),
                f.coalesce(col("x"), lit(float("nan"))).alias("x"),
            )

        aliases: dict[int, Any] = {}
        for index, node in enumerate(pointwise_alias_nodes(expr)):
            name = f"__dbm_pointwise_value_{index}"
            value = self._value(
                node,
                source,
                base=col("x"),
                aliases=aliases,
            )
            frame = frame.with_column(name, value)
            aliases[id(node)] = col(name)

        value = self._value(expr, source, base=col("x"), aliases=aliases)
        return frame.select(col("i"), col("j"), value.alias("x"))

    def _value(
        self,
        expr: MatrixExpr,
        source: Source,
        *,
        base: Any,
        aliases: dict[int, Any] | None = None,
    ) -> Any:
        from datafusion import lit

        return pointwise_value(
            expr,
            source,
            base=base,
            literal=lit,
            unary=self._unary,
            binary=self._binary,
            aliases=aliases,
        )

    @staticmethod
    def _unary(op: UnaryOp, value: Any) -> Any:
        import datafusion.functions as f
        from datafusion import lit

        functions = {
            UnaryOp.ABSOLUTE: f.abs,
            UnaryOp.SQRT: f.sqrt,
            UnaryOp.EXP: f.exp,
            UnaryOp.LOG: f.ln,
            UnaryOp.SIN: f.sin,
            UnaryOp.COS: f.cos,
            UnaryOp.TAN: f.tan,
            UnaryOp.FLOOR: f.floor,
            UnaryOp.CEIL: f.ceil,
            UnaryOp.LOG2: f.log2,
            UnaryOp.LOG10: f.log10,
            UnaryOp.SINH: f.sinh,
            UnaryOp.COSH: f.cosh,
            UnaryOp.TANH: f.tanh,
            UnaryOp.ARCSIN: f.asin,
            UnaryOp.ARCCOS: f.acos,
            UnaryOp.ARCTAN: f.atan,
        }
        if op is UnaryOp.NEGATIVE:
            return value * -1.0
        if op is UnaryOp.SQRT:
            negative_nonzero = (value < 0.0) & ~f.iszero(value)
            return f.when(negative_nonzero, lit(float("nan"))).otherwise(f.sqrt(value))
        if op is UnaryOp.EXPM1:
            series = value * (
                1.0
                + value
                * (
                    0.5
                    + value * (1.0 / 6.0 + value * (1.0 / 24.0 + value * (1.0 / 120.0)))
                )
            )
            return f.when(f.abs(value) < 1.0e-5, series).otherwise(f.exp(value) - 1.0)
        if op is UnaryOp.LOG:
            negative_nonzero = (value < 0.0) & ~f.iszero(value)
            return (
                f.when(negative_nonzero, lit(float("nan")))
                .when(f.iszero(value), lit(float("-inf")))
                .otherwise(f.ln(value))
            )
        if op is UnaryOp.LOG1P:
            series = value * (
                1.0
                + value * (-0.5 + value * (1.0 / 3.0 + value * (-0.25 + value * 0.2)))
            )
            return (
                f.when(value < -1.0, lit(float("nan")))
                .when(value == -1.0, lit(float("-inf")))
                .when(f.abs(value) < 1.0e-4, series)
                .otherwise(f.ln(value + 1.0))
            )
        if op in {UnaryOp.SIN, UnaryOp.COS, UnaryOp.TAN}:
            is_infinite = (value == lit(float("inf"))) | (value == lit(float("-inf")))
            return f.when(is_infinite, lit(float("nan"))).otherwise(
                functions[op](value)
            )
        if op is UnaryOp.SIGN:
            return (
                f.when(f.isnan(value), value)
                .when(f.iszero(value), lit(0.0))
                .when(value > 0.0, lit(1.0))
                .otherwise(lit(-1.0))
            )
        if op is UnaryOp.TRUNC:
            keep_input = f.isnan(value) | f.iszero(value)
            truncated = f.when(value < 0.0, f.ceil(value)).otherwise(f.floor(value))
            return f.when(keep_input, value).otherwise(truncated)
        if op is UnaryOp.ISNAN:
            return f.isnan(value).cast(pa.float64())
        if op in {UnaryOp.LOG2, UnaryOp.LOG10}:
            is_zero = f.iszero(value)
            negative_nonzero = (value < 0.0) & ~is_zero
            return (
                f.when(f.isnan(value), value)
                .when(negative_nonzero, lit(float("nan")))
                .when(is_zero, lit(float("-inf")))
                .otherwise(functions[op](value))
            )
        if op in {UnaryOp.ARCSIN, UnaryOp.ARCCOS}:
            outside_domain = (value < -1.0) | (value > 1.0)
            return f.when(outside_domain, lit(float("nan"))).otherwise(
                functions[op](value)
            )
        return functions[op](value)

    @staticmethod
    def _binary(op: BinaryOp, left: Any, right: Any) -> Any:
        import datafusion.functions as f

        if op is BinaryOp.ADD:
            result = left + right
        elif op is BinaryOp.SUBTRACT:
            result = left - right
        elif op is BinaryOp.MULTIPLY:
            result = left * right
        elif op is BinaryOp.TRUE_DIVIDE:
            result = left / right
        elif op is BinaryOp.POWER:
            from datafusion import lit

            is_zero = f.iszero(left)
            unsafe = is_zero & (right < 0.0)
            square_root_zero = is_zero & (right == 0.5)
            negative_zero = (1.0 / left) < 0.0
            odd_integer = (right == f.floor(right)) & (f.abs(right % 2.0) == 1.0)
            signed_infinity = f.when(
                negative_zero & odd_integer, lit(float("-inf"))
            ).otherwise(lit(float("inf")))
            safe_left = f.when(unsafe, lit(1.0)).otherwise(left)
            result = (
                f.when(unsafe, signed_infinity)
                .when(square_root_zero, left)
                .otherwise(f.power(safe_left, right))
            )
        elif op is BinaryOp.GREATER:
            zero_pair = f.iszero(left) & f.iszero(right)
            result = ~(f.isnan(left) | f.isnan(right) | zero_pair) & (left > right)
        elif op is BinaryOp.GREATER_EQUAL:
            zero_pair = f.iszero(left) & f.iszero(right)
            result = ~(f.isnan(left) | f.isnan(right)) & (zero_pair | (left >= right))
        elif op is BinaryOp.LESS:
            zero_pair = f.iszero(left) & f.iszero(right)
            result = ~(f.isnan(left) | f.isnan(right) | zero_pair) & (left < right)
        elif op is BinaryOp.LESS_EQUAL:
            zero_pair = f.iszero(left) & f.iszero(right)
            result = ~(f.isnan(left) | f.isnan(right)) & (zero_pair | (left <= right))
        elif op is BinaryOp.EQUAL:
            zero_pair = f.iszero(left) & f.iszero(right)
            result = ~(f.isnan(left) | f.isnan(right)) & (zero_pair | (left == right))
        elif op is BinaryOp.NOT_EQUAL:
            zero_pair = f.iszero(left) & f.iszero(right)
            result = f.isnan(left) | f.isnan(right) | (~zero_pair & (left != right))
        else:
            raise AssertionError(f"unsupported binary operation: {op}")
        return result.cast(pa.float64()) if op in COMPARISON_OPS else result
