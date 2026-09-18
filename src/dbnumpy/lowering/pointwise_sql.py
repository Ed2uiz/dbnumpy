"""Iterative SQL rendering for arbitrarily deep single-source pointwise trees."""

from __future__ import annotations

import math

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


class PointwiseSQLCompiler:
    """Render the hot pointwise path without recursive compiler traversal.

    The renderer covers only semantic nodes that preserve coordinates. More
    complex relational plans continue through Ibis. Its explicit bounded scope
    keeps DBVerse from growing an accidental general-purpose SQL compiler.
    """

    def compile(
        self,
        expr: MatrixExpr,
        *,
        dialect: str,
        name_prefix: str = "__dbm_pointwise",
    ) -> str:
        source = single_pointwise_source(expr)
        if source is None:
            raise ValueError("pointwise SQL compilation requires exactly one source")

        output_storage = storage_kind(expr)
        shared = pointwise_alias_nodes(expr)
        if shared:
            return self._compile_shared(
                expr,
                source,
                shared,
                output_storage=output_storage,
                dialect=dialect,
                name_prefix=name_prefix,
            )
        if source.storage is StorageKind.SPARSE and output_storage is StorageKind.DENSE:
            nan = self._literal(float("nan"))
            base = (
                f'CASE WHEN "v"."i" IS NULL THEN 0.0 ELSE COALESCE("v"."x", {nan}) END'
            )
            value = self._value(expr, source, base=base, dialect=dialect)
            return (
                'SELECT "r"."i" AS "i", "c"."j" AS "j", '
                f'{value} AS "x" '
                f'FROM {self._quote(source.rows_relation)} AS "r" '
                f'CROSS JOIN {self._quote(source.cols_relation)} AS "c" '
                f'LEFT OUTER JOIN {self._quote(source.relation)} AS "v" '
                'ON "r"."i" = "v"."i" AND "c"."j" = "v"."j"'
            )

        value = self._value(
            expr,
            source,
            base=f'COALESCE("v"."x", {self._literal(float("nan"))})',
            dialect=dialect,
        )
        return (
            'SELECT "v"."i" AS "i", "v"."j" AS "j", '
            f'{value} AS "x" FROM {self._quote(source.relation)} AS "v"'
        )

    def _compile_shared(
        self,
        expr: MatrixExpr,
        source: Source,
        shared: tuple[MatrixExpr, ...],
        *,
        output_storage: StorageKind,
        dialect: str,
        name_prefix: str,
    ) -> str:
        prefix = name_prefix
        forbidden = {
            source.relation,
            source.rows_relation,
            source.cols_relation,
        }
        while any(
            f"{prefix}_{suffix}" in forbidden
            for suffix in ["base", *(str(index) for index in range(len(shared)))]
        ):
            prefix += "_"

        base_name = f"{prefix}_base"
        base_column = "__dbm_pointwise_x"
        if source.storage is StorageKind.SPARSE and output_storage is StorageKind.DENSE:
            nan = self._literal(float("nan"))
            base_query = (
                'SELECT "r"."i" AS "i", "c"."j" AS "j", '
                'CASE WHEN "v"."i" IS NULL THEN 0.0 '
                f'ELSE COALESCE("v"."x", {nan}) END '
                f"AS {self._quote(base_column)} "
                f'FROM {self._quote(source.rows_relation)} AS "r" '
                f'CROSS JOIN {self._quote(source.cols_relation)} AS "c" '
                f'LEFT OUTER JOIN {self._quote(source.relation)} AS "v" '
                'ON "r"."i" = "v"."i" AND "c"."j" = "v"."j"'
            )
        else:
            nan = self._literal(float("nan"))
            base_query = (
                'SELECT "v"."i" AS "i", "v"."j" AS "j", '
                f'COALESCE("v"."x", {nan}) AS {self._quote(base_column)} '
                f'FROM {self._quote(source.relation)} AS "v"'
            )

        ctes = [f"{self._quote(base_name)} AS ({base_query})"]
        aliases: dict[int, str] = {}
        previous = base_name
        for index, node in enumerate(shared):
            stage_name = f"{prefix}_{index}"
            value_name = f"__dbm_pointwise_value_{index}"
            value = self._value(
                node,
                source,
                base=self._quote(base_column),
                dialect=dialect,
                aliases=aliases,
            )
            ctes.append(
                f"{self._quote(stage_name)} AS ("
                f"SELECT *, {value} AS {self._quote(value_name)} "
                f"FROM {self._quote(previous)})"
            )
            aliases[id(node)] = self._quote(value_name)
            previous = stage_name

        value = self._value(
            expr,
            source,
            base=self._quote(base_column),
            dialect=dialect,
            aliases=aliases,
        )
        return (
            f"WITH {', '.join(ctes)} "
            f'SELECT "i" AS "i", "j" AS "j", {value} AS "x" '
            f"FROM {self._quote(previous)}"
        )

    def _value(
        self,
        expr: MatrixExpr,
        source: Source,
        *,
        base: str,
        dialect: str,
        aliases: dict[int, str] | None = None,
    ) -> str:
        return pointwise_value(
            expr,
            source,
            base=base,
            literal=self._literal,
            unary=self._unary,
            binary=lambda op, left, right: self._binary(
                op, left, right, dialect=dialect
            ),
            aliases=aliases,
        )

    @staticmethod
    def _unary(op: UnaryOp, value: str) -> str:
        if op is UnaryOp.NEGATIVE:
            return f"(-{value})"
        if op is UnaryOp.ABSOLUTE:
            return f"ABS({value})"
        if op is UnaryOp.SQRT:
            nan = PointwiseSQLCompiler._literal(float("nan"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            is_zero = PointwiseSQLCompiler._is_zero(
                value, negative_infinity=negative_infinity
            )
            return (
                f"CASE WHEN {value} < 0.0 AND NOT {is_zero} "
                f"THEN {nan} ELSE SQRT({value}) END"
            )
        if op is UnaryOp.EXP:
            return f"EXP({value})"
        if op is UnaryOp.EXPM1:
            series = (
                f"({value} * (1.0 + {value} * (0.5 + {value} * "
                f"(0.16666666666666666 + {value} * "
                f"(0.041666666666666664 + {value} * 0.008333333333333333)))))"
            )
            return (
                f"CASE WHEN ABS({value}) < 1e-5 THEN {series} "
                f"ELSE (EXP({value}) - 1.0) END"
            )
        if op is UnaryOp.LOG:
            nan = PointwiseSQLCompiler._literal(float("nan"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            is_zero = PointwiseSQLCompiler._is_zero(
                value, negative_infinity=negative_infinity
            )
            return (
                f"CASE WHEN {value} < 0.0 AND NOT {is_zero} THEN {nan} "
                f"WHEN {is_zero} THEN {negative_infinity} "
                f"ELSE LN({value}) END"
            )
        if op is UnaryOp.LOG1P:
            nan = PointwiseSQLCompiler._literal(float("nan"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            series = (
                f"({value} * (1.0 + {value} * (-0.5 + {value} * "
                f"(0.3333333333333333 + {value} * (-0.25 + {value} * 0.2)))))"
            )
            return (
                f"CASE WHEN {value} < -1.0 THEN {nan} "
                f"WHEN {value} = -1.0 THEN {negative_infinity} "
                f"WHEN ABS({value}) < 1e-4 THEN {series} "
                f"ELSE LN({value} + 1.0) END"
            )
        if op is UnaryOp.SIN:
            infinity = PointwiseSQLCompiler._literal(float("inf"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            nan = PointwiseSQLCompiler._literal(float("nan"))
            return (
                f"CASE WHEN {value} = {infinity} OR {value} = {negative_infinity} "
                f"THEN {nan} ELSE SIN({value}) END"
            )
        if op is UnaryOp.COS:
            infinity = PointwiseSQLCompiler._literal(float("inf"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            nan = PointwiseSQLCompiler._literal(float("nan"))
            return (
                f"CASE WHEN {value} = {infinity} OR {value} = {negative_infinity} "
                f"THEN {nan} ELSE COS({value}) END"
            )
        if op is UnaryOp.TAN:
            infinity = PointwiseSQLCompiler._literal(float("inf"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            nan = PointwiseSQLCompiler._literal(float("nan"))
            return (
                f"CASE WHEN {value} = {infinity} OR {value} = {negative_infinity} "
                f"THEN {nan} ELSE TAN({value}) END"
            )
        if op is UnaryOp.FLOOR:
            return f"FLOOR({value})"
        if op is UnaryOp.CEIL:
            return f"CEIL({value})"
        if op is UnaryOp.SIGN:
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            is_zero = PointwiseSQLCompiler._is_zero(
                value, negative_infinity=negative_infinity
            )
            return (
                f"CASE WHEN ISNAN({value}) THEN {value} WHEN {is_zero} THEN 0.0 "
                f"WHEN {value} > 0.0 THEN 1.0 ELSE -1.0 END"
            )
        if op is UnaryOp.TRUNC:
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            is_zero = PointwiseSQLCompiler._is_zero(
                value, negative_infinity=negative_infinity
            )
            return (
                f"CASE WHEN ISNAN({value}) OR {is_zero} THEN {value} "
                f"WHEN {value} < 0.0 THEN CEIL({value}) ELSE FLOOR({value}) END"
            )
        if op is UnaryOp.ISNAN:
            return f"CAST(ISNAN({value}) AS DOUBLE)"
        if op in {UnaryOp.LOG2, UnaryOp.LOG10}:
            nan = PointwiseSQLCompiler._literal(float("nan"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            is_zero = PointwiseSQLCompiler._is_zero(
                value, negative_infinity=negative_infinity
            )
            function = "LOG2" if op is UnaryOp.LOG2 else "LOG10"
            return (
                f"CASE WHEN ISNAN({value}) THEN {value} "
                f"WHEN {value} < 0.0 AND NOT {is_zero} THEN {nan} "
                f"WHEN {is_zero} THEN {negative_infinity} "
                f"ELSE {function}({value}) END"
            )
        if op is UnaryOp.SINH:
            return f"SINH({value})"
        if op is UnaryOp.COSH:
            return f"COSH({value})"
        if op is UnaryOp.TANH:
            return f"TANH({value})"
        if op in {UnaryOp.ARCSIN, UnaryOp.ARCCOS}:
            nan = PointwiseSQLCompiler._literal(float("nan"))
            function = "ASIN" if op is UnaryOp.ARCSIN else "ACOS"
            return (
                f"CASE WHEN {value} < -1.0 OR {value} > 1.0 "
                f"THEN {nan} ELSE {function}({value}) END"
            )
        if op is UnaryOp.ARCTAN:
            return f"ATAN({value})"
        raise AssertionError(f"unsupported unary operation: {op}")

    @staticmethod
    def _binary(op: BinaryOp, left: str, right: str, *, dialect: str) -> str:
        symbols = {
            BinaryOp.ADD: "+",
            BinaryOp.SUBTRACT: "-",
            BinaryOp.MULTIPLY: "*",
            BinaryOp.TRUE_DIVIDE: "/",
            BinaryOp.GREATER: ">",
            BinaryOp.GREATER_EQUAL: ">=",
            BinaryOp.LESS: "<",
            BinaryOp.LESS_EQUAL: "<=",
            BinaryOp.EQUAL: "=",
            BinaryOp.NOT_EQUAL: "<>",
        }
        if op is BinaryOp.POWER:
            infinity = PointwiseSQLCompiler._literal(float("inf"))
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            signed_infinity = (
                f"CASE WHEN (1.0 / {left}) < 0.0 "
                f"AND {right} = FLOOR({right}) "
                f"AND ABS({right} % 2.0) = 1.0 "
                f"THEN {negative_infinity} ELSE {infinity} END"
            )
            unsafe = (
                f"({left} = 0.0 OR (1.0 / {left}) = {negative_infinity}) "
                f"AND {right} < 0.0"
            )
            square_root_zero = (
                f"({left} = 0.0 OR (1.0 / {left}) = {negative_infinity}) "
                f"AND {right} = 0.5"
            )
            safe_left = f"CASE WHEN {unsafe} THEN 1.0 ELSE {left} END"
            result = (
                f"CASE WHEN {unsafe} "
                f"THEN {signed_infinity} "
                f"WHEN {square_root_zero} THEN {left} "
                f"ELSE POWER({safe_left}, {right}) END"
            )
        elif op is BinaryOp.EQUAL:
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            zero_pair = PointwiseSQLCompiler._zero_pair(
                left, right, negative_infinity=negative_infinity
            )
            result = (
                f"((NOT ISNAN({left})) AND (NOT ISNAN({right})) "
                f"AND ({zero_pair} OR ({left} = {right})))"
            )
        elif op is BinaryOp.NOT_EQUAL:
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            zero_pair = PointwiseSQLCompiler._zero_pair(
                left, right, negative_infinity=negative_infinity
            )
            result = (
                f"(ISNAN({left}) OR ISNAN({right}) "
                f"OR ((NOT {zero_pair}) AND ({left} <> {right})))"
            )
        elif op in COMPARISON_OPS:
            negative_infinity = PointwiseSQLCompiler._literal(float("-inf"))
            zero_pair = PointwiseSQLCompiler._zero_pair(
                left, right, negative_infinity=negative_infinity
            )
            if op in {BinaryOp.GREATER_EQUAL, BinaryOp.LESS_EQUAL}:
                comparison = f"({zero_pair} OR ({left} {symbols[op]} {right}))"
            else:
                comparison = f"((NOT {zero_pair}) AND ({left} {symbols[op]} {right}))"
            result = f"((NOT ISNAN({left})) AND (NOT ISNAN({right})) AND {comparison})"
        else:
            result = f"({left} {symbols[op]} {right})"
        if op in COMPARISON_OPS:
            sql_type = "DOUBLE" if dialect == "duckdb" else "DOUBLE PRECISION"
            return f"CAST({result} AS {sql_type})"
        return result

    @staticmethod
    def _zero_pair(left: str, right: str, *, negative_infinity: str) -> str:
        left_zero = PointwiseSQLCompiler._is_zero(
            left, negative_infinity=negative_infinity
        )
        right_zero = PointwiseSQLCompiler._is_zero(
            right, negative_infinity=negative_infinity
        )
        return f"({left_zero} AND {right_zero})"

    @staticmethod
    def _is_zero(value: str, *, negative_infinity: str) -> str:
        return f"({value} = 0.0 OR (1.0 / {value}) = {negative_infinity})"

    @staticmethod
    def _literal(value: bool | int | float) -> str:
        numeric = float(value)
        if math.isnan(numeric):
            return "CAST('NaN' AS DOUBLE)"
        if math.isinf(numeric):
            sign = "-Infinity" if numeric < 0 else "Infinity"
            return f"CAST('{sign}' AS DOUBLE)"
        return repr(numeric)

    @staticmethod
    def _quote(identifier: str) -> str:
        return f'"{identifier.replace(chr(34), chr(34) * 2)}"'
