"""Rewrite dbnumpy's DuckDB-dialect SQL for the Polars SQL engine."""

from __future__ import annotations

import math
from typing import cast

import sqlglot
from sqlglot import expressions as sge

# Polars SQL has no hyperbolic functions or expm1. Near zero, a Taylor
# polynomial avoids the cancellation in exp(v) - exp(-v); its truncation error
# below 0.1 is under 1e-17 relative.
_SMALL_SINH = (
    "CASE WHEN ABS(v) < 0.1 THEN v * (1.0 + v * v * (1.0 / 6 + v * v * "
    "(1.0 / 120 + v * v * (1.0 / 5040 + v * v / 362880)))) "
    "ELSE (EXP(v) - EXP(-v)) / 2 END"
)
_SMALL_COSH = "(EXP(v) + EXP(-v)) / 2"
# Above 20, exp(-|v|) is negligible. Halving the exponent keeps exp(|v|) / 2
# finite for |v| in (709.8, 710.5), where exp(|v|) alone overflows.
_HALF_EXP = "EXP(ABS(v) / 2) * (EXP(ABS(v) / 2) / 2)"
_SINH = f"CASE WHEN ABS(v) > 20 THEN SIGN(v) * {_HALF_EXP} ELSE {_SMALL_SINH} END"
_COSH = f"CASE WHEN ABS(v) > 20 THEN {_HALF_EXP} ELSE {_SMALL_COSH} END"
# tanh(20) rounds to 1.0, and SIGN keeps exp() from overflowing to inf / inf.
_TANH = f"CASE WHEN ABS(v) > 20 THEN SIGN(v) ELSE ({_SMALL_SINH}) / ({_SMALL_COSH}) END"
_HYPERBOLIC: dict[type[sge.Expression], sge.Expression] = {
    function: cast(sge.Expression, sqlglot.parse_one(template, read="duckdb"))
    for function, template in ((sge.Sinh, _SINH), (sge.Cosh, _COSH), (sge.Tanh, _TANH))
}


def _hyperbolic(node: sge.Expression) -> sge.Expression:
    template = _HYPERBOLIC.get(type(node))
    if template is None:
        return node
    return sge.paren(
        template.transform(
            lambda part: (
                node.this.copy()
                if isinstance(part, sge.Column) and part.name == "v"
                else part
            )
        )
    )


def _nan_test(node: sge.Expression) -> sge.Expression:
    # Polars SQL has no isnan, but NaN compares equal to NaN in Polars.
    if isinstance(node, sge.IsNan):
        arg = node.this
    elif isinstance(node, sge.Anonymous) and node.name.lower() == "isnan":
        arg = node.expressions[0]
    else:
        return node
    nan = sge.cast(sge.Literal.string("NaN"), "DOUBLE")
    return sge.paren(sge.EQ(this=arg.copy(), expression=nan))


def _bool_aggregates(node: sge.Expression) -> sge.Expression:
    name = node.sql_name().lower() if isinstance(node, sge.Func) else ""
    one = sge.Literal.number(1)
    flags: sge.Expression
    if isinstance(node, sge.LogicalOr) or name in {"bool_or", "logical_or"}:
        flags = sge.Max(this=sge.cast(node.this.copy(), "INT"))
    elif isinstance(node, sge.LogicalAnd) or name in {"bool_and", "logical_and"}:
        flags = sge.Min(this=sge.cast(node.this.copy(), "INT"))
    else:
        return node
    return sge.paren(sge.EQ(this=flags, expression=one))


def _log_bases(node: sge.Expression) -> sge.Expression:
    # SQLGlot writes LOG2(v) as LOG(2, v); Polars reads LOG(a, b) as base b.
    if isinstance(node, sge.Log) and node.expression is not None:
        return sge.Div(
            this=sge.Ln(this=node.expression.copy()),
            expression=sge.Ln(this=node.this.copy()),
        )
    return node


def _constant(node: sge.Expression) -> float | None:
    if isinstance(node, (sge.Paren, sge.Cast)):
        return _constant(node.this)
    if isinstance(node, sge.Neg):
        value = _constant(node.this)
        return None if value is None else -value
    if isinstance(node, sge.Div):
        # Ibis writes -0.0 as -(1.0) / CAST('inf' AS DOUBLE).
        left, right = _constant(node.this), _constant(node.expression)
        return None if left is None or not right else left / right
    if isinstance(node, sge.Literal):
        try:
            return float(node.this)
        except ValueError:
            return None
    return None


def _is_zero(node: sge.Expression, *, negative: bool) -> bool:
    value = _constant(node)
    if value is None or value != 0.0:
        return False
    return (math.copysign(1.0, value) < 0) == negative


def _zero_identities(node: sge.Expression) -> sge.Expression:
    # Polars folds x + 0.0 and x - -0.0 to x, and 0.0 - x to -x, which keeps
    # the sign of a zero x. NumPy gives +0.0 there, so test for zero instead.
    value: sge.Expression
    if isinstance(node, sge.Add) and _is_zero(node.expression, negative=False):
        value = node.this.copy()
    elif isinstance(node, sge.Add) and _is_zero(node.this, negative=False):
        value = node.expression.copy()
    elif isinstance(node, sge.Sub) and _is_zero(node.expression, negative=True):
        value = node.this.copy()
    elif isinstance(node, sge.Sub) and _is_zero(node.this, negative=False):
        value = sge.Neg(this=sge.paren(node.expression.copy()))
    else:
        return node
    zero = sge.EQ(this=value.copy(), expression=sge.Literal.number("0.0"))
    return sge.paren(
        sge.Case(ifs=[sge.If(this=zero, true=sge.Literal.number("0.0"))], default=value)
    )


def _double_literals(node: sge.Expression) -> sge.Expression:
    # Polars reads 1.0 as a decimal, so 1.0 / 0.0 raises instead of giving inf.
    if (
        isinstance(node, sge.Literal)
        and not node.is_string
        and "." in node.this
        and not isinstance(node.parent, sge.Cast)
    ):
        return sge.cast(node.copy(), "DOUBLE")
    return node


def to_polars_sql(sql: str) -> str:
    """Return ``sql`` with the DuckDB functions Polars lacks rewritten."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    # One pass per rewrite: transform() does not revisit a replaced node's
    # children, and later passes must see the SQL that earlier ones produce.
    for rewrite in (
        _hyperbolic,
        _nan_test,
        _bool_aggregates,
        _log_bases,
        _zero_identities,
        _double_literals,
    ):
        tree = tree.transform(rewrite)
    return tree.sql(dialect="duckdb")
