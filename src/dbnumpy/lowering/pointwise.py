"""Shared input ordering and operand rules for elementwise query builders."""

from collections.abc import Callable

from dbnumpy.ir import (
    BinaryOp,
    ElementwiseBinary,
    MatrixExpr,
    ScalarBinary,
    Source,
    Unary,
    UnaryOp,
    postorder,
)


def pointwise_value[T](
    expr: MatrixExpr,
    source: Source,
    *,
    base: T,
    literal: Callable[[float], T],
    unary: Callable[[UnaryOp, T], T],
    binary: Callable[[BinaryOp, T, T], T],
    aliases: dict[int, T] | None = None,
) -> T:
    """Build each value once; reuse named steps without expanding their inputs."""
    aliases = aliases or {}
    values: dict[int, T] = {}

    def named(node: MatrixExpr) -> bool:
        return node is not expr and id(node) in aliases

    for node in postorder(expr, stop=named):
        key = id(node)
        if named(node):
            values[key] = aliases[key]
        elif isinstance(node, Source):
            if node != source:
                raise AssertionError("pointwise expression contains a second source")
            values[key] = base
        elif isinstance(node, Unary):
            values[key] = unary(node.op, values[id(node.arg)])
        elif isinstance(node, ScalarBinary):
            arg, scalar = values[id(node.arg)], literal(float(node.scalar))
            left, right = (scalar, arg) if node.reverse else (arg, scalar)
            values[key] = binary(node.op, left, right)
        elif isinstance(node, ElementwiseBinary):
            values[key] = binary(node.op, values[id(node.left)], values[id(node.right)])
        else:
            raise AssertionError(f"non-pointwise expression: {type(node)!r}")
    return values[id(expr)]
