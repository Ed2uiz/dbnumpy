"""Randomly generated programs match NumPy end to end."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.parity

SPECIAL = [0.0, 0.0, 0.0, -1.0, 1.0, 2.0, 0.25, 1e-10, np.nan, np.inf, -np.inf]
UNARY = [np.sin, np.cos, np.sqrt, np.log1p, np.abs, np.negative, np.tanh, np.arctan]
SCALAR_OPS = [np.add, np.subtract, np.multiply, np.true_divide, np.power, np.greater]
MATRIX_OPS = [np.add, np.subtract, np.multiply]


def random_step(
    rng: np.random.Generator,
    shape: tuple[int, int],
    pool: list[tuple[Any, np.ndarray]],
    fresh: Callable[[tuple[int, int]], tuple[Any, np.ndarray]],
) -> tuple[str, Callable[[Any], Any]]:
    """Pick one operation; it applies equally to a DBArray and to NumPy."""
    rows, cols = shape
    kind = int(rng.integers(7))
    if kind == 0:
        op = UNARY[int(rng.integers(len(UNARY)))]
        return op.__name__, op
    if kind == 1:
        op = SCALAR_OPS[int(rng.integers(len(SCALAR_OPS)))]
        scalar = float(rng.choice([-1.0, 0.0, 0.25, 2.0]))
        if rng.integers(2):
            return f"{op.__name__}(x, {scalar})", lambda x: op(x, scalar)
        return f"{op.__name__}({scalar}, x)", lambda x: op(scalar, x)
    if kind == 2:
        return "transpose", lambda x: x.T
    if kind == 3:
        index = rng.integers(0, rows, size=int(rng.integers(1, rows + 2))).tolist()
        return f"rows {index}", lambda x: x[index, :]
    if kind == 4:
        step = int(rng.choice([-2, -1, 1, 2]))
        columns = slice(1, None) if cols > 1 else slice(None)
        return f"slice ::{step}", lambda x: x[::step, columns]
    if kind == 5:
        vector = rng.normal(size=cols)
        return "add row vector", lambda x: x + vector
    op = MATRIX_OPS[int(rng.integers(len(MATRIX_OPS)))]
    # Reusing an earlier result turns the program into a DAG with shared branches.
    peers = [entry for entry in pool if entry[1].shape == shape]
    peer = peers[int(rng.integers(len(peers)))] if rng.integers(2) else fresh(shape)
    return op.__name__, lambda x: op(
        x, peer[1] if isinstance(x, np.ndarray) else peer[0]
    )


@pytest.mark.parametrize("seed", range(12))
def test_generated_program_matches_numpy(upload: Any, seed: int) -> None:
    """Chain random steps; matrix operands may reuse earlier results."""
    rng = np.random.default_rng(20261011 + seed)

    def fresh(shape: tuple[int, ...]) -> tuple[Any, np.ndarray]:
        values = rng.choice(SPECIAL, size=shape)
        return upload(values, sparse_input=bool(rng.integers(2)))

    pool = [fresh((4, 5))]
    names = []
    with np.errstate(all="ignore"):
        for _ in range(16):
            matrix, values = pool[-1 - int(rng.integers(min(len(pool), 3)))]
            name, step = random_step(rng, values.shape, pool, fresh)
            pool.append((step(matrix), np.asarray(step(values), dtype=np.float64)))
            names.append(name)

        try:
            for matrix, values in pool[-3:]:
                for actual in (
                    matrix.to_numpy(),
                    matrix.compute().to_numpy(),
                    matrix.sum(axis=0),
                ):
                    expected = values.sum(axis=0) if actual.ndim == 1 else values
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-11, atol=1e-12, equal_nan=True
                    )
        except AssertionError as error:
            error.add_note(f"seed={20261011 + seed}; steps={names}")
            raise
