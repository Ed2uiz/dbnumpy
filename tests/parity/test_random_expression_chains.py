"""Deterministic generated expression chains for cross-layer differential testing."""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend
from dbnumpy.ir import ReductionOp


@pytest.mark.parity
@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_generated_expression_chain_matches_numpy(
    backend: Backend, seed: int, sparse_input: bool
) -> None:
    rng = np.random.default_rng(20_260_710 + seed)
    reference = rng.integers(-4, 5, size=(4, 5)).astype(np.float64)
    reference[rng.random(reference.shape) < 0.45] = 0.0
    matrix = (
        backend.from_scipy(sparse.csr_array(reference))
        if sparse_input
        else backend.from_numpy(reference)
    )

    for _ in range(18):
        operation = int(rng.integers(0, 20))
        if operation == 0:
            scalar = float(rng.choice([-2.0, 0.0, 1.5]))
            matrix = matrix + scalar
            reference = reference + scalar
        elif operation == 1:
            scalar = float(rng.choice([-1.5, 0.0, 0.75]))
            matrix = matrix * scalar
            reference = reference * scalar
        elif operation == 2:
            scalar = float(rng.choice([-1.0, 0.5, 2.0]))
            matrix = scalar - matrix
            reference = scalar - reference
        elif operation == 3:
            matrix = np.sin(matrix)
            reference = np.sin(reference)
        elif operation == 4:
            matrix = np.cos(matrix)
            reference = np.cos(reference)
        elif operation == 5:
            matrix = np.sqrt(abs(matrix) + 0.25)
            reference = np.sqrt(np.abs(reference) + 0.25)
        elif operation == 6:
            matrix = np.log1p(abs(matrix) * 0.1)
            reference = np.log1p(np.abs(reference) * 0.1)
        elif operation == 7:
            matrix = matrix.T
            reference = reference.T
        elif operation == 8:
            matrix = (matrix + matrix) * 0.5
            reference = (reference + reference) * 0.5
        elif operation == 9:
            row = rng.normal(scale=0.1, size=matrix.shape[1])
            matrix = matrix + row
            reference = reference + row
        elif operation == 10:
            threshold = float(rng.choice([-0.5, 0.0, 0.5]))
            matrix = matrix > threshold
            reference = (reference > threshold).astype(np.float64)
        elif operation == 11:
            scalar = float(rng.choice([-2.0, 0.5, 2.0]))
            matrix = matrix / scalar
            reference = reference / scalar
        elif operation == 12:
            matrix = matrix**2.0 * 0.05
            reference = reference**2.0 * 0.05
        elif operation == 13:
            matrix = np.sign(matrix)
            reference = np.sign(reference)
        elif operation == 14:
            matrix = np.trunc(matrix)
            reference = np.trunc(reference)
        elif operation == 15:
            matrix = np.tanh(matrix)
            reference = np.tanh(reference)
        elif operation == 16:
            matrix = np.log2(abs(matrix) + 1.0)
            reference = np.log2(np.abs(reference) + 1.0)
        elif operation == 17:
            matrix = np.arcsin(np.tanh(matrix))
            reference = np.arcsin(np.tanh(reference))
        elif operation == 18:
            matrix = np.arccos(np.tanh(matrix))
            reference = np.arccos(np.tanh(reference))
        else:
            matrix = np.arctan(matrix)
            reference = np.arctan(reference)

    actual = matrix.to_numpy()
    np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(matrix.sum(axis=0), reference.sum(axis=0))
    np.testing.assert_allclose(matrix.mean(axis=1), reference.mean(axis=1))
    np.testing.assert_allclose(matrix.var(), reference.var(), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(matrix.min(axis=0), reference.min(axis=0))
    np.testing.assert_allclose(matrix.max(axis=1), reference.max(axis=1))
    np.testing.assert_array_equal(matrix.any(axis=0), reference.any(axis=0))
    sql = matrix.compile()
    variance_sql, _ = backend.lowerer.compile_reduction(
        matrix._expr,  # noqa: SLF001 - optimizer-shape regression
        ReductionOp.VAR,
        None,
        dialect=backend.dialect,
        ddof=0,
    )
    physical_relations = {
        node["relation"] for node in matrix.plan()["nodes"] if node["node"] == "source"
    }
    assert all(sql.count(f'"{relation}"') == 1 for relation in physical_relations)
    assert all(
        variance_sql.count(f'"{relation}"') == 1 for relation in physical_relations
    )
    matrix_cte = re.search(
        r'WITH ("__dbm_internal_\d+_matrix_reduction") AS',
        variance_sql,
    )
    assert matrix_cte is not None
    assert variance_sql.count(matrix_cte.group(1)) == 2
