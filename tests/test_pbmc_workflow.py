"""Check the shared example against dense NumPy, including filtering edges."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

MODULE_PATH = Path(__file__).parents[1] / "examples" / "pbmc_workflow.py"
SPEC = importlib.util.spec_from_file_location("pbmc_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)
preprocess = workflow.preprocess


@pytest.mark.backend
@pytest.mark.parity
def test_preprocessing_matches_dense(backend):
    counts = np.array(
        [
            [0, 0, 0, 0],
            [1, 2, 0, 0],
            [3, 0, 4, 0],
            [0, 5, 6, 0],
            [0, 0, 0, 9],
            [1, 0, 0, 0],
        ],
        dtype=float,
    )
    # The rare fourth gene is removed, leaving its cell with zero counts.
    reference = preprocess(counts, min_genes=1, min_cells=2)
    for x in [sparse.csr_array(counts), backend.from_scipy(sparse.csr_array(counts))]:
        result = preprocess(x, min_genes=1, min_cells=2)
        actual = (
            result.values.to_scipy()
            if hasattr(result.values, "to_scipy")
            else result.values
        ).toarray()
        np.testing.assert_array_equal(result.cells, [1, 2, 3, 5])
        np.testing.assert_array_equal(result.genes, [0, 1, 2])
        np.testing.assert_allclose(actual, reference.values, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.gene_mean, reference.values.mean(axis=0))
        np.testing.assert_allclose(
            result.gene_variance, reference.values.var(axis=0, ddof=1), rtol=1e-10
        )
        np.testing.assert_array_equal(result.total_counts, counts.sum(axis=1))
        np.testing.assert_array_equal(result.detected_genes, (counts > 0).sum(axis=1))


@pytest.mark.backend
@pytest.mark.parametrize("counts", [np.zeros((3, 4)), np.ones((1, 4))])
def test_preprocessing_rejects_undefined_summary(backend, counts):
    for x in [sparse.csr_array(counts), backend.from_scipy(sparse.csr_array(counts))]:
        with pytest.raises(ValueError, match="No genes|At least two"):
            preprocess(x, min_genes=1, min_cells=1)


@pytest.mark.backend
@pytest.mark.parity
@pytest.mark.parametrize("seed", range(8))
def test_preprocessing_against_independent_dense_calculation(backend, seed):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.geomspace(0.001, 1000, 13), (47, 13)).astype(float)
    counts[0] = 0
    counts[:, 0] = 0
    counts[1, 0] = 1  # Gene present in only one cell.
    if seed == 0:
        counts[2:] = 1  # Constant transformed genes exercise near-zero variance.
    keep_cells = np.count_nonzero(counts, axis=1) >= 3
    keep_genes = np.count_nonzero(counts[keep_cells], axis=0) >= 2
    expected = counts[keep_cells][:, keep_genes]
    positive = expected.sum(axis=1) > 0
    expected = expected[positive]
    expected = np.log1p(expected / expected.sum(axis=1, keepdims=True) * 10_000)
    for values in (
        sparse.csr_array(counts),
        backend.from_scipy(sparse.csr_array(counts)),
    ):
        result = preprocess(values, min_genes=3, min_cells=2)
        output = result.values
        if hasattr(output, "to_scipy"):
            output = output.to_scipy()
        np.testing.assert_array_equal(
            result.cells, np.flatnonzero(keep_cells)[positive]
        )
        np.testing.assert_array_equal(result.genes, np.flatnonzero(keep_genes))
        np.testing.assert_allclose(output.toarray(), expected, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(result.gene_mean, expected.mean(axis=0), atol=1e-11)
        np.testing.assert_allclose(
            result.gene_variance, expected.var(axis=0, ddof=1), rtol=1e-9, atol=1e-11
        )
