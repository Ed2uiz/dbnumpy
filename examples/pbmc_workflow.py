"""One NumPy-style preprocessing function for SciPy arrays and dbnumpy arrays.

Rows are cells; columns are genes. Input counts must be finite and nonnegative.
Use a SciPy sparse *array*, whose multiplication is elementwise, not csr_matrix.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PreprocessingResult:
    values: Any
    cells: np.ndarray
    genes: np.ndarray
    total_counts: np.ndarray
    detected_genes: np.ndarray
    gene_mean: np.ndarray
    gene_variance: np.ndarray


def preprocess(x, *, min_genes=200, min_cells=3, target_sum=10_000.0):
    """Filter, normalize, log-transform and summarize a cell-by-gene matrix.

    Returned cell/gene indices refer to the original input. Variance uses ddof=1.
    Reject an empty result or fewer than two retained cells. Zero-total cells
    after gene filtering are also removed, so every normalized row has a scale.
    """
    total_counts = np.asarray(np.sum(x, axis=1)).reshape(-1)
    detected_genes = np.asarray(np.sum(x > 0, axis=1)).reshape(-1)
    cells = np.flatnonzero(detected_genes >= min_genes)
    x = x[cells, :]
    detected_cells = np.asarray(np.sum(x > 0, axis=0)).reshape(-1)
    genes = np.flatnonzero(detected_cells >= min_cells)
    if len(genes) == 0:
        raise ValueError("No genes remain after filtering")
    x = x[:, genes]
    totals = np.asarray(np.sum(x, axis=1)).reshape(-1)
    positive = totals > 0
    cells = cells[positive]
    x = x[positive, :]
    totals = totals[positive]
    if len(cells) < 2:
        raise ValueError("At least two cells must remain after filtering")
    scale = target_sum / totals
    values = np.log1p(x * scale[:, None])
    mean = np.asarray(np.mean(values, axis=0)).reshape(-1)
    second_moment = np.asarray(np.mean(values * values, axis=0)).reshape(-1)
    variance = (second_moment - mean * mean) * len(cells) / (len(cells) - 1)
    return PreprocessingResult(
        values, cells, genes, total_counts, detected_genes, mean, variance
    )
