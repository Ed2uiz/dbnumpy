<h1 align="center">dbnumpy</h1>

<p align="center">
  NumPy-style matrix operations in DuckDB and Apache DataFusion.
</p>

<p align="center">
  <a href="https://github.com/Ed2uiz/dbnumpy/actions/workflows/ci.yml"><img src="https://github.com/Ed2uiz/dbnumpy/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.12%2B-555555" alt="Python 3.12+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-555555" alt="MIT license"></a>
  <a href="#support"><img src="https://img.shields.io/badge/Status-Pre--alpha-555555" alt="Status: pre-alpha"></a>
  <a href="#install"><img src="https://img.shields.io/badge/PyPI-Not%20published-555555" alt="PyPI: not published"></a>
</p>

<p align="center">
  <a href="docs/api.md">API</a> ·
  <a href="docs/guides/overview.md">Examples</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/guides/r-parity.md">R comparison</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

dbnumpy builds lazy calculations over dense and sparse matrices. Operations run
in the database; results are collected into NumPy when needed. It builds on
ideas from [dbverse](https://dbverse-org.github.io/dbverse/).

## Install

Requires Python 3.12 or newer. Not yet published on [PyPI](https://pypi.org/).
Install from a local checkout:

```bash
git clone https://github.com/Ed2uiz/dbnumpy.git
cd dbnumpy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -e '.[duckdb,datafusion,scipy]'
```

These commands work on Linux, macOS and WSL 2. Install just the `duckdb` or
`datafusion` extra if you only need one engine.

## Example

```python
import numpy as np
import dbnumpy as dnp

with dnp.DuckDBBackend.connect() as backend:
    x = backend.from_numpy(np.arange(12.0).reshape(3, 4))
    y = np.sqrt(x + 1) * (x > 2)

    print(y.shape)       # inspect without calculating values
    print(y.to_numpy())  # run the calculation and collect the result
```

Use `DataFusionBackend` to run the same operations in DataFusion.
Both engines are checked against NumPy and SciPy in CI.

## Support

| Area | Supported | Limits |
|---|---|---|
| Engines | DuckDB and DataFusion | File-loading options differ |
| Arrays | Dense and sparse matrices, vectors and scalars | Float64; no higher-rank arrays or labels |
| Operations | Arithmetic, broadcasting and selected NumPy functions | Partial API; may change during pre-alpha |
| Indexing | Slices, transpose, repeated indices and one-axis Boolean selection | No assignment or full Boolean masks |
| Reductions | Sums, means, variance, standard deviation and others | Execute immediately; return Python or NumPy results |
| File input | Matrix Market in DuckDB; coordinate Parquet in DataFusion | No general DataFusion Matrix Market loader |
| Storage | Coordinate tables with configurable size limits | Dense storage and filling sparse zeros can be costly |
| Inspection | Saved operation steps, SQL and engine plans | DataFusion may execute an equivalent native plan |
| Linear algebra | Experimental 2D matrix multiplication | No matrix-vector multiplication or PCA/SVD |

## NumPy features

These examples use the same matrix. Each creates a new result; `x` stays unchanged.

```python
import numpy as np
import dbnumpy as dnp

backend = dnp.DuckDBBackend.connect()  # or dnp.DataFusionBackend.connect()
x = backend.from_numpy(np.array([[1., 2., 3.], [4., 5., 6.]]))
print(x.to_numpy())
```

```text
[[1. 2. 3.]
 [4. 5. 6.]]
```

### Broadcasting

Add a vector to every row.

```python
y = x + np.array([10., 20., 30.])
print(y.to_numpy())
```

```text
[[11. 22. 33.]
 [14. 25. 36.]]
```

### Elementwise functions

Apply a NumPy function to every value. Round the displayed result to three decimals.

```python
y = np.sqrt(x)
print(np.round(y.to_numpy(), 3))
```

```text
[[1.    1.414 1.732]
 [2.    2.236 2.449]]
```

### Indexing and transpose

Select columns in a new order, then transpose the result.

```python
y = x[:, [2, 0]].T
print(y.to_numpy())
```

```text
[[3. 6.]
 [1. 4.]]
```

### Reductions

Calculate column means, then subtract them from each row.

```python
means = np.mean(x, axis=0)
y = x - means
print(means)
print(y.to_numpy())
```

```text
[2.5 3.5 4.5]
[[-1.5 -1.5 -1.5]
 [ 1.5  1.5  1.5]]
```

Close the backend when finished:

```python
backend.close()
```

## Benchmark

As a proof of concept, below is a sample workflow commonly used in bioinformatics that preprocesses a cell-by-gene count matrix. Each value in this matrix represents the count of a specific feature (gene) in a sample (cell). The workflow filters cells and genes, normalizes counts, and calculates
gene statistics using the same underlying NumPy code across SciPy, dbnumpy with DuckDB and dbnumpy with DataFusion. We tested 5,000–250,000 cells
sampled from the [10x 1.3M mouse brain dataset](https://www.10xgenomics.com/datasets/1-3-million-brain-cells-from-e-18-mice-2-standard-1-3-0).

<details>
<summary>See code</summary>

Rows are cells; columns are genes. The function below is shared by all three paths. SciPy uses a sparse `csr_array`; the database paths use dbnumpy arrays.

```python
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
```

After loading the input matrix, the benchmark calls `result = preprocess(values)`.
It also stores the complete normalized matrix:

```python
# backend is None for SciPy, whose result is already computed.
materialized = result.values if backend is None else result.values.compute()
```

[Workflow source](examples/pbmc_workflow.py) ·
[Benchmark runner](benchmarks/brain_preprocessing.py)

</details>

![Single-cell preprocessing runtime including storing the normalized matrix](docs/assets/brain-native-compute.png)

One run per point in WSL 2 with a 7.51 GiB RAM ceiling. Times exclude matrix preparation, export and validation.
The runtime includes `compute()` to store the normalized matrix in the database.
× marks an out-of-memory failure.

DuckDB completed 250,000 cells, including `compute()`, in 185 seconds. DataFusion
completed 250,000 cells in 108 seconds using temporary Parquet storage. SciPy
ran out of memory at 50,000 cells and above. These are preliminary measurements;
successful runs passed checks of gene summaries, entry counts and sampled values.

DataFusion was rerun at all six sizes using Linux temporary storage. The other
paths show the earlier runs with Windows-mounted temporary storage; these
are not controlled measurements of relative speed.

[Method and results](docs/guides/brain-native-compute.md) ·
[Earlier PBMC benchmark](docs/guides/pbmc-benchmark.md)
