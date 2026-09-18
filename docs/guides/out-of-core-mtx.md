# File input

DuckDB can read Matrix Market files without building a full SciPy matrix or
Arrow input table in Python. DataFusion can scan existing coordinate Parquet.

## DuckDB Matrix Market

```python
--8<-- "examples/vignettes/out_of_core_mtx.py"
```

`from_mtx()` accepts plain or gzip coordinate files with real, integer or
pattern values. It supports general, symmetric, skew-symmetric and real-valued
Hermitian input. The default validates coordinates, expands symmetry, sums
duplicates and removes zeros. Pattern entries become `1.0`.

For trusted general files, `assume_canonical=True` skips duplicate aggregation,
sorting, symmetry expansion and zero removal. It still checks row counts,
integer syntax and bounds. **You must guarantee unique coordinates.** Duplicate
rows in this mode can give incorrect reductions.

Use a file-backed `database`, a name and `temporary=False` to keep a table
across connections. `temporary=True` limits it to the connection.
`overwrite=True` can replace ordinary database tables, but cannot replace
copied inputs or selector maps. Failed loads preserve the old destination.

The loader manages its own transaction. Do not call it inside another DuckDB
transaction. Set `temp_directory` when connecting to choose a spill location.

## DataFusion Parquet

```python
from dbnumpy import DataFusionBackend

with DataFusionBackend.connect() as backend:
    x = backend.from_parquet(
        "coordinates.parquet", shape=(20_000, 1_000_000), storage="sparse"
    )
    means = x.mean(axis=0)
```

Files must contain canonical, zero-based `(i, j, x)` coordinates. Bounds,
uniqueness and the declared shape/storage are the caller's responsibility.
Files must stay available and unchanged while expressions use them.
There is no general DataFusion MTX loader.

## Memory and measurement

Engine memory limits do not cover every Python, NumPy or Arrow allocation.
Output vectors still need host memory. See [resource limits](../api.md#resource-behavior).

Measure file preparation, loading, reopening, cold queries, warm queries and
result collection separately. A warm reduction time is not an ingestion time.

Sparse column means divide each column sum by the full row count. For a column
`[3, 0, 9]`, the mean is `4`; averaging only its stored values would give `6`.
