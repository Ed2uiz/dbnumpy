# Overview

`DBArray` is a lazy matrix. `DBDenseArray` stores every coordinate;
`DBSparseArray` treats missing coordinates as zero. Shape and storage metadata
are available without running a value query.

## Example

```python
--8<-- "examples/vignettes/overview.py"
```

## Input choices

| Input | Method | Where data is prepared |
|---|---|---|
| NumPy array | `from_numpy()` | Python; copied before registration |
| SciPy sparse array | `from_scipy()` | Python |
| Coordinate arrays | `from_coo()` | Python |
| Existing `(i, j, x)` relation | `from_relation()` | Already in the engine |
| Matrix Market file | DuckDB `from_mtx()` | DuckDB |
| Coordinate Parquet | DataFusion `from_parquet()` | Lazy external scan |

Relations use zero-based int64 coordinates and float64 values. Shape is stored
separately, so an all-zero row or column still exists even when it has no
stored values.

`from_relation()` trusts the supplied shape and coordinate contract. It does
not scan the whole input to validate it. See [API and limits](../api.md).

## When work runs

Arithmetic, indexing and transpose build plans. Reductions run immediately.
`to_numpy()` and `to_scipy()` collect data into Python. `compute()` stores a
result in the engine and returns a new lazy object.

Both engines use the same matrix rules. Inputs must belong to the same backend
instance to be combined. Closing the backend releases its registered resources.
