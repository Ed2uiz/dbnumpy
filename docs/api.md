# API and limits

dbnumpy 0.4.0a0 is a pre-alpha matrix package. Both engines share the operations
below. The API is not a complete NumPy implementation.

## Objects

| Type | Meaning |
|---|---|
| `DBArray` | Base class for lazy two-dimensional matrices |
| `DBDenseArray` | Every coordinate is stored |
| `DBSparseArray` | Missing coordinates mean zero |
| `DBVector` | Lazy one-dimensional indexing result |
| `DBScalar` | Lazy scalar indexing result |

Values use float64, including integer and Boolean inputs. Integers above
`2**53` can lose value precision. Coordinates use exact signed int64.
Complex values are unsupported. Comparison and `isnan` expressions store
float64 `0.0`/`1.0`, rather than a Boolean dtype.

## Connect and import

```python
import numpy as np
from dbnumpy import DuckDBBackend

with DuckDBBackend.connect() as backend:
    x = backend.from_numpy(np.arange(12.0).reshape(3, 4))
    result = np.sqrt(x + 1).to_numpy()
```

`DataFusionBackend.connect()` provides the same matrix API. Objects combined
in an expression must belong to the same backend instance.

| Method | Contract |
|---|---|
| `from_numpy(values)` | Copies a numeric/Boolean 2D input into owned float64 storage |
| `from_coo(row, col, data, shape=...)` | Validates 1D coordinates and values; rejects duplicates; removes zeros |
| `from_scipy(values)` | Copies sparse input, sums duplicates and removes zeros |
| `from_relation(name, shape=..., storage=...)` | Wraps an existing coordinate relation without copying values |
| DuckDB `from_mtx(path, ...)` | Reads and validates Matrix Market input in DuckDB |
| DataFusion `from_parquet(paths, shape=..., storage=...)` | Scans external canonical coordinate Parquet |

Relations use `(i, j, x)`: integer row/column coordinates and numeric values.
External relation/Parquet wrappers check the schema but do not scan for bounds,
uniqueness or dense completeness. The caller must guarantee those properties.
External files must stay available and unchanged. SQL NULL values become NaN.

Use `from_parquet()` for DataFusion Parquet input. It protects NaNs from a
DataFusion 54 statistics optimization. `from_relation()` retains the existing
relation's scan settings; it cannot repair a scan configured outside dbnumpy.

Host imports are independent of later input-array mutations. `from_relation()`
is a live reference. Direct changes to managed registrations through raw engine
APIs are outside the package contract.

DuckDB MTX loading is transactional. Its default sums duplicates and expands
supported symmetry. `assume_canonical=True` requires a trusted general file
with unique coordinates; it skips duplicate checks and aggregation. Both modes
check integer syntax and coordinate bounds. Copied input and selector names
cannot be overwritten, even with `overwrite=True`. See [file input](guides/out-of-core-mtx.md).

## Lazy operations

Supported operators: `+`, `-`, `*`, `/`, `**`, six numeric comparisons,
negation and absolute value. Reflected operations work too.

Supported NumPy ufuncs include:

- `sqrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`;
- `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`;
- `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `sign`, `trunc`, `isnan`;
- the supported arithmetic/comparison ufuncs and experimental 2D `matmul`.

Math functions also have matrix methods. `asin`, `acos` and `atan` are aliases.
Real-domain errors such as `sqrt(-1)` and `arcsin(2)` produce NaN. Eager NumPy
warning behavior is not reproduced when a lazy expression is built.

One- and two-dimensional NumPy arrays, lists and tuples are uploaded as
operands. Raw SciPy sparse operands must first pass through `from_scipy()`.
Shapes follow NumPy broadcasting. Vectors align with the trailing matrix
dimension, regardless of how they were extracted; length one broadcasts.

`A.T` and `np.transpose(A)` are lazy. `A @ B` accepts two-dimensional
operands in either direction. Matrix-vector multiplication is deferred.
Dense multiplication follows IEEE zero arithmetic; omitted sparse coordinates
use structural-zero multiplication. These can differ for `Inf * 0`.

## Indexing

```python
block = x[1:, ::-1]          # matrix
row = x[1, :]               # DBVector
cell = x[1, 2]              # DBScalar
reordered = x[[2, 0, 2], :] # order and repeats retained
```

Indices are zero-based. Negative indices count from the end. Slices support
signed nonzero steps. One axis may use a 1D integer or Boolean selector.
`take()` and `np.take(..., axis=0/1)` use the same rules.

For rectangular two-axis selection, write `x[rows, :][:, cols]`. Paired
advanced selectors, full matrix masks, new axes, higher-rank results,
flattened `take`, labels and assignment are unsupported.

## Reductions

Matrices support `sum`, `mean`, `var`, `std`, `min`, `max`, `any`, `all`,
`nansum`, `nanmean`, `nanvar`, `nanstd`, `nanmin` and `nanmax`, including
matching NumPy function calls. They run immediately and return host results.
Vectors currently support only `sum` and `mean`.

- Matrix axes: `None`, `0`, `1`, `-2`, `-1`; tuple axes are unsupported.
- `keepdims=True` preserves reduced dimensions in the host result.
- Variance/SD accept finite, nonnegative `ddof`, or its `correction` alias.
- Ordinary reductions propagate NaN; `nan*` variants skip stored NaNs.
- Implicit sparse zeros count in all reductions, including `nan*` variants.
- `any` and `all` return Boolean scalars/arrays; NaN and infinity are truthy.
- Empty extrema raise `ValueError` when the reduced dimension has length zero.
- For ordinary variance/SD, `ddof >= count` gives infinity for nonconstant
  slices and NaN for constant slices. The `nan*` versions give NaN when
  `ddof` reaches the effective non-NaN count.

The stable variance formula can differ from NumPy's intermediate rounding at
extreme magnitudes. See [correctness and limits](architecture.md#correctness-and-limits).

Product reductions are deferred. A log/exp substitute would not preserve
zeros, signs, nonfinite values or accuracy across both engines.

## Execution

| Call | Effect |
|---|---|
| `shape`, `dtype`, `size`, `storage` | Read metadata |
| Arithmetic, indexing, transpose | Build a lazy plan; may upload operands/selectors |
| `plan()` | Return the semantic plan as a JSON-compatible DAG |
| `compile()` | Generate SQL without executing values |
| `explain()` | Ask the engine for its query plan |
| `compute()` | Store results in the engine; preserve wrapper rank |
| `to_numpy()`, `np.asarray()` | Execute and collect a dense host array |
| `to_scipy()` | Execute and collect a sparse host array |
| `DBScalar.item()` | Execute and collect one value |
| Reductions | Execute and collect a scalar or NumPy array |

`plan()` refers to live source and selector relations. It is not a portable
replay file. Unsupported NumPy functions fail through dispatch. `np.asarray()`
is an explicitly supported eager conversion.

`compute()` returns an array backed by a completed result. DuckDB uses temporary
tables; DataFusion uses engine-written temporary Parquet files. Both support
subsequent matrix operations. DataFusion's result files stay available until
the backend closes, including when the original input is no longer available.
The files are temporary storage, not a persistent dataset export.

`DataFusionBackend.connect(temp_directory=...)` selects an existing writable
directory for result files and engine spill files. With a caller-supplied
`SessionContext`, `DataFusionBackend(context, temp_directory=...)` selects only
the result-file directory; the context retains its own spill configuration.
Closing the backend removes its result files, preserving the supplied directory
and caller-owned files. Use a context manager to ensure cleanup on exceptions.

In WSL 2, use a Linux temporary directory such as `/tmp`. Large DataFusion 54
writes failed with `Upload aborted` on `/mnt/c` in our tests; the same writes
completed on the Linux filesystem.

## Resource behavior

| Setting | Default | Bounds |
|---|---:|---|
| `max_densify_cells` | 5,000,000 | Dense expansion, dense input and dense collection |
| `max_host_values` | 50,000,000 | Vector collection and eager axis results |
| `max_sparse_host_values` | 5,000,000 | Sparse export coordinate rows, including zeros |
| `max_selector_values` | 5,000,000 | Total retained selector positions |
| `max_selector_relations` | 256 | Retained selector maps |

An operation such as `exp(A)` or `A == 0` fills missing sparse coordinates.
It must fit the dense-expansion limit. These limits are separate from engine
memory/spill settings and do not cap total process memory or query cost.

Finite row/column/scalar multiplication can avoid dense expansion. The proof
comes from copied inputs and coordinate-only transforms. Arithmetic results,
external relations and materialized outputs are not certified finite.
Unknown/nonfinite factors use guarded broadcasting. This optimization does
not cover sparse division or every broadcast form.

Sparse export requests at most its limit plus one rows, then rejects an
oversized result before allocating full NumPy/SciPy coordinate arrays.
`to_scipy()` drops explicit zeros. Lazy results and `compute()` may retain them,
so export can change later structural-zero multiplication with infinity.

Resources stay registered until the backend closes. Deleting a wrapper does
not release its data. Failed package operations remove newly created resources
without deleting borrowed relations. Streaming export and early release are
future work.

Expression depth is capped at 768; use intermediate `compute()` calls.
This is an upper bound, not a promise that every shallower backend plan works.
Long chains of relational gathers remain a known compiler limit.

## Unsupported options

No general dtype promotion, complex arrays, physical Boolean matrices,
arbitrary-rank tensors, mutation or labels. Ufuncs accept default options only;
masked `where`, `out`, requested casting/dtypes and generalized reductions
remain unsupported. Reductions reject `initial`, precomputed means and
non-float64 output dtypes; all-true scalar masks are accepted.

Matrix-vector multiplication and PCA/SVD remain outside the supported API.
