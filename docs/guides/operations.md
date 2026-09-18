# Operations

Matrix transformations are lazy. Reductions return Python scalars or NumPy
arrays immediately.

| Task | Expression |
|---|---|
| Column sums / means | `A.sum(axis=0)` / `A.mean(axis=0)` |
| Row sums / means | `A.sum(axis=1)` / `A.mean(axis=1)` |
| Sample column variance / SD | `A.var(axis=0, ddof=1)` / `A.std(axis=0, ddof=1)` |
| Skip NaNs | `A.nanmean(axis=0)` and other `nan*` methods |
| Extrema | `A.min()` / `A.max()` |
| Truth reductions | `A.any()` / `A.all()` |
| Transpose | `A.T` |
| Store result in the engine | `A.compute()` |
| Collect into Python | `A.to_numpy()` / `A.to_scipy()` |

Implicit sparse zeros participate in reductions. Ordinary reductions propagate
NaN; `nan*` variants skip NaNs. `any` and `all` return Boolean results.
Vectors currently provide only `sum` and `mean` reductions.

## Examples

```python
--8<-- "examples/vignettes/operations.py"
```

```python
--8<-- "examples/vignettes/numerical_surface.py"
```

## Numerical limits

Values are float64. Comparison and `isnan` expressions use numeric 0/1 values.
Variance and SD use a stable formula that can avoid intermediate overflow or
underflow seen in a direct NumPy calculation at extreme magnitudes.

| Input and operation | NumPy reference in the recorded tests | dbnumpy |
|---|---:|---:|
| Variance of `[-1e154, 1e154]` | `inf` | `1e308` |
| SD of `[-1e-200, 1e-200]` | `0.0` | `1e-200` |

These cases are tested on both engines. See [correctness and limits](../architecture.md#correctness-and-limits)
and the [full API contract](../api.md) for options and edge cases.
