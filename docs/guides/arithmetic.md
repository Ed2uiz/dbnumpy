# Arithmetic

Use `+`, `-`, `*`, `/`, `**`, comparisons, negation and absolute value on lazy
matrices. NumPy arrays, lists and tuples used as operands are uploaded to the
same backend. Shapes follow NumPy broadcasting rules.

```python
--8<-- "examples/vignettes/arithmetic.py"
```

## Sparse results

An operation stays sparse when its result at a missing zero is still zero.

| Expression | Result at zero | Storage |
|---|---:|---|
| `A * 2` | 0 | Sparse |
| `A > 0` | 0 | Sparse |
| `np.log1p(A)` | 0 | Sparse |
| `A + 1` | 1 | Dense |
| `A == 0` | 1 | Dense |
| `np.exp(A)` | 1 | Dense |

Dense expansion is checked against `max_densify_cells`. Finite row, column
and scalar factors can multiply stored sparse coordinates directly when the
compiler can prove they are finite. See [resource rules](../api.md#resource-behavior).

## Broadcasting

One-dimensional operands align with columns. Use a two-dimensional column
operand to scale rows:

```python
scaled_columns = matrix * np.array([1.0, 2.0, 3.0, 4.0])
scaled_rows = matrix * row_factors[:, None]
```

`DBVector` follows the same rule whether it came from a row or column slice.
Vectors have no public row/column orientation. Scalars broadcast to either.

## Matrix multiplication

`A @ B` and `np.matmul(A, B)` support two-dimensional operands. This path is
experimental. Matrix-vector products, PCA/SVD and further matmul development
are deferred. `A * B` remains elementwise multiplication.
