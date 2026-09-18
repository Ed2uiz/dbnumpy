# Indexing

Indices are zero-based. Negative integers count from the end. Slices accept
positive or negative steps and clip to the known shape.

```python
block = matrix[1:4, ::2]  # DBArray
row = matrix[2, :]       # DBVector, shape (n_columns,)
column = matrix[:, 3]    # DBVector, shape (n_rows,)
value = matrix[2, 3]     # DBScalar, shape ()
```

Results stay lazy. Use `row.to_numpy()` or `value.item()` to collect them.
`compute()` stores a result in the engine and preserves its rank.

## Select and reorder

One axis can use a one-dimensional integer or Boolean selector. Order and
repeated indices are preserved.

```python
rows = matrix[[3, 1, 3], :]
columns = matrix[:, [4, 0, 4]]
same_rows = np.take(matrix, [3, 1, 3], axis=0)
```

A Boolean selector must match its axis length. An empty selector produces an
empty axis. Selectors are registered as coordinate maps rather than expanded
into long SQL expressions.

## Select both axes

For a rectangular selection, apply the selectors in sequence:

```python
selected = matrix[[3, 1], :][:, [4, 0]]
```

NumPy's `matrix[[3, 1], [4, 0]]` selects paired points. That form is not
implemented. Neither are full matrix masks, new axes, multidimensional
selectors, flattened `take` or assignment.

## Cost and lifetime

Select before an operation that fills zeros when possible. For example,
`np.exp(A[rows, :])` expands only the selected shape. The compiler does not
automatically reorder it from `np.exp(A)[rows, :]`.

Selector maps remain registered until the backend closes. Identical selectors
reuse a map. Defaults allow 5,000,000 retained positions across 256 maps.
Repeated selections can also increase the number of output rows.

## Example

```python
--8<-- "examples/vignettes/indexing.py"
```
