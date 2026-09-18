# R behavior reference

The original audit used R `dbMatrix` at local `dev` commit
`47d63c42086a9277cf97d7d95cb4557042817f8c`. It included two local commits beyond
the public ancestor `6d5188c972e013a077588cfe87ab02ab26abf149`.

Public reference files:

- [Classes](https://github.com/dbverse-org/dbmatrix-r/blob/6d5188c972e013a077588cfe87ab02ab26abf149/R/classes.R)
- [Operations](https://github.com/dbverse-org/dbmatrix-r/blob/6d5188c972e013a077588cfe87ab02ab26abf149/R/operations.R)
- [Construction](https://github.com/dbverse-org/dbmatrix-r/blob/6d5188c972e013a077588cfe87ab02ab26abf149/R/dbMatrix.R)
- [Indexing](https://github.com/dbverse-org/dbmatrix-r/blob/6d5188c972e013a077588cfe87ab02ab26abf149/R/extract.R)
- [Tests](https://github.com/dbverse-org/dbmatrix-r/tree/6d5188c972e013a077588cfe87ab02ab26abf149/tests/testthat)

## What Python retains

Lazy dense/sparse objects, coordinate storage, separate shape metadata,
implicit zeros, arithmetic, transpose, summaries and explicit collection.
Pointwise work over one source should avoid a self-join.

Selected R cases were translated into Python tests. Those tests use NumPy and
SciPy references; there is no live R runner or R-produced fixture set.

## Intentional differences

Python uses zero-based indexing and broadcasting. R uses one-based indexing,
negative exclusion and recycling. Python has lazy vector/scalar index results.
R's current row/column statistics collect vectors even where `memory=FALSE`
appears in the API.

R missing-value defaults vary by method. Python instead distinguishes ordinary
reductions from `nan*` reductions. Both lazy missing-value predicates use
numeric 0/1 storage in the audited implementations.

## Audited gaps

R transformed only stored sparse rows for operations such as `exp(A)`,
`cos(A)` and `A == 0`. That misses the changed value at absent coordinates.
Python checks that value and expands the shape when needed.

The audited R matrix-comparison shape check accepted a match in only one
dimension. Python requires broadcast-compatible shapes.

R's masked assignment and some `head`/`tail` paths are source-defined but
untested. Its SVD/PCA tests check shapes/types and arguments, not numerical
factors. Those paths are not complete numerical reference implementations.

See the [feature comparison](../guides/r-parity.md) for the supported surface.
