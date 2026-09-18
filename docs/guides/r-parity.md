# R comparison

dbnumpy follows the R `dbMatrix` idea: familiar matrix operations execute in a
database. It uses Python indexing and NumPy broadcasting.

This comparison uses the recorded R audit at `47d63c4`. Python tests port
selected R cases, but do not execute R or compare R-produced fixtures. It is
not a claim of complete cross-language equivalence.

| Capability | R reference | dbnumpy |
|---|---|---|
| Dense/sparse matrices | Supported | Supported |
| Engines | DuckDB | DuckDB and DataFusion |
| Shape, transpose, host conversion | Supported | Supported |
| Scalar/elementwise arithmetic | R dispatch and recycling | Selected operators and NumPy broadcasting |
| Sums, means, variance, SD | Host results | Host results; `ddof=1` for sample variance/SD |
| Math functions | Broad S4 dispatch; some members unsupported or lightly tested | Explicit tested subset listed in the API |
| Min/max/any/all | Source-defined; uneven direct coverage | Tested on both engines |
| Missing values | Method-specific R behavior | Ordinary and explicit `nan*` reductions |
| Indexing | Integer, logical and names | Integer and one-axis Boolean; order/repeats retained |
| Labels | Row/column names | Missing |
| Assignment | Narrow, untested mask/scalar path | Missing; expressions are immutable |
| Membership | Dense `%in%` collects; sparse errors | Missing |
| Event tables and named export | Supported | Missing |
| File I/O and persistence | Broader metadata and export APIs | DuckDB MTX input, DataFusion Parquet scans, backend materialization |
| PCA/SVD | Sparse Arrow/Eigen path with a memory guard; limited numerical tests | Deferred |
| Matrix multiplication | Listed as TODO in the audited vignette | Experimental 2D `@` |
| Value dtypes | Several input storage types | float64 |

## Common translations

| R | Python |
|---|---|
| `dim(A)` | `A.shape` |
| `t(A)` | `A.T` |
| `rowSums(A)` / `colSums(A)` | `A.sum(axis=1)` / `A.sum(axis=0)` |
| `rowMeans(A)` / `colMeans(A)` | `A.mean(axis=1)` / `A.mean(axis=0)` |
| `colVars(A)` / `colSds(A)` | `A.var(axis=0, ddof=1)` / `A.std(axis=0, ddof=1)` |
| `is.na(A)` | `np.isnan(A)` under the float64 contract |
| Cartesian row/column selection | `A[rows, :][:, cols]` |

Python indices start at zero. Negative indices count from the end rather than
exclude positions. Vectors broadcast along the trailing dimension; R recycling
does not apply. Lazy comparison results are currently numeric 0/1.

## Numerical scope

Both support the common logs, exponentials, rounding and trigonometric
functions. Python also has explicit NaN-skipping reductions. Cumulative
functions, exact product, inverse hyperbolic functions and special functions
remain incomplete or deferred; R generic registration alone does not establish
that every member works.

The Python sparse rules correct audited R gaps such as `exp(A)` and `A == 0`:
missing coordinates must become nonzero values. Sparse means and variances
include structural zeros. See the [R behavior notes](../reference/dbmatrix-r-behavior.md).

## Single-cell workflows

Counts, detection totals, filtering, fixed-target normalization, `log1p`, means
and variance can be expressed with the current Python operations. Reductions
return host vectors. Centering a sparse matrix generally makes it dense.

PCA/SVD remains deferred.
