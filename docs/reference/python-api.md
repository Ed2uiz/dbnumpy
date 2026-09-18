# Python reference

Signatures and docstrings from the source. See [API and limits](../api.md)
for supported behavior.

## Matrix objects

::: dbnumpy.matrix.DBArray
    options:
      members:
        - backend
        - shape
        - dtype
        - ndim
        - size
        - storage
        - T
        - __len__
        - __bool__
        - __getitem__
        - __array__
        - __array_ufunc__
        - __array_function__
        - __neg__
        - __abs__
        - __add__
        - __radd__
        - __sub__
        - __rsub__
        - __mul__
        - __rmul__
        - __truediv__
        - __rtruediv__
        - __pow__
        - __rpow__
        - __gt__
        - __ge__
        - __lt__
        - __le__
        - __eq__
        - __ne__
        - __matmul__
        - __rmatmul__
        - transpose
        - take
        - compile
        - explain
        - plan
        - compute
        - to_numpy
        - toarray
        - to_scipy
        - sum
        - mean
        - var
        - std
        - min
        - max
        - any
        - all
        - nansum
        - nanmean
        - nanvar
        - nanstd
        - nanmin
        - nanmax
        - sqrt
        - exp
        - expm1
        - log
        - log1p
        - sin
        - cos
        - tan
        - arcsin
        - asin
        - arccos
        - acos
        - arctan
        - atan
        - floor
        - ceil
        - sign
        - trunc
        - isnan
        - log2
        - log10
        - sinh
        - cosh
        - tanh

::: dbnumpy.matrix.DBDenseArray

::: dbnumpy.matrix.DBSparseArray

## Ranked indexing results

::: dbnumpy.indexing.DBVector
    options:
      members:
        - backend
        - shape
        - dtype
        - ndim
        - size
        - storage
        - T
        - __getitem__
        - compile
        - explain
        - plan
        - compute
        - to_numpy
        - transpose
        - sum
        - mean

::: dbnumpy.indexing.DBScalar
    options:
      members:
        - backend
        - shape
        - dtype
        - ndim
        - size
        - storage
        - compile
        - explain
        - plan
        - compute
        - item
        - to_numpy

## Backends

::: dbnumpy.backends.base.Backend
    options:
      members:
        - name
        - capabilities
        - from_numpy
        - from_coo
        - from_scipy
        - from_relation
        - compile
        - collect_matrix
        - collect_coordinates
        - collect_reduction
        - explain
        - materialize
        - close

::: dbnumpy.backends.duckdb.DuckDBBackend
    options:
      members:
        - connect
        - from_mtx

::: dbnumpy.backends.datafusion.DataFusionBackend
    options:
      members:
        - connect
        - from_parquet

## Capabilities and storage

::: dbnumpy.backends.capabilities.BackendCapabilities

::: dbnumpy.ir.StorageKind

## Exceptions

::: dbnumpy.exceptions
    options:
      members:
        - DBArrayError
        - BackendMismatchError
        - DensificationError
        - UnsupportedOperationError
