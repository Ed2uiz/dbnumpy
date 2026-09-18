"""Python-native scientific matrices backed by analytical databases."""

from dbnumpy._version import __version__
from dbnumpy.backends import (
    Backend,
    BackendCapabilities,
    DataFusionBackend,
    DuckDBBackend,
)
from dbnumpy.exceptions import (
    BackendMismatchError,
    DBArrayError,
    DensificationError,
    UnsupportedOperationError,
)
from dbnumpy.indexing import DBScalar, DBVector
from dbnumpy.io import MatrixMarketHeader, read_matrix_market_header
from dbnumpy.ir import StorageKind
from dbnumpy.matrix import DBArray, DBDenseArray, DBSparseArray

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendMismatchError",
    "DBArray",
    "DBArrayError",
    "DBDenseArray",
    "DBScalar",
    "DBSparseArray",
    "DBVector",
    "DataFusionBackend",
    "DensificationError",
    "DuckDBBackend",
    "MatrixMarketHeader",
    "StorageKind",
    "UnsupportedOperationError",
    "__version__",
    "read_matrix_market_header",
]
