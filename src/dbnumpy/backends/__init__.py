"""Analytical execution backends."""

from dbnumpy.backends.base import Backend
from dbnumpy.backends.capabilities import BackendCapabilities
from dbnumpy.backends.datafusion import DataFusionBackend
from dbnumpy.backends.duckdb import DuckDBBackend

__all__ = ["Backend", "BackendCapabilities", "DataFusionBackend", "DuckDBBackend"]
