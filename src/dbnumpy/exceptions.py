"""Project-specific exception hierarchy."""


class DBArrayError(Exception):
    """Base exception for dbnumpy."""


class BackendMismatchError(DBArrayError, ValueError):
    """Raised when one expression mixes matrices from different backends."""


class DensificationError(DBArrayError, MemoryError):
    """Raised when an operation would exceed the configured densification cap."""


class UnsupportedOperationError(DBArrayError, NotImplementedError):
    """Raised when a semantic operation has no safe backend lowering."""
