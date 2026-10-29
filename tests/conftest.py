from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend, DataFusionBackend, DuckDBBackend, PolarsBackend


@pytest.fixture(
    params=[DuckDBBackend, DataFusionBackend, PolarsBackend],
    ids=["duckdb", "datafusion", "polars"],
)
def backend(request: pytest.FixtureRequest) -> Iterator[Backend]:
    backend_type = request.param
    instance = backend_type.connect(max_densify_cells=10_000)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def upload(backend: Backend) -> Callable[..., tuple[Any, np.ndarray]]:
    """Upload ``values`` and return the matrix with the values it stores.

    Sparse storage drops explicit zeros, so ``-0.0`` comes back as ``+0.0``.
    """

    def _upload(values: Any, *, sparse_input: bool) -> tuple[Any, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        if not sparse_input:
            return backend.from_numpy(values), values
        return backend.from_scipy(sparse.csr_array(values)), values + 0.0

    return _upload
