from __future__ import annotations

from collections.abc import Iterator

import pytest

from dbnumpy.backends import Backend, DataFusionBackend, DuckDBBackend


@pytest.fixture(params=[DuckDBBackend, DataFusionBackend], ids=["duckdb", "datafusion"])
def backend(request: pytest.FixtureRequest) -> Iterator[Backend]:
    backend_type = request.param
    instance = backend_type.connect(max_densify_cells=10_000)
    try:
        yield instance
    finally:
        instance.close()
