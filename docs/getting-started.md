# Get started

Use Python 3.12 or newer on Linux, macOS or WSL 2:

```bash
git clone https://github.com/Ed2uiz/dbnumpy.git
cd dbnumpy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -e '.[duckdb,datafusion,scipy]'
```

Choose an engine and keep its connection open while using lazy results:

```python
import numpy as np
from dbnumpy import DuckDBBackend

with DuckDBBackend.connect() as backend:
    x = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    print(np.log1p(x).to_numpy())
```

Replace `DuckDBBackend` with `DataFusionBackend` for the same matrix API.
Input APIs differ: DuckDB reads Matrix Market files; DataFusion can scan
canonical coordinate Parquet. See [file input](guides/out-of-core-mtx.md).

## Examples and docs

Each example checks its results against NumPy or SciPy:

```bash
for example in examples/vignettes/*.py; do python "$example"; done
```

To serve the docs locally:

```bash
python -m pip install -e '.[docs]'
mkdocs serve
```

Open <http://127.0.0.1:8000/>. Use `mkdocs build --strict` to check the site.
See [API and limits](api.md) before adapting an existing NumPy workflow.
