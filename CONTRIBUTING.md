# Contributing

Use Python 3.12 or newer. Install both engines for development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -e '.[dev,docs]'
```

## Checks

Run these before opening a pull request:

```bash
ruff check src tests benchmarks examples scripts
mypy src/dbnumpy
pytest --cov=dbnumpy --cov-report=term-missing --cov-fail-under=80
python scripts/query_checks.py  # Linux or WSL 2
for example in examples/vignettes/*.py; do python "$example"; done
mkdocs build --strict
python -m build
python scripts/sdist_content_smoke.py dist/dbnumpy-0.4.0a0.tar.gz
```

Check installed packages in a clean environment too:

```bash
python -m venv .wheel-smoke
.wheel-smoke/bin/python -m pip install --only-binary=:all: 'dist/dbnumpy-0.4.0a0-py3-none-any.whl[duckdb,datafusion,scipy]'
.wheel-smoke/bin/python -m pip check
.wheel-smoke/bin/python scripts/artifact_smoke.py
.wheel-smoke/bin/python -m pip install --no-deps --force-reinstall dist/dbnumpy-0.4.0a0.tar.gz
.wheel-smoke/bin/python scripts/artifact_smoke.py
```

CI runs tests on Python 3.12, 3.13 and 3.14. Tests run serially with bounded
engine memory and thread counts. Do not use `pytest-xdist` by default.

A separate CI job installs `benchmarks/requirements-pbmc.txt` and checks the
shared workflow and independent benchmark reference on small synthetic data.
It does not download datasets or run the large benchmarks.

The query checks use fresh processes, different hash seeds, and one or two
engine threads/partitions. They compare both execution paths with NumPy and
check stable SQL for fixed inputs. Each process has a 60-second timeout and
a sampled 1.5 GiB memory limit. These catch regressions in the tested workloads;
they do not guarantee the same limits for other queries or machines.

## Code and docs

- Keep matrix rules in the operation plan (IR), separate from SQL generation.
  See [how the IR works](docs/architecture.md#what-is-the-ir).
- Keep IR nodes immutable and free of engine objects or Python callbacks.
- Test supported behavior on both engines against NumPy or SciPy.
- For sparse operations, test what happens at missing coordinates.
- Check generated plans when a change promises fewer scans or joins.
- Use database expressions instead of Python UDFs for relational operations.
- Measure a compiler bottleneck before adding another execution path.
- Keep prose short. Record limits, decisions and measured results clearly.

Run `mkdocs serve` to preview docs at <http://127.0.0.1:8000/>.
The generated `site/` directory is ignored. Documentation is not auto-published.

The R `dbMatrix` source is a behavior reference. Python keeps NumPy indexing,
broadcasting and missing-value conventions. R comparisons are documented in
`docs/guides/r-parity.md`.

The project uses the MIT license. Keep changes focused and update the API
documentation when supported behavior changes.
