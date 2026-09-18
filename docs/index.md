# dbnumpy

dbnumpy provides lazy matrix operations in DuckDB and DataFusion. Python
expressions build a plan. The database runs it when you collect a result,
request a reduction or call `compute()`.

This is a pre-alpha package with a limited NumPy API. It uses float64 values
and supports matrices, plus vectors and scalars returned by indexing.

## Start here

- [Install and run](getting-started.md)
- [Examples](guides/overview.md)
- [API and limits](api.md)
- [How the package works](architecture.md)
- [R comparison](guides/r-parity.md)

Both engines run the shared NumPy/SciPy reference tests in
[CI](https://github.com/Ed2uiz/dbnumpy/actions). CI also checks types, lint,
examples, documentation and installed packages.
