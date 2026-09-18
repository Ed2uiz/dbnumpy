# How dbnumpy works

dbnumpy records matrix calculations, turns them into database queries, and runs
those queries in DuckDB or DataFusion.

```text
NumPy-style expression → saved matrix steps → database query → result
```

## What is the IR?

IR means **intermediate representation**. In dbnumpy it is simply the saved
steps of a calculation. Each step names an operation and refers to its inputs.

```python
import numpy as np
import dbnumpy as dnp

with dnp.DuckDBBackend.connect() as backend:
    x = backend.from_numpy(np.arange(12.0).reshape(3, 4))
    y = np.sqrt(x + 1)[:, ::2]
    result = y.to_numpy()
```

| Expression | Saved step |
|---|---|
| `x` | Read the registered input with shape `(3, 4)` |
| `x + 1` | Add one to each value |
| `np.sqrt(...)` | Take the square root |
| `[:, ::2]` | Keep every row and columns 0 and 2 |

`from_numpy()` uploads the input. The next steps record the calculation.
`to_numpy()` runs it and returns the values.

The saved steps contain operations, constants, input references, and matrix
metadata. They contain no database expression objects. Adding a new operation
leaves the earlier steps unchanged. Two branches can refer to the same step.

## Why keep these steps?

They let us check matrix rules before writing a database query. For example,
missing sparse entries mean zero. `sqrt(0)` remains zero, but `exp(0)` becomes
one and may produce a much larger result. Both engines need the same rule.

We keep this small plan because it already describes the supported matrix
operations and gives us one place to inspect their meaning. The query builders
share input ordering and operand rules, while each engine handles its own query
syntax and execution.

A package could also keep Ibis expressions directly and put matrix rules in a
shared Python layer. Our own plan is a design choice, not a requirement for
lazy execution or multiple databases. We do not aim to build a universal plan
format for every language or matrix library. That would need a separate design
and real users outside this package.

## How a query is built

The code that turns saved steps into a query is sometimes called a *compiler*.
It reads the plan; it is not the plan itself.

| Query builder | Used for |
|---|---|
| Ibis | Joins, indexing, broadcasting, and reductions, translated into SQL |
| Small SQL writer | Elementwise calculations, including long chains and shared steps |
| Native DataFusion expressions | Elementwise calculations using one source in DataFusion |

Ibis builds database queries from Python expressions. It uses SQLGlot to write
SQL for each database. dbnumpy also uses SQLGlot for a narrow integer-division
rewrite, so slice coordinates stay exact. SQLGlot does not define our matrix
rules.

The SQL and native DataFusion paths share the code that visits inputs and
orders operands. Engine-specific numerical expressions remain separate where
the engines behave differently. Tests compare their results with NumPy.

DataFusion chooses its execution path in one place. Collection, `compute()`,
and `explain()` use that choice. `compile()` always returns equivalent SQL;
it does not claim that DataFusion will execute that SQL text.

Long relational plans use named SQL stages to limit query nesting. A stage is
part of the same query, not a stored intermediate result. The database still
chooses how to execute it. Stages do not guarantee that shared work runs only
once, or that every deep expression will fit in memory.

## Inspecting and running a calculation

| Method | Meaning |
|---|---|
| `y.plan()` | Describe the saved steps as a Python dictionary |
| `y.compile()` | Generate SQL without running the value query |
| `y.explain()` | Show the selected engine plan |
| `y.compute()` | Run the calculation and keep its result in the database |
| `y.to_numpy()` | Run the calculation and return a NumPy array |
| `y.mean(axis=0)` | Run a reduction and return a NumPy vector immediately |

`plan()` refers to data in the current backend. Saving the dictionary does not
save the data or make a portable, runnable file. `compute()` returns another
lazy array backed by the stored result. Reusing an expression can reuse its
cached query; rebuilding an equivalent expression may require a new query.

DuckDB stores computed results in temporary tables and manages their memory
and disk use. DataFusion writes computed results directly to temporary Parquet
files through its streaming writer, then registers a scan of those files.
It does not collect the whole result in Python or keep it in an in-memory cache.
DataFusion currently writes even small computed results to disk, which adds I/O.
Its query operators can also spill intermediate data; that is separate from
storing the completed matrix.

For external Parquet input, dbnumpy disables DataFusion's file statistics
collection when defining the scan. DataFusion 54 can otherwise replace NaN
values with a finite constant because Parquet min/max statistics omit NaNs.
The scan stays lazy, uses the backend's execution limits, and can still prune
row groups. This protection may reduce statistics-based query optimizations.
Computed files separately omit value-column statistics when written.

Computed files belong to the backend and are removed when it closes. Failed
or interrupted writes remove their partial files. Use the backend as a context
manager and set `temp_directory` to an existing writable directory if the system
temporary directory is unsuitable. Abrupt process termination can leave files
behind. Input files supplied by the caller are never deleted.

Building an expression can upload host operands or register index tables.
The matrix calculation itself waits until an execution method is called.

## Correctness and limits

Both engines store coordinates `i` and `j` as int64 and values `x` as float64.
Dense storage has a row for every coordinate. Sparse storage can omit zeros.
The shape is stored separately, preserving empty dimensions and all-zero rows.
SQL NULL values become NaN. Dense inputs preserve signed zeros; sparse imports
remove explicit zeros, including negative zero.

Coordinate tables suit sparse data but add overhead for dense arrays. Filling
missing sparse coordinates must fit the configured dense-size limit. Output
limits do not cap all query costs or process memory. The backend owns imported
and temporary data until it closes.

Expression depth is capped at 768. Some query shapes reach engine or query
builder limits sooner. `compute()` can shorten a long calculation by storing
an intermediate result. See [API and limits](api.md) for the full contract.

## How we check the results

CI checks both engines against NumPy and SciPy. Tests include empty shapes,
repeated indices, sparse zeros, NaN, infinities, signed zeros, and branches
that reuse earlier results. Seeded mixed calculations exercise combinations
of operations as well as individual functions.

A separate Linux check runs long and shared expressions in fresh processes.
It compares explicit SQL and normal execution, changes Python hash seeds and
engine parallelism, and enforces time and sampled process-memory limits.
These checks cover selected workloads, not every possible expression.

Numerical agreement uses explicit tolerances and separate checks for NaN and
infinity. Parallel reductions can change the last few bits because floating
point addition depends on order. We do not promise identical result bytes
across engines, versions, machines, or thread counts.
