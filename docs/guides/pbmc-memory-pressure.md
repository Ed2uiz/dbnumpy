# PBMC preprocessing under memory limits

This pilot tests the [same preprocessing function](pbmc-benchmark.md) with
hard limits of 1 GiB and 2 GiB. It starts with the same in-memory input and
collects the full sparse output. It does not test database-resident input.

## Pilot results

One trial per configuration, in WSL 2 with Python 3.14.7. These are exploratory timings, not three-repeat medians.

| RAM ceiling | Cells | NumPy / SciPy | dbnumpy DuckDB | dbnumpy DataFusion |
|---|---:|---:|---:|---:|
| 1 GiB | 10,000 | 1.23 s | 8.77 s | 5.33 s |
| 1 GiB | 30,000 | Memory limit (workflow) | Memory limit (setup) | Memory limit (setup) |
| 1 GiB | 68,579 | Memory limit (workflow) | Memory limit (setup) | Memory limit (setup) |
| 2 GiB | 10,000 | 1.08 s | 7.62 s | 5.01 s |
| 2 GiB | 30,000 | 3.94 s | 18.08 s | Memory limit (sparse_collection) |
| 2 GiB | 68,579 | Memory limit (workflow) | Memory limit (setup) | Memory limit (setup) |

8 trials passed full numerical validation; 10 were stopped by the kernel at the memory ceiling. No failure is shown as a runtime.

Lowering the memory limit did not establish a dbnumpy advantage in this in-memory-input workflow. Database workers still need the original SciPy input and additional buffers during loading. A larger cell count alone does not remove that cost.

[Raw measurements and failure phases](../assets/pbmc-memory-pressure-results.json)

## Method

Each worker runs in a fresh Linux cgroup with `memory.max` set to the ceiling
and `memory.swap.max=0`. The limit covers imports, input reading, setup,
calculations, collection and saving the output. It covers the worker's memory
and attributable file cache. Filesystem caches are not flushed between trials.
The operating system enforces the cap; it is not a sampled RSS watchdog.

Both databases receive an internal execution budget equal to half the ceiling:
512 MiB under 1 GiB, and 1 GiB under 2 GiB. The remainder is available for
Python, inputs and results, but is not a guarantee that those allocations fit.
NumPy's thread libraries use one thread; DuckDB uses one thread and DataFusion
uses one partition. Each worker has a 300-second deadline.

The scientific function contains no SQL or backend branches. As in the original
benchmark, timing includes setup, preprocessing and full sparse collection.
Imports, input reading, connection cleanup, saving and verification are outside
the timer. Small phase markers identify where a worker stops.

Successful outputs are saved and checked in a separate process outside the
memory cap. This prevents validation from charging a worker for a second full
reference matrix. Every output value and coordinate is checked against the
NumPy/SciPy reference. Selections and count summaries must match exactly. Gene
means and variances are checked against independent NumPy calculations on
Scanpy output, with `rtol=1e-9` and `atol=1e-11`.

The runner records the kernel's memory peak and out-of-memory counters, the
last recorded phase, source hashes, and errors. A killed process has no runtime
result. A completed calculation without successful verification is not a pass.

## Reproduce

First prepare and validate the inputs using the commands in the
[original benchmark](pbmc-benchmark.md#reproduce). Run the supervisor as root
inside Linux or WSL 2, using the Python environment containing the dependencies:

```bash
sudo /absolute/path/to/venv/bin/python benchmarks/pbmc_memory_pressure.py run \
  --data benchmark-results/pbmc/prepared \
  --output benchmark-results/pbmc/memory-pressure \
  --limits 1 2 --sizes 10000 30000 68579 --repeats 1 \
  --uid "$(id -u)" --gid "$(id -g)"
```

The supervisor needs permission to create cgroups. Workers drop to the supplied
non-root user and group before importing scientific packages. Only dedicated,
temporary cgroups are changed; system-wide memory and swap settings are untouched.
The output directory must be new, and writable by the worker user.

## Larger dataset

A suitable next dataset is 10x Genomics'
[1.3 Million Brain Cells from E18 Mice](https://www.10xgenomics.com/datasets/1-3-million-brain-cells-from-e-18-mice-2-standard-1-3-0).
It contains 1,306,127 cells from two mice and is licensed under CC BY 4.0.
The [filtered HDF5 matrix](https://s3-us-west-2.amazonaws.com/10x.files/samples/cell/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5)
is 4,216,018,749 bytes; its download endpoint was checked for this pilot.
It has not been downloaded or benchmarked here.

For that experiment, use nested samples of actual cells, for example 100,000,
250,000, 500,000 and the full dataset. Record the number of nonzero values too:
cell count alone does not describe the amount of computation.

A separate database-resident comparison would let DuckDB read stored tables
and DataFusion scan coordinate Parquet, avoiding a full SciPy input in each
database worker. Keep the same scientific function and verify equivalent
outputs. Report file preparation, input loading and computation separately.
Collecting the full result would still require host memory. A summary-only
experiment can test a different use case, but should be labeled separately.
