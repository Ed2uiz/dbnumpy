# Computing and storing the normalized matrix

The summary benchmark already computes the filtering, normalization and gene statistics. This follow-up adds `result.values.compute()` after the same workflow. It stores the complete normalized matrix inside the database, without exporting that matrix to Python.

![Full preprocessing runtime including compute](../assets/brain-native-compute.png)

## Results with 14 threads (October 29, 2026)

Measured with dbnumpy `dce82b4`, DuckDB 2.0.0, DataFusion 55.0.0, Polars 2.1.0,
NumPy 2.5.3 and SciPy 1.18.1 on Python 3.12, in WSL 2 on an Intel Core Ultra 5
125U. Every engine and BLAS library uses all 14 CPU threads.

| Input cells | Method | Median total (range) | Of which `compute()` | Peak memory |
|---:|---|---:|---:|---:|
| 5,000 | NumPy / SciPy | 1.9 s (1.9–1.9) | — | 0.98 GiB |
| 5,000 | DuckDB 2.0 | 0.7 s (0.6–0.7) | 0.1 s | 0.48 GiB |
| 5,000 | DataFusion 55 | 1.0 s (1.0–1.0) | 0.3 s | 0.59 GiB |
| 5,000 | Polars 2.1 | 1.4 s (1.4–1.4) | 0.3 s | 0.69 GiB |
| 10,000 | NumPy / SciPy | 3.9 s (3.9–4.8) | — | 1.88 GiB |
| 10,000 | DuckDB 2.0 | 0.9 s (0.9–0.9) | 0.2 s | 0.76 GiB |
| 10,000 | DataFusion 55 | 1.4 s (1.4–1.4) | 0.5 s | 0.74 GiB |
| 10,000 | Polars 2.1 | 2.2 s (2.2–2.2) | 0.5 s | 1.05 GiB |
| 25,000 | NumPy / SciPy | 12.7 s (11.0–14.6) | — | 4.58 GiB |
| 25,000 | DuckDB 2.0 | 1.9 s (1.8–2.1) | 0.6 s | 1.57 GiB |
| 25,000 | DataFusion 55 | 3.4 s (3.4–3.6) | 1.3 s | 1.04 GiB |
| 25,000 | Polars 2.1 | 4.9 s (4.8–5.0) | 1.3 s | 1.08 GiB |
| 50,000 | NumPy / SciPy | × | — | — |
| 50,000 | DuckDB 2.0 | 3.3 s (3.3–4.4) | 1.0 s | 2.93 GiB |
| 50,000 | DataFusion 55 | 6.8 s (6.7–6.9) | 2.7 s | 1.63 GiB |
| 50,000 | Polars 2.1 | 9.5 s (9.5–9.9) | 2.6 s | 1.99 GiB |
| 100,000 | NumPy / SciPy | × | — | — |
| 100,000 | DuckDB 2.0 | 7.3 s (6.9–8.3) | 2.8 s | 4.09 GiB |
| 100,000 | DataFusion 55 | 14.3 s (13.9–14.3) | 5.4 s | 2.33 GiB |
| 100,000 | Polars 2.1 | 18.8 s (18.8–18.9) | 5.2 s | 2.14 GiB |
| 250,000 | NumPy / SciPy | × | — | — |
| 250,000 | DuckDB 2.0 | 20.9 s (20.5–31.4) | 10.0 s | 4.25 GiB |
| 250,000 | DataFusion 55 | 35.7 s (35.3–35.8) | 14.1 s | 4.00 GiB |
| 250,000 | Polars 2.1 | 48.4 s (48.0–56.9) | 14.0 s | 2.27 GiB |

× means all three runs ran out of memory. All 63 completed runs passed the
correctness checks below. The protocol is the same as the one-thread runs below,
except for the thread count and that Polars, which has no memory limit, is only
bounded by the process cap.

DuckDB gains the most from more threads: 20.9 s at 250,000 cells, against 79.1 s
with one thread. DataFusion takes 35.7 s, against 60.2 s with one thread.
dbnumpy raises DataFusion's broadcast-join limits so that small selector and
scale-vector tables are copied to every partition; with DataFusion's defaults
the whole coordinate relation was hash-repartitioned for each such join, and
the same run took 84.2 s. Polars uses the least memory from 100,000 cells.

[Raw results, versions and source hashes](../assets/brain-native-compute-threads-results.json).

## Results with one thread (October 2026)

Measured with dbnumpy `ffef40a`, DuckDB 2.0.0, DataFusion 54.0.0, NumPy 2.5.3 and
SciPy 1.18.1 on Python 3.12, in WSL 2 on an Intel Core Ultra 5 125U.

| Input cells | Method | Median total (range) | Of which `compute()` | Peak memory |
|---:|---|---:|---:|---:|
| 5,000 | NumPy / SciPy | 2.0 s (2.0–2.4) | — | 0.98 GiB |
| 5,000 | DuckDB 2.0 | 1.7 s (1.6–2.1) | 0.5 s | 0.45 GiB |
| 5,000 | DataFusion 54 | 2.2 s (2.2–2.2) | 0.5 s | 0.46 GiB |
| 10,000 | NumPy / SciPy | 4.2 s (4.1–5.0) | — | 1.88 GiB |
| 10,000 | DuckDB 2.0 | 3.0 s (2.9–3.2) | 1.0 s | 0.72 GiB |
| 10,000 | DataFusion 54 | 2.3 s (2.3–2.3) | 0.7 s | 0.60 GiB |
| 25,000 | NumPy / SciPy | 15.4 s (11.6–18.4) | — | 4.58 GiB |
| 25,000 | DuckDB 2.0 | 7.3 s (7.1–7.8) | 2.6 s | 1.53 GiB |
| 25,000 | DataFusion 54 | 6.3 s (6.3–6.3) | 1.8 s | 0.77 GiB |
| 50,000 | NumPy / SciPy | × | — | — |
| 50,000 | DuckDB 2.0 | 14.3 s (14.0–15.4) | 5.1 s | 2.89 GiB |
| 50,000 | DataFusion 54 | 10.6 s (10.6–10.9) | 3.2 s | 1.00 GiB |
| 100,000 | NumPy / SciPy | × | — | — |
| 100,000 | DuckDB 2.0 | 29.2 s (28.2–31.4) | 10.9 s | 4.02 GiB |
| 100,000 | DataFusion 54 | 21.1 s (21.1–21.2) | 6.0 s | 1.16 GiB |
| 250,000 | NumPy / SciPy | × | — | — |
| 250,000 | DuckDB 2.0 | 79.1 s (76.6–95.6) | 33.6 s | 4.15 GiB |
| 250,000 | DataFusion 54 | 60.2 s (60.0–61.0) | 16.7 s | 1.80 GiB |

× means all three runs ran out of memory. All 45 completed runs passed the
correctness checks below.

How the runs were made:

- Three repeats per point; the table and figure show the median.
- Each run is a fresh process in a systemd scope capped at 7.51 GiB (the WSL
  total) with no swap. DuckDB's memory limit and DataFusion's memory pool are set
  to half of that. Engines use one thread.
- Each repeat runs one method through every size before the next method starts,
  and the method order rotates between repeats. Pending writes are flushed and the
  run's input files are dropped from the page cache before every run.
- A run starts only if no other process is using more than 20% of a CPU core.
- Inputs are read from the Linux filesystem.

[Raw results, versions and source hashes](../assets/brain-native-compute-isolated-results.json).

## September 2026 results

These earlier runs used DuckDB 1.5.5, one run per point, and are kept for
reference. They are not comparable with the October results above.

### Updated DataFusion results

DataFusion now stores computed matrices in temporary Parquet files. It completed
250,000 input cells in 107.97 seconds, including 25.54 seconds for `compute()`.
The scientific function, inputs, dependency versions and reference data are
unchanged. All six rerun points passed the summary, entry-count and sampled-value
checks described below.

| Input cells | Through summaries (s) | Added compute (s) | Total (s) |
|---:|---:|---:|---:|
| 5,000 | 4.30 | 0.76 | 5.06 |
| 10,000 | 4.87 | 1.00 | 5.87 |
| 25,000 | 10.26 | 3.00 | 13.27 |
| 50,000 | 16.82 | 4.62 | 21.45 |
| 100,000 | 29.15 | 9.17 | 38.32 |
| 250,000 | 82.43 | 25.54 | 107.97 |

The figure uses all six Parquet-backed DataFusion measurements and the earlier
NumPy/SciPy and DuckDB measurements. The 10,000, 25,000 and 50,000-cell cases
were added in a second run with the same scientific code, package source,
dependency versions, data and validation checks. No cache-based DataFusion
measurements are mixed into the updated line.

The rerun used Linux temporary storage. Large writes to the Windows-mounted
directory failed with `Upload aborted`; the same writer succeeded on Linux.
The earlier runs used Windows-mounted temporary storage. This difference and
the single trial per point prevent a controlled comparison of relative speed.

The [combined result record](../assets/brain-datafusion-disk-results.json) identifies
the source run for every point. The [initial three points](../assets/brain-datafusion-initial-results.json)
were recovered from captured supervisor output after their temporary directory
disappeared; their original validation files are not retained. The
[three added points](../assets/brain-datafusion-missing-results.json) come from the
original result JSON, with their logs and validation files archived locally
before the WSL session ended.

### Original results: in-memory DataFusion cache

These results describe the previous implementation and are retained for comparison.

![Original paired timings](../assets/brain-native-compute-original.png)

One trial per engine and size, with a 7.510 GiB WSL RAM ceiling. 14 of 18 trials completed the stored-matrix endpoint and its checks. 15 trials produced verified workflow summaries.

At 100,000 cells, compute added 28.02 seconds for DuckDB (70.4%) and 7.39 seconds for DataFusion (26.3%). DuckDB also completed 250,000 cells in 185.01 seconds including compute. DataFusion ran out of memory while caching that largest result. SciPy ran out of memory during the workflow at 50,000 cells and above.

Both columns come from the same process. The added time is measured around `compute()` itself; it is not the difference between separate benchmark runs. SciPy's matrix is already computed, so its added time is zero.

| Input cells | Engine | Through summaries (s) | Added compute (s) | Added time (%) | Total (s) |
|---:|---|---:|---:|---:|---:|
| 5,000 | NumPy / SciPy | 2.92 | 0.00 | 0.0% | 2.92 |
| 5,000 | DuckDB | 5.47 | 0.83 | 15.2% | 6.30 |
| 5,000 | DataFusion | 4.20 | 0.59 | 14.0% | 4.79 |
| 10,000 | NumPy / SciPy | 5.86 | 0.00 | 0.0% | 5.86 |
| 10,000 | DuckDB | 7.39 | 1.75 | 23.7% | 9.14 |
| 10,000 | DataFusion | 4.54 | 0.77 | 16.9% | 5.30 |
| 25,000 | NumPy / SciPy | 15.58 | 0.00 | 0.0% | 15.58 |
| 25,000 | DuckDB | 12.71 | 4.36 | 34.3% | 17.07 |
| 25,000 | DataFusion | 9.20 | 2.01 | 21.9% | 11.21 |
| 50,000 | NumPy / SciPy | × | × | — | × |
| 50,000 | DuckDB | 21.88 | 8.76 | 40.1% | 30.64 |
| 50,000 | DataFusion | 14.62 | 3.54 | 24.2% | 18.16 |
| 100,000 | NumPy / SciPy | × | × | — | × |
| 100,000 | DuckDB | 39.80 | 28.02 | 70.4% | 67.82 |
| 100,000 | DataFusion | 28.13 | 7.39 | 26.3% | 35.52 |
| 250,000 | NumPy / SciPy | × | × | — | × |
| 250,000 | DuckDB | 93.08 | 91.93 | 98.8% | 185.01 |
| 250,000 | DataFusion | 78.47 | × | — | × |

× means the endpoint did not complete and pass validation. It is not a measured runtime.

| Failed trial | Failure | Last stage | Stored-matrix endpoint reached? |
|---|---|---|---|
| NumPy / SciPy, 50,000 cells | memory_limit | workflow | No |
| NumPy / SciPy, 100,000 cells | memory_limit | workflow | No |
| NumPy / SciPy, 250,000 cells | memory_limit | workflow | No |
| DataFusion, 250,000 cells | memory_limit | native_compute | No |

## What the extra step does

- DuckDB executes the matrix expression into a temporary table. Its buffer manager can spill to the configured temporary directory.
- The original DataFusion run used `cache()` to retain an in-memory table. The updated implementation streams the result to temporary Parquet files and registers a scan of those files. The backend removes its result files when it closes.
- SciPy retains its existing sparse result. No artificial copy or extra calculation is added.

The database paths evaluate the normalization expression again to store it. The earlier reductions return gene summaries, but do not retain the whole normalized matrix. These timings measure the current API sequence, including that repeated work. Computing before the final reductions might give a different total; that is a separate experiment.

## Correctness checks

Validation runs after the timer. Workflow cell/gene selections and count vectors are compared exactly with the independent reference. Means and sample variances are checked with `rtol=1e-9`, `atol=1e-11`.

After materialization, all gene means and variances are calculated again from the stored matrix. Its positive entry count is checked against the reference. All stored input counts are positive and normalization preserves positivity. Sixty-four complete rows, spread across the result, are then collected and checked coordinate by coordinate against independent Scanpy normalization. This bounded sample is outside the timer.

The largest absolute error in sampled normalized values was 8.88e-16. The large runs do not compare every matrix entry. A separate small fixture does compare every entry for all three engines, including zero cells, a rare gene and unsorted source coordinates. The test also confirms that the verifier rejects an intentionally altered value.

## Conditions and limits (September runs)

- Same nested samples of actual cells from the 10x 1.3M mouse brain dataset: 5,000 through 250,000 input cells. The entire 1.3M-cell matrix was not benchmarked.
- The scientific function, package source, prepared dataset and independent references match the previous full-RAM study. No handwritten scientific SQL was added.
- Input opening is timed. Downloading, preparing input formats, imports and correctness checks are excluded. The full matrix is never exported during this benchmark.
- Workers run serially in fresh processes. Each cgroup has the full WSL RAM ceiling and no swap; database memory budgets are half that ceiling. WSL itself has overhead, so Linux can kill a worker before it reaches its cgroup ceiling.
- BLAS/OpenMP thread settings, DuckDB threads and DataFusion target partitions are set to one. This does not establish identical CPU use across engines; CPU affinity was not imposed.
- Each worker has a 600-second wall deadline, including imports and the post-compute checks. The external reference comparison runs separately.
- Linux input-cache eviction is requested before each trial; the Windows host cache is uncontrolled. The engines read different prepared formats.
- This is an ordinary eager NumPy/SciPy baseline, not an optimized in-place or blockwise implementation. One trial per point cannot establish stable speedups.
- An initial pilot was stopped after the validator's COO row selection was found to allocate an unnecessarily large temporary. Sampling was corrected outside the timed section, the fixture was rerun, and all 18 reported trials were restarted with the corrected runner. The interrupted pilot is excluded.

## Reproduce

To regenerate the current figure from the 14-thread measurements:

```sh
python benchmarks/plot_brain_compute.py \
  --results docs/assets/brain-native-compute-threads-results.json \
  --output docs/assets/brain-native-compute
```

To plot the one-thread October measurements instead, pass
`docs/assets/brain-native-compute-isolated-results.json` as `--results`.

To regenerate the September figure from its recorded measurements:

```sh
python benchmarks/plot_brain_compute.py \
  --results docs/assets/brain-native-compute-results.json \
  --datafusion-results docs/assets/brain-datafusion-disk-results.json \
  --output docs/assets/brain-native-compute
```

The command below runs a new benchmark with the current implementation. For WSL,
use Linux storage for the run directory, and archive the results to a persistent
location before the WSL session ends. The worker places temporary data under the
run directory. Omit `--engines` to run all engines, or use `--engines datafusion`
for a focused rerun.

Use the prepared inputs and references from the [disk-backed benchmark](brain-benchmark.md). Inside WSL, run the supervisor as root with non-root worker uid/gid:

```sh
python benchmarks/brain_preprocessing.py run \
  --data /path/to/brain-prepared --output /path/to/new-compute-runs \
  --endpoint native --limits max --repeats 1 --timeout 600
python benchmarks/plot_brain_compute.py \
  --results /path/to/new-compute-runs/results.json \
  --output docs/assets/brain-native-compute
```

[Raw results and source hashes](../assets/brain-native-compute-results.json). The previous [full-RAM export study](brain-max-ram.md) remains a separate benchmark.
The measured runner called SciPy's eager storage `eager_csr`; the returned normalized array is COO. That descriptive field is corrected in the published JSON with a correction record. Timings and outcomes are unchanged. The [original measurement file](../assets/brain-native-compute-measured-results.json) is also retained.
