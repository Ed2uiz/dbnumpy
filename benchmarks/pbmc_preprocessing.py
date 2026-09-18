"""Reproduce the PBMC preprocessing comparison without handwritten SQL.

Run from the repository root; see docs/guides/pbmc-benchmark.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENGINES = ("scipy", "duckdb", "datafusion")
SOURCE_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
for variable in THREAD_VARS:
    os.environ[variable] = "1"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    import tarfile

    import numpy as np
    from scipy import io, sparse

    from examples.pbmc_workflow import preprocess

    args.data.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("matrix.mtx"))
        with archive.extractfile(member) as stream:
            x = sparse.csr_array(io.mmread(stream, spmatrix=False).T, dtype=np.float64)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError("Expected finite, nonnegative counts")
    permutation = np.random.default_rng(args.seed).permutation(x.shape[0])
    metadata = dict(
        source_url=SOURCE_URL,
        archive_sha256=sha256(args.archive),
        input_shape=list(x.shape),
        input_nnz=x.nnz,
        seed=args.seed,
        subsets=[],
    )
    for size in args.sizes:
        if size > x.shape[0]:
            raise ValueError("Subset exceeds the number of actual cells")
        indices = permutation[:size]
        subset = x[indices, :]
        path = args.data / f"input-{size}.npz"
        sparse.save_npz(path, subset, compressed=False)
        np.save(args.data / f"cells-{size}.npy", indices)
        result = preprocess(subset)
        save_reference(args.data, size, result)
        entry = dict(
            cells=size,
            genes=x.shape[1],
            nnz=subset.nnz,
            input_sha256=sha256(path),
            cell_indices_sha256=sha256(args.data / f"cells-{size}.npy"),
            retained_cells=len(result.cells),
            retained_genes=len(result.genes),
            output_nnz=result.values.nnz,
        )
        metadata["subsets"].append(entry)
        print(json.dumps(entry), flush=True)
    write_json(args.data / "dataset.json", metadata)


def save_reference(folder, size, result):
    import numpy as np
    from scipy import sparse

    sparse.save_npz(folder / f"reference-{size}.npz", result.values, compressed=False)
    np.savez(
        folder / f"summary-{size}.npz",
        **{
            name: getattr(result, name)
            for name in (
                "cells",
                "genes",
                "total_counts",
                "detected_genes",
                "gene_mean",
                "gene_variance",
            )
        },
    )


def compare(actual, expected):
    import numpy as np
    from scipy import sparse

    actual = sparse.csr_array(actual)
    expected = sparse.csr_array(expected)
    for x in (actual, expected):
        x.sum_duplicates()
        x.eliminate_zeros()
        x.sort_indices()
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-9, atol=1e-11)
    return float(np.max(np.abs(actual.data - expected.data), initial=0))


def validate_scanpy(args):
    import anndata as ad
    import numpy as np
    import scanpy as sc
    from scipy import sparse

    rows = []
    for size in args.sizes:
        x = sparse.load_npz(args.data / f"input-{size}.npz")
        a = ad.AnnData(sparse.csr_matrix(x))
        sc.pp.filter_cells(a, min_genes=200)
        sc.pp.filter_genes(a, min_cells=3)
        a = a[np.asarray(a.X.sum(axis=1)).ravel() > 0, :].copy()
        sc.pp.normalize_total(a, target_sum=10_000)
        sc.pp.log1p(a)
        with np.load(args.data / f"summary-{size}.npz") as summary:
            np.testing.assert_array_equal(a.obs_names.astype(int), summary["cells"])
            np.testing.assert_array_equal(a.var_names.astype(int), summary["genes"])
            # Use dense column blocks and NumPy's centered variance, rather
            # than repeating the shared workflow's second-moment formula.
            columns = a.X.tocsc()
            mean_error = variance_error = 0.0
            independent_mean = np.empty(a.n_vars)
            independent_variance = np.empty(a.n_vars)
            for start in range(0, a.n_vars, 64):
                stop = min(start + 64, a.n_vars)
                block = columns[:, start:stop].toarray()
                for name, expected in (
                    ("gene_mean", np.mean(block, axis=0)),
                    ("gene_variance", np.var(block, axis=0, ddof=1)),
                ):
                    observed = summary[name][start:stop]
                    np.testing.assert_allclose(
                        observed, expected, rtol=1e-9, atol=1e-11
                    )
                    difference = float(np.max(np.abs(observed - expected), initial=0))
                    if name == "gene_mean":
                        mean_error = max(mean_error, difference)
                        independent_mean[start:stop] = expected
                    else:
                        variance_error = max(variance_error, difference)
                        independent_variance[start:stop] = expected
            del columns, block
            np.savez(
                args.data / f"independent-summary-{size}.npz",
                gene_mean=independent_mean,
                gene_variance=independent_variance,
            )
        error = compare(a.X, sparse.load_npz(args.data / f"reference-{size}.npz"))
        rows.append(
            dict(
                cells=size,
                max_abs_error=error,
                status="passed",
                max_mean_abs_error=mean_error,
                max_variance_abs_error=variance_error,
            )
        )
        print(rows[-1], flush=True)
    write_json(
        args.data / "scanpy-validation.json",
        dict(
            scanpy=version("scanpy"),
            anndata=version("anndata"),
            rtol=1e-9,
            atol=1e-11,
            checks=rows,
        ),
    )


def worker(args):
    import numpy as np
    from scipy import sparse

    import dbnumpy as dnp
    from examples.pbmc_workflow import preprocess

    # Imports and the common input read are deliberately outside the timer.
    x = sparse.csr_array(sparse.load_npz(args.data / f"input-{args.size}.npz"))
    if args.engine != "scipy":
        import_module(args.engine)
    backend = None
    started = time.perf_counter()
    if args.engine == "scipy":
        values = x.copy()
    elif args.engine == "duckdb":
        backend = dnp.DuckDBBackend.connect(
            threads=1,
            memory_limit="2GB",
            max_sparse_host_values=100_000_000,
        )
        values = backend.from_scipy(x)
    else:
        backend = dnp.DataFusionBackend.connect(
            target_partitions=1,
            memory_limit_bytes=2 * 1024**3,
            max_sparse_host_values=100_000_000,
        )
        values = backend.from_scipy(x)
    ingested = time.perf_counter()
    result = preprocess(values)
    calculated = time.perf_counter()
    output = sparse.csr_array(
        result.values if backend is None else result.values.to_scipy()
    )
    collected = time.perf_counter()
    if backend is not None:
        backend.close()
    # Full sparse values, selections and summaries are checked after timing.
    error = compare(output, sparse.load_npz(args.data / f"reference-{args.size}.npz"))
    with np.load(args.data / f"summary-{args.size}.npz") as reference:
        for name in ("cells", "genes", "total_counts", "detected_genes"):
            np.testing.assert_array_equal(getattr(result, name), reference[name])
    summary_errors = {}
    with np.load(args.data / f"independent-summary-{args.size}.npz") as reference:
        for name in ("gene_mean", "gene_variance"):
            np.testing.assert_allclose(
                getattr(result, name),
                reference[name],
                rtol=1e-9,
                atol=1e-11,
            )
            summary_errors[name] = float(
                np.max(np.abs(getattr(result, name) - reference[name]), initial=0)
            )
    write_json(
        args.output,
        dict(
            engine=args.engine,
            cells=args.size,
            repeat=args.repeat,
            status="passed",
            total_seconds=collected - started,
            setup_seconds=ingested - started,
            workflow_seconds=calculated - ingested,
            collect_seconds=collected - calculated,
            max_abs_error=error,
            max_mean_abs_error=summary_errors["gene_mean"],
            max_variance_abs_error=summary_errors["gene_variance"],
            retained_cells=len(result.cells),
            retained_genes=len(result.genes),
            output_nnz=output.nnz,
        ),
    )


def run(args):
    import psutil

    args.output.mkdir(parents=True, exist_ok=True)
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    metadata = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        cpu=cpu,
        threads=1,
        engine_memory_limit="DuckDB 2GB; DataFusion 2GiB",
        process_rss_limit_gib=args.rss_limit,
        timeout_seconds=args.timeout,
        repeats=args.repeats,
        sizes=args.sizes,
        versions={
            name: version(name)
            for name in (
                "dbnumpy",
                "numpy",
                "scipy",
                "duckdb",
                "datafusion",
                "ibis-framework",
                "pyarrow",
                "sqlglot",
                "matplotlib",
            )
        },
        workflow_sha256=sha256(ROOT / "examples/pbmc_workflow.py"),
        runner_sha256=sha256(Path(__file__)),
        package_source_sha256={
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "src/dbnumpy").rglob("*.py"))
        },
        dataset=json.loads((args.data / "dataset.json").read_text()),
        scanpy_validation=json.loads(
            (args.data / "scanpy-validation.json").read_text()
        ),
        timings=(
            "setup + shared workflow + sparse collection; excludes imports/read/checks"
        ),
        trials=[],
    )
    for size in args.sizes:
        for repeat in range(args.repeats):
            order = ENGINES[repeat % 3 :] + ENGINES[: repeat % 3]
            for engine in order:
                name = f"{size}-{repeat}-{engine}"
                output = args.output / f"{name}.json"
                log = args.output / f"{name}.log"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--data",
                    str(args.data.resolve()),
                    "--size",
                    str(size),
                    "--engine",
                    engine,
                    "--repeat",
                    str(repeat),
                    "--output",
                    str(output.resolve()),
                ]
                print(f"Starting {name}", flush=True)
                start = time.monotonic()
                failure = None
                with log.open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=stream)
                    watched = psutil.Process(process.pid)
                    while process.poll() is None:
                        try:
                            rss = watched.memory_info().rss
                        except psutil.NoSuchProcess:
                            break
                        if time.monotonic() - start > args.timeout:
                            failure = "timeout"
                        if rss > args.rss_limit * 1024**3:
                            failure = "resource_limit"
                        if failure:
                            process.kill()
                            break
                        time.sleep(0.25)
                    process.wait()
                if process.returncode or failure:
                    row = dict(
                        engine=engine,
                        cells=size,
                        repeat=repeat,
                        status=failure or "error",
                        error=log.read_text()[-3000:],
                    )
                else:
                    row = json.loads(output.read_text())
                metadata["trials"].append(row)
                write_json(args.output / "results.json", metadata)
                print(json.dumps(row), flush=True)


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    results = json.loads(args.results.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    colors = ("#343d46", "#197a72", "#bc5925")
    labels = ("NumPy / SciPy", "dbnumpy · DuckDB", "dbnumpy · DataFusion")
    for engine, label, color, marker in zip(
        ENGINES, labels, colors, ("o", "s", "^"), strict=True
    ):
        medians, lows, highs = [], [], []
        for size in results["sizes"]:
            rows = [
                row
                for row in results["trials"]
                if row["engine"] == engine and row["cells"] == size
            ]
            times = [row["total_seconds"] for row in rows if row["status"] == "passed"]
            complete = len(times) == results["repeats"]
            medians.append(np.median(times) if complete else np.nan)
            lows.append(min(times) if complete else np.nan)
            highs.append(max(times) if complete else np.nan)
        ax.plot(
            results["sizes"],
            medians,
            marker=marker,
            ms=5,
            lw=1.8,
            color=color,
            label=label,
        )
        ax.fill_between(results["sizes"], lows, highs, color=color, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Input cells")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("PBMC preprocessing · same NumPy-style code", loc="left", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.text(
        0,
        -0.24,
        "68,579 real cells · one thread · median and range of 3 runs\n"
        "Includes setup, filtering, normalization, log1p, summaries and collection",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "validate-scanpy", "run", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        if name != "worker":
            command.add_argument(
                "--sizes",
                type=int,
                nargs="+",
                default=[1000, 3000, 10_000, 30_000, 68_579],
            )
        if name == "prepare":
            command.add_argument("--archive", type=Path, required=True)
            command.add_argument("--seed", type=int, default=20260915)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--timeout", type=float, default=300)
            command.add_argument("--rss-limit", type=float, default=5.5)
        if name == "worker":
            command.add_argument("--size", type=int, required=True)
            command.add_argument("--engine", choices=ENGINES, required=True)
            command.add_argument("--repeat", type=int, required=True)
            command.add_argument("--output", type=Path, required=True)
    chart = commands.add_parser("plot")
    chart.add_argument("--results", type=Path, required=True)
    chart.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "validate-scanpy": validate_scanpy,
        "run": run,
        "worker": worker,
        "plot": plot,
    }[args.action](args)


if __name__ == "__main__":
    main()
