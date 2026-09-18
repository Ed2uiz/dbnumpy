"""Compare the unchanged PBMC workflow on disk-backed 10x brain cell counts.

Preparation and independent verification stream small blocks. Timed workers
run the ordinary shared function, without SQL or substitute implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.pbmc_memory_pressure import mark, read_counters  # noqa: E402
from benchmarks.pbmc_preprocessing import (  # noqa: E402
    ENGINES,
    sha256,
    write_json,
)

SUMMARY_FIELDS = (
    "cells",
    "genes",
    "total_counts",
    "detected_genes",
    "gene_mean",
    "gene_variance",
)
SOURCE_URL = (
    "https://www.10xgenomics.com/datasets/"
    "1-3-million-brain-cells-from-e-18-mice-2-standard-1-3-0"
)


def mapped_input(folder, size):
    """A read-only CSR view used only by preparation and independent checks."""
    import numpy as np
    from scipy import sparse

    manifest = json.loads((folder / "dataset.json").read_text())
    pointers = np.load(folder / "indptr.npy", mmap_mode="r")[: size + 1]
    end = int(pointers[-1])
    return sparse.csr_array(
        (
            np.load(folder / "data.npy", mmap_mode="r")[:end],
            np.load(folder / "indices.npy", mmap_mode="r")[:end],
            pointers,
        ),
        shape=(size, manifest["genes"]),
        copy=False,
    )


def prepare_csr(args):
    import h5py
    import numpy as np

    args.data.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sizes = sorted(set(args.sizes))
    count = sizes[-1]
    with h5py.File(args.archive, "r") as source:
        matrix = source["mm10"]
        genes, cells = map(int, matrix["shape"][:])
        if count > cells:
            raise ValueError("Sample exceeds the number of actual cells")
        original_ptr = matrix["indptr"][:].astype(np.int64)
        chosen = np.random.default_rng(args.seed).permutation(cells)[:count]
        np.save(args.data / "original_cells.npy", chosen)
        lengths = np.diff(original_ptr)[chosen]
        pointers = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
        index_dtype = np.int32 if pointers[-1] < 2**31 and count < 2**31 else np.int64
        pointers = pointers.astype(index_dtype)
        np.save(args.data / "indptr.npy", pointers)
        target_data = np.lib.format.open_memmap(
            args.data / "data.npy",
            mode="w+",
            dtype=np.float64,
            shape=(int(pointers[-1]),),
        )
        target_indices = np.lib.format.open_memmap(
            args.data / "indices.npy",
            mode="w+",
            dtype=index_dtype,
            shape=(int(pointers[-1]),),
        )
        rank = np.full(cells, -1, dtype=np.int64)
        rank[chosen] = np.arange(count)
        for start in range(0, cells, args.block):
            stop = min(cells, start + args.block)
            selected = np.flatnonzero(rank[start:stop] >= 0)
            if not len(selected):
                continue
            left, right = int(original_ptr[start]), int(original_ptr[stop])
            data = matrix["data"][left:right]
            indices = matrix["indices"][left:right]
            for local in selected:
                original = start + int(local)
                dest = int(rank[original])
                a, b = original_ptr[original : original + 2] - left
                out_a, out_b = pointers[dest : dest + 2]
                target_data[out_a:out_b] = data[a:b]
                target_indices[out_a:out_b] = indices[a:b]
            if start // args.block % 25 == 0:
                print(f"Read {stop:,}/{cells:,} source cells", flush=True)
        target_data.flush()
        target_indices.flush()
        del target_data, target_indices
        manifest = dict(
            source_url=SOURCE_URL,
            source_cells=cells,
            genes=genes,
            source_nnz=int(original_ptr[-1]),
            seed=args.seed,
            sizes=sizes,
            selected_cells=count,
            selected_nnz=int(pointers[-1]),
            selected_indices_sha256=sha256(args.data / "original_cells.npy"),
            archive_sha256=sha256(args.archive),
            csr_preparation_seconds=time.perf_counter() - started,
            partitions=[],
        )
    write_json(args.data / "dataset.json", manifest)


def prepare(args):
    prepare_csr(args)
    prepare_stores(args)


def prepare_stores(args):
    import duckdb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy import sparse

    if (args.data / "counts.duckdb").exists():
        raise ValueError("Prepared database already exists; use a new directory")
    started = time.perf_counter()
    manifest = json.loads((args.data / "dataset.json").read_text())
    count, genes, sizes = (
        manifest["selected_cells"],
        manifest["genes"],
        manifest["sizes"],
    )
    pointers = np.load(args.data / "indptr.npy", mmap_mode="r")
    values = mapped_input(args.data, count)
    target_data = np.load(args.data / "data.npy", mmap_mode="r+")
    target_indices = np.load(args.data / "indices.npy", mmap_mode="r+")
    for first in range(0, count, args.block):
        last = min(first + args.block, count)
        block = sparse.csr_array(values[first:last])
        entries = block.nnz
        block.sum_duplicates()
        if block.nnz != entries:
            raise ValueError("Duplicate source coordinates require coalescing")
        left, right = pointers[first], pointers[last]
        target_data[left:right] = block.data
        target_indices[left:right] = block.indices
        if first // args.block % 25 == 0:
            print(f"Sorted {last:,}/{count:,} selected cells", flush=True)
    target_data.flush()
    target_indices.flush()
    del target_data, target_indices, values
    manifest["csr_canonicalization_seconds"] = time.perf_counter() - started
    values = mapped_input(args.data, count)
    manifest["partitions"] = []
    before_parquet = time.perf_counter()
    for begin, end in zip([0, *sizes[:-1]], sizes, strict=True):
        path = args.data / f"counts-{begin}-{end}.parquet"
        schema = pa.schema([("i", pa.int64()), ("j", pa.int64()), ("x", pa.float64())])
        with pq.ParquetWriter(path, schema=schema, compression="zstd") as writer:
            for first in range(begin, end, args.block):
                last = min(first + args.block, end)
                block = sparse.csr_array(values[first:last])
                if not block.has_canonical_format:
                    raise ValueError("Source coordinates are not unique and sorted")
                if not np.isfinite(block.data).all() or not (block.data > 0).all():
                    raise ValueError("Expected positive, finite stored counts")
                if (block.indices < 0).any() or (block.indices >= genes).any():
                    raise ValueError("Source gene coordinates are out of bounds")
                table = pa.table(
                    {
                        "i": np.repeat(np.arange(first, last), np.diff(block.indptr)),
                        "j": block.indices.astype(np.int64),
                        "x": block.data,
                    },
                    schema=schema,
                )
                writer.write_table(table, row_group_size=262_144)
        manifest["partitions"].append(
            dict(
                begin=begin,
                end=end,
                file=path.name,
                nnz=int(pointers[end] - pointers[begin]),
                bytes=path.stat().st_size,
                sha256=sha256(path),
            )
        )
        print(f"Parquet prepared through {end:,} cells", flush=True)
    manifest["parquet_preparation_seconds"] = time.perf_counter() - before_parquet
    del values
    before_duckdb = time.perf_counter()
    # Public relation API for file preparation; no custom SQL or matrix logic.
    with duckdb.connect(
        str(args.data / "counts.duckdb"),
        config={"threads": 1, "memory_limit": "1GB"},
    ) as connection:
        connection.read_parquet(
            [str(args.data / part["file"]) for part in manifest["partitions"]]
        ).create("counts")
    manifest["duckdb_preparation_seconds"] = time.perf_counter() - before_duckdb
    manifest["duckdb_bytes"] = (args.data / "counts.duckdb").stat().st_size
    for name in ("data.npy", "indices.npy", "indptr.npy"):
        manifest[f"{name}_sha256"] = sha256(args.data / name)
    manifest["total_preparation_seconds"] = (
        manifest["csr_preparation_seconds"] + time.perf_counter() - started
    )
    write_json(args.data / "dataset.json", manifest)
    print(json.dumps(manifest), flush=True)


def normalized_blocks(x, cells, genes, block_size):
    """Independent normalization via Scanpy, after fixed global selection."""
    import anndata as ad
    import numpy as np
    import scanpy as sc
    from scipy import sparse

    for start in range(0, len(cells), block_size):
        block = sparse.csr_matrix(x[cells[start : start + block_size], :][:, genes])
        annotated = ad.AnnData(block)
        sc.pp.normalize_total(annotated, target_sum=10_000)
        sc.pp.log1p(annotated)
        normalized = annotated.X
        if not np.isfinite(normalized.data).all():
            raise ValueError("Nonfinite reference output")
        yield start, sparse.csr_array(normalized)


def reference(args):
    began = time.perf_counter()
    import numpy as np
    import scanpy as sc

    x = mapped_input(args.data, args.size)
    totals = np.empty(args.size)
    detected = np.empty(args.size, dtype=np.int64)
    detected_cells = np.zeros(x.shape[1], dtype=np.int64)
    for start in range(0, args.size, args.block):
        stop = min(start + args.block, args.size)
        block = x[start:stop]
        keep, count = sc.pp.filter_cells(block, min_genes=200, inplace=False)
        totals[start:stop] = np.asarray(block.sum(axis=1)).ravel()
        detected[start:stop] = count
        _, gene_count = sc.pp.filter_genes(block[keep], min_cells=1, inplace=False)
        detected_cells += gene_count
    genes = np.flatnonzero(detected_cells >= 3)
    cells = np.flatnonzero(detected >= 200)
    positive = np.empty(len(cells), dtype=bool)
    for first in range(0, len(cells), args.block):
        block = x[cells[first : first + args.block], :][:, genes]
        positive[first : first + args.block] = np.asarray(block.sum(axis=1)).ravel() > 0
    cells = cells[positive]
    if len(cells) < 2 or len(genes) == 0:
        raise ValueError("Independent reference has too few cells or no genes")
    sums = np.zeros(len(genes))
    stored_counts = np.zeros(len(genes), dtype=np.int64)
    nnz = 0
    for _, block in normalized_blocks(x, cells, genes, args.block):
        sums += np.asarray(block.sum(axis=0)).ravel()
        stored_counts += np.bincount(block.indices, minlength=len(genes))
        nnz += block.nnz
    mean = sums / len(cells)
    deviations = np.zeros(len(genes))
    for _, block in normalized_blocks(x, cells, genes, args.block):
        centered = block.data - mean[block.indices]
        deviations += np.bincount(
            block.indices, weights=centered * centered, minlength=len(genes)
        )
    # Account for implicit zeros in the centered variance, rather than using
    # the second-moment formula in the timed shared function.
    deviations += (len(cells) - stored_counts) * mean * mean
    np.savez(
        args.data / f"reference-{args.size}.npz",
        cells=cells,
        genes=genes,
        total_counts=totals,
        detected_genes=detected,
        gene_mean=mean,
        gene_variance=deviations / (len(cells) - 1),
        output_nnz=nnz,
    )
    write_json(
        args.data / f"reference-{args.size}.json",
        dict(
            cells=args.size,
            block_size=args.block,
            retained_cells=len(cells),
            retained_genes=len(genes),
            output_nnz=nnz,
            validation_seconds=time.perf_counter() - began,
            reference_sha256=sha256(args.data / f"reference-{args.size}.npz"),
        ),
    )
    print(
        f"Independent reference: {args.size:,} input, {len(cells):,} retained cells, "
        f"{nnz:,} output entries",
        flush=True,
    )


def save_summary(path, result):
    import numpy as np

    np.savez(path, **{name: getattr(result, name) for name in SUMMARY_FIELDS})


def compute_native(args, result, backend, timing):
    """Time native materialization, then inspect its stored values separately."""
    import numpy as np
    from scipy import sparse

    mark(args.output, "native_compute")
    before = time.perf_counter()
    materialized = result.values if backend is None else result.values.compute()
    elapsed = time.perf_counter() - before if backend is not None else 0.0
    timing.update(
        native_compute_seconds=elapsed,
        total_seconds=sum(timing.values()) + elapsed,
        retained_cells=len(result.cells),
        retained_genes=len(result.genes),
        native_compute_completed=True,
        native_storage=f"eager_{materialized.format}"
        if backend is None
        else "temporary_table"
        if args.engine == "duckdb"
        else "parquet",
    )
    write_json(args.output / "timing.json", timing)
    # This is after the timer: no full matrix crosses into Python. Check all
    # gene means/second moments and the positive entry count, plus complete
    # rows spread across the result. The external verifier uses Scanpy.
    mark(args.output, "native_validation")
    validation_start = time.perf_counter()
    if backend is not None:
        from dbnumpy.ir import Source

        assert isinstance(materialized._expr, Source)
    assert materialized.shape == (len(result.cells), len(result.genes))
    mean = np.asarray(np.mean(materialized, axis=0)).ravel()
    second = np.asarray(np.mean(materialized * materialized, axis=0)).ravel()
    variance = (second - mean * mean) * len(result.cells) / (len(result.cells) - 1)
    nnz = (
        materialized.count_nonzero()
        if backend is None
        else int(np.sum(materialized > 0))
    )
    rows = np.unique(
        np.linspace(0, len(result.cells) - 1, min(64, len(result.cells)), dtype=int)
    )
    if backend is None and materialized.format == "coo":
        # SciPy COO fancy indexing builds an nnz-by-number_of_rows mask.
        # Select coordinates directly so validation does not need that large
        # temporary. This is outside the timed scientific workflow.
        selected = np.isin(materialized.row, rows, kind="table")
        sample = sparse.coo_array(
            (
                materialized.data[selected],
                (
                    np.searchsorted(rows, materialized.row[selected]),
                    materialized.col[selected],
                ),
            ),
            shape=(len(rows), materialized.shape[1]),
        )
    else:
        sample = materialized[rows, :]
    sample = sparse.csr_array(
        sample if backend is None else sample.to_scipy(format="csr")
    )
    np.savez(
        args.output / "native-checks.npz",
        gene_mean=mean,
        gene_variance=variance,
        output_nnz=nnz,
        rows=rows,
        data=sample.data,
        indices=sample.indices,
        indptr=sample.indptr,
    )
    timing.update(
        output_nnz=int(nnz),
        native_validation_seconds=time.perf_counter() - validation_start,
    )
    write_json(args.output / "timing.json", timing)


def worker(args):
    spill = (args.output / "spill").resolve()
    spill.mkdir()
    os.environ["TMPDIR"] = str(spill)
    mark(args.output, "imports")
    import numpy as np
    from scipy import sparse

    from examples.pbmc_workflow import preprocess

    manifest = json.loads((args.data / "dataset.json").read_text())
    if args.engine != "scipy":
        import dbnumpy as dnp

        __import__(args.engine)
    mark(args.output, "input_open")
    started = time.perf_counter()
    backend = None
    if args.engine == "scipy":
        # Read only this prefix and own its buffers. No additional x.copy().
        pointers = np.load(args.data / "indptr.npy", mmap_mode="r")[
            : args.size + 1
        ].copy()
        end = int(pointers[-1])
        data = np.load(args.data / "data.npy", mmap_mode="r")[:end].copy()
        indices = np.load(args.data / "indices.npy", mmap_mode="r")[:end].copy()
        values = sparse.csr_array(
            (data, indices, pointers), shape=(args.size, manifest["genes"])
        )
        del data, indices, pointers
    elif args.engine == "duckdb":
        import duckdb

        backend = dnp.DuckDBBackend.connect(
            args.data / "counts.duckdb",
            threads=1,
            memory_limit=f"{args.engine_bytes}B",
            temp_directory=spill,
            max_sparse_host_values=2_000_000_000,
        )
        backend.connection.table("counts").filter(
            duckdb.ColumnExpression("i") < args.size
        ).create_view("benchmark_subset")
        values = backend.from_relation(
            "benchmark_subset", shape=(args.size, manifest["genes"]), storage="sparse"
        )
    else:
        backend = dnp.DataFusionBackend.connect(
            target_partitions=1,
            memory_limit_bytes=args.engine_bytes,
            temp_directory=spill,
            max_sparse_host_values=2_000_000_000,
        )
        values = backend.from_parquet(
            [
                args.data / part["file"]
                for part in manifest["partitions"]
                if part["end"] <= args.size
            ],
            shape=(args.size, manifest["genes"]),
        )
    opened = time.perf_counter()
    mark(args.output, "workflow")
    result = preprocess(values)
    calculated = time.perf_counter()
    # Save small summaries before full collection, so its separate boundary
    # remains visible even if collection is killed by the kernel.
    mark(args.output, "save_summaries")
    write_json(
        args.output / "timing.json",
        dict(
            input_open_seconds=opened - started,
            workflow_seconds=calculated - opened,
        ),
    )
    save_summary(args.output / "summary.npz", result)
    if getattr(args, "endpoint", "export") == "native":
        compute_native(
            args,
            result,
            backend,
            dict(
                input_open_seconds=opened - started,
                workflow_seconds=calculated - opened,
            ),
        )
        if backend is not None:
            backend.close()
        mark(args.output, "finished")
        return
    mark(args.output, "sparse_collection")
    before_collection = time.perf_counter()
    output = sparse.csr_array(
        result.values if backend is None else result.values.to_scipy(format="csr")
    )
    collected = time.perf_counter()
    write_json(
        args.output / "timing.json",
        dict(
            input_open_seconds=opened - started,
            workflow_seconds=calculated - opened,
            collect_seconds=collected - before_collection,
            total_seconds=(calculated - started) + (collected - before_collection),
            retained_cells=len(result.cells),
            retained_genes=len(result.genes),
            output_nnz=output.nnz,
        ),
    )
    mark(args.output, "cleanup")
    if backend is not None:
        backend.close()
    result.values = None
    del values, backend
    mark(args.output, "save_for_validation")
    for name in ("data", "indices", "indptr"):
        np.save(args.output / f"output-{name}.npy", getattr(output, name))
    mark(args.output, "finished")


def verify(args):
    import numpy as np
    from scipy import sparse

    reference_path = args.data / f"reference-{args.size}.npz"
    checks = {"scope": "summaries_only"}
    with (
        np.load(reference_path) as expected,
        np.load(args.output / "summary.npz") as actual,
    ):
        for name in SUMMARY_FIELDS[:4]:
            np.testing.assert_array_equal(actual[name], expected[name])
        for name in SUMMARY_FIELDS[4:]:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-9, atol=1e-11
            )
            checks[f"max_{name}_abs_error"] = float(
                np.max(np.abs(actual[name] - expected[name]))
            )
        cells, genes = expected["cells"], expected["genes"]
        expected_nnz = int(expected["output_nnz"])
    if getattr(args, "endpoint", "export") == "native":
        native_path = args.output / "native-checks.npz"
        if native_path.exists():
            with np.load(native_path) as actual, np.load(reference_path) as expected:
                assert int(actual["output_nnz"]) == expected_nnz
                for name in SUMMARY_FIELDS[4:]:
                    np.testing.assert_allclose(
                        actual[name], expected[name], rtol=1e-9, atol=1e-11
                    )
                    checks[f"native_max_{name}_abs_error"] = float(
                        np.max(np.abs(actual[name] - expected[name]))
                    )
                rows = actual["rows"]
                output = sparse.csr_array(
                    (actual["data"], actual["indices"], actual["indptr"]),
                    shape=(len(rows), len(genes)),
                )
                x = mapped_input(args.data, args.size)
                error = 0.0
                for start, block in normalized_blocks(
                    x, cells[rows], genes, args.block
                ):
                    sample = output[start : start + block.shape[0]]
                    np.testing.assert_array_equal(sample.indptr, block.indptr)
                    np.testing.assert_array_equal(sample.indices, block.indices)
                    np.testing.assert_allclose(
                        sample.data, block.data, rtol=1e-9, atol=1e-11
                    )
                    error = max(
                        error,
                        float(np.max(np.abs(sample.data - block.data), initial=0)),
                    )
                checks.update(
                    scope="native_summaries_count_and_sample",
                    sample_rows=len(rows),
                    sample_values=output.nnz,
                    sample_max_abs_error=error,
                )
        write_json(args.output / "verification.json", checks)
        print(json.dumps(checks), flush=True)
        return
    if (args.output / "phase.json").exists() and json.loads(
        (args.output / "phase.json").read_text()
    )["phase"] == "finished":
        output = sparse.csr_array(
            tuple(
                np.load(args.output / f"output-{name}.npy", mmap_mode="r")
                for name in ("data", "indices", "indptr")
            ),
            shape=(len(cells), len(genes)),
        )
        assert output.nnz == expected_nnz
        x = mapped_input(args.data, args.size)
        error = 0.0
        for start, expected in normalized_blocks(x, cells, genes, args.block):
            actual = output[start : start + expected.shape[0]]
            np.testing.assert_array_equal(actual.indptr, expected.indptr)
            np.testing.assert_array_equal(actual.indices, expected.indices)
            np.testing.assert_allclose(
                actual.data, expected.data, rtol=1e-9, atol=1e-11
            )
            error = max(
                error, float(np.max(np.abs(actual.data - expected.data), initial=0))
            )
        checks.update(scope="full_output_and_summaries", max_abs_error=error)
    write_json(args.output / "verification.json", checks)
    print(json.dumps(checks), flush=True)


def evict_input_cache(folder, engine, size, manifest):
    """Request eviction of only this trial's clean Linux input-file cache."""
    if engine == "scipy":
        paths = [folder / name for name in ("data.npy", "indices.npy", "indptr.npy")]
    elif engine == "duckdb":
        paths = [folder / "counts.duckdb"]
    else:
        paths = [
            folder / part["file"]
            for part in manifest["partitions"]
            if part["end"] <= size
        ]
    for path in paths:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return [path.name for path in paths]


def run(args):
    if os.geteuid() != 0 or args.uid == 0 or args.gid == 0:
        raise SystemExit("Run supervisor as root, with a non-root worker uid/gid")
    if len(set(args.engines)) != len(args.engines):
        raise ValueError("Each engine must be listed once")
    memory = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":", 1)[0] in ("MemTotal", "MemAvailable", "SwapTotal")
    }
    requested_limits = args.limits
    args.limits = [
        memory["MemTotal"] / 1024**3 if str(value) == "max" else float(value)
        for value in requested_limits
    ]
    if any(not math.isfinite(value) or value <= 0 for value in args.limits):
        raise ValueError("Memory ceilings must be positive and finite")
    # At the full VM ceiling, Linux may run out before the cgroup does. Prefer
    # killing the benchmark worker over unrelated processes in that situation.
    worker_oom_score = 500 if "max" in requested_limits else 0
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        method="cgroup v2; memory.max=ceiling, memory.swap.max=0; serial workers",
        scope=(
            "disk-backed input; shared workflow; native materialization; "
            "validate stored summaries, entry count and up to 64 complete rows"
            if args.endpoint == "native"
            else "disk-backed input; shared workflow; full output required for pass"
        ),
        endpoint=args.endpoint,
        input_cache_policy=(
            "POSIX_FADV_DONTNEED requested on each worker's input files; "
            "Windows host cache is not controlled"
        ),
        platform=platform.platform(),
        python=platform.python_version(),
        cpu=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        versions={
            n: version(n)
            for n in (
                "numpy",
                "scipy",
                "duckdb",
                "datafusion",
                "pyarrow",
                "h5py",
                "scanpy",
            )
        },
        limits_gib=args.limits,
        requested_limits=requested_limits,
        system_memory_bytes=memory,
        worker_oom_score_adj=worker_oom_score,
        sizes=args.sizes,
        engines=args.engines,
        repeats=args.repeats,
        timeout_seconds=args.timeout,
        disk_free_floor_bytes=5 * 1024**3,
        engine_budget_fraction=0.5,
        workflow_sha256=sha256(ROOT / "examples/pbmc_workflow.py"),
        runner_sha256=sha256(Path(__file__)),
        package_source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "src/dbnumpy").rglob("*.py"))
        },
        reference_sha256={
            str(size): sha256(args.data / f"reference-{size}.npz")
            for size in args.sizes
        },
        dataset=json.loads((args.data / "dataset.json").read_text()),
        trials=[],
    )
    if not set(args.sizes).issubset(metadata["dataset"]["sizes"]):
        raise ValueError("Sizes must match prepared Parquet partition boundaries")
    for size in args.sizes:
        if not (args.data / f"reference-{size}.npz").is_file():
            raise ValueError(f"Independent reference missing for {size} cells")
    for ceiling in args.limits:
        for size in args.sizes:
            for repeat in range(args.repeats):
                offset = repeat % len(args.engines)
                for engine in args.engines[offset:] + args.engines[:offset]:
                    evicted = evict_input_cache(
                        args.data, engine, size, metadata["dataset"]
                    )
                    folder = args.output / f"{ceiling:g}GiB-{size}-{repeat}-{engine}"
                    folder.mkdir()
                    os.chown(folder, args.uid, args.gid)
                    group = Path("/sys/fs/cgroup") / f"dbnumpy-brain-{uuid4().hex}"
                    group.mkdir()
                    cap = int(ceiling * 1024**3)
                    process = None
                    try:
                        (group / "memory.max").write_text(str(cap))
                        (group / "memory.swap.max").write_text("0")
                        (group / "memory.oom.group").write_text("1")
                        assert int((group / "memory.max").read_text()) == cap
                        assert int((group / "memory.swap.max").read_text()) == 0

                        def enter(group=group):
                            (group / "cgroup.procs").write_text(str(os.getpid()))
                            Path("/proc/self/oom_score_adj").write_text(
                                str(worker_oom_score)
                            )
                            os.setgroups([])
                            os.setgid(args.gid)
                            os.setuid(args.uid)

                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "worker",
                            "--data",
                            str(args.data.resolve()),
                            "--output",
                            str(folder.resolve()),
                            "--size",
                            str(size),
                            "--engine",
                            engine,
                            "--engine-bytes",
                            str(cap // 2),
                            "--endpoint",
                            args.endpoint,
                        ]
                        print(f"Starting {folder.name}", flush=True)
                        timeout = False
                        disk_limit = False
                        with (folder / "worker.log").open("w") as stream:
                            process = subprocess.Popen(
                                command, stdout=stream, stderr=stream, preexec_fn=enter
                            )
                            began = time.monotonic()
                            while process.poll() is None:
                                timeout = time.monotonic() - began > args.timeout
                                disk_limit = (
                                    shutil.disk_usage(args.data).free < 5 * 1024**3
                                )
                                if timeout or disk_limit:
                                    (group / "cgroup.kill").write_text("1")
                                    break
                                time.sleep(0.5)
                            process.wait()
                        events = read_counters(group / "memory.events")
                        row = dict(
                            engine=engine,
                            cells=size,
                            repeat=repeat,
                            limit_gib=ceiling,
                            engine_budget_bytes=cap // 2,
                            memory_max_bytes=cap,
                            peak_bytes=int((group / "memory.peak").read_text()),
                            memory_events=events,
                            returncode=process.returncode,
                            input_files_eviction_requested=evicted,
                            last_phase=json.loads((folder / "phase.json").read_text())[
                                "phase"
                            ]
                            if (folder / "phase.json").exists()
                            else "startup",
                            status="disk_limit"
                            if disk_limit
                            else "timeout"
                            if timeout
                            else "memory_limit"
                            if events["oom_kill"]
                            else "error"
                            if process.returncode
                            else "pending",
                        )
                    finally:
                        if process is not None and process.poll() is None:
                            (group / "cgroup.kill").write_text("1")
                            process.wait()
                        group.rmdir()
                    spill = (folder / "spill").resolve()
                    if spill.is_relative_to(folder.resolve()) and spill.is_dir():
                        # Only this worker's generated spill directory is removed.
                        row["spill_bytes_after_exit"] = sum(
                            p.stat().st_size for p in spill.rglob("*") if p.is_file()
                        )
                        shutil.rmtree(spill)
                    if (folder / "timing.json").exists():
                        row.update(json.loads((folder / "timing.json").read_text()))
                    if (folder / "summary.npz").exists():
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "verify",
                            "--data",
                            str(args.data.resolve()),
                            "--output",
                            str(folder.resolve()),
                            "--size",
                            str(size),
                            "--endpoint",
                            args.endpoint,
                        ]
                        with (folder / "verification.log").open("w") as stream:
                            check = subprocess.run(
                                command,
                                stdout=stream,
                                stderr=stream,
                                timeout=args.timeout,
                            )
                        if check.returncode:
                            row["verification"] = "failed"
                            if row["status"] == "pending":
                                row["status"] = "verification_failed"
                        else:
                            row["verification"] = json.loads(
                                (folder / "verification.json").read_text()
                            )
                            if row["status"] == "pending":
                                if row["verification"]["scope"] != (
                                    "native_summaries_count_and_sample"
                                    if args.endpoint == "native"
                                    else "full_output_and_summaries"
                                ):
                                    raise RuntimeError("Missing endpoint verification")
                                row["status"] = "passed"
                    if row["status"] == "pending":
                        row["status"] = "missing_output"
                    if row["status"] != "passed":
                        row["error"] = (folder / "worker.log").read_text()[-3000:]
                    metadata["trials"].append(row)
                    write_json(args.output / "results.json", metadata)
                    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "prepare-stores", "reference", "worker", "verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--block", type=int, default=2048)
        if name in ("worker", "verify", "run"):
            p.add_argument("--endpoint", choices=("export", "native"), default="export")
        if name in ("prepare", "run"):
            p.add_argument(
                "--sizes",
                type=int,
                nargs="+",
                default=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
            )
        if name == "prepare":
            p.add_argument("--archive", type=Path, required=True)
            p.add_argument("--seed", type=int, default=20260916)
        elif name == "prepare-stores":
            pass
        elif name == "run":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--engines", choices=ENGINES, nargs="+", default=ENGINES)
            p.add_argument(
                "--limits",
                nargs="+",
                default=["1", "2"],
                help="Memory ceilings in GiB, or max for Linux MemTotal",
            )
            p.add_argument("--repeats", type=int, default=1)
            p.add_argument("--timeout", type=int, default=600)
            p.add_argument("--uid", type=int, default=1000)
            p.add_argument("--gid", type=int, default=1000)
        else:
            p.add_argument("--size", type=int, required=True)
            if name != "reference":
                p.add_argument("--output", type=Path, required=True)
            if name == "worker":
                p.add_argument("--engine", choices=ENGINES, required=True)
                p.add_argument("--engine-bytes", type=int, required=True)
    args = parser.parse_args()
    globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    main()
