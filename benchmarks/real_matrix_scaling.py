#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the Figure 1 sparse-matrix scaling benchmark safely.

The parent process never imports the numerical stack.  Every dataset/backend
pair is executed in a fresh spawned process so a timeout or RSS safety stop does
not end the complete benchmark run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from html import escape
from importlib import metadata
from pathlib import Path
from typing import Any

# The 30k-column / ~600M-NNZ case is intentionally opt-in on a 16 GB laptop.
DEFAULT_SIZES = (1000, 3000, 10_000)
DEFAULT_BACKENDS = (
    "numpy-host",
    "duckdb-host",
    "datafusion-host",
    "duckdb-native",
)
RESULT_SCHEMA_VERSION = 3
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RESULT_COLUMNS = (
    "size",
    "backend",
    "status",
    "error",
    "source",
    "rows",
    "cols",
    "nnz",
    "density",
    "compressed_bytes",
    "native_input",
    "prepared_source",
    "prepare_sec",
    "connect_sec",
    "parse_sec",
    "ingest_sec",
    "checkpoint_sec",
    "reopen_sec",
    "wrap_sec",
    "cold_query_sec",
    "ready_sec",
    "query_median_sec",
    "query_p10_sec",
    "query_p90_sec",
    "query_min_sec",
    "query_max_sec",
    "query_samples_json",
    "output_checksum",
    "output_bytes",
    "max_abs_error",
    "execution_path",
    "host_coo_materialized",
    "scipy_loaded",
    "worker_tmp_dir",
    "requested_spill_dir",
    "effective_spill_dir",
    "spill_directory_observation",
    "threads",
    "engine_memory_gb",
    "process_rss_limit_gb",
    "case_order",
    "case_order_policy",
    "package_versions_json",
    "peak_rss_bytes",
    "rss_scope",
    "rss_sampling_note",
    "last_phase",
)


@dataclass(frozen=True)
class Dataset:
    size: int
    path: Path
    rows: int
    cols: int
    nnz: int
    compressed_bytes: int

    @property
    def density(self) -> float:
        cells = self.rows * self.cols
        return self.nnz / cells if cells else 0.0


def _open_mtx(path: Path) -> Any:
    return (
        gzip.open(path, "rt", encoding="ascii", errors="strict")
        if path.suffix == ".gz"
        else path.open("rt", encoding="ascii", errors="strict")
    )


def read_mtx_header(path: Path) -> tuple[int, int, int]:
    """Read only the Matrix Market banner/comments/dimension line."""

    with _open_mtx(path) as stream:
        banner = stream.readline().strip().split()
        if len(banner) < 5 or banner[:2] != ["%%MatrixMarket", "matrix"]:
            raise ValueError(f"{path}: invalid Matrix Market banner")
        if banner[2].lower() != "coordinate":
            raise ValueError(f"{path}: expected coordinate Matrix Market data")
        if banner[3].lower() not in {"real", "integer"}:
            raise ValueError(f"{path}: expected real or integer Matrix Market data")
        if banner[4].lower() != "general":
            raise ValueError(f"{path}: expected general Matrix Market symmetry")
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(f"{path}: invalid Matrix Market dimensions")
            rows, cols, nnz = (int(value) for value in fields)
            if min(rows, cols, nnz) < 0:
                raise ValueError(f"{path}: negative Matrix Market dimension")
            return rows, cols, nnz
    raise ValueError(f"{path}: missing Matrix Market dimensions")


def prepare_plain_mtx(
    source: Path, destination: Path, *, reuse: bool
) -> tuple[Path, float]:
    """Boundedly decompress ``source`` without replacing any existing file."""

    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        if not reuse:
            raise FileExistsError(
                f"prepared MTX already exists: {destination}; pass --reuse-prepared-mtx to validate and reuse it"
            )
        if read_mtx_header(destination) != read_mtx_header(source):
            raise ValueError(
                f"prepared MTX header does not match source: {destination}"
            )
        return destination, 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    started = time.perf_counter()
    try:
        with gzip.open(source, "rb") as incoming, temporary.open("xb") as outgoing:
            while chunk := incoming.read(4 * 1024 * 1024):
                outgoing.write(chunk)
        if read_mtx_header(temporary) != read_mtx_header(source):
            raise ValueError(f"prepared MTX header does not match source: {temporary}")
        # Hard-link publication is atomic and fails rather than overwriting a
        # destination created concurrently.
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, time.perf_counter() - started


def discover_datasets(data_dir: Path) -> dict[int, Dataset]:
    """Discover exact Figure 1 directory names without reading matrix bodies."""

    found: dict[int, Dataset] = {}
    prefix, suffix = "10x_synth_20000g_", "c_sp99"
    if not data_dir.is_dir():
        raise ValueError(f"data directory does not exist: {data_dir}")
    for directory in sorted(data_dir.iterdir()):
        name = directory.name
        if (
            not directory.is_dir()
            or not name.startswith(prefix)
            or not name.endswith(suffix)
        ):
            continue
        raw_size = name[len(prefix) : -len(suffix)]
        if not raw_size.isascii() or not raw_size.isdigit():
            continue
        path = directory / "matrix.mtx.gz"
        if not path.is_file():
            continue
        size = int(raw_size)
        if size in found:
            raise ValueError(f"duplicate dataset size {size}")
        rows, cols, nnz = read_mtx_header(path)
        if cols != size:
            raise ValueError(
                f"{path}: header has {cols} columns but directory declares {size}"
            )
        found[size] = Dataset(
            size, path.resolve(), rows, cols, nnz, path.stat().st_size
        )
    return dict(sorted(found.items()))


def sample_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "query_median_sec": statistics.median(ordered),
        "query_p10_sec": percentile(0.1),
        "query_p90_sec": percentile(0.9),
        "query_min_sec": ordered[0],
        "query_max_sec": ordered[-1],
    }


def _versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in ("dbnumpy", "numpy", "scipy", "duckdb", "datafusion"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _checksum(array: Any) -> str:
    contiguous = array.astype("<f8", copy=False)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _worker(case: dict[str, Any], connection: Any) -> None:
    """Execute one case and return a JSON-compatible row over a pipe."""

    for variable in THREAD_ENV_VARS:
        os.environ[variable] = str(case["threads"])
    os.environ["TMPDIR"] = case["requested_spill_dir"]
    backend_object: Any = None
    effective_spill_dir = case["effective_spill_dir"]
    spill_directory_observation = case["spill_directory_observation"]
    current_phase = "starting"
    backend_name = case["backend"]

    def phase(name: str) -> None:
        nonlocal current_phase
        current_phase = name
        connection.send({"message_type": "phase", "phase": name})

    try:
        import tempfile

        tempfile.tempdir = case["requested_spill_dir"]
        rows, cols = case["rows"], case["cols"]
        prepare_sec = 0.0
        parse_sec = 0.0
        connect_sec = 0.0
        ingest_sec = 0.0
        checkpoint_sec = 0.0
        reopen_sec = 0.0
        wrap_sec = 0.0
        cold_query_sec = 0.0
        prepared_source = ""
        effective_spill_dir = tempfile.gettempdir()
        spill_directory_observation = "python_tempfile_gettempdir"

        # Deliberately branch before importing SciPy: this path gives the MTX
        # file to DuckDB and never constructs a host COO matrix.
        if backend_name == "duckdb-native":
            from dbnumpy.backends import DuckDBBackend

            native_source = Path(case["source"])
            if case["native_input"] == "plain":
                phase("prepare_plain_mtx")
                native_source, prepare_sec = prepare_plain_mtx(
                    native_source,
                    Path(case["prepared_source_target"]),
                    reuse=case["reuse_prepared_mtx"],
                )
                prepared_source = str(native_source)

            database = Path(case["worker_tmp_dir"]) / "benchmark.duckdb"
            phase("native_connect")
            started = time.perf_counter()
            backend_object = DuckDBBackend.connect(
                database,
                threads=case["threads"],
                memory_limit=f"{case['engine_memory_gb']}GiB",
                temp_directory=case["requested_spill_dir"],
                max_densify_cells=max(rows, cols),
            )
            connect_sec = time.perf_counter() - started
            effective_spill_dir = backend_object.connection.execute(
                "SELECT current_setting('temp_directory')"
            ).fetchone()[0]
            spill_directory_observation = "duckdb_current_setting_temp_directory"

            phase("native_ingest")
            started = time.perf_counter()
            backend_object.from_mtx(
                native_source,
                name="benchmark_matrix",
                temporary=False,
                overwrite=False,
            )
            ingest_sec = time.perf_counter() - started

            phase("checkpoint")
            started = time.perf_counter()
            backend_object.connection.execute("CHECKPOINT")
            checkpoint_sec = time.perf_counter() - started
            backend_object.close()
            backend_object = None

            phase("reopen")
            started = time.perf_counter()
            backend_object = DuckDBBackend.connect(
                database,
                threads=case["threads"],
                memory_limit=f"{case['engine_memory_gb']}GiB",
                temp_directory=case["requested_spill_dir"],
                max_densify_cells=max(rows, cols),
            )
            reopen_sec = time.perf_counter() - started
            phase("wrap_relation")
            started = time.perf_counter()
            matrix = backend_object.from_relation(
                "benchmark_matrix", shape=(rows, cols), storage="sparse"
            )
            wrap_sec = time.perf_counter() - started
            execution_path = "duckdb_native_mtx_disk_table"

            def query() -> Any:
                return matrix.mean(axis=0)

            phase("cold_colmeans")
            started = time.perf_counter()
            output = query()
            cold_query_sec = time.perf_counter() - started

            phase("direct_sql_validation")
            # Independent structural-zero reference: AVG(x) is intentionally
            # not used because absent sparse coordinates represent zeros.
            direct_rows = backend_object.connection.execute(
                'SELECT "j", SUM("x") / ? AS mean FROM "benchmark_matrix" GROUP BY "j" ORDER BY "j"',
                [rows],
            ).fetchall()
            import numpy as np

            reference = np.zeros(cols, dtype=np.float64)
            for j, value in direct_rows:
                reference[int(j)] = float(value)
            actual_nnz = int(
                backend_object.connection.execute(
                    'SELECT COUNT(*) FROM "benchmark_matrix"'
                ).fetchone()[0]
            )
        else:
            phase("host_parse_scipy")
            import numpy as np
            from scipy.io import mmread

            parse_started = time.perf_counter()
            with gzip.open(case["source"], "rb") as stream:
                coo = mmread(stream).tocoo(copy=False)
            coo.sum_duplicates()
            coo.eliminate_zeros()
            parse_sec = time.perf_counter() - parse_started
            rows, cols = coo.shape
            actual_nnz = int(coo.nnz)
            phase("host_reference_colmeans")
            reference = np.asarray(coo.mean(axis=0), dtype=np.float64).reshape(-1)

        if backend_name == "numpy-host":
            execution_path = "optimized_numpy_bincount"

            def query() -> Any:
                return (
                    np.bincount(
                        coo.col,
                        weights=np.asarray(coo.data, dtype=np.float64),
                        minlength=cols,
                    ).astype(np.float64, copy=False)
                    / rows
                )

        elif backend_name in {"duckdb-host", "datafusion-host"}:
            phase("host_backend_connect")
            connect_started = time.perf_counter()
            if backend_name == "duckdb-host":
                from dbnumpy.backends import DuckDBBackend

                backend_object = DuckDBBackend.connect(
                    threads=case["threads"],
                    memory_limit=f"{case['engine_memory_gb']}GiB",
                    max_densify_cells=max(rows, cols),
                )
                backend_object.connection.execute(
                    "SET temp_directory = ?", [case["requested_spill_dir"]]
                )
                effective_spill_dir = backend_object.connection.execute(
                    "SELECT current_setting('temp_directory')"
                ).fetchone()[0]
                spill_directory_observation = "duckdb_current_setting_temp_directory"
            else:
                from dbnumpy.backends import DataFusionBackend

                backend_object = DataFusionBackend.connect(
                    target_partitions=case["threads"],
                    memory_limit_bytes=int(case["engine_memory_gb"] * 1024**3),
                    max_densify_cells=max(rows, cols),
                )
                spill_directory_observation = (
                    "python_os_temp_path_not_queryable_datafusion_engine_setting"
                )
            connect_sec = time.perf_counter() - connect_started
            phase("host_arrow_upload")
            ingest_started = time.perf_counter()
            matrix = backend_object.from_coo(
                coo.row,
                coo.col,
                coo.data,
                shape=(rows, cols),
                name="benchmark_matrix",
            )
            ingest_sec = time.perf_counter() - ingest_started
            execution_path = (
                "datafusion_sql_reduction_native_pointwise_not_used"
                if backend_name == "datafusion-host"
                else "duckdb_host_arrow_sql_reduction"
            )

            def query() -> Any:
                return matrix.mean(axis=0)

        # Source preparation is deliberately reported separately. ``ready_sec``
        # measures only the selected execution path's database/host readiness.
        ready_sec = (
            parse_sec
            + connect_sec
            + ingest_sec
            + checkpoint_sec
            + reopen_sec
            + wrap_sec
        )
        if backend_name != "duckdb-native":
            phase("cold_colmeans")
            started = time.perf_counter()
            output = query()
            cold_query_sec = time.perf_counter() - started
        samples: list[float] = []
        phase("warm_colmeans")
        for _ in range(case["repeats"]):
            query_started = time.perf_counter()
            output = query()
            samples.append(time.perf_counter() - query_started)
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (cols,):
            raise AssertionError(
                f"column mean output shape is {output.shape}, expected {(cols,)}"
            )
        np.testing.assert_allclose(output, reference, rtol=1e-10, atol=1e-12)
        max_abs_error = (
            float(np.max(np.abs(output - reference))) if output.size else 0.0
        )
        row = {
            **case,
            "status": "ok",
            "error": "",
            "rows": rows,
            "cols": cols,
            "nnz": actual_nnz,
            "density": actual_nnz / (rows * cols) if rows * cols else 0.0,
            "prepared_source": prepared_source,
            "prepare_sec": prepare_sec,
            "connect_sec": connect_sec,
            "parse_sec": parse_sec,
            "ingest_sec": ingest_sec,
            "checkpoint_sec": checkpoint_sec,
            "reopen_sec": reopen_sec,
            "wrap_sec": wrap_sec,
            "cold_query_sec": cold_query_sec,
            "ready_sec": ready_sec,
            "query_samples": samples,
            **sample_stats(samples),
            "output_checksum": _checksum(output),
            "output_bytes": output.nbytes,
            "max_abs_error": max_abs_error,
            "execution_path": execution_path,
            "host_coo_materialized": backend_name != "duckdb-native",
            "scipy_loaded": "scipy" in sys.modules,
            "effective_spill_dir": str(effective_spill_dir),
            "spill_directory_observation": spill_directory_observation,
            "package_versions": _versions(),
            "last_phase": "complete",
            "message_type": "result",
        }
        connection.send(row)
    except BaseException as exc:
        connection.send(
            {
                **case,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "effective_spill_dir": str(effective_spill_dir),
                "spill_directory_observation": spill_directory_observation,
                "package_versions": _versions(),
                "last_phase": current_phase,
                "host_coo_materialized": backend_name != "duckdb-native",
                "scipy_loaded": "scipy" in sys.modules,
                "message_type": "result",
            }
        )
    finally:
        if backend_object is not None:
            backend_object.close()
        connection.close()


def read_rss_bytes(pid: int) -> int | None:
    """Return resident bytes using the portable macOS/Linux ``ps`` interface."""

    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        return int(output.splitlines()[-1].strip()) * 1024 if output else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _base_row(dataset: Dataset, backend: str) -> dict[str, Any]:
    execution_path = {
        "numpy-host": "optimized_numpy_bincount",
        "duckdb-host": "duckdb_host_arrow_sql_reduction",
        "datafusion-host": "datafusion_host_arrow_sql_reduction_native_pointwise_not_used",
        "duckdb-native": "duckdb_native_mtx_disk_table",
    }[backend]
    return {
        "size": dataset.size,
        "backend": backend,
        "source": str(dataset.path),
        "rows": dataset.rows,
        "cols": dataset.cols,
        "nnz": dataset.nnz,
        "density": dataset.density,
        "compressed_bytes": dataset.compressed_bytes,
        "execution_path": execution_path,
    }


def run_case(
    dataset: Dataset,
    backend: str,
    *,
    repeats: int,
    threads: int,
    engine_memory_gb: float,
    rss_limit_bytes: int,
    process_rss_limit_gb: float,
    timeout_sec: float,
    output_dir: Path,
    case_order: int,
    case_order_policy: str,
    native_input: str,
    reuse_prepared_mtx: bool,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    worker_tmp_dir = (
        output_dir / "worker-tmp" / f"{dataset.size}-{backend}-{uuid.uuid4().hex}"
    )
    worker_tmp_dir.mkdir(parents=True, exist_ok=True)
    case = {
        **_base_row(dataset, backend),
        "repeats": repeats,
        "threads": threads,
        "engine_memory_gb": engine_memory_gb,
        "process_rss_limit_gb": process_rss_limit_gb,
        "worker_tmp_dir": str(worker_tmp_dir.resolve()),
        "requested_spill_dir": str(worker_tmp_dir.resolve()),
        "effective_spill_dir": "",
        "spill_directory_observation": "unconfirmed",
        "case_order": case_order,
        "case_order_policy": case_order_policy,
        "native_input": native_input,
        "reuse_prepared_mtx": reuse_prepared_mtx,
        "prepared_source_target": str(
            (
                output_dir
                / "prepared-mtx"
                / f"{dataset.size}-{dataset.rows}x{dataset.cols}-{dataset.nnz}.mtx"
            ).resolve()
        ),
    }
    process = context.Process(target=_worker, args=(case, sending))
    process.start()
    sending.close()
    peak_rss = 0
    status: str | None = None
    last_phase = "worker_starting"
    final_row: dict[str, Any] | None = None
    deadline = time.monotonic() + timeout_sec
    while process.is_alive():
        while receiving.poll(0):
            try:
                message = receiving.recv()
            except EOFError:
                break
            if message.get("message_type") == "phase":
                last_phase = str(message.get("phase", "unknown"))
            else:
                final_row = message
        rss = read_rss_bytes(process.pid) if process.pid is not None else None
        if rss is not None:
            peak_rss = max(peak_rss, rss)
            if rss > rss_limit_bytes:
                status = "rss_guard"
                break
        if time.monotonic() >= deadline:
            status = "timeout"
            break
        process.join(0.1)
    if status is not None:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
    else:
        process.join()
    if status is not None:
        row = {
            **case,
            "status": status,
            "error": (
                f"RSS exceeded {rss_limit_bytes} bytes during phase {last_phase!r}"
                if status == "rss_guard"
                else f"case exceeded {timeout_sec:g} seconds during phase {last_phase!r}"
            ),
            "package_versions": _versions(),
            "effective_spill_dir": "unconfirmed_worker_terminated",
            "spill_directory_observation": "unconfirmed",
            "last_phase": last_phase,
        }
    else:
        while receiving.poll(0.2):
            try:
                message = receiving.recv()
            except EOFError:
                break
            if message.get("message_type") == "phase":
                last_phase = str(message.get("phase", "unknown"))
            else:
                final_row = message
        if final_row is not None:
            row = final_row
        else:
            row = {
                **case,
                "status": "error",
                "error": f"worker exited with code {process.exitcode} without a result; last phase={last_phase!r}",
                "package_versions": _versions(),
                "effective_spill_dir": "unconfirmed_worker_exit",
                "spill_directory_observation": "unconfirmed",
                "last_phase": last_phase,
            }
    receiving.close()
    row["peak_rss_bytes"] = peak_rss
    row["rss_scope"] = "worker_process_sampled_via_ps_includes_embedded_engine"
    row["rss_sampling_note"] = (
        "approximately_100ms_polling; sampled_peak_not_allocator_high_water_mark"
    )
    return row


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for original in rows:
            row = dict(original)
            row["query_samples_json"] = json.dumps(row.get("query_samples", []))
            row["package_versions_json"] = json.dumps(
                row.get("package_versions", {}), sort_keys=True
            )
            writer.writerow(row)
    temporary.replace(path)


def _config(args: argparse.Namespace, datasets: list[Dataset]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_semantics": "explicit_ingress_paths_structural_zero_colmeans_v1",
        "data_dir": str(args.data_dir.resolve()),
        "sizes": list(args.sizes),
        "backends": list(args.backends),
        "repeats": args.repeats,
        "threads": args.threads,
        "engine_memory_gb": args.engine_memory_gb,
        "process_rss_limit_gb": args.rss_limit_gb,
        "timeout_sec": args.timeout_sec,
        "native_input": args.native_input,
        "reuse_prepared_mtx": args.reuse_prepared_mtx,
        "case_order_policy": "rotate_backends_left_by_dataset_index",
        "sources": [str(dataset.path) for dataset in datasets],
    }


def run_benchmark(args: argparse.Namespace) -> int:
    discovered = discover_datasets(args.data_dir)
    missing = [size for size in args.sizes if size not in discovered]
    if missing:
        raise ValueError(f"requested dataset sizes not found: {missing}")
    datasets = [discovered[size] for size in args.sizes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    config = _config(args, datasets)
    if json_path.exists() or csv_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"results already exist in {args.output_dir}; pass --resume to continue"
            )
        if not json_path.is_file():
            raise ValueError("resume requires canonical results.json")
        document = json.loads(json_path.read_text(encoding="utf-8"))
        if document.get("config") != config:
            raise ValueError("existing results configuration does not match this run")
        rows = document.get("results")
        if not isinstance(rows, list):
            raise ValueError("existing results.json has an invalid results list")
        canonical: dict[tuple[Any, Any], dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict) and row.get("status") == "ok":
                canonical[(row.get("size"), row.get("backend"))] = row
        rows = list(canonical.values())
        document["results"] = rows
        _atomic_json(json_path, document)
        _write_csv(csv_path, rows)
    else:
        rows = []
        document = {"config": config, "results": rows}
        _atomic_json(json_path, document)
        _write_csv(csv_path, rows)
    completed = {
        (row.get("size"), row.get("backend"))
        for row in rows
        if row.get("status") == "ok"
    }
    policy = config["case_order_policy"]
    ordered_cases: list[tuple[Dataset, str]] = []
    for dataset_index, dataset in enumerate(datasets):
        offset = dataset_index % len(args.backends)
        rotated = args.backends[offset:] + args.backends[:offset]
        ordered_cases.extend((dataset, backend) for backend in rotated)
    for case_order, (dataset, backend) in enumerate(ordered_cases):
        if (dataset.size, backend) in completed:
            continue
        print(f"running size={dataset.size} backend={backend}", flush=True)
        row = run_case(
            dataset,
            backend,
            repeats=args.repeats,
            threads=args.threads,
            engine_memory_gb=args.engine_memory_gb,
            rss_limit_bytes=int(args.rss_limit_gb * 1024**3),
            process_rss_limit_gb=args.rss_limit_gb,
            timeout_sec=args.timeout_sec,
            output_dir=args.output_dir,
            case_order=case_order,
            case_order_policy=policy,
            native_input=args.native_input,
            reuse_prepared_mtx=args.reuse_prepared_mtx,
        )
        rows.append(row)
        _atomic_json(json_path, document)
        _write_csv(csv_path, rows)
        print(f"  status={row['status']}", flush=True)
    return int(any(row.get("status") != "ok" for row in rows))


def _load_one_plot_input(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(value, dict)
        and "config" in value
        and value.get("config", {}).get("schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{path}: incompatible benchmark schema; expected {RESULT_SCHEMA_VERSION}"
        )
    rows = value.get("results") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("plot input must contain a result-row list")
    return rows


def _load_plot_rows(
    paths: Path | list[Path] | tuple[Path, ...],
) -> list[dict[str, Any]]:
    """Load one or more result files and reject ambiguous duplicate points."""

    inputs = [paths] if isinstance(paths, Path) else list(paths)
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[int, str], Path] = {}
    for path in inputs:
        for row in _load_one_plot_input(path):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: plot result rows must be objects")
            try:
                key = (int(row["size"]), str(row["backend"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: every plot row needs an integer size and backend"
                ) from exc
            if key[1] not in DEFAULT_BACKENDS:
                raise ValueError(
                    f"{path}: ambiguous or unsupported execution path {key[1]!r}; expected an explicit host/native label"
                )
            previous = seen.get(key)
            if previous is not None:
                raise ValueError(
                    f"duplicate plot point size={key[0]} backend={key[1]!r} "
                    f"in {previous} and {path}"
                )
            seen[key] = path
            rows.append(row)
    return rows


def write_svg(
    rows: list[dict[str, Any]], output: Path, *, overwrite: bool = False
) -> None:
    """Write a dependency-free, accessible two-panel log-log benchmark plot."""

    if output.exists() and not overwrite:
        raise FileExistsError(f"plot already exists: {output}")
    width, height = 1120, 520
    colors = {
        "numpy-host": "#2468a2",
        "duckdb-host": "#d1495b",
        "datafusion-host": "#2a9d6f",
        "duckdb-native": "#7a3db8",
    }
    panels = (
        (70, "ready_sec", "Execution path → ready (preparation excluded)"),
        (590, "query_median_sec", "Warm colMeans time"),
    )
    plot_w, plot_h, top = 450, 330, 85
    numeric_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for key in (
            "nnz",
            "ready_sec",
            "query_median_sec",
            "query_p10_sec",
            "query_p90_sec",
        ):
            try:
                row[key] = float(row[key])
            except (KeyError, TypeError, ValueError):
                row[key] = math.nan
        numeric_rows.append(row)
    positive_x = [row["nnz"] for row in numeric_rows if row["nnz"] > 0]
    if not positive_x:
        raise ValueError("plot rows need positive NNZ and timing values")
    x_min, x_max = min(positive_x), max(positive_x)
    if x_min == x_max:
        x_min /= 1.5
        x_max *= 1.5

    def scale(
        value: float, low: float, high: float, start: float, extent: float
    ) -> float:
        return (
            start
            + (math.log10(value) - math.log10(low))
            / (math.log10(high) - math.log10(low))
            * extent
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Sparse matrix scaling benchmark</title>',
        '<desc id="desc">Two log-log line plots compare backend-ready and warm column-mean times by nonzero count. Whiskers show the tenth to ninetieth percentile; crosses, diamonds, and triangles mark timeout, process-RSS guard stops, and other failures.</desc>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="560" y="34" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="bold">Sparse matrix scaling benchmark</text>',
    ]
    for left, key, label in panels:
        y_keys = (
            ("query_median_sec", "query_p10_sec", "query_p90_sec")
            if key == "query_median_sec"
            else (key,)
        )
        positive_y = [
            row[y_key]
            for row in numeric_rows
            for y_key in y_keys
            if math.isfinite(row[y_key]) and row[y_key] > 0
        ]
        if not positive_y:
            raise ValueError(f"plot rows need positive {key} values")
        y_min, y_max = min(positive_y), max(positive_y)
        if y_min == y_max:
            y_min /= 1.5
            y_max *= 1.5
        bottom = top + plot_h
        parts.extend(
            [
                f'<text x="{left + plot_w / 2}" y="64" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{escape(label)}</text>',
                f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#777"/>',
                f'<text x="{left + plot_w / 2}" y="462" text-anchor="middle" font-family="sans-serif" font-size="13">NNZ (log scale)</text>',
                f'<text x="{left - 49}" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 {left - 49} {top + plot_h / 2})" font-family="sans-serif" font-size="13">Seconds (log scale)</text>',
            ]
        )
        for fraction in (0.0, 0.5, 1.0):
            x_value = 10 ** (
                math.log10(x_min) + fraction * (math.log10(x_max) - math.log10(x_min))
            )
            y_value = 10 ** (
                math.log10(y_min) + fraction * (math.log10(y_max) - math.log10(y_min))
            )
            x = left + fraction * plot_w
            y = bottom - fraction * plot_h
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#ddd"/>',
                    f'<text x="{x:.1f}" y="{bottom + 19}" text-anchor="middle" font-family="sans-serif" font-size="11">{x_value:.2g}</text>',
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>',
                    f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y_value:.2g}</text>',
                ]
            )
        for backend, color in colors.items():
            series = sorted(
                (
                    row
                    for row in numeric_rows
                    if row.get("backend") == backend
                    and row.get("status") == "ok"
                    and row["nnz"] > 0
                    and math.isfinite(row[key])
                    and row[key] > 0
                ),
                key=lambda row: row["nnz"],
            )
            points = [
                (
                    scale(row["nnz"], x_min, x_max, left, plot_w),
                    bottom - scale(row[key], y_min, y_max, 0, plot_h),
                    row,
                )
                for row in series
            ]
            if points:
                coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
                parts.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                )
            for x, y, row in points:
                if (
                    key == "query_median_sec"
                    and row["query_p10_sec"] > 0
                    and row["query_p90_sec"] > 0
                ):
                    high = bottom - scale(row["query_p90_sec"], y_min, y_max, 0, plot_h)
                    low = bottom - scale(row["query_p10_sec"], y_min, y_max, 0, plot_h)
                    parts.append(
                        f'<path d="M{x:.1f} {high:.1f}V{low:.1f}M{x - 4:.1f} {high:.1f}H{x + 4:.1f}M{x - 4:.1f} {low:.1f}H{x + 4:.1f}" stroke="{color}" fill="none"/>'
                    )
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white"/>'
                )
            failures = [
                row
                for row in numeric_rows
                if row.get("backend") == backend
                and row.get("status") != "ok"
                and row["nnz"] > 0
            ]
            for row in failures:
                x = scale(row["nnz"], x_min, x_max, left, plot_w)
                y = (
                    top + 13
                    if row["status"] == "timeout"
                    else top + 29
                    if row["status"] == "rss_guard"
                    else top + 45
                )
                y += {
                    "numpy-host": -10,
                    "duckdb-host": -3,
                    "datafusion-host": 4,
                    "duckdb-native": 11,
                }[backend]
                if row["status"] == "timeout":
                    parts.append(
                        f'<path d="M{x - 5:.1f} {y - 5}L{x + 5:.1f} {y + 5}M{x + 5:.1f} {y - 5}L{x - 5:.1f} {y + 5}" stroke="{color}" stroke-width="2.5"><title>{backend} timeout</title></path>'
                    )
                elif row["status"] == "rss_guard":
                    parts.append(
                        f'<path d="M{x:.1f} {y - 6}L{x + 6:.1f} {y}L{x:.1f} {y + 6}L{x - 6:.1f} {y}Z" fill="none" stroke="{color}" stroke-width="2.5"><title>{backend} RSS guard</title></path>'
                    )
                else:
                    status = escape(str(row.get("status") or "error"))
                    parts.append(
                        f'<path d="M{x:.1f} {y - 7}L{x + 7:.1f} {y + 6}L{x - 7:.1f} {y + 6}Z" fill="none" stroke="{color}" stroke-width="2.5"><title>{backend} {status}</title></path>'
                    )
    legend_x = 160
    for index, (backend, color) in enumerate(colors.items()):
        x = legend_x + index * 195
        parts.extend(
            [
                f'<line x1="{x}" y1="493" x2="{x + 25}" y2="493" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x + 12.5}" cy="493" r="4" fill="{color}"/>',
                f'<text x="{x + 33}" y="498" font-family="sans-serif" font-size="13">{backend}</text>',
            ]
        )
    parts.append(
        '<text x="1060" y="498" text-anchor="end" font-family="sans-serif" font-size="11">× timeout  ◇ RSS guard  △ other</text>'
    )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(parts) + "\n", encoding="utf-8")
    temporary.replace(output)


def _csv_list(value: str, *, choices: tuple[str, ...] | None = None) -> tuple[Any, ...]:
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if not fields:
        raise argparse.ArgumentTypeError(
            "value must be a non-empty comma-separated list"
        )
    if choices is not None:
        invalid = sorted(set(fields) - set(choices))
        if invalid:
            raise argparse.ArgumentTypeError(
                f"unsupported values: {', '.join(invalid)}"
            )
        if len(set(fields)) != len(fields):
            raise argparse.ArgumentTypeError("values must not be repeated")
        return fields
    try:
        sizes = tuple(int(field) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be positive integers") from exc
    if any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be unique positive integers")
    return sizes


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def repeat_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("repeats must be at most 100")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect available MTX headers"
    )
    inspect_parser.add_argument("--data-dir", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="run isolated benchmark cases")
    run_parser.add_argument("--data-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--sizes", type=_csv_list, default=DEFAULT_SIZES)
    run_parser.add_argument(
        "--backends",
        type=lambda value: _csv_list(value, choices=DEFAULT_BACKENDS),
        default=DEFAULT_BACKENDS,
    )
    run_parser.add_argument("--repeats", type=repeat_count, default=5)
    run_parser.add_argument("--threads", type=positive_int, default=2)
    run_parser.add_argument("--engine-memory-gb", type=positive_float, default=1.0)
    run_parser.add_argument("--rss-limit-gb", type=positive_float, default=5.0)
    run_parser.add_argument("--timeout-sec", type=positive_float, default=300.0)
    run_parser.add_argument(
        "--native-input",
        choices=("gzip", "plain"),
        default="gzip",
        help="give DuckDB the gzip source directly, or boundedly prepare a plain MTX under --output-dir",
    )
    run_parser.add_argument(
        "--reuse-prepared-mtx",
        action="store_true",
        help="validate and reuse an existing prepared plain MTX; never overwrite it",
    )
    run_parser.add_argument("--resume", action="store_true")

    plot_parser = subparsers.add_parser("plot", help="render result JSON/CSV as SVG")
    plot_parser.add_argument("--input", type=Path, nargs="+", required=True)
    plot_parser.add_argument("--output", type=Path, required=True)
    plot_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            datasets = discover_datasets(args.data_dir)
            print(
                json.dumps(
                    [
                        asdict(dataset)
                        | {"path": str(dataset.path), "density": dataset.density}
                        for dataset in datasets.values()
                    ],
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            return run_benchmark(args)
        write_svg(_load_plot_rows(args.input), args.output, overwrite=args.overwrite)
        return 0
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
