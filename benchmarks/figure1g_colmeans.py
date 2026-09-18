#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproduce Figure 1g column means with honest out-of-core pipelines.

Preparation is deliberately excluded from the measured operation.  Each case
runs in a fresh spawned process and measures one cold call followed by five
warm calls to the public matrix API.  Engine memory budgets are allowed to
spill; the supervisor has only a high, system-derived emergency RSS stop.

The DataFusion MTX preparation is intentionally fixture-specific.  Figure 1g
files contain a Matrix Market dimension row followed by unique, all-one,
general coordinates.  DataFusion reads that file as CSV, validates those
assumptions, removes the known dimension row, converts to zero-based indices,
and streams canonical Parquet without a Python/PyArrow coordinate table.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

ROWS = 20_000
WARM_SAMPLES = 5
SCHEMA_VERSION = 1
BACKENDS = (
    "scipy-host",
    "duckdb-mtx-table",
    "duckdb-parquet-scan",
    "datafusion-parquet-scan",
)
BACKEND_LABELS = {
    "scipy-host": "SciPy sparse: MTX parsed into host memory",
    "duckdb-mtx-table": "DuckDB: MTX ingested to persistent DuckDB table",
    "duckdb-parquet-scan": "DuckDB: external canonical Parquet scan",
    "datafusion-parquet-scan": "DataFusion: external canonical Parquet scan",
}


@dataclass(frozen=True)
class Figure1GSpec:
    columns: int
    sparsity: str
    nnz: int
    expected_mean: float

    @property
    def directory_name(self) -> str:
        return f"10x_synth_{ROWS}g_{self.columns}c_{self.sparsity}"


# This is an audit contract, not a filename heuristic.  Every backend receives
# the identical source for a given row in this series.
FIGURE1G_SERIES = (
    Figure1GSpec(1_000, "sp95", 1_000_000, 0.05),
    Figure1GSpec(3_000, "sp95", 3_000_000, 0.05),
    Figure1GSpec(10_000, "sp95", 10_000_000, 0.05),
    Figure1GSpec(30_000, "sp95", 30_000_000, 0.05),
    Figure1GSpec(100_000, "sp95", 100_000_000, 0.05),
    Figure1GSpec(1_000_000, "sp99", 200_000_000, 0.01),
    Figure1GSpec(3_000_000, "sp99", 600_000_000, 0.01),
    Figure1GSpec(10_000_000, "sp99", 2_000_000_000, 0.01),
)
SPEC_BY_COLUMNS = {spec.columns: spec for spec in FIGURE1G_SERIES}


@dataclass(frozen=True)
class Dataset:
    spec: Figure1GSpec
    path: Path
    compressed_bytes: int


def _read_header(path: Path) -> tuple[int, int, int, str, str]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as stream:
        banner = stream.readline().strip().split()
        if len(banner) != 5 or banner[:3] != [
            "%%MatrixMarket",
            "matrix",
            "coordinate",
        ]:
            raise ValueError(f"{path}: expected coordinate Matrix Market data")
        field, symmetry = banner[3].lower(), banner[4].lower()
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                dimensions = stripped.split()
                if len(dimensions) != 3:
                    break
                rows, columns, entries = (int(value) for value in dimensions)
                return rows, columns, entries, field, symmetry
    raise ValueError(f"{path}: missing Matrix Market dimensions")


def discover_datasets(data_dir: Path, columns: tuple[int, ...]) -> list[Dataset]:
    datasets: list[Dataset] = []
    for count in columns:
        try:
            spec = SPEC_BY_COLUMNS[count]
        except KeyError as exc:
            raise ValueError(f"{count} is not a Figure 1g column count") from exc
        directory = data_dir / spec.directory_name
        candidates = (directory / "matrix.mtx.gz", directory / "matrix.mtx")
        paths = [path for path in candidates if path.is_file()]
        if len(paths) != 1:
            raise ValueError(
                "expected exactly one matrix.mtx[.gz] in "
                f"{directory}, found {len(paths)}"
            )
        path = paths[0].resolve()
        rows, found_columns, nnz, field, symmetry = _read_header(path)
        expected = (ROWS, spec.columns, spec.nnz)
        if (rows, found_columns, nnz) != expected:
            raise ValueError(
                f"{path}: header {(rows, found_columns, nnz)} != expected {expected}"
            )
        if field not in {"integer", "real"} or symmetry != "general":
            raise ValueError(
                f"{path}: Figure 1g requires real/integer general coordinates"
            )
        datasets.append(Dataset(spec, path, path.stat().st_size))
    return datasets


def physical_memory_bytes() -> int:
    if platform.system() == "Darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def default_emergency_rss_bytes(total: int) -> int:
    """High last-resort stop; this is not an engine or benchmark limit."""

    # 80% gives a 16 GiB laptop roughly 3.2 GiB of system headroom.
    return int(total * 0.80)


def swap_snapshot() -> dict[str, int] | None:
    """Return system-wide macOS swap totals, not worker-owned memory."""

    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    unit_bytes = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    values = {
        key: int(float(value) * unit_bytes[unit])
        for key, value, unit in re.findall(
            r"\b(total|used|free)\s*=\s*([0-9.]+)([KMGT])", output
        )
    }
    return values if values.keys() == {"total", "used", "free"} else None


def swap_safety_limit_reached(
    snapshot: dict[str, int],
    *,
    minimum_free_bytes: int,
    minimum_used_bytes: int,
) -> bool:
    """Detect severe system swap pressure without flagging an idle swap pool."""

    # macOS grows its swap pool dynamically. A healthy machine may therefore
    # report total=free=0, or a small allocated pool with little absolute use.
    # Low free allocated swap is only an emergency signal once use is also high.
    return (
        snapshot["total"] > 0
        and snapshot["used"] >= minimum_used_bytes
        and snapshot["free"] < minimum_free_bytes
    )


def scipy_memory_lower_bounds(nnz: int, columns: int) -> dict[str, int]:
    """Deterministic payload minima, excluding allocator/library overhead."""

    return {
        "scipy_mmread_coo_lower_bound_bytes": nnz * (4 + 4 + 8),
        "scipy_csc_lower_bound_bytes": nnz * (4 + 8) + (columns + 1) * 4,
    }


def disk_snapshot(path: Path) -> dict[str, int]:
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    return {
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
    }


def tree_size(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    try:
        candidates = path.rglob("*")
        for item in candidates:
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                # Spill files can disappear between directory enumeration and stat.
                continue
    except OSError:
        pass
    return total


def _checksum(values: Any) -> str:
    import numpy as np

    array = np.asarray(values, dtype="<f8").reshape(-1)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate_result(values: Any, dataset: Dataset) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != dataset.spec.columns:
        raise ValueError(
            f"column mean length {array.size} != {dataset.spec.columns}"
        )
    error = float(np.max(np.abs(array - dataset.spec.expected_mean)))
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError(
            f"column means differ from {dataset.spec.expected_mean}; max error={error}"
        )
    return {
        "output_length": int(array.size),
        "output_checksum": _checksum(array),
        "max_abs_error": error,
        "reference_mean": dataset.spec.expected_mean,
        "reference_semantics": f"SUM(x) / {ROWS}; structural zeros included",
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact manifest: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_matching_manifest(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileExistsError(
            f"artifact exists without reusable manifest {path}; refusing overwrite"
        )
    actual = json.loads(path.read_text())
    if actual != expected:
        raise ValueError(f"artifact manifest does not match this case: {path}")


def _load_parquet_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileExistsError(
            f"artifact exists without reusable manifest {path}; refusing overwrite"
        )
    actual = json.loads(path.read_text())
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(f"artifact manifest does not match this case: {path}")
    validation = actual.get("parquet_validation")
    if not isinstance(validation, dict):
        raise ValueError(f"artifact manifest lacks Parquet validation: {path}")
    return validation


def _artifact_manifest(dataset: Dataset, kind: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source": str(dataset.path),
        "source_size": dataset.compressed_bytes,
        "source_mtime_ns": dataset.path.stat().st_mtime_ns,
        "rows": ROWS,
        "columns": dataset.spec.columns,
        "nnz": dataset.spec.nnz,
        "all_values": 1.0,
        "zero_based": True,
    }


def _prepare_datafusion_parquet(
    dataset: Dataset,
    destination: Path,
    *,
    memory_bytes: int,
    threads: int,
) -> tuple[float, bool, dict[str, Any]]:
    """Stream the known Figure 1g MTX fixture to canonical Parquet."""

    manifest = destination.with_suffix(".manifest.json")
    expected = _artifact_manifest(dataset, "datafusion-canonical-parquet")
    if destination.exists() or manifest.exists():
        if not destination.exists() or not manifest.is_file():
            raise FileExistsError(
                f"partial Parquet artifact exists; refusing overwrite: {destination}"
            )
        # This artifact is intentionally shared by both scan cases in one run.
        # On a resumed run the same manifest validation is required.
        validation = _load_parquet_manifest(manifest, expected)
        return 0.0, True, validation

    import pyarrow as pa
    from datafusion import (
        CsvReadOptions,
        RuntimeEnvBuilder,
        SessionConfig,
        SessionContext,
    )

    started = time.perf_counter()
    config = SessionConfig().with_target_partitions(threads)
    runtime = RuntimeEnvBuilder().with_disk_manager_os().with_fair_spill_pool(
        memory_bytes
    )
    context = SessionContext(config=config, runtime=runtime)
    schema = pa.schema(
        [("i", pa.int64()), ("j", pa.int64()), ("x", pa.float64())]
    )
    csv_options: dict[str, Any] = dict(
        has_header=False,
        delimiter=" ",
        comment="%",
        schema=schema,
        file_extension=dataset.path.suffix,
    )
    if dataset.path.suffix == ".gz":
        csv_options["file_compression_type"] = "GZIP"
    options = CsvReadOptions(**csv_options)
    context.register_csv("figure1g_raw", str(dataset.path), options=options)
    rows, columns, nnz = ROWS, dataset.spec.columns, dataset.spec.nnz
    canonical = context.sql(
        f"""
        SELECT cast(i - 1 AS BIGINT) AS i, cast(j - 1 AS BIGINT) AS j, x
        FROM figure1g_raw
        WHERE NOT (i={rows} AND j={columns} AND x={nnz}.0)
        """
    )
    # DataFusion creates a directory. It refuses the pre-existing destination;
    # a failed partial directory is retained for inspection and never deleted.
    canonical.write_parquet(str(destination), compression="zstd")
    parquet_paths = _parquet_files(destination)
    validation_frames = [context.read_parquet(str(path)) for path in parquet_paths]
    validation_frame = validation_frames[0]
    for other in validation_frames[1:]:
        validation_frame = validation_frame.union(other)
    context.register_table("figure1g_canonical_validation", validation_frame)
    validation = context.sql(
        f"""
        SELECT count(*) AS entries,
               min(i) AS min_i, max(i) AS max_i,
               min(j) AS min_j, max(j) AS max_j,
               min(x) AS min_x, max(x) AS max_x,
               sum(x) AS sum_x,
               sum(CASE WHEN i < 0 OR i >= {rows} OR j < 0 OR j >= {columns}
                         OR x <> 1.0 OR x IS NULL THEN 1 ELSE 0 END) AS invalid
        FROM figure1g_canonical_validation
        """
    ).to_arrow_table().to_pylist()[0]
    if (
        validation["entries"] != nnz
        or validation["invalid"] != 0
        or validation["min_i"] != 0
        or validation["max_i"] != rows - 1
        or validation["min_j"] != 0
        or validation["max_j"] != columns - 1
        or validation["min_x"] != 1.0
        or validation["max_x"] != 1.0
        or validation["sum_x"] != float(nnz)
    ):
        raise ValueError(f"Figure 1g Parquet validation failed: {validation}")
    _write_manifest(manifest, {**expected, "parquet_validation": validation})
    return time.perf_counter() - started, False, validation


def _parquet_files(path: Path) -> tuple[Path, ...]:
    files = (path,) if path.is_file() else tuple(sorted(path.glob("*.parquet")))
    if not files or any(not item.is_file() for item in files):
        raise ValueError(f"no regular Parquet part files found at {path}")
    return files


def _open_case(
    backend_name: str,
    dataset: Dataset,
    case_dir: Path,
    *,
    memory_bytes: int,
    duckdb_ingest_memory_bytes: int,
    threads: int,
    resume: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return (matrix, owner, preparation telemetry)."""

    prep: dict[str, Any] = {
        "preparation_sec": 0.0,
        "preparation_reused": False,
        "database_bytes": 0,
        "parquet_bytes": 0,
        "host_coordinates_materialized": False,
        "preparation_engine": None,
        "host_result_limit_values": 50_000_000,
        "host_result_limit_role": (
            "dbnumpy default max_host_values for eager result vectors; "
            "independent of dense-shape and engine-memory budgets"
        ),
    }
    if backend_name == "scipy-host":
        import scipy.io

        started = time.perf_counter()
        matrix = scipy.io.mmread(dataset.path).tocsc()
        prep["preparation_sec"] = time.perf_counter() - started
        prep["host_coordinates_materialized"] = True
        prep["preparation_engine"] = "SciPy"
        return matrix, None, prep

    spill = case_dir / f"spill-attempt-{time.time_ns()}"
    spill.mkdir(exist_ok=False)
    os.environ["TMPDIR"] = str(spill)
    prep["effective_spill_directory"] = str(spill)
    parquet = Path(case_dir.parent / "shared" / "canonical.parquet")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if backend_name in {"duckdb-parquet-scan", "datafusion-parquet-scan"}:
        seconds, reused, validation = _prepare_datafusion_parquet(
            dataset,
            parquet,
            memory_bytes=memory_bytes,
            threads=threads,
        )
        prep.update(
            preparation_sec=seconds,
            preparation_reused=reused,
            parquet_bytes=tree_size(parquet),
            preparation_engine="DataFusion streaming CSV-to-Parquet",
            parquet_validation=validation,
        )

    if backend_name == "duckdb-mtx-table":
        from dbnumpy import DuckDBBackend

        database = case_dir / "matrix.duckdb"
        manifest = case_dir / "matrix.duckdb.manifest.json"
        expected = {
            **_artifact_manifest(dataset, "duckdb-persistent-table"),
            "duckdb_mtx_reader_mode": "assume-canonical-direct-v1",
        }
        if database.exists() != manifest.exists():
            raise FileExistsError(
                "partial DuckDB artifact exists; refusing to create, replace, "
                f"or reuse it: {database}"
            )
        existed = database.exists() or manifest.exists()
        if existed and not resume:
            raise FileExistsError(f"refusing to overwrite DuckDB artifact: {database}")
        ingest_spill = spill / "ingest"
        query_spill = spill / "query"
        ingest_spill.mkdir()
        query_spill.mkdir()
        prep.update(
            duckdb_ingest_memory_budget_bytes=duckdb_ingest_memory_bytes,
            duckdb_ingest_memory_budget_role=(
                "persistent MTX-to-table ingestion and checkpoint only"
            ),
            duckdb_query_memory_budget_bytes=memory_bytes,
            duckdb_query_memory_budget_role="public mean(axis=0) operation only",
            duckdb_ingest_spill_directory=str(ingest_spill),
            duckdb_query_spill_directory=str(query_spill),
            duckdb_mtx_reader_mode="assume-canonical-direct-v1",
            duckdb_preserve_insertion_order=True,
        )
        if existed:
            _load_matching_manifest(manifest, expected)
            prep["preparation_reused"] = True
        else:
            ingest_backend = None
            try:
                ingest_backend = DuckDBBackend.connect(
                    database,
                    memory_limit=f"{duckdb_ingest_memory_bytes}B",
                    threads=threads,
                    temp_directory=ingest_spill,
                )
                started = time.perf_counter()
                ingest_backend.from_mtx(
                    dataset.path,
                    name="figure1g_matrix",
                    assume_canonical=True,
                )
                ingest_backend.connection.execute("CHECKPOINT")
                prep["preparation_sec"] = time.perf_counter() - started
            finally:
                if ingest_backend is not None:
                    ingest_backend.close()
            _write_manifest(manifest, expected)
        prep["database_bytes"] = tree_size(database)
        prep["preparation_engine"] = "DuckDB native MTX-to-table"
        query_backend = None
        try:
            query_backend = DuckDBBackend.connect(
                database,
                memory_limit=f"{memory_bytes}B",
                threads=threads,
                temp_directory=query_spill,
            )
            matrix = query_backend.from_relation(
                "figure1g_matrix",
                shape=(ROWS, dataset.spec.columns),
                storage="sparse",
            )
            return matrix, query_backend, prep
        except Exception:
            if query_backend is not None:
                try:
                    query_backend.close()
                except Exception:
                    pass
            raise

    if backend_name == "duckdb-parquet-scan":
        from dbnumpy import DuckDBBackend

        duck_backend = DuckDBBackend.connect(
            memory_limit=f"{memory_bytes}B",
            threads=threads,
            temp_directory=spill,
        )
        parquet_files = _parquet_files(parquet)
        parquet_source = [str(path).replace("'", "''") for path in parquet_files]
        parquet_argument = (
            f"'{parquet_source[0]}'"
            if len(parquet_source) == 1
            else "[" + ",".join(f"'{path}'" for path in parquet_source) + "]"
        )
        duck_backend.connection.execute(
            "CREATE VIEW figure1g_matrix AS SELECT i, j, x "
            f"FROM read_parquet({parquet_argument})"
        )
        matrix = duck_backend.from_relation(
            "figure1g_matrix",
            shape=(ROWS, dataset.spec.columns),
            storage="sparse",
        )
        return matrix, duck_backend, prep

    if backend_name == "datafusion-parquet-scan":
        from dbnumpy import DataFusionBackend

        datafusion_backend: Any = DataFusionBackend.connect(
            target_partitions=threads,
            memory_limit_bytes=memory_bytes,
        )
        matrix = datafusion_backend.from_parquet(
            _parquet_files(parquet),
            shape=(ROWS, dataset.spec.columns),
            name="figure1g_matrix",
        )
        return matrix, datafusion_backend, prep

    raise ValueError(f"unknown backend {backend_name!r}")


def _run_worker(case: dict[str, Any], connection: Any) -> None:
    row = dict(case["row"])
    owner = None
    try:
        case_dir = Path(case["case_dir"])
        # A resume gets a new spill directory because stale spill files are never
        # removed. Persistent benchmark artifacts are reused only by manifest.
        case_dir.mkdir(parents=True, exist_ok=case["resume"])
        row.update(disk_before=disk_snapshot(case_dir.parent))
        connection.send({"event": "phase", "phase": "preparation"})
        dataset = Dataset(
            Figure1GSpec(**case["spec"]),
            Path(row["source"]),
            case["compressed_bytes"],
        )
        matrix, owner, preparation = _open_case(
            case["backend"],
            dataset,
            case_dir,
            memory_bytes=case["memory_bytes"],
            duckdb_ingest_memory_bytes=case["duckdb_ingest_memory_bytes"],
            threads=case["threads"],
            resume=case["resume"],
        )
        row.update(preparation)
        connection.send({"event": "phase", "phase": "cold-column-means"})
        gc.collect()
        started = time.perf_counter()
        cold = matrix.mean(axis=0)
        row["cold_query_sec"] = time.perf_counter() - started
        validation = _validate_result(cold, dataset)

        samples = []
        for index in range(WARM_SAMPLES):
            connection.send(
                {"event": "phase", "phase": f"warm-column-means-{index + 1}"}
            )
            gc.collect()
            started = time.perf_counter()
            warm = matrix.mean(axis=0)
            samples.append(time.perf_counter() - started)
            current = _validate_result(warm, dataset)
            if current["output_checksum"] != validation["output_checksum"]:
                raise ValueError("column means checksum changed between calls")
        row.update(validation)
        row.update(
            status="success",
            outcome_kind="backend_result",
            error=None,
            warm_query_samples_sec=samples,
            warm_query_median_sec=statistics.median(samples),
            disk_after=disk_snapshot(case_dir.parent),
            spill_bytes=tree_size(Path(preparation["effective_spill_directory"]))
            if "effective_spill_directory" in preparation
            else 0,
        )
    except Exception as exc:
        row.update(
            status="error",
            outcome_kind="backend_result",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            disk_after=disk_snapshot(Path(case["case_dir"]).parent),
            spill_bytes=sum(
                tree_size(path)
                for path in Path(case["case_dir"]).glob("spill-attempt-*")
            ),
        )
    finally:
        if owner is not None:
            try:
                owner.close()
            except Exception:
                pass
        connection.send({"event": "result", "row": row})
        connection.close()


def _rss_bytes(pid: int) -> int:
    try:
        if platform.system() == "Darwin":
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(output) * 1024 if output else 0
        statm = Path(f"/proc/{pid}/statm").read_text().split()
        return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def run_case(
    case: dict[str, Any],
    emergency_bytes: int,
    timeout_seconds: float,
    emergency_free_swap_bytes: int | None = None,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_run_worker, args=(case, child))
    process.start()
    assert process.pid is not None
    worker_pid = process.pid
    child.close()
    started = time.monotonic()
    peak = 0
    peak_spill = 0
    peak_case_bytes = 0
    minimum_free_disk = disk_snapshot(Path(case["case_dir"]).parent)["disk_free_bytes"]
    initial_swap = swap_snapshot()
    minimum_free_swap = initial_swap["free"] if initial_swap is not None else None
    peak_used_swap = initial_swap["used"] if initial_swap is not None else None
    phase = "worker-start"
    phase_peaks: dict[str, int] = {}
    phase_spill_peaks: dict[str, int] = {}
    result: dict[str, Any] | None = None
    pipe_open = True
    while process.is_alive() or (pipe_open and parent.poll()):
        while pipe_open and parent.poll():
            try:
                message = parent.recv()
            except EOFError:
                pipe_open = False
                break
            if message["event"] == "phase":
                phase = message["phase"]
            else:
                result = message["row"]
        rss = _rss_bytes(worker_pid)
        case_dir = Path(case["case_dir"])
        spill = sum(tree_size(path) for path in case_dir.glob("spill-attempt-*"))
        case_bytes = tree_size(case_dir)
        free_disk = disk_snapshot(case_dir.parent)["disk_free_bytes"]
        swap = swap_snapshot()
        peak = max(peak, rss)
        peak_spill = max(peak_spill, spill)
        peak_case_bytes = max(peak_case_bytes, case_bytes)
        minimum_free_disk = min(minimum_free_disk, free_disk)
        if swap is not None:
            minimum_free_swap = min(
                swap["free"] if minimum_free_swap is None else minimum_free_swap,
                swap["free"],
            )
            peak_used_swap = max(
                swap["used"] if peak_used_swap is None else peak_used_swap,
                swap["used"],
            )
        phase_peaks[phase] = max(phase_peaks.get(phase, 0), rss)
        phase_spill_peaks[phase] = max(phase_spill_peaks.get(phase, 0), spill)
        if rss > emergency_bytes:
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            result = dict(case["row"])
            result.update(
                status="system-emergency-safety-stop",
                outcome_kind="system_safety_outcome",
                error=(
                    f"total worker-tree RSS {rss} exceeded emergency threshold "
                    f"{emergency_bytes}; this is not an engine memory-limit result"
                ),
                emergency_phase=phase,
                disk_after=disk_snapshot(Path(case["case_dir"]).parent),
            )
            break
        if (
            emergency_free_swap_bytes is not None
            and swap is not None
            and swap_safety_limit_reached(
                swap,
                minimum_free_bytes=emergency_free_swap_bytes,
                minimum_used_bytes=emergency_bytes,
            )
        ):
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            result = dict(case["row"])
            result.update(
                status="system-swap-safety-stop",
                outcome_kind="system_safety_outcome",
                error=(
                    f"system swap use {swap['used']} was at least "
                    f"{emergency_bytes} while free allocated swap "
                    f"{swap['free']} fell below {emergency_free_swap_bytes}; "
                    "this is not an engine memory-limit result"
                ),
                emergency_phase=phase,
                disk_after=disk_snapshot(Path(case["case_dir"]).parent),
            )
            break
        if time.monotonic() - started > timeout_seconds:
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            result = dict(case["row"])
            result.update(
                status="system-timeout-safety-stop",
                outcome_kind="system_safety_outcome",
                error=(
                    f"fresh worker exceeded {timeout_seconds} seconds; this is "
                    "not an engine memory-limit result"
                ),
                emergency_phase=phase,
                disk_after=disk_snapshot(Path(case["case_dir"]).parent),
            )
            break
        time.sleep(0.2)
    process.join()
    if result is None:
        result = dict(case["row"])
        result.update(
            status="worker-exit",
            outcome_kind="system_safety_outcome",
            error=f"worker exited with code {process.exitcode} without a result",
        )
    result.update(
        peak_rss_bytes=peak,
        phase_peak_rss_bytes=phase_peaks,
        peak_spill_bytes=peak_spill,
        phase_peak_spill_bytes=phase_spill_peaks,
        peak_case_directory_bytes=peak_case_bytes,
        minimum_filesystem_free_bytes=minimum_free_disk,
        emergency_rss_threshold_bytes=emergency_bytes,
        emergency_free_swap_threshold_bytes=emergency_free_swap_bytes,
        emergency_swap_used_floor_bytes=emergency_bytes,
        initial_swap_bytes=initial_swap,
        minimum_free_swap_bytes=minimum_free_swap,
        peak_used_swap_bytes=peak_used_swap,
        timeout_seconds=timeout_seconds,
        rss_scope="fresh worker process, sampled every 0.2 seconds",
        disk_sampling_note=(
            "spill bytes, case-directory bytes, and filesystem free bytes sampled "
            "every 0.2 seconds; transient changes between samples may be missed"
        ),
    )
    return result


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True)
        if isinstance(value, (dict, list))
        else value
        for key, value in row.items()
    }


def write_results(
    rows: list[dict[str, Any]], output_dir: Path, run_config: dict[str, Any]
) -> None:
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_config": run_config,
        "results": rows,
    }
    temporary = output_dir / f".results.{os.getpid()}.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, json_path)
    flattened = [_flatten(row) for row in rows]
    keys = sorted({key for row in flattened for key in row})
    temporary_csv = output_dir / f".results.{os.getpid()}.csv.tmp"
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flattened)
    os.replace(temporary_csv, csv_path)


PLOT_COLORS = dict(
    zip(BACKENDS, ("#4d4d4d", "#6a3d9a", "#d95f02", "#087f5b"), strict=True)
)


def _log_scale(value: float, low: float, high: float, start: float, span: float) -> float:
    return start + (math.log10(value) - math.log10(low)) / (
        math.log10(high) - math.log10(low)
    ) * span


def _format_nnz(value: float) -> str:
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _format_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1e6:g} µs"
    if value < 1:
        return f"{value * 1e3:g} ms"
    return f"{value:g} s"


def _log_ticks(low: float, high: float) -> list[float]:
    ticks = [
        10.0**exponent
        for exponent in range(math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1)
    ]
    if len(ticks) < 2:
        ticks = [low, high]
    return ticks


def _staggered_labels(
    desired: dict[str, float], *, low: float, high: float, gap: float = 16
) -> dict[str, float]:
    ordered = sorted(desired, key=lambda key: (desired[key], key))
    positions: dict[str, float] = {}
    cursor = low
    for key in ordered:
        positions[key] = max(desired[key], cursor)
        cursor = positions[key] + gap
    cursor = high
    for key in reversed(ordered):
        positions[key] = min(positions[key], cursor)
        cursor = positions[key] - gap
    return positions


def _write_svg(destination: Path, parts: list[str]) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(parts) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def plot_results(
    rows: list[dict[str, Any]],
    destination: Path,
    *,
    backends: tuple[str, ...],
    title: str,
) -> None:
    """Write operation-only cold and warm timing panels."""

    selected = [row for row in rows if row["backend"] in backends and row["nnz"] > 0]
    if not selected:
        return
    x_min, x_max = min(float(row["nnz"]) for row in selected), max(
        float(row["nnz"]) for row in selected
    )
    if x_min == x_max:
        x_min /= 1.5
        x_max *= 1.5
    width, height = 980, 710
    left, plot_width, plot_height = 82, 565, 235
    panels = (
        (92, "cold_query_sec", "Cold mean(axis=0)"),
        (400, "warm_query_median_sec", "Warm mean(axis=0), median of 5"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}" role="img" aria-labelledby="operation-title operation-desc">',
        f'<title id="operation-title">{escape(title)}</title>',
        '<desc id="operation-desc">Cold and warm column-mean operation time in seconds versus stored nonzero entries. Input preparation is excluded. Successful timings are connected circles with direct full-path labels. Failures are plain text and are not plotted as measurements.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">{escape(title)}</text>',
        '<text x="490" y="50" text-anchor="middle" font-family="sans-serif" font-size="12">Operation time only · input preparation excluded · both axes logarithmic</text>',
    ]
    scipy_rows = [row for row in selected if row["backend"] == "scipy-host"]
    if scipy_rows:
        largest = max(scipy_rows, key=lambda row: row["nnz"])
        bounds = scipy_memory_lower_bounds(int(largest["nnz"]), int(largest.get("columns", 0)))
        parts.append(
            f'<text x="490" y="68" text-anchor="middle" font-family="sans-serif" font-size="10.5">SciPy payload minimum at {_format_nnz(float(largest["nnz"]))} NNZ: COO {bounds["scipy_mmread_coo_lower_bound_bytes"] / 1e9:.2f} GB; CSC {bounds["scipy_csc_lower_bound_bytes"] / 1e9:.2f} GB, before overhead</text>'
        )
    for top, metric, panel_label in panels:
        bottom = top + plot_height
        timings = [
            float(row[metric])
            for row in selected
            if row.get("status") == "success" and float(row.get(metric, 0)) > 0
        ]
        y_min, y_max = (min(timings) / 1.8, max(timings) * 1.8) if timings else (0.01, 1.0)
        if math.isclose(y_min, y_max):
            y_min /= 2
            y_max *= 2
        parts.extend(
            [
                f'<text x="{left}" y="{top - 10}" font-family="sans-serif" font-size="13" font-weight="bold">{escape(panel_label)}</text>',
                f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#777"/>',
                f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">Elapsed time</text>',
            ]
        )
        for x_value in _log_ticks(x_min, x_max):
            x = _log_scale(x_value, x_min, x_max, left, plot_width)
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#ddd"/>',
                    f'<text x="{x:.1f}" y="{bottom + 17}" text-anchor="middle" font-family="sans-serif" font-size="10.5">{escape(_format_nnz(x_value))}</text>',
                ]
            )
        for y_value in _log_ticks(y_min, y_max):
            y = bottom - _log_scale(y_value, y_min, y_max, 0, plot_height)
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#ddd"/>',
                    f'<text x="{left - 7}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10.5">{escape(_format_seconds(y_value))}</text>',
                ]
            )
        endpoints: dict[str, tuple[float, float]] = {}
        for backend in backends:
            series = sorted(
                (row for row in selected if row["backend"] == backend and row.get("status") == "success" and float(row.get(metric, 0)) > 0),
                key=lambda row: row["nnz"],
            )
            points = [
                (
                    _log_scale(float(row["nnz"]), x_min, x_max, left, plot_width),
                    bottom - _log_scale(float(row[metric]), y_min, y_max, 0, plot_height),
                )
                for row in series
            ]
            if points:
                coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{PLOT_COLORS[backend]}" stroke-width="2.4"/>')
                parts.extend(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PLOT_COLORS[backend]}" stroke="white"><title>{escape(BACKEND_LABELS[backend])}</title></circle>'
                    for x, y in points
                )
                endpoints[backend] = points[-1]
        label_y = _staggered_labels(
            {backend: point[1] for backend, point in endpoints.items()},
            low=top + 12,
            high=bottom - 8,
        )
        for backend, (point_x, point_y) in endpoints.items():
            y = label_y[backend]
            parts.extend(
                [
                    f'<path d="M{point_x + 5:.1f} {point_y:.1f} L{left + plot_width + 12:.1f} {y:.1f}" fill="none" stroke="{PLOT_COLORS[backend]}" stroke-width="1"/>',
                    f'<text x="{left + plot_width + 17}" y="{y + 4:.1f}" font-family="sans-serif" font-size="10.5" fill="{PLOT_COLORS[backend]}">{escape(BACKEND_LABELS[backend])}</text>',
                ]
            )
    failures = [row for row in selected if row.get("status") != "success"]
    for index, row in enumerate(failures):
        parts.append(
            f'<text x="{left}" y="{354 + index * 12}" font-family="sans-serif" font-size="9.5" fill="{PLOT_COLORS[row["backend"]]}">No operation timing at {_format_nnz(float(row["nnz"]))} NNZ — {escape(BACKEND_LABELS[row["backend"]])}: {escape(str(row["status"]))}</text>'
        )
    parts.extend(
        [
            '<text x="365" y="694" text-anchor="middle" font-family="sans-serif" font-size="12">Stored nonzero entries in the common MTX input</text>',
            "</svg>",
        ]
    )
    _write_svg(destination, parts)


def plot_preparation_results(
    rows: list[dict[str, Any]], destination: Path, *, backends: tuple[str, ...]
) -> None:
    """Write a preparation-only plot; query timing and query spill are absent."""

    selected = [row for row in rows if row["backend"] in backends and row["nnz"] > 0]
    if not selected:
        return
    x_min, x_max = min(float(row["nnz"]) for row in selected), max(float(row["nnz"]) for row in selected)
    if x_min == x_max:
        x_min /= 1.5
        x_max *= 1.5
    measured = [
        row
        for row in selected
        if not row.get("preparation_reused") and float(row.get("preparation_sec", 0)) > 0
    ]
    timings = [float(row["preparation_sec"]) for row in measured]
    y_min, y_max = (min(timings) / 1.8, max(timings) * 1.8) if timings else (0.01, 1.0)
    if math.isclose(y_min, y_max):
        y_min /= 2
        y_max *= 2
    width, height = 980, 550
    left, top, plot_width, plot_height = 82, 100, 565, 285
    bottom = top + plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}" role="img" aria-labelledby="preparation-title preparation-desc">',
        '<title id="preparation-title">Input preparation time</title>',
        '<desc id="preparation-desc">Input preparation time in seconds versus stored nonzero entries. Reused artifacts are identified with plain text and are not plotted as zero-second measurements. Column-mean operation timing and operation spill are not included.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="490" y="30" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">Input preparation time</text>',
        '<text x="490" y="52" text-anchor="middle" font-family="sans-serif" font-size="12">Preparation only · mean(axis=0) and operation spill are not shown</text>',
        '<text x="490" y="70" text-anchor="middle" font-family="sans-serif" font-size="10.5">“Reused artifact” means preparation happened in an earlier case or resumed run and was not timed again.</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#777"/>',
        f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">Preparation time</text>',
    ]
    for x_value in _log_ticks(x_min, x_max):
        x = _log_scale(x_value, x_min, x_max, left, plot_width)
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#ddd"/>',
                f'<text x="{x:.1f}" y="{bottom + 17}" text-anchor="middle" font-family="sans-serif" font-size="10.5">{escape(_format_nnz(x_value))}</text>',
            ]
        )
    for y_value in _log_ticks(y_min, y_max):
        y = bottom - _log_scale(y_value, y_min, y_max, 0, plot_height)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left - 7}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10.5">{escape(_format_seconds(y_value))}</text>',
            ]
        )
    endpoints: dict[str, tuple[float, float]] = {}
    for backend in backends:
        series = sorted((row for row in measured if row["backend"] == backend), key=lambda row: row["nnz"])
        points = [
            (
                _log_scale(float(row["nnz"]), x_min, x_max, left, plot_width),
                bottom - _log_scale(float(row["preparation_sec"]), y_min, y_max, 0, plot_height),
            )
            for row in series
        ]
        if points:
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{PLOT_COLORS[backend]}" stroke-width="2.4"/>')
            parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PLOT_COLORS[backend]}" stroke="white"/>' for x, y in points)
            endpoints[backend] = points[-1]
    label_y = _staggered_labels(
        {backend: point[1] for backend, point in endpoints.items()},
        low=top + 12,
        high=bottom - 8,
    )
    for backend, (point_x, point_y) in endpoints.items():
        y = label_y[backend]
        parts.extend(
            [
                f'<path d="M{point_x + 5:.1f} {point_y:.1f} L{left + plot_width + 12:.1f} {y:.1f}" fill="none" stroke="{PLOT_COLORS[backend]}" stroke-width="1"/>',
                f'<text x="{left + plot_width + 17}" y="{y + 4:.1f}" font-family="sans-serif" font-size="10.5" fill="{PLOT_COLORS[backend]}">{escape(BACKEND_LABELS[backend])}</text>',
            ]
        )
    note_groups: list[tuple[str, str, int]] = []
    for backend in backends:
        reused = [row for row in selected if row["backend"] == backend and row.get("preparation_reused")]
        failures = [row for row in selected if row["backend"] == backend and row.get("status") != "success" and not row.get("preparation_reused")]
        if reused:
            note_groups.append((backend, f"{len(reused)} reused artifact case(s); preparation not retimed", max(int(row["nnz"]) for row in reused)))
        if failures:
            note_groups.append((backend, f"{len(failures)} case(s) without preparation timing: {failures[-1]['status']}", max(int(row["nnz"]) for row in failures)))
    if note_groups:
        parts.append('<text x="82" y="425" font-family="sans-serif" font-size="10" font-weight="bold">Not plotted as preparation measurements:</text>')
    for index, (backend, note, largest_nnz) in enumerate(note_groups):
        parts.append(
            f'<text x="82" y="{442 + index * 13}" font-family="sans-serif" font-size="9.5" fill="{PLOT_COLORS[backend]}">{escape(BACKEND_LABELS[backend])} — {escape(note)}; through {_format_nnz(float(largest_nnz))} NNZ</text>'
        )
    parts.extend(
        [
            '<text x="365" y="535" text-anchor="middle" font-family="sans-serif" font-size="12">Stored nonzero entries in the common MTX input</text>',
            "</svg>",
        ]
    )
    _write_svg(destination, parts)


def parse_sizes(raw: str) -> tuple[int, ...]:
    if raw == "all":
        return tuple(spec.columns for spec in FIGURE1G_SERIES)
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("--sizes must contain unique Figure 1g column counts")
    unknown = set(values) - SPEC_BY_COLUMNS.keys()
    if unknown:
        raise ValueError(f"unknown Figure 1g sizes: {sorted(unknown)}")
    return values


def resolve_duckdb_ingest_memory_gb(
    engine_memory_gb: float, override_gb: float | None
) -> float:
    return engine_memory_gb if override_gb is None else override_gb


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        default="1000,3000,10000",
        help="comma-separated column counts, or 'all' (includes 2B NNZ)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=86_400.0,
        help="supervisor timeout for each fresh-process case (default: 24 hours)",
    )
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--engine-memory-gb", type=float, default=1.0)
    parser.add_argument(
        "--duckdb-ingest-memory-gb",
        type=float,
        help=(
            "DuckDB MTX-to-table preparation budget; defaults to "
            "--engine-memory-gb (operation budget remains unchanged)"
        ),
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--emergency-rss-gb",
        type=float,
        help="last-resort system safety stop; default is 80%% of physical RAM",
    )
    parser.add_argument(
        "--emergency-free-swap-gb",
        type=float,
        default=1.0 if platform.system() == "Darwin" else None,
        help=(
            "last-resort macOS free-swap floor; default 1 GiB on macOS and "
            "disabled elsewhere"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.engine_memory_gb <= 0
        or (
            args.duckdb_ingest_memory_gb is not None
            and args.duckdb_ingest_memory_gb <= 0
        )
        or (
            args.emergency_free_swap_gb is not None
            and args.emergency_free_swap_gb <= 0
        )
        or args.threads < 1
        or args.timeout_sec <= 0
    ):
        raise ValueError("memory budgets, threads, and timeout must be positive")
    columns = parse_sizes(args.sizes)
    datasets = discover_datasets(args.data_dir.resolve(), columns)
    total_memory = physical_memory_bytes()
    emergency = (
        int(args.emergency_rss_gb * 1024**3)
        if args.emergency_rss_gb is not None
        else default_emergency_rss_bytes(total_memory)
    )
    memory_bytes = int(args.engine_memory_gb * 1024**3)
    emergency_free_swap_bytes = (
        int(args.emergency_free_swap_gb * 1024**3)
        if args.emergency_free_swap_gb is not None
        else None
    )
    duckdb_ingest_memory_bytes = int(
        resolve_duckdb_ingest_memory_gb(
            args.engine_memory_gb, args.duckdb_ingest_memory_gb
        )
        * 1024**3
    )
    run_config = {
        "data_dir": str(args.data_dir.resolve()),
        "datasets": [
            {
                "columns": dataset.spec.columns,
                "source": str(dataset.path),
                "source_bytes": dataset.compressed_bytes,
                "source_mtime_ns": dataset.path.stat().st_mtime_ns,
            }
            for dataset in datasets
        ],
        "backends": list(args.backends),
        "engine_memory_budget_bytes": memory_bytes,
        "engine_memory_budget_role": "timed public operations and scan spill budget",
        "duckdb_ingest_memory_budget_bytes": duckdb_ingest_memory_bytes,
        "duckdb_ingest_memory_budget_role": (
            "duckdb-mtx-table ingestion and checkpoint only"
        ),
        "threads": args.threads,
        "emergency_rss_threshold_bytes": emergency,
        "emergency_free_swap_threshold_bytes": emergency_free_swap_bytes,
        "timeout_seconds": args.timeout_sec,
        "warm_samples": WARM_SAMPLES,
    }
    identity = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = args.output_dir.resolve()
    results_path = output / "results.json"
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"output directory exists: {output}; refusing overwrite (use --resume)"
        )
    rows: list[dict[str, Any]] = []
    if args.resume and results_path.is_file():
        payload = json.loads(results_path.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cannot resume a different result schema")
        if payload.get("run_config") != run_config:
            raise ValueError(
                "resume configuration differs from results.json; use a new output "
                "directory instead of mixing runs"
            )
        rows = payload["results"]
    output.mkdir(parents=True, exist_ok=args.resume)
    completed = {
        (row["columns"], row["backend"])
        for row in rows
        if row.get("status") == "success"
    }
    for dataset in datasets:
        for backend in args.backends:
            key = (dataset.spec.columns, backend)
            if key in completed:
                continue
            case_dir = output / "artifacts" / str(dataset.spec.columns) / backend
            row = {
                "schema_version": SCHEMA_VERSION,
                "backend": backend,
                "backend_label": BACKEND_LABELS[backend],
                "source": str(dataset.path),
                "rows": ROWS,
                "columns": dataset.spec.columns,
                "nnz": dataset.spec.nnz,
                "sparsity_fixture": dataset.spec.sparsity,
                "compressed_source_bytes": dataset.compressed_bytes,
                "engine_memory_budget_bytes": memory_bytes,
                "engine_memory_budget_role": (
                    "timed public operations and query/scan spill budget; not a "
                    "process cutoff"
                ),
                "duckdb_ingest_memory_budget_bytes": duckdb_ingest_memory_bytes,
                "duckdb_ingest_memory_budget_role": (
                    "duckdb-mtx-table ingestion and checkpoint only; other "
                    "backends do not use this budget"
                ),
                "threads": args.threads,
                "case_dir": str(case_dir),
                "requested_spill_directory_parent": str(case_dir),
                "spill_configuration": (
                    "fresh spill-attempt-* directory; DuckDB temp_directory or "
                    "DataFusion OS disk manager with TMPDIR"
                ),
                "fresh_worker_process": True,
                "run_config_sha256": identity,
                "timeout_seconds": args.timeout_sec,
            }
            if backend == "scipy-host":
                row.update(
                    scipy_memory_lower_bounds(
                        dataset.spec.nnz, dataset.spec.columns
                    )
                )
            case = {
                "row": row,
                "backend": backend,
                "spec": asdict(dataset.spec),
                "compressed_bytes": dataset.compressed_bytes,
                "case_dir": str(case_dir),
                "memory_bytes": memory_bytes,
                "duckdb_ingest_memory_bytes": duckdb_ingest_memory_bytes,
                "threads": args.threads,
                "resume": args.resume,
            }
            result = run_case(
                case,
                emergency,
                args.timeout_sec,
                emergency_free_swap_bytes,
            )
            # A resumed failed/safety attempt is replaced, not retained as a
            # second contradictory point for the same dataset/backend key.
            rows = [
                existing
                for existing in rows
                if (existing["columns"], existing["backend"]) != key
            ]
            rows.append(result)
            write_results(rows, output, run_config)
            plot_results(
                rows,
                output / "figure1g-colmeans.svg",
                backends=(
                    "scipy-host",
                    "duckdb-mtx-table",
                    "datafusion-parquet-scan",
                ),
                title="Column means: practical pipelines",
            )
            plot_results(
                rows,
                output / "figure1g-parquet-control.svg",
                backends=("duckdb-parquet-scan", "datafusion-parquet-scan"),
                title="Column means: shared Parquet control",
            )
            plot_preparation_results(
                rows,
                output / "figure1g-preparation.svg",
                backends=BACKENDS,
            )
    return int(any(row.get("status") != "success" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
