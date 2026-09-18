#!/usr/bin/env python3
"""Stress Figure 1g column means under a small engine spill budget.

This runner consumes already-prepared canonical artifacts. It never parses MTX,
constructs host coordinate arrays, rewrites the persistent DuckDB database, or
deletes source files. DuckDB and DataFusion scan the exact same Parquet parts.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROWS = 20_000
THREADS = 2
SCHEMA_VERSION = 1
BACKENDS = ("duckdb-table", "duckdb-parquet", "datafusion-parquet")


@dataclass(frozen=True)
class ArtifactContract:
    source: str
    source_size: int
    source_mtime_ns: int
    rows: int
    columns: int
    nnz: int
    all_values: float
    zero_based: bool

    @property
    def expected_mean(self) -> float:
        return self.nnz / (self.rows * self.columns)


def physical_memory_bytes() -> int:
    if platform.system() == "Darwin":
        return int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        )
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def default_emergency_rss_bytes(total: int) -> int:
    """Return a high workstation safety stop, not an engine memory budget."""

    return int(total * 0.80)


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


def tree_size(path: Path) -> int:
    """Return a best-effort byte count while transient spill files change."""

    try:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def parquet_files(path: Path) -> tuple[Path, ...]:
    path = path.expanduser().resolve()
    files = (path,) if path.is_file() else tuple(sorted(path.glob("*.parquet")))
    if not files or any(not item.is_file() for item in files):
        raise ValueError(f"no regular Parquet part files found at {path}")
    return files


def _read_manifest(
    path: Path, *, expected_kind: str
) -> tuple[ArtifactContract, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"artifact manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "source",
        "source_size",
        "source_mtime_ns",
        "rows",
        "columns",
        "nnz",
        "all_values",
        "zero_based",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"artifact manifest {path} lacks fields: {missing}")
    if payload.get("schema_version") != 1 or payload.get("kind") != expected_kind:
        raise ValueError(
            f"artifact manifest {path} is not schema-1 {expected_kind!r}"
        )
    contract = ArtifactContract(**{key: payload[key] for key in required})
    if contract.rows <= 0 or contract.columns <= 0 or contract.nnz <= 0:
        raise ValueError(f"artifact manifest {path} has nonpositive dimensions")
    if contract.all_values != 1.0 or contract.zero_based is not True:
        raise ValueError(f"artifact manifest {path} is not canonical all-one COO")
    return contract, payload


def load_artifact_contract(
    duckdb_manifest: Path,
    parquet_manifest: Path,
    *,
    columns: int,
    nnz: int,
) -> ArtifactContract:
    duck_contract, _ = _read_manifest(
        duckdb_manifest, expected_kind="duckdb-persistent-table"
    )
    parquet_contract, parquet_payload = _read_manifest(
        parquet_manifest, expected_kind="datafusion-canonical-parquet"
    )
    if duck_contract != parquet_contract:
        raise ValueError(
            "DuckDB and Parquet manifests do not describe the exact same source"
        )
    if (duck_contract.rows, duck_contract.columns, duck_contract.nnz) != (
        ROWS,
        columns,
        nnz,
    ):
        raise ValueError(
            "artifact contract does not match requested Figure 1g dimensions: "
            f"found {(duck_contract.rows, duck_contract.columns, duck_contract.nnz)}"
        )
    expected_mean = nnz / (ROWS * columns)
    if expected_mean not in {0.01, 0.05}:
        raise ValueError(
            f"requested fixture density {expected_mean} is not Figure 1g sp95/sp99"
        )
    validation = parquet_payload.get("parquet_validation")
    if not isinstance(validation, dict):
        raise ValueError("Parquet manifest lacks full canonical validation")
    expected_validation = {
        "entries": nnz,
        "invalid": 0,
        "min_i": 0,
        "max_i": ROWS - 1,
        "min_j": 0,
        "max_j": columns - 1,
        "min_x": 1.0,
        "max_x": 1.0,
        "sum_x": float(nnz),
    }
    if any(validation.get(key) != value for key, value in expected_validation.items()):
        raise ValueError("Parquet manifest canonical validation does not match request")
    return duck_contract


def _duckdb_parquet_view(connection: Any, paths: tuple[Path, ...]) -> None:
    escaped = [str(path).replace("'", "''") for path in paths]
    argument = (
        f"'{escaped[0]}'"
        if len(escaped) == 1
        else "[" + ",".join(f"'{path}'" for path in escaped) + "]"
    )
    connection.execute(
        "CREATE VIEW figure1g_spill_source AS SELECT i, j, x "
        f"FROM read_parquet({argument})"
    )


def _validate_means(values: Any, contract: ArtifactContract) -> dict[str, Any]:
    import numpy as np

    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size != contract.columns:
        raise ValueError(
            f"column mean length {result.size} != expected {contract.columns}"
        )
    error = float(np.max(np.abs(result - contract.expected_mean)))
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError(
            f"column means differ from {contract.expected_mean}; max error={error}"
        )
    return {
        "output_length": int(result.size),
        "expected_mean": contract.expected_mean,
        "max_abs_error": error,
        "reference_semantics": f"SUM(x) / {contract.rows}; structural zeros included",
    }


def _worker(case: dict[str, Any], connection: Any) -> None:
    owner: Any = None
    spill = Path(case["spill_dir"])
    os.environ["TMPDIR"] = str(spill)
    row = {key: value for key, value in case.items() if key != "contract"}
    try:
        contract = ArtifactContract(**case["contract"])
        backend_name = case["backend"]
        paths = tuple(Path(path) for path in case["parquet_files"])
        if backend_name == "duckdb-table":
            import duckdb

            from dbnumpy import DuckDBBackend

            raw = duckdb.connect(
                case["duckdb_database"],
                read_only=True,
                config={
                    "memory_limit": f'{case["memory_budget_bytes"]}B',
                    "threads": THREADS,
                    "temp_directory": str(spill),
                },
            )
            owner = DuckDBBackend(raw)
            matrix = owner.from_relation(
                case["duckdb_table"],
                shape=(contract.rows, contract.columns),
                storage="sparse",
            )
        elif backend_name == "duckdb-parquet":
            from dbnumpy import DuckDBBackend

            owner = DuckDBBackend.connect(
                memory_limit=f'{case["memory_budget_bytes"]}B',
                threads=THREADS,
                temp_directory=spill,
            )
            _duckdb_parquet_view(owner.connection, paths)
            matrix = owner.from_relation(
                "figure1g_spill_source",
                shape=(contract.rows, contract.columns),
                storage="sparse",
            )
        elif backend_name == "datafusion-parquet":
            from dbnumpy import DataFusionBackend

            owner = DataFusionBackend.connect(
                target_partitions=THREADS,
                memory_limit_bytes=case["memory_budget_bytes"],
            )
            matrix = owner.from_parquet(
                paths,
                shape=(contract.rows, contract.columns),
                name="figure1g_spill_source",
            )
        else:
            raise ValueError(f"unknown backend: {backend_name}")

        connection.send({"event": "query-start"})
        started = time.perf_counter()
        values = matrix.mean(axis=0)
        query_seconds = time.perf_counter() - started
        row.update(
            status="success",
            error=None,
            query_seconds=query_seconds,
            final_spill_bytes=tree_size(spill),
            **_validate_means(values, contract),
        )
    except Exception as exc:
        row.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            final_spill_bytes=tree_size(spill),
        )
    finally:
        if owner is not None:
            try:
                owner.close()
            except Exception:
                pass
        connection.send({"event": "result", "row": row})
        connection.close()


def run_case(
    case: dict[str, Any],
    *,
    timeout_seconds: float,
    emergency_rss_bytes: int,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(case, child))
    process.start()
    assert process.pid is not None
    child.close()
    started = time.monotonic()
    peak_spill = 0
    peak_rss = 0
    result: dict[str, Any] | None = None
    pipe_open = True
    while process.is_alive() or (pipe_open and parent.poll()):
        while pipe_open and parent.poll():
            try:
                message = parent.recv()
            except EOFError:
                pipe_open = False
                break
            if message["event"] == "result":
                result = message["row"]
        peak_spill = max(peak_spill, tree_size(Path(case["spill_dir"])))
        rss = _rss_bytes(process.pid)
        peak_rss = max(peak_rss, rss)
        if rss > emergency_rss_bytes:
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            result = {
                **{key: value for key, value in case.items() if key != "contract"},
                "status": "system-emergency-safety-stop",
                "error": (
                    f"worker RSS {rss} exceeded emergency threshold "
                    f"{emergency_rss_bytes}; this is not an engine memory-limit "
                    "result"
                ),
                "final_spill_bytes": tree_size(Path(case["spill_dir"])),
            }
            break
        if time.monotonic() - started > timeout_seconds:
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            result = {
                **{key: value for key, value in case.items() if key != "contract"},
                "status": "timeout",
                "error": f"worker exceeded {timeout_seconds} seconds",
                "final_spill_bytes": tree_size(Path(case["spill_dir"])),
            }
            break
        time.sleep(0.1)
    process.join()
    if result is None:
        result = {
            **{key: value for key, value in case.items() if key != "contract"},
            "status": "worker-exit",
            "error": f"worker exited with code {process.exitcode} without a result",
            "final_spill_bytes": tree_size(Path(case["spill_dir"])),
        }
    result["peak_spill_bytes"] = max(peak_spill, result["final_spill_bytes"])
    result["peak_rss_bytes"] = peak_rss
    result["emergency_rss_threshold_bytes"] = emergency_rss_bytes
    result["rss_scope"] = "fresh worker process, sampled every 0.1 seconds"
    result["spill_sampling_interval_seconds"] = 0.1
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb-database", type=Path, required=True)
    parser.add_argument("--duckdb-manifest", type=Path, required=True)
    parser.add_argument("--duckdb-table", default="figure1g_matrix")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--parquet-manifest", type=Path, required=True)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--nnz", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-mib", type=int, default=128)
    parser.add_argument(
        "--emergency-rss-gib",
        type=float,
        help="last-resort workstation safety stop; default is 80%% of physical RAM",
    )
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.columns <= 0 or args.nnz <= 0 or args.memory_mib <= 0:
        raise ValueError("columns, nnz, and memory-mib must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if args.emergency_rss_gib is not None and args.emergency_rss_gib <= 0:
        raise ValueError("emergency-rss-gib must be positive")
    database = args.duckdb_database.expanduser().resolve()
    if not database.is_file():
        raise ValueError(f"DuckDB database does not exist: {database}")
    duckdb_manifest = args.duckdb_manifest.expanduser().resolve()
    expected_duckdb_manifest = Path(f"{database}.manifest.json")
    if duckdb_manifest != expected_duckdb_manifest:
        raise ValueError(
            "DuckDB manifest must be the canonical adjacent manifest: "
            f"{expected_duckdb_manifest}"
        )
    parquet_artifact = args.parquet.expanduser().resolve()
    parquet_manifest = args.parquet_manifest.expanduser().resolve()
    expected_parquet_manifest = parquet_artifact.with_suffix(".manifest.json")
    if parquet_manifest != expected_parquet_manifest:
        raise ValueError(
            "Parquet manifest must be the canonical adjacent manifest: "
            f"{expected_parquet_manifest}"
        )
    paths = parquet_files(parquet_artifact)
    contract = load_artifact_contract(
        duckdb_manifest,
        parquet_manifest,
        columns=args.columns,
        nnz=args.nnz,
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    memory_bytes = args.memory_mib * 1024**2
    emergency_rss_bytes = (
        int(args.emergency_rss_gib * 1024**3)
        if args.emergency_rss_gib is not None
        else default_emergency_rss_bytes(physical_memory_bytes())
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "duckdb_database": str(database),
        "duckdb_table": args.duckdb_table,
        "parquet_files": [str(path) for path in paths],
        "contract": asdict(contract),
        "rows": contract.rows,
        "columns": contract.columns,
        "nnz": contract.nnz,
        "memory_budget_bytes": memory_bytes,
        "memory_budget_role": "engine spill budget, not a process RSS cutoff",
        "host_result_limit_values": 50_000_000,
        "host_result_limit_role": (
            "dbnumpy default max_host_values for eager result vectors; "
            "independent of dense-shape and engine-memory budgets"
        ),
        "threads": THREADS,
        "input_coordinates_materialized_in_python": False,
    }
    results = []
    for backend in args.backends:
        case_dir = output / backend
        spill = case_dir / "spill"
        spill.mkdir(parents=True)
        case = {
            **common,
            "backend": backend,
            "spill_dir": str(spill),
        }
        results.append(
            run_case(
                case,
                timeout_seconds=args.timeout_seconds,
                emergency_rss_bytes=emergency_rss_bytes,
            )
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_contract": asdict(contract),
        "parquet_files_shared_by_scan_backends": [str(path) for path in paths],
        "memory_budget_bytes": memory_bytes,
        "emergency_rss_threshold_bytes": emergency_rss_bytes,
        "threads": THREADS,
        "results": results,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in results:
        if row["status"] != "success":
            print(f'{row["backend"]}: {row["error"]}')
    return int(any(row["status"] != "success" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
