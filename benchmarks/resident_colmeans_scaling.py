#!/usr/bin/env python3
"""Benchmark the cost of returning very wide resident ``colMeans`` results.

This intentionally does not read Matrix Market files.  Every case has 20,000
logical rows and one stored value in every output column, and runs in a fresh
spawned child process supervised for elapsed time and RSS by the parent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
import uuid
from html import escape
from importlib import metadata
from pathlib import Path
from typing import Any

ROWS = 20_000
DEFAULT_SIZES = (100_000, 300_000, 1_000_000, 3_000_000)
DEFAULT_BACKENDS = ("numpy", "duckdb", "datafusion")
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
    "rows",
    "cols",
    "nnz",
    "density",
    "setup_sec",
    "ready_sec",
    "query_median_sec",
    "query_p10_sec",
    "query_p90_sec",
    "query_min_sec",
    "query_max_sec",
    "query_samples_json",
    "output_length",
    "output_bytes",
    "peak_rss_bytes",
    "execution_path",
    "worker_tmp_dir",
    "requested_temp_directory",
    "effective_temp_directory",
    "temp_directory_observation",
    "threads",
    "engine_memory_gib",
    "rss_limit_gib",
    "timeout_sec",
    "package_versions_json",
)


def sample_stats(samples: list[float]) -> dict[str, float]:
    """Return median and linearly interpolated percentile summaries."""

    if not samples:
        raise ValueError("at least one timing sample is required")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("timing samples must be finite and nonnegative")
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
    for distribution in ("dbnumpy", "numpy", "duckdb", "datafusion"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _base_case(
    size: int,
    backend: str,
    *,
    repeats: int,
    threads: int,
    engine_memory_gib: float,
    rss_limit_gib: float,
    timeout_sec: float,
    worker_tmp_dir: Path,
) -> dict[str, Any]:
    return {
        "size": size,
        "backend": backend,
        "rows": ROWS,
        "cols": size,
        "nnz": size,
        "density": 1.0 / ROWS,
        "repeats": repeats,
        "threads": threads,
        "engine_memory_gib": engine_memory_gib,
        "rss_limit_gib": rss_limit_gib,
        "timeout_sec": timeout_sec,
        "worker_tmp_dir": str(worker_tmp_dir.resolve()),
        "requested_temp_directory": str(worker_tmp_dir.resolve()),
        "effective_temp_directory": "",
        "temp_directory_observation": "unconfirmed",
        "execution_path": {
            "numpy": "numpy_bincount_resident_coordinates",
            "duckdb": "duckdb_lazy_range_public_from_relation_mean_axis_0",
            "datafusion": (
                "datafusion_lazy_generate_series_registered_dataframe_"
                "public_from_relation_mean_axis_0"
            ),
        }[backend],
    }


def _worker(case: dict[str, Any], connection: Any) -> None:
    """Execute one case; imports and allocations stay in the spawned child."""

    for variable in THREAD_ENV_VARS:
        os.environ[variable] = str(case["threads"])
    os.environ["TMPDIR"] = case["requested_temp_directory"]
    backend_object: Any = None
    try:
        import tempfile

        tempfile.tempdir = case["requested_temp_directory"]
        import numpy as np

        setup_started = time.perf_counter()
        ncols = case["cols"]
        backend_name = case["backend"]
        effective_temp = tempfile.gettempdir()
        temp_observation = "python_tempfile_gettempdir"
        if backend_name == "numpy":
            cols = np.arange(ncols, dtype=np.int64)
            ones = np.ones(ncols, dtype=np.float64)

            def query() -> Any:
                return np.bincount(cols, weights=ones, minlength=ncols) / ROWS

        elif backend_name == "duckdb":
            from dbnumpy.backends import DuckDBBackend

            backend_object = DuckDBBackend.connect(
                threads=case["threads"],
                memory_limit=f"{case['engine_memory_gib']}GiB",
                max_densify_cells=ncols,
            )
            backend_object.connection.execute(
                "SET temp_directory = ?", [case["requested_temp_directory"]]
            )
            effective_temp = backend_object.connection.execute(
                "SELECT current_setting('temp_directory')"
            ).fetchone()[0]
            temp_observation = "duckdb_current_setting_temp_directory"
            backend_object.connection.execute(
                "CREATE TEMP VIEW resident_values AS "
                "SELECT CAST(0 AS BIGINT) AS i, "
                "CAST(range AS BIGINT) AS j, CAST(1.0 AS DOUBLE) AS x "
                f"FROM range({ncols})"
            )
            matrix = backend_object.from_relation(
                "resident_values", shape=(ROWS, ncols), storage="sparse"
            )

            def query() -> Any:
                return matrix.mean(axis=0)

        elif backend_name == "datafusion":
            from dbnumpy.backends import DataFusionBackend

            backend_object = DataFusionBackend.connect(
                target_partitions=case["threads"],
                memory_limit_bytes=int(case["engine_memory_gib"] * 1024**3),
                max_densify_cells=ncols,
            )
            # DataFusion's Python API has no stable queryable spill-path setting.
            # The runtime's OS disk manager observes TMPDIR set before import.
            frame = backend_object.context.sql(
                "SELECT CAST(0 AS BIGINT) AS i, "
                "CAST(value AS BIGINT) AS j, CAST(1.0 AS DOUBLE) AS x "
                f"FROM generate_series(0, {ncols - 1})"
            )
            backend_object.context.register_table("resident_values", frame)
            temp_observation = "python_tempfile_path_for_datafusion_os_disk_manager"
            matrix = backend_object.from_relation(
                "resident_values", shape=(ROWS, ncols), storage="sparse"
            )

            def query() -> Any:
                return matrix.mean(axis=0)

        else:  # guarded by CLI validation
            raise ValueError(f"unsupported backend: {backend_name}")

        setup_sec = time.perf_counter() - setup_started
        query()  # untimed warmup
        samples: list[float] = []
        output: Any = None
        for _ in range(case["repeats"]):
            started = time.perf_counter()
            output = query()
            samples.append(time.perf_counter() - started)
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (ncols,):
            raise AssertionError(f"output shape is {output.shape}, expected {(ncols,)}")
        expected = 1.0 / ROWS
        if not np.all(output == expected):
            maximum_error = float(np.max(np.abs(output - expected)))
            raise AssertionError(f"incorrect column means; max error {maximum_error}")
        connection.send(
            {
                **case,
                "status": "ok",
                "error": "",
                "setup_sec": setup_sec,
                "ready_sec": setup_sec,
                "query_samples": samples,
                **sample_stats(samples),
                "output_length": int(output.size),
                "output_bytes": int(output.nbytes),
                "effective_temp_directory": str(effective_temp),
                "temp_directory_observation": temp_observation,
                "package_versions": _versions(),
            }
        )
    except BaseException as exc:
        connection.send(
            {
                **case,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "package_versions": _versions(),
            }
        )
    finally:
        if backend_object is not None:
            backend_object.close()
        connection.close()


def read_rss_bytes(pid: int) -> int | None:
    """Sample RSS through the common macOS/Linux ``ps`` interface."""

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


def run_case(
    size: int,
    backend: str,
    *,
    repeats: int,
    threads: int,
    engine_memory_gib: float,
    rss_limit_bytes: int,
    rss_limit_gib: float,
    timeout_sec: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Run and supervise one fresh spawned process."""

    worker_tmp = output_dir / "worker-tmp" / (f"{size}-{backend}-{uuid.uuid4().hex}")
    worker_tmp.mkdir(parents=True, exist_ok=False)
    case = _base_case(
        size,
        backend,
        repeats=repeats,
        threads=threads,
        engine_memory_gib=engine_memory_gib,
        rss_limit_gib=rss_limit_gib,
        timeout_sec=timeout_sec,
        worker_tmp_dir=worker_tmp,
    )
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(case, sending))
    process.start()
    sending.close()
    peak_rss = 0
    parent_status: str | None = None
    deadline = time.monotonic() + timeout_sec
    while process.is_alive():
        rss = read_rss_bytes(process.pid) if process.pid is not None else None
        if rss is not None:
            peak_rss = max(peak_rss, rss)
            if rss > rss_limit_bytes:
                parent_status = "memory_limit"
                break
        if time.monotonic() >= deadline:
            parent_status = "timeout"
            break
        process.join(0.1)
    if parent_status is not None:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        row = {
            **case,
            "status": parent_status,
            "error": (
                f"RSS exceeded {rss_limit_bytes} bytes"
                if parent_status == "memory_limit"
                else f"case exceeded {timeout_sec:g} seconds"
            ),
            "effective_temp_directory": "unconfirmed_worker_terminated",
            "package_versions": _versions(),
        }
    else:
        process.join()
        if receiving.poll(0.2):
            row = receiving.recv()
        else:
            row = {
                **case,
                "status": "error",
                "error": (
                    f"worker exited with code {process.exitcode} without a result"
                ),
                "package_versions": _versions(),
            }
    receiving.close()
    row["peak_rss_bytes"] = peak_rss
    return row


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
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


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workload": "resident-wide-colmeans-one-coordinate-per-column",
        "rows": ROWS,
        "sizes": list(args.sizes),
        "backends": list(args.backends),
        "repeats": args.repeats,
        "threads": args.threads,
        "engine_memory_gib": args.engine_memory_gib,
        "rss_limit_gib": args.rss_limit_gib,
        "timeout_sec": args.timeout_sec,
    }


def run_benchmark(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    config = _config(args)
    if json_path.exists() or csv_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"results already exist in {args.output_dir}; pass --resume"
            )
        if not json_path.is_file():
            raise ValueError("resume requires canonical results.json")
        document = json.loads(json_path.read_text(encoding="utf-8"))
        if document.get("config") != config:
            raise ValueError("existing results configuration does not match")
        rows = document.get("results")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("existing results.json has an invalid results list")
        keys = [(row.get("size"), row.get("backend")) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("existing results.json contains duplicate cases")
        _atomic_csv(csv_path, rows)
    else:
        rows = []
        document = {"config": config, "results": rows}
        _atomic_json(json_path, document)
        _atomic_csv(csv_path, rows)
    completed = {(row["size"], row["backend"]) for row in rows}
    for size in args.sizes:
        for backend in args.backends:
            if (size, backend) in completed:
                continue
            print(f"running result_columns={size} backend={backend}", flush=True)
            row = run_case(
                size,
                backend,
                repeats=args.repeats,
                threads=args.threads,
                engine_memory_gib=args.engine_memory_gib,
                rss_limit_bytes=int(args.rss_limit_gib * 1024**3),
                rss_limit_gib=args.rss_limit_gib,
                timeout_sec=args.timeout_sec,
                output_dir=args.output_dir,
            )
            rows.append(row)
            _atomic_json(json_path, document)
            _atomic_csv(csv_path, rows)
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


def _number(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def write_svg(rows: list[dict[str, Any]], output: Path) -> None:
    """Write a dependency-free accessible log/log result-width chart."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    usable_x = [value for row in rows if (value := _number(row, "cols")) is not None]
    successes = [
        row
        for row in rows
        if row.get("status") == "ok"
        and _number(row, "cols") is not None
        and _number(row, "query_median_sec") is not None
    ]
    if not usable_x:
        raise ValueError("plot needs at least one row with positive result columns")
    if not successes:
        raise ValueError("plot needs at least one successful positive timing")
    x_min, x_max = min(usable_x), max(usable_x)
    y_values = [
        value
        for row in successes
        for key in ("query_p10_sec", "query_median_sec", "query_p90_sec")
        if (value := _number(row, key)) is not None
    ]
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min /= 1.5
        x_max *= 1.5
    if y_min == y_max:
        y_min /= 1.5
        y_max *= 1.5
    left, top, width, height = 92.0, 112.0, 800.0, 400.0
    bottom = top + height

    def log_scale(value: float, low: float, high: float, span: float) -> float:
        return math.log10(value / low) / math.log10(high / low) * span

    colors = {"numpy": "#2468a2", "duckdb": "#c43c4e", "datafusion": "#087f5b"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" '
        'viewBox="0 0 1000 620" role="img" aria-labelledby="title desc">',
        '<title id="title">Resident wide colMeans result-return scaling</title>',
        '<desc id="desc">Log scale plot of warm query seconds against output '
        "vector length, with percentile whiskers and failure markers.</desc>",
        '<rect width="1000" height="620" fill="white"/>',
        '<text x="500" y="34" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Resident wide colMeans scaling</text>',
        '<text x="500" y="61" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">20,000 logical rows; one NNZ per output column. '
        "Width is returned result length, not Figure 1 MTX density.</text>",
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        'fill="#fafafa" stroke="#666"/>',
    ]
    for fraction in (0.0, 0.5, 1.0):
        x = left + fraction * width
        y = bottom - fraction * height
        x_value = 10 ** (math.log10(x_min) + fraction * math.log10(x_max / x_min))
        y_value = 10 ** (math.log10(y_min) + fraction * math.log10(y_max / y_min))
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{bottom}" stroke="#ddd"/>',
                f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{x_value:.2g}</text>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" '
                f'y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11">{y_value:.2g}</text>',
            ]
        )
    for backend, color in colors.items():
        series = sorted(
            (row for row in successes if row.get("backend") == backend),
            key=lambda row: float(row["cols"]),
        )
        points: list[tuple[float, float, dict[str, Any]]] = []
        for row in series:
            x = left + log_scale(float(row["cols"]), x_min, x_max, width)
            median = float(row["query_median_sec"])
            y = bottom - log_scale(median, y_min, y_max, height)
            points.append((x, y, row))
        if points:
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            parts.append(
                f'<polyline points="{coordinates}" fill="none" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
        for x, y, row in points:
            p10 = _number(row, "query_p10_sec")
            p90 = _number(row, "query_p90_sec")
            if p10 is not None and p90 is not None:
                high = bottom - log_scale(p90, y_min, y_max, height)
                low = bottom - log_scale(p10, y_min, y_max, height)
                parts.append(
                    f'<path d="M{x:.1f} {high:.1f}V{low:.1f} '
                    f"M{x - 4:.1f} {high:.1f}H{x + 4:.1f} "
                    f'M{x - 4:.1f} {low:.1f}H{x + 4:.1f}" '
                    f'stroke="{color}" fill="none"/>'
                )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
                f"<title>{backend} {row['cols']} columns</title></circle>"
            )
        failures = [
            row
            for row in rows
            if row.get("backend") == backend
            and row.get("status") != "ok"
            and _number(row, "cols") is not None
        ]
        for row in failures:
            x = left + log_scale(float(row["cols"]), x_min, x_max, width)
            status = escape(str(row.get("status") or "error"))
            y = top + 18
            parts.append(
                f'<path d="M{x - 6:.1f} {y - 6}L{x + 6:.1f} {y + 6}'
                f'M{x + 6:.1f} {y - 6}L{x - 6:.1f} {y + 6}" '
                f'stroke="{color}" stroke-width="2.5">'
                f"<title>{backend} {status}</title></path>"
            )
    parts.extend(
        [
            '<text x="492" y="570" text-anchor="middle" '
            'font-family="sans-serif" font-size="13">Result columns / output '
            "vector length (log scale)</text>",
            '<text x="25" y="312" text-anchor="middle" '
            'transform="rotate(-90 25 312)" font-family="sans-serif" '
            'font-size="13">Warm matrix.mean(axis=0) seconds (log scale)</text>',
            '<text x="92" y="91" font-family="sans-serif" font-size="11">'
            "Whiskers: p10–p90; × marks timeout, memory limit, or error</text>",
        ]
    )
    for index, (backend, color) in enumerate(colors.items()):
        x = 260 + index * 180
        parts.append(
            f'<line x1="{x}" y1="600" x2="{x + 25}" y2="600" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{x + 33}" y="605" font-family="sans-serif" '
            f'font-size="13">{backend}</text>'
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text("\n".join(parts) + "\n", encoding="utf-8")
    temporary.replace(output)


def plot_results(args: argparse.Namespace) -> int:
    document = json.loads(args.input.read_text(encoding="utf-8"))
    rows = document.get("results") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("input JSON must contain a results list")
    write_svg(rows, args.output)
    return 0


def _csv_list(value: str, *, choices: tuple[str, ...] | None = None) -> tuple[Any, ...]:
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if not fields:
        raise argparse.ArgumentTypeError("value must be a non-empty CSV list")
    if len(fields) != len(set(fields)):
        raise argparse.ArgumentTypeError("values must not be repeated")
    if choices is not None:
        invalid = sorted(set(fields) - set(choices))
        if invalid:
            raise argparse.ArgumentTypeError(
                f"unsupported values: {', '.join(invalid)}"
            )
        return fields
    try:
        parsed = tuple(int(field) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be positive integers") from exc
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return parsed


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
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run isolated resident cases")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--sizes", type=_csv_list, default=DEFAULT_SIZES)
    run.add_argument(
        "--backends",
        type=lambda value: _csv_list(value, choices=DEFAULT_BACKENDS),
        default=DEFAULT_BACKENDS,
    )
    run.add_argument("--repeats", type=repeat_count, default=5)
    run.add_argument("--threads", type=positive_int, default=2)
    run.add_argument("--engine-memory-gib", type=positive_float, default=1.0)
    run.add_argument("--rss-limit-gib", type=positive_float, default=5.0)
    run.add_argument("--timeout-sec", type=positive_float, default=300.0)
    run.add_argument("--resume", action="store_true")
    plot = commands.add_parser("plot", help="plot a results JSON as SVG")
    plot.add_argument("--input", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_benchmark(args)
    return plot_results(args)


if __name__ == "__main__":
    raise SystemExit(main())
