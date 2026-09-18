"""Run the shared PBMC workflow under Linux cgroup v2 memory limits.

Run the supervisor as root inside WSL/Linux; workers drop to --uid/--gid.
Only fresh, dedicated cgroups are changed. See the memory-pressure guide.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.pbmc_preprocessing import (  # noqa: E402
    ENGINES,
    THREAD_VARS,
    sha256,
    write_json,
)

for variable in THREAD_VARS:
    os.environ[variable] = "1"


def mark(folder, phase):
    write_json(folder / "phase.json", {"phase": phase})


def worker(args):
    folder = args.output
    mark(folder, "imports")
    import numpy as np
    from scipy import sparse

    import dbnumpy as dnp
    from examples.pbmc_workflow import preprocess

    if args.engine != "scipy":
        __import__(args.engine)
    mark(folder, "input_read")
    x = sparse.csr_array(sparse.load_npz(args.data / f"input-{args.size}.npz"))
    mark(folder, "setup")
    started = time.perf_counter()
    backend = None
    if args.engine == "scipy":
        values = x.copy()
    elif args.engine == "duckdb":
        backend = dnp.DuckDBBackend.connect(
            threads=1,
            memory_limit=f"{args.engine_bytes}B",
            max_sparse_host_values=100_000_000,
        )
        values = backend.from_scipy(x)
    else:
        backend = dnp.DataFusionBackend.connect(
            target_partitions=1,
            memory_limit_bytes=args.engine_bytes,
            max_sparse_host_values=100_000_000,
        )
        values = backend.from_scipy(x)
    ingested = time.perf_counter()
    mark(folder, "workflow")
    result = preprocess(values)
    calculated = time.perf_counter()
    mark(folder, "sparse_collection")
    output = sparse.csr_array(
        result.values if backend is None else result.values.to_scipy()
    )
    collected = time.perf_counter()
    write_json(
        folder / "timing.json",
        dict(
            total_seconds=collected - started,
            setup_seconds=ingested - started,
            workflow_seconds=calculated - ingested,
            collect_seconds=collected - calculated,
            retained_cells=len(result.cells),
            retained_genes=len(result.genes),
            output_nnz=output.nnz,
        ),
    )
    mark(folder, "cleanup")
    if backend is not None:
        backend.close()
    result.values = None
    del x, values, backend
    mark(folder, "save_for_validation")
    sparse.save_npz(folder / "output.npz", output, compressed=False)
    np.savez(
        folder / "summary.npz",
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
    mark(folder, "finished")


def verify(args):
    import numpy as np
    from scipy import sparse

    from benchmarks.pbmc_preprocessing import compare

    error = compare(
        sparse.load_npz(args.output / "output.npz"),
        sparse.load_npz(args.data / f"reference-{args.size}.npz"),
    )
    checks = {"max_abs_error": error}
    with (
        np.load(args.output / "summary.npz") as result,
        np.load(args.data / f"summary-{args.size}.npz") as reference,
        np.load(args.data / f"independent-summary-{args.size}.npz") as independent,
    ):
        for name in ("cells", "genes", "total_counts", "detected_genes"):
            np.testing.assert_array_equal(result[name], reference[name])
        for name in ("gene_mean", "gene_variance"):
            np.testing.assert_allclose(
                result[name], independent[name], rtol=1e-9, atol=1e-11
            )
            checks[f"max_{name}_abs_error"] = float(
                np.max(np.abs(result[name] - independent[name]), initial=0)
            )
    write_json(args.output / "verification.json", checks)


def read_counters(path):
    return {
        key: int(value)
        for key, value in (line.split() for line in path.read_text().splitlines())
    }


def run(args):
    if os.geteuid() != 0:
        raise SystemExit(
            "The cgroup supervisor must run as root; workers drop privileges."
        )
    if args.uid == 0 or args.gid == 0:
        raise SystemExit("Select a non-root worker uid and gid.")
    parent = Path("/sys/fs/cgroup")
    if "memory" not in (parent / "cgroup.subtree_control").read_text().split():
        raise SystemExit("The root cgroup must already have memory control enabled.")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        method="cgroup v2 memory.max; memory.swap.max=0; serial fresh workers",
        scope="in-memory input, full sparse output; validation in a separate process",
        timing=(
            "setup + shared workflow + sparse collection; "
            "excludes imports/read/checks/save"
        ),
        versions={
            name: version(name)
            for name in ("numpy", "scipy", "duckdb", "datafusion", "pyarrow")
        },
        limits_gib=args.limits,
        engine_budget_fraction=0.5,
        timeout_seconds=args.timeout,
        sizes=args.sizes,
        repeats=args.repeats,
        workflow_sha256=sha256(ROOT / "examples/pbmc_workflow.py"),
        runner_sha256=sha256(Path(__file__)),
        reference_runner_sha256=sha256(ROOT / "benchmarks/pbmc_preprocessing.py"),
        package_source_sha256={
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "src/dbnumpy").rglob("*.py"))
        },
        dataset=json.loads((args.data / "dataset.json").read_text()),
        scanpy_validation=json.loads(
            (args.data / "scanpy-validation.json").read_text()
        ),
        trials=[],
    )
    for ceiling in args.limits:
        cap = int(ceiling * 1024**3)
        for size in args.sizes:
            for repeat in range(args.repeats):
                order = ENGINES[repeat % 3 :] + ENGINES[: repeat % 3]
                for engine in order:
                    name = f"{ceiling:g}GiB-{size}-{repeat}-{engine}"
                    folder = args.output / name
                    folder.mkdir()
                    os.chown(folder, args.uid, args.gid)
                    group = parent / f"dbnumpy-benchmark-{uuid4().hex}"
                    group.mkdir()
                    process = None
                    try:
                        (group / "memory.max").write_text(str(cap))
                        (group / "memory.swap.max").write_text("0")
                        (group / "memory.oom.group").write_text("1")
                        assert int((group / "memory.max").read_text()) == cap
                        assert int((group / "memory.swap.max").read_text()) == 0

                        def enter_worker_group(group=group):
                            (group / "cgroup.procs").write_text(str(os.getpid()))
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
                        ]
                        print(f"Starting {name}", flush=True)
                        timeout = False
                        with (folder / "worker.log").open("w") as stream:
                            process = subprocess.Popen(
                                command,
                                stdout=stream,
                                stderr=stream,
                                preexec_fn=enter_worker_group,
                            )
                            try:
                                process.wait(timeout=args.timeout)
                            except subprocess.TimeoutExpired:
                                timeout = True
                                (group / "cgroup.kill").write_text("1")
                                process.wait()
                        events = read_counters(group / "memory.events")
                        row = dict(
                            engine=engine,
                            cells=size,
                            repeat=repeat,
                            limit_gib=ceiling,
                            engine_budget_bytes=cap // 2,
                            memory_peak_bytes=int((group / "memory.peak").read_text()),
                            memory_events=events,
                            returncode=process.returncode,
                            last_phase=json.loads((folder / "phase.json").read_text())[
                                "phase"
                            ]
                            if (folder / "phase.json").exists()
                            else "startup",
                            status="pending",
                        )
                    finally:
                        if process is not None and process.poll() is None:
                            (group / "cgroup.kill").write_text("1")
                            process.wait()
                        group.rmdir()
                    if timeout:
                        row["status"] = "timeout"
                    elif events["oom_kill"]:
                        row["status"] = "memory_limit"
                    elif process.returncode:
                        row["status"] = "error"
                    else:
                        check = subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "verify",
                                "--data",
                                str(args.data.resolve()),
                                "--size",
                                str(size),
                                "--output",
                                str(folder.resolve()),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                        )
                        (folder / "verification.log").write_text(
                            check.stdout + check.stderr
                        )
                        if check.returncode:
                            row["status"] = "verification_failed"
                            row["error"] = (check.stdout + check.stderr)[-3000:]
                        else:
                            row["status"] = "passed"
                            row.update(json.loads((folder / "timing.json").read_text()))
                            row.update(
                                json.loads((folder / "verification.json").read_text())
                            )
                    if row["status"] != "passed" and "error" not in row:
                        row["error"] = (folder / "worker.log").read_text()[-3000:]
                    metadata["trials"].append(row)
                    write_json(args.output / "results.json", metadata)
                    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--limits", type=float, nargs="+", default=[1, 2])
            command.add_argument(
                "--sizes", type=int, nargs="+", default=[10_000, 30_000, 68_579]
            )
            command.add_argument("--repeats", type=int, default=1)
            command.add_argument("--timeout", type=int, default=300)
            command.add_argument("--uid", type=int, default=1000)
            command.add_argument("--gid", type=int, default=1000)
        else:
            command.add_argument("--size", type=int, required=True)
            if name == "worker":
                command.add_argument("--engine", choices=ENGINES, required=True)
                command.add_argument("--engine-bytes", type=int, required=True)
    args = parser.parse_args()
    {"run": run, "worker": worker, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
