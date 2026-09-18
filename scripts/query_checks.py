"""Run bounded Linux query checks in fresh processes on both engines.

Checks numerical agreement, stable SQL for fixed inputs, and deep/shared plans.
The limits catch large regressions; they are not performance guarantees.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def worker(engine: str, partitions: int) -> dict:
    import numpy as np
    from scipy import sparse

    import dbnumpy as dnp

    backend = (
        dnp.DuckDBBackend.connect(threads=partitions, memory_limit="512MB")
        if engine == "duckdb"
        else dnp.DataFusionBackend.connect(
            target_partitions=partitions, memory_limit_bytes=512 * 1024**2
        )
    )
    queries = {}
    timings = {}

    def check(label, expression, expected):
        start = time.perf_counter()
        sql = expression.compile()
        queries[label] = hashlib.sha256(sql.encode()).hexdigest()
        # Explicit SQL also exercises the alternative to DataFusion's native path.
        table = backend._execute_sql(sql)
        sql_values = np.zeros(expected.shape)
        sql_values[table["i"].to_numpy(), table["j"].to_numpy()] = table["x"].to_numpy()
        for actual in (sql_values, expression.to_numpy(), expression.to_numpy()):
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            expression.sum(axis=0), expected.sum(axis=0), rtol=1e-11, atol=1e-12
        )
        timings[label] = round(time.perf_counter() - start, 3)

    try:
        values = np.arange(64.0).reshape(8, 8)
        matrix = backend.from_numpy(values, name="query_input")
        gathered = matrix
        expected = values
        for _ in range(100):
            indices = [1, 0, 2, 3, 4, 5, 6, 7]
            gathered = gathered[indices, :]
            expected = expected[indices, :]
        check("100_gathers", gathered, expected)

        # Shared inputs must stay shared when guards refer to values repeatedly.
        expression, expected = matrix / 100, values / 100
        for _ in range(50):
            expression = np.log1p(expression + expression) / 3
            expected = np.log1p(expected + expected) / 3
        check("50_shared_guarded_steps", expression, expected)

        rng = np.random.default_rng(20260915)
        values = rng.normal(size=(512, 32))
        values[rng.random(values.shape) < 0.9] = 0
        matrix = backend.from_scipy(sparse.csr_array(values), name="sparse_input")
        check("sparse_workload", np.expm1(matrix) * 0.5, np.expm1(values) * 0.5)
        np.testing.assert_allclose(matrix.var(axis=0), values.var(axis=0), rtol=1e-11)
        return {"queries": queries, "seconds": timings}
    finally:
        backend.close()


def bounded_worker(engine: str, partitions: int, seed: int) -> dict:
    start = time.monotonic()
    with subprocess.Popen(
        [sys.executable, __file__, "--worker", engine, str(partitions)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
    ) as process:
        peak_kib = 0
        try:
            while True:
                status = Path(f"/proc/{process.pid}/status")
                if status.exists():
                    for line in status.read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            peak_kib = max(peak_kib, int(line.split()[1]))
                if peak_kib > 1536 * 1024:
                    raise RuntimeError(f"{engine} exceeded the 1.5 GiB process limit")
                if time.monotonic() - start > 60:
                    raise RuntimeError(f"{engine} exceeded the 60 second time limit")
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"{engine} query checks failed:\n{stderr}")
    return {**json.loads(stdout), "peak_rss_mib": round(peak_kib / 1024, 1)}


def main() -> None:
    if sys.argv[1:2] == ["--worker"]:
        print(json.dumps(worker(sys.argv[2], int(sys.argv[3]))))
        return
    if not sys.platform.startswith("linux"):
        raise SystemExit("Run these process-memory checks on Linux (including WSL 2).")
    for engine in ("duckdb", "datafusion"):
        reference = None
        for partitions, seed in ((1, 0), (1, 1), (2, 2)):
            result = bounded_worker(engine, partitions, seed)
            if reference is not None and result["queries"] != reference:
                raise AssertionError(f"{engine}: SQL changed across fresh processes")
            reference = result["queries"]
            print(json.dumps({"engine": engine, "partitions": partitions, **result}))


if __name__ == "__main__":
    main()
