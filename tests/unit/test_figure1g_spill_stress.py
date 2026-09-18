from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "figure1g_spill_stress.py"
SPEC = importlib.util.spec_from_file_location("figure1g_spill_stress", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _manifest_payload(kind: str, *, columns: int, nnz: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": kind,
        "source": "/canonical/matrix.mtx.gz",
        "source_size": 123,
        "source_mtime_ns": 456,
        "rows": 20_000,
        "columns": columns,
        "nnz": nnz,
        "all_values": 1.0,
        "zero_based": True,
    }


def test_contract_requires_identical_canonical_artifacts(tmp_path: Path) -> None:
    duck = tmp_path / "duck.json"
    parquet = tmp_path / "parquet.json"
    duck.write_text(
        json.dumps(
            _manifest_payload("duckdb-persistent-table", columns=2, nnz=2_000)
        )
    )
    parquet_payload = _manifest_payload(
        "datafusion-canonical-parquet", columns=2, nnz=2_000
    )
    parquet_payload["parquet_validation"] = {
        "entries": 2_000,
        "invalid": 0,
        "min_i": 0,
        "max_i": 19_999,
        "min_j": 0,
        "max_j": 1,
        "min_x": 1.0,
        "max_x": 1.0,
        "sum_x": 2_000.0,
    }
    parquet.write_text(json.dumps(parquet_payload))
    contract = benchmark.load_artifact_contract(
        duck, parquet, columns=2, nnz=2_000
    )
    assert contract.expected_mean == 0.05

    parquet_payload["source_size"] = 124
    parquet.write_text(json.dumps(parquet_payload))
    with pytest.raises(ValueError, match="exact same source"):
        benchmark.load_artifact_contract(duck, parquet, columns=2, nnz=2_000)


@pytest.mark.backend
def test_cli_scans_tiny_artifacts_with_all_backends(tmp_path: Path) -> None:
    import duckdb

    columns, nnz = 2, 2_000
    rows = list(range(999)) + [19_999]
    i = rows + rows
    j = [0] * 1_000 + [1] * 1_000
    x = [1.0] * nnz
    parquet_dir = tmp_path / "canonical.parquet"
    parquet_dir.mkdir()
    pq.write_table(pa.table({"i": i, "j": j, "x": x}), parquet_dir / "part.parquet")

    database = tmp_path / "matrix.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(
        "CREATE TABLE figure1g_matrix AS SELECT * FROM read_parquet(?)",
        [str(parquet_dir / "part.parquet")],
    )
    connection.close()

    duck_manifest = tmp_path / "matrix.duckdb.manifest.json"
    parquet_manifest = tmp_path / "canonical.manifest.json"
    duck_manifest.write_text(
        json.dumps(
            _manifest_payload(
                "duckdb-persistent-table", columns=columns, nnz=nnz
            )
        )
    )
    parquet_payload = _manifest_payload(
        "datafusion-canonical-parquet", columns=columns, nnz=nnz
    )
    parquet_payload["parquet_validation"] = {
        "entries": nnz,
        "invalid": 0,
        "min_i": 0,
        "max_i": 19_999,
        "min_j": 0,
        "max_j": 1,
        "min_x": 1.0,
        "max_x": 1.0,
        "sum_x": float(nnz),
    }
    parquet_manifest.write_text(json.dumps(parquet_payload))

    output = tmp_path / "results"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--duckdb-database",
            str(database),
            "--duckdb-manifest",
            str(duck_manifest),
            "--parquet",
            str(parquet_dir),
            "--parquet-manifest",
            str(parquet_manifest),
            "--columns",
            str(columns),
            "--nnz",
            str(nnz),
            "--output-dir",
            str(output),
            "--memory-mib",
            "64",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads((output / "results.json").read_text())
    assert [row["status"] for row in payload["results"]] == [
        "success",
        "success",
        "success",
    ]
    assert all(row["max_abs_error"] == 0 for row in payload["results"])
    assert all(row["threads"] == 2 for row in payload["results"])
    assert all("peak_spill_bytes" in row for row in payload["results"])
    assert all("peak_rss_bytes" in row for row in payload["results"])
    assert all(
        row["emergency_rss_threshold_bytes"]
        == payload["emergency_rss_threshold_bytes"]
        for row in payload["results"]
    )
    assert payload["results"][0]["expected_mean"] == 0.05
    assert (
        payload["results"][1]["parquet_files"]
        == payload["results"][2]["parquet_files"]
    )
