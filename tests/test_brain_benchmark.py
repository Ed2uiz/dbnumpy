"""Check benchmark preparation and its independent reference on a small fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse


def test_brain_file_inputs_and_independent_reference(tmp_path):
    if any(find_spec(name) is None for name in ("h5py", "scanpy")):
        pytest.skip("Optional benchmark dependencies are not installed")
    # Start a new interpreter, as the actual runner does. Changing Numba's
    # thread limit after a pytest plugin imported it is not supported.
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "NUMBA_NUM_THREADS": "1"},
        check=True,
    )


def _check_fixture(tmp_path):
    import h5py

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks import brain_preprocessing as benchmark

    rng = np.random.default_rng(714)
    counts = rng.poisson(2, size=(80, 320)).astype(float)
    counts[0] = 0
    counts[:, 0] = 0
    counts[1, 0] = 5  # A rare gene, removed by the global filter.
    counts[:, 1] = 2  # A constant input column.
    original = sparse.csr_array(counts)
    # Cell Ranger HDF5 rows need not be sorted; preserve each index/value pair.
    for row in range(original.shape[0]):
        left, right = original.indptr[row : row + 2]
        original.indices[left:right] = original.indices[left:right][::-1].copy()
        original.data[left:right] = original.data[left:right][::-1].copy()
    archive = tmp_path / "counts.h5"
    with h5py.File(archive, "w") as handle:
        group = handle.create_group("mm10")
        for key, value in {
            "shape": original.shape[::-1],
            "data": original.data,
            "indices": original.indices,
            "indptr": original.indptr,
        }.items():
            group.create_dataset(key, data=value)
    data = tmp_path / "prepared"
    args = SimpleNamespace(archive=archive, data=data, sizes=[20, 50], seed=19, block=7)
    benchmark.prepare(args)
    selected = np.load(data / "original_cells.npy")
    for size in args.sizes:
        np.testing.assert_array_equal(
            benchmark.mapped_input(data, size).toarray(), counts[selected[:size]]
        )
        current = SimpleNamespace(data=data, size=size, block=9)
        benchmark.reference(current)
        # A dense calculation independently checks selection, zeros and variance.
        dense = counts[selected[:size]]
        cells = np.flatnonzero(np.count_nonzero(dense, axis=1) >= 200)
        genes = np.flatnonzero(np.count_nonzero(dense[cells], axis=0) >= 3)
        filtered = dense[np.ix_(cells, genes)]
        keep = filtered.sum(axis=1) > 0
        cells, filtered = cells[keep], filtered[keep]
        expected = np.log1p(filtered / filtered.sum(axis=1, keepdims=True) * 10_000)
        with np.load(data / f"reference-{size}.npz") as summary:
            np.testing.assert_array_equal(summary["cells"], cells)
            np.testing.assert_array_equal(summary["genes"], genes)
            np.testing.assert_allclose(summary["gene_mean"], expected.mean(axis=0))
            np.testing.assert_allclose(
                summary["gene_variance"], expected.var(axis=0, ddof=1), atol=1e-12
            )
        for engine in benchmark.ENGINES:
            output = tmp_path / f"{size}-{engine}"
            output.mkdir()
            trial = SimpleNamespace(
                data=data,
                output=output,
                size=size,
                engine=engine,
                engine_bytes=512 * 1024**2,
                block=11,
            )
            benchmark.worker(trial)
            benchmark.verify(trial)
            checked = json.loads((output / "verification.json").read_text())
            assert checked["scope"] == "full_output_and_summaries"
            # The verifier must reject a real output error.
            path = output / "output-data.npy"
            changed = np.load(path)
            changed[0] += 1.0
            np.save(path, changed)
            with pytest.raises(AssertionError):
                benchmark.verify(trial)
            # Native materialization must preserve every fixture value. The
            # production sample covers all rows here (fewer than 64 retained).
            native_output = tmp_path / f"{size}-{engine}-native"
            native_output.mkdir()
            trial.output = native_output
            trial.endpoint = "native"
            benchmark.worker(trial)
            benchmark.verify(trial)
            checked = json.loads((native_output / "verification.json").read_text())
            timing = json.loads((native_output / "timing.json").read_text())
            assert checked["scope"] == "native_summaries_count_and_sample"
            assert checked["sample_rows"] == len(cells)
            assert timing["native_compute_completed"]
            assert timing["total_seconds"] == pytest.approx(
                timing["input_open_seconds"]
                + timing["workflow_seconds"]
                + timing["native_compute_seconds"]
            )
            assert (timing["native_compute_seconds"] == 0) == (engine == "scipy")
            path = native_output / "native-checks.npz"
            with np.load(path) as saved:
                changed = dict(saved)
            changed["data"][0] += 1.0
            np.savez(path, **changed)
            with pytest.raises(AssertionError):
                benchmark.verify(trial)


if __name__ == "__main__":
    _check_fixture(Path(sys.argv[1]))
