from __future__ import annotations

import tracemalloc
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
from scipy import sparse

from dbnumpy import (
    DataFusionBackend,
    DensificationError,
    DuckDBBackend,
    StorageKind,
    UnsupportedOperationError,
)
from dbnumpy.backends import Backend
from dbnumpy.ir import BinaryOp, ScalarBinary


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_numpy_ingestion_owns_array_and_operand_values(
    backend: Backend, layout: str
) -> None:
    original = np.arange(16.0).reshape(4, 4)
    values = {
        "contiguous": original,
        "transpose": original.T,
        "strided": original[::2, ::2],
    }[layout]
    expected = values.copy()
    matrix = backend.from_numpy(values)
    expression = matrix + values
    original[:] = 999
    np.testing.assert_array_equal(matrix.to_numpy(), expected)
    np.testing.assert_array_equal(expression.to_numpy(), expected * 2)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("position", [2**54 + 2, 2**63 - 2])
def test_int64_slice_coordinates_are_exact(
    backend: Backend, axis: int, reverse: bool, position: int
) -> None:
    shape = (position + 1, 1) if axis == 0 else (1, position + 1)
    row, col = (position, 0) if axis == 0 else (0, position)
    matrix = backend.from_coo([row], [col], [7.0], shape=shape)
    selector = slice(None, None, -2 if reverse else 2)
    result = matrix[selector, :] if axis == 0 else matrix[:, selector]
    expected_index = 0 if reverse else position // 2
    item = result[expected_index, 0] if axis == 0 else result[0, expected_index]
    assert item.item() == 7
    # A second relational operation must not reintroduce lossy coordinates.
    assert result.T.sum() == 7


@pytest.mark.parametrize("step", [10**100, -(10**100)])
def test_extreme_slice_step_is_valid(backend: Backend, step: int) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    np.testing.assert_array_equal(
        matrix[::step, ::step].to_numpy(), values[::step, ::step]
    )


@pytest.mark.parametrize("factor_shape", [(1, 11), (11, 1), (1, 1)])
@pytest.mark.parametrize("reverse", [False, True])
def test_finite_scaling_preserves_sparse_plan_and_values(
    backend: Backend, factor_shape: tuple[int, int], reverse: bool
) -> None:
    backend.max_densify_cells = 100
    matrix = backend.from_coo([0, 10], [1, 10], [2.0, 4.0], shape=(11, 11))
    factors = np.arange(np.prod(factor_shape), dtype=float).reshape(factor_shape) + 2
    expression = factors * matrix if reverse else matrix * factors
    assert expression.storage is StorageKind.SPARSE
    sql = expression.compile()
    assert "dbm_dim" not in sql
    assert backend._execute_sql(sql).num_rows == 2
    expected = np.zeros((11, 11))
    expected[0, 1], expected[10, 10] = 2, 4
    np.testing.assert_array_equal(expression.to_scipy().toarray(), expected * factors)


def test_lazy_finite_scalar_scaling_and_composition(backend: Backend) -> None:
    backend.max_densify_cells = 100
    matrix = backend.from_coo([0, 10], [0, 10], [2.0, 4.0], shape=(11, 11))
    scalar = matrix[0, 0]
    assert (matrix * scalar).sum() == 12
    assert (scalar * matrix).T.sum() == 12
    assert (matrix[:, 0] * scalar).sum() == 4
    assert (matrix * matrix[1, 1]).sum() == 0
    scaled = matrix * backend.from_numpy(np.ones((1, 11)))[0, :]
    assert (scaled.T * 3).sum() == 18


@pytest.mark.parametrize("factor", [np.inf, -np.inf, np.nan])
def test_nonfinite_scaling_retains_guarded_ieee_semantics(
    backend: Backend, factor: float
) -> None:
    values = np.array([[0.0, 2.0], [3.0, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    factors = np.array([factor, 2.0])
    with np.errstate(all="ignore"):
        expected = values * factors
    np.testing.assert_allclose((matrix * factors).to_numpy(), expected, equal_nan=True)
    backend.max_densify_cells = 3
    with pytest.raises(DensificationError):
        matrix * factors


def test_arithmetic_does_not_inherit_finite_proof(backend: Backend) -> None:
    backend.max_densify_cells = 100
    matrix = backend.from_coo([0], [0], [2.0], shape=(11, 11))
    factors = backend.from_numpy(np.full((1, 11), 1e308)) * 2
    with pytest.raises(DensificationError):
        matrix * factors


def test_sparse_export_limit_bounds_arrow_result(backend: Backend) -> None:
    matrix = backend.from_coo([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0], shape=(4, 4))
    backend.max_sparse_host_values = 2
    with patch.object(backend, "_execute_sql", wraps=backend._execute_sql) as execute:
        with pytest.raises(DensificationError, match="max_sparse_host_values=2"):
            matrix.to_scipy()
        assert "LIMIT 3" in execute.call_args.args[0]
    backend.max_sparse_host_values = 3
    assert matrix.to_scipy().nnz == 3


@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
def test_sparse_export_is_writable_and_does_not_alias_source(
    backend: Backend, format: str
) -> None:
    expected = np.diag([2.0, 3.0, 4.0])
    matrix = backend.from_scipy(sparse.csr_array(expected))
    exported = matrix.to_scipy(format=format)
    exported.data[:] = -10.0
    # A zero-copy shortcut must not expose the database's registered buffers.
    np.testing.assert_array_equal(matrix.to_numpy(), expected)
    np.testing.assert_array_equal(exported.toarray(), np.diag([-10.0] * 3))


@pytest.mark.parametrize("drop_zeros", [False, True])
def test_sparse_export_handles_chunk_boundaries_and_nonfinite_values(
    backend: Backend, drop_zeros: bool
) -> None:
    matrix = backend.from_coo([0], [0], [1.0], shape=(2, 3))
    values = [2.0, np.nan, np.inf, -np.inf, 0.0 if drop_zeros else 5.0]
    table = pa.table(
        {
            "i": pa.chunked_array([[0], [], [1, 0, 1, 1]], type=pa.int32()),
            "j": pa.chunked_array([[0, 1, 2], [0, 2]], type=pa.int64()),
            "x": pa.chunked_array([values[:2], [], values[2:]], type=pa.float64()),
        }
    )
    with patch.object(backend, "_execute_sql", return_value=table):
        result = matrix.to_scipy(format="coo")
    expected = np.array([[2.0, 0.0, np.inf], [-np.inf, np.nan, values[-1]]])
    np.testing.assert_array_equal(result.toarray(), expected)
    assert result.nnz == (4 if drop_zeros else 5)
    result.data[:] = 7
    np.testing.assert_array_equal(table.column("x").to_numpy(), values)


def test_sparse_export_bounds_temporary_array_memory(
    backend: Backend,
) -> None:
    size = 262_144
    matrix = backend.from_coo(
        np.arange(size),
        np.zeros(size, dtype=np.int64),
        np.ones(size),
        shape=(size, 1),
    )
    backend.compile(matrix._expr)
    table = pa.table(
        {
            "i": pa.chunked_array(
                [np.arange(0, size // 2), np.arange(size // 2, size)]
            ),
            "j": pa.chunked_array([np.zeros(size // 2, dtype=np.int64)] * 2),
            "x": pa.chunked_array([np.ones(size // 2)] * 2),
        }
    )
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        with patch.object(backend, "_execute_sql", return_value=table):
            result = matrix.to_scipy(format="coo")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    np.testing.assert_array_equal(result.data, np.ones(size))
    # Three owned 8-byte columns and a 1-byte mask, with room for Python
    # overhead but not another full column or a second set of result arrays.
    assert peak - baseline < size * 28 + 256 * 1024


def test_failed_import_cleans_up_partial_registration(backend: Backend) -> None:
    owned = backend._owned_relations.copy()
    register = backend._register_dimension
    calls = 0

    def fail_second(name: str, *, column: str, size: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected dimension failure")
        register(name, column=column, size=size)

    with patch.object(backend, "_register_dimension", side_effect=fail_second):
        with pytest.raises(RuntimeError, match="injected"):
            backend.from_numpy(np.ones((2, 3)), name="retryable")
    assert backend._owned_relations == owned
    assert not backend._created_relations
    matrix = backend.from_numpy(np.ones((2, 3)), name="retryable")
    assert matrix.sum() == 6


def test_failed_operand_and_materialization_release_names(backend: Backend) -> None:
    matrix = backend.from_numpy(np.ones((2, 3)))
    owned = backend._owned_relations.copy()
    with pytest.raises(ValueError, match="broadcastable"):
        matrix + np.ones((4, 4))
    assert backend._owned_relations == owned
    with patch.object(backend, "_materialize_expr", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            matrix.compute(name="retryable")
    assert backend._owned_relations == owned
    assert matrix.compute(name="retryable").sum() == 6


def test_failed_wrap_does_not_delete_borrowed_relation(backend: Backend) -> None:
    backend._register_arrow("borrowed", pa.table({"i": [0], "j": [0], "x": [3.0]}))
    with patch.object(
        backend, "_register_dimension", side_effect=RuntimeError("failed")
    ):
        with pytest.raises(RuntimeError, match="failed"):
            backend.from_relation("borrowed", shape=(2, 2), storage="sparse")
    assert backend._execute_sql('SELECT x FROM "borrowed"').column("x")[0].as_py() == 3
    wrapped = backend.from_relation("borrowed", shape=(2, 2), storage="sparse")
    assert not wrapped._expr.finite_values
    assert wrapped.sum() == 3


def test_failed_gather_releases_selector_resources(backend: Backend) -> None:
    matrix = backend.from_numpy(np.ones((3, 4)))
    owned = backend._owned_relations.copy()
    dimension_relations = backend.dimension_relations

    def fail_gather_shape(shape: tuple[int, int]) -> tuple[str, str]:
        if shape == (2, 4):
            raise RuntimeError("failed gather dimensions")
        return dimension_relations(shape)

    with patch.object(backend, "dimension_relations", side_effect=fail_gather_shape):
        with pytest.raises(RuntimeError, match="failed gather"):
            matrix[[2, 0], :]
    assert backend._owned_relations == owned
    assert not backend._selector_cache
    assert backend._selector_values_registered == 0
    assert matrix[[2, 0], :].sum() == 8


def test_sparse_factors_preserve_missing_zero_and_infinite_products(
    backend: Backend,
) -> None:
    matrix = backend.from_coo([0, 1], [0, 1], [np.inf, 4.0], shape=(2, 2))
    factors = backend.from_coo([0], [1], [2.0], shape=(1, 2))
    with np.errstate(all="ignore"):
        expected = matrix.to_numpy() * factors.to_numpy()
    np.testing.assert_allclose((matrix * factors).to_numpy(), expected, equal_nan=True)


def test_deep_metadata_and_explicit_execution_boundary(backend: Backend) -> None:
    matrix = backend.from_numpy(np.ones((1, 1)))
    node = matrix._expr
    for _ in range(1200):
        node = ScalarBinary(BinaryOp.MULTIPLY, node, 1.0)
    assert node.shape == (1, 1)
    assert node.dtype == "float64"
    with pytest.raises(UnsupportedOperationError, match="depth of 768"):
        backend.compile(node)


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("body", ["1.4 1 8", "1 1e0 8", "1 1 2.5", "1 1 2e0"])
def test_mtx_integer_grammar_and_failed_overwrite(
    tmp_path: Path, canonical: bool, body: str
) -> None:
    path = tmp_path / "input.mtx"
    banner = "%%MatrixMarket matrix coordinate integer general\n2 2 1\n"
    path.write_text(banner + "1 1 3\n")
    with DuckDBBackend.connect() as backend:
        matrix = backend.from_mtx(path, name="values")
        path.write_text(banner + body + "\n")
        with pytest.raises(ValueError, match="malformed"):
            backend.from_mtx(
                path, name="values", overwrite=True, assume_canonical=canonical
            )
        assert matrix.sum() == 3


def test_sparse_export_canonicalizes_explicit_zero_arithmetic(backend: Backend) -> None:
    matrix = backend.from_coo([0], [0], [1.0], shape=(2, 2))
    infinite = backend.from_coo([0], [0], [np.inf], shape=(2, 2))
    assert np.isnan(((matrix * 0) @ infinite).to_numpy()[0, 0])
    assert ((matrix * 0).to_scipy() @ infinite.to_scipy()).toarray()[0, 0] == 0


@pytest.mark.parametrize("backend_type", [DuckDBBackend, DataFusionBackend])
def test_sparse_export_budget_configuration(backend_type: type[Backend]) -> None:
    with pytest.raises(ValueError, match="max_sparse_host_values"):
        backend_type.connect(max_sparse_host_values=-1)
