from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
from scipy import sparse

from dbnumpy import (
    BackendMismatchError,
    DataFusionBackend,
    DBDenseArray,
    DBSparseArray,
    DensificationError,
    DuckDBBackend,
    StorageKind,
    UnsupportedOperationError,
)
from dbnumpy.backends import Backend
from dbnumpy.ir import ReductionOp


@pytest.mark.backend
def test_dense_and_sparse_ingestion(backend: Backend) -> None:
    values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    dense = backend.from_numpy(values)
    sparse_matrix = backend.from_scipy(sparse.csr_array(values))

    assert isinstance(dense, DBDenseArray)
    assert isinstance(sparse_matrix, DBSparseArray)
    assert dense.shape == sparse_matrix.shape == values.shape
    np.testing.assert_array_equal(dense.to_numpy(), values)
    np.testing.assert_array_equal(sparse_matrix.to_numpy(), values)
    np.testing.assert_array_equal(sparse_matrix.to_scipy().toarray(), values)


@pytest.mark.backend
def test_materialization_produces_a_source_backed_equivalent(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    lazy = np.sqrt(matrix + 1.0) * 2.0
    materialized = lazy.compute()

    np.testing.assert_allclose(materialized.to_numpy(), np.sqrt(values + 1.0) * 2.0)
    semantic_plan = materialized.plan()
    root = next(
        node for node in semantic_plan["nodes"] if node["id"] == semantic_plan["root"]
    )
    assert root["node"] == "source"
    assert "SQRT" not in materialized.compile().upper()


@pytest.mark.backend
def test_explain_exposes_engine_plan(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    expression = (matrix + 1.0) * 2.0
    plan = expression.explain()
    assert len(plan) > 20
    assert "PROJECTION" in plan.upper()
    assert "JOIN" not in expression.compile().upper()
    assert "SCAN" in plan.upper()


@pytest.mark.backend
def test_variance_explain_keeps_windowed_statistics_optimizer_visible(
    backend: Backend,
) -> None:
    values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    matrix = backend.from_numpy(values)
    sql, _ = backend._compile_reduction(  # noqa: SLF001 - optimizer plan probe
        matrix._expr,  # noqa: SLF001
        ReductionOp.VAR,
        None,
        ddof=0.5,
    )
    plan = backend._explain_sql(sql).upper()  # noqa: SLF001

    assert sql.count('FROM "dbm_0"') == 1
    assert "OVER (" in sql.upper()
    assert "CROSS JOIN" not in sql.upper()
    assert "WINDOW" in plan
    assert "SCAN" in plan
    np.testing.assert_allclose(matrix.var(ddof=0.5), np.var(values, ddof=0.5))


@pytest.mark.backend
def test_emitted_pointwise_sql_matches_numpy_edge_semantics(backend: Backend) -> None:
    values = np.array([[-0.0, 0.0, -1.0, 1.0, np.nan, np.inf]])
    matrix = backend.from_numpy(values)
    with np.errstate(all="ignore"):
        cases = [
            (np.sqrt(matrix), np.sqrt(values)),
            (np.log(matrix), np.log(values)),
            (np.log1p(matrix), np.log1p(values)),
            (np.expm1(matrix * 1.0e-12), np.expm1(values * 1.0e-12)),
            (np.sign(matrix), np.sign(values)),
            (np.trunc(matrix), np.trunc(values)),
            (np.isnan(matrix), np.isnan(values)),
            (np.log2(matrix), np.log2(values)),
            (np.log10(matrix), np.log10(values)),
            (np.sinh(matrix), np.sinh(values)),
            (np.cosh(matrix), np.cosh(values)),
            (np.tanh(matrix), np.tanh(values)),
            (matrix**-3.0, values**-3.0),
            (matrix**0.5, values**0.5),
            (matrix == 0.0, values == 0.0),
            (matrix < 0.0, values < 0.0),
        ]

    for expression, expected in cases:
        table = backend._execute_sql(  # noqa: SLF001 - emitted-SQL contract probe
            expression.compile()
        )
        actual = np.zeros(expression.shape, dtype=np.float64)
        if table.num_rows:
            i = table.column("i").to_numpy(zero_copy_only=False).astype(np.intp)
            j = table.column("j").to_numpy(zero_copy_only=False).astype(np.intp)
            actual[i, j] = table.column("x").to_numpy(zero_copy_only=False)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=0.0, equal_nan=True
        )
        non_nan = ~np.isnan(expected)
        np.testing.assert_array_equal(
            np.signbit(actual[non_nan]), np.signbit(expected[non_nan])
        )


@pytest.mark.backend
def test_matmul_relational_operators_reach_backend_optimizer(
    backend: Backend,
) -> None:
    left = backend.from_scipy(sparse.csr_array(np.eye(3)))
    right = backend.from_scipy(sparse.csr_array(np.eye(3)))
    plan = (left @ right).explain().upper()
    assert "JOIN" in plan
    assert "AGGREGATE" in plan or "GROUP_BY" in plan
    assert plan.count("SCAN") >= 2


@pytest.mark.backend
def test_cross_backend_operations_are_explicitly_rejected(backend: Backend) -> None:
    other = DuckDBBackend.connect()
    try:
        left = backend.from_numpy(np.ones((2, 2)))
        right = other.from_numpy(np.ones((2, 2)))
        with pytest.raises(BackendMismatchError):
            _ = left + right
    finally:
        other.close()


@pytest.mark.backend
def test_densification_guard_is_checked_before_execution(backend: Backend) -> None:
    backend.max_densify_cells = 3
    matrix = backend.from_scipy(sparse.csr_array(np.eye(2)))
    with pytest.raises(DensificationError, match="enumerate 4 cells"):
        _ = matrix + 1.0


@pytest.mark.backend
def test_storage_metadata_tracks_sparse_zero_image(backend: Backend) -> None:
    matrix = backend.from_scipy(sparse.csr_array(np.eye(3)))
    assert (matrix * 2.0).storage is StorageKind.SPARSE
    assert (matrix + 1.0).storage is StorageKind.DENSE
    assert (matrix == 0.0).storage is StorageKind.DENSE
    assert np.exp(matrix).storage is StorageKind.DENSE


@pytest.mark.backend
def test_depth_100_pointwise_chain_remains_fused_and_executable(
    backend: Backend,
) -> None:
    values = np.arange(16.0).reshape(4, 4) + 1.0
    matrix = backend.from_numpy(values)
    expected = values
    for _ in range(100):
        matrix = np.sqrt(matrix * 1.000001 + 0.000001)
        expected = np.sqrt(expected * 1.000001 + 0.000001)

    sql = matrix.compile()
    assert "JOIN" not in sql.upper()
    assert sql.count('FROM "dbm_0"') == 1
    np.testing.assert_allclose(matrix.to_numpy(), expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(matrix.sum(), expected.sum(), rtol=1e-11, atol=1e-11)
    extrema_sql, _ = backend._compile_reduction(  # noqa: SLF001 - plan probe
        matrix._expr,
        ReductionOp.NANMAX,
        None,
        ddof=0,  # noqa: SLF001
    )
    assert extrema_sql.count('FROM "dbm_0"') == 1
    assert len(extrema_sql.encode()) < 100_000
    np.testing.assert_allclose(matrix.nanmax(), np.nanmax(expected))

    transposed = matrix.T
    transposed_sql = transposed.compile()
    assert transposed_sql.count('"dbm_0"') == 1
    assert len(transposed_sql) < 40_000
    np.testing.assert_allclose(
        transposed.to_numpy(), expected.T, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        transposed.mean(axis=0), expected.T.mean(axis=0), rtol=1e-11, atol=1e-11
    )

    post_structural = matrix.T
    expected_post_structural = expected.T
    for _ in range(100):
        post_structural = np.sqrt(post_structural * 1.000001 + 0.000001)
        expected_post_structural = np.sqrt(
            expected_post_structural * 1.000001 + 0.000001
        )
    post_structural_sql = post_structural.compile()
    assert post_structural_sql.count('"dbm_0"') == 1
    assert len(post_structural_sql) < 80_000
    np.testing.assert_allclose(
        post_structural.to_numpy(),
        expected_post_structural,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.backend
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_shared_pointwise_dag_has_linear_sql_and_executes_once(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.arange(6.0).reshape(2, 3)
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    for _ in range(25):
        matrix = (matrix + matrix) * 0.5
    matrix = matrix + 1.0

    sql = matrix.compile()
    assert len(sql) < 12_000
    assert sql.count('"dbm_0"') == 1
    np.testing.assert_allclose(matrix.to_numpy(), values + 1.0)
    np.testing.assert_allclose(matrix.var(axis=0), (values + 1.0).var(axis=0))


@pytest.mark.backend
def test_nonfinite_scalar_zero_image_matches_numpy(backend: Backend) -> None:
    values = np.array([[0.0, 2.0], [3.0, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    with np.errstate(invalid="ignore"):
        expected = values * np.inf
    np.testing.assert_allclose((matrix * np.inf).to_numpy(), expected, equal_nan=True)


@pytest.mark.backend
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_zero_sized_dense_matrices_preserve_shape(
    backend: Backend, shape: tuple[int, int]
) -> None:
    matrix = backend.from_numpy(np.empty(shape))
    assert matrix.shape == shape
    assert matrix.to_numpy().shape == shape
    assert matrix.T.to_numpy().shape == shape[::-1]
    assert matrix.sum() == 0.0
    np.testing.assert_array_equal(matrix.sum(axis=0), np.zeros(shape[1]))
    np.testing.assert_array_equal(matrix.sum(axis=1), np.zeros(shape[0]))

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for axis in (None, 0, 1):
            np.testing.assert_allclose(
                matrix.mean(axis=axis),
                np.mean(np.empty(shape), axis=axis),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                matrix.var(axis=axis),
                np.var(np.empty(shape), axis=axis),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                matrix.std(axis=axis, ddof=1),
                np.std(np.empty(shape), axis=axis, ddof=1),
                equal_nan=True,
            )


@pytest.mark.backend
def test_closed_backend_rejects_new_work() -> None:
    backend = DuckDBBackend.connect()
    matrix = backend.from_numpy(np.eye(2))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        matrix.to_numpy()


@pytest.mark.backend
def test_invalid_backend_resource_limits_fail_before_connection() -> None:
    with pytest.raises(ValueError, match="threads must be positive"):
        DuckDBBackend.connect(threads=0)
    with pytest.raises(ValueError, match="max_densify_cells"):
        DuckDBBackend.connect(max_densify_cells=-1)
    with pytest.raises(ValueError, match="max_host_values"):
        DuckDBBackend.connect(max_host_values=-1)
    with pytest.raises(ValueError, match="max_selector_values"):
        DuckDBBackend.connect(max_selector_values=-1)
    with pytest.raises(ValueError, match="max_selector_relations"):
        DuckDBBackend.connect(max_selector_relations=-1)
    with pytest.raises(ValueError, match="target_partitions"):
        DataFusionBackend.connect(target_partitions=0)
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        DataFusionBackend.connect(memory_limit_bytes=0)
    with pytest.raises(ValueError, match="recursion_limit"):
        DataFusionBackend.connect(sql_parser_recursion_limit=0)
    with pytest.raises(ValueError, match="max_densify_cells"):
        DataFusionBackend.connect(max_densify_cells=-1)
    with pytest.raises(ValueError, match="max_host_values"):
        DataFusionBackend.connect(max_host_values=-1)
    with pytest.raises(ValueError, match="max_selector_values"):
        DataFusionBackend.connect(max_selector_values=-1)
    with pytest.raises(ValueError, match="max_selector_relations"):
        DataFusionBackend.connect(max_selector_relations=-1)


@pytest.mark.backend
def test_compilation_is_cached_per_immutable_plan(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    plan = np.sqrt(matrix + 1.0)
    with patch.object(
        backend.lowerer,
        "compile_matrix",
        wraps=backend.lowerer.compile_matrix,
    ) as compile_matrix:
        first = plan.compile()
        second = plan.compile()
        np.testing.assert_allclose(
            plan.to_numpy(), np.sqrt(np.arange(6.0).reshape(2, 3) + 1.0)
        )
    assert first == second
    assert compile_matrix.call_count == 1


@pytest.mark.backend
def test_plan_caches_evict_at_the_configured_bound(backend: Backend) -> None:
    backend._compile_cache_limit = 3  # noqa: SLF001 - bounded-cache contract probe
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    plans = [matrix + float(index) for index in range(5)]

    for plan in plans:
        plan.compile()
        backend._compile_reduction(  # noqa: SLF001 - bounded-cache contract probe
            plan._expr,  # noqa: SLF001 - bounded-cache contract probe
            ReductionOp.SUM,
            None,
            ddof=0,
        )
        if isinstance(backend, DataFusionBackend):
            plan.to_numpy()

    assert len(backend._compile_cache) == 3  # noqa: SLF001
    assert len(backend._reduction_cache) == 3  # noqa: SLF001
    assert id(plans[0]._expr) not in backend._compile_cache  # noqa: SLF001
    if isinstance(backend, DataFusionBackend):
        assert len(backend._native_plan_cache) == 3  # noqa: SLF001


@pytest.mark.backend
def test_datafusion_native_pointwise_plan_is_cached() -> None:
    backend = DataFusionBackend.connect()
    try:
        matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
        plan = np.sqrt(matrix + 1.0)
        with patch.object(
            backend.native_pointwise_lowerer,
            "lower",
            wraps=backend.native_pointwise_lowerer.lower,
        ) as lower:
            plan.to_numpy()
            plan.to_numpy()
            plan.explain()
        assert lower.call_count == 1
    finally:
        backend.close()


@pytest.mark.backend
def test_capabilities_are_explicit_and_backend_specific(backend: Backend) -> None:
    capabilities = backend.capabilities
    assert capabilities.backend_name == backend.name
    assert capabilities.sql_dialect == backend.dialect
    assert capabilities.supports_node("matmul")
    assert capabilities.supports_node("gather")
    assert capabilities.supports_reduction("var")
    assert not capabilities.supports_node("svd")
    assert capabilities.native_pointwise_execution is isinstance(
        backend, DataFusionBackend
    )
    assert capabilities.native_dimension_ranges
    assert capabilities.native_matrix_market_ingestion is isinstance(
        backend, DuckDBBackend
    )
    assert capabilities.external_parquet_scan is isinstance(
        backend, DataFusionBackend
    )


@pytest.mark.backend
def test_duplicate_user_relation_names_are_rejected(backend: Backend) -> None:
    backend.from_numpy(np.eye(2), name="owned_matrix")
    with pytest.raises(ValueError, match="already owned"):
        backend.from_numpy(np.ones((2, 2)), name="owned_matrix")


@pytest.mark.backend
def test_existing_coordinate_relation_can_be_wrapped(backend: Backend) -> None:
    backend._register_arrow(  # noqa: SLF001 - backend conformance probe
        "existing_values",
        pa.table(
            {
                "i": pa.array([0, 1], type=pa.int64()),
                "j": pa.array([1, 0], type=pa.int64()),
                "x": pa.array([2.0, 3.0], type=pa.float64()),
            }
        ),
    )
    matrix = backend.from_relation("existing_values", shape=(2, 2), storage="sparse")
    np.testing.assert_array_equal(matrix.to_numpy(), np.array([[0.0, 2.0], [3.0, 0.0]]))


@pytest.mark.backend
def test_integer_value_relation_normalizes_to_float64(backend: Backend) -> None:
    backend._register_arrow(  # noqa: SLF001 - backend conformance probe
        "integer_values",
        pa.table(
            {
                "i": pa.array([0, 1], type=pa.int32()),
                "j": pa.array([1, 0], type=pa.int32()),
                "x": pa.array([None, 3], type=pa.int32()),
            }
        ),
    )
    matrix = backend.from_relation("integer_values", shape=(2, 2), storage="sparse")
    expected = np.array([[0.0, np.nan], [3.0, 0.0]])
    assert matrix.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(matrix.to_numpy(), expected, equal_nan=True)
    np.testing.assert_allclose(
        (matrix + 0.5).to_numpy(), expected + 0.5, equal_nan=True
    )


@pytest.mark.backend
def test_internal_dimension_names_do_not_replace_wrapped_relations(
    backend: Backend,
) -> None:
    relation = "dbm_dim_2_2_rows_0"
    backend._register_arrow(  # noqa: SLF001 - backend conformance probe
        relation,
        pa.table(
            {
                "i": pa.array([0, 1], type=pa.int64()),
                "j": pa.array([1, 0], type=pa.int64()),
                "x": pa.array([2.0, 3.0], type=pa.float64()),
            }
        ),
    )
    matrix = backend.from_relation(relation, shape=(2, 2), storage="sparse")
    np.testing.assert_array_equal(matrix.to_numpy(), np.array([[0.0, 2.0], [3.0, 0.0]]))
    source = matrix._expr  # noqa: SLF001 - ownership contract probe
    assert source.rows_relation != relation


@pytest.mark.backend
def test_large_sparse_shape_uses_lazy_native_dimension_ranges(
    backend: Backend,
) -> None:
    shape = (10_000_000, 20_000_000)
    with patch.object(
        backend,
        "_register_arrow",
        wraps=backend._register_arrow,  # noqa: SLF001 - resource contract probe
    ) as register_arrow:
        rows_relation, cols_relation = backend.dimension_relations(shape)
    assert register_arrow.call_count == 0
    assert backend.dimension_relations(shape) == (rows_relation, cols_relation)

    matrix = backend.from_coo([], [], [], shape=shape)
    assert matrix.shape == shape
    assert matrix.sum() == 0.0
    host_sparse = matrix.to_scipy()
    assert host_sparse.shape == shape
    assert host_sparse.nnz == 0
    backend.max_host_values = 19_999_999
    with pytest.raises(
        DensificationError,
        match=r"20,000,000 host values.*max_host_values=19,999,999",
    ):
        matrix.sum(axis=0)


@pytest.mark.backend
def test_default_host_limit_allows_figure1g_column_mean_width(
    backend: Backend,
) -> None:
    matrix = backend.from_coo([], [], [], shape=(1, 10_000_000))

    result = matrix.mean(axis=0)

    assert result.shape == (10_000_000,)
    assert result.nbytes == 80_000_000
    assert not result.any()


@pytest.mark.backend
def test_existing_relation_schema_is_validated(backend: Backend) -> None:
    backend._register_arrow(  # noqa: SLF001 - backend conformance probe
        "invalid_values",
        pa.table({"not_x": pa.array([1.0])}),
    )
    with pytest.raises(ValueError, match="expose numeric i, j, x"):
        backend.from_relation("invalid_values", shape=(1, 1), storage="dense")

    backend._register_arrow(  # noqa: SLF001 - backend conformance probe
        "string_values",
        pa.table({"i": ["0"], "j": ["0"], "x": ["1.0"]}),
    )
    with pytest.raises(ValueError, match="integer i/j and numeric x"):
        backend.from_relation("string_values", shape=(1, 1), storage="dense")


@pytest.mark.backend
@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_sql_null_values_normalize_to_numpy_nan(backend: Backend, storage: str) -> None:
    if storage == "dense":
        table = pa.table(
            {
                "i": pa.array([0, 0], type=pa.int64()),
                "j": pa.array([0, 1], type=pa.int64()),
                "x": pa.array([None, 0.0], type=pa.float64()),
            }
        )
        expected = np.array([[np.nan, 0.0]])
    else:
        table = pa.table(
            {
                "i": pa.array([0], type=pa.int64()),
                "j": pa.array([0], type=pa.int64()),
                "x": pa.array([None], type=pa.float64()),
            }
        )
        expected = np.array([[np.nan, 0.0]])
    backend._register_arrow(  # noqa: SLF001 - backend conformance probe
        "nullable_values", table
    )
    matrix = backend.from_relation("nullable_values", shape=(1, 2), storage=storage)

    np.testing.assert_allclose(matrix.to_numpy(), expected, equal_nan=True)
    np.testing.assert_array_equal((matrix == 0.0).to_numpy(), expected == 0.0)
    np.testing.assert_allclose(
        np.exp(matrix).to_numpy(), np.exp(expected), equal_nan=True
    )
    np.testing.assert_allclose(
        np.arctan(matrix).to_numpy(), np.arctan(expected), equal_nan=True
    )
    np.testing.assert_allclose(
        np.arccos(matrix).to_numpy(), np.arccos(expected), equal_nan=True
    )
    np.testing.assert_allclose(matrix.T.to_numpy(), expected.T, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(matrix).to_numpy(), np.isnan(expected))
    assert np.isnan(matrix.sum())
    assert np.isnan(matrix.min())
    assert matrix.nansum() == np.nansum(expected)
    assert matrix.nanmean() == np.nanmean(expected)
    assert matrix.nanmin() == np.nanmin(expected)
    assert matrix.nanmax() == np.nanmax(expected)


@pytest.mark.backend
def test_unsupported_numpy_options_fail_explicitly(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    with pytest.raises(UnsupportedOperationError, match="float64"):
        matrix.sum(dtype=np.float32)
    with pytest.raises(UnsupportedOperationError, match="out="):
        matrix.sum(out=np.empty(3))
    with pytest.raises(UnsupportedOperationError, match="initial="):
        matrix.sum(initial=2.0)
    with pytest.raises(UnsupportedOperationError, match="where="):
        matrix.mean(where=False)
    with pytest.raises(UnsupportedOperationError, match="precomputed mean"):
        matrix.var(mean=np.zeros(3))
    with pytest.raises(np.exceptions.AxisError, match="out of bounds"):
        matrix.sum(axis=2)
    with pytest.raises(ValueError, match="ddof must"):
        matrix.var(ddof=-1)


@pytest.mark.backend
def test_invalid_array_operands_and_transpose_axes_fail(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        _ = matrix + np.ones((1, 2, 3))
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = matrix + (1.0 + 2.0j)
    with pytest.raises(ValueError, match="axes"):
        np.transpose(matrix, axes=(0, 0))
    with pytest.raises(ValueError, match="axes"):
        np.transpose(matrix, axes=(0, 2))


@pytest.mark.backend
def test_ingestion_validates_shape_coordinates_and_names(backend: Backend) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        backend.from_numpy(np.ones(3))
    with pytest.raises(ValueError, match="lengths must match"):
        backend.from_coo([0], [0, 1], [1.0], shape=(2, 2))
    with pytest.raises(IndexError, match="row index"):
        backend.from_coo([2], [0], [1.0], shape=(2, 2))
    with pytest.raises(IndexError, match="column index"):
        backend.from_coo([0], [-1], [1.0], shape=(2, 2))
    with pytest.raises(ValueError, match="unique"):
        backend.from_coo([0, 0], [1, 1], [1.0, 2.0], shape=(2, 2))
    with pytest.raises(ValueError, match="relation names"):
        backend.from_numpy(np.eye(2), name="unsafe-name")
    with pytest.raises(TypeError, match="SciPy sparse"):
        backend.from_scipy(np.eye(2))
    with pytest.raises(TypeError, match="dimensions must be integers"):
        backend.from_coo([], [], [], shape=(2.5, 3))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="row indices.*integer dtype"):
        backend.from_coo([0.5], [0], [1.0], shape=(2, 2))
    with pytest.raises(TypeError, match="column indices.*integer dtype"):
        backend.from_coo([0], ["1"], [1.0], shape=(2, 2))
    with pytest.raises(TypeError, match="dense matrix values must be numeric"):
        backend.from_numpy(np.array([["1.0"]]))
    with pytest.raises(TypeError, match="complex matrix values"):
        backend.from_numpy(np.array([[1.0 + 2.0j]]))
    with pytest.raises(TypeError, match="complex COO data"):
        backend.from_coo([0], [0], [1.0 + 2.0j], shape=(1, 1))
    with pytest.raises(OverflowError, match="signed int64"):
        backend.from_coo([], [], [], shape=(2**63, 1))


@pytest.mark.backend
def test_dense_ingestion_obeys_explicit_cell_cap(backend: Backend) -> None:
    backend.max_densify_cells = 3
    with pytest.raises(DensificationError, match="dense ingestion"):
        backend.from_numpy(np.ones((2, 2)))


@pytest.mark.backend
def test_host_collection_and_dense_shape_expansion_obey_cell_cap(
    backend: Backend,
) -> None:
    backend.max_densify_cells = 10
    sparse_matrix = backend.from_coo([0], [0], [1.0], shape=(4, 4))
    with pytest.raises(DensificationError, match=r"to_numpy\(\).*16 host values"):
        sparse_matrix.to_numpy()
    assert sparse_matrix.to_scipy().shape == (4, 4)

    column = backend.from_numpy(np.ones((4, 1)))
    row = backend.from_numpy(np.ones((1, 4)))
    with pytest.raises(DensificationError, match="broadcasting.*16-cell output"):
        _ = column + row
    with pytest.raises(
        DensificationError, match="dense matrix multiplication.*16-cell output"
    ):
        _ = column @ row
