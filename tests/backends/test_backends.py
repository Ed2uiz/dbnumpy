from __future__ import annotations

import json
from typing import Any
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
    PolarsBackend,
    UnsupportedOperationError,
)
from dbnumpy.backends import Backend
from dbnumpy.ir import BinaryOp, ReductionOp, ScalarBinary


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
    assert "PROJECT" in plan.upper()
    assert "JOIN" not in expression.compile().upper()
    # Polars shows an in-memory table as DF rather than a scan.
    assert ("DF [" if isinstance(backend, PolarsBackend) else "SCAN") in plan.upper()


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
    if isinstance(backend, PolarsBackend):
        # Polars plans these windows as aggregates over one input table.
        assert plan.count("DF [") == 1
    else:
        assert "WINDOW" in plan
        assert "SCAN" in plan
    np.testing.assert_allclose(matrix.var(ddof=0.5), np.var(values, ddof=0.5))


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
def test_closed_backend_rejects_new_work() -> None:
    backend = DuckDBBackend.connect()
    matrix = backend.from_numpy(np.eye(2))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        matrix.to_numpy()


@pytest.mark.backend
def test_plans_deeper_than_the_cap_fail_before_compilation(backend: Backend) -> None:
    node = backend.from_numpy(np.ones((1, 1)))._expr
    for _ in range(1200):
        node = ScalarBinary(BinaryOp.MULTIPLY, node, 1.0)
    assert node.shape == (1, 1)
    with pytest.raises(UnsupportedOperationError, match="depth of 768"):
        backend.compile(node)


@pytest.mark.backend
def test_shared_work_is_compiled_once(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    branch = matrix.T + np.array([0.25, -0.5, 0.75])
    for result, expected in (
        ((matrix * 2.0) * (matrix > 1.0), (values * 2.0) * (values > 1.0)),
        ((branch + branch) * 0.5, values.T + np.array([0.25, -0.5, 0.75])),
        (matrix.T + matrix.T, values.T * 2),
    ):
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.compile().count('"dbm_0"') == 1
    assert matrix.T.T._expr is matrix._expr


@pytest.mark.backend
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_ingestion_copies_host_arrays(backend: Backend, layout: str) -> None:
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


def _fail_on_call(method: Any, call: int) -> Any:
    calls = 0

    def failing(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == call:
            raise RuntimeError("injected")
        return method(*args, **kwargs)

    return failing


@pytest.mark.backend
@pytest.mark.parametrize(
    ("target", "call", "action"),
    [
        (
            "_register_dimension",
            2,
            lambda b, m: b.from_numpy(np.ones((2, 3)), name="retry"),
        ),
        ("_materialize_expr", 1, lambda b, m: m.compute(name="retry")),
        ("dimension_relations", 1, lambda b, m: m[[2, 0], :]),
        (
            "_register_dimension",
            1,
            lambda b, m: b.from_relation("borrowed", shape=(5, 5), storage="sparse"),
        ),
    ],
    ids=["import", "compute", "gather", "wrap"],
)
def test_failed_registration_rolls_back(
    backend: Backend, target: str, call: int, action: Any
) -> None:
    matrix = backend.from_numpy(np.ones((3, 4)))
    backend._register_arrow("borrowed", pa.table({"i": [0], "j": [0], "x": [3.0]}))
    owned, selectors = backend._owned_relations.copy(), dict(backend._selector_cache)
    with patch.object(backend, target, _fail_on_call(getattr(backend, target), call)):
        with pytest.raises(RuntimeError, match="injected"):
            action(backend, matrix)
    assert backend._owned_relations == owned
    assert backend._selector_cache == selectors
    assert backend._execute_sql('SELECT x FROM "borrowed"').num_rows == 1
    action(backend, matrix)  # the same names are free to use again


@pytest.mark.backend
@pytest.mark.parametrize(
    ("backend_type", "option"),
    [
        (DuckDBBackend, {"threads": 0}),
        (DataFusionBackend, {"target_partitions": 0}),
        (DataFusionBackend, {"memory_limit_bytes": 0}),
        (DataFusionBackend, {"sql_parser_recursion_limit": 0}),
        *(
            (backend_type, {limit: -1})
            for backend_type in (DuckDBBackend, DataFusionBackend)
            for limit in (
                "max_densify_cells",
                "max_host_values",
                "max_sparse_host_values",
                "max_selector_values",
                "max_selector_relations",
            )
        ),
    ],
)
def test_invalid_resource_limits_fail_before_connection(
    backend_type: type[Backend], option: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match=next(iter(option))):
        backend_type.connect(**option)


@pytest.mark.backend
def test_compilation_is_cached_per_immutable_plan(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    plan = np.sqrt(matrix + 1.0)
    assert plan.compile() is plan.compile()


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
        backend, (DataFusionBackend, PolarsBackend)
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
def test_default_host_limit_allows_wide_axis_reductions(
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
def test_plan_is_plain_json(backend: Backend) -> None:
    matrix = backend.from_numpy(np.ones((2, 3)))
    expression = (matrix.T + np.ones(2)) * np.nan + np.inf + (-np.inf) - 2.5
    plan = json.loads(json.dumps(expression.plan(), allow_nan=False))
    nodes = {node["node"] for node in plan["nodes"]}
    assert {"source", "transpose", "broadcast", "scalar_binary"} <= nodes
    scalars = [node["scalar"] for node in plan["nodes"] if "scalar" in node]
    assert {"special_float": "nan"} in scalars
    assert {"special_float": "+inf"} in scalars
    assert {"special_float": "-inf"} in scalars
    assert 2.5 in scalars


@pytest.mark.backend
def test_numpy_container_conventions(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    assert (len(matrix), matrix.ndim, matrix.size) == (3, 2, 12)
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal((-matrix).to_numpy(), -values)
    np.testing.assert_array_equal(abs(-matrix).to_numpy(), values)
    np.testing.assert_array_equal(np.asarray(matrix, dtype=np.float32), values)
    assert np.transpose(matrix, axes=(-2, -1)) is matrix
    np.testing.assert_array_equal(np.transpose(matrix).to_numpy(), values.T)
    np.testing.assert_array_equal(
        np.transpose(matrix, axes=(1, 0)).to_numpy(), values.T
    )
    with pytest.raises(ValueError, match=r"truth value.*ambiguous"):
        bool(matrix)
    with pytest.raises(NotImplementedError, match="newaxis"):
        matrix[None, :]


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
