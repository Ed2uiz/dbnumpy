from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from dbnumpy import DBArray, DBScalar, DBVector
from dbnumpy.backends import Backend
from dbnumpy.exceptions import DensificationError, UnsupportedOperationError


def _collect_index_result(result: DBArray[Any] | DBVector | DBScalar) -> Any:
    return result.item() if isinstance(result, DBScalar) else result.to_numpy()


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_basic_indexing_has_numpy_rank_and_values(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    values[values % 4 == 0] = 0.0
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )

    row = matrix[1]
    column = matrix[:, -2]
    scalar = matrix[-1, -2]
    assert isinstance(row, DBVector)
    assert isinstance(column, DBVector)
    assert isinstance(scalar, DBScalar)
    assert row.shape == (5,) and row.ndim == 1 and row.size == 5
    assert column.shape == (4,) and column.ndim == 1 and column.size == 4
    assert scalar.shape == () and scalar.ndim == 0 and scalar.size == 1
    np.testing.assert_array_equal(row.to_numpy(), values[1])
    np.testing.assert_array_equal(column.to_numpy(), values[:, -2])
    assert scalar.item() == values[-1, -2]
    assert scalar.to_numpy().shape == ()


@pytest.mark.parity
def test_ellipsis_expansion_matches_numpy_for_matrix_and_vector(
    backend: Backend,
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    matrix = backend.from_numpy(values)
    selector = np.array([3, 1], dtype=np.int64)

    np.testing.assert_array_equal(matrix[...].to_numpy(), values[...])
    np.testing.assert_array_equal(
        matrix[selector, ...].to_numpy(), values[selector, ...]
    )
    np.testing.assert_array_equal(
        matrix[..., selector].to_numpy(), values[..., selector]
    )
    assert matrix[..., 1, 2].item() == values[..., 1, 2]
    assert matrix[1, ..., 2].item() == values[1, ..., 2]
    assert matrix[1, 2, ...].item() == values[1, 2, ...]

    vector = matrix[1, :]
    np.testing.assert_array_equal(vector[()].to_numpy(), values[1, :][()])
    np.testing.assert_array_equal(vector[...].to_numpy(), values[1, ...])
    np.testing.assert_array_equal(vector[(...,)].to_numpy(), values[1, ...])
    assert vector[1,].item() == values[1, :][1,]
    with pytest.raises(IndexError, match="too many indices"):
        _ = vector[1, 2]


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_signed_slices_match_numpy_without_expanding_selector_domains(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.arange(30.0).reshape(5, 6)
    values[values % 5 == 0] = 0.0
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )

    for key in (
        (slice(None, None, -1), slice(None, None, -2)),
        (slice(3, 0, -2), slice(4, 1, -1)),
        (slice(1, 1, -1), slice(None)),
        (slice(20, -20, -3), slice(-20, 20, 2)),
    ):
        result = matrix[key]
        expected = values[key]
        assert isinstance(result, DBArray)
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parity
@pytest.mark.parametrize("axis", [0, 1], ids=["rows", "columns"])
def test_integer_gather_preserves_order_repetition_and_negative_indices(
    backend: Backend, axis: int
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    matrix = backend.from_numpy(values)
    selector = np.array([-1, 0, -1, 1], dtype=np.int64)
    key = (selector, slice(None)) if axis == 0 else (slice(None), selector)
    result = matrix[key]

    expected = values[key]
    assert isinstance(result, DBArray)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.to_numpy(), expected)
    sql = result.compile()
    assert "source_index" in sql and "output_index" in sql


@pytest.mark.parity
def test_boolean_empty_and_mixed_gathers_match_numpy(backend: Backend) -> None:
    values = np.arange(20.0).reshape(4, 5)
    matrix = backend.from_numpy(values)

    cases: tuple[tuple[Any, Any], ...] = (
        ([True, False, True, False], slice(None)),
        (slice(None), [False, True, False, True, False]),
        (np.array([], dtype=np.int64), slice(None)),
        (slice(None), []),
        ([3, 1], slice(1, 5, 2)),
        (slice(1, 4), [4, 1, 4]),
    )
    for key in cases:
        result = matrix[key]
        expected = values[key]
        assert isinstance(result, DBArray)
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.to_numpy(), expected)

    row_vector = matrix[[3, 1], 2]
    column_vector = matrix[1, [4, 0, 4]]
    assert isinstance(row_vector, DBVector)
    assert isinstance(column_vector, DBVector)
    np.testing.assert_array_equal(row_vector.to_numpy(), values[[3, 1], 2])
    np.testing.assert_array_equal(column_vector.to_numpy(), values[1, [4, 0, 4]])


@pytest.mark.parity
def test_sequential_advanced_indexing_is_explicit_cartesian_selection(
    backend: Backend,
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    matrix = backend.from_numpy(values)
    with pytest.raises(UnsupportedOperationError, match="paired semantics"):
        _ = matrix[[3, 1], [4, 0]]

    result = matrix[[3, 1], :][:, [4, 0]]
    expected = values[[3, 1], :][:, [4, 0]]
    np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parity
def test_take_method_and_numpy_dispatch_use_the_same_lazy_gather(
    backend: Backend,
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    matrix = backend.from_numpy(values)

    rows = matrix.take([3, 1, 3], axis=0)
    columns = np.take(matrix, [4, 0, 4], axis=-1)
    one_row = np.take(matrix, 2, axis=0)
    np.testing.assert_array_equal(rows.to_numpy(), np.take(values, [3, 1, 3], axis=0))
    np.testing.assert_array_equal(
        columns.to_numpy(), np.take(values, [4, 0, 4], axis=-1)
    )
    assert isinstance(one_row, DBVector)
    np.testing.assert_array_equal(one_row.to_numpy(), np.take(values, 2, axis=0))
    with pytest.raises(UnsupportedOperationError, match="flatten"):
        matrix.take([0, 1])
    with pytest.raises(UnsupportedOperationError, match="mode='raise'"):
        matrix.take([0, 1], axis=0, mode="wrap")


@pytest.mark.parity
def test_sparse_gather_restores_implicit_zeros_and_scalar_zero(
    backend: Backend,
) -> None:
    values = np.array(
        [
            [0.0, 11.0, 0.0, 13.0, 0.0],
            [20.0, 0.0, 22.0, 0.0, 0.0],
            [0.0, 31.0, 0.0, 33.0, 0.0],
        ]
    )
    matrix = backend.from_scipy(sparse.csr_array(values))

    result = matrix[:, [4, 2, 4]]
    expected = values[:, [4, 2, 4]]
    assert result.storage.value == "sparse"
    np.testing.assert_array_equal(result.to_numpy(), expected)
    assert matrix[0, 0].item() == 0.0
    np.testing.assert_array_equal(
        matrix[[2, 1, 0], 1].to_numpy(), values[[2, 1, 0], 1]
    )

    densified = matrix[[2, 0, 2], :] + 1.0
    np.testing.assert_array_equal(densified.to_numpy(), values[[2, 0, 2], :] + 1.0)
    implicit_column = matrix[:, 4]
    np.testing.assert_array_equal(
        np.exp(implicit_column).to_numpy(), np.exp(values[:, 4])
    )


@pytest.mark.parity
def test_gather_before_densifying_transform_bounds_the_output_domain(
    backend: Backend,
) -> None:
    values = np.zeros((20, 20), dtype=np.float64)
    values[0, 1] = 2.0
    values[1, 3] = -1.0
    matrix = backend.from_scipy(sparse.csr_array(values))
    backend.max_densify_cells = 50

    selected_first = np.exp(matrix[[0, 1], :])
    np.testing.assert_allclose(selected_first.to_numpy(), np.exp(values[[0, 1], :]))

    with pytest.raises(DensificationError, match="enumerate 400 cells"):
        _ = np.exp(matrix)


@pytest.mark.parity
def test_gather_composes_lazily_with_pointwise_and_ranked_operations(
    backend: Backend,
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    matrix = backend.from_numpy(values)

    gathered = matrix[[3, 1, 3], 1]
    assert isinstance(gathered, DBVector)
    np.testing.assert_array_equal(
        (gathered * 2 + 1).to_numpy(), values[[3, 1, 3], 1] * 2 + 1
    )
    np.testing.assert_array_equal(gathered[::-1].to_numpy(), values[[3, 1, 3], 1][::-1])
    assert gathered[1].item() == values[1, 1]
    scalar = matrix[2, 3]
    assert (scalar * 2 + 1).item() == values[2, 3] * 2 + 1


@pytest.mark.parity
def test_vector_layout_is_not_public_orientation(backend: Backend) -> None:
    values = np.arange(16.0).reshape(4, 4)
    matrix = backend.from_numpy(values)
    row = matrix[1, :]
    column = matrix[:, 2]

    assert row.T is row and column.T is column
    assert np.transpose(row) is row
    assert np.transpose(row, axes=(0,)) is row
    assert np.transpose(row, axes=(-1,)) is row
    with pytest.raises(np.exceptions.AxisError):
        np.transpose(row, axes=(1,))
    with pytest.raises(np.exceptions.AxisError):
        np.transpose(row, axes=(-2,))
    with pytest.raises(ValueError, match="axes must"):
        np.transpose(row, axes=())
    assert np.sum(row) == np.sum(values[1, :])
    assert np.mean(column) == np.mean(values[:, 2])
    np.testing.assert_array_equal(
        np.sum(row, keepdims=True), np.sum(values[1, :], keepdims=True)
    )
    np.testing.assert_array_equal(
        (row + column).to_numpy(), values[1, :] + values[:, 2]
    )
    np.testing.assert_allclose(np.sqrt(row + 1).to_numpy(), np.sqrt(values[1, :] + 1))
    np.testing.assert_array_equal((column > 5).to_numpy(), values[:, 2] > 5)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_ranked_results_follow_minimal_numpy_broadcasting(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.arange(16.0).reshape(4, 4)
    values[values % 3 == 0] = 0.0
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    scalar = matrix[1, 1]
    vector = matrix[:, 2]

    np.testing.assert_array_equal((matrix + scalar).to_numpy(), values + values[1, 1])
    np.testing.assert_array_equal((scalar - matrix).to_numpy(), values[1, 1] - values)
    np.testing.assert_array_equal(
        (vector + scalar).to_numpy(), values[:, 2] + values[1, 1]
    )
    np.testing.assert_array_equal(
        (scalar - vector).to_numpy(), values[1, 1] - values[:, 2]
    )
    np.testing.assert_array_equal((matrix + vector).to_numpy(), values + values[:, 2])
    np.testing.assert_array_equal((vector + matrix).to_numpy(), values[:, 2] + values)
    np.testing.assert_array_equal(
        (vector + np.arange(4.0)).to_numpy(), values[:, 2] + np.arange(4.0)
    )


@pytest.mark.parity
def test_lazy_scalar_broadcasts_with_host_ranked_arrays(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)
    scalar = matrix[1, 2]
    scalar_value = values[1, 2]

    assert (scalar + np.array(2.0)).item() == scalar_value + 2.0
    vector = np.arange(4.0)
    np.testing.assert_array_equal(
        (scalar - vector).to_numpy(), scalar_value - vector
    )
    np.testing.assert_array_equal(
        (vector - scalar).to_numpy(), vector - scalar_value
    )
    host_matrix = np.arange(8.0).reshape(2, 4)
    np.testing.assert_array_equal(
        (scalar + host_matrix).to_numpy(), scalar_value + host_matrix
    )
    with pytest.raises(ValueError, match="at most two dimensions"):
        _ = scalar + np.zeros((1, 1, 1))


@pytest.mark.parity
def test_sparse_implicit_scalar_densification_uses_singleton_domain(
    backend: Backend,
) -> None:
    values = np.array([[0.0, 2.0], [3.0, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    zero = matrix[0, 0]

    assert np.exp(zero).item() == 1.0
    assert (zero + 1).item() == 1.0
    assert np.isinf((1 / zero).item())
    np.testing.assert_array_equal((matrix + zero).to_numpy(), values)


@pytest.mark.parity
def test_rank_wrappers_survive_backend_materialization(backend: Backend) -> None:
    values = np.arange(12.0).reshape(3, 4)
    matrix = backend.from_numpy(values)

    vector = (matrix[2, :] + 1).compute()
    scalar = (matrix[1, 2] * 2).compute()
    assert isinstance(vector, DBVector)
    assert isinstance(scalar, DBScalar)
    np.testing.assert_array_equal(vector.to_numpy(), values[2, :] + 1)
    assert scalar.item() == values[1, 2] * 2
    assert np.transpose(scalar) is scalar
    assert np.transpose(scalar, axes=()) is scalar
    with pytest.raises(ValueError, match="axes must"):
        np.transpose(scalar, axes=(0,))


@pytest.mark.parity
def test_index_validation_fails_before_query_execution(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(20.0).reshape(4, 5))

    for key in ((4, slice(None)), (-5, slice(None)), (slice(None), 5)):
        with pytest.raises(IndexError, match="out of bounds"):
            _ = matrix[key]
    with pytest.raises(IndexError, match="out of bounds"):
        _ = matrix[[0, 4], :]
    with pytest.raises(IndexError, match="Boolean index"):
        _ = matrix[[True, False], :]
    with pytest.raises(IndexError, match="integer or Boolean"):
        _ = matrix[[0.0, 1.0], :]
    with pytest.raises(UnsupportedOperationError, match="one-dimensional"):
        _ = matrix[np.array([[0, 1]]), :]
    with pytest.raises(UnsupportedOperationError, match="Boolean scalar"):
        _ = matrix[True, :]
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        _ = matrix[::0, :]


@pytest.mark.parity
def test_index_plan_construction_does_not_collect_source(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(20.0).reshape(4, 5))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("index construction must not collect source values")

    backend.collect_matrix = forbidden  # type: ignore[method-assign]
    backend.collect_vector = forbidden  # type: ignore[method-assign]
    backend.collect_scalar = forbidden  # type: ignore[method-assign]
    result = matrix[[3, 1, 3], 2]
    assert result.shape == (3,)
    assert result.ndim == 1
    assert result.plan()["result"]["rank"] == 1
    assert "JOIN" in result.compile().upper()


@pytest.mark.parity
def test_large_gather_has_bounded_sql_and_host_selector_guard(
    backend: Backend,
) -> None:
    matrix = backend.from_numpy(np.arange(20.0).reshape(4, 5))
    selector = np.arange(100_000, dtype=np.int64) % 4
    result = matrix[selector, :]
    assert len(result.compile()) < 5_000
    nodes = result.plan()["nodes"]
    assert [node["node"] for node in nodes] == ["source", "gather"]

    backend.max_selector_values = 2
    with pytest.raises(DensificationError, match="max_selector_values"):
        _ = matrix[[0, 1, 2], :]


@pytest.mark.parity
def test_selector_resources_are_deduplicated_and_bounded(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(20.0).reshape(4, 5))
    backend.max_selector_relations = 1
    first = matrix[[3, 1, 3], :]
    second = matrix[np.array([3, 1, 3]), :]
    first_root = first.plan()["nodes"][-1]
    second_root = second.plan()["nodes"][-1]
    assert first_root["map_relation"] == second_root["map_relation"]
    assert len(backend._selector_cache) == 1  # noqa: SLF001 - lifecycle contract

    with pytest.raises(DensificationError, match="max_selector_relations"):
        _ = matrix[[2, 0], :]


@pytest.mark.parity
def test_deep_gather_plan_stays_iterative_and_reuses_selector_resource(
    backend: Backend,
) -> None:
    values = np.arange(8.0).reshape(2, 4)
    matrix = backend.from_numpy(values)
    for _ in range(50):
        matrix = matrix[[1, 0], :]

    assert len(matrix.plan()["nodes"]) == 51
    np.testing.assert_array_equal(matrix.to_numpy(), values)

    for _ in range(50):
        matrix = matrix[[1, 0], :]
    plan = matrix.plan()
    assert len(plan["nodes"]) == 101
    assert len(backend._selector_cache) == 1  # noqa: SLF001 - lifecycle contract


@pytest.mark.parity
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_seeded_indexing_differential(
    backend: Backend, seed: int, sparse_input: bool
) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(7, 9))
    values[rng.random(values.shape) < 0.55] = 0.0
    matrix = (
        backend.from_scipy(sparse.csr_array(values))
        if sparse_input
        else backend.from_numpy(values)
    )
    row_indices = rng.integers(-7, 7, size=6)
    col_indices = rng.integers(-9, 9, size=5)
    cases: tuple[Any, ...] = (
        (slice(None, None, -1), slice(8, None, -2)),
        (slice(6, 0, -2), slice(1, 9, 3)),
        (int(rng.integers(-7, 7)), slice(None)),
        (slice(None), int(rng.integers(-9, 9))),
        (int(rng.integers(-7, 7)), int(rng.integers(-9, 9))),
        (row_indices, slice(1, 8, 2)),
        (slice(1, 7, 2), col_indices),
        (row_indices, int(rng.integers(-9, 9))),
        (int(rng.integers(-7, 7)), col_indices),
    )
    for key in cases:
        result = matrix[key]
        expected = values[key]
        actual = _collect_index_result(result)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        assert result.shape == np.shape(expected)
