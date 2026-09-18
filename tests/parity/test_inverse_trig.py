from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend
from dbnumpy.exceptions import DensificationError
from dbnumpy.ir import StorageKind
from dbnumpy.matrix import DBArray

_VALUES = np.array(
    [
        [-np.inf, -2.0, -1.0, -0.0, 0.0],
        [0.5, 1.0, 2.0, np.inf, np.nan],
    ],
    dtype=np.float64,
)

_INVERSE_TRIG: tuple[tuple[str, str, np.ufunc], ...] = (
    ("arcsin", "asin", np.arcsin),
    ("arccos", "acos", np.arccos),
    ("arctan", "atan", np.arctan),
)


def _matrix(backend: Backend, *, sparse_input: bool) -> DBArray[Any]:
    if sparse_input:
        return backend.from_scipy(sparse.csr_array(_VALUES))
    return backend.from_numpy(_VALUES)


def _expected(operation: Callable[[Any], Any], values: Any = _VALUES) -> Any:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return operation(values)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    ("canonical_name", "alias_name", "operation"),
    _INVERSE_TRIG,
    ids=["arcsin", "arccos", "arctan"],
)
def test_inverse_trig_methods_aliases_and_numpy_ufuncs_match_numpy(
    backend: Backend,
    sparse_input: bool,
    canonical_name: str,
    alias_name: str,
    operation: np.ufunc,
) -> None:
    matrix = _matrix(backend, sparse_input=sparse_input)
    expected = _expected(operation)

    ufunc_result = operation(matrix)
    canonical_result = getattr(matrix, canonical_name)()
    alias_result = getattr(matrix, alias_name)()

    for result in (ufunc_result, canonical_result, alias_result):
        np.testing.assert_allclose(
            result.to_numpy(), expected, rtol=1.0e-15, atol=1.0e-15, equal_nan=True
        )

    np.testing.assert_allclose(
        canonical_result.to_numpy(),
        ufunc_result.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        alias_result.to_numpy(),
        ufunc_result.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method_name", "operation"),
    [("arcsin", np.arcsin), ("arccos", np.arccos)],
)
def test_inverse_trig_domain_errors_are_nan(
    backend: Backend, method_name: str, operation: np.ufunc
) -> None:
    actual = getattr(backend.from_numpy(_VALUES), method_name)().to_numpy()
    expected = _expected(operation)
    outside_domain = np.isinf(_VALUES) | (np.abs(_VALUES) > 1.0)

    assert np.isnan(actual[outside_domain]).all()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method_name", "operation"),
    [("arcsin", np.arcsin), ("arctan", np.arctan)],
)
def test_inverse_trig_preserves_dense_signed_zero(
    backend: Backend, method_name: str, operation: np.ufunc
) -> None:
    values = np.array([[-0.0, 0.0]], dtype=np.float64)
    actual = getattr(backend.from_numpy(values), method_name)().to_numpy()
    expected = operation(values)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parity
def test_inverse_trig_sparse_zero_image_controls_storage_and_values(
    backend: Backend,
) -> None:
    values = np.array([[0.0, -0.5, 0.0], [1.0, 0.0, -1.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))

    for method_name, operation in (("arcsin", np.arcsin), ("arctan", np.arctan)):
        result = getattr(matrix, method_name)()
        assert result.storage is StorageKind.SPARSE
        np.testing.assert_allclose(result.to_scipy().toarray(), operation(values))

    arccos_result = matrix.arccos()
    assert arccos_result.storage is StorageKind.DENSE
    np.testing.assert_allclose(arccos_result.to_numpy(), np.arccos(values))
    implicit_zero = values == 0.0
    np.testing.assert_allclose(
        arccos_result.to_numpy()[implicit_zero], np.full(3, np.pi / 2.0)
    )


@pytest.mark.parity
def test_arccos_sparse_densification_guard_fails_before_execution(
    backend: Backend,
) -> None:
    matrix = backend.from_scipy(sparse.csr_array(np.eye(4)))
    backend.max_densify_cells = 15

    with pytest.raises(DensificationError, match="enumerate 16 cells"):
        _ = matrix.arccos()

    # Zero-preserving inverse functions remain legal under the same guard.
    assert matrix.arcsin().storage is StorageKind.SPARSE
    assert matrix.arctan().storage is StorageKind.SPARSE


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method_name", "compiled_marker"),
    [("arcsin", "asin"), ("arccos", "acos"), ("arctan", "atan")],
)
def test_inverse_trig_semantic_and_compiled_plans_keep_operation_markers(
    backend: Backend, method_name: str, compiled_marker: str
) -> None:
    result = getattr(backend.from_numpy(np.array([[0.0, 0.5]])), method_name)()
    plan = result.plan()
    root = next(node for node in plan["nodes"] if node["id"] == plan["root"])

    assert root["node"] == "unary"
    assert root["op"] == method_name
    assert compiled_marker in result.compile().lower()


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method_name", "operation"),
    [
        ("arcsin", np.arcsin),
        ("arccos", np.arccos),
        ("arctan", np.arctan),
    ],
)
def test_inverse_trig_composes_after_a_distinct_source_join(
    backend: Backend,
    method_name: str,
    operation: np.ufunc,
) -> None:
    left_values = np.array([[0.25, 2.0], [-0.75, -2.0]])
    right_values = np.array([[0.25, 1.0], [-0.5, -1.0]])
    left = backend.from_numpy(left_values)
    right = backend.from_numpy(right_values)
    result = getattr(left + right, method_name)()

    expected = _expected(operation, left_values + right_values)
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)
    assert result.compile().upper().count("JOIN") >= 1
