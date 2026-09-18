from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from dbnumpy.backends import Backend
from dbnumpy.exceptions import DensificationError, UnsupportedOperationError
from dbnumpy.matrix import DBArray

_UNARY_OPERATIONS: tuple[
    tuple[str, Callable[[Any], Any], np.ndarray[Any, np.dtype[np.float64]]], ...
] = (
    (
        "sign",
        np.sign,
        np.array([[-np.inf, -2.75, -0.0, 0.0], [1.25, np.inf, np.nan, 0.0]]),
    ),
    (
        "trunc",
        np.trunc,
        np.array([[-np.inf, -2.75, -0.0, 0.0], [1.25, np.inf, np.nan, 0.0]]),
    ),
    (
        "isnan",
        np.isnan,
        np.array([[-np.inf, -2.75, -0.0, 0.0], [1.25, np.inf, np.nan, 0.0]]),
    ),
    (
        "log2",
        np.log2,
        np.array(
            [
                [-np.inf, -4.0, -0.0, 0.0, 0.25],
                [1.0, 2.0, 16.0, np.inf, np.nan],
            ]
        ),
    ),
    (
        "log10",
        np.log10,
        np.array(
            [
                [-np.inf, -10.0, -0.0, 0.0, 0.1],
                [1.0, 10.0, 100.0, np.inf, np.nan],
            ]
        ),
    ),
    (
        "sinh",
        np.sinh,
        np.array(
            [
                [-np.inf, -1000.0, -2.0, -0.0, 0.0],
                [0.5, 2.0, 1000.0, np.inf, np.nan],
            ]
        ),
    ),
    (
        "cosh",
        np.cosh,
        np.array(
            [
                [-np.inf, -1000.0, -2.0, -0.0, 0.0],
                [0.5, 2.0, 1000.0, np.inf, np.nan],
            ]
        ),
    ),
    (
        "tanh",
        np.tanh,
        np.array(
            [
                [-np.inf, -1000.0, -2.0, -0.0, 0.0],
                [0.5, 2.0, 1000.0, np.inf, np.nan],
            ]
        ),
    ),
)

_REDUCTIONS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("min", np.min),
    ("max", np.max),
    ("any", np.any),
    ("all", np.all),
    ("nansum", np.nansum),
    ("nanmean", np.nanmean),
    ("nanvar", np.nanvar),
    ("nanstd", np.nanstd),
    ("nanmin", np.nanmin),
    ("nanmax", np.nanmax),
)


def _matrix(
    backend: Backend,
    values: np.ndarray[Any, np.dtype[np.float64]],
    *,
    sparse_input: bool,
) -> DBArray[Any]:
    if sparse_input:
        return backend.from_scipy(sparse.csr_array(values))
    return backend.from_numpy(values)


def _numpy_call(operation: Callable[..., Any], values: Any, **kwargs: Any) -> Any:
    # NumPy deliberately warns for zero logarithms and all-NaN/empty slices.
    # Those values are part of the contract under test, not test-suite noise.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return operation(values, **kwargs)


def _assert_same(actual: Any, expected: Any) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape
    np.testing.assert_allclose(actual_array, expected_array, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_extended_unary_numpy_parity(backend: Backend, sparse_input: bool) -> None:
    for name, operation, values in _UNARY_OPERATIONS:
        matrix = _matrix(backend, values, sparse_input=sparse_input)
        actual = operation(matrix).to_numpy()
        expected = _numpy_call(operation, values)
        _assert_same(actual, expected)
        _assert_same(getattr(matrix, name)().to_numpy(), expected)
        if name == "isnan":
            assert actual.dtype == np.dtype(np.float64)

        # Dense ingestion can preserve IEEE negative zero.  Check its sign bit
        # explicitly because numerical equality considers -0.0 and 0.0 equal.
        if not sparse_input and name in {"sign", "trunc", "sinh", "tanh"}:
            zero = expected == 0.0
            np.testing.assert_array_equal(
                np.signbit(actual[zero]), np.signbit(expected[zero])
            )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_extended_unary_composes_after_relational_barriers(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[0.0, -2.0, 4.0], [1.5, 0.0, np.nan]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)
    relational = matrix.T[1:, :]
    expected_input = values.T[1:, :]
    for _, operation, _ in _UNARY_OPERATIONS:
        _assert_same(
            operation(relational).to_numpy(),
            _numpy_call(operation, expected_input),
        )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_isnan_composes_with_reductions(backend: Backend, sparse_input: bool) -> None:
    values = np.array([[0.0, np.nan, 2.0], [np.nan, 0.0, -1.0]])
    flags = np.isnan(_matrix(backend, values, sparse_input=sparse_input))
    expected = np.isnan(values).astype(np.float64)

    _assert_same(flags.sum(axis=0), expected.sum(axis=0))
    np.testing.assert_array_equal(flags.any(axis=1), expected.any(axis=1))
    assert flags.all() == expected.all()


@pytest.mark.parity
def test_extended_unary_materialization_remains_composable(backend: Backend) -> None:
    values = np.array([[0.0, -2.0, 0.0], [1.5, np.nan, 3.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))
    materialized = np.tanh(matrix).compute()
    expected = np.tanh(values)

    _assert_same(materialized.to_numpy(), expected)
    _assert_same(materialized.nanmax(axis=0), np.nanmax(expected, axis=0))


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_extended_reduction_methods_and_numpy_dispatch(
    backend: Backend, sparse_input: bool
) -> None:
    # Several zeros are absent from CSR storage, so each reduction must account
    # for the full logical shape rather than only explicit coordinate rows.
    values = np.array(
        [
            [0.0, np.nan, -2.0, 0.0],
            [0.0, 5.0, 0.0, 4.0],
            [0.0, np.nan, 3.0, 0.0],
        ]
    )
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name, numpy_operation in _REDUCTIONS:
        method = getattr(matrix, name)
        for axis in (None, 0, 1):
            expected = _numpy_call(numpy_operation, values, axis=axis)
            actual = method(axis=axis)
            _assert_same(actual, expected)
            if name in {"any", "all"}:
                assert np.asarray(actual).dtype == np.dtype(bool)
            else:
                assert np.asarray(actual).dtype == np.dtype(np.float64)

            expected_keepdims = _numpy_call(
                numpy_operation, values, axis=axis, keepdims=True
            )
            actual_keepdims = numpy_operation(matrix, axis=axis, keepdims=True)
            _assert_same(actual_keepdims, expected_keepdims)
            if name in {"any", "all"}:
                assert np.asarray(actual_keepdims).dtype == np.dtype(bool)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nan_variance_ddof_parity(backend: Backend, sparse_input: bool) -> None:
    values = np.array([[0.0, np.nan, 2.0], [3.0, 4.0, np.nan], [0.0, 6.0, 8.0]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name, operation in (("nanvar", np.nanvar), ("nanstd", np.nanstd)):
        for axis in (None, 0, 1):
            expected = _numpy_call(operation, values, axis=axis, ddof=1)
            _assert_same(getattr(matrix, name)(axis=axis, ddof=1), expected)
            _assert_same(
                operation(matrix, axis=axis, ddof=1),
                expected,
            )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_fractional_ddof_and_correction_are_not_truncated(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[0.0, np.nan, 2.0], [3.0, 4.0, 5.0]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    _assert_same(matrix.var(ddof=0.5), np.var(values, ddof=0.5))
    _assert_same(matrix.std(correction=0.5), np.std(values, correction=0.5))
    _assert_same(matrix.nanvar(ddof=0.5), np.nanvar(values, ddof=0.5))
    _assert_same(
        np.nanstd(matrix, correction=0.5),
        np.nanstd(values, correction=0.5),
    )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    ("name", "operation", "skip_nan"),
    [
        ("var", np.var, False),
        ("std", np.std, False),
        ("nanvar", np.nanvar, True),
        ("nanstd", np.nanstd, True),
    ],
)
def test_variance_ddof_at_or_above_effective_count_matches_numpy(
    backend: Backend,
    sparse_input: bool,
    name: str,
    operation: Callable[..., Any],
    skip_nan: bool,
) -> None:
    values = np.array(
        [[0.0, 2.0, np.nan], [0.0, 0.0, np.nan]]
        if skip_nan
        else [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
    )
    matrix = _matrix(backend, values, sparse_input=sparse_input)
    axis_count = 2 if skip_nan else 3
    total_count = axis_count * 2

    for ddof in (axis_count, axis_count + 0.5):
        _assert_same(
            getattr(matrix, name)(axis=1, ddof=ddof),
            _numpy_call(operation, values, axis=1, ddof=ddof),
        )
    for ddof in (total_count, total_count + 0.5):
        _assert_same(
            getattr(matrix, name)(ddof=ddof),
            _numpy_call(operation, values, ddof=ddof),
        )


@pytest.mark.parity
def test_extended_reduction_negative_axis_spots(backend: Backend) -> None:
    values = np.array([[0.0, np.nan, 2.0], [-1.0, 4.0, 0.0]])
    matrix = backend.from_scipy(sparse.csr_array(values))

    for name, operation, axis in (
        ("min", np.min, -2),
        ("any", np.any, -1),
        ("nanmean", np.nanmean, -2),
        ("nanmax", np.nanmax, -1),
    ):
        expected = _numpy_call(operation, values, axis=axis, keepdims=True)
        _assert_same(getattr(matrix, name)(axis=axis, keepdims=True), expected)
        _assert_same(operation(matrix, axis=axis, keepdims=True), expected)

    _assert_same(np.amin(matrix, axis=0), np.amin(values, axis=0))
    _assert_same(np.amax(matrix, axis=1), np.amax(values, axis=1))


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nan_reductions_on_all_nan_slices(backend: Backend, sparse_input: bool) -> None:
    values = np.full((2, 3), np.nan)
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name, operation in _REDUCTIONS[4:]:
        for axis in (None, 0, 1):
            expected = _numpy_call(operation, values, axis=axis)
            _assert_same(getattr(matrix, name)(axis=axis), expected)
            _assert_same(operation(matrix, axis=axis), expected)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_extrema_with_infinities_and_nan(backend: Backend, sparse_input: bool) -> None:
    values = np.array([[0.0, np.inf, np.nan], [-np.inf, 2.0, 0.0]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name, operation in (
        ("min", np.min),
        ("max", np.max),
        ("nanmin", np.nanmin),
        ("nanmax", np.nanmax),
    ):
        for axis in (None, 0, 1):
            _assert_same(
                getattr(matrix, name)(axis=axis),
                _numpy_call(operation, values, axis=axis),
            )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_extrema_preserve_singleton_infinities(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[-np.inf, np.inf, -np.inf], [np.inf, -np.inf, np.inf]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name, operation in (
        ("min", np.min),
        ("max", np.max),
        ("nanmin", np.nanmin),
        ("nanmax", np.nanmax),
    ):
        for axis in (None, 0, 1):
            _assert_same(
                getattr(matrix, name)(axis=axis),
                operation(values, axis=axis),
            )


@pytest.mark.parity
def test_extrema_use_deterministic_signed_zero_ties(
    backend: Backend,
) -> None:
    values = np.array([[0.0, -0.0], [-0.0, 0.0]])
    # Canonical sparse ingestion intentionally removes every explicit zero and
    # therefore cannot retain a negative-zero sign bit. Dense storage does.
    matrix = backend.from_numpy(values)

    for name, operation in (
        ("min", np.min),
        ("max", np.max),
        ("nanmin", np.nanmin),
        ("nanmax", np.nanmax),
    ):
        for axis in (None, 0, 1):
            actual = np.asarray(getattr(matrix, name)(axis=axis))
            expected = np.asarray(operation(values, axis=axis))
            np.testing.assert_array_equal(actual, expected)
            # NumPy does not document which equal zero operand wins and its
            # reduction sign differs between SIMD/platform builds. DBVerse uses
            # one deterministic IEEE-style tie: min prefers -0, max prefers +0.
            expected_sign = name in {"min", "nanmin"}
            np.testing.assert_array_equal(
                np.signbit(actual),
                np.full(actual.shape, expected_sign, dtype=np.bool_),
            )


@pytest.mark.parity
def test_sparse_lazy_negative_zero_participates_in_extrema(
    backend: Backend,
) -> None:
    values = np.array([[-0.5, 0.0], [0.0, 0.5]])
    matrix = np.trunc(backend.from_scipy(sparse.csr_array(values)))
    expected = np.trunc(values)

    for name, operation in (
        ("min", np.min),
        ("max", np.max),
        ("nanmin", np.nanmin),
        ("nanmax", np.nanmax),
    ):
        for axis in (None, 0, 1):
            actual = np.asarray(getattr(matrix, name)(axis=axis))
            reference = np.asarray(operation(expected, axis=axis))
            np.testing.assert_array_equal(actual, reference)
            signbits = np.signbit(expected) & (expected == 0.0)
            if name in {"min", "nanmin"}:
                expected_sign = np.any(signbits, axis=axis)
            else:
                positive_zero = ~np.signbit(expected) & (expected == 0.0)
                expected_sign = ~np.any(positive_zero, axis=axis)
            np.testing.assert_array_equal(np.signbit(actual), expected_sign)


@pytest.mark.parity
def test_sparse_extrema_fold_implicit_zero_only_when_present(backend: Backend) -> None:
    positive = np.array([[2.0, 0.0], [5.0, 3.0]])
    negative = np.array([[-2.0, 0.0], [-5.0, -3.0]])
    positive_matrix = backend.from_scipy(sparse.csr_array(positive))
    negative_matrix = backend.from_scipy(sparse.csr_array(negative))

    for name, operation, matrix, values in (
        ("min", np.min, positive_matrix, positive),
        ("nanmin", np.nanmin, positive_matrix, positive),
        ("max", np.max, negative_matrix, negative),
        ("nanmax", np.nanmax, negative_matrix, negative),
    ):
        for axis in (None, 0, 1):
            _assert_same(
                getattr(matrix, name)(axis=axis),
                operation(values, axis=axis),
            )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nan_reductions_with_opposite_infinities(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[np.inf, np.nan, 0.0], [-np.inf, 2.0, 0.0]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)
    for name, operation in (
        ("nansum", np.nansum),
        ("nanmean", np.nanmean),
        ("nanvar", np.nanvar),
        ("nanstd", np.nanstd),
        ("nanmin", np.nanmin),
        ("nanmax", np.nanmax),
    ):
        for axis in (None, 0, 1):
            _assert_same(
                getattr(matrix, name)(axis=axis),
                _numpy_call(operation, values, axis=axis),
            )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nan_variance_is_stable_for_large_offsets(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array(
        [
            [1.0e12 + 1.0, np.nan, 1.0e12 + 3.0],
            [1.0e12 + 4.0, 1.0e12 + 5.0, 1.0e12 + 6.0],
        ]
    )
    matrix = _matrix(backend, values, sparse_input=sparse_input)
    # The lowering subtracts an explicit finite anchor before scaling, so the
    # small spread is retained even when the common offset is large.
    for axis in (None, 0, 1):
        np.testing.assert_allclose(
            matrix.nanvar(axis=axis),
            _numpy_call(np.nanvar, values, axis=axis),
            rtol=2.0e-6,
            atol=2.0e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            matrix.nanstd(axis=axis),
            _numpy_call(np.nanstd, values, axis=axis),
            rtol=2.0e-6,
            atol=2.0e-6,
            equal_nan=True,
        )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_variance_does_not_overflow_for_large_constant_values(
    backend: Backend, sparse_input: bool
) -> None:
    # Two equal values keep NumPy's reference mean exact while still making
    # the old ``mean**2 * implicit_count`` merge overflow as ``inf * 0``.
    values = np.full((1, 2), 1.0e200)
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name, operation in (
        ("var", np.var),
        ("std", np.std),
        ("nanvar", np.nanvar),
        ("nanstd", np.nanstd),
    ):
        for axis in (None, 0, 1):
            _assert_same(
                getattr(matrix, name)(axis=axis),
                operation(values, axis=axis),
            )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_variance_retains_a_one_ulp_spread_at_huge_magnitude(
    backend: Backend, sparse_input: bool
) -> None:
    lower = np.float64(1.0e160)
    upper = np.nextafter(lower, np.inf)
    values = np.array([[lower, upper]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)
    half_delta = (upper - lower) / 2.0
    expected_variance = half_delta * half_delta

    # NumPy's rounded intermediate mean makes np.var twice the mathematical
    # population variance for this exact two-point input. The relational
    # lowering centers the representable delta instead and stays nonzero.
    np.testing.assert_allclose(matrix.var(), expected_variance, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.nanvar(), expected_variance, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.std(), half_delta, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.nanstd(), half_delta, rtol=2.0e-15)


@pytest.mark.parity
def test_variance_avoids_intermediate_overflow_for_opposite_large_values(
    backend: Backend,
) -> None:
    values = np.array([[-1.0e154, 1.0e154]])
    matrix = backend.from_numpy(values)

    # Squaring and summing the original values overflows in NumPy, although
    # the population variance and standard deviation remain representable.
    np.testing.assert_allclose(matrix.var(), 1.0e308, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.nanvar(), 1.0e308, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.std(), 1.0e154, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.nanstd(), 1.0e154, rtol=2.0e-15)


@pytest.mark.parity
def test_standard_deviation_avoids_intermediate_variance_underflow(
    backend: Backend,
) -> None:
    values = np.array([[-1.0e-200, 1.0e-200]])
    matrix = backend.from_numpy(values)

    # NumPy forms the variance first, which underflows to zero before sqrt.
    # The lowering rescales standard deviation directly, while float64
    # variance itself still has the expected representational zero.
    assert matrix.var() == 0.0
    assert matrix.nanvar() == 0.0
    np.testing.assert_allclose(matrix.std(), 1.0e-200, rtol=2.0e-15)
    np.testing.assert_allclose(matrix.nanstd(), 1.0e-200, rtol=2.0e-15)


@pytest.mark.parity
def test_variance_scales_before_squaring_huge_sparse_outlier(
    backend: Backend,
) -> None:
    dimension = 1_000_000_000
    logical_count = float(dimension * dimension)
    outlier = 1.0e163
    matrix = backend.from_coo([0], [0], [outlier], shape=(dimension, dimension))
    expected_variance = (outlier / np.sqrt(logical_count)) ** 2 * (
        (logical_count - 1.0) / logical_count
    )

    assert np.isfinite(expected_variance)
    np.testing.assert_allclose(matrix.var(), expected_variance, rtol=1.0e-15)
    np.testing.assert_allclose(matrix.nanvar(), expected_variance, rtol=1.0e-15)
    np.testing.assert_allclose(matrix.std(), np.sqrt(expected_variance), rtol=1.0e-15)
    np.testing.assert_allclose(
        matrix.nanstd(), np.sqrt(expected_variance), rtol=1.0e-15
    )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_boolean_reductions_treat_nan_and_infinity_as_truthy(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.array([[np.nan, np.inf, -2.0], [1.0, 2.0, 3.0]])
    matrix = _matrix(backend, values, sparse_input=sparse_input)
    for axis in (None, 0, 1):
        np.testing.assert_array_equal(matrix.any(axis=axis), np.any(values, axis=axis))
        np.testing.assert_array_equal(matrix.all(axis=axis), np.all(values, axis=axis))

    with_implicit_zero = backend.from_scipy(
        sparse.csr_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
    )
    np.testing.assert_array_equal(
        with_implicit_zero.all(axis=1), np.array([False, True])
    )


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_empty_reduction_extrema_and_identities(
    backend: Backend, sparse_input: bool
) -> None:
    values = np.empty((0, 3), dtype=np.float64)
    matrix = _matrix(backend, values, sparse_input=sparse_input)

    for name in ("min", "max", "nanmin", "nanmax"):
        with pytest.raises(ValueError):
            getattr(matrix, name)()
        with pytest.raises(ValueError):
            getattr(matrix, name)(axis=0)
        _assert_same(getattr(matrix, name)(axis=1), np.empty(0))

    for name, operation in (
        ("any", np.any),
        ("all", np.all),
        ("nansum", np.nansum),
        ("nanmean", np.nanmean),
        ("nanvar", np.nanvar),
        ("nanstd", np.nanstd),
    ):
        for axis in (None, 0, 1):
            _assert_same(
                getattr(matrix, name)(axis=axis),
                _numpy_call(operation, values, axis=axis),
            )

    transposed_values = np.empty((3, 0), dtype=np.float64)
    transposed = _matrix(backend, transposed_values, sparse_input=sparse_input)
    for name in ("min", "max", "nanmin", "nanmax"):
        with pytest.raises(ValueError):
            getattr(transposed, name)()
        _assert_same(getattr(transposed, name)(axis=0), np.empty(0))
        with pytest.raises(ValueError):
            getattr(transposed, name)(axis=1)

    empty_square_values = np.empty((0, 0), dtype=np.float64)
    empty_square = _matrix(
        backend,
        empty_square_values,
        sparse_input=sparse_input,
    )
    for name in ("min", "max", "nanmin", "nanmax"):
        with pytest.raises(ValueError):
            getattr(empty_square, name)()
        with pytest.raises(ValueError):
            getattr(empty_square, name)(axis=0)
        with pytest.raises(ValueError):
            getattr(empty_square, name)(axis=1)


@pytest.mark.parity
def test_unsupported_reduction_options_do_not_execute(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))

    with patch.object(
        backend,
        "collect_reduction",
        side_effect=AssertionError("unsupported options must fail before execution"),
    ):
        with pytest.raises(UnsupportedOperationError, match="out="):
            matrix.min(out=np.empty(2))
        with pytest.raises(UnsupportedOperationError, match="initial="):
            matrix.max(initial=0.0)
        with pytest.raises(UnsupportedOperationError, match="where="):
            matrix.any(where=False)
        with pytest.raises(UnsupportedOperationError, match="float64"):
            matrix.nanmean(dtype=np.float32)
        with pytest.raises(UnsupportedOperationError, match="precomputed mean"):
            matrix.nanvar(mean=np.zeros(3))
        with pytest.raises(UnsupportedOperationError, match="initial="):
            np.nanmax(matrix, initial=0.0)
        with pytest.raises(ValueError, match="ddof must be nonnegative"):
            matrix.var(ddof=-0.5)
        with pytest.raises(ValueError, match="ddof must be nonnegative"):
            matrix.nanstd(correction=-1)
        with pytest.raises(ValueError, match="ddof must be finite"):
            matrix.var(ddof=np.inf)
        with pytest.raises(ValueError, match="ddof must be finite"):
            matrix.nanvar(correction=np.nan)
        for invalid_ddof in ("1", None, 1 + 0j, np.array(0.5)):
            with pytest.raises(TypeError, match="ddof must be a real number"):
                matrix.var(ddof=invalid_ddof)


@pytest.mark.parity
@pytest.mark.parametrize("axis", [0.0, True, np.bool_(False)])
def test_invalid_axis_types_do_not_execute(backend: Backend, axis: Any) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))

    with patch.object(
        backend,
        "collect_reduction",
        side_effect=AssertionError("invalid axes must fail before execution"),
    ):
        for name, _ in _REDUCTIONS:
            with pytest.raises(TypeError, match="axis must be an integer"):
                getattr(matrix, name)(axis=axis)


@pytest.mark.parity
@pytest.mark.parametrize("axis", [2, -3])
def test_out_of_range_axes_use_numpy_axis_error(backend: Backend, axis: int) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))

    with patch.object(
        backend,
        "collect_reduction",
        side_effect=AssertionError("out-of-range axes must fail before execution"),
    ):
        with pytest.raises(np.exceptions.AxisError, match="out of bounds"):
            matrix.nanmean(axis=axis)
        with pytest.raises(np.exceptions.AxisError, match="out of bounds"):
            np.nanmean(matrix, axis=axis)


@pytest.mark.parity
@pytest.mark.parametrize(
    "keepdims", ["yes", None, 1.0, np.bool_(True), np.array([True])]
)
def test_invalid_keepdims_does_not_execute(backend: Backend, keepdims: Any) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))

    with patch.object(
        backend,
        "collect_reduction",
        side_effect=AssertionError("invalid keepdims must fail before execution"),
    ):
        for name in ("sum", "any", "nanmax"):
            with pytest.raises(TypeError, match="keepdims must be an integer"):
                getattr(matrix, name)(keepdims=keepdims)
        for operation in (np.sum, np.any, np.nanmax):
            with pytest.raises(TypeError, match="keepdims must be an integer"):
                operation(matrix, keepdims=keepdims)


@pytest.mark.parity
@pytest.mark.parametrize("keepdims", [2, -1, np.int64(1), np.array(1)])
def test_integer_keepdims_matches_numpy(backend: Backend, keepdims: Any) -> None:
    values = np.arange(6.0).reshape(2, 3)
    matrix = backend.from_numpy(values)

    _assert_same(
        matrix.nanmax(axis=0, keepdims=keepdims),
        np.nanmax(values, axis=0, keepdims=keepdims),
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    "where", [True, np.bool_(True), np.array(True), 1, 2.0, "selected"]
)
def test_scalar_true_where_matches_full_reduction(backend: Backend, where: Any) -> None:
    values = np.arange(6.0).reshape(2, 3)
    matrix = backend.from_numpy(values)

    _assert_same(matrix.sum(where=where), np.sum(values, where=where))
    _assert_same(np.nanmean(matrix, where=where), np.nanmean(values, where=where))


@pytest.mark.parity
def test_extended_reductions_are_advertised(backend: Backend) -> None:
    expected = {name for name, _ in _REDUCTIONS}
    assert all(backend.capabilities.supports_reduction(name) for name in expected)
    assert backend.capabilities.supports_node("unary")


@pytest.mark.parity
def test_huge_all_zero_sparse_scalar_reductions_do_not_enumerate(
    backend: Backend,
) -> None:
    matrix = backend.from_coo([], [], [], shape=(1_000_000, 1_000_000))
    expected = {
        "min": 0.0,
        "max": 0.0,
        "any": False,
        "all": False,
        "nansum": 0.0,
        "nanmean": 0.0,
        "nanvar": 0.0,
        "nanstd": 0.0,
        "nanmin": 0.0,
        "nanmax": 0.0,
    }
    for name, value in expected.items():
        assert getattr(matrix, name)() == value
    assert "CROSS JOIN" not in matrix.compile().upper()


@pytest.mark.parity
def test_scalar_reductions_support_logical_counts_above_int64(
    backend: Backend,
) -> None:
    matrix = backend.from_coo([], [], [], shape=(10_000_000_000, 10_000_000_000))
    assert matrix.size > np.iinfo(np.int64).max
    expected = {
        "sum": 0.0,
        "mean": 0.0,
        "var": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "any": False,
        "all": False,
        "nansum": 0.0,
        "nanmean": 0.0,
        "nanvar": 0.0,
        "nanstd": 0.0,
        "nanmin": 0.0,
        "nanmax": 0.0,
    }

    for name, value in expected.items():
        assert getattr(matrix, name)() == value


@pytest.mark.parity
def test_variance_ddof_boundary_above_exact_float_integer_range(
    backend: Backend,
) -> None:
    logical_count = 2**53 + 1
    matrix = backend.from_coo([], [], [], shape=(1, logical_count))

    for name in ("var", "std", "nanvar", "nanstd"):
        assert getattr(matrix, name)(ddof=logical_count - 1) == 0.0

    larger_count = 2**53 + 2
    one = backend.from_coo([0], [0], [1.0], shape=(1, larger_count))
    expected_variance = (larger_count - 1) / larger_count
    np.testing.assert_allclose(one.var(ddof=larger_count - 1), expected_variance)
    np.testing.assert_allclose(one.nanvar(ddof=larger_count - 1), expected_variance)
    np.testing.assert_allclose(
        one.std(ddof=larger_count - 1), np.sqrt(expected_variance)
    )
    np.testing.assert_allclose(
        one.nanstd(correction=larger_count - 1), np.sqrt(expected_variance)
    )


@pytest.mark.parity
def test_extended_axis_reduction_respects_host_output_guard(backend: Backend) -> None:
    assert backend.max_host_values >= 10_000_000
    backend.max_densify_cells = 3
    matrix = backend.from_coo([], [], [], shape=(4, 5))
    _assert_same(matrix.nanmax(axis=1), np.zeros(4))

    backend.max_host_values = 3
    with pytest.raises(DensificationError, match="max_host_values=3"):
        matrix.nanmax(axis=1)


@pytest.mark.parity
def test_unsupported_numpy_dispatch_never_collects(backend: Backend) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    with patch.object(
        backend,
        "collect_matrix",
        side_effect=AssertionError("unsupported NumPy dispatch must not collect"),
    ):
        with pytest.raises(TypeError):
            np.median(matrix)
        with pytest.raises(TypeError):
            np.add.reduce(matrix)


@pytest.mark.parity
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_seeded_random_extended_reduction_parity(
    backend: Backend, sparse_input: bool
) -> None:
    rng = np.random.default_rng(20260711)
    operations = (
        ("sum", np.sum),
        ("mean", np.mean),
        ("var", np.var),
        ("std", np.std),
        *_REDUCTIONS,
    )
    special_values = np.array(
        [0.0, -0.0, 1.0, -1.0, 3.5, -2.25, np.nan, np.inf, -np.inf]
    )

    for case, shape in enumerate(((1, 5), (5, 1), (3, 4), (5, 5))):
        values = rng.normal(size=shape)
        special = rng.random(size=shape) < 0.4
        values[special] = rng.choice(special_values, size=int(special.sum()))
        matrix = _matrix(backend, values, sparse_input=sparse_input)

        for name, operation in operations:
            kwargs: dict[str, Any] = {}
            if name in {"var", "std", "nanvar", "nanstd"}:
                kwargs["ddof"] = (case % 3) * 0.5
            for axis in (None, 0, 1):
                expected = _numpy_call(operation, values, axis=axis, **kwargs)
                actual = getattr(matrix, name)(axis=axis, **kwargs)
                if name in {"any", "all"}:
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=3.0e-6,
                        atol=3.0e-6,
                        equal_nan=True,
                    )
