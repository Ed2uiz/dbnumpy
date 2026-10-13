"""Reductions match NumPy, count implicit sparse zeros, and validate options."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from dbnumpy import DensificationError, UnsupportedOperationError
from dbnumpy.backends import Backend

pytestmark = pytest.mark.parity

REDUCTIONS = [
    "sum",
    "mean",
    "var",
    "std",
    "min",
    "max",
    "any",
    "all",
    "nansum",
    "nanmean",
    "nanvar",
    "nanstd",
    "nanmin",
    "nanmax",
]
VARIANCES = {"var", "std", "nanvar", "nanstd"}
EXTREMA = {"min", "max", "nanmin", "nanmax"}

inf, nan = np.inf, np.nan
VALUE_SETS = {
    "implicit_zeros": [
        [0.0, nan, -2.0, 0.0],
        [0.0, 5.0, 0.0, 4.0],
        [0.0, nan, 3.0, 0.0],
    ],
    "nonfinite": [[inf, nan, 0.0, -2.25], [-inf, 2.0, 0.0, 3.5]],
    "infinities": [[-inf, inf, -inf], [inf, -inf, inf]],
    "all_nan": [[nan, nan, nan], [nan, nan, nan]],
    "all_zero": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "positive": [[2.0, 0.0], [5.0, 3.0]],
    "negative": [[-2.0, 0.0], [-5.0, -3.0]],
    "large_offset": [[1e12, 1e12 + 1, 1e12 + 2], [1e12 + 3, nan, 1e12 + 5]],
}


def numpy_reduce(name: str, values: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return getattr(np, name)(values, **kwargs)


def assert_reduction(actual: Any, expected: Any, name: str, **kwargs: Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    msg = f"{name}({kwargs})"
    assert actual.shape == expected.shape, msg
    assert actual.dtype == (bool if name in {"any", "all"} else np.float64), msg
    np.testing.assert_allclose(actual, expected, rtol=1e-9, equal_nan=True, err_msg=msg)


@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("value_set", VALUE_SETS)
def test_reductions(upload: Any, value_set: str, sparse_input: bool) -> None:
    matrix, values = upload(VALUE_SETS[value_set], sparse_input=sparse_input)
    for name in REDUCTIONS:
        for axis in (None, 0, 1, -1):
            assert_reduction(
                getattr(matrix, name)(axis=axis),
                numpy_reduce(name, values, axis=axis),
                name,
                axis=axis,
            )
        # NumPy dispatch and keepdims share the method implementation.
        for axis in (None, -2):
            assert_reduction(
                getattr(np, name)(matrix, axis=axis, keepdims=True),
                numpy_reduce(name, values, axis=axis, keepdims=True),
                name,
                axis=axis,
                keepdims=True,
            )
    for name in VARIANCES:
        for ddof in (1, 0.5, 2, 6.5):
            for axis in (None, 1):
                assert_reduction(
                    getattr(matrix, name)(axis=axis, ddof=ddof),
                    numpy_reduce(name, values, axis=axis, ddof=ddof),
                    name,
                    axis=axis,
                    ddof=ddof,
                )
        assert_reduction(
            getattr(matrix, name)(correction=0.5),
            numpy_reduce(name, values, ddof=0.5),
            name,
        )


def huge_sparse_outlier(backend: Backend) -> tuple[Any, float, float]:
    count = float(10**18)
    matrix = backend.from_coo([0], [0], [1e163], shape=(10**9, 10**9))
    variance = (1e163 / np.sqrt(count)) ** 2 * ((count - 1.0) / count)
    return matrix, variance, np.sqrt(variance)


ONE_ULP = np.nextafter(1e160, inf) - 1e160
# values, expected variance, expected standard deviation. Where NumPy loses
# precision (rounded mean, overflow, underflow), the exact result is used.
STABILITY_CASES = {
    "large_constant": ([[1e200, 1e200]], 0.0, 0.0),
    "one_ulp_spread": ([[1e160, 1e160 + ONE_ULP]], (ONE_ULP / 2) ** 2, ONE_ULP / 2),
    "opposite_large": ([[-1e154, 1e154]], 1e308, 1e154),
    "variance_underflow": ([[-1e-200, 1e-200]], 0.0, 1e-200),
    "large_offset": (
        1e12 + np.arange(9.0).reshape(3, 3),
        np.var(np.arange(9.0)),
        np.std(np.arange(9.0)),
    ),
    "huge_sparse_outlier": (huge_sparse_outlier, None, None),
}


@pytest.mark.parametrize("case", STABILITY_CASES)
def test_variance_is_numerically_stable(
    backend: Backend, upload: Any, case: str
) -> None:
    values, variance, std = STABILITY_CASES[case]
    if callable(values):
        matrix, variance, std = values(backend)
    else:
        matrix, _ = upload(values, sparse_input=False)
    for name, expected in (
        ("var", variance),
        ("nanvar", variance),
        ("std", std),
        ("nanstd", std),
    ):
        np.testing.assert_allclose(
            getattr(matrix, name)(), expected, rtol=2e-15, err_msg=name
        )


def test_extrema_signed_zero_ties(backend: Backend) -> None:
    # Min prefers -0.0 and max prefers +0.0, including lazily produced -0.0.
    dense_values = np.array([[0.0, -0.0], [-0.0, 0.0]])
    cases = [
        (backend.from_numpy(dense_values), dense_values),
        (
            np.trunc(backend.from_coo([0], [0], [-0.5], shape=(2, 2))),
            np.array([[-0.0, 0.0], [0.0, 0.0]]),
        ),
    ]
    for matrix, values in cases:
        for name in EXTREMA:
            for axis in (None, 0, 1):
                actual = np.asarray(getattr(matrix, name)(axis=axis))
                if name in {"min", "nanmin"}:
                    negative = np.any(np.signbit(values), axis=axis)
                else:
                    negative = ~np.any(~np.signbit(values), axis=axis)
                assert (actual == 0.0).all()
                np.testing.assert_array_equal(
                    np.signbit(actual), negative, err_msg=name
                )


@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_reductions(
    upload: Any, shape: tuple[int, int], sparse_input: bool
) -> None:
    matrix, values = upload(np.empty(shape), sparse_input=sparse_input)
    assert matrix.T.to_numpy().shape == shape[::-1]
    for name in REDUCTIONS:
        for axis in (None, 0, 1):
            if name in EXTREMA and (axis is None or shape[axis] == 0):
                with pytest.raises(ValueError, match="zero-size"):
                    getattr(matrix, name)(axis=axis)
            else:
                assert_reduction(
                    getattr(matrix, name)(axis=axis),
                    numpy_reduce(name, values, axis=axis),
                    name,
                    axis=axis,
                )


def test_huge_all_zero_matrices_reduce_without_enumeration(backend: Backend) -> None:
    matrix = backend.from_coo([], [], [], shape=(10**10, 10**10))
    assert matrix.size > np.iinfo(np.int64).max
    for name in REDUCTIONS:
        assert getattr(matrix, name)() == (False if name in {"any", "all"} else 0.0)
    assert "CROSS JOIN" not in matrix.compile().upper()

    # Logical counts above 2**53 are not rounded when ddof is subtracted.
    count = 2**53 + 2
    one = backend.from_coo([0], [0], [1.0], shape=(1, count))
    np.testing.assert_allclose(one.var(ddof=count - 1), (count - 1) / count)
    assert backend.from_coo([], [], [], shape=(1, count - 1)).var(ddof=count - 2) == 0.0


def test_axis_reductions_respect_host_output_limit(backend: Backend) -> None:
    matrix = backend.from_coo([], [], [], shape=(4, 5))
    backend.max_host_values = 3
    with pytest.raises(DensificationError, match="max_host_values=3"):
        matrix.nanmax(axis=1)


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda x: x.min(out=np.empty(2)), UnsupportedOperationError, "out="),
        (lambda x: x.max(initial=0.0), UnsupportedOperationError, "initial="),
        (lambda x: np.nanmax(x, initial=0.0), UnsupportedOperationError, "initial="),
        (lambda x: x.any(where=False), UnsupportedOperationError, "where="),
        (lambda x: x.nanmean(dtype=np.float32), UnsupportedOperationError, "float64"),
        (
            lambda x: x.nanvar(mean=np.zeros(3)),
            UnsupportedOperationError,
            "precomputed mean",
        ),
        (lambda x: x.var(ddof=-0.5), ValueError, "ddof must be nonnegative"),
        (lambda x: x.nanstd(correction=-1), ValueError, "ddof must be nonnegative"),
        (lambda x: x.var(ddof=np.inf), ValueError, "ddof must be finite"),
        (lambda x: x.var(ddof="1"), TypeError, "ddof must be a real number"),
        (lambda x: x.var(ddof=np.array(0.5)), TypeError, "ddof must be a real number"),
        (lambda x: x.sum(axis=0.0), TypeError, "axis must be an integer"),
        (lambda x: x.sum(axis=True), TypeError, "axis must be an integer"),
        (lambda x: x.nanmean(axis=2), np.exceptions.AxisError, "out of bounds"),
        (lambda x: np.nanmean(x, axis=-3), np.exceptions.AxisError, "out of bounds"),
        (lambda x: x.sum(keepdims="yes"), TypeError, "keepdims must be an integer"),
        (
            lambda x: np.any(x, keepdims=np.bool_(True)),
            TypeError,
            "keepdims must be an integer",
        ),
        (lambda x: np.median(x), TypeError, None),
        (lambda x: np.add.reduce(x), TypeError, None),
    ],
)
def test_invalid_reduction_options(
    backend: Backend, call: Any, error: type, match: str | None
) -> None:
    matrix = backend.from_numpy(np.arange(6.0).reshape(2, 3))
    with pytest.raises(error, match=match):
        call(matrix)


def test_accepted_reduction_option_spellings(backend: Backend) -> None:
    values = np.arange(6.0).reshape(2, 3)
    matrix = backend.from_numpy(values)
    for keepdims in (2, -1, np.int64(1), np.array(1)):
        np.testing.assert_array_equal(
            matrix.nanmax(axis=0, keepdims=keepdims),
            np.nanmax(values, axis=0, keepdims=keepdims),
        )
    for where in (True, np.bool_(True), np.array(True), 1, 2.0, "selected"):
        assert matrix.sum(where=where) == np.sum(values, where=where)
