"""Tests for the bounded float32 ``asnumpy.diff`` adapter.

The CPU-only NumPy-oracle tests are always collected and never import
``asnumpy`` at module scope. Public/device tests are isolated behind the
``asnumpy`` fixture and skip unless the runtime gate is explicitly enabled.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


_RUNTIME_ENV = "ASNUMPY_DIFF_RUN_RUNTIME"
_OBSERVED_SHAPE = (1024, 1024)
_OBSERVED_OUTPUT_SHAPE = (1024, 1023)
_OBSERVED_AXIS = 1

# Declared before device qualification.  Exact equality is additionally
# required for expected zeros so that signed-zero semantics are preserved.
_FLOAT32_RTOL = 1e-6
_FLOAT32_ATOL = 1e-6


def numpy_oracle_diff(a, axis=_OBSERVED_AXIS):
    """Semantic source for the bounded adapter: ``numpy.diff``."""
    return np.diff(a, axis=axis)


def _to_numpy(value):
    """Convert a public-API result without importing optional runtimes."""
    if isinstance(value, np.ndarray):
        return value

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()

    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()

    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()

    return np.asarray(candidate)


def _assert_float32_matches(actual, expected):
    """Compare finite float32 values and preserve special-value classes."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    assert actual.dtype == np.dtype("float32"), f"actual dtype is {actual.dtype}"
    assert expected.dtype == np.dtype("float32"), (
        f"oracle dtype is {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"actual shape {actual.shape} != expected shape {expected.shape}"
    )

    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    assert np.array_equal(actual_nan, expected_nan), "NaN placement differs"

    actual_posinf = np.isposinf(actual)
    expected_posinf = np.isposinf(expected)
    assert np.array_equal(actual_posinf, expected_posinf), (
        "positive-infinity placement differs"
    )

    actual_neginf = np.isneginf(actual)
    expected_neginf = np.isneginf(expected)
    assert np.array_equal(actual_neginf, expected_neginf), (
        "negative-infinity placement differs"
    )

    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    assert np.array_equal(actual_finite, expected_finite), (
        "finite/non-finite placement differs"
    )

    if np.any(expected_finite):
        np.testing.assert_allclose(
            actual[expected_finite],
            expected[expected_finite],
            rtol=_FLOAT32_RTOL,
            atol=_FLOAT32_ATOL,
            equal_nan=False,
            err_msg="finite float32 diff values differ",
        )

    expected_zero = expected_finite & (expected == np.float32(0.0))
    if np.any(expected_zero):
        np.testing.assert_array_equal(
            actual[expected_zero],
            expected[expected_zero],
            err_msg="expected-zero values must be exactly zero",
        )
        assert np.array_equal(
            np.signbit(actual[expected_zero]),
            np.signbit(expected[expected_zero]),
        ), "signed-zero signbit is not preserved"


@pytest.fixture(scope="module")
def asnumpy():
    """Return the public AsNumpy module only for explicitly gated tests."""
    if os.environ.get(_RUNTIME_ENV) != "1":
        pytest.skip(
            "device runtime tests disabled; set "
            f"{_RUNTIME_ENV}=1 after installing a verified asnumpy/operator build"
        )

    module = pytest.importorskip(
        "asnumpy",
        reason="public asnumpy runtime is not installed in this environment",
    )
    if not callable(getattr(module, "diff", None)):
        pytest.fail("asnumpy is importable but asnumpy.diff is not callable")
    return module


def test_oracle_observed_profile_shape_and_dtype():
    """The observed 2-D profile produces float32 ``[1024, 1023]``."""
    x = np.arange(
        _OBSERVED_SHAPE[0] * _OBSERVED_SHAPE[1], dtype=np.float32
    ).reshape(_OBSERVED_SHAPE)
    y = numpy_oracle_diff(x, axis=_OBSERVED_AXIS)

    assert y.shape == _OBSERVED_OUTPUT_SHAPE
    assert y.dtype == np.dtype("float32")
    _assert_float32_matches(
        y, np.ones(_OBSERVED_OUTPUT_SHAPE, dtype=np.float32)
    )


def test_oracle_axis_one_subtracts_adjacent_columns():
    x = np.array(
        [
            [1.0, 4.0, 9.0, 16.0],
            [2.0, 5.0, 10.0, 17.0],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [3.0, 5.0, 7.0],
            [3.0, 5.0, 7.0],
        ],
        dtype=np.float32,
    )
    _assert_float32_matches(numpy_oracle_diff(x, axis=1), expected)


def test_oracle_axis_zero_subtracts_adjacent_rows():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 7.0, 12.0],
            [9.0, 16.0, 27.0],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [3.0, 5.0, 9.0],
            [5.0, 9.0, 15.0],
        ],
        dtype=np.float32,
    )
    _assert_float32_matches(numpy_oracle_diff(x, axis=0), expected)


def test_oracle_negative_axis_is_normalized():
    x = np.array(
        [
            [0.5, 1.5, 3.0, 6.0],
            [-1.0, 2.0, 5.0, 9.0],
        ],
        dtype=np.float32,
    )
    positive = numpy_oracle_diff(x, axis=1)
    negative = numpy_oracle_diff(x, axis=-1)
    _assert_float32_matches(negative, positive)


def test_oracle_nan_inf_and_signed_zero_propagation():
    x = np.array(
        [[np.nan, 1.0, 2.0, np.inf, -np.inf, 0.0, -0.0, 1.0]],
        dtype=np.float32,
    )
    expected = np.array(
        [[np.nan, 1.0, np.inf, -np.inf, np.inf, -0.0, 1.0]],
        dtype=np.float32,
    )
    _assert_float32_matches(numpy_oracle_diff(x, axis=1), expected)


def test_oracle_empty_dimensions_on_observed_rank():
    cases = (
        ((0, 5), 1, (0, 4)),
        ((3, 1), 1, (3, 0)),
        ((1, 5), 0, (0, 5)),
        ((5, 0), 0, (4, 0)),
    )
    for shape, axis, expected_shape in cases:
        x = np.empty(shape, dtype=np.float32)
        y = numpy_oracle_diff(x, axis=axis)
        assert y.shape == expected_shape
        assert y.dtype == np.dtype("float32")
        _assert_float32_matches(
            y, np.empty(expected_shape, dtype=np.float32)
        )


def test_oracle_rejects_axis_out_of_range():
    x = np.ones((2, 3), dtype=np.float32)
    with pytest.raises((ValueError, IndexError)):
        numpy_oracle_diff(x, axis=2)


def test_oracle_does_not_mutate_input():
    x = np.array(
        [
            [0.0, 1.0, 4.0, 9.0],
            [-0.0, 2.5, 6.5, 12.5],
        ],
        dtype=np.float32,
    )
    before = x.copy()
    _ = numpy_oracle_diff(x, axis=1)
    np.testing.assert_array_equal(x, before)


def test_public_asnumpy_diff_observed_profile(asnumpy):
    """Call the public API on the exact observed float32 profile."""
    x = np.arange(
        _OBSERVED_SHAPE[0] * _OBSERVED_SHAPE[1], dtype=np.float32
    ).reshape(_OBSERVED_SHAPE)
    expected = numpy_oracle_diff(x, axis=_OBSERVED_AXIS)

    actual = _to_numpy(asnumpy.diff(x, axis=_OBSERVED_AXIS))

    assert actual.shape == _OBSERVED_OUTPUT_SHAPE
    _assert_float32_matches(actual, expected)


def test_public_asnumpy_diff_special_values(asnumpy):
    x = np.array(
        [[np.nan, -np.inf, -1.0, -0.0, 0.0, 1.0, np.inf]],
        dtype=np.float32,
    )
    expected = numpy_oracle_diff(x, axis=1)

    actual = _to_numpy(asnumpy.diff(x, axis=1))

    _assert_float32_matches(actual, expected)


def test_public_asnumpy_diff_repeated_calls_are_stable(asnumpy):
    x = np.array(
        [
            [0.0, 1.0, 4.0, 9.0, 16.0],
            [-2.0, -1.0, 1.0, 4.0, 8.0],
            [3.5, 3.5, 3.5, 3.5, 3.5],
        ],
        dtype=np.float32,
    )
    before = x.copy()
    expected = numpy_oracle_diff(x, axis=1)

    first = _to_numpy(asnumpy.diff(x, axis=1))
    second = _to_numpy(asnumpy.diff(x, axis=1))

    np.testing.assert_array_equal(x, before)
    _assert_float32_matches(first, expected)
    _assert_float32_matches(second, expected)
    np.testing.assert_array_equal(first, second)


def test_public_asnumpy_diff_rejects_float64_without_implicit_cast(asnumpy):
    x = np.ones((4, 5), dtype=np.float64)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError)
    ):
        asnumpy.diff(x, axis=1)


def test_public_asnumpy_diff_rejects_unsupported_n_keyword(asnumpy):
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    with pytest.raises((TypeError, ValueError)):
        asnumpy.diff(x, n=2, axis=1)