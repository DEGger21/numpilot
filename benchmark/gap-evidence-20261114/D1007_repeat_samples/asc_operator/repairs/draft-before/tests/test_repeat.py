"""NumPy-oracle and public AsNumpy adapter tests for ``repeat``.

The oracle tests in this file use only NumPy and therefore collect and pass
when neither ``asnumpy`` nor the custom Repeat operator is installed.  The
public-API tests import ``asnumpy`` lazily inside each runtime test; when the
runtime is unavailable they skip explicitly rather than pretending to be CPU
golden passes.

``numpy.repeat`` is a pure selection/data-movement operation for the admitted
float32 profile.  No arithmetic rounding is introduced, so the numerical
contract is exact float32 bit equality, including NaN, infinities, and signed
zero.  Set ``NUMPILOT_REQUIRE_ASNUMPY=1`` to turn missing-runtime skips into
failures during activation validation.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

OBSERVED_INPUT_SHAPE = (1024, 1024)
OBSERVED_OUTPUT_SHAPE = (1024, 3072)
OBSERVED_REPEATS = 3
OBSERVED_AXIS = 1

# Repeat is data movement, not arithmetic.  Keep the declared tolerance at
# zero and verify raw float32 bits below.
RTOL = 0.0
ATOL = 0.0

_REQUIRE_RUNTIME = os.environ.get("NUMPILOT_REQUIRE_ASNUMPY") == "1"
_AXIS_ERROR = getattr(getattr(np, "exceptions", np), "AxisError", IndexError)


def numpy_oracle(*args, **kwargs):
    """Semantic oracle from the GapSpec/NumPy taskbook."""
    return np.repeat(*args, **kwargs)


def _repeat_indices(length, repeats):
    """Build the index map for repeating a single axis without np.repeat."""
    if repeats < 0:
        raise ValueError("repeats must be non-negative")
    if repeats == 0:
        return np.empty(0, dtype=np.intp)
    base = np.arange(length, dtype=np.intp).reshape(length, 1)
    return np.broadcast_to(base, (length, repeats)).reshape(-1)


def _manual_repeat_axis1(array, repeats):
    """Reference expansion along axis 1 using explicit indices only."""
    return array[:, _repeat_indices(array.shape[1], repeats)]


def _assert_exact_float32(actual, expected, *, context=""):
    """Require identical float32 dtype, shape, and raw bit patterns."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)

    if expected_array.dtype != np.dtype(np.float32):
        pytest.fail(
            f"{context}: test expected array must be float32, "
            f"got {expected_array.dtype}"
        )
    if actual_array.dtype != np.dtype(np.float32):
        pytest.fail(
            f"{context}: expected float32 output, got {actual_array.dtype}"
        )
    if actual_array.shape != expected_array.shape:
        pytest.fail(
            f"{context}: expected shape {expected_array.shape}, "
            f"got {actual_array.shape}"
        )

    actual_bits = np.ascontiguousarray(actual_array).view(np.uint32)
    expected_bits = np.ascontiguousarray(expected_array).view(np.uint32)
    try:
        np.testing.assert_array_equal(actual_bits, expected_bits)
    except AssertionError as exc:
        raise AssertionError(
            f"{context}: float32 bitwise mismatch; repeat must preserve exact "
            f"values, NaNs, infinities, and signed zero\n{exc}"
        ) from exc


def _require_asnumpy_or_skip():
    """Import the public AsNumpy module lazily and guard the runtime tests."""
    try:
        import asnumpy
    except ImportError as exc:
        message = (
            "asnumpy.repeat runtime is unavailable; public-API tests require "
            "the built AsNumpy package and the Repeat custom operator. "
            f"Import detail: {exc}"
        )
        if _REQUIRE_RUNTIME:
            pytest.fail(message)
        pytest.skip(message)

    if not hasattr(asnumpy, "repeat"):
        pytest.fail(
            "asnumpy imported, but the public API asnumpy.repeat is missing; "
            "this is a runtime integration failure"
        )
    if not callable(asnumpy.repeat):
        pytest.fail("asnumpy.repeat exists but is not callable")
    return asnumpy


def _observed_input():
    """Deterministic input for the observed float32 [1024, 1024] profile."""
    element_count = OBSERVED_INPUT_SHAPE[0] * OBSERVED_INPUT_SHAPE[1]
    return np.arange(element_count, dtype=np.float32).reshape(OBSERVED_INPUT_SHAPE)


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests: these must pass without asnumpy or the custom op.
# ---------------------------------------------------------------------------


def test_numpy_oracle_observed_profile_shape_layout_and_bits():
    x = _observed_input()
    actual = numpy_oracle(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)
    expected = _manual_repeat_axis1(x, OBSERVED_REPEATS)

    assert actual.shape == OBSERVED_OUTPUT_SHAPE
    assert actual.dtype == np.dtype(np.float32)
    assert actual.flags.c_contiguous
    _assert_exact_float32(
        actual, expected, context="numpy oracle observed profile"
    )


def test_numpy_oracle_axis_normalization_and_flattening():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    axis1 = numpy_oracle(x, 2, axis=1)
    axis1_expected = _manual_repeat_axis1(x, 2)
    _assert_exact_float32(axis1, axis1_expected, context="oracle axis=1")
    assert axis1.shape == (2, 6)

    axis0 = numpy_oracle(x, 2, axis=0)
    axis0_expected = x[_repeat_indices(x.shape[0], 2), :]
    _assert_exact_float32(axis0, axis0_expected, context="oracle axis=0")
    assert axis0.shape == (4, 3)

    axis_neg = numpy_oracle(x, 2, axis=-1)
    _assert_exact_float32(axis_neg, axis1_expected, context="oracle axis=-1")
    assert axis_neg.shape == (2, 6)

    flat = x.reshape(-1)
    flat_expected = flat[_repeat_indices(flat.size, 2)]
    axis_none = numpy_oracle(x, 2, axis=None)
    _assert_exact_float32(
        axis_none, flat_expected, context="oracle axis=None"
    )
    assert axis_none.shape == (12,)

    default_axis = numpy_oracle(x, 2)
    _assert_exact_float32(
        default_axis, flat_expected, context="oracle default axis"
    )
    assert default_axis.shape == (12,)


def test_numpy_oracle_special_values_preserved_exactly():
    x = np.array([[np.nan, np.inf, -np.inf, -0.0, 0.0]], dtype=np.float32)
    actual = numpy_oracle(x, 3, axis=1)
    expected = _manual_repeat_axis1(x, 3)

    _assert_exact_float32(
        actual, expected, context="oracle NaN/Inf/signed-zero"
    )
    assert actual.shape == (1, 15)
    assert np.isnan(actual[0, 0])
    assert np.isinf(actual[0, 3]) and actual[0, 3] > 0
    assert np.isinf(actual[0, 6]) and actual[0, 6] < 0
    assert np.signbit(actual[0, 9])
    assert not np.signbit(actual[0, 12])


def test_numpy_oracle_empty_inputs_and_zero_repeats():
    empty_columns = np.empty((2, 0), dtype=np.float32)
    y_columns = numpy_oracle(empty_columns, 3, axis=1)
    _assert_exact_float32(
        y_columns,
        empty_columns[:, _repeat_indices(0, 3)],
        context="oracle empty columns",
    )
    assert y_columns.shape == (2, 0)

    empty_rows = np.empty((0, 3), dtype=np.float32)
    y_rows = numpy_oracle(empty_rows, 2, axis=0)
    _assert_exact_float32(
        y_rows,
        empty_rows[_repeat_indices(0, 2), :],
        context="oracle empty rows",
    )
    assert y_rows.shape == (0, 3)

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y_zero = numpy_oracle(x, 0, axis=1)
    _assert_exact_float32(
        y_zero,
        np.empty((2, 0), dtype=np.float32),
        context="oracle zero repeats",
    )
    assert y_zero.shape == (2, 0)


def test_numpy_oracle_rejects_invalid_axis_and_repeats():
    x = np.ones((2, 3), dtype=np.float32)

    with pytest.raises((_AXIS_ERROR, IndexError)):
        numpy_oracle(x, 2, axis=2)
    with pytest.raises((_AXIS_ERROR, IndexError)):
        numpy_oracle(x, 2, axis=-3)
    with pytest.raises(ValueError):
        numpy_oracle(x, -1, axis=1)
    with pytest.raises(TypeError):
        numpy_oracle(x, 1.5, axis=1)


def test_numpy_oracle_bool_output_dtype_preserved():
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    actual = numpy_oracle(x, 3, axis=1)
    expected = _manual_repeat_axis1(x, 3)

    assert actual.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(actual, expected)


def test_numpy_oracle_scalar_materialization_without_axis():
    x = np.array(7.0, dtype=np.float32)
    actual = numpy_oracle(x, 3)
    expected = np.array([7.0, 7.0, 7.0], dtype=np.float32)

    _assert_exact_float32(
        actual, expected, context="oracle scalar materialization"
    )
    assert actual.shape == (3,)


# ---------------------------------------------------------------------------
# Public asnumpy.repeat adapter tests: explicitly skipped when absent.
# ---------------------------------------------------------------------------


def test_public_api_repeat_observed_profile_exact_bits():
    asnumpy = _require_asnumpy_or_skip()
    x = _observed_input()
    expected = numpy_oracle(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)
    actual = asnumpy.repeat(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)
    actual_array = np.asarray(actual)

    assert actual_array.shape == OBSERVED_OUTPUT_SHAPE
    assert actual_array.dtype == np.dtype(np.float32)
    assert actual_array.flags.c_contiguous
    _assert_exact_float32(
        actual_array, expected, context="asnumpy.repeat observed profile"
    )


def test_public_api_repeat_is_not_numpy_repeat_alias():
    asnumpy = _require_asnumpy_or_skip()
    assert asnumpy.repeat is not np.repeat, (
        "asnumpy.repeat must bind to the custom runtime, not alias "
        "numpy.repeat as a silent CPU fallback"
    )


def test_public_api_repeat_repeated_calls_and_input_lifecycle():
    asnumpy = _require_asnumpy_or_skip()
    x = _observed_input()
    x_bits_before = np.ascontiguousarray(x).view(np.uint32).copy()
    expected = numpy_oracle(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)

    first = asnumpy.repeat(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)
    second = asnumpy.repeat(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)

    _assert_exact_float32(
        first, expected, context="first asnumpy.repeat call"
    )
    _assert_exact_float32(
        second, expected, context="second asnumpy.repeat call"
    )
    np.testing.assert_array_equal(
        np.ascontiguousarray(x).view(np.uint32), x_bits_before
    )


def test_public_api_repeat_special_values_exact_bits():
    asnumpy = _require_asnumpy_or_skip()
    x = _observed_input()
    x[0, :5] = np.array(
        [np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=np.float32
    )
    expected = numpy_oracle(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)
    actual = asnumpy.repeat(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)
    actual_array = np.asarray(actual)

    _assert_exact_float32(
        actual_array, expected, context="asnumpy.repeat special values"
    )
    assert actual_array.shape == OBSERVED_OUTPUT_SHAPE
    assert np.isnan(actual_array[0, 0])
    assert np.isinf(actual_array[0, 3]) and actual_array[0, 3] > 0
    assert np.isinf(actual_array[0, 6]) and actual_array[0, 6] < 0
    assert np.signbit(actual_array[0, 9])
    assert not np.signbit(actual_array[0, 12])


@pytest.mark.parametrize(
    "dtype", (np.float64, np.float16, np.int32, np.bool_)
)
def test_public_api_repeat_rejects_non_float32_input(dtype):
    asnumpy = _require_asnumpy_or_skip()
    x = np.ones(OBSERVED_INPUT_SHAPE, dtype=dtype)

    with pytest.raises(
        (TypeError, ValueError, NotImplementedError, RuntimeError)
    ):
        asnumpy.repeat(x, OBSERVED_REPEATS, axis=OBSERVED_AXIS)


@pytest.mark.parametrize("axis", (2, -3))
def test_public_api_repeat_rejects_invalid_axis(axis):
    asnumpy = _require_asnumpy_or_skip()
    x = _observed_input()

    with pytest.raises(
        (_AXIS_ERROR, IndexError, ValueError, TypeError, RuntimeError)
    ):
        asnumpy.repeat(x, OBSERVED_REPEATS, axis=axis)


@pytest.mark.parametrize("repeats", (-1, 1.5))
def test_public_api_repeat_rejects_invalid_repeats(repeats):
    asnumpy = _require_asnumpy_or_skip()
    x = _observed_input()

    with pytest.raises(
        (TypeError, ValueError, NotImplementedError, RuntimeError)
    ):
        asnumpy.repeat(x, repeats, axis=OBSERVED_AXIS)


def test_public_api_repeat_rejects_scalar_input_for_axis():
    asnumpy = _require_asnumpy_or_skip()
    x = np.asarray(7.0, dtype=np.float32)

    with pytest.raises(
        (_AXIS_ERROR, IndexError, ValueError, TypeError, RuntimeError)
    ):
        asnumpy.repeat(x, OBSERVED_REPEATS, axis=0)


def test_public_api_repeat_rejects_unsupported_keyword():
    asnumpy = _require_asnumpy_or_skip()
    x = _observed_input()

    with pytest.raises(TypeError):
        asnumpy.repeat(
            x, OBSERVED_REPEATS, axis=OBSERVED_AXIS, dtype=np.float32
        )