"""Tests for the generated ``asnumpy.roll`` adapter.

The NumPy oracle tests are unconditional and exercise the exact semantic contract
that the custom operator must match.  Public runtime tests are guarded and are
skipped when ``asnumpy``, a native public API, or an Ascend device is absent.

The observed support envelope for this generation is:
  * input dtype: float32
  * input shape: [1024, 1024]
  * input layout: C-contiguous
  * call signature: roll(a, shift, axis=...)
  * observed call: roll(a, 1, axis=1)

Out-of-envelope inputs must be rejected explicitly rather than silently cast or
handled by a CPU fallback.
"""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest


# NumPy roll is a pure data movement operation.  Exact bitwise agreement is
# therefore required for float32 values, including NaN payloads and signed zero.
_FLOAT32_RTOL = 0.0
_FLOAT32_ATOL = 0.0


try:
    _AXIS_ERROR = np.AxisError
except AttributeError:  # NumPy >= 2.0
    from numpy.exceptions import AxisError as _AXIS_ERROR


_REJECTION_ERRORS = (
    TypeError,
    ValueError,
    RuntimeError,
    NotImplementedError,
)


def numpy_oracle(*args, **kwargs):
    """Oracle retained from the gap specification."""
    return np.roll(*args, **kwargs)


def _as_float32_bits(array):
    array = np.ascontiguousarray(array)
    if array.dtype != np.dtype("float32"):
        raise AssertionError(
            "bitwise float32 comparison requires float32, got {}".format(array.dtype)
        )
    return array.view(np.uint32)


def _assert_float32_bitwise_equal(actual, expected):
    """Compare float32 outputs exactly, preserving NaN/Inf/signed-zero bits."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == np.dtype("float32"), actual.dtype
    assert expected.dtype == np.dtype("float32"), expected.dtype
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_array_equal(
        _as_float32_bits(actual),
        _as_float32_bits(expected),
        err_msg="float32 bit patterns differ from the NumPy oracle",
    )


def _observed_input():
    """Create a deterministic tensor representative of the observed callsite."""
    rng = np.random.default_rng(0x5EED5011)
    array = rng.standard_normal((1024, 1024), dtype=np.float32)
    # Include values that commonly expose invalid passthrough/cast behaviour.
    array[0, 0] = np.nan
    array[0, 1] = np.inf
    array[0, 2] = -np.inf
    array[0, 3] = 0.0
    array[0, 4] = -0.0
    return array


# ---------------------------------------------------------------------------
# NumPy oracle tests: always collected, no asnumpy or custom operator needed.
# ---------------------------------------------------------------------------


def test_numpy_oracle_1d_shifts_and_dtype_preservation():
    array = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    cases = [
        (0, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
        (1, np.array([4.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)),
        (-1, np.array([1.0, 2.0, 3.0, 4.0, 0.0], dtype=np.float32)),
        (7, np.array([3.0, 4.0, 0.0, 1.0, 2.0], dtype=np.float32)),
        (-7, np.array([2.0, 3.0, 4.0, 0.0, 1.0], dtype=np.float32)),
    ]

    for shift, expected in cases:
        actual = numpy_oracle(array, shift)
        assert actual.dtype == np.dtype("float32")
        assert actual.shape == array.shape
        _assert_float32_bitwise_equal(actual, expected)


def test_numpy_oracle_2d_axis_0_and_axis_1():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)

    axis_0_positive = numpy_oracle(array, 2, axis=0)
    expected_axis_0_positive = np.concatenate(
        (array[-2:, :], array[:-2, :]), axis=0
    )
    _assert_float32_bitwise_equal(axis_0_positive, expected_axis_0_positive)

    axis_1_negative = numpy_oracle(array, -2, axis=1)
    expected_axis_1_negative = np.concatenate(
        (array[:, 2:], array[:, :2]), axis=1
    )
    _assert_float32_bitwise_equal(axis_1_negative, expected_axis_1_negative)

    axis_1_negative_axis_number = numpy_oracle(array, 1, axis=-1)
    expected_axis_1_negative_axis_number = np.concatenate(
        (array[:, -1:], array[:, :-1]), axis=1
    )
    _assert_float32_bitwise_equal(
        axis_1_negative_axis_number, expected_axis_1_negative_axis_number
    )


def test_numpy_oracle_axis_none_flattens_then_restores_shape():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    flat = array.ravel()
    shift = 3

    actual = numpy_oracle(array, shift, axis=None)
    expected = np.concatenate((flat[-shift:], flat[:-shift])).reshape(array.shape)
    _assert_float32_bitwise_equal(actual, expected)


def test_numpy_oracle_tuple_shift_and_axis_are_applied_in_order():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    actual = numpy_oracle(array, (1, -2), axis=(0, 1))

    expected = np.concatenate((array[-1:, :], array[:-1, :]), axis=0)
    expected = np.concatenate((expected[:, 2:], expected[:, :2]), axis=1)
    _assert_float32_bitwise_equal(actual, expected)


def test_numpy_oracle_preserves_nan_inf_and_signed_zero():
    array = np.array(
        [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0],
        dtype=np.float32,
    )

    actual = numpy_oracle(array, 3)
    expected = np.concatenate((array[-3:], array[:-3]))
    _assert_float32_bitwise_equal(actual, expected)

    assert np.signbit(actual[0])
    assert np.isnan(actual[3])
    assert np.isposinf(actual[4])
    assert np.isneginf(actual[5])
    assert actual[6] == np.float32(0.0)
    assert not np.signbit(actual[6])


def test_numpy_oracle_observed_1024x1024_profile():
    array = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    actual = numpy_oracle(array, 1, axis=1)
    expected = np.concatenate((array[:, -1:], array[:, :-1]), axis=1)

    assert actual.shape == (1024, 1024)
    assert actual.dtype == np.dtype("float32")
    assert actual is not array
    _assert_float32_bitwise_equal(actual, expected)


def test_numpy_oracle_empty_input_returns_empty_same_shape():
    array = np.empty((0, 4), dtype=np.float32)
    actual = numpy_oracle(array, 3, axis=0)

    assert actual.shape == (0, 4)
    assert actual.dtype == np.dtype("float32")
    assert actual.size == 0


def test_numpy_oracle_invalid_axis_and_mismatched_shift_axis_raise():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    with pytest.raises(_AXIS_ERROR):
        numpy_oracle(array, 1, axis=2)

    with pytest.raises(ValueError):
        numpy_oracle(array, (1, 2), axis=(0,))


# ---------------------------------------------------------------------------
# Public API runtime tests: guarded so collection succeeds without asnumpy.
# ---------------------------------------------------------------------------


def _npu_device_present():
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None:
        value = os.environ["ASCEND_RT_VISIBLE_DEVICES"].strip()
        return value not in ("", "-1")

    try:
        return any(Path("/dev").glob("davinci*"))
    except OSError:
        return False


def _import_asnumpy():
    try:
        asnumpy = importlib.import_module("asnumpy")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(
            "asnumpy is not importable in this runtime: {}: {}".format(
                type(exc).__name__, exc
            )
        )

    if not hasattr(asnumpy, "roll"):
        pytest.skip("public API asnumpy.roll is not installed")
    return asnumpy


def _require_public_api():
    if os.environ.get("NUMPILOT_SKIP_DEVICE_TESTS") == "1":
        pytest.skip("device tests disabled by NUMPILOT_SKIP_DEVICE_TESTS=1")

    if (
        os.environ.get("NUMPILOT_FORCE_DEVICE_TESTS") != "1"
        and not _npu_device_present()
    ):
        pytest.skip("no Ascend NPU device is visible in this runtime")

    asnumpy = _import_asnumpy()
    roll = asnumpy.roll

    if roll is np.roll:
        pytest.skip("asnumpy.roll is an alias of NumPy roll, not the custom operator")

    module_name = getattr(roll, "__module__", "") or ""
    if module_name == "numpy" or module_name.startswith("numpy."):
        pytest.skip(
            "asnumpy.roll resolves to a NumPy CPU fallback; native operator is absent"
        )

    return asnumpy


def _as_runtime_tensor(asnumpy, value):
    """Convert a NumPy array to the runtime tensor type when the runtime has one."""
    for name in ("asarray", "from_numpy", "array"):
        converter = getattr(asnumpy, name, None)
        if not callable(converter):
            continue
        try:
            return converter(value)
        except Exception:
            continue

    ndarray_type = getattr(asnumpy, "ndarray", None)
    if isinstance(ndarray_type, type):
        try:
            return ndarray_type(value)
        except Exception:
            pass

    # Some adapters intentionally accept NumPy arrays directly.
    return value


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value

    for name in ("numpy", "to_numpy", "cpu"):
        converter = getattr(value, name, None)
        if not callable(converter):
            continue
        try:
            value = converter()
        except Exception:
            continue
        if isinstance(value, np.ndarray):
            return value
        break

    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - runtime dependent
        pytest.fail("public API result cannot be converted to NumPy: {}".format(exc))


def test_public_api_observed_profile_matches_numpy_oracle():
    asnumpy = _require_public_api()
    input_np = _observed_input()
    input_rt = _as_runtime_tensor(asnumpy, input_np)

    output_np = _to_numpy(asnumpy.roll(input_rt, 1, axis=1))
    expected = numpy_oracle(input_np, 1, axis=1)

    assert output_np.shape == (1024, 1024)
    assert output_np.dtype == np.dtype("float32")
    _assert_float32_bitwise_equal(output_np, expected)


def test_public_api_repeated_calls_are_stable_and_do_not_mutate_input():
    asnumpy = _require_public_api()
    input_np = _observed_input()
    original = input_np.copy()
    input_rt = _as_runtime_tensor(asnumpy, input_np)

    first = _to_numpy(asnumpy.roll(input_rt, 1, axis=1))
    second = _to_numpy(asnumpy.roll(input_rt, 1, axis=1))
    expected = numpy_oracle(input_np, 1, axis=1)

    _assert_float32_bitwise_equal(first, expected)
    _assert_float32_bitwise_equal(second, expected)
    _assert_float32_bitwise_equal(first, second)
    _assert_float32_bitwise_equal(input_np, original)


def test_public_api_rejects_unsupported_float16_input():
    asnumpy = _require_public_api()
    input_np = np.zeros((1024, 1024), dtype=np.float16)
    input_rt = _as_runtime_tensor(asnumpy, input_np)

    with pytest.raises(_REJECTION_ERRORS):
        asnumpy.roll(input_rt, 1, axis=1)


def test_public_api_rejects_out_of_envelope_shape():
    asnumpy = _require_public_api()
    input_np = np.arange(4, dtype=np.float32).reshape(2, 2)
    input_rt = _as_runtime_tensor(asnumpy, input_np)

    with pytest.raises(_REJECTION_ERRORS):
        asnumpy.roll(input_rt, 1, axis=1)


def test_public_api_rejects_invalid_axis():
    asnumpy = _require_public_api()
    input_np = _observed_input()
    input_rt = _as_runtime_tensor(asnumpy, input_np)

    with pytest.raises(_REJECTION_ERRORS):
        asnumpy.roll(input_rt, 1, axis=2)


def test_public_api_rejects_unsupported_out_keyword():
    asnumpy = _require_public_api()
    input_np = _observed_input()
    input_rt = _as_runtime_tensor(asnumpy, input_np)

    with pytest.raises(_REJECTION_ERRORS):
        asnumpy.roll(input_rt, 1, axis=1, out=input_rt)