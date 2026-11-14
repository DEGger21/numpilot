"""Contract tests for ``asnumpy.nextafter`` and the NumPy oracle.

The pure NumPy tests in this module always collect and pass without AsNumpy or
the custom operator installed.  Public-API/runtime tests are opt-in and skipped
unless ``NUMPILOT_RUN_ASCEND_TESTS=1`` is set; they never substitute a CPU
golden result for a missing device implementation.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


PUBLIC_MODULE = "asnumpy"
PUBLIC_API = "nextafter"
RUNTIME_ENV = "NUMPILOT_RUN_ASCEND_TESTS"
OBSERVED_SHAPE = (1024, 1024)
FLOAT32 = np.dtype(np.float32)

_INF = np.float32(np.inf)
_NEG_INF = np.float32(-np.inf)
_NAN = np.float32(np.nan)
_FINFO32 = np.finfo(np.float32)
_MAX = np.float32(_FINFO32.max)
_NEG_MAX = np.float32(-_FINFO32.max)
_TINY_POS = np.float32(np.nextafter(np.float32(0.0), np.float32(1.0)))
_NEG_TINY_POS = np.float32(-_TINY_POS)

# The first rows of the profile fixture exercise exact IEEE-754 neighbors,
# signed zero, infinities, NaNs, and representable-limit transitions.
_SPECIAL_X = np.array(
    [
        0.0,
        -0.0,
        0.0,
        -0.0,
        _TINY_POS,
        _NEG_TINY_POS,
        _MAX,
        _NEG_MAX,
        _INF,
        _NEG_INF,
        _INF,
        _NEG_INF,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        _NAN,
        _INF,
    ],
    dtype=np.float32,
)

_SPECIAL_Y = np.array(
    [
        0.0,
        0.0,
        -0.0,
        -0.0,
        _INF,
        _NEG_INF,
        _INF,
        _NEG_INF,
        0.0,
        0.0,
        _INF,
        _NEG_INF,
        2.0,
        0.0,
        0.0,
        -2.0,
        _NAN,
        _NAN,
        1.0,
        _MAX,
    ],
    dtype=np.float32,
)

_SPECIAL_NAN_MASK = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ],
    dtype=np.bool_,
)

# Raw float32 bit patterns for the non-NaN positions above.  Comparing bit
# patterns instead of values preserves signed-zero semantics.
_SPECIAL_EXPECTED_BITS = np.array(
    [
        0x00000000,
        0x00000000,
        0x80000000,
        0x80000000,
        0x00000002,
        0x80000002,
        0x7F800000,
        0xFF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x3F800001,
        0x3F7FFFFF,
        0xBF7FFFFF,
        0xBF800001,
        0x7FC00000,
        0x7FC00000,
        0x7FC00000,
        0x7F7FFFFF,
    ],
    dtype=np.uint32,
)


def _as_u32(array: np.ndarray) -> np.ndarray:
    """Return a float32 view as raw uint32 bits."""
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _assert_float32_exact(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    context: str = "nextafter",
) -> None:
    """Compare float32 outputs exactly, treating all NaNs as equivalent.

    Signed zero and every finite/infinite value must match bit-for-bit.  NaN
    payloads are intentionally ignored because the NumPy oracle does not require
    a particular quiet-NaN payload.
    """
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)

    assert actual_array.dtype == FLOAT32, (
        f"{context}: expected float32 output, got dtype {actual_array.dtype!r}"
    )
    assert expected_array.dtype == FLOAT32, (
        f"{context}: oracle unexpectedly produced dtype {expected_array.dtype!r}"
    )
    assert actual_array.shape == expected_array.shape, (
        f"{context}: shape mismatch: actual={actual_array.shape}, "
        f"expected={expected_array.shape}"
    )
    assert actual_array.flags.c_contiguous, (
        f"{context}: observed-profile output must be C-contiguous"
    )

    actual_nan = np.isnan(actual_array)
    expected_nan = np.isnan(expected_array)
    if not np.array_equal(actual_nan, expected_nan):
        mismatch = np.argwhere(actual_nan != expected_nan)
        first = tuple(int(v) for v in mismatch[0])
        raise AssertionError(
            f"{context}: NaN mask mismatch at {first}: "
            f"actual={actual_array[first]!r}, expected={expected_array[first]!r}"
        )

    finite_mask = ~expected_nan
    if np.any(finite_mask):
        actual_bits = _as_u32(actual_array[finite_mask])
        expected_bits = _as_u32(expected_array[finite_mask])
        bit_mismatch = actual_bits != expected_bits
        if np.any(bit_mismatch):
            finite_positions = np.flatnonzero(finite_mask)
            first = int(finite_positions[np.flatnonzero(bit_mismatch)[0]])
            actual_flat = actual_array.reshape(-1)
            expected_flat = expected_array.reshape(-1)
            raise AssertionError(
                f"{context}: bit mismatch at flat index {first}: "
                f"actual={actual_flat[first]!r} "
                f"(0x{_as_u32(actual_flat[first : first + 1])[0]:08x}), "
                f"expected={expected_flat[first]!r} "
                f"(0x{_as_u32(expected_flat[first : first + 1])[0]:08x})"
            )


def _profile_inputs(seed: int = 0x5EED) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic float32 inputs for the observed [1024, 1024] profile."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(OBSERVED_SHAPE).astype(np.float32)
    y = rng.standard_normal(OBSERVED_SHAPE).astype(np.float32)
    x *= np.float32(16.0)
    y *= np.float32(16.0)

    count = int(_SPECIAL_X.size)
    x.reshape(-1)[:count] = _SPECIAL_X
    y.reshape(-1)[:count] = _SPECIAL_Y
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _require_runtime_enabled() -> None:
    if os.environ.get(RUNTIME_ENV) != "1":
        pytest.skip(
            f"runtime/device tests disabled; set {RUNTIME_ENV}=1 to run "
            "the public AsNumpy/ACLNN checks"
        )


def _load_public_nextafter():
    """Load and validate ``asnumpy.nextafter`` only inside guarded tests."""
    _require_runtime_enabled()
    try:
        module = importlib.import_module(PUBLIC_MODULE)
    except (ImportError, OSError) as exc:
        pytest.skip(f"{PUBLIC_MODULE} unavailable: {exc}")

    fn = getattr(module, PUBLIC_API, None)
    if fn is None:
        pytest.skip(
            f"{PUBLIC_MODULE}.{PUBLIC_API} is not present; custom operator "
            "is not active in this environment"
        )
    if not callable(fn):
        pytest.fail(f"{PUBLIC_MODULE}.{PUBLIC_API} exists but is not callable")
    return module, fn


def test_observed_profile_fixture_is_float32_rank2_c_contiguous() -> None:
    x, y = _profile_inputs()
    for name, value in (("x", x), ("y", y)):
        assert value.dtype == FLOAT32, f"{name}: unexpected dtype {value.dtype!r}"
        assert value.shape == OBSERVED_SHAPE, f"{name}: unexpected shape {value.shape}"
        assert value.ndim == 2, f"{name}: expected rank 2"
        assert value.flags.c_contiguous, f"{name}: expected C-contiguous layout"


def test_numpy_oracle_observed_profile_output_contract() -> None:
    x, y = _profile_inputs()
    result = np.nextafter(x, y)

    assert result.dtype == FLOAT32
    assert result.shape == OBSERVED_SHAPE
    assert result.ndim == 2
    assert result.flags.c_contiguous


def test_numpy_oracle_profile_special_values_are_bit_exact() -> None:
    x, y = _profile_inputs()
    result = np.nextafter(x, y)
    first = np.asarray(result).reshape(-1)[: int(_SPECIAL_X.size)]

    actual_nan_mask = np.isnan(first)
    np.testing.assert_array_equal(actual_nan_mask, _SPECIAL_NAN_MASK)

    actual_non_nan_bits = _as_u32(first)[~actual_nan_mask]
    expected_non_nan_bits = _SPECIAL_EXPECTED_BITS[~_SPECIAL_NAN_MASK]
    np.testing.assert_array_equal(actual_non_nan_bits, expected_non_nan_bits)


def test_numpy_oracle_is_deterministic_and_does_not_mutate_inputs() -> None:
    x, y = _profile_inputs()
    x_bits_before = _as_u32(x).copy()
    y_bits_before = _as_u32(y).copy()

    first = np.nextafter(x, y)
    second = np.nextafter(x, y)

    _assert_float32_exact(second, first, context="deterministic NumPy oracle")
    np.testing.assert_array_equal(_as_u32(x), x_bits_before)
    np.testing.assert_array_equal(_as_u32(y), y_bits_before)


def test_numpy_oracle_float64_promotion_is_an_out_of_scope_boundary() -> None:
    # The adapter supports observed float32 inputs only.  The NumPy oracle keeps
    # float64 as float64, so any adapter that silently casts or returns float64
    # violates the bounded support scope.
    x = np.ones((2, 2), dtype=np.float64)
    y = np.zeros((2, 2), dtype=np.float64)
    result = np.nextafter(x, y)
    assert result.dtype == np.dtype(np.float64)


def test_public_asnumpy_nextafter_is_exposed() -> None:
    module, fn = _load_public_nextafter()
    assert module.__name__ == PUBLIC_MODULE
    assert callable(fn)


def test_public_asnumpy_nextafter_observed_profile_matches_oracle() -> None:
    _, fn = _load_public_nextafter()
    x, y = _profile_inputs()
    expected = np.nextafter(x, y)

    x_bits_before = _as_u32(x).copy()
    y_bits_before = _as_u32(y).copy()

    actual = fn(x, y)

    assert isinstance(actual, np.ndarray), (
        "asnumpy.nextafter must return a NumPy ndarray for the observed profile"
    )
    _assert_float32_exact(actual, expected, context="public asnumpy.nextafter")
    np.testing.assert_array_equal(_as_u32(x), x_bits_before)
    np.testing.assert_array_equal(_as_u32(y), y_bits_before)


def test_public_asnumpy_nextafter_special_values_and_signed_zero() -> None:
    _, fn = _load_public_nextafter()
    x, y = _profile_inputs()
    actual = fn(x, y)

    first = np.asarray(actual).reshape(-1)[: int(_SPECIAL_X.size)]
    actual_nan_mask = np.isnan(first)
    np.testing.assert_array_equal(actual_nan_mask, _SPECIAL_NAN_MASK)
    np.testing.assert_array_equal(
        _as_u32(first)[~actual_nan_mask],
        _SPECIAL_EXPECTED_BITS[~_SPECIAL_NAN_MASK],
    )


def test_public_asnumpy_nextafter_repeated_input_lifecycle_is_stable() -> None:
    _, fn = _load_public_nextafter()
    x, y = _profile_inputs()
    expected = np.nextafter(x, y)

    first = fn(x, y)
    second = fn(x, y)

    assert isinstance(first, np.ndarray)
    assert isinstance(second, np.ndarray)
    _assert_float32_exact(first, expected, context="first public call")
    _assert_float32_exact(second, expected, context="second public call")


def test_public_asnumpy_nextafter_rejects_keyword_arguments() -> None:
    _, fn = _load_public_nextafter()
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    y = np.ones(OBSERVED_SHAPE, dtype=np.float32)

    # Signature contract: exactly two positional arguments, no keywords.
    with pytest.raises((TypeError, ValueError)):
        fn(x, y, dtype=np.float32)

    with pytest.raises((TypeError, ValueError)):
        fn(x, y, out=x)


def test_public_asnumpy_nextafter_rejects_float64_without_silent_cast() -> None:
    _, fn = _load_public_nextafter()
    x = np.ones(OBSERVED_SHAPE, dtype=np.float64)
    y = np.zeros(OBSERVED_SHAPE, dtype=np.float64)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        fn(x, y)


def test_public_asnumpy_nextafter_rejects_broadcastable_shape_mismatch() -> None:
    _, fn = _load_public_nextafter()
    x = np.ones(OBSERVED_SHAPE, dtype=np.float32)
    y = np.ones((OBSERVED_SHAPE[0], 1), dtype=np.float32)

    # No broadcasting was observed for this gap, and the bounded support scope
    # admits only equal observed-profile shapes.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        fn(x, y)