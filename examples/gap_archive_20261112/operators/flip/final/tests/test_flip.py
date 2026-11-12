"""Tests for the NumPy ``flip`` oracle and the public ``asnumpy.flip`` API.

The CPU oracle tests in this module always run with NumPy and never import
AsNumpy.  Public-API/runtime tests import AsNumpy lazily and skip when the
unverified runtime or custom operator is not installed; a skipped runtime test
must never be reported as a CPU golden-test pass.

The observed admitted profile is float32 ``[2048, 2048]`` with ``axis=1``.
The operator is a pure permutation, so comparisons are bit-exact, including
NaN payloads, infinities and signed zeros.
"""

from __future__ import annotations

import numpy as np
import pytest


OBSERVED_SHAPE = (2048, 2048)
OBSERVED_AXIS = 1
FLOAT32 = np.float32


def _assert_exact(expected: np.ndarray, actual: object) -> None:
    """Assert shape, dtype and bit-exact values against a NumPy oracle result."""
    actual_array = np.asarray(actual)
    assert actual_array.shape == expected.shape, (
        f"shape mismatch: expected {expected.shape}, got {actual_array.shape}"
    )
    assert actual_array.dtype == expected.dtype, (
        f"dtype mismatch: expected {expected.dtype}, got {actual_array.dtype}"
    )

    if expected.dtype == np.float32:
        expected_bits = np.ascontiguousarray(expected).view(np.uint32)
        actual_bits = np.ascontiguousarray(actual_array).view(np.uint32)
        assert np.array_equal(expected_bits, actual_bits), (
            "float32 values are not bit-exact"
        )
    elif expected.dtype == np.float64:
        expected_bits = np.ascontiguousarray(expected).view(np.uint64)
        actual_bits = np.ascontiguousarray(actual_array).view(np.uint64)
        assert np.array_equal(expected_bits, actual_bits), (
            "float64 values are not bit-exact"
        )
    else:
        assert np.array_equal(expected, actual_array), (
            "values are not exactly equal"
        )


def _import_asnumpy():
    """Import the public AsNumpy package lazily, skipping when unavailable."""
    try:
        import asnumpy  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime
        pytest.skip(
            "asnumpy runtime is unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    flip = getattr(asnumpy, "flip", None)
    if not callable(flip):
        pytest.skip("asnumpy.flip public API is not installed")
    return asnumpy


# ---------------------------------------------------------------------------
# CPU NumPy oracle tests: these must collect and pass without AsNumpy.
# ---------------------------------------------------------------------------


def test_oracle_axis1_observed_profile_is_exact_reversal() -> None:
    a = np.arange(OBSERVED_SHAPE[0] * OBSERVED_SHAPE[1], dtype=FLOAT32)
    a = (a % 1024).reshape(OBSERVED_SHAPE)
    expected = a[:, ::-1]
    actual = np.flip(a, axis=OBSERVED_AXIS)
    _assert_exact(expected, actual)


def test_oracle_axis0_and_axis1_reverse_correct_dimensions() -> None:
    a = np.arange(4 * 6, dtype=FLOAT32).reshape(4, 6)

    _assert_exact(a[:, ::-1], np.flip(a, axis=1))
    _assert_exact(a[::-1, :], np.flip(a, axis=0))


def test_oracle_negative_axis_is_normalized() -> None:
    a = np.arange(3 * 5, dtype=FLOAT32).reshape(3, 5)

    _assert_exact(a[:, ::-1], np.flip(a, axis=-1))
    _assert_exact(a[::-1, :], np.flip(a, axis=-2))


def test_oracle_axis_none_and_axis_tuple() -> None:
    a = np.arange(2 * 3 * 4, dtype=FLOAT32).reshape(2, 3, 4)

    _assert_exact(a[::-1, ::-1, ::-1], np.flip(a))
    _assert_exact(a[::-1, :, ::-1], np.flip(a, axis=(0, 2)))
    _assert_exact(a[:, ::-1, ::-1], np.flip(a, axis=(1, 2)))


def test_oracle_accepts_non_contiguous_input() -> None:
    base = np.arange(6 * 8, dtype=FLOAT32).reshape(6, 8)
    a = base[::2, 1::2]
    assert not a.flags.c_contiguous

    _assert_exact(a[:, ::-1], np.flip(a, axis=1))
    _assert_exact(a[::-1, :], np.flip(a, axis=0))


def test_oracle_special_values_and_signed_zero_are_preserved() -> None:
    a = np.array(
        [
            [
                np.nan,
                np.inf,
                -np.inf,
                np.float32(0.0),
                np.float32(-0.0),
                np.float32(1.5),
                np.float32(-2.25),
                np.float32(3.0),
            ],
            [
                np.float32(-0.0),
                np.nan,
                np.float32(-7.0),
                np.float32(0.0),
                -np.inf,
                np.inf,
                np.float32(-0.5),
                np.float32(0.5),
            ],
        ],
        dtype=FLOAT32,
    )

    expected = a[:, ::-1]
    actual = np.flip(a, axis=1)
    _assert_exact(expected, actual)
    assert np.array_equal(np.signbit(actual), np.signbit(expected))


def test_oracle_bool_output_dtype_is_preserved() -> None:
    a = np.array(
        [[True, False, False], [False, True, False]],
        dtype=np.bool_,
    )

    expected = a[:, ::-1]
    actual = np.flip(a, axis=1)
    _assert_exact(expected, actual)
    assert actual.dtype == np.bool_


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0, 3), 1),
        ((3, 0), 0),
        ((2, 0, 4), 1),
        ((0, 0), 0),
    ],
)
def test_oracle_empty_tensors_preserve_shape(shape: tuple[int, ...], axis: int) -> None:
    a = np.empty(shape, dtype=FLOAT32)
    expected = np.empty(shape, dtype=FLOAT32)
    actual = np.flip(a, axis=axis)
    _assert_exact(expected, actual)


def test_oracle_does_not_mutate_input() -> None:
    a = np.arange(12, dtype=FLOAT32).reshape(3, 4)
    before = a.copy()

    _ = np.flip(a, axis=0)
    _ = np.flip(a, axis=1)

    _assert_exact(before, a)


def test_oracle_invalid_axis_raises() -> None:
    a = np.zeros((2, 3), dtype=FLOAT32)
    axis_errors = (ValueError, IndexError, getattr(np, "AxisError", ValueError))

    with pytest.raises(axis_errors):
        np.flip(a, axis=2)

    with pytest.raises(ValueError):
        np.flip(a, axis=(0, 0))


# ---------------------------------------------------------------------------
# Public AsNumpy runtime tests: lazily guarded and skipped when unavailable.
# ---------------------------------------------------------------------------


def test_public_flip_observed_profile() -> None:
    asnumpy = _import_asnumpy()

    a = np.arange(OBSERVED_SHAPE[0] * OBSERVED_SHAPE[1], dtype=FLOAT32)
    a = a.reshape(OBSERVED_SHAPE)
    before = a.copy()

    expected = np.flip(a, axis=OBSERVED_AXIS)
    actual = asnumpy.flip(a, axis=OBSERVED_AXIS)

    _assert_exact(expected, actual)
    _assert_exact(before, a)


def test_public_flip_special_values_and_signed_zero() -> None:
    asnumpy = _import_asnumpy()

    a = np.array(
        [
            [
                np.nan,
                np.inf,
                -np.inf,
                np.float32(0.0),
                np.float32(-0.0),
                np.float32(1.25),
                np.float32(-4.5),
                np.float32(8.0),
            ],
            [
                np.float32(-0.0),
                np.nan,
                np.float32(-3.0),
                np.float32(0.0),
                -np.inf,
                np.inf,
                np.float32(-0.125),
                np.float32(0.125),
            ],
        ],
        dtype=FLOAT32,
    )

    expected = np.flip(a, axis=1)
    actual = asnumpy.flip(a, axis=1)

    _assert_exact(expected, actual)
    assert np.array_equal(np.signbit(actual), np.signbit(expected))


def test_public_flip_repeated_calls_are_consistent() -> None:
    asnumpy = _import_asnumpy()

    a = np.arange(24, dtype=FLOAT32).reshape(4, 6)
    before = a.copy()
    expected = np.flip(a, axis=1)

    for _ in range(3):
        actual = asnumpy.flip(a, axis=1)
        _assert_exact(expected, actual)

    _assert_exact(before, a)


def test_public_flip_rejects_float64_without_silent_cast() -> None:
    asnumpy = _import_asnumpy()

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    with pytest.raises(Exception):
        asnumpy.flip(a, axis=1)


def test_public_flip_rejects_unsupported_keywords() -> None:
    asnumpy = _import_asnumpy()

    a = np.zeros((2, 3), dtype=FLOAT32)
    rejection_errors = (TypeError, ValueError, NotImplementedError)

    with pytest.raises(rejection_errors):
        asnumpy.flip(a, axis=1, dtype=FLOAT32)

    with pytest.raises(rejection_errors):
        asnumpy.flip(a, axis=1, out=np.empty_like(a))


def test_public_flip_rejects_out_of_range_axis() -> None:
    asnumpy = _import_asnumpy()

    a = np.zeros((2, 3), dtype=FLOAT32)
    with pytest.raises(Exception):
        asnumpy.flip(a, axis=2)