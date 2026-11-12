"""Tests for the bounded ``asnumpy.pad`` adapter.

The NumPy oracle tests in this module are pure host-side tests and must collect
and pass without an AsNumpy installation or the custom Pad operator.  Runtime
and public-API tests are kept in separate, explicitly skipped test cases so a
missing runtime can never be reported as a successful CPU golden comparison.

Targeted support scope for this draft:
  * ``asnumpy.pad(a, pad_width, mode="constant")``
  * float32 input, rank-2, C-contiguous
  * observed profile: ``(2048, 2048)`` with ``((1, 1), (2, 2))``
  * output shape: ``(2050, 2052)``, float32
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest


# Constant padding only copies values and inserts the constant zero.  There is
# no floating-point arithmetic in the semantic path, so exact comparison is the
# correct tolerance for the supported float32 profile.
FLOAT32_RTOL = 0.0
FLOAT32_ATOL = 0.0

OBSERVED_INPUT_SHAPE = (2048, 2048)
OBSERVED_PAD_WIDTH = ((1, 1), (2, 2))
OBSERVED_OUTPUT_SHAPE = (2050, 2052)
OBSERVED_MODE = "constant"


def _numpy_oracle(
    a: np.ndarray,
    pad_width: object,
    *,
    mode: str = OBSERVED_MODE,
) -> np.ndarray:
    """Oracle wrapper preserving the NumPy parameter order used by the gap."""

    return np.pad(a, pad_width, mode=mode)


def _make_observed_input() -> np.ndarray:
    """Create a deterministic C-contiguous float32 array of the observed shape."""

    rng = np.random.default_rng(20261112)
    return rng.random(OBSERVED_INPUT_SHAPE, dtype=np.float32)


def _assert_float32_equal(actual: object, expected: object) -> None:
    """Compare float32 tensors exactly, including NaNs and signed zero."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)

    assert actual_array.shape == expected_array.shape
    assert actual_array.dtype == np.dtype(np.float32)
    assert expected_array.dtype == np.dtype(np.float32)

    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.signbit(actual_array),
        np.signbit(expected_array),
    )


@pytest.fixture(scope="module")
def asnumpy_pad() -> Callable[..., np.ndarray]:
    """Load the public AsNumpy API only inside guarded runtime tests."""

    try:
        import asnumpy  # noqa: PLC0415 - intentionally not imported at module scope
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        pytest.skip(f"asnumpy runtime is not importable: {exc!r}")

    candidate = getattr(asnumpy, "pad", None)
    if candidate is None or not callable(candidate):
        pytest.skip("asnumpy.pad is not available in the installed runtime")

    return candidate


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests: no AsNumpy and no custom operator required.
# ---------------------------------------------------------------------------


def test_numpy_oracle_matches_literal_constant_pad_example() -> None:
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    actual = _numpy_oracle(a, ((1, 1), (2, 2)), mode="constant")

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    _assert_float32_equal(actual, expected)


def test_numpy_oracle_preserves_float32_special_values_and_signbit() -> None:
    a = np.array(
        [
            [np.nan, np.inf, -np.inf],
            [-0.0, 0.0, 1.5],
        ],
        dtype=np.float32,
    )

    actual = _numpy_oracle(a, ((1, 1), (2, 2)), mode="constant")

    assert actual.shape == (4, 7)
    assert actual.dtype == np.dtype(np.float32)

    interior = actual[1:3, 2:5]
    _assert_float32_equal(interior, a)

    assert np.isnan(interior[0, 0])
    assert np.isposinf(interior[0, 1])
    assert np.isneginf(interior[0, 2])
    assert np.signbit(interior[1, 0])
    assert not np.signbit(interior[1, 1])

    padding_mask = np.ones(actual.shape, dtype=bool)
    padding_mask[1:3, 2:5] = False
    assert np.all(actual[padding_mask] == np.float32(0.0))
    assert not np.any(np.signbit(actual[padding_mask]))


def test_numpy_oracle_observed_profile_shape_and_layout() -> None:
    a = _make_observed_input()
    assert a.shape == OBSERVED_INPUT_SHAPE
    assert a.dtype == np.dtype(np.float32)
    assert a.flags.c_contiguous

    actual = _numpy_oracle(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)

    assert actual.shape == OBSERVED_OUTPUT_SHAPE
    assert actual.dtype == np.dtype(np.float32)
    assert actual.flags.c_contiguous

    _assert_float32_equal(
        actual[1 : 1 + OBSERVED_INPUT_SHAPE[0], 2 : 2 + OBSERVED_INPUT_SHAPE[1]],
        a,
    )
    assert np.all(actual[0, :] == np.float32(0.0))
    assert np.all(actual[-1, :] == np.float32(0.0))
    assert np.all(actual[:, 0] == np.float32(0.0))
    assert np.all(actual[:, 1] == np.float32(0.0))
    assert np.all(actual[:, -2] == np.float32(0.0))
    assert np.all(actual[:, -1] == np.float32(0.0))


def test_numpy_oracle_empty_2d_float32_constant_pad() -> None:
    a = np.empty((0, 0), dtype=np.float32)

    actual = _numpy_oracle(a, ((1, 1), (1, 1)), mode="constant")

    assert actual.shape == (2, 2)
    assert actual.dtype == np.dtype(np.float32)
    _assert_float32_equal(actual, np.zeros((2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Public API tests: separately guarded and skipped when AsNumpy is unavailable.
# ---------------------------------------------------------------------------


def test_public_asnumpy_pad_observed_profile_matches_oracle(
    asnumpy_pad: Callable[..., np.ndarray],
) -> None:
    a = _make_observed_input()
    expected = _numpy_oracle(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)

    # Exactly two positional arguments; mode remains a keyword argument.
    actual = asnumpy_pad(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)

    assert np.asarray(actual).shape == OBSERVED_OUTPUT_SHAPE
    assert np.asarray(actual).dtype == np.dtype(np.float32)
    assert np.asarray(actual).flags.c_contiguous
    _assert_float32_equal(actual, expected)


def test_public_asnumpy_pad_special_values_exact(
    asnumpy_pad: Callable[..., np.ndarray],
) -> None:
    a = np.zeros(OBSERVED_INPUT_SHAPE, dtype=np.float32)
    a[0, 0] = np.nan
    a[0, 1] = np.inf
    a[0, 2] = -np.inf
    a[0, 3] = np.float32(-0.0)
    a[0, 4] = np.float32(0.0)
    a[1, 0] = np.float32(1.25)
    a[-1, -1] = np.float32(-2.5)

    expected = _numpy_oracle(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)
    actual = asnumpy_pad(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)

    _assert_float32_equal(actual, expected)

    actual_array = np.asarray(actual)
    interior = actual_array[
        1 : 1 + OBSERVED_INPUT_SHAPE[0],
        2 : 2 + OBSERVED_INPUT_SHAPE[1],
    ]
    assert np.isnan(interior[0, 0])
    assert np.isposinf(interior[0, 1])
    assert np.isneginf(interior[0, 2])
    assert np.signbit(interior[0, 3])
    assert not np.signbit(interior[0, 4])


def test_public_asnumpy_pad_repeated_input_lifecycle(
    asnumpy_pad: Callable[..., np.ndarray],
) -> None:
    a = _make_observed_input()
    original = a.copy()
    expected = _numpy_oracle(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)

    first = asnumpy_pad(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)
    second = asnumpy_pad(a, OBSERVED_PAD_WIDTH, mode=OBSERVED_MODE)

    _assert_float32_equal(a, original)
    _assert_float32_equal(first, expected)
    _assert_float32_equal(second, expected)
    _assert_float32_equal(first, second)


@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float16, np.int32],
)
def test_public_asnumpy_pad_rejects_non_float32_input(
    asnumpy_pad: Callable[..., np.ndarray],
    dtype: object,
) -> None:
    a = np.ones((8, 8), dtype=dtype)

    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        asnumpy_pad(a, ((1, 1), (2, 2)), mode=OBSERVED_MODE)


def test_public_asnumpy_pad_rejects_unsupported_mode(
    asnumpy_pad: Callable[..., np.ndarray],
) -> None:
    a = np.zeros(OBSERVED_INPUT_SHAPE, dtype=np.float32)

    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        asnumpy_pad(a, OBSERVED_PAD_WIDTH, mode="reflect")


@pytest.mark.parametrize(
    "bad_pad_width",
    [
        ((1, 1),),
        ((1, 1), (2, 2), (3, 3)),
        ((1,), (2, 2)),
    ],
)
def test_public_asnumpy_pad_rejects_malformed_pad_width(
    asnumpy_pad: Callable[..., np.ndarray],
    bad_pad_width: object,
) -> None:
    a = np.zeros(OBSERVED_INPUT_SHAPE, dtype=np.float32)

    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        asnumpy_pad(a, bad_pad_width, mode=OBSERVED_MODE)


def test_public_asnumpy_pad_rejects_non_2d_input(
    asnumpy_pad: Callable[..., np.ndarray],
) -> None:
    a = np.ones((16,), dtype=np.float32)

    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        asnumpy_pad(a, ((1, 1),), mode=OBSERVED_MODE)