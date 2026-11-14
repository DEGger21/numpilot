"""Contract tests for the draft ``asnumpy.spacing`` operator.

The tests in this module are split into two groups:

* Pure NumPy oracle tests.  These only use ``numpy.spacing`` and therefore
  collect and pass on a CPU-only host where neither ``asnumpy`` nor the
  custom Ascend operator is installed.
* Runtime/public-API tests.  These import and call ``asnumpy.spacing`` only
  inside guarded test bodies.  A missing runtime is reported as an explicit
  pytest skip and is never counted as a successful CPU golden test.

This file corresponds to the still-unverified Ascend C draft for
``numpy.spacing`` -> ``asnumpy.spacing``.  It does not assert qualification,
performance, or activation.
"""

import numpy as np
import pytest


_FLOAT32 = np.dtype(np.float32)
_OBSERVED_SHAPE = (1024, 1024)

# NumPy's spacing is a bit-level contract.  The custom operator is expected
# to match the float32 oracle exactly, including subnormals and signed zero.
_RTOL = 0.0
_ATOL = 0.0


def _smallest_positive_float32():
    """Return the smallest positive float32 subnormal without warnings."""
    return np.float32(
        np.nextafter(np.float32(0.0), np.float32(1.0))
    )


def _assert_spacing_equal(actual, expected):
    """Assert exact float32 spacing equality including NaN/sign handling."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    assert actual.dtype == _FLOAT32, f"expected float32, got {actual.dtype}"
    assert expected.dtype == _FLOAT32, f"oracle dtype is {expected.dtype}"
    assert actual.shape == expected.shape, (
        f"shape mismatch: actual {actual.shape}, expected {expected.shape}"
    )

    expected_nan = np.isnan(expected)
    actual_nan = np.isnan(actual)
    np.testing.assert_array_equal(
        actual_nan,
        expected_nan,
        err_msg="NaN positions differ from the NumPy oracle",
    )

    valid = ~expected_nan
    if np.any(valid):
        np.testing.assert_array_equal(
            actual[valid],
            expected[valid],
            err_msg="spacing values differ from the NumPy oracle",
        )
        np.testing.assert_array_equal(
            np.signbit(actual[valid]),
            np.signbit(expected[valid]),
            err_msg="spacing sign bits differ from the NumPy oracle",
        )

    # Keep the declared exact tolerances explicit; these are intentional
    # constants for a later numerical-qualification receipt.
    assert _RTOL == 0.0 and _ATOL == 0.0


def _asnumpy_spacing_or_skip():
    """Import ``asnumpy.spacing`` lazily, skipping only when unavailable."""
    try:
        import asnumpy  # local import: never import at module scope
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"asnumpy runtime is not importable: {exc!r}")

    spacing = getattr(asnumpy, "spacing", None)
    if spacing is None:
        pytest.skip(
            "asnumpy.spacing is not exposed; the custom adapter is not active"
        )
    if not callable(spacing):
        pytest.fail("asnumpy.spacing exists but is not callable")
    if spacing is np.spacing:
        pytest.fail(
            "asnumpy.spacing must not alias numpy.spacing; host fallback is forbidden"
        )
    return spacing


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests: no asnumpy import, no custom-operator dependency.
# ---------------------------------------------------------------------------


def test_numpy_oracle_spacing_observed_profile_shape_and_dtype():
    values = np.linspace(
        -10.0,
        10.0,
        num=_OBSERVED_SHAPE[0] * _OBSERVED_SHAPE[1],
        dtype=np.float32,
    ).reshape(_OBSERVED_SHAPE)

    actual = np.spacing(values)

    assert actual.shape == _OBSERVED_SHAPE
    assert actual.dtype == _FLOAT32
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= np.float32(0.0))
    assert not np.any(np.signbit(actual))


def test_numpy_oracle_spacing_float32_special_values():
    smallest = _smallest_positive_float32()
    tiny = np.finfo(np.float32).tiny
    float_max = np.finfo(np.float32).max

    values = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            tiny,
            -tiny,
            smallest,
            -smallest,
            np.inf,
            -np.inf,
            np.nan,
            float_max,
            -float_max,
        ],
        dtype=np.float32,
    )

    with np.errstate(all="ignore"):
        actual = np.spacing(values)

    assert actual.shape == values.shape
    assert actual.dtype == _FLOAT32

    np.testing.assert_array_equal(
        actual[0:2],
        np.array([smallest, smallest], dtype=np.float32),
    )
    assert not np.any(np.signbit(actual[0:2]))

    assert actual[2] == np.float32(2.0 ** -23)
    assert np.isfinite(actual[3])
    assert actual[3] > np.float32(0.0)
    assert not np.signbit(actual[3])

    assert actual[4] == smallest
    assert np.isfinite(actual[5])
    assert actual[5] > np.float32(0.0)
    assert not np.signbit(actual[5])
    assert actual[6] == smallest
    assert np.isfinite(actual[7])
    assert actual[7] > np.float32(0.0)
    assert not np.signbit(actual[7])

    assert np.isnan(actual[8])
    assert np.isnan(actual[9])
    assert np.isnan(actual[10])

    assert not np.isnan(actual[11])
    assert actual[11] > np.float32(0.0)
    assert not np.signbit(actual[11])
    assert not np.isnan(actual[12])
    assert actual[12] > np.float32(0.0)
    assert not np.signbit(actual[12])


def test_numpy_oracle_spacing_signed_zero_and_subnormal_contract():
    smallest = _smallest_positive_float32()
    tiny = np.finfo(np.float32).tiny

    values = np.array(
        [0.0, -0.0, smallest, -smallest, tiny, -tiny],
        dtype=np.float32,
    )
    expected = np.full(values.shape, smallest, dtype=np.float32)

    actual = np.spacing(values)

    _assert_spacing_equal(actual, expected)
    assert not np.any(np.signbit(actual))


def test_numpy_oracle_spacing_infinity_and_nan_contract():
    values = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    actual = np.spacing(values)

    assert actual.dtype == _FLOAT32
    assert actual.shape == values.shape
    assert np.all(np.isnan(actual))
    assert not np.any(np.signbit(actual))


def test_numpy_oracle_spacing_empty_float32_input():
    values = np.empty((0, 5), dtype=np.float32)
    actual = np.spacing(values)

    assert actual.shape == values.shape
    assert actual.dtype == _FLOAT32
    assert actual.size == 0


def test_numpy_oracle_spacing_positive_normal_formula():
    x = np.array(
        [
            _smallest_positive_float32(),
            np.float32(1.0e-30),
            np.float32(0.5),
            np.float32(1.0),
            np.float32(2.0),
            np.float32(3.0),
            np.float32(17.0),
            np.float32(1.0e30),
        ],
        dtype=np.float32,
    )

    forward = np.nextafter(x, np.float32(np.inf)).astype(np.float32)
    expected = (forward - x).astype(np.float32)
    actual = np.spacing(x)

    _assert_spacing_equal(actual, expected)


def test_numpy_oracle_spacing_preserves_float32_shape():
    values = np.arange(-6, 6, dtype=np.float32).reshape(3, 4)
    actual = np.spacing(values)

    assert actual.dtype == _FLOAT32
    assert actual.shape == values.shape


# ---------------------------------------------------------------------------
# Guarded public-API tests: require an activated asnumpy runtime and the
# matching custom operator.
# ---------------------------------------------------------------------------


def test_asnumpy_spacing_public_api_is_exposed():
    spacing = _asnumpy_spacing_or_skip()
    assert callable(spacing)


def test_asnumpy_spacing_matches_numpy_oracle_observed_profile():
    spacing = _asnumpy_spacing_or_skip()

    rng = np.random.default_rng(0x5FAC1A)
    values = rng.standard_normal(_OBSERVED_SHAPE).astype(np.float32)
    values[0, 0] = np.float32(0.0)
    values[0, 1] = np.float32(-0.0)
    values[0, 2] = np.float32(np.inf)
    values[0, 3] = np.float32(-np.inf)
    values[0, 4] = np.float32(np.nan)

    expected = np.spacing(values)
    actual = spacing(values)

    _assert_spacing_equal(actual, expected)


def test_asnumpy_spacing_matches_numpy_oracle_special_values():
    spacing = _asnumpy_spacing_or_skip()

    smallest = _smallest_positive_float32()
    tiny = np.finfo(np.float32).tiny
    float_max = np.finfo(np.float32).max

    values = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            tiny,
            -tiny,
            smallest,
            -smallest,
            np.inf,
            -np.inf,
            np.nan,
            float_max,
            -float_max,
        ],
        dtype=np.float32,
    )

    expected = np.spacing(values)
    actual = spacing(values.copy())

    _assert_spacing_equal(actual, expected)


def test_asnumpy_spacing_empty_float32_input():
    spacing = _asnumpy_spacing_or_skip()

    values = np.empty((0, 7), dtype=np.float32)
    expected = np.spacing(values)
    actual = spacing(values)

    _assert_spacing_equal(actual, expected)


def test_asnumpy_spacing_repeated_calls_do_not_mutate_input():
    spacing = _asnumpy_spacing_or_skip()

    rng = np.random.default_rng(0xA5C11)
    values = rng.standard_normal((16, 16)).astype(np.float32)
    before = values.copy()

    expected = np.spacing(values)
    first = spacing(values)
    second = spacing(values)

    _assert_spacing_equal(first, expected)
    _assert_spacing_equal(second, expected)
    np.testing.assert_array_equal(
        values,
        before,
        err_msg="asnumpy.spacing modified its input tensor",
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.int64])
def test_asnumpy_spacing_rejects_non_float32_dtype(dtype):
    spacing = _asnumpy_spacing_or_skip()

    # A supported probe makes device/runtime failures distinguishable from
    # the intended dtype rejection when this runs on an NPU host.
    probe = np.ones((2, 2), dtype=np.float32)
    spacing(probe)

    values = np.ones((2, 2), dtype=dtype)
    with pytest.raises((TypeError, ValueError)):
        spacing(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": np.float32},
        {"out": np.empty((2, 2), dtype=np.float32)},
        {"where": np.ones((2, 2), dtype=bool)},
    ],
)
def test_asnumpy_spacing_rejects_unsupported_keyword_arguments(kwargs):
    spacing = _asnumpy_spacing_or_skip()

    values = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        spacing(values, **kwargs)