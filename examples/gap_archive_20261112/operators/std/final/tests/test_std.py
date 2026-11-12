"""Tests for the observed `numpy.std` to `asnumpy.std` gap.

The CPU tests in this module are pure NumPy oracle tests and must collect and
pass without AsNumpy or the custom operator being installed.  Public API tests
import AsNumpy only inside a fixture, call ``asnumpy.std`` rather than a private
pybind module, and are skipped when the public runtime or custom operator is
unavailable.

Observed admitted profile:
    input:  float32, shape (2048, 2048), C-contiguous
    axis:   1
    output: float32, shape (2048,), single tensor

Other axes, shapes, layouts, and input dtypes are outside this unverified
candidate's admitted profile and are not claimed by these tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

OBSERVED_SHAPE = (2048, 2048)
OBSERVED_AXIS = 1
OBSERVED_OUTPUT_SHAPE = (2048,)

# Tolerance for the observed float32 profile.  These values are deliberately
# recorded here because the taskbook tolerance is still unresolved.
FLOAT32_RTOL = 2.0e-4
FLOAT32_ATOL = 2.0e-4
ZERO_ATOL = 1.0e-6

BASE_SEED = 0x5EED


def _observed_input(seed: int = BASE_SEED) -> np.ndarray:
    """Return deterministic float32 data with the observed input shape."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(OBSERVED_SHAPE).astype(np.float32)


def _to_numpy(value):
    """Convert an AsNumpy return value to a NumPy array without importing it."""
    if isinstance(value, np.ndarray):
        return value
    for attr in ("numpy", "asnumpy", "to_numpy"):
        candidate = getattr(value, attr, None)
        if candidate is None:
            continue
        if callable(candidate):
            candidate = candidate()
        return np.asarray(candidate)
    return np.asarray(value)


def _call_public_std(asnumpy_module, std_fn, value, axis: int = OBSERVED_AXIS):
    """Call the public AsNumpy API, accepting NumPy or AsNumpy inputs."""
    try:
        return std_fn(value, axis=axis)
    except TypeError:
        asarray = getattr(asnumpy_module, "asarray", None)
        if asarray is None:
            raise
        return std_fn(asarray(value), axis=axis)


def _looks_like_missing_runtime(exc: BaseException) -> bool:
    """Best-effort classification of an unavailable public runtime.

    Numerical failures after a successful probe are not skipped; only import,
    symbol, device, or missing-operator errors are treated as unavailable.
    """
    if isinstance(exc, (ImportError, ModuleNotFoundError, OSError)):
        return True

    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "aclnn",
        "undefined symbol",
        "cannot open shared object",
        "no module named",
        "module not found",
        "has no attribute",
        "not registered",
        "operator does not exist",
        "operator not found",
        "cannot find operator",
        "cannot find kernel",
        "cann",
        "ascend",
        "npu",
        "device",
    )
    return any(marker in text for marker in markers)


@pytest.fixture(scope="module")
def public_std_runtime():
    """Load ``asnumpy.std`` and skip only when its runtime is unavailable."""
    try:
        import asnumpy  # noqa: PLC0415 - intentionally lazy
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"asnumpy is unavailable: {type(exc).__name__}: {exc}")

    std_fn = getattr(asnumpy, "std", None)
    if not callable(std_fn):
        pytest.skip("asnumpy.std is not exposed by the public AsNumpy module")

    probe = _observed_input(BASE_SEED + 1)
    try:
        _ = _to_numpy(_call_public_std(asnumpy, std_fn, probe))
    except Exception as exc:  # pragma: no cover - environment dependent
        if _looks_like_missing_runtime(exc):
            pytest.skip(
                "asnumpy.std runtime is unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        raise

    return asnumpy, std_fn


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests.  These do not import AsNumpy at module scope.
# ---------------------------------------------------------------------------


def test_oracle_std_observed_profile_shape_dtype_and_value():
    """CPU oracle: observed profile returns float32, shape (2048,)."""
    a = _observed_input()
    expected = np.std(a, axis=OBSERVED_AXIS)

    assert isinstance(expected, np.ndarray)
    assert expected.shape == OBSERVED_OUTPUT_SHAPE
    assert expected.dtype == np.dtype(np.float32)
    assert np.all(np.isfinite(expected))
    assert np.all(expected >= np.float32(0.0))

    # Independent CPU formulation of the same oracle, evaluated in float64.
    mean = np.mean(a, axis=OBSERVED_AXIS, keepdims=True, dtype=np.float64)
    deviations = a.astype(np.float64) - mean
    manual = np.sqrt(np.mean(np.square(deviations), axis=OBSERVED_AXIS))
    np.testing.assert_allclose(
        expected,
        manual,
        rtol=1.0e-5,
        atol=1.0e-6,
        err_msg="numpy.std should implement the two-pass standard deviation",
    )


def test_oracle_std_nan_inf_and_signed_zero_semantics():
    """CPU oracle: record NaN, infinity, and signed-zero behavior."""
    a = np.array(
        [
            [1.0, 1.0],
            [-2.0, -2.0],
            [0.0, 0.0],
            [np.nan, 1.0],
            [np.inf, 1.0],
            [-np.inf, 2.0],
            [np.inf, -np.inf],
        ],
        dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.std(a, axis=1)

    assert out.shape == (7,)
    assert out.dtype == np.dtype(np.float32)

    # Constant rows have positive zero standard deviation.
    np.testing.assert_allclose(out[:3], np.zeros(3, dtype=np.float32), rtol=0.0, atol=0.0)
    assert not np.any(np.signbit(out[:3]))

    # Non-finite input rows produce NaN for this oracle.
    assert np.isnan(out[3:]).all()


def test_oracle_axis_one_is_the_admitted_reduction_axis():
    """CPU oracle: axis=1 reduces the observed 2048x2048 matrix by rows."""
    a = _observed_input(BASE_SEED + 2)
    expected = np.std(a, axis=OBSERVED_AXIS)

    assert expected.shape == (OBSERVED_SHAPE[0],)
    assert expected.shape != (OBSERVED_SHAPE[1],)

    # Explicitly ensure that this is a per-row reduction, not a scalar or
    # per-column reduction.
    row_ids = np.array([0, 17, 1024, 2047], dtype=np.int64)
    for row in row_ids:
        assert np.allclose(
            expected[int(row)],
            np.std(a[int(row), :], axis=0),
            rtol=1.0e-6,
            atol=1.0e-7,
        )


def test_cpu_golden_observed_profile_is_repeatable():
    """CPU oracle: repeated golden evaluation is deterministic."""
    a = _observed_input(BASE_SEED + 3)
    first = np.std(a, axis=OBSERVED_AXIS)
    second = np.std(a, axis=OBSERVED_AXIS)
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Public AsNumpy adapter tests.  These are skipped when the runtime is absent.
# ---------------------------------------------------------------------------


def test_public_api_std_observed_profile_matches_numpy(public_std_runtime):
    """Call public ``asnumpy.std`` on the observed profile and compare to NumPy."""
    asnumpy, std_fn = public_std_runtime
    a = _observed_input()
    original = a.copy()

    out = _to_numpy(_call_public_std(asnumpy, std_fn, a))

    assert isinstance(out, np.ndarray)
    assert out.shape == OBSERVED_OUTPUT_SHAPE
    assert out.dtype == np.dtype(np.float32)

    expected = np.std(a, axis=OBSERVED_AXIS)
    np.testing.assert_allclose(
        out,
        expected,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
        equal_nan=True,
        err_msg="asnumpy.std differs from numpy.std on the observed profile",
    )

    # The adapter must not modify the caller's input in place.
    np.testing.assert_array_equal(a, original)


def test_public_api_std_special_values_and_signed_zero(public_std_runtime):
    """Check NaN propagation and positive-zero results through the public API."""
    asnumpy, std_fn = public_std_runtime

    a = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    a[0, :] = np.float32(1.0)
    a[1, :] = np.float32(-2.5)
    a[2, :] = np.float32(0.0)
    a[3, 0] = np.nan
    a[3, 1:] = np.float32(1.0)
    a[4, 0] = np.inf
    a[4, 1:] = np.float32(1.0)
    a[5, 0] = -np.inf
    a[5, 1:] = np.float32(2.0)
    a[6, 0] = np.inf
    a[6, 1] = -np.inf
    a[6, 2:] = np.float32(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.std(a, axis=OBSERVED_AXIS)

    out = _to_numpy(_call_public_std(asnumpy, std_fn, a))

    assert isinstance(out, np.ndarray)
    assert out.shape == OBSERVED_OUTPUT_SHAPE
    assert out.dtype == np.dtype(np.float32)

    # Constant rows: finite positive zero.
    np.testing.assert_allclose(
        out[:3],
        expected[:3],
        rtol=0.0,
        atol=ZERO_ATOL,
        equal_nan=True,
    )
    assert not np.any(np.signbit(out[:3]))

    # Rows containing NaN or infinity produce NaN for numpy.std.
    assert np.isnan(expected[3:7]).all()
    np.testing.assert_allclose(
        out[3:7],
        expected[3:7],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert np.isnan(out[3:7]).all()

    # Remaining zero rows must also be non-negative zero, not negative zero.
    np.testing.assert_allclose(
        out[7:],
        expected[7:],
        rtol=0.0,
        atol=ZERO_ATOL,
        equal_nan=True,
    )
    assert not np.any(np.signbit(out[7:]))


def test_public_api_std_rejects_float64_without_silent_cast(public_std_runtime):
    """Unadmitted float64 input must be rejected, not silently cast to float32."""
    asnumpy, std_fn = public_std_runtime
    a = np.zeros(OBSERVED_SHAPE, dtype=np.float64)

    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        _call_public_std(asnumpy, std_fn, a)


def test_public_api_std_rejects_unadmitted_dtype_keyword(public_std_runtime):
    """The declared adapter signature admits only ``axis``, not ``dtype``."""
    asnumpy, std_fn = public_std_runtime
    a = _observed_input(BASE_SEED + 4)

    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        std_fn(a, axis=OBSERVED_AXIS, dtype=np.float64)