"""Contract and adapter tests for ``asnumpy.ediff1d``.

The CPU-only tests below use ``numpy.ediff1d`` as the semantic oracle and do
not import ``asnumpy`` at module import time.  Public AsNumpy tests import the
package lazily and are skipped when the runtime or custom operator is absent.

Name mapping expected from the integration:
``asnumpy.ediff1d`` -> Python adapter wrapper -> pybind native binding ->
``aclnnEdiff1dGetWorkspaceSize`` / ``aclnnEdiff1d``.
"""

import importlib.util

import numpy as np
import pytest


OBSERVED_SHAPE = (2048, 2048)
OBSERVED_ELEMENTS = OBSERVED_SHAPE[0] * OBSERVED_SHAPE[1]
OBSERVED_OUTPUT_SHAPE = (OBSERVED_ELEMENTS - 1,)

# Explicit float32 validation thresholds for this draft candidate.  The
# operation is a direct subtraction, so finite results should be bitwise or
# nearly bitwise identical; signed zeros, NaNs, and infinities are checked
# exactly by the helper below.
FLOAT32_RTOL = 1e-6
FLOAT32_ATOL = 1e-6

_MISSING_RUNTIME_MARKERS = (
    "aclnn",
    "ascend",
    "npu",
    "cann",
    "custom op",
    "custom operator",
    "operator not found",
    "not registered",
    "not built",
    "run package",
    "undefined symbol",
    "cannot find",
    "failed to find",
    "device",
)


def _looks_like_missing_runtime(exc):
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in _MISSING_RUNTIME_MARKERS)


def _import_asnumpy_ediff1d():
    """Return the public adapter callable, or skip if unavailable."""
    if importlib.util.find_spec("asnumpy") is None:
        pytest.skip("asnumpy is not installed in this test environment")

    try:
        import asnumpy  # pylint: disable=import-outside-toplevel
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"asnumpy runtime is unavailable: {exc}")
    except Exception as exc:  # pragma: no cover - exact environment exception
        if _looks_like_missing_runtime(exc):
            pytest.skip(f"asnumpy runtime is unavailable: {exc}")
        raise

    public_fn = getattr(asnumpy, "ediff1d", None)
    if public_fn is None:
        pytest.skip("asnumpy.ediff1d is not present in this build")
    if not callable(public_fn):
        pytest.fail("asnumpy.ediff1d exists but is not callable")
    return public_fn


def _call_public(public_fn, x):
    """Call asnumpy.ediff1d and skip only for unavailable runtime cases."""
    try:
        return public_fn(x)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"asnumpy runtime is unavailable: {exc}")
    except RuntimeError as exc:
        if _looks_like_missing_runtime(exc):
            pytest.skip(f"custom ediff1d operator is unavailable: {exc}")
        raise
    except Exception as exc:  # pragma: no cover - exact environment exception
        if isinstance(exc, AttributeError) and _looks_like_missing_runtime(exc):
            pytest.skip(f"custom ediff1d operator is unavailable: {exc}")
        raise


def _expect_rejection(public_fn, x, *, args=(), kwargs=None, allowed=(TypeError, ValueError)):
    """Require an explicit adapter rejection without hiding missing runtime."""
    call_kwargs = {} if kwargs is None else kwargs
    try:
        public_fn(x, *args, **call_kwargs)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"asnumpy runtime is unavailable: {exc}")
    except RuntimeError as exc:
        if _looks_like_missing_runtime(exc):
            pytest.skip(f"custom ediff1d operator is unavailable: {exc}")
        if RuntimeError in allowed:
            return
        raise
    except Exception as exc:
        if isinstance(exc, allowed):
            return
        if isinstance(exc, AttributeError) and _looks_like_missing_runtime(exc):
            pytest.skip(f"custom ediff1d operator is unavailable: {exc}")
        raise
    else:
        pytest.fail("unsupported input was accepted instead of explicitly rejected")


def _assert_float32_matches(actual, expected, *, rtol, atol):
    """Compare float32 outputs while preserving special values and sign bits."""
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)

    assert expected_arr.dtype == np.dtype(np.float32)
    assert actual_arr.dtype == np.dtype(np.float32)
    assert actual_arr.shape == expected_arr.shape

    actual_finite = np.isfinite(actual_arr)
    expected_finite = np.isfinite(expected_arr)
    np.testing.assert_array_equal(
        actual_finite,
        expected_finite,
        err_msg="finite/non-finite masks differ",
    )

    finite_mask = expected_finite
    if np.any(finite_mask):
        np.testing.assert_allclose(
            actual_arr[finite_mask],
            expected_arr[finite_mask],
            rtol=rtol,
            atol=atol,
            equal_nan=False,
            err_msg="finite float32 values differ from the NumPy oracle",
        )

    inf_mask = np.isinf(expected_arr)
    if np.any(inf_mask):
        assert np.isinf(actual_arr[inf_mask]).all(), (
            "actual output contains a non-infinity where the oracle has infinity"
        )
        np.testing.assert_array_equal(
            np.signbit(actual_arr[inf_mask]),
            np.signbit(expected_arr[inf_mask]),
            err_msg="positive/negative infinity sign differs from the NumPy oracle",
        )

    nan_mask = np.isnan(expected_arr)
    if np.any(nan_mask):
        assert np.isnan(actual_arr[nan_mask]).all(), (
            "actual output contains a non-NaN where the oracle has NaN"
        )

    zero_mask = expected_arr == np.float32(0.0)
    if np.any(zero_mask):
        assert np.all(actual_arr[zero_mask] == np.float32(0.0)), (
            "actual output does not preserve exact zeros where the oracle has zeros"
        )
        np.testing.assert_array_equal(
            np.signbit(actual_arr[zero_mask]),
            np.signbit(expected_arr[zero_mask]),
            err_msg="signed-zero sign differs from the NumPy oracle",
        )


def _make_observed_input():
    """Create a deterministic float32 input at the observed gap profile."""
    flat = np.linspace(
        np.float32(-2.0),
        np.float32(3.0),
        num=OBSERVED_ELEMENTS,
        dtype=np.float32,
    )

    # Equal neighbours produce exact +0.0.
    flat[0] = np.float32(7.0)
    flat[1] = np.float32(7.0)

    # Exercise signed-zero differences.
    flat[2] = np.float32(0.0)
    flat[3] = np.float32(-0.0)
    flat[4] = np.float32(-0.0)

    # Exercise NaN and infinities in the same observed profile.
    flat[1000] = np.float32(np.inf)
    flat[1001] = np.float32(np.inf)
    flat[1002] = np.float32(-np.inf)
    flat[1003] = np.float32(np.nan)
    flat[1004] = np.float32(1.0)

    # Exercise repeated signed-zero differences in another region.
    flat[2_000_000] = np.float32(-0.0)
    flat[2_000_001] = np.float32(-0.0)

    return flat.reshape(OBSERVED_SHAPE)


@pytest.fixture(scope="session")
def live_public_api():
    """Import and smoke-check the public adapter at the observed profile."""
    public_fn = _import_asnumpy_ediff1d()
    x = _make_observed_input()
    expected = np.ediff1d(x)
    first_output = _call_public(public_fn, x)
    _assert_float32_matches(
        first_output,
        expected,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    return {
        "public_fn": public_fn,
        "input": x,
        "expected": expected,
        "first_output": first_output,
    }


def test_oracle_flattens_in_c_order_and_subtracts_adjacent_values():
    x = np.array(
        [[1.0, 3.0, 6.0], [10.0, 15.0, 21.0]],
        dtype=np.float32,
    )
    expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    actual = np.ediff1d(x)

    assert actual.shape == (5,)
    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (1,),
        (0, 2),
        (2, 0),
        (1, 1),
    ],
)
def test_oracle_empty_and_singleton_inputs(shape):
    x = np.empty(shape, dtype=np.float32)

    actual = np.ediff1d(x)

    assert actual.shape == (max(x.size - 1, 0),)
    assert actual.dtype == np.dtype(np.float32)


def test_oracle_preserves_float32_dtype():
    x = np.array([np.float32(0.5), np.float32(0.25)], dtype=np.float32)

    actual = np.ediff1d(x)

    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual, np.array([-0.25], dtype=np.float32))


def test_oracle_preserves_signed_zero_signs():
    x = np.array([0.0, -0.0, -0.0, 0.0], dtype=np.float32)

    actual = np.ediff1d(x)

    assert actual.dtype == np.dtype(np.float32)
    assert actual.shape == (3,)
    np.testing.assert_array_equal(
        np.signbit(actual),
        np.array([True, False, False], dtype=bool),
    )


def test_oracle_preserves_nan_and_infinity_results():
    x = np.array([np.inf, np.inf, -np.inf, np.nan, 1.0], dtype=np.float32)

    actual = np.ediff1d(x)

    assert actual.dtype == np.dtype(np.float32)
    assert actual.shape == (4,)
    assert np.isnan(actual[0])
    assert np.isneginf(actual[1])
    assert np.isnan(actual[2])
    assert np.isnan(actual[3])


def test_oracle_positive_infinity_result():
    x = np.array([-np.inf, np.inf], dtype=np.float32)

    actual = np.ediff1d(x)

    assert actual.dtype == np.dtype(np.float32)
    assert actual.shape == (1,)
    assert np.isposinf(actual[0])


def test_observed_profile_oracle_contract():
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    assert x.flags.c_contiguous

    actual = np.ediff1d(x)

    assert actual.shape == OBSERVED_OUTPUT_SHAPE
    assert actual.dtype == np.dtype(np.float32)
    assert np.count_nonzero(actual) == 0


def test_oracle_keyword_extensions_are_outside_declared_public_signature():
    x = np.array([1.0, 3.0], dtype=np.float32)

    actual = np.ediff1d(
        x,
        to_begin=np.array([-1.0], dtype=np.float32),
        to_end=np.array([9.0], dtype=np.float32),
    )

    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        actual,
        np.array([-1.0, 2.0, 9.0], dtype=np.float32),
    )


def test_public_api_observed_profile_matches_numpy_oracle(live_public_api):
    public_fn = live_public_api["public_fn"]
    x = live_public_api["input"]
    expected = live_public_api["expected"]

    assert expected.shape == OBSERVED_OUTPUT_SHAPE
    assert expected.dtype == np.dtype(np.float32)

    first_output = live_public_api["first_output"]
    _assert_float32_matches(
        first_output,
        expected,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )

    second_output = _call_public(public_fn, x)
    _assert_float32_matches(
        second_output,
        expected,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


def test_public_api_is_not_the_numpy_cpu_function(live_public_api):
    public_fn = live_public_api["public_fn"]
    assert public_fn is not np.ediff1d


def test_public_api_rejects_float64_without_silent_cast(live_public_api):
    public_fn = live_public_api["public_fn"]
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float64)

    _expect_rejection(
        public_fn,
        x,
        allowed=(TypeError, ValueError, RuntimeError),
    )


@pytest.mark.parametrize("keyword_name", ["to_begin", "to_end"])
def test_public_api_rejects_numpy_extension_keywords(live_public_api, keyword_name):
    public_fn = live_public_api["public_fn"]
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    extra = np.array([0.0], dtype=np.float32)

    _expect_rejection(
        public_fn,
        x,
        kwargs={keyword_name: extra},
        allowed=(TypeError, ValueError),
    )


def test_public_api_rejects_second_positional_argument(live_public_api):
    public_fn = live_public_api["public_fn"]
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)

    _expect_rejection(
        public_fn,
        x,
        args=(x,),
        allowed=(TypeError,),
    )