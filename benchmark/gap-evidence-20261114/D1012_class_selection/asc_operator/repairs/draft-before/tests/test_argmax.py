"""NumPy-oracle and guarded public-API tests for asnumpy.argmax.

The CPU/oracle tests in this file always collect and run with only NumPy.
The public AsNumpy tests import asnumpy lazily and skip when either AsNumpy
or the custom Ascend operator runtime is unavailable. A skipped runtime test
is never reported as a CPU-golden pass.

This is an unverified draft test module for the observed float32 profile:
    input:  shape (1024, 1024), dtype float32, C-contiguous
    axis:   1
    output: shape (1024,), dtype int64, C-contiguous
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


OBSERVED_SHAPE = (1024, 1024)
OBSERVED_AXIS = 1
OBSERVED_OUTPUT_SHAPE = (1024,)
RANDOM_SEED = 0xA6A1

# argmax returns integer indices. The required comparison is exact; there is
# no floating-point tolerance for the returned index values themselves.
INDEX_RTOL = 0.0
INDEX_ATOL = 0.0


def _make_observed_input() -> np.ndarray:
    """Build a deterministic float32 input matching the observed profile."""
    rng = np.random.default_rng(RANDOM_SEED)
    data = rng.standard_normal(OBSERVED_SHAPE).astype(np.float32, copy=False)

    # Inject ties and signed zeros so first-occurrence behavior is exercised
    # by the observed-profile numerical check as well as by small oracle tests.
    data[:, 5] = data[:, 4]
    data[:, 11] = 0.0
    data[:, 12] = -0.0
    return np.ascontiguousarray(data, dtype=np.float32)


def _import_asnumpy_or_skip():
    """Import the public AsNumpy package only when a runtime test needs it."""
    try:
        return importlib.import_module("asnumpy")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"asnumpy is not importable in this environment: {exc}")


_RUNTIME_UNAVAILABLE_HINTS = (
    "asnumpy",
    "aclnn",
    "ascend",
    "cann",
    "npu",
    "device",
    "custom op",
    "operator library",
    "not found",
    "cannot find",
    "could not find",
    "failed to load",
    "cannot load",
    "could not load",
    "not loaded",
    "not registered",
    "undefined symbol",
    "no such file",
    "libascend",
    "runtime",
)


def _is_runtime_unavailable(exc: BaseException) -> bool:
    """Best-effort detection of a missing native/device runtime.

    This is used only for the valid-profile preflight probe. Once the preflight
    succeeds, numerical mismatches and real implementation errors are allowed
    to fail the tests rather than being converted into skips.
    """
    if isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError, OSError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _RUNTIME_UNAVAILABLE_HINTS)


def _to_numpy(value) -> np.ndarray:
    """Convert a public-API result to a NumPy array without host fallback."""
    if isinstance(value, np.ndarray):
        return np.asarray(value)

    for name in ("numpy", "to_numpy"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                converted = method()
            except TypeError:
                continue
            return np.asarray(converted)

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            converted = cpu()
        except TypeError:
            converted = None
        if converted is not None:
            return _to_numpy(converted)

    return np.asarray(value)


def _call_public_argmax(module, value: np.ndarray, axis: int) -> np.ndarray:
    """Call the public API asnumpy.argmax(value, axis=axis)."""
    result = module.argmax(value, axis=axis)
    return _to_numpy(result)


@pytest.fixture(scope="module")
def asnumpy_runtime():
    """Return the public AsNumpy module after a valid-profile runtime probe."""
    module = _import_asnumpy_or_skip()
    if not callable(getattr(module, "argmax", None)):
        pytest.skip("asnumpy.argmax is not available")

    probe = _make_observed_input()
    try:
        module.argmax(probe, axis=OBSERVED_AXIS)
    except Exception as exc:  # pragma: no cover - environment dependent
        if _is_runtime_unavailable(exc):
            pytest.skip(f"custom Ascend argmax runtime is unavailable: {exc}")
        raise

    return module


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests: these do not import asnumpy and always run.
# ---------------------------------------------------------------------------


def test_oracle_axis_one_known_values() -> None:
    """Lock the NumPy axis=1 index contract on a hand-checkable matrix."""
    x = np.array(
        [
            [1.0, 3.0, 3.0, 0.0],
            [-1.0, -2.0, -0.5, -3.0],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )

    actual = np.argmax(x, axis=1)
    expected = np.array([1, 2, 0], dtype=np.int64)

    assert actual.shape == (3,)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)
    assert INDEX_RTOL == 0.0
    assert INDEX_ATOL == 0.0


def test_oracle_tie_breaking_is_first_occurrence() -> None:
    """NumPy argmax returns the first index on ties and equal signed zeros."""
    x = np.array(
        [
            [7.0, 7.0, 1.0, 7.0],
            [0.0, 9.0, 9.0, 9.0],
            [-0.0, 0.0, -1.0, -0.0],
        ],
        dtype=np.float32,
    )

    actual = np.argmax(x, axis=1)
    expected = np.array([0, 1, 0], dtype=np.int64)

    np.testing.assert_array_equal(actual, expected)


def test_oracle_nan_inf_signed_zero_semantics() -> None:
    """Document NumPy argmax behavior for NaN, infinities, and signed zeros."""
    x = np.array(
        [
            [1.0, np.nan, 2.0, 0.0],
            [np.nan, 1.0, np.inf, -np.inf],
            [np.inf, 1.0, np.inf, 0.0],
            [-np.inf, -1.0, -np.inf, -2.0],
            [0.0, -0.0, 0.0, -0.0],
        ],
        dtype=np.float32,
    )

    # NumPy argmax treats the first NaN in a reduction row as the selected
    # value, and otherwise selects the first occurrence of the maximum.
    expected = np.array([1, 0, 0, 1, 0], dtype=np.int64)
    actual = np.argmax(x, axis=1)

    np.testing.assert_array_equal(actual, expected)


def test_oracle_observed_output_metadata() -> None:
    """The observed profile must produce shape (1024,), dtype int64."""
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    actual = np.argmax(x, axis=OBSERVED_AXIS)

    assert actual.shape == OBSERVED_OUTPUT_SHAPE
    assert actual.dtype == np.int64
    assert actual.flags.c_contiguous


def test_oracle_axis_out_of_range_raises() -> None:
    """NumPy rejects an axis outside the input rank."""
    x = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises((ValueError, IndexError)):
        np.argmax(x, axis=2)


def test_oracle_does_not_mutate_input() -> None:
    """The oracle must not change its input tensor."""
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    before = x.copy()

    _ = np.argmax(x, axis=1)

    np.testing.assert_array_equal(x, before)


# ---------------------------------------------------------------------------
# Public AsNumpy tests: lazily imported and skipped without a runtime.
# ---------------------------------------------------------------------------


def test_public_api_observed_profile(asnumpy_runtime) -> None:
    """Compare asnumpy.argmax against NumPy on the observed profile."""
    x = _make_observed_input()
    before = x.copy()

    actual = _call_public_argmax(asnumpy_runtime, x, OBSERVED_AXIS)
    expected = np.argmax(x, axis=OBSERVED_AXIS)

    assert actual.shape == expected.shape == OBSERVED_OUTPUT_SHAPE
    assert actual.dtype == np.int64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, before)


def test_public_api_nan_inf_signed_zero(asnumpy_runtime) -> None:
    """Compare special-value behavior against the NumPy oracle."""
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    rows = np.arange(OBSERVED_SHAPE[0])

    x[:, 0] = -np.inf
    x[:, 1] = 0.0
    x[:, 2] = np.inf
    x[:, 3] = np.nan
    x[:, 4] = -0.0

    x[rows % 4 == 0, 5] = np.nan
    x[rows % 4 == 1, 5] = np.inf
    x[rows % 4 == 2, 5] = -np.inf
    x[rows % 4 == 3, 5] = 0.0

    expected = np.argmax(x, axis=OBSERVED_AXIS)
    actual = _call_public_argmax(asnumpy_runtime, x, OBSERVED_AXIS)

    assert actual.shape == expected.shape
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)


def test_public_api_repeated_calls_are_deterministic(asnumpy_runtime) -> None:
    """Repeated calls with the same retained input must be deterministic."""
    x = _make_observed_input()
    before = x.copy()

    first = _call_public_argmax(asnumpy_runtime, x, OBSERVED_AXIS)
    second = _call_public_argmax(asnumpy_runtime, x, OBSERVED_AXIS)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.argmax(x, axis=OBSERVED_AXIS))
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32])
def test_public_api_rejects_unprofiled_dtype(asnumpy_runtime, dtype) -> None:
    """Only float32 is in scope; other dtypes must not be silently cast."""
    x = np.zeros(OBSERVED_SHAPE, dtype=dtype)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        asnumpy_runtime.argmax(x, axis=OBSERVED_AXIS)


def test_public_api_rejects_unprofiled_rank(asnumpy_runtime) -> None:
    """The observed input rank is 2; an unprofiled rank must be rejected."""
    x = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        asnumpy_runtime.argmax(x, axis=1)


def test_public_api_rejects_invalid_axis(asnumpy_runtime) -> None:
    """An axis outside the input rank must be rejected explicitly."""
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)

    with pytest.raises((TypeError, ValueError, IndexError, RuntimeError)):
        asnumpy_runtime.argmax(x, axis=2)


def test_public_api_rejects_unprofiled_keywords(asnumpy_runtime) -> None:
    """The generated public signature admits the axis keyword only."""
    x = np.zeros(OBSERVED_SHAPE, dtype=np.float32)
    out = np.empty(OBSERVED_OUTPUT_SHAPE, dtype=np.int64)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        asnumpy_runtime.argmax(x, axis=OBSERVED_AXIS, out=out)