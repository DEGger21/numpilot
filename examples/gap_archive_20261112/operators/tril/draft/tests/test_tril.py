"""Tests for the public ``asnumpy.tril`` adapter and the NumPy oracle.

The CPU tests in this file only depend on NumPy and pytest.  They always run,
including in environments where ``asnumpy`` and the custom ACLNN operator are
not installed.

Public-API tests against ``asnumpy.tril`` are deliberately not imported at
module scope.  They are opt-in device tests and can be enabled with:

    NUMPILOT_ENABLE_DEVICE_TESTS=1

When that variable is unset, or when ``asnumpy``/``asnumpy.tril`` is absent,
the runtime tests skip.  A skip is never used as a substitute for a passing
CPU golden test.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Tuple

import numpy as np
import pytest

# Public API mapping:
#   asnumpy.tril -> Python adapter -> pybind native wrapper
#                -> ACLNN aclnnTrilGetWorkspaceSize / aclnnTril
TRIL_PUBLIC_API = "asnumpy.tril"
DEVICE_ENV = "NUMPILOT_ENABLE_DEVICE_TESTS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FLOAT32 = np.dtype("float32")
_OBSERVED_SHAPE = (2048, 2048)


def numpy_oracle(a: Any) -> np.ndarray:
    """Semantic source of truth for this operator."""
    return np.tril(a)


def _to_numpy_host(value: Any, asnumpy_module: Any | None = None) -> np.ndarray:
    """Convert an AsNumpy/torch-like tensor to a host NumPy array."""
    if isinstance(value, np.ndarray):
        return value

    if asnumpy_module is not None:
        converter = getattr(asnumpy_module, "asnumpy", None)
        if callable(converter):
            try:
                converted = converter(value)
            except (TypeError, ValueError):
                converted = None
            except RuntimeError:
                converted = None
            if converted is not None and converted is not value:
                return _to_numpy_host(converted, asnumpy_module)

    for method_name in ("asnumpy", "numpy", "to_numpy"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
            except TypeError:
                continue
            if converted is not value:
                return _to_numpy_host(converted, asnumpy_module)

    cpu_method = getattr(value, "cpu", None)
    if callable(cpu_method):
        try:
            converted = cpu_method()
        except TypeError:
            converted = None
        if converted is not None and converted is not value:
            return _to_numpy_host(converted, asnumpy_module)

    try:
        converted = np.asarray(value)
    except Exception as exc:  # pragma: no cover - conversion failure path
        raise AssertionError(
            f"cannot convert value of type {type(value)!r} to a host NumPy array"
        ) from exc
    return converted


def _assert_float32_semantics(
    actual: Any,
    expected: Any,
    *,
    context: str = "",
) -> None:
    """Exact float32 comparison preserving NaN, Inf and zero sign bits."""
    actual_array = np.asarray(_to_numpy_host(actual))
    expected_array = np.asarray(expected)

    prefix = f"{context}: " if context else ""

    if actual_array.dtype != _FLOAT32:
        raise AssertionError(
            f"{prefix}expected float32 output, got dtype {actual_array.dtype!r}"
        )
    if expected_array.dtype != _FLOAT32:
        raise AssertionError(
            f"{prefix}oracle expected dtype float32, got {expected_array.dtype!r}"
        )
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{prefix}expected shape {expected_array.shape}, "
            f"got {actual_array.shape}"
        )

    actual_nan = np.isnan(actual_array)
    expected_nan = np.isnan(expected_array)
    np.testing.assert_array_equal(
        actual_nan,
        expected_nan,
        err_msg=f"{prefix}NaN positions differ",
    )

    finite = ~expected_nan
    np.testing.assert_array_equal(
        actual_array[finite],
        expected_array[finite],
        err_msg=f"{prefix}finite/Inf elements differ from NumPy oracle",
    )

    zero_mask = finite & (expected_array == np.float32(0.0))
    if np.any(zero_mask):
        np.testing.assert_array_equal(
            np.signbit(actual_array[zero_mask]),
            np.signbit(expected_array[zero_mask]),
            err_msg=f"{prefix}signed-zero bits differ from NumPy oracle",
        )


def _sample_input(rows: int, cols: int) -> np.ndarray:
    """Deterministic non-trivial float32 input, including signed zero."""
    row_index = np.arange(rows, dtype=np.float32).reshape(rows, 1)
    col_index = np.arange(cols, dtype=np.float32).reshape(1, cols)
    values = (
        row_index * np.float32(13.0)
        - col_index * np.float32(7.0)
        + np.float32(3.0)
    ).astype(np.float32, copy=False)

    if rows > 0 and cols > 0:
        values[0, 0] = np.float32(-0.0)
    if rows > 1 and cols > 1:
        values[1, 1] = np.float32(0.0)
    return values


def _sample_special_input() -> np.ndarray:
    """Input covering NaN, +Inf, -Inf, +0.0 and -0.0 below/above the diagonal."""
    return np.array(
        [
            [np.nan, np.inf, -np.inf, -0.0, 0.0],
            [1.0, -0.0, np.inf, 2.0, -2.0],
            [np.nan, np.inf, 0.0, -0.0, 3.5],
            [4.0, 5.0, 6.0, 7.0, -np.inf],
        ],
        dtype=np.float32,
    )


def _device_tests_enabled() -> bool:
    return os.environ.get(DEVICE_ENV, "").strip().lower() in _TRUTHY


def _require_public_tril() -> Tuple[Any, Any]:
    """Return (asnumpy module, asnumpy.tril) or skip with a clear reason."""
    if not _device_tests_enabled():
        pytest.skip(
            f"public-API device tests are opt-in; set {DEVICE_ENV}=1 to run them"
        )

    try:
        asnumpy_module = importlib.import_module("asnumpy")
    except ImportError as exc:
        pytest.skip(f"asnumpy is not importable in this environment: {exc}")

    tril_function = getattr(asnumpy_module, "tril", None)
    if not callable(tril_function):
        pytest.skip("asnumpy.tril is not registered by the installed runtime")

    return asnumpy_module, tril_function


def _make_device_input(asnumpy_module: Any, host_array: np.ndarray) -> Any:
    """Create an AsNumpy input tensor without importing asnumpy at module scope."""
    for factory_name in ("array", "asarray"):
        factory = getattr(asnumpy_module, factory_name, None)
        if callable(factory):
            return factory(host_array)
    raise AssertionError(
        "asnumpy exposes neither asnumpy.array nor asnumpy.asarray; "
        "cannot construct a public-API input tensor"
    )


def _call_public_tril(
    asnumpy_module: Any,
    tril_function: Any,
    host_array: np.ndarray,
) -> np.ndarray:
    tensor = _make_device_input(asnumpy_module, host_array)
    result = tril_function(tensor)
    return _to_numpy_host(result, asnumpy_module)


# ---------------------------------------------------------------------------
# CPU golden tests (always available; no asnumpy import)
# ---------------------------------------------------------------------------


def test_cpu_oracle_default_lower_triangle() -> None:
    x = _sample_input(4, 5)
    actual = numpy_oracle(x)
    expected = np.where(np.tri(4, 5, dtype=bool), x, np.float32(0.0))
    _assert_float32_semantics(actual, expected)


def test_cpu_oracle_default_is_k_zero() -> None:
    x = _sample_input(6, 9)
    np.testing.assert_array_equal(
        numpy_oracle(x),
        np.tril(x, k=0),
        err_msg="np.tril default must be equivalent to k=0",
    )


def test_cpu_oracle_above_diagonal_values_become_positive_zero() -> None:
    x = np.full((4, 5), -0.0, dtype=np.float32)
    actual = numpy_oracle(x)
    above_diagonal = ~np.tri(4, 5, dtype=bool)
    np.testing.assert_array_equal(
        actual[above_diagonal],
        np.zeros_like(actual[above_diagonal]),
    )
    assert not np.any(np.signbit(actual[above_diagonal]))


def test_cpu_oracle_preserves_special_values_and_signed_zero() -> None:
    x = _sample_special_input()
    expected = np.where(np.tri(4, 5, dtype=bool), x, np.float32(0.0))
    _assert_float32_semantics(numpy_oracle(x), expected)


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 8), (8, 1), (3, 7), (0, 4), (4, 0), (0, 0)],
)
def test_cpu_oracle_shape_and_dtype(shape: Tuple[int, int]) -> None:
    rows, cols = shape
    x = _sample_input(rows, cols)
    actual = numpy_oracle(x)

    assert actual.dtype == _FLOAT32
    assert actual.shape == shape

    expected = np.where(np.tri(rows, cols, dtype=bool), x, np.float32(0.0))
    _assert_float32_semantics(actual, expected)


def test_cpu_oracle_does_not_mutate_input() -> None:
    x = _sample_input(7, 9)
    before = x.copy()
    _ = numpy_oracle(x)
    np.testing.assert_array_equal(
        x.view(np.uint32),
        before.view(np.uint32),
        err_msg="np.tril must not mutate its input",
    )


def test_cpu_oracle_observed_2048_profile_contract() -> None:
    rows, cols = _OBSERVED_SHAPE
    x = _sample_input(rows, cols)
    actual = numpy_oracle(x)

    assert actual.dtype == x.dtype == _FLOAT32
    assert actual.shape == x.shape == _OBSERVED_SHAPE

    mask = np.tri(rows, cols, dtype=bool)
    np.testing.assert_array_equal(actual[mask], x[mask])
    np.testing.assert_array_equal(
        actual[~mask],
        np.zeros_like(actual[~mask]),
    )


# ---------------------------------------------------------------------------
# Public asnumpy.tril adapter tests (opt-in; import happens lazily)
# ---------------------------------------------------------------------------


def test_public_api_symbol_is_bound() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    assert callable(tril_function)
    assert tril_function is getattr(asnumpy_module, "tril")


def test_public_api_small_float32_matches_numpy_oracle() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    x = _sample_input(5, 7)
    actual = _call_public_tril(asnumpy_module, tril_function, x)
    _assert_float32_semantics(actual, numpy_oracle(x), context="asnumpy.tril")


def test_public_api_preserves_special_values_and_signed_zero() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    x = _sample_special_input()
    actual = _call_public_tril(asnumpy_module, tril_function, x)
    _assert_float32_semantics(
        actual,
        numpy_oracle(x),
        context="asnumpy.tril special values",
    )


def test_public_api_observed_2048_profile_matches_numpy_oracle() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    rows, cols = _OBSERVED_SHAPE
    x = _sample_input(rows, cols)
    actual = _call_public_tril(asnumpy_module, tril_function, x)
    _assert_float32_semantics(
        actual,
        numpy_oracle(x),
        context="asnumpy.tril observed 2048x2048 profile",
    )


def test_public_api_repeated_calls_and_input_lifecycle() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    x = _sample_input(6, 8)
    before = x.copy()

    first = _call_public_tril(asnumpy_module, tril_function, x)
    second = _call_public_tril(asnumpy_module, tril_function, x)

    _assert_float32_semantics(first, numpy_oracle(x), context="first call")
    _assert_float32_semantics(second, numpy_oracle(x), context="second call")
    np.testing.assert_array_equal(
        x.view(np.uint32),
        before.view(np.uint32),
        err_msg="asnumpy.tril must not mutate the host input",
    )


def test_public_api_rejects_float64_without_implicit_cast() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    x64 = _sample_input(3, 4).astype(np.float64)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        tensor = _make_device_input(asnumpy_module, x64)
        tril_function(tensor)


def test_public_api_rejects_non_2d_rank() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    x3 = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        tensor = _make_device_input(asnumpy_module, x3)
        tril_function(tensor)


def test_public_api_rejects_extra_call_forms() -> None:
    asnumpy_module, tril_function = _require_public_tril()
    tensor = _make_device_input(asnumpy_module, _sample_input(3, 4))

    with pytest.raises(TypeError):
        tril_function(tensor, 1)

    with pytest.raises(TypeError):
        tril_function(tensor, k=1)