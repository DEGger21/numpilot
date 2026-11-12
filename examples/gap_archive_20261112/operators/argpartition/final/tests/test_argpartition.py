"""NumPy-oracle and public-API tests for ``asnumpy.argpartition``.

The module intentionally does not import ``asnumpy`` at import time.  The pure
NumPy tests below always collect and pass without the AsNumpy runtime or the
custom Argpartition operator.  Public-API tests are guarded by a runtime
fixture and are skipped when the runtime, device, or custom operator is absent.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

FLOAT32 = np.dtype(np.float32)
INT64 = np.dtype(np.int64)
OBSERVED_SHAPE = (32, 16)
OBSERVED_KTH = 3
OBSERVED_AXIS = 1


def _normalize_axis(axis: int, ndim: int) -> int:
    value = int(axis)
    if value < 0:
        value += ndim
    if value < 0 or value >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return value


def _normalize_kth(kth: int, axis_length: int) -> int:
    if axis_length <= 0:
        raise ValueError("cannot partition an empty axis")
    value = int(kth)
    if value < -axis_length or value >= axis_length:
        raise IndexError(
            f"kth {kth} is out of bounds for axis of length {axis_length}"
        )
    return value % axis_length


def _rows_along_axis(values: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(values, axis, -1)
    return moved.reshape((-1, moved.shape[-1]))


def _assert_index_permutation(indices: np.ndarray, axis: int) -> None:
    axis_length = indices.shape[axis]
    expected_shape = [1] * indices.ndim
    expected_shape[axis] = axis_length
    expected = np.broadcast_to(
        np.arange(axis_length, dtype=indices.dtype).reshape(expected_shape),
        indices.shape,
    )
    np.testing.assert_array_equal(
        np.sort(indices, axis=axis),
        expected,
        err_msg="argpartition indices must be a permutation of each axis slice",
    )


def _assert_partition_condition(gathered: np.ndarray, kth: int, axis: int) -> None:
    rows = _rows_along_axis(gathered, axis)
    for row_index, row in enumerate(rows):
        pivot = row[kth]
        left = row[:kth]
        right = row[kth + 1 :]

        if np.isnan(pivot):
            if right.size and not np.all(np.isnan(right)):
                raise AssertionError(
                    f"row {row_index}: NaN pivot requires all right-side values to be NaN"
                )
            continue

        if left.size and np.any(np.isnan(left)):
            raise AssertionError(
                f"row {row_index}: NaN must not appear left of a non-NaN pivot"
            )
        if left.size and not np.all(left <= pivot):
            raise AssertionError(
                f"row {row_index}: left-side values are not <= the kth value"
            )

        finite_right = right[~np.isnan(right)]
        if finite_right.size and not np.all(finite_right >= pivot):
            raise AssertionError(
                f"row {row_index}: right-side values are not >= the kth value"
            )


def _assert_argpartition_contract(
    indices: Any,
    source: np.ndarray,
    kth: int,
    axis: int,
) -> None:
    values = np.asarray(source)
    result = np.asarray(indices)
    axis_normalized = _normalize_axis(axis, values.ndim)
    kth_normalized = _normalize_kth(kth, values.shape[axis_normalized])

    if result.shape != values.shape:
        raise AssertionError(
            f"argpartition shape mismatch: expected {values.shape}, got {result.shape}"
        )
    if result.dtype != INT64:
        raise AssertionError(
            f"argpartition output dtype must be int64, got {result.dtype}"
        )

    _assert_index_permutation(result, axis_normalized)
    gathered = np.take_along_axis(values, result, axis=axis_normalized)

    np.testing.assert_array_equal(
        np.sort(gathered, axis=axis_normalized),
        np.sort(values, axis=axis_normalized),
        err_msg="argpartition indices do not select the input values exactly once",
    )

    expected_kth = np.take(
        np.sort(values, axis=axis_normalized), kth_normalized, axis=axis_normalized
    )
    actual_kth = np.take(gathered, kth_normalized, axis=axis_normalized)
    np.testing.assert_array_equal(
        actual_kth,
        expected_kth,
        err_msg="argpartition kth element does not match the NumPy oracle",
    )

    _assert_partition_condition(gathered, kth_normalized, axis_normalized)


def _observed_profile_source(seed: int = 20261112) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(np.prod(OBSERVED_SHAPE), dtype=FLOAT32).reshape(OBSERVED_SHAPE)
    noise = rng.standard_normal(OBSERVED_SHAPE).astype(FLOAT32) * FLOAT32.type(0.125)
    return np.ascontiguousarray(base + noise, dtype=FLOAT32)


def _special_observed_source() -> np.ndarray:
    source = _observed_profile_source(seed=20261113)
    source[0, :] = np.asarray(
        [
            np.nan,
            np.inf,
            -np.inf,
            0.0,
            -0.0,
            1.0,
            -1.0,
            np.nan,
            np.inf,
            -np.inf,
            2.0,
            -2.0,
            0.0,
            -0.0,
            3.0,
            -3.0,
        ],
        dtype=FLOAT32,
    )
    source[1, :] = np.nan
    source[2, ::2] = np.inf
    source[3, 1::2] = -np.inf
    source[4, ::4] = 0.0
    source[4, 2::4] = np.float32(-0.0)
    source[5, :8] = 1.0
    source[6, 8:] = -1.0
    return np.ascontiguousarray(source, dtype=FLOAT32)


def _materialize_result(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return result
    for method_name in ("asnumpy", "to_numpy", "numpy", "cpu"):
        method = getattr(result, method_name, None)
        if callable(method):
            converted = method()
            if converted is not result:
                return np.asarray(converted)
    return np.asarray(result)


def _assert_runtime_result(
    result: Any,
    source: np.ndarray,
    kth: int,
    axis: int,
) -> None:
    materialized = _materialize_result(result)
    if not isinstance(materialized, np.ndarray):
        raise AssertionError(
            f"public API must return an array-like result, got {type(materialized)!r}"
        )
    if materialized.dtype != INT64:
        raise AssertionError(
            f"public API output dtype must be int64, got {materialized.dtype}"
        )
    if materialized.shape != source.shape:
        raise AssertionError(
            f"public API output shape mismatch: expected {source.shape}, "
            f"got {materialized.shape}"
        )
    if not materialized.flags.c_contiguous:
        raise AssertionError("public API output must be C-contiguous")
    _assert_argpartition_contract(materialized, source, kth, axis)


def _is_runtime_unavailable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "no module named",
        "cannot import name",
        "undefined symbol",
        "not found",
        "cannot find",
        "could not find",
        "not registered",
        "not installed",
        "is not available",
        "not available",
        "not supported",
        "unsupported",
        "no kernel",
        "kernel not found",
        "failed to load",
        "could not load",
        "cannot load",
        "cannot open",
        "no such file",
        "no device",
        "device not",
        "device is not",
        "npu",
    )
    return any(marker in text for marker in markers)


@pytest.fixture(scope="module")
def argpartition_api() -> Any:
    try:
        module = importlib.import_module("asnumpy")
    except Exception as exc:  # pragma: no cover - depends on local runtime
        pytest.skip(f"asnumpy runtime is unavailable: {exc!r}")

    api = getattr(module, "argpartition", None)
    if api is None or not callable(api):
        pytest.skip("asnumpy.argpartition is not available")

    if api is np.argpartition:
        pytest.fail("asnumpy.argpartition must not alias numpy.argpartition")

    probe = _observed_profile_source(seed=101)
    try:
        probe_result = api(probe, OBSERVED_KTH, axis=OBSERVED_AXIS)
    except Exception as exc:  # pragma: no cover - depends on local runtime
        if _is_runtime_unavailable(exc):
            pytest.skip(f"Argpartition runtime is unavailable: {exc!r}")
        raise

    _assert_runtime_result(probe_result, probe, OBSERVED_KTH, OBSERVED_AXIS)
    return api


class TestNumpyOracleContract:
    def test_observed_profile_oracle_contract(self) -> None:
        source = _observed_profile_source()
        expected = np.argpartition(source, OBSERVED_KTH, axis=OBSERVED_AXIS)

        assert expected.dtype == INT64
        assert expected.shape == source.shape
        _assert_argpartition_contract(
            expected, source, OBSERVED_KTH, OBSERVED_AXIS
        )

    def test_duplicate_values_and_negative_axis(self) -> None:
        rng = np.random.default_rng(20261114)
        source = rng.integers(-2, 3, size=(8, 16)).astype(FLOAT32)

        expected = np.argpartition(source, 2, axis=-1)

        assert expected.dtype == INT64
        assert expected.shape == source.shape
        _assert_argpartition_contract(expected, source, 2, -1)

    def test_special_values_nan_inf_and_signed_zero(self) -> None:
        source = _special_observed_source()

        for kth in (0, 3, 7, 15):
            expected = np.argpartition(source, kth, axis=OBSERVED_AXIS)
            assert expected.dtype == INT64
            assert expected.shape == source.shape
            _assert_argpartition_contract(
                expected, source, kth, OBSERVED_AXIS
            )

    def test_empty_batch(self) -> None:
        source = np.empty((0, 16), dtype=FLOAT32)

        expected = np.argpartition(source, OBSERVED_KTH, axis=OBSERVED_AXIS)

        assert expected.dtype == INT64
        assert expected.shape == source.shape
        _assert_argpartition_contract(
            expected, source, OBSERVED_KTH, OBSERVED_AXIS
        )

    def test_invalid_kth_and_axis_raise(self) -> None:
        source = np.zeros((2, 3), dtype=FLOAT32)

        with pytest.raises((ValueError, IndexError)):
            np.argpartition(source, 3, axis=1)

        with pytest.raises((ValueError, IndexError)):
            np.argpartition(source, 1, axis=2)

    def test_semantic_checker_rejects_non_permutation(self) -> None:
        source = np.arange(12, dtype=FLOAT32).reshape(3, 4)
        good = np.argpartition(source, 1, axis=1)
        _assert_argpartition_contract(good, source, 1, 1)

        bad = good.copy()
        bad[0, 0] = bad[0, 1]

        with pytest.raises(AssertionError):
            _assert_argpartition_contract(bad, source, 1, 1)


class TestPublicAsNumpyArgpartition:
    def test_public_api_is_not_numpy_alias(self, argpartition_api: Any) -> None:
        assert argpartition_api is not np.argpartition

    def test_observed_profile_matches_numpy_partition_semantics(
        self, argpartition_api: Any
    ) -> None:
        source = _observed_profile_source(seed=20261201)

        result = argpartition_api(source, OBSERVED_KTH, axis=OBSERVED_AXIS)

        _assert_runtime_result(result, source, OBSERVED_KTH, OBSERVED_AXIS)

    def test_special_values_match_numpy_partition_semantics(
        self, argpartition_api: Any
    ) -> None:
        source = _special_observed_source()

        result = argpartition_api(source, OBSERVED_KTH, axis=OBSERVED_AXIS)

        _assert_runtime_result(result, source, OBSERVED_KTH, OBSERVED_AXIS)

    @pytest.mark.parametrize("dtype", [np.float64, np.int32])
    def test_public_api_rejects_unsupported_input_dtype_without_cast(
        self, argpartition_api: Any, dtype: type
    ) -> None:
        source = np.arange(np.prod(OBSERVED_SHAPE), dtype=dtype).reshape(
            OBSERVED_SHAPE
        )

        with pytest.raises((TypeError, ValueError, RuntimeError)):
            argpartition_api(source, OBSERVED_KTH, axis=OBSERVED_AXIS)

    def test_public_api_rejects_invalid_axis_and_kth(
        self, argpartition_api: Any
    ) -> None:
        source = _observed_profile_source(seed=20261204)

        with pytest.raises((ValueError, IndexError, RuntimeError)):
            argpartition_api(source, OBSERVED_KTH, axis=2)

        with pytest.raises((ValueError, IndexError, RuntimeError)):
            argpartition_api(source, source.shape[OBSERVED_AXIS], axis=OBSERVED_AXIS)

    def test_public_api_rejects_unsupported_keywords(
        self, argpartition_api: Any
    ) -> None:
        source = _observed_profile_source(seed=20261205)

        with pytest.raises((TypeError, ValueError)):
            argpartition_api(
                source, OBSERVED_KTH, axis=OBSERVED_AXIS, kind="quicksort"
            )

        with pytest.raises((TypeError, ValueError)):
            argpartition_api(source, OBSERVED_KTH, axis=OBSERVED_AXIS, out=None)

    def test_public_api_does_not_delegate_to_numpy_argpartition(
        self, argpartition_api: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _observed_profile_source(seed=20261202)

        def forbidden_numpy_argpartition(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "asnumpy.argpartition must not delegate to numpy.argpartition"
            )

        monkeypatch.setattr(np, "argpartition", forbidden_numpy_argpartition)

        result = argpartition_api(source, OBSERVED_KTH, axis=OBSERVED_AXIS)

        _assert_runtime_result(result, source, OBSERVED_KTH, OBSERVED_AXIS)

    def test_public_api_does_not_mutate_input_and_is_repeatable(
        self, argpartition_api: Any
    ) -> None:
        source = _observed_profile_source(seed=20261203)
        before = source.copy()

        first = argpartition_api(source, OBSERVED_KTH, axis=OBSERVED_AXIS)
        second = argpartition_api(source.copy(), OBSERVED_KTH, axis=OBSERVED_AXIS)

        np.testing.assert_array_equal(
            source, before, err_msg="public API mutated its input tensor"
        )
        _assert_runtime_result(first, source, OBSERVED_KTH, OBSERVED_AXIS)
        _assert_runtime_result(second, source, OBSERVED_KTH, OBSERVED_AXIS)
        np.testing.assert_array_equal(
            _materialize_result(first),
            _materialize_result(second),
            err_msg="public API is not repeatable for the admitted profile",
        )