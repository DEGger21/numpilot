#!/usr/bin/env python3
"""Deterministic API correctness gate for the reviewed concatenate adapter.

This file is copied into an integration transaction and executed by a
separate CPython process with an explicitly constructed ``PYTHONPATH``.  It
must remain standalone: it imports only the standard library, NumPy, and the
isolated AsNumpy build under test.  NumPilot never delegates this comparison
to a model.
"""

from __future__ import annotations

import argparse
import json
import importlib
import os
import resource
import sys


SCHEMA_VERSION = 1
API_NAME = "numpy.concatenate"
MAX_DETAIL_CHARS = 2048
REPEATED_CALLS = 100
REPRESENTATIVE_EXCEPTIONS = (TypeError, ValueError)


def _write_result(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _detail(exc):
    text = "%s: %s" % (type(exc).__name__, exc)
    return text[:MAX_DETAIL_CHARS]


def _install_dispatch_probe(array_module, native_module):
    """Require the reviewed binding and trap the pre-existing D2D route.

    Numerical equivalence alone cannot distinguish the old copy implementation
    from the new ACLNN adapter. Keep this check NumPilot-owned, independent of
    tests supplied by the generating agent.
    """
    native = getattr(native_module, "concatenate_aclnn", None)
    if not callable(native) or getattr(array_module, "_concatenate", None) is not native:
        raise RuntimeError("public concatenate does not select the reviewed ACLNN binding")
    observed = {"native_calls": 0, "legacy_calls": 0}

    def checked_native(*args, **kwargs):
        observed["native_calls"] += 1
        return native(*args, **kwargs)

    def forbidden_legacy(*args, **kwargs):
        observed["legacy_calls"] += 1
        raise RuntimeError("public concatenate entered the legacy copy route")

    array_module._concatenate = checked_native
    array_module._concatenate_legacy = forbidden_legacy
    return observed


CBRT_API_NAME = "numpy.cbrt"
INVERT_API_NAME = "numpy.invert"
ARGSORT_API_NAME = "numpy.argsort"
SEARCHSORTED_API_NAME = "numpy.searchsorted"
MEDIAN_API_NAME = "numpy.median"


def _finish(result, publish):
    failed = [case for case in result["cases"] if case["status"] != "passed"]
    if failed:
        result["status"] = "failed"
        result["error"] = {
            "type": "GateCaseFailure",
            "message": "%d gate case(s) failed" % len(failed),
        }
        publish()
        return 2
    result["status"] = "passed"
    result["error"] = None
    publish()
    return 0


def _assert_cbrt(actual, expected, numpy, *, rtol=1e-6):
    """Compare one cbrt output, including signed zero and NaN handling.

    ``rtol`` defaults to the strict float32 target used for I001's bounded
    input range.  The dedicated large-magnitude case passes a wider float32
    tolerance because a single-precision ``pow`` implementation can land
    around a dozen ulp from NumPy's cbrt at 1e30; the oracle still rejects any
    percent-level error.
    """

    if actual.dtype != expected.dtype:
        raise AssertionError("dtype %r != %r" % (actual.dtype, expected.dtype))
    if actual.shape != expected.shape:
        raise AssertionError("shape %r != %r" % (actual.shape, expected.shape))
    if not numpy.array_equal(numpy.isnan(actual), numpy.isnan(expected)):
        raise AssertionError("NaN propagation pattern differs from NumPy")
    mask = ~numpy.isnan(expected)
    if not numpy.array_equal(numpy.signbit(actual[mask]), numpy.signbit(expected[mask])):
        raise AssertionError("signbit/signed-zero differs from NumPy cbrt; np.cbrt(-0.0) has signbit set")
    if not numpy.allclose(actual[mask], expected[mask], rtol=rtol, atol=1e-7):
        delta = numpy.abs(
            actual[mask].astype(numpy.float64) - expected[mask].astype(numpy.float64)
        )
        scale = numpy.maximum(
            numpy.abs(expected[mask].astype(numpy.float64)), 1.0
        )
        relative = numpy.max(delta / scale)
        raise AssertionError(
            "max absolute error %.9g, max relative error %.9g at rtol %.1e"
            % (float(numpy.max(delta)), float(relative), rtol)
        )


def _run_cbrt(arguments, result, publish, record, numpy, ap):
    """Deterministic NumPy oracle suite for the reviewed cbrt adapter."""

    import importlib
    import resource

    if not callable(getattr(ap, "cbrt", None)):
        raise RuntimeError("asnumpy.cbrt is not available")
    math_module = importlib.import_module("asnumpy.math")
    native_module = importlib.import_module("asnumpy._core.math")
    native = getattr(native_module, "cbrt", None)
    if not callable(native):
        raise RuntimeError("reviewed native cbrt binding is missing")
    if getattr(math_module, "_cbrt", None) is not native:
        raise RuntimeError(
            "public cbrt does not select the reviewed native binding"
        )
    observed = {"native_calls": 0, "fallback_calls": 0}

    def checked_native(value):
        observed["native_calls"] += 1
        return native(value)

    def forbidden_fallback(value):
        observed["fallback_calls"] += 1
        raise RuntimeError("public cbrt entered a fallback route")

    # The reviewed Python wrapper resolves ``_cbrt`` from its module globals on
    # every call, so this patch observes the real dispatch path without
    # modifying the installed extension.
    math_module._cbrt = checked_native
    math_module._cbrt_legacy = forbidden_fallback

    def compare(name, values, *, rtol=1e-6):
        try:
            expected = numpy.cbrt(values)
            actual = ap.cbrt(ap.ndarray.from_numpy(values)).to_numpy()
            _assert_cbrt(actual, expected, numpy, rtol=rtol)
            result["oracle_comparisons"] += 1
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))

    mixed = numpy.array(
        [
            -27.0, -8.0, -1.0, -0.125, -1e-6, 0.0, 1e-6, 0.125,
            1.0, 8.0, 27.0, 1e6, -1e6,
        ],
        dtype=numpy.float32,
    )
    compare("mixed_real_values", mixed)
    compare(
        "signed_zero_preserved",
        numpy.array([-0.0, 0.0], dtype=numpy.float32),
    )
    compare(
        "large_magnitude_values",
        numpy.array([1e30, -1e30], dtype=numpy.float32),
        rtol=2e-6,
    )
    compare(
        "negative_values",
        numpy.array([-27.0, -8.0, -1.0, -0.125, -1e-6], dtype=numpy.float32),
    )
    compare(
        "positive_values",
        numpy.array([0.125, 1e-6, 1.0, 8.0, 27.0], dtype=numpy.float32),
    )
    compare(
        "infinities",
        numpy.array([numpy.inf, -numpy.inf], dtype=numpy.float32),
    )
    compare(
        "nan_propagation",
        numpy.array([numpy.nan, -numpy.nan], dtype=numpy.float32),
    )
    compare(
        "single_value",
        numpy.array([27.0], dtype=numpy.float32),
    )
    compare(
        "rank_two_input",
        numpy.linspace(-27.0, 27.0, 12, dtype=numpy.float32).reshape(3, 4),
    )
    compare(
        "rank_three_input",
        numpy.linspace(-27.0, 27.0, 24, dtype=numpy.float32).reshape(2, 3, 4),
    )

    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    repeated_failure = None
    expected_mixed = numpy.cbrt(mixed)
    for index in range(REPEATED_CALLS):
        try:
            actual = ap.cbrt(ap.ndarray.from_numpy(mixed)).to_numpy()
            _assert_cbrt(actual, expected_mixed, numpy)
            result["oracle_comparisons"] += 1
        except BaseException as exc:
            repeated_failure = "iteration %d: %s" % (index, _detail(exc))
            break
    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_kib"] = end_rss
    result["repeated_calls"] = REPEATED_CALLS
    result["rss_start_kib"] = start_rss
    result["rss_end_kib"] = end_rss
    if repeated_failure is not None:
        record("repeated_calls_stable", "failed", repeated_failure)
    else:
        record("repeated_calls_stable", "passed")

    result["dispatch_observation"] = dict(observed)
    dispatch_passed = (
        observed["native_calls"] >= result["oracle_comparisons"] > 0
        and observed["fallback_calls"] == 0
    )
    record(
        "official_aclnn_dispatch",
        "passed" if dispatch_passed else "failed",
        "native=%d fallback=%d"
        % (observed["native_calls"], observed["fallback_calls"]),
    )
    return _finish(result, publish)


def _run_invert(arguments, result, publish, record, numpy, ap):
    """Deterministic NumPy oracle suite for the reviewed invert adapter."""

    if not callable(getattr(ap, "invert", None)):
        raise RuntimeError("asnumpy.invert is not available")
    public_module = importlib.import_module("asnumpy.logic")
    core_module = importlib.import_module("asnumpy._core.logic")
    native = getattr(core_module, "invert", None)
    if not callable(native):
        raise RuntimeError("reviewed native invert binding is missing")
    aliases = [
        name for name, value in vars(public_module).items() if value is native
    ]
    if not aliases:
        raise RuntimeError(
            "public logic module does not import the reviewed invert binding"
        )

    originals = {name: getattr(public_module, name) for name in aliases}

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("official invert binding probe")

    # Binding probe: the public wrapper must resolve one of these module-level
    # aliases at call time, otherwise patching it cannot intercept the call.
    for name in aliases:
        setattr(public_module, name, forbidden)
    try:
        probe = numpy.array([1], dtype=numpy.int32)
        try:
            ap.invert(ap.ndarray.from_numpy(probe))
        except BaseException as exc:
            if "official invert binding probe" in str(exc):
                record("native_binding_selected", "passed")
            else:
                record("native_binding_selected", "failed", _detail(exc))
        else:
            record(
                "native_binding_selected",
                "failed",
                "public invert bypassed the reviewed native binding",
            )
    finally:
        for name, value in originals.items():
            setattr(public_module, name, value)

    observed = {"native_calls": 0}

    def checked_native(*args, **kwargs):
        observed["native_calls"] += 1
        return native(*args, **kwargs)

    for name in aliases:
        setattr(public_module, name, checked_native)

    def compare(name, values):
        try:
            expected = numpy.invert(values)
            actual = ap.invert(ap.ndarray.from_numpy(values)).to_numpy()
            if actual.dtype != expected.dtype:
                raise AssertionError(
                    "dtype %r != %r" % (actual.dtype, expected.dtype)
                )
            if actual.shape != expected.shape:
                raise AssertionError(
                    "shape %r != %r" % (actual.shape, expected.shape)
                )
            numpy.testing.assert_array_equal(actual, expected)
            result["oracle_comparisons"] += 1
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))

    def reject(name, values):
        try:
            ap.invert(ap.ndarray.from_numpy(values))
        except (TypeError, ValueError, RuntimeError):
            # The C++ dtype guard surfaces as std::runtime_error.
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))
        else:
            record(name, "failed", "unsupported dtype was accepted")

    compare(
        "bool_values",
        numpy.array([True, False, True, True, False], dtype=numpy.bool_),
    )
    compare(
        "int8_boundaries",
        numpy.array([-128, -2, -1, 0, 1, 126, 127], dtype=numpy.int8),
    )
    compare(
        "int16_boundaries",
        numpy.array([-32768, -1, 0, 1, 32767], dtype=numpy.int16),
    )
    compare(
        "int32_boundaries",
        numpy.array([-2147483648, -1, 0, 1, 2147483647], dtype=numpy.int32),
    )
    compare(
        "int64_boundaries",
        numpy.array([-9223372036854775808, -1, 0, 1, 9223372036854775807],
                    dtype=numpy.int64),
    )
    compare(
        "uint8_values",
        numpy.array([0, 1, 2, 127, 254, 255], dtype=numpy.uint8),
    )
    compare(
        "rank_two_int32",
        numpy.arange(-6, 6, dtype=numpy.int32).reshape(3, 4),
    )
    reject("float32_rejected", numpy.array([1.0, -2.0], dtype=numpy.float32))
    reject("float64_rejected", numpy.array([1.0, -2.0], dtype=numpy.float64))
    reject("uint16_rejected", numpy.array([1, 2], dtype=numpy.uint16))
    reject("uint32_rejected", numpy.array([1, 2], dtype=numpy.uint32))

    base = numpy.array([-128, -1, 0, 1, 127], dtype=numpy.int8)
    expected_base = numpy.invert(base)
    repeated_failure = None
    for index in range(REPEATED_CALLS):
        try:
            actual = ap.invert(ap.ndarray.from_numpy(base)).to_numpy()
            numpy.testing.assert_array_equal(actual, expected_base)
            result["oracle_comparisons"] += 1
        except BaseException as exc:
            repeated_failure = "iteration %d: %s" % (index, _detail(exc))
            break
    if repeated_failure is not None:
        record("repeated_calls_stable", "failed", repeated_failure)
    else:
        record("repeated_calls_stable", "passed")

    result["dispatch_observation"] = dict(observed)
    dispatch_passed = observed["native_calls"] >= result["oracle_comparisons"] > 0
    record(
        "official_aclnn_dispatch",
        "passed" if dispatch_passed else "failed",
        "native=%d" % observed["native_calls"],
    )
    return _finish(result, publish)


def _bind_sorting_native(public_module, core_module, name, api):
    """Return the module-level aliases bound to the reviewed extension."""

    native = getattr(core_module, name, None)
    if not callable(native):
        raise RuntimeError("reviewed native %s binding is missing" % api)
    aliases = [
        alias for alias, value in vars(public_module).items() if value is native
    ]
    if not aliases:
        raise RuntimeError(
            "public sorting module does not import the reviewed %s binding" % api
        )
    return native, aliases


def _probe_sorting_alias(public_module, aliases, call, record, name):
    """Insert a trap on every alias and require the public call to hit it."""

    originals = {alias: getattr(public_module, alias) for alias in aliases}

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("official %s binding probe" % name)

    for alias in aliases:
        setattr(public_module, alias, forbidden)
    try:
        try:
            call()
        except BaseException as exc:
            if "official %s binding probe" % name in str(exc):
                record("native_binding_selected", "passed")
            else:
                record("native_binding_selected", "failed", _detail(exc))
        else:
            record(
                "native_binding_selected",
                "failed",
                "public %s bypassed the reviewed native binding" % name,
            )
    finally:
        for alias, value in originals.items():
            setattr(public_module, alias, value)


def _install_sorting_counter(public_module, aliases, native):
    observed = {"native_calls": 0}

    def checked(*args, **kwargs):
        observed["native_calls"] += 1
        return native(*args, **kwargs)

    for alias in aliases:
        setattr(public_module, alias, checked)
    return observed


def _run_argsort(arguments, result, publish, record, numpy, ap):
    """Deterministic NumPy oracle suite for the reviewed argsort adapter."""

    if not callable(getattr(ap, "argsort", None)):
        raise RuntimeError("asnumpy.argsort is not available")
    public_module = importlib.import_module("asnumpy.sorting")
    core_module = importlib.import_module("asnumpy._core.sorting")
    native, aliases = _bind_sorting_native(
        public_module, core_module, "argsort", "argsort"
    )
    probe_values = numpy.array([3.0, 1.0], dtype=numpy.float32)
    _probe_sorting_alias(
        public_module,
        aliases,
        lambda: ap.argsort(ap.ndarray.from_numpy(probe_values)),
        record,
        "argsort",
    )
    observed = _install_sorting_counter(public_module, aliases, native)

    def compare(name, values, axis=-1):
        try:
            expected = numpy.argsort(values, axis=axis)
            actual = ap.argsort(ap.ndarray.from_numpy(values), axis=axis).to_numpy()
            if actual.dtype != numpy.int64:
                raise AssertionError("dtype %r != int64" % (actual.dtype,))
            if actual.shape != expected.shape:
                raise AssertionError(
                    "shape %r != %r" % (actual.shape, expected.shape)
                )
            numpy.testing.assert_array_equal(actual, expected)
            result["oracle_comparisons"] += 1
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))

    def reject(name, values, axis=-1):
        try:
            ap.argsort(ap.ndarray.from_numpy(values), axis=axis)
        except (TypeError, ValueError, RuntimeError, IndexError):
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))
        else:
            record(name, "failed", "unsupported input was accepted")

    compare(
        "descending_ramp_1d",
        numpy.array([9.0, 7.0, 5.0, 3.0, 1.0], dtype=numpy.float32),
    )
    compare(
        "ascending_ramp_1d",
        numpy.array([1.0, 5.0, 3.0, 7.0, 9.0], dtype=numpy.float32),
    )
    compare(
        "irregular_values_1d",
        numpy.array([0.5, -2.25, 3.75, -1.0, 2.0, 0.0], dtype=numpy.float32),
    )
    compare("single_element", numpy.array([-4.0], dtype=numpy.float32))
    compare(
        "rank_two_default_axis",
        numpy.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype=numpy.float32),
    )
    compare(
        "rank_two_axis_zero",
        numpy.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype=numpy.float32),
        axis=0,
    )
    compare(
        "rank_three_axis_one",
        numpy.linspace(-3.0, 3.0, 12, dtype=numpy.float32).reshape(2, 3, 2),
        axis=1,
    )
    reject(
        "int32_rejected",
        numpy.array([3, 1, 2], dtype=numpy.int32),
    )
    reject(
        "float64_rejected",
        numpy.array([3.0, 1.0, 2.0], dtype=numpy.float64),
    )
    reject(
        "axis_out_of_range_rejected",
        numpy.array([3.0, 1.0, 2.0], dtype=numpy.float32),
        axis=2,
    )
    reject(
        "negative_axis_out_of_range_rejected",
        numpy.array([3.0, 1.0, 2.0], dtype=numpy.float32),
        axis=-2,
    )

    base = numpy.array([9.0, 7.0, 5.0, 3.0, 1.0], dtype=numpy.float32)
    expected_base = numpy.argsort(base)
    repeated_failure = None
    for index in range(REPEATED_CALLS):
        try:
            actual = ap.argsort(ap.ndarray.from_numpy(base)).to_numpy()
            numpy.testing.assert_array_equal(actual, expected_base)
            result["oracle_comparisons"] += 1
        except BaseException as exc:
            repeated_failure = "iteration %d: %s" % (index, _detail(exc))
            break
    if repeated_failure is not None:
        record("repeated_calls_stable", "failed", repeated_failure)
    else:
        record("repeated_calls_stable", "passed")

    result["dispatch_observation"] = dict(observed)
    dispatch_passed = observed["native_calls"] >= result["oracle_comparisons"] > 0
    record(
        "official_aclnn_dispatch",
        "passed" if dispatch_passed else "failed",
        "native=%d" % observed["native_calls"],
    )
    return _finish(result, publish)


def _run_searchsorted(arguments, result, publish, record, numpy, ap):
    """Deterministic NumPy oracle suite for the reviewed searchsorted adapter."""

    if not callable(getattr(ap, "searchsorted", None)):
        raise RuntimeError("asnumpy.searchsorted is not available")
    public_module = importlib.import_module("asnumpy.sorting")
    core_module = importlib.import_module("asnumpy._core.sorting")
    native, aliases = _bind_sorting_native(
        public_module, core_module, "searchsorted", "searchsorted"
    )
    probe_boundaries = numpy.array([0.0, 1.0, 2.0], dtype=numpy.float32)
    probe_values = numpy.array([0.5, 1.5], dtype=numpy.float32)
    _probe_sorting_alias(
        public_module,
        aliases,
        lambda: ap.searchsorted(
            ap.ndarray.from_numpy(probe_boundaries),
            ap.ndarray.from_numpy(probe_values),
        ),
        record,
        "searchsorted",
    )
    observed = _install_sorting_counter(public_module, aliases, native)

    def compare(name, boundaries, probes, right=False):
        try:
            expected = numpy.searchsorted(
                boundaries, probes, side="right" if right else "left"
            )
            actual = ap.searchsorted(
                ap.ndarray.from_numpy(boundaries),
                ap.ndarray.from_numpy(probes),
                right=right,
            ).to_numpy()
            if actual.dtype != numpy.int64:
                raise AssertionError("dtype %r != int64" % (actual.dtype,))
            if actual.shape != expected.shape:
                raise AssertionError(
                    "shape %r != %r" % (actual.shape, expected.shape)
                )
            numpy.testing.assert_array_equal(actual, expected)
            result["oracle_comparisons"] += 1
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))

    def reject(name, boundaries, probes, right=False):
        try:
            ap.searchsorted(
                ap.ndarray.from_numpy(boundaries),
                ap.ndarray.from_numpy(probes),
                right=right,
            )
        except (TypeError, ValueError, RuntimeError):
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))
        else:
            record(name, "failed", "unsupported input was accepted")

    ramp = numpy.linspace(0.0, 1.0, 11, dtype=numpy.float32)
    compare(
        "probes_inside_and_outside_left",
        ramp,
        numpy.array([-0.5, 0.0, 0.05, 0.5, 0.95, 1.0, 1.5], dtype=numpy.float32),
    )
    compare(
        "probes_inside_and_outside_right",
        ramp,
        numpy.array([-0.5, 0.0, 0.05, 0.5, 0.95, 1.0, 1.5], dtype=numpy.float32),
        right=True,
    )
    duplicates = numpy.array([0.0, 0.0, 1.0, 1.0, 2.0], dtype=numpy.float32)
    compare(
        "duplicate_boundaries_left",
        duplicates,
        numpy.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=numpy.float32),
    )
    compare(
        "duplicate_boundaries_right",
        duplicates,
        numpy.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=numpy.float32),
        right=True,
    )
    compare(
        "all_probes_below_range",
        ramp,
        numpy.array([-3.0, -2.0, -1.0], dtype=numpy.float32),
    )
    compare(
        "all_probes_above_range",
        ramp,
        numpy.array([2.0, 3.0, 4.0], dtype=numpy.float32),
    )
    compare(
        "single_probe_exact_hit",
        ramp,
        numpy.array([0.5], dtype=numpy.float32),
    )
    reject(
        "two_dimensional_sequence_rejected",
        numpy.zeros((2, 3), dtype=numpy.float32),
        numpy.array([0.5], dtype=numpy.float32),
    )
    reject(
        "float64_inputs_rejected",
        ramp.astype(numpy.float64),
        numpy.array([0.5], dtype=numpy.float64),
    )
    # Documented reviewed boundary: the ACLNN pair has no empty-tensor route,
    # so both empty operands must be refused rather than silently answered.
    reject(
        "empty_sequence_boundary_rejected",
        numpy.array([], dtype=numpy.float32),
        numpy.array([0.5], dtype=numpy.float32),
    )
    reject(
        "empty_values_boundary_rejected",
        ramp,
        numpy.array([], dtype=numpy.float32),
    )

    base_boundaries = numpy.linspace(0.0, 1.0, 11, dtype=numpy.float32)
    base_probes = numpy.array([-0.5, 0.25, 0.75, 1.5], dtype=numpy.float32)
    expected_base = numpy.searchsorted(base_boundaries, base_probes)
    repeated_failure = None
    for index in range(REPEATED_CALLS):
        try:
            actual = ap.searchsorted(
                ap.ndarray.from_numpy(base_boundaries),
                ap.ndarray.from_numpy(base_probes),
            ).to_numpy()
            numpy.testing.assert_array_equal(actual, expected_base)
            result["oracle_comparisons"] += 1
        except BaseException as exc:
            repeated_failure = "iteration %d: %s" % (index, _detail(exc))
            break
    if repeated_failure is not None:
        record("repeated_calls_stable", "failed", repeated_failure)
    else:
        record("repeated_calls_stable", "passed")

    result["dispatch_observation"] = dict(observed)
    dispatch_passed = observed["native_calls"] >= result["oracle_comparisons"] > 0
    record(
        "official_aclnn_dispatch",
        "passed" if dispatch_passed else "failed",
        "native=%d" % observed["native_calls"],
    )
    return _finish(result, publish)


def _run_median(arguments, result, publish, record, numpy, ap):
    """Deterministic NumPy oracle suite for the reviewed median adapter."""

    if not callable(getattr(ap, "median", None)):
        raise RuntimeError("asnumpy.median is not available")
    public_module = importlib.import_module("asnumpy.statistics")
    core_module = importlib.import_module("asnumpy._core.statistics")
    native = getattr(core_module, "median", None)
    if not callable(native):
        raise RuntimeError("reviewed native median binding is missing")
    aliases = [
        name for name, value in vars(public_module).items() if value is native
    ]
    if not aliases:
        raise RuntimeError(
            "public statistics module does not import the reviewed median binding"
        )

    originals = {name: getattr(public_module, name) for name in aliases}

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("official median binding probe")

    for name in aliases:
        setattr(public_module, name, forbidden)
    try:
        try:
            ap.median(ap.ndarray.from_numpy(
                numpy.array([1.0], dtype=numpy.float32)
            ))
        except BaseException as exc:
            if "official median binding probe" in str(exc):
                record("native_binding_selected", "passed")
            else:
                record("native_binding_selected", "failed", _detail(exc))
        else:
            record(
                "native_binding_selected",
                "failed",
                "public median bypassed the reviewed native binding",
            )
    finally:
        for name, value in originals.items():
            setattr(public_module, name, value)

    observed = {"native_calls": 0}

    def checked_native(*args, **kwargs):
        observed["native_calls"] += 1
        return native(*args, **kwargs)

    for name in aliases:
        setattr(public_module, name, checked_native)

    def scalar_of(actual):
        payload = actual
        if hasattr(payload, "to_numpy"):
            payload = payload.to_numpy()
        payload = numpy.asarray(payload)
        if payload.size != 1:
            raise AssertionError(
                "median returned %d values, expected one" % payload.size
            )
        return float(payload.reshape(-1)[0])

    def compare(name, values):
        try:
            expected = float(numpy.median(values))
            actual = scalar_of(ap.median(ap.ndarray.from_numpy(values)))
            if not numpy.isclose(actual, expected, rtol=1e-6, atol=1e-6):
                raise AssertionError(
                    "median %.9g != numpy %.9g" % (actual, expected)
                )
            result["oracle_comparisons"] += 1
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))

    def reject(name, values):
        try:
            ap.median(ap.ndarray.from_numpy(values))
        except (TypeError, ValueError, RuntimeError):
            record(name, "passed")
        except BaseException as exc:
            record(name, "failed", _detail(exc))
        else:
            record(name, "failed", "unsupported input was accepted")

    compare(
        "odd_length_1d",
        numpy.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=numpy.float32),
    )
    compare(
        "even_length_1d",
        numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32),
    )
    compare("single_element", numpy.array([7.5], dtype=numpy.float32))
    compare(
        "negative_values",
        numpy.array([-4.0, -1.0, -3.0, -2.0], dtype=numpy.float32),
    )
    compare(
        "all_equal",
        numpy.full(6, 2.25, dtype=numpy.float32),
    )
    compare(
        "rank_two_flattened",
        numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32),
    )
    compare(
        "rank_three_flattened",
        numpy.linspace(-3.0, 3.0, 12, dtype=numpy.float32).reshape(2, 3, 2),
    )
    reject("int32_rejected", numpy.array([1, 2, 3], dtype=numpy.int32))
    reject("float64_rejected", numpy.array([1.0, 2.0], dtype=numpy.float64))
    reject("empty_rejected", numpy.array([], dtype=numpy.float32))

    base = numpy.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=numpy.float32)
    expected_base = float(numpy.median(base))
    repeated_failure = None
    for index in range(REPEATED_CALLS):
        try:
            actual = scalar_of(ap.median(ap.ndarray.from_numpy(base)))
            if not numpy.isclose(actual, expected_base, rtol=1e-6, atol=1e-6):
                raise AssertionError(
                    "median %.9g != numpy %.9g" % (actual, expected_base)
                )
            result["oracle_comparisons"] += 1
        except BaseException as exc:
            repeated_failure = "iteration %d: %s" % (index, _detail(exc))
            break
    if repeated_failure is not None:
        record("repeated_calls_stable", "failed", repeated_failure)
    else:
        record("repeated_calls_stable", "passed")

    result["dispatch_observation"] = dict(observed)
    dispatch_passed = observed["native_calls"] >= result["oracle_comparisons"] > 0
    record(
        "official_aclnn_dispatch",
        "passed" if dispatch_passed else "failed",
        "native=%d" % observed["native_calls"],
    )
    return _finish(result, publish)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--api", default=API_NAME)
    arguments = parser.parse_args(argv)

    expected_root = os.path.realpath(arguments.expected_root)
    result = {
        "schema_version": SCHEMA_VERSION,
        "gate": "numpilot.asnumpy_api_correctness",
        "api": arguments.api,
        "status": "failed",
        "asnumpy_file": "",
        "asnumpy_file_under_expected_root": False,
        "asnumpy_version": "",
        "numpy_version": "",
        "python_version": "",
        "cases": [],
        "completed_cases": 0,
        "oracle_comparisons": 0,
        "peak_rss_kib": 0,
        "error": None,
    }

    def publish():
        result["completed_cases"] = len(result["cases"])
        _write_result(arguments.result, result)

    def record(name, status, detail=""):
        result["cases"].append({
            "name": name,
            "status": status,
            "detail": str(detail)[:MAX_DETAIL_CHARS],
        })
        publish()

    try:
        import numpy
        import asnumpy as ap

        module_file = os.path.realpath(getattr(ap, "__file__", ""))
        result["asnumpy_file"] = module_file
        result["asnumpy_file_under_expected_root"] = (
            module_file == expected_root
            or module_file.startswith(expected_root + os.sep)
        )
        result["asnumpy_version"] = str(getattr(ap, "__version__", ""))
        result["numpy_version"] = str(numpy.__version__)
        result["python_version"] = sys.version.split()[0]
        if not result["asnumpy_file_under_expected_root"]:
            raise RuntimeError(
                "imported asnumpy does not come from the isolated runtime"
            )
        if arguments.api == CBRT_API_NAME:
            return _run_cbrt(arguments, result, publish, record, numpy, ap)
        if arguments.api == INVERT_API_NAME:
            return _run_invert(arguments, result, publish, record, numpy, ap)
        if arguments.api == ARGSORT_API_NAME:
            return _run_argsort(arguments, result, publish, record, numpy, ap)
        if arguments.api == SEARCHSORTED_API_NAME:
            return _run_searchsorted(
                arguments, result, publish, record, numpy, ap
            )
        if arguments.api == MEDIAN_API_NAME:
            return _run_median(arguments, result, publish, record, numpy, ap)
        if arguments.api != API_NAME:
            raise RuntimeError(
                "unsupported integration gate API: %s" % arguments.api
            )
        if not callable(getattr(ap, "concatenate", None)):
            raise RuntimeError("asnumpy.concatenate is not available")

        array_module = importlib.import_module("asnumpy.array")
        native_module = importlib.import_module("asnumpy._core").array
        dispatch = _install_dispatch_probe(array_module, native_module)

        def _compare(name, arrays, axis, expected):
            try:
                actual = ap.concatenate(arrays, axis=axis)
                payload = actual.to_numpy()
                numpy.testing.assert_array_equal(payload, expected)
                if payload.dtype != expected.dtype:
                    raise AssertionError(
                        "dtype %r != %r" % (payload.dtype, expected.dtype)
                    )
                if payload.shape != expected.shape:
                    raise AssertionError(
                        "shape %r != %r" % (payload.shape, expected.shape)
                    )
                result["oracle_comparisons"] += 1
                record(name, "passed")
            except BaseException as exc:
                record(name, "failed", _detail(exc))

        def _reject(name, arrays, axis, **kwargs):
            import asnumpy.array as ap_array

            original = ap_array._concatenate
            entered = []

            def _unexpected(*_args, **_kwargs):
                entered.append(True)
                raise AssertionError("native concatenate path was entered")

            ap_array._concatenate = _unexpected
            try:
                try:
                    ap_array.concatenate(arrays, axis=axis, **kwargs)
                except REPRESENTATIVE_EXCEPTIONS as exc:
                    if entered:
                        record(
                            name,
                            "failed",
                            "native path was entered before %s" % type(exc).__name__,
                        )
                    else:
                        record(name, "passed", type(exc).__name__)
                except BaseException as exc:
                    record(name, "failed", _detail(exc))
                else:
                    record(name, "failed", "invalid input was accepted")
            finally:
                ap_array._concatenate = original

        a2x3 = ap.full((2, 3), 1.0, dtype=numpy.float32)
        b2x2 = ap.full((2, 2), 2.0, dtype=numpy.float32)
        np_a2x3 = numpy.full((2, 3), 1.0, dtype=numpy.float32)
        np_b2x2 = numpy.full((2, 2), 2.0, dtype=numpy.float32)

        _compare(
            "axis_negative_one",
            [a2x3, b2x2],
            -1,
            numpy.concatenate([np_a2x3, np_b2x2], axis=-1),
        )
        _compare(
            "single_input_identity",
            [a2x3],
            0,
            np_a2x3,
        )
        _compare(
            "tuple_input_axis_one",
            (a2x3, b2x2),
            1,
            numpy.concatenate([np_a2x3, np_b2x2], axis=1),
        )

        a1 = ap.full((4,), 3.0, dtype=numpy.float32)
        b1 = ap.full((2,), 4.0, dtype=numpy.float32)
        _compare(
            "one_dimensional_axis_zero",
            [a1, b1],
            0,
            numpy.concatenate(
                [
                    numpy.full((4,), 3.0, dtype=numpy.float32),
                    numpy.full((2,), 4.0, dtype=numpy.float32),
                ],
                axis=0,
            ),
        )

        a3 = ap.full((2, 2, 2), 1.0, dtype=numpy.float32)
        b3 = ap.full((2, 3, 2), 2.0, dtype=numpy.float32)
        _compare(
            "three_dimensional_axis_one",
            [a3, b3],
            1,
            numpy.concatenate(
                [
                    numpy.full((2, 2, 2), 1.0, dtype=numpy.float32),
                    numpy.full((2, 3, 2), 2.0, dtype=numpy.float32),
                ],
                axis=1,
            ),
        )

        _reject("empty_sequence_rejected", [], 0)
        _reject("axis_none_rejected", [a2x3], None)
        _reject("axis_out_of_range_rejected", [a2x3], 2)
        _reject(
            "mismatched_non_axis_extent_rejected",
            [a2x3, ap.zeros((3, 3), dtype=numpy.float32)],
            1,
        )
        _reject(
            "mismatched_rank_rejected",
            [a2x3, ap.zeros((2, 3, 1), dtype=numpy.float32)],
            0,
        )
        _reject(
            "mixed_dtype_rejected",
            [a2x3, ap.zeros((2, 2), dtype=numpy.float64)],
            1,
        )
        _reject(
            "numpy_input_rejected",
            [numpy.zeros((2, 2), dtype=numpy.float32)],
            0,
        )
        _reject(
            "generator_input_rejected",
            (item for item in [a2x3]),
            0,
        )
        _reject(
            "dtype_keyword_rejected",
            [a2x3],
            0,
            dtype=numpy.float32,
        )

        class NonContiguous(ap.ndarray):
            @property
            def strides(self):
                base = super().strides
                return (base[1], base[0])

        _reject(
            "non_contiguous_input_rejected",
            [
                NonContiguous(ap.zeros((2, 3), dtype=numpy.float32)),
                ap.zeros((2, 3), dtype=numpy.float32),
            ],
            0,
        )

        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        repeated_failure = None
        for index in range(REPEATED_CALLS):
            try:
                actual = ap.concatenate([a2x3, b2x2], axis=1)
                numpy.testing.assert_array_equal(
                    actual.to_numpy(),
                    numpy.concatenate([np_a2x3, np_b2x2], axis=1),
                )
                result["oracle_comparisons"] += 1
            except BaseException as exc:
                repeated_failure = "iteration %d: %s" % (index, _detail(exc))
                break
        end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_kib"] = end_rss
        result["repeated_calls"] = REPEATED_CALLS
        result["rss_start_kib"] = start_rss
        result["rss_end_kib"] = end_rss
        if repeated_failure is not None:
            record("repeated_calls_stable", "failed", repeated_failure)
        else:
            record("repeated_calls_stable", "passed")

        result["dispatch_observation"] = dict(dispatch)
        dispatch_passed = (
            dispatch["native_calls"] >= result["oracle_comparisons"] > 0
            and dispatch["legacy_calls"] == 0
        )
        record("official_aclnn_dispatch", "passed" if dispatch_passed else "failed",
               "native=%d legacy=%d" % (dispatch["native_calls"], dispatch["legacy_calls"]))
        return _finish(result, publish)
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": _detail(exc)}
        try:
            publish()
        except BaseException:
            pass
        return 2


if __name__ == "__main__":
    sys.exit(main())
