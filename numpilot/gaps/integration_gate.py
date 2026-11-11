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
