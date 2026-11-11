from types import SimpleNamespace

import pytest

import numpy

from numpilot.gaps.integration import IntegrationError, _gate_profile_for_api
from numpilot.gaps.integration_gate import (
    _assert_cbrt,
    _install_dispatch_probe,
)


def test_probe_rejects_existing_copy_binding_even_if_numerically_equivalent():
    old = lambda inputs, axis: inputs
    official = lambda inputs, axis: inputs
    with pytest.raises(RuntimeError, match='reviewed ACLNN binding'):
        _install_dispatch_probe(SimpleNamespace(_concatenate=old),
                                SimpleNamespace(concatenate_aclnn=official))


def test_probe_distinguishes_public_wrapper_dispatch_from_legacy_copy():
    official = lambda inputs, axis: ('native', inputs, axis)
    array_module = SimpleNamespace(_concatenate=official, _concatenate_legacy=lambda *a: a)
    counts = _install_dispatch_probe(array_module, SimpleNamespace(concatenate_aclnn=official))
    assert array_module._concatenate([1], axis=0) == ('native', [1], 0)
    assert counts == {'native_calls': 1, 'legacy_calls': 0}
    with pytest.raises(RuntimeError, match='legacy copy route'):
        array_module._concatenate_legacy([1], 0)
    assert counts == {'native_calls': 1, 'legacy_calls': 1}


def test_gate_profile_binds_cbrt_and_fails_closed_for_unknown_api():
    profile = _gate_profile_for_api("numpy.cbrt")
    assert profile["api"] == "numpy.cbrt"
    assert profile["minimum_oracle_comparisons"] >= 10
    assert profile["minimum_cases"] >= 10
    assert profile["candidate_tests"].endswith(
        "math_tests/test_exponents_and_logarithms.py"
    )
    concatenate = _gate_profile_for_api("numpy.concatenate")
    assert concatenate["api"] == "numpy.concatenate"
    with pytest.raises(IntegrationError):
        _gate_profile_for_api("numpy.tan")


def test_cbrt_oracle_rejects_signed_zero_loss():
    expected = numpy.cbrt(
        numpy.array([-0.0, 0.0], dtype=numpy.float32)
    )
    positive_zero = numpy.array([0.0, 0.0], dtype=numpy.float32)
    with pytest.raises(AssertionError, match="signbit"):
        _assert_cbrt(positive_zero, expected, numpy)
    _assert_cbrt(expected.copy(), expected, numpy)


def test_cbrt_oracle_uses_float32_tolerance_only_for_extreme_magnitudes():
    values = numpy.array([1e30, -1e30], dtype=numpy.float32)
    expected = numpy.cbrt(values)
    observed = expected.copy()
    observed[0] = numpy.float32(float(expected[0]) + 12288.0)
    observed[1] = numpy.float32(float(expected[1]) - 12288.0)
    _assert_cbrt(observed, expected, numpy, rtol=2e-6)
    with pytest.raises(AssertionError, match="relative error"):
        _assert_cbrt(observed, expected, numpy, rtol=1e-6)
    percent_error = expected * numpy.float32(1.01)
    with pytest.raises(AssertionError, match="relative error"):
        _assert_cbrt(percent_error, expected, numpy, rtol=2e-6)


def test_cbrt_oracle_accepts_special_values_and_rejects_nan_pattern_drift():
    values = numpy.array(
        [-27.0, -0.0, 0.0, 27.0, numpy.inf, -numpy.inf, numpy.nan],
        dtype=numpy.float32,
    )
    expected = numpy.cbrt(values)
    _assert_cbrt(expected.copy(), expected, numpy)
    wrong_nan = expected.copy()
    wrong_nan[0] = numpy.nan
    with pytest.raises(AssertionError, match="NaN"):
        _assert_cbrt(wrong_nan, expected, numpy)
