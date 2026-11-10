"""Sequence operands keep every tensor and the integer axis in scoped receipts."""
from dataclasses import replace

import numpy as np
import pytest

from numpilot.capabilities import (
    CapabilityManifest, ExecutionProvenance, OperatorCallProfile,
    OperatorCapability, OperatorSignature, RuntimeProfileError, RuntimeProfileGuard,
    match_capability_profile, profile_concatenate_call, profile_operator_call,
)
from numpilot.benchmark_overlay import build_analysis_profiles
from tests.test_benchmark_overlay import _overlay


def _capability(name='concatenate'):
    return OperatorCapability(
        name=name, available=True, provenance=ExecutionProvenance.NATIVE_NPU,
        evidence='test-only scoped capability',
        constraints=('dtype=float32', 'rank>=1', 'rank<=3', 'contiguous=true'),
        signature_variants=(OperatorSignature(1, ('axis',)), OperatorSignature(1), OperatorSignature(2)),
    )


def _guard(name='concatenate'):
    return RuntimeProfileGuard(
        CapabilityManifest(backend='npu', backend_version='test-only', operators={name: _capability(name)}, audit_id='test-scope'),
        (name,),
    )


@pytest.mark.parametrize('count', [1, 2, 3, 7])
@pytest.mark.parametrize('container', [tuple, list])
@pytest.mark.parametrize('axis', [1, -2, np.int64(1)])
def test_sequence_all_tensor_metadata_and_axis_reach_runtime_guard(count, container, axis):
    arrays = container(np.ones((3, i + 1, 5), np.float32) for i in range(count))
    args = (arrays,)
    kwargs = {'axis': axis}
    profile = profile_concatenate_call(args, kwargs)
    assert len(profile.inputs) == count
    assert profile.sequence_axis == int(axis)
    assert profile.unprofiled_operands == ()
    assert OperatorCallProfile.from_dict(profile.to_dict()) == profile
    guard = _guard()
    guard.validate_call('concatenate', args, kwargs)
    assert guard.ok and guard.runtime_profile_ok
    check = guard.checks['concatenate'][0]
    assert check['matched'] is True
    assert len(check['profile']['inputs']) == count
    assert check['profile']['sequence_axis'] == int(axis)


@pytest.mark.parametrize('style', ['default', 'positional', 'keyword'])
def test_axis_signature_is_preserved(style):
    arrays = [np.ones((2, 3), np.float32), np.ones((1, 3), np.float32)]
    args, kwargs = ((arrays,), {}) if style == 'default' else ((arrays, 0), {}) if style == 'positional' else ((arrays,), {'axis': 0})
    p = profile_concatenate_call(args, kwargs)
    assert p.sequence_axis == 0
    assert p.positional_args == len(args)
    assert p.keyword_args == tuple(kwargs)
    assert match_capability_profile(_capability(), p).matched


@pytest.mark.parametrize('failure', [
    'empty_sequence', 'nested_sequence', 'scalar_element', 'axis_none',
    'axis_bool', 'axis_numpy_bool', 'axis_float', 'axis_high', 'axis_low',
    'axis_duplicate', 'axis_huge', 'out', 'dtype_keyword', 'casting',
    'mixed_dtype', 'rank_mismatch', 'shape_mismatch', 'strided', 'empty_extent',
    'rank_zero', 'rank_four', 'generator',
])
def test_bad_sequence_or_scope_never_passes_before_dispatch(failure):
    arrays = [np.ones((3, 4, 5), np.float32), np.ones((3, 2, 5), np.float32)]
    args, kwargs = (arrays,), {'axis': 1}
    if failure == 'empty_sequence': args = ([],)
    elif failure == 'nested_sequence': args = ([arrays],)
    elif failure == 'scalar_element': arrays[1] = 2
    elif failure == 'axis_none': kwargs['axis'] = None
    elif failure == 'axis_bool': kwargs['axis'] = True
    elif failure == 'axis_numpy_bool': kwargs['axis'] = np.bool_(True)
    elif failure == 'axis_float': kwargs['axis'] = 1.0
    elif failure == 'axis_high': kwargs['axis'] = 3
    elif failure == 'axis_low': kwargs['axis'] = -4
    elif failure == 'axis_duplicate': args = (arrays, 1)
    elif failure == 'axis_huge': kwargs['axis'] = 2 ** 64
    elif failure in ('out', 'dtype_keyword', 'casting'): kwargs[{'dtype_keyword': 'dtype'}.get(failure, failure)] = None
    elif failure == 'mixed_dtype': arrays[1] = arrays[1].astype(np.float64)
    elif failure == 'rank_mismatch': arrays[1] = np.ones((3, 5), np.float32)
    elif failure == 'shape_mismatch': arrays[1] = np.ones((4, 2, 5), np.float32)
    elif failure == 'strided': arrays[1] = np.ones((3, 4, 5), np.float32)[:, ::2, :]
    elif failure == 'empty_extent': arrays[1] = np.ones((3, 0, 5), np.float32)
    elif failure == 'rank_zero': args, kwargs = ([np.array(1., dtype=np.float32)],), {'axis': 0}
    elif failure == 'rank_four': args, kwargs = ([np.ones((1, 2, 3, 4), np.float32)],), {'axis': 0}
    elif failure == 'generator': args = (iter(arrays),)
    with pytest.raises(RuntimeProfileError):
        _guard().validate_call('concatenate', args, kwargs)


def test_sequence_contract_cannot_promote_unknown_api_or_drop_scalar_checks():
    arrays = [np.ones((2, 3), np.float32), np.ones((2, 3), np.float32)]
    ordinary = profile_operator_call((arrays,), {'axis': 1})
    assert ordinary.sequence_axis is None
    assert ordinary.unprofiled_operands == ('arg[0]', 'kwarg[axis]')
    assert 'sequence_axis' not in ordinary.to_dict()
    assert OperatorCallProfile.from_dict(ordinary.to_dict()) == ordinary
    with pytest.raises(RuntimeProfileError):
        _guard('unknown_api').validate_call('unknown_api', (arrays,), {'axis': 1})
    concat = profile_concatenate_call((arrays,), {'axis': 1})
    assert not match_capability_profile(_capability('unknown_api'), concat).matched


@pytest.mark.parametrize('field,value', [('sequence_axis', None), ('sequence_axis', True), ('sequence_axis', '1'), ('keyword_args', ['out'])])
def test_sequence_profile_decoder_rejects_corrupt_axis_or_signature(field, value):
    profile = profile_concatenate_call(([np.ones((2, 3), np.float32)],), {'axis': 1})
    raw = profile.to_dict()
    raw[field] = value
    with pytest.raises(ValueError):
        OperatorCallProfile.from_dict(raw)


def test_static_and_runtime_profiles_share_sequence_contract_for_i006():
    from pathlib import Path
    source = (Path(__file__).parents[1] / 'benchmark/cases/I_gap_model/I006_concatenate_axis/program.py').read_text()
    result = build_analysis_profiles(source, _overlay())
    assert result.matched_callsites == result.attempted_callsites == 1
    recorded = result.profiles.callsites[0].profile
    scope = {}
    exec(compile(source, 'I006-test.py', 'exec'), scope)
    inputs = scope['build_inputs']()
    actual = profile_concatenate_call(((inputs['x'], inputs['y']),), {'axis': 1})
    assert recorded == actual
    assert [p.shape for p in actual.inputs] == [(3, 4, 5), (3, 2, 5)]
    assert match_capability_profile(_capability(), actual).matched


@pytest.mark.parametrize('style', ['single_ndarray', 'tensor_keyword_axis', 'tensor_positional_axis'])
def test_tensor_metadata_cannot_promote_an_invalid_concatenate_convention(style):
    tensor = np.ones((2, 3), np.float32)
    args, kwargs = (tensor,), {}
    if style == 'tensor_keyword_axis':
        kwargs = {'axis': tensor}
    elif style == 'tensor_positional_axis':
        args = (tensor, tensor)
    actual = profile_concatenate_call(args, kwargs)
    assert actual.sequence_axis is None
    assert actual.inputs == ()
    assert len(actual.unprofiled_operands) == len(args) + len(kwargs)
    assert not match_capability_profile(_capability(), actual).matched
    with pytest.raises(RuntimeProfileError):
        _guard().validate_call('concatenate', args, kwargs)


def test_generic_serialized_tensor_profile_is_not_a_concatenate_sequence_profile():
    tensor = np.ones((2, 3), np.float32)
    generic = profile_operator_call((tensor,), {})
    assert generic.inputs and not generic.unprofiled_operands
    restored = OperatorCallProfile.from_dict(generic.to_dict())
    result = match_capability_profile(_capability(), restored)
    assert not result.matched
    assert 'concatenate requires an explicit sequence-axis profile' in result.reasons


def test_static_profiler_does_not_admit_numpy_single_array_concatenate():
    source = """import numpy as np
def build_inputs():
    return {'x': np.ones((2, 3), dtype=np.float32)}
def forward(x):
    return np.concatenate(x)
"""
    recorded = build_analysis_profiles(source, _overlay())
    assert recorded.attempted_callsites == 1
    assert recorded.matched_callsites == 0
    assert recorded.profiles is None
