"""CPU-only metadata contracts for exact baseline API qualifications."""
from dataclasses import replace
import json

import numpy as np
import pytest

from numpilot.capabilities import (
    BackendCapabilityError, CapabilityManifest, ExecutionProvenance,
    OperatorCallProfile, OperatorCapability, OperatorSignature,
    RuntimeProfileError, RuntimeProfileGuard, ScalarValueProfile, TensorProfile,
    match_capability_profile, profile_api_call, profile_operator_call,
)


def capability(name, profiles=(), *, signatures=None):
    return OperatorCapability(name, True, ExecutionProvenance.NATIVE_NPU,
        'CPU test capability fixture',
        constraints=('dtype=float32', 'rank>=1', 'rank<=3', 'contiguous=true'),
        signature_variants=signatures or tuple(dict.fromkeys(
            OperatorSignature(profile.positional_args, profile.keyword_args) for profile in profiles)),
        qualified_profiles=tuple(profiles))


def guard(cap):
    return RuntimeProfileGuard(CapabilityManifest('npu', 'CPU-test-only', {cap.name: cap}), (cap.name,))


@pytest.mark.parametrize('axis', [0, 1, -1, -2])
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('positional', [False, True])
def test_mean_preserves_explicit_axis_keepdims_signature_and_exact_allowlist(axis, keepdims, positional):
    value = np.ones((4, 16), np.float32)
    args, kwargs = ((value, axis), {'keepdims': keepdims}) if positional else ((value,), {'axis': axis, 'keepdims': keepdims})
    profile = profile_api_call('mean', args, kwargs)
    assert profile.inputs == (TensorProfile('float32', (4, 16), True),)
    assert profile.reduction_axis == axis and profile.reduction_keepdims is keepdims
    assert not profile.unprofiled_operands and profile.creation_output is None
    assert OperatorCallProfile.from_dict(profile.to_dict()) == profile
    cap = capability('mean', (profile,))
    assert not cap.strict_npu
    assert OperatorCapability.from_dict(cap.to_dict()) == cap
    session = guard(cap)
    assert session.profile_status == 'pending'
    session.validate_call('mean', args, kwargs)
    assert session.runtime_profile_ok
    assert session.checks['mean'][0]['profile'] == profile.to_dict()


def test_mean_missing_keepdims_is_explicit_false_but_original_signature_remains():
    x = np.ones((4, 16), np.float32)
    profile = profile_api_call('mean', (x,), {'axis': 1})
    assert profile.reduction_keepdims is False
    assert profile.keyword_args == ('axis',)
    cap = capability('mean', (profile,))
    session = guard(cap)
    session.validate_call('mean', (x,), {'axis': 1})
    with pytest.raises(RuntimeProfileError):
        session.validate_call('mean', (x,), {'axis': 1, 'keepdims': False})


@pytest.mark.parametrize('case', ['missing_axis','axis_none','axis_bool','axis_numpy_bool', 'axis_numpy_int',
    'axis_float','tuple_axis','axis_high','axis_low','dtype_none','dtype_fp32','out','keepdims_int',
    'keepdims_numpy_bool','duplicate_axis','three_positional','unknown_kw','rank_zero'])
def test_mean_rejects_unreviewed_scalar_and_reduction_conventions(case):
    x = np.ones((4, 16), np.float32)
    accepted = profile_api_call('mean', (x,), {'axis': 1})
    cap = capability('mean', (accepted,))
    args, kwargs = (x,), {'axis': 1}
    if case == 'missing_axis': kwargs = {}
    elif case == 'axis_none': kwargs['axis'] = None
    elif case == 'axis_bool': kwargs['axis'] = True
    elif case == 'axis_numpy_bool': kwargs['axis'] = np.bool_(True)
    elif case == 'axis_numpy_int': kwargs['axis'] = np.int64(1)
    elif case == 'axis_float': kwargs['axis'] = 1.0
    elif case == 'tuple_axis': kwargs['axis'] = (1,)
    elif case == 'axis_high': kwargs['axis'] = 2
    elif case == 'axis_low': kwargs['axis'] = -3
    elif case == 'dtype_none': kwargs['dtype'] = None
    elif case == 'dtype_fp32': kwargs['dtype'] = np.float32
    elif case == 'out': kwargs['out'] = None
    elif case == 'keepdims_int': kwargs['keepdims'] = 1
    elif case == 'keepdims_numpy_bool': kwargs['keepdims'] = np.bool_(True)
    elif case == 'duplicate_axis': args = (x, 1)
    elif case == 'three_positional': args, kwargs = (x, 1, None), {}
    elif case == 'unknown_kw': kwargs['where'] = True
    elif case == 'rank_zero': args, kwargs = (np.array(1, np.float32),), {'axis': 0}
    profile = profile_api_call('mean', args, kwargs)
    assert not match_capability_profile(cap, profile).matched
    with pytest.raises(RuntimeProfileError): guard(cap).validate_call('mean', args, kwargs)


@pytest.mark.parametrize('shape', [(4, 8), [4, 16], 7, (2, 3, 4)])
@pytest.mark.parametrize('fill', [0., -0., .25, 1, np.float32(.3), np.int64(2)])
def test_full_has_no_fake_input_and_records_exact_creation_metadata(shape, fill):
    def forbidden_extract(value):
        raise AssertionError('creation must not extract a fake tensor operand')
    profile = profile_api_call('full', (shape, fill), {'dtype': np.dtype('float32')},
                               tensor_profile_extractor=forbidden_extract)
    dims = (shape,) if isinstance(shape, int) else tuple(shape)
    assert profile.inputs == () and not profile.unprofiled_operands
    assert profile.creation_output == TensorProfile('float32', dims, True)
    assert profile.creation_fill is not None
    assert OperatorCallProfile.from_dict(profile.to_dict()) == profile
    cap = capability('full', (profile,))
    assert match_capability_profile(cap, profile).matched
    session = guard(cap)
    session.validate_call('full', (shape, fill), {'dtype': np.float32})
    assert session.runtime_profile_ok
    assert session.checks['full'][0]['profile']['inputs'] == []


def test_full_signed_zero_and_numpy_scalar_precision_are_not_normalized_away():
    positive = profile_api_call('full', ((4, 8), 0.), {'dtype': np.float32})
    negative = profile_api_call('full', ((4, 8), -0.), {'dtype': np.float32})
    assert positive != negative
    assert positive.creation_fill.value == '0x0.0p+0'
    assert negative.creation_fill.value == '-0x0.0p+0'
    double = profile_api_call('full', ((4, 8), .3), {'dtype': np.float32})
    single = profile_api_call('full', ((4, 8), np.float32(.3)), {'dtype': np.float32})
    assert double.creation_fill != single.creation_fill
    assert single.creation_fill.kind == 'numpy:float32'
    assert single.creation_fill.value == float(np.float32(.3)).hex()
    with pytest.raises(RuntimeProfileError):
        guard(capability('full', (positive,))).validate_call('full', ((4, 8), -0.), {'dtype': np.float32})


@pytest.mark.parametrize('case', ['dtype_missing','dtype_none','dtype_fp64','dtype_positional','out','shape_empty',
    'shape_zero','shape_negative','shape_rank4','shape_bool','shape_float','shape_generator','fill_bool',
    'fill_numpy_bool','fill_nan','fill_inf','fill_negative_inf','fill_complex','fill_string','fill_array',
    'fill_overflow','fill_large_int','unknown_kw'])
def test_full_unsupported_metadata_stays_unprofiled(case):
    args, kwargs = ((4, 8), .5), {'dtype': np.float32}
    if case == 'dtype_missing': kwargs = {}
    elif case == 'dtype_none': kwargs['dtype'] = None
    elif case == 'dtype_fp64': kwargs['dtype'] = np.float64
    elif case == 'dtype_positional': args, kwargs = ((4, 8), .5, np.float32), {}
    elif case == 'out': kwargs['out'] = None
    elif case == 'shape_empty': args = ((), .5)
    elif case == 'shape_zero': args = ((4, 0), .5)
    elif case == 'shape_negative': args = ((4, -1), .5)
    elif case == 'shape_rank4': args = ((1, 2, 3, 4), .5)
    elif case == 'shape_bool': args = ((True, 8), .5)
    elif case == 'shape_float': args = ((4., 8), .5)
    elif case == 'shape_generator': args = (iter((4, 8)), .5)
    elif case == 'fill_bool': args = ((4, 8), True)
    elif case == 'fill_numpy_bool': args = ((4, 8), np.bool_(True))
    elif case == 'fill_nan': args = ((4, 8), float('nan'))
    elif case == 'fill_inf': args = ((4, 8), float('inf'))
    elif case == 'fill_negative_inf': args = ((4, 8), -float('inf'))
    elif case == 'fill_complex': args = ((4, 8), 1j)
    elif case == 'fill_string': args = ((4, 8), '1')
    elif case == 'fill_array': args = ((4, 8), np.array(.5, np.float32))
    elif case == 'fill_overflow': args = ((4, 8), 1e39)
    elif case == 'fill_large_int': args = ((4, 8), 2 ** 80)
    elif case == 'unknown_kw': kwargs['order'] = 'C'
    profile = profile_api_call('full', args, kwargs)
    assert profile.creation_output is None
    assert profile.unprofiled_operands
    accepted = profile_api_call('full', ((4, 8), .5), {'dtype': np.float32})
    with pytest.raises(RuntimeProfileError): guard(capability('full', (accepted,))).validate_call('full', args, kwargs)


@pytest.mark.parametrize('api', ['mean', 'full', 'concatenate'])
def test_api_markers_cannot_be_used_by_another_operator(api):
    x = np.ones((4, 8), np.float32)
    args, kwargs = {'mean': ((x,), {'axis': 1}), 'full': (((4, 8), .5), {'dtype': np.float32}),
                    'concatenate': (((x, x),), {'axis': 1})}[api]
    profile = profile_api_call(api, args, kwargs)
    for name in ('unknown', 'add', *({'mean','full','concatenate'} - {api})):
        assert not match_capability_profile(capability(name, (profile,)), profile).matched


def test_exact_allowlist_rejects_unaudited_shapes_layouts_and_values():
    x = np.ones((4, 8), np.float32)
    profile = profile_api_call('add', (x, x), {})
    cap = capability('add', (profile,))
    for y in [np.ones((8, 4), np.float32), np.ones((4, 8), np.float64), np.ones((4, 16), np.float32)[:, ::2]]:
        with pytest.raises(RuntimeProfileError): guard(cap).validate_call('add', (y, y), {})
    full = profile_api_call('full', ((4, 8), .25), {'dtype': np.float32})
    with pytest.raises(RuntimeProfileError):
        guard(capability('full', (full,))).validate_call('full', ((4, 8), .5), {'dtype': np.float32})


@pytest.mark.parametrize('array', [np.ones((2, 3), np.float64), np.ones((0, 3), np.float32),
                                  np.ones((1, 2, 3, 4), np.float32), np.ones((4, 8), np.float32)[:, ::2]])
def test_exact_allowlist_cannot_grant_outside_fp32_positive_contiguous_rank1_3(array):
    profile = profile_operator_call((array,), {})
    cap = capability('absolute', (profile,))
    assert not match_capability_profile(cap, profile).matched


def test_incomplete_allowlist_capability_is_never_name_only_strict():
    profile = profile_api_call('absolute', (np.ones((4, 8), np.float32),), {})
    cap = OperatorCapability('absolute', True, ExecutionProvenance.NATIVE_NPU,
                             qualified_profiles=(profile,))
    assert cap.strict_npu is False
    with pytest.raises(BackendCapabilityError): guard(cap)
    assert not match_capability_profile(cap, profile).matched


def test_old_serialization_keeps_exact_fields_and_new_empty_fields_are_rejected():
    profile = profile_operator_call((np.ones((4, 8), np.float32),), {})
    assert set(profile.to_dict()) == {'inputs', 'positional_args', 'keyword_args', 'unprofiled_operands'}
    cap = OperatorCapability('absolute', True, ExecutionProvenance.NATIVE_NPU)
    assert set(cap.to_dict()) == {'name','available','provenance','strict_npu','evidence','constraints','signature_variants'}
    assert OperatorCapability.from_dict(json.loads(json.dumps(cap.to_dict()))) == cap
    raw = cap.to_dict(); raw['qualified_profiles'] = []
    with pytest.raises(ValueError): OperatorCapability.from_dict(raw)
    for field in ('reduction_axis','creation_output','creation_fill'):
        raw = profile.to_dict(); raw[field] = None
        with pytest.raises(ValueError): OperatorCallProfile.from_dict(raw)


def test_serialized_api_metadata_and_allowlist_are_strictly_validated():
    full = profile_api_call('full', ((4, 8), .5), {'dtype': np.float32})
    mean = profile_api_call('mean', (np.ones((4, 8), np.float32),), {'axis': 1})
    mutations = []
    raw=full.to_dict();raw['creation_fill']['value']='nan';mutations.append(raw)
    raw=full.to_dict();raw['creation_output']['dtype']='float64';mutations.append(raw)
    raw=full.to_dict();raw['creation_fill']['kind']='python:bool';mutations.append(raw)
    raw=mean.to_dict();raw['reduction_axis']=True;mutations.append(raw)
    raw=mean.to_dict();raw['reduction_keepdims']=1;mutations.append(raw)
    raw=mean.to_dict();raw['creation_output']=full.creation_output.to_dict();mutations.append(raw)
    for raw in mutations:
        with pytest.raises(ValueError): OperatorCallProfile.from_dict(raw)
    with pytest.raises(ValueError): replace(full, reduction_axis=1, reduction_keepdims=False)
    with pytest.raises(ValueError): ScalarValueProfile('python:int', '01')
    with pytest.raises(ValueError): ScalarValueProfile('numpy:float32', float(.3).hex())
    with pytest.raises(ValueError): capability('full', (full, full))


def test_scalar_profile_decoder_rejects_non_scalar_kind_without_untyped_exceptions():
    for kind in ([], {}, None, 1, True):
        with pytest.raises(ValueError):
            ScalarValueProfile.from_dict({'kind': kind, 'value': '0'})
