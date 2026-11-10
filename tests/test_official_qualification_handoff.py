"""Portable official handoffs and rank admission retain all evidence bindings."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.gaps.official_candidate_handoff import _candidate_evidence
from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    OfficialGapResolution,
    OfficialOperatorResolutionReport,
    gap_spec_sha256,
)
from numpilot.gaps.official_qualification import (
    OfficialFileRef,
    QualificationError,
    _verify_extended_concat_ranks,
    _verify_gap,
    validate_official_handoff_evidence,
)
from numpilot.gaps.spec import build_gap_specs


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _seal(request):
    request['handoff_id'] = _sha(_canonical({key: value for key, value in request.items() if key != 'handoff_id'}))
    return request


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o600)


@pytest.fixture
def official_handoff():
    source_path = '/unavailable-original-machine/benchmark/fixtures/concat.py'
    source = (
        'import numpy as np\n'
        'def forward(a, b):\n'
        '    return {"out": np.concatenate((a, b), axis=1)}\n'
    ).encode()
    analysis = AnalysisAgent().analyze(source.decode(), source_path)
    specs = build_gap_specs(source.decode(), analysis, target_soc='Ascend910B3', cann_version='9.2.0-beta.2')
    assert len(specs) == 1
    spec = specs[0]
    decision = OfficialGapResolution(
        gap_id=spec.gap_id,
        qualified_api=spec.qualified_api,
        gap_spec_sha256=gap_spec_sha256(spec),
        route=INSTALLED_ADAPTER_ROUTE,
        reason='synthetic installed ACLNN evidence',
        installed_cann={
            'status': 'exact_candidate', 'adapter_candidate': True,
            'missing_symbols': [], 'candidate': {'candidate_id': 'numpy.concatenate-aclnn-cat'},
            'selected_header': {'relative_path': 'include/aclnn_cat.h', 'sha256': 'a' * 64},
            'selected_library': {'relative_path': 'lib64/libopapi.so', 'sha256': 'b' * 64},
            'required_symbols': ['aclnnCatGetWorkspaceSize', 'aclnnCat'],
            'semantic_equivalence': 'unverified',
        },
        ops_nn_source=None, new_kernel_generation_allowed=False,
        adapter_generation_required=True, activation_allowed=False,
    )
    resolution = OfficialOperatorResolutionReport(
        source_path=source_path, source_sha256=_sha(source),
        target_soc='Ascend910B3', cann_version='9.2.0-beta.2', cann_root='/original/cann',
        ops_nn_root='', cann_osl_accepted=False, status='official_candidate_found', decisions=(decision,),
    )
    resolution_bytes = _canonical(resolution.to_dict())
    request = _seal({
        'schema': 'numpilot.official-candidate-gap.v1', 'schema_version': 1,
        'request_type': 'official_candidate_gap', 'status': 'candidate_adapter_required',
        'source_path': source_path, 'source_sha256': _sha(source),
        'target_soc': 'Ascend910B3', 'cann_version': '9.2.0-beta.2',
        'official_resolution_path': '/original-tmp/official-resolution.json',
        'official_resolution_sha256': _sha(resolution_bytes), 'resolution_id': resolution.resolution_id,
        'gap_specs': [spec.to_dict()], 'decisions': [_candidate_evidence(decision)],
        'generated_code': False, 'qualification_performed': False,
        'activation_performed': False, 'network_access_performed': False,
        'system_write_performed': False,
    })
    return request, source, resolution_bytes


def test_official_handoff_is_validated_from_portable_bytes(official_handoff):
    request, source, resolution = official_handoff
    assert not Path(request['source_path']).exists()
    assert not Path(request['official_resolution_path']).exists()
    validate_official_handoff_evidence(request, source, resolution)
    assert request['request_type'] == 'official_candidate_gap'


@pytest.mark.parametrize('change', [
    'unsafe_flag', 'wrong_schema', 'bool_version', 'relative_source', 'source_label',
    'spec_content', 'decision_library', 'decision_missing', 'resolution_id', 'target_soc',
])
def test_resealed_official_handoff_cannot_change_semantic_bindings(official_handoff, change):
    request, source, resolution = official_handoff
    request = deepcopy(request)
    if change == 'unsafe_flag':
        request['activation_performed'] = True
    elif change == 'wrong_schema':
        request['schema'] = 'script_api_gap'
    elif change == 'bool_version':
        request['schema_version'] = True
    elif change == 'relative_source':
        request['source_path'] = 'other.py'
    elif change == 'source_label':
        request['source_path'] = '/other-machine/I006.py'
        request['gap_specs'][0]['source_path'] = request['source_path']
    elif change == 'spec_content':
        request['gap_specs'][0]['signature_variants'] = [{'positional_args': 2, 'keyword_args': []}]
    elif change == 'decision_library':
        request['decisions'][0]['library_sha256'] = 'f' * 64
    elif change == 'decision_missing':
        request['decisions'] = []
    elif change == 'resolution_id':
        request['resolution_id'] = 'f' * 64
    else:
        request['target_soc'] = 'Ascend310P'
    _seal(request)
    with pytest.raises(QualificationError):
        validate_official_handoff_evidence(request, source, resolution)


@pytest.mark.parametrize('change', ['source_bytes', 'resolution_bytes', 'handoff_id'])
def test_official_handoff_rejects_byte_identity_drift(official_handoff, change):
    request, source, resolution = official_handoff
    if change == 'source_bytes':
        source += b'\n'
    elif change == 'resolution_bytes':
        resolution += b'\n'
    else:
        request['handoff_id'] = 'f' * 64
    with pytest.raises(QualificationError):
        validate_official_handoff_evidence(request, source, resolution)


def test_official_gap_validator_uses_portable_resolution_and_bound_source(official_handoff, tmp_path):
    from numpilot.capabilities import OperatorSignature

    request, source, resolution = official_handoff
    _write(tmp_path / 'gap-request.json', _canonical(request))
    _write(tmp_path / 'official-resolution.json', resolution)
    _write(tmp_path / 'source/fixture.py', source)
    overlay = SimpleNamespace(
        gap_spec=SimpleNamespace(
            request=OfficialFileRef('gap-request.json', _sha(_canonical(request))),
            source_sha256=_sha(source), gap_id=request['gap_specs'][0]['gap_id'],
        ),
        source=SimpleNamespace(bundle_path='source/fixture.py', sha256=_sha(source)),
        runtime=SimpleNamespace(cann_version='9.2.0-beta.2', soc_model='Ascend910B3'),
        entries=(SimpleNamespace(
            qualified_api='numpy.concatenate', asnumpy_api='asnumpy.concatenate',
            signature_variants=(OperatorSignature(1, ('axis',)),),
        ),),
    )
    _verify_gap(overlay, str(tmp_path))
    overlay.runtime.soc_model = 'Ascend310P'
    with pytest.raises(QualificationError, match='target and live runtime'):
        _verify_gap(overlay, str(tmp_path))


def _rank_evidence(tmp_path):
    cases = [
        {'name': 'rank_%d' % rank, 'ok': True, 'comparisons': 3, 'shape': [2] * rank, 'dtype': 'float32'}
        for rank in (1, 2, 3)
    ]
    totals = {'cases': 3, 'oracle_comparisons': 9, 'mismatches': 0, 'tested_ranks': [1, 2, 3]}
    raw = {
        'schema': 'numpilot.official-aclnn-gate-evidence', 'schema_version': 1,
        'gate': 'numerical_validation', 'target': 'concatenate', 'oracle': 'numpy',
        'cases': cases, 'totals': totals,
    }
    overlay = SimpleNamespace(entries=(SimpleNamespace(
        registry_key='concatenate', constraints=('dtype=float32', 'rank>=1', 'rank<=3', 'contiguous=true'),
    ),))
    report = SimpleNamespace(claims=deepcopy(totals), evidence={'files': []})
    return overlay, report, raw


def _bind_raw(tmp_path, report, raw):
    relative = 'gates/r1/numerical_validation.concatenate.json'
    payload = _canonical(raw)
    _write(tmp_path / relative, payload)
    report.evidence['files'] = [{'path': relative, 'sha256': _sha(payload)}]


def test_rank_three_admission_matches_hash_bound_raw_oracles(tmp_path):
    overlay, report, raw = _rank_evidence(tmp_path)
    _bind_raw(tmp_path, report, raw)
    _verify_extended_concat_ranks(overlay, report, str(tmp_path))


@pytest.mark.parametrize('change', ['missing_rank', 'failed_case', 'dtype', 'shape', 'total', 'claims', 'unbounded'])
def test_rank_three_cannot_be_advertised_without_passing_raw_cases(tmp_path, change):
    overlay, report, raw = _rank_evidence(tmp_path)
    if change == 'missing_rank':
        raw['cases'][2]['shape'] = [2, 2]
    elif change == 'failed_case':
        raw['cases'][2]['ok'] = False
    elif change == 'dtype':
        raw['cases'][2]['dtype'] = 'float64'
    elif change == 'shape':
        raw['cases'][2]['shape'] = [2, 0, 2]
    elif change == 'total':
        raw['totals']['cases'] = 4
    elif change == 'claims':
        report.claims['tested_ranks'] = [2]
    else:
        overlay.entries[0].constraints = ('dtype=float32', 'rank>=1', 'contiguous=true')
    _bind_raw(tmp_path, report, raw)
    with pytest.raises(QualificationError):
        _verify_extended_concat_ranks(overlay, report, str(tmp_path))


def _kernel_evidence(tmp_path):
    worker = b'# Synthetic worker evidence; never executed.\n'
    native_sha, module_sha, array_sha = 'a' * 64, 'b' * 64, 'c' * 64
    transaction_sha, transaction_file_sha = 'd' * 64, 'e' * 64
    execution = {
        'schema': 'numpilot.cat-profile-worker.v1', 'status': 'passed',
        'worker_sha256': _sha(worker), 'transaction_sha256': transaction_file_sha,
        'runtime': {key: {'sha256': digest} for key, digest in (
            ('core', native_sha), ('module', module_sha), ('array', array_sha))},
        'dispatch': {'native_calls': 4, 'legacy_calls': 0},
        'cases': [
            {'rank': rank, 'shape': [2] * rank, 'axis': 1, 'dtype': 'float32', 'exact_equal': True}
            for rank in (2, 2, 3, 3)
        ],
    }
    trace = (
        'Op Name,OP Type,Task Type,Device_id,Task Duration(us)\n'
        + ''.join('aclnnCat_ConcatD_%d,ConcatD,AI_VECTOR_CORE,0,12.5\n' % i for i in range(4))
    ).encode()
    profile = {
        'schema': 'numpilot.concatenate-kernel-profile.v1', 'status': 'passed',
        'api': 'numpy.concatenate', 'performance_claim': False,
        'transaction_sha256': transaction_sha, 'transaction_file_sha256': transaction_file_sha,
        'runtime_core_sha256': native_sha, 'runtime_module_sha256': module_sha,
        'runtime_array_sha256': array_sha, 'tested_ranks': [2, 3], 'native_calls': 4,
        'cat_kernel_rows': 4, 'op_type_counts': {'ConcatD': 4},
    }
    overlay = SimpleNamespace(
        entries=(SimpleNamespace(registry_key='concatenate', constraints=(
            'dtype=float32', 'rank>=1', 'rank<=3', 'contiguous=true')) ,),
        integration=SimpleNamespace(transaction_sha256=transaction_sha,
            transaction=SimpleNamespace(sha256=transaction_file_sha), runtime_core_sha256=native_sha),
        runtime=SimpleNamespace(module_sha256=module_sha, wrapper_sha256={'array.py': array_sha}),
    )
    report = SimpleNamespace(claims={
        'execution_profile': 'official_concat_rank1_3_aclnn_v1',
        'kernel_evidence_scope': 'independent_profile', 'device_count': 1,
        'kernel_launches': 4, 'native_calls': 2,
    }, evidence={'files': []})
    return overlay, report, profile, execution, trace, worker


def _bind_kernel(tmp_path, report, profile, execution, trace, worker):
    root = 'gates/r1/kernel-profile/'
    members = {'worker.py': worker, 'worker.json': _canonical(execution), 'raw/op_summary.csv': trace}
    for name, payload in members.items():
        _write(tmp_path / (root + name), payload)
    profile['worker'] = {'path': 'worker.py', 'sha256': _sha(worker)}
    profile['execution'] = {'path': 'worker.json', 'sha256': _sha(members['worker.json'])}
    profile['traces'] = [{'path': 'raw/op_summary.csv', 'sha256': _sha(trace), 'rows': 4}]
    profile_bytes = _canonical(profile)
    _write(tmp_path / (root + 'kernel-profile.json'), profile_bytes)
    report.evidence['files'] = [
        {'path': root + name, 'sha256': _sha(payload)}
        for name, payload in {**members, 'kernel-profile.json': profile_bytes}.items()
    ]


def test_kernel_counts_are_independent_of_python_dispatch_counts(tmp_path):
    from numpilot.gaps.official_qualification import _verify_integrated_cat_kernel_profile

    overlay, report, profile, execution, trace, worker = _kernel_evidence(tmp_path)
    _bind_kernel(tmp_path, report, profile, execution, trace, worker)
    assert report.claims['native_calls'] == 2
    assert report.claims['kernel_launches'] == 4
    _verify_integrated_cat_kernel_profile(overlay, report, str(tmp_path))


@pytest.mark.parametrize('change', [
    'transaction', 'runtime', 'worker_identity', 'legacy_dispatch', 'incorrect_values',
    'unlisted_member', 'device', 'task', 'op_type', 'inflated_claim', 'profile_count',
    'nan_duration', 'zero_duration',
])
def test_kernel_profiles_fail_closed_after_bound_evidence_tampering(tmp_path, change):
    from numpilot.gaps.official_qualification import _verify_integrated_cat_kernel_profile

    overlay, report, profile, execution, trace, worker = _kernel_evidence(tmp_path)
    if change == 'transaction':
        profile['transaction_sha256'] = 'f' * 64
    elif change == 'runtime':
        execution['runtime']['core']['sha256'] = 'f' * 64
    elif change == 'worker_identity':
        execution['worker_sha256'] = 'f' * 64
    elif change == 'legacy_dispatch':
        execution['dispatch']['legacy_calls'] = 1
    elif change == 'incorrect_values':
        execution['cases'][2]['exact_equal'] = False
    elif change == 'device':
        trace = trace.replace(b',0,12.5\n', b',1,12.5\n')
    elif change == 'task':
        trace = trace.replace(b'AI_VECTOR_CORE', b'AICPU')
    elif change == 'op_type':
        trace = trace.replace(b'ConcatD', b'Add')
    elif change == 'inflated_claim':
        report.claims['kernel_launches'] = 99
    elif change == 'profile_count':
        profile['cat_kernel_rows'] = 99
    elif change == 'nan_duration':
        trace = trace.replace(b',12.5\n', b',nan\n')
    elif change == 'zero_duration':
        trace = trace.replace(b',12.5\n', b',0\n')
    _bind_kernel(tmp_path, report, profile, execution, trace, worker)
    if change == 'unlisted_member':
        report.evidence['files'] = [ref for ref in report.evidence['files'] if not ref['path'].endswith('/worker.py')]
    with pytest.raises(QualificationError):
        _verify_integrated_cat_kernel_profile(overlay, report, str(tmp_path))
