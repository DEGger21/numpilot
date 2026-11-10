"""CPU-only scope replay tests; synthetic execution facts never touch a device."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpilot.scoped_qualification as sq
from numpilot.benchmark_lowering import lower_benchmark_program
from numpilot.capabilities import (CapabilityManifest, ExecutionProvenance,
    OperatorCapability, OperatorSignature, RuntimeIdentity)
from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.gaps.official_qualification import OfficialOverlayEntry
from numpilot.operator_artifacts import OperatorArtifactStore, OperatorArtifactStoreError
from numpilot.operator_artifact_audit import audit_store


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    path.chmod(0o600)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / 'case'
    root.mkdir(mode=0o700)
    identity = RuntimeIdentity(
        asnumpy_repository='unknown', asnumpy_commit='unknown', asnumpy_tree='unknown',
        asnumpy_version='0.4.1', wrapper_sha256={'__init__.py': 'a' * 64, 'array.py': 'b' * 64},
        module_path='/runtime/asnumpy/__init__.py', module_sha256='a' * 64,
        native_module_path='/runtime/asnumpy/_core.so', native_module_sha256='c' * 64,
        cann_home='/cann', cann_version='9.2.0', soc_model='Ascend910B3',
        python_executable='/bin/python3', python_version='3.12.0',
        bootstrap_receipt_path='/receipt.json', bootstrap_receipt_sha256='unknown',
        audit_id='unaudited-asnumpy-version', status='audit_required', issues=('not globally audited',),
    ).to_dict()
    qualification = 'official-aclnn-' + 'a' * 20
    capability = OperatorCapability(name='concatenate', available=True,
        provenance=ExecutionProvenance.NATIVE_NPU,
        evidence='official ACLNN qualification ' + qualification,
        constraints=('dtype=float32', 'rank>=1', 'rank<=3', 'contiguous=true'),
        signature_variants=(OperatorSignature(positional_args=1, keyword_args=('axis',)),))
    manifest = CapabilityManifest(backend='npu', backend_version='0.4.1',
        operators={'concatenate': capability}, environment={},
        audit_id='unaudited-asnumpy-version+official-qualification-' + 'a' * 20)
    overlay = SimpleNamespace(qualification_id=qualification,
        entries=[OfficialOverlayEntry(registry_key='concatenate', qualified_api='numpy.concatenate',
            asnumpy_api='asnumpy.concatenate', provenance='native_npu',
            constraints=capability.constraints, signature_variants=capability.signature_variants)])
    calls = []
    def official(root, evidence_root, runtime):
        calls.append((root, evidence_root, runtime))
        assert runtime == identity
        return overlay, manifest
    monkeypatch.setattr(sq, '_load_official_admission', official)
    source = ('import numpy as np\n\ndef build_inputs():\n'
              '    return (np.ones((3, 4, 5), dtype=np.float32), np.ones((3, 2, 5), dtype=np.float32))\n'
              '\ndef forward(a, b):\n    return np.concatenate((a, b), axis=1)\n')
    analysis = lower_benchmark_program(source).source
    generated = AstRewriteEngine().rewrite(analysis, OffloadPlan({'concatenate': 'NPU'}, 0))
    for name, value in [('original.py', source), ('analysis_source.py', analysis), ('migrated.py', generated)]:
        (root / name).write_text(value)
    tree = __import__('ast').parse(analysis)
    call = next(node for node in __import__('ast').walk(tree) if isinstance(node, __import__('ast').Call)
                and isinstance(node.func, __import__('ast').Attribute) and node.func.attr == 'concatenate')
    profile = {'inputs': [{'shape': [3, 4, 5], 'rank': 3, 'dtype': 'float32', 'c_contiguous': True},
                           {'shape': [3, 2, 5], 'rank': 3, 'dtype': 'float32', 'c_contiguous': True}],
               'positional_args': 1, 'keyword_args': ['axis'], 'sequence_axis': 1, 'unprofiled_operands': []}
    profiles = {'matched_callsites': 1, 'attempted_callsites': 1, 'notes': [], 'profiles': {
        'schema_version': 1, 'source_sha256': sha(analysis.encode()), 'callsites': [{
            'qualified_name': 'numpy.concatenate', 'lineno': call.lineno, 'col_offset': call.col_offset,
            'profile': profile}]}}
    dump(root / 'profiles.json', profiles)
    dump(root / 'runtime-before.json', identity)
    dump(root / 'runtime-after.json', identity)
    dump(root / 'capability-overlay.json', {'synthetic': 'unit fixture uses explicit admission double'})
    dump(root / 'base-manifest.json', manifest.to_dict())
    admission = {'qualification_id': qualification, 'overlay_sha256': sha((root/'capability-overlay.json').read_bytes()),
        'runtime_identity_sha256': identity['identity_sha256'], 'operators': ['concatenate'],
        'qualified_names': ['numpy.concatenate'], 'admission': 'official_qualification_overlay',
        'execution': 'strict_npu_mainline'}
    dump(root / 'overlay-admission.json', admission)
    guard = {'schema_version': '1.1', 'backend': 'npu', 'backend_version': '0.4.1', 'audit_id': manifest.audit_id,
        'required_ops': ['concatenate'], 'profile_scoped': ['concatenate'], 'profile_status': 'matched',
        'missing': [], 'non_strict': [], 'runtime_target_integrity_errors': [],
        'runtime_call_counts': {'kernel': {'concatenate': 1}, 'artifact': {'concatenate': 1}},
        'runtime_profile_checks': {'concatenate': [
            {'matched': True, 'reasons': [], 'profile': profile}, {'matched': True, 'reasons': [], 'profile': profile}]},
        'environment': {'module_path': identity['module_path'], 'cann_version': identity['cann_version'],
                        'python_version': identity['python_version'], 'soc_version': identity['soc_model']},
        'capabilities': {'concatenate': capability.to_dict()}}
    for name in ('strict_npu', 'ok', 'preflight_ok', 'preflight_strict_npu', 'runtime_profile_ok',
                 'runtime_target_integrity_ok', 'runtime_execution_ok', 'runtime_trace_ok'):
        guard[name] = True
    validation = {'backend': 'npu', 'status': 'pass', 'source_sha256': sha(source.encode()),
        'migrated_sha256': sha(generated.encode()), 'backend_provenance': guard,
        'comparisons': {name: {'consistent': True, 'n_elements': 90, 'max_abs_diff': 0.0,
                               'max_rel_diff': 0.0, 'atol': 1e-5, 'rtol': 1e-4,
                               'failure_path': None, 'failure_index': None, 'reason': None}
                        for name in ('inputs', 'kernel', 'artifact')}}
    for name in ('numerically_consistent', 'input_consistent', 'kernel_consistent', 'artifact_consistent', 'artifact_inputs_executed'):
        validation[name] = True
    dump(root / 'validation-evidence.json', validation)
    result = {'status': 'pass', 'engine': 'ast', 'rewrite_integrity': True,
              'rewrites': {'concatenate': 1}, 'expected_rewrites': {'concatenate': 1},
              'source_sha256': sha(source.encode()), 'analysis_source_sha256': sha(analysis.encode())}
    dump(root / 'mainline-result.json', result)
    kwargs = dict(sample_id='I_gap_model/I006_concatenate_axis', source_sha256=sha(source.encode()),
        analysis_source_sha256=sha(analysis.encode()), overlay_path=root/'capability-overlay.json',
        overlay_evidence_root=tmp_path/'official', expected_runtime_identity=identity)
    return SimpleNamespace(root=root, identity=identity, kwargs=kwargs, calls=calls,
                           validation=validation, profiles=profiles, manifest=manifest, overlay=overlay)


def build(case):
    return sq.build_scoped_qualification(case.root, **case.kwargs)


def runtime_receipt(case, ref):
    return {'backend': 'npu', 'strict_npu_qualified': True, 'numerical_verified': True,
            'performance_qualified': False, 'runtime_identity_status': 'audit_required',
            'runtime_identity_sha256': case.identity['identity_sha256'],
            'source_sha256': case.kwargs['source_sha256'], 'runtime_qualification_scope': ref}


def test_scope_replay_is_exact_and_does_not_promote_global_audit(case):
    ref = build(case)
    scope = sq.validate_scoped_qualification(ref, expected_runtime_identity=case.identity)
    assert scope['runtime_identity_status'] == 'audit_required'
    assert scope['global_runtime_audit_promoted'] is False
    assert scope['performance_valid'] is False
    assert scope['required_ops'] == ['concatenate']
    assert len(case.calls) == 2  # full official admission on seal and each replay


def test_store_and_audit_accept_complete_scope(case, tmp_path):
    ref = build(case)
    store = OperatorArtifactStore(tmp_path / 'store')
    artifact = store.put_runtime_qualification(case.kwargs['sample_id'], runtime_receipt(case, ref))
    stored = json.loads(Path(artifact.receipt_path).read_text())
    assert stored['status'] == 'qualified'
    assert stored['runtime_identity_status'] == 'audit_required'
    assert stored['identity_scope'] == 'official_overlay_callsite'
    assert audit_store(store.root)['ok'] is True


@pytest.mark.parametrize('field,value', [('schema', 'unknown'), ('schema_version', True),
    ('identity_scope', 'global'), ('global_runtime_audit_promoted', True), ('sample_id', 'another')])
def test_reference_tamper_rejected(case, field, value):
    ref = build(case)
    ref[field] = value
    with pytest.raises(sq.ScopedQualificationError):
        sq.validate_scoped_qualification(ref)


@pytest.mark.parametrize('name', ['original.py', 'analysis_source.py', 'migrated.py', 'profiles.json',
    'validation-evidence.json', 'runtime-after.json', 'base-manifest.json', 'capability-overlay.json'])
def test_replay_rechecks_file_hashes(case, name):
    ref = build(case)
    with (case.root / name).open('a') as stream:
        stream.write(' ')
    with pytest.raises(sq.ScopedQualificationError, match='hash mismatch'):
        sq.validate_scoped_qualification(ref)


@pytest.mark.parametrize('fact', ['strict_npu', 'runtime_target_integrity_ok', 'runtime_trace_ok', 'runtime_execution_ok'])
def test_unguarded_success_flags_do_not_qualify(case, fact):
    case.validation['backend_provenance'][fact] = False
    dump(case.root / 'validation-evidence.json', case.validation)
    with pytest.raises(sq.ScopedQualificationError, match='guard failed'):
        build(case)
    assert not (case.root/'scope.json').exists()


@pytest.mark.parametrize('field,value', [('dtype', 'float64'), ('c_contiguous', False), ('rank', 4)])
def test_dtype_rank_layout_profile_must_match_official_constraints(case, field, value):
    profile = case.profiles['profiles']['callsites'][0]['profile']
    profile['inputs'][0][field] = value
    dump(case.root/'profiles.json', case.profiles)
    with pytest.raises(sq.ScopedQualificationError):
        build(case)


def test_actual_call_profile_cannot_differ_from_static_profile(case):
    case.validation['backend_provenance']['runtime_profile_checks']['concatenate'][0]['profile'] = {
        **copy.deepcopy(case.profiles['profiles']['callsites'][0]['profile']), 'sequence_axis': 0}
    dump(case.root/'validation-evidence.json', case.validation)
    with pytest.raises(sq.ScopedQualificationError, match='actual runtime profiles differ'):
        build(case)


def test_rehashed_generated_source_cannot_claim_another_ast(case):
    generated = (case.root/'migrated.py').read_text() + '\n# another source'
    (case.root/'migrated.py').write_text(generated)
    case.validation['migrated_sha256'] = sha(generated.encode())
    dump(case.root/'validation-evidence.json', case.validation)
    with pytest.raises(sq.ScopedQualificationError, match='deterministic AST'):
        build(case)


def test_fake_global_status_rejected(case, tmp_path):
    ref = build(case)
    receipt = runtime_receipt(case, ref)
    receipt['runtime_identity_status'] = 'matched'
    with pytest.raises(OperatorArtifactStoreError, match='cannot promote'):
        OperatorArtifactStore(tmp_path/'store').put_runtime_qualification(case.kwargs['sample_id'], receipt)


@pytest.mark.parametrize('fake', ['official_overlay_callsite', {'identity_scope': 'official_overlay_callsite'}, None])
def test_scope_label_never_bypasses_matched_gate(case, tmp_path, fake):
    receipt = runtime_receipt(case, fake)
    with pytest.raises(OperatorArtifactStoreError):
        OperatorArtifactStore(tmp_path/'store').put_runtime_qualification(case.kwargs['sample_id'], receipt)


def test_foreign_sample_source_and_runtime_rejected(case):
    ref = build(case)
    for kwargs in ({'expected_sample_id': 'other'}, {'expected_source_sha256': 'd'*64},
                   {'expected_runtime_identity_sha256': 'd'*64}):
        with pytest.raises(sq.ScopedQualificationError, match='expected'):
            sq.validate_scoped_qualification(ref, **kwargs)


def test_symlink_cannot_replace_bound_artifact(case, tmp_path):
    ref = build(case)
    path = case.root/'migrated.py'
    same = tmp_path/'same.py'
    same.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(same)
    with pytest.raises(sq.ScopedQualificationError, match='symlink'):
        sq.validate_scoped_qualification(ref)


def test_archive_relocation_keeps_all_hashes(case, tmp_path):
    import shutil
    ref = build(case)
    restored = tmp_path/'restored'
    shutil.copytree(case.root, restored)
    shutil.rmtree(case.root)
    scope = sq.validate_scoped_qualification(ref, evidence_dir=restored,
                                            official_evidence_root=tmp_path/'restored-official')
    assert scope['evidence']['root'] == str(restored)
    assert case.calls[-1][1] == str(tmp_path/'restored-official')


def test_official_chain_loader_uses_real_overlay_and_gate_fixtures(tmp_path):
    from tests.test_official_qualification import _build_fixture
    fixture = _build_fixture(tmp_path / 'official')
    case_root = tmp_path/'case'
    case_root.mkdir(mode=0o700)
    dump(case_root/'capability-overlay.json', fixture.overlay.to_dict())
    dump(case_root/'base-manifest.json', fixture.base.to_dict())
    overlay, admitted = sq._load_official_admission(case_root, str(fixture.root), fixture.identity)
    assert overlay.entries and isinstance(overlay.entries[0], OfficialOverlayEntry)
    assert admitted.operators['concatenate'].provenance is ExecutionProvenance.NATIVE_NPU
    # The loader must actually recheck full integration evidence, not only the overlay JSON.
    fixture.wheel_path.write_bytes(b'tampered wheel')
    with pytest.raises(Exception, match='(?i)hash|sha|wheel'):
        sq._load_official_admission(case_root, str(fixture.root), fixture.identity)


def test_axis_outside_rank_does_not_qualify(case):
    case.profiles['profiles']['callsites'][0]['profile']['sequence_axis'] = 3
    dump(case.root/'profiles.json', case.profiles)
    with pytest.raises(sq.ScopedQualificationError):
        build(case)


def test_actual_runtime_count_requires_integers(case):
    case.validation['backend_provenance']['runtime_call_counts']['kernel']['concatenate'] = True
    dump(case.root/'validation-evidence.json', case.validation)
    with pytest.raises(sq.ScopedQualificationError, match='runtime calls'):
        build(case)


def test_audit_rechecks_external_scope_artifacts(case, tmp_path):
    ref = build(case)
    store = OperatorArtifactStore(tmp_path/'store')
    store.put_runtime_qualification(case.kwargs['sample_id'], runtime_receipt(case, ref))
    (case.root/'migrated.py').write_text('forged generated code')
    report = audit_store(store.root)
    assert report['ok'] is False
    assert any('hash mismatch' in error for error in report['errors'])


def test_scope_does_not_turn_unmeasured_performance_into_qualification(case, tmp_path):
    ref = build(case)
    receipt = runtime_receipt(case, ref)
    receipt['performance_qualified'] = True
    receipt['performance_evidence'] = {
        'schema': 'numpilot.performance-evidence-ref', 'schema_version': 1,
        'status': 'passed', 'performance_valid': True, 'receipt_sha256': 'a'*64,
        'runtime_identity_sha256': case.identity['identity_sha256'],
        'source_sha256': case.kwargs['source_sha256']}
    with pytest.raises(OperatorArtifactStoreError, match='separate exact-sample'):
        OperatorArtifactStore(tmp_path/'store').put_runtime_qualification(case.kwargs['sample_id'], receipt)
