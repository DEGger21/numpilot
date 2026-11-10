"""CPU-only receipt/measurement checks; fake NPU records are contract fixtures."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from numpilot import scoped_performance as contract
from numpilot.capabilities import profile_concatenate_call
from numpilot.performance_evidence import PerformanceEvidenceCatalog, PerformanceEvidenceError, verify_reference
from tools import qualified_i006_performance as tool

ROOT = Path(__file__).parents[1]


def _write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def _fixture(tmp_path, monkeypatch):
    """Only the independent official scope verifier is mocked; all local files,
    worker hashes, raw samples and NumPy outputs are actually re-read."""
    root = tmp_path / 'performance'
    scope_root = tmp_path / 'scope'
    root.mkdir()
    scope_root.mkdir()
    source = ROOT / 'benchmark/cases/I_gap_model/I006_concatenate_axis/program.py'
    (scope_root / 'original.py').write_bytes(source.read_bytes())
    (scope_root / 'migrated.py').write_text('# CPU contract fixture, not NPU proof\n')
    _write(scope_root / 'mainline-result.json', {'fixture': True})
    a, b = np.ones((3, 4, 5), np.float32), np.ones((3, 2, 5), np.float32)
    profile = profile_concatenate_call(((a, b),), {'axis': 1}).to_dict()
    _write(scope_root / 'profiles.json', {'profiles': {'callsites': [{'profile': profile}]}})
    runtime = {'identity_sha256': 'b' * 64, 'status': 'audit_required'}
    _write(scope_root / 'runtime-before.json', runtime)
    _write(scope_root / 'runtime-after.json', runtime)
    _write(scope_root / 'capability-overlay.json', {'fixture': True})
    scope = {'sample_id': contract.SAMPLE_ID, 'source_sha256': contract.SOURCE_SHA256,
             'analysis_source_sha256': 'c' * 64, 'generated_source_sha256': contract.digest(scope_root / 'migrated.py'),
             'profile_sha256': contract.digest(scope_root / 'profiles.json'), 'runtime_identity_sha256': 'b' * 64,
             'overlay_sha256': 'd' * 64, 'qualification_id': 'official-fixture',
             'required_ops': ['concatenate'], 'evidence': {'root': str(scope_root)}}
    ref = {'receipt_sha256': 'e' * 64}
    current = {'receipt_sha256': 'f' * 64}
    scopes = {ref['receipt_sha256']: scope, current['receipt_sha256']: deepcopy(scope)}
    calls = []
    def verified_scope(value):
        calls.append(value)
        if value.get('receipt_sha256') not in scopes:
            raise ValueError('scope chain not verified')
        return deepcopy(scopes[value['receipt_sha256']])
    monkeypatch.setattr(contract, 'load_scope', verified_scope)
    for name in ('original.py', 'migrated.py', 'mainline-result.json'):
        shutil.copyfile(scope_root / name, root / name)
    (root / 'runner.py').write_text('# frozen CPU fixture runner\n')
    binding = {'source_sha256': contract.SOURCE_SHA256, 'generated_sha256': contract.digest(root / 'migrated.py'),
               'migration_report_sha256': contract.digest(root / 'mainline-result.json'),
               'scoped_qualification_sha256': ref['receipt_sha256'], 'producer_sha256': contract.digest(root / 'runner.py')}
    protocol = {'scope': contract.SCOPE, 'warmup': 1, 'samples': 7, 'core': 0,
                'includes_runtime_guard': True, 'cpu_threads_requested': 1, 'cold_start_included': False}
    pool = [{'user_api': 'blas', 'num_threads': 1}]
    common = {'schema_version': 1, 'status': 'passed', 'binding': binding,
              'affinity': {'before': [0], 'after': [0]},
              'thread_environment': {key: '1' for key in contract.THREAD_KEYS},
              'threadpools': {'before': pool, 'after': pool}, 'inputs': tool._arrays({'x': a, 'y': b})}
    cpu = {**deepcopy(common), 'role': 'cpu', 'pid': 101,
           'timings': {'cpu': contract.timing_summary([10], [10] * 7)}}
    npu = {**deepcopy(common), 'role': 'npu', 'pid': 102,
           'timings': {'resident': contract.timing_summary([100], [100] * 7),
                       'warm_e2e': contract.timing_summary([200], [200] * 7)},
           'device_before': {'device_index': 0, 'visible_device_count': 1},
           'device_after': {'device_index': 0, 'visible_device_count': 1},
           'runtime_identity_before': runtime, 'runtime_identity_after': runtime,
           'runtime_call_counts': {'resident': {'concatenate': 8}, 'warm_e2e': {'concatenate': 8}},
           'synchronization': {'api': 'aclrtSynchronizeDevice', 'before_and_after_each_call': True,
                               'setup_calls': 1, 'observed_calls': 33, 'expected_calls': 33},
           'transfers': {'resident_setup_h2d_arrays': 2, 'resident_measured_h2d_arrays': 0,
                         'resident_measured_d2h_arrays': 0, 'resident_final_d2h_arrays': 1,
                         'warm_e2e_h2d_arrays_per_call': 2, 'warm_e2e_d2h_arrays_per_call': 1}}
    npu['profile_guard'] = {**{key: True for key in ('strict_npu', 'ok', 'preflight_ok', 'preflight_strict_npu',
                                                   'runtime_profile_ok', 'runtime_target_integrity_ok')},
        'profile_status': 'matched', 'required_ops': ['concatenate'], 'profile_scoped': ['concatenate'],
        'missing': [], 'non_strict': [], 'runtime_target_integrity_errors': [],
        'runtime_profile_checks': {'concatenate': [{'matched': True, 'reasons': [], 'profile': profile} for _ in range(16)]}}
    qualification_guard = {'schema_version': '1.1', 'backend': 'npu', 'backend_version': 'test-only',
        'audit_id': 'test-qualification', 'environment': {'test': True},
        'capabilities': {'concatenate': {'provenance': 'native_npu'}}}
    npu['profile_guard'].update(qualification_guard)
    _write(scope_root / 'validation-evidence.json', {'backend_provenance': qualification_guard})
    workers = {'cpu': cpu, 'npu': npu}
    worker_artifacts = {}
    for role, worker in workers.items():
        (root / role).mkdir()
        _write(root / role / 'worker.json', worker)
        worker_artifacts[role] = {'path': role + '/worker.json', 'sha256': contract.digest(root / role / 'worker.json')}
    output = np.concatenate((a, b), axis=1)
    outputs = {}
    for name in ('cpu', 'resident', 'warm_e2e'):
        np.save(root / (name + '.npy'), output, allow_pickle=False)
        outputs[name] = {'path': name + '.npy', 'sha256': contract.digest(root / (name + '.npy'))}
    gates = contract.worker_gates(cpu, npu, protocol, profile, runtime, binding, qualification_guard)
    gates['numerically_consistent'] = True
    receipt = {'schema_version': 1, 'receipt_type': contract.RECEIPT_TYPE, 'producer': contract.PRODUCER,
        'sample_id': contract.SAMPLE_ID, 'status': 'passed', 'scoped_qualification_ref': ref,
        'framework_identity': {'runner_path': 'runner.py', 'runner_sha256': binding['producer_sha256']},
        'binding': {field: {'path': name, 'sha256': contract.digest(root / name)} for field, name in (
            ('source', 'original.py'), ('migrated', 'migrated.py'), ('migration_report', 'mainline-result.json'))},
        'protocol': protocol, 'workers': workers, 'worker_artifacts': worker_artifacts, 'outputs': outputs,
        'evidence_boundary': {'steady_state': True, 'device_resident': True, 'end_to_end': False,
            'kernel_only': False, 'includes_runtime_guard': True, 'warm_e2e_measured_separately': True,
            'global_runtime_audit_promoted': False, 'workload': 'unchanged_original_sample'},
        'gates': gates, 'comparisons': {name: {'consistent': True, 'exact_equal': True, 'shape': [3, 6, 5], 'dtype': 'float32'}
                                      for name in ('resident', 'warm_e2e')},
        'performance': {'performance_valid': True, 'issues': [],
            **contract.ratio_metrics(cpu['timings']['cpu'], npu['timings']['resident'], contract.SOURCE_SHA256),
            'warm_end_to_end': {'directly_sampled': True,
                **contract.ratio_metrics(cpu['timings']['cpu'], npu['timings']['warm_e2e'], contract.SOURCE_SHA256)}}}
    return root, receipt, current, scopes, calls


def _catalog(root, receipt):
    _write(root / 'receipt.json', receipt)
    _write(root / 'manifest.json', {'schema': 'numpilot.performance-evidence-manifest', 'schema_version': 1,
        'entries': {contract.SAMPLE_ID: {'path': 'receipt.json', 'sha256': contract.digest(root / 'receipt.json')}}})
    return PerformanceEvidenceCatalog(root)


def _resolve(catalog, scope):
    return catalog.resolve(contract.SAMPLE_ID, source_sha256=contract.SOURCE_SHA256, backend='npu',
        strict_npu_qualified=True, numerical_verified=True, runtime_identity_sha256='b' * 64,
        scoped_qualification=scope)


def test_valid_slower_measurements_qualify_without_promoting_global_runtime(tmp_path, monkeypatch):
    root, receipt, current, scopes, calls = _fixture(tmp_path, monkeypatch)
    validated = contract.validate_receipt(root, receipt)
    assert validated['metrics']['median_speedup'] == .1
    assert validated['metrics']['accelerated'] is False
    assert validated['runtime_identity_status'] == 'audit_required'
    resolution = _resolve(_catalog(root, receipt), current)
    assert resolution.qualified is True
    assert resolution.ref['current_scoped_qualification_receipt_sha256'] == current['receipt_sha256']
    assert resolution.ref['scoped_qualification_receipt_sha256'] != current['receipt_sha256']
    assert verify_reference(resolution.ref, expected_sample_id=contract.SAMPLE_ID,
                            expected_scoped_qualification=current) == resolution.ref
    assert calls  # Real production code invokes the independent scope verifier.


@pytest.mark.parametrize('field', ['source_sha256', 'analysis_source_sha256', 'generated_source_sha256',
                                  'profile_sha256', 'runtime_identity_sha256', 'overlay_sha256',
                                  'qualification_id', 'required_ops'])
def test_current_case_drift_prevents_cross_run_performance_reuse(tmp_path, monkeypatch, field):
    root, receipt, current, scopes, calls = _fixture(tmp_path, monkeypatch)
    scopes[current['receipt_sha256']][field] = ['add'] if field == 'required_ops' else 'changed'
    outcome = _resolve(_catalog(root, receipt), current)
    assert outcome.qualified is False
    assert field in outcome.reason


@pytest.mark.parametrize('field', ['raw_timing', 'claimed_ratio', 'cpu_threads', 'affinity', 'npu_count',
                                  'runtime', 'sync_count', 'missing_call', 'profile_axis', 'source_file',
                                  'worker_file', 'output_value', 'runner_file', 'boundary', 'receipt_schema_bool',
                                  'worker_schema', 'guard_schema', 'guard_audit', 'guard_capability', 'guard_reasons'])
def test_real_evidence_changes_are_rejected_not_only_pass_flags(tmp_path, monkeypatch, field):
    root, receipt, current, scopes, calls = _fixture(tmp_path, monkeypatch)
    npu = receipt['workers']['npu']
    cpu = receipt['workers']['cpu']
    if field == 'raw_timing': npu['timings']['resident']['samples_ns'][0] = -1
    elif field == 'claimed_ratio': receipt['performance']['median_speedup'] = 1000.
    elif field == 'cpu_threads': cpu['threadpools']['after'][0]['num_threads'] = 24
    elif field == 'affinity': cpu['affinity']['after'] = [0, 1]
    elif field == 'npu_count': npu['device_after']['visible_device_count'] = 2
    elif field == 'runtime': npu['runtime_identity_after'] = {'status': 'matched', 'identity_sha256': 'b' * 64}
    elif field == 'sync_count': npu['synchronization']['observed_calls'] -= 1
    elif field == 'missing_call': npu['runtime_call_counts']['resident']['concatenate'] -= 1
    elif field == 'profile_axis': npu['profile_guard']['runtime_profile_checks']['concatenate'][0]['profile'] = {'sequence_axis': 0}
    elif field == 'source_file': (root / 'migrated.py').write_text('# mutated\n')
    elif field == 'worker_file': (root / 'cpu/worker.json').write_text('{}')
    elif field == 'output_value':
        data = np.load(root / 'resident.npy')
        data[0, 0, 0] += 1
        np.save(root / 'resident.npy', data)
        receipt['outputs']['resident']['sha256'] = contract.digest(root / 'resident.npy')
    elif field == 'runner_file': (root / 'runner.py').write_text('# changed runner\n')
    elif field == 'boundary': receipt['evidence_boundary']['kernel_only'] = True
    elif field == 'receipt_schema_bool': receipt['schema_version'] = True
    elif field == 'worker_schema': npu['schema_version'] = 'unknown'
    elif field == 'guard_schema': npu['profile_guard']['schema_version'] = 'unknown'
    elif field == 'guard_audit': npu['profile_guard']['audit_id'] = 'unknown'
    elif field == 'guard_capability': npu['profile_guard']['capabilities'] = {}
    elif field == 'guard_reasons': npu['profile_guard']['runtime_profile_checks']['concatenate'][0]['reasons'] = ['unverified']
    # Rehash changed worker records so validation must examine the facts, not
    # merely catch a stale outer file digest.
    if field not in ('worker_file',):
        for role in ('cpu', 'npu'):
            _write(root / role / 'worker.json', receipt['workers'][role])
            receipt['worker_artifacts'][role]['sha256'] = contract.digest(root / role / 'worker.json')
    with pytest.raises(ValueError):
        contract.validate_receipt(root, receipt)
    assert _resolve(_catalog(root, receipt), current).qualified is False


def test_scope_ref_is_required_and_reference_cannot_be_forged_after_catalog(tmp_path, monkeypatch):
    root, receipt, current, scopes, calls = _fixture(tmp_path, monkeypatch)
    catalog = _catalog(root, receipt)
    assert _resolve(catalog, None).qualified is False
    ref = _resolve(catalog, current).ref
    ref['metrics']['median_speedup'] = 3.
    with pytest.raises(PerformanceEvidenceError, match='does not match'):
        verify_reference(ref, expected_scoped_qualification=current)


@pytest.mark.skipif(not hasattr(os, 'sched_setaffinity'), reason='CPU affinity requires Linux')
def test_cpu_worker_runs_original_i006_on_one_core_without_asnumpy(tmp_path):
    core = min(os.sched_getaffinity(0))
    source = ROOT / 'benchmark/cases/I_gap_model/I006_concatenate_axis/program.py'
    shutil.copyfile(source, tmp_path / 'original.py')
    (tmp_path / 'migrated.py').write_text('# CPU test does not execute this file\n')
    _write(tmp_path / 'mainline-result.json', {})
    shutil.copyfile(tool.__file__, tmp_path / 'runner.py')
    binding = {'source_sha256': contract.digest(tmp_path / 'original.py'),
        'generated_sha256': contract.digest(tmp_path / 'migrated.py'),
        'migration_report_sha256': contract.digest(tmp_path / 'mainline-result.json'),
        'producer_sha256': contract.digest(tmp_path / 'runner.py')}
    _write(tmp_path / 'config.json', {'repo_root': str(ROOT), 'out': str(tmp_path), 'binding': binding,
                                    'protocol': {'core': core, 'warmup': 1, 'samples': 7}})
    out = tmp_path / 'cpu'
    out.mkdir()
    command = [sys.executable, str(tmp_path / 'runner.py'), '--worker', 'cpu',
               '--config', str(tmp_path / 'config.json'), '--worker-out', str(out)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=tool._environment(ROOT), timeout=30)
    assert completed.returncode == 0, completed.stderr
    record = json.loads((out / 'worker.json').read_text())
    assert record['affinity'] == {'before': [core], 'after': [core]}
    assert all(pool['num_threads'] == 1 for pool in record['threadpools']['before'])
    assert len(record['timings']['cpu']['samples_ns']) == 7
    assert 'runtime_identity_before' not in record
    original = tool._namespace(source)
    expected = original['forward'](**original['build_inputs']())['out']
    np.testing.assert_array_equal(np.load(out / 'cpu.npy'), expected)


def test_measure_synchronizes_before_and_after_each_call():
    calls = []
    def operation(): calls.append('forward'); return 1
    def synchronize(): calls.append('sync')
    timing, output = tool.measure(operation, 1, 7, synchronize)
    assert calls == ['sync', 'forward', 'sync'] * 8
    assert output == 1 and len(timing['samples_ns']) == 7
