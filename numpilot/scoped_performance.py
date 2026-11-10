"""Exact-sample performance receipts for a verified official-overlay migration.

Performance qualification means that measurement evidence is valid; it does
not require an acceleration ratio greater than one. Global runtime audit
status is preserved and never promoted by this evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics

RECEIPT_TYPE = 'numpilot_scoped_sample_benchmark'
PRODUCER = 'numpilot-scoped-sample-benchmark-v1'
SCOPE = 'steady_device_resident_generated_forward'
SAMPLE_ID = 'I_gap_model/I006_concatenate_axis'
SOURCE_SHA256 = '17909c8f44bfc183a61e068c96e5df5d5f5e3678e8a8e854f3cc67246fd3823b'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular_file(root, relative):
    root = Path(root).resolve(strict=True)
    if not isinstance(relative, str) or not relative or '\x00' in relative:
        raise ValueError('artifact path is invalid')
    parts = Path(relative)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('artifact path escapes evidence root')
    current = root
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('artifact path contains a symlink')
    if not current.is_file():
        raise ValueError('artifact is not a regular file: ' + relative)
    return current


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(warmup_ns, samples_ns):
    for values in (warmup_ns, samples_ns):
        if not values or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError('timing samples must be positive integer nanoseconds')
    return {'clock': 'time.perf_counter_ns', 'warmup_ns': list(warmup_ns),
            'samples_ns': list(samples_ns), 'median_ns': float(statistics.median(samples_ns)),
            'p10_ns': percentile(samples_ns, .1), 'p90_ns': percentile(samples_ns, .9)}


def ratio_metrics(cpu, npu, source_sha256):
    numerator = cpu['samples_ns']
    denominator = npu['samples_ns']
    rng = random.Random(int(source_sha256[:16], 16))
    ratios = [statistics.median(rng.choices(numerator, k=len(numerator))) /
              statistics.median(rng.choices(denominator, k=len(denominator)))
              for _ in range(2000)]
    ratio = statistics.median(numerator) / statistics.median(denominator)
    return {'cpu_median_ms': statistics.median(numerator) / 1e6,
            'npu_median_ms': statistics.median(denominator) / 1e6,
            'median_speedup': ratio,
            'accelerated': ratio > 1,
            'conservative_cpu_p10_over_npu_p90': percentile(numerator, .1) / percentile(denominator, .9),
            'median_speedup_bootstrap_95ci': [percentile(ratios, .025), percentile(ratios, .975)]}


def load_scope(ref):
    from numpilot.scoped_qualification import validate_scoped_qualification
    scope = validate_scoped_qualification(
        ref, expected_sample_id=SAMPLE_ID, expected_source_sha256=SOURCE_SHA256)
    if scope.get('required_ops') != ['concatenate']:
        raise ValueError('performance producer supports only original I006 concatenate')
    return scope


def scope_files(scope):
    root = Path(scope['evidence']['root'])
    return {name: regular_file(root, name) for name in (
        'original.py', 'migrated.py', 'mainline-result.json', 'profiles.json',
        'runtime-before.json', 'runtime-after.json', 'capability-overlay.json', 'validation-evidence.json')}


def worker_gates(cpu, npu, protocol, profile, runtime_identity, binding, qualification_guard):
    warmup, samples, core = protocol['warmup'], protocol['samples'], protocol['core']
    iterations = warmup + samples
    timings = [('cpu', cpu.get('timings', {}).get('cpu')),
               ('resident', npu.get('timings', {}).get('resident')),
               ('warm_e2e', npu.get('timings', {}).get('warm_e2e'))]
    timing_ok = True
    for _, timing in timings:
        try:
            expected = timing_summary(timing['warmup_ns'], timing['samples_ns'])
            timing_ok &= timing == expected
            timing_ok &= len(timing['warmup_ns']) == warmup and len(timing['samples_ns']) == samples
        except (TypeError, KeyError, ValueError):
            timing_ok = False
    def threadpools_ok(worker):
        for phase in ('before', 'after'):
            pools = worker.get('threadpools', {}).get(phase)
            if not isinstance(pools, list) or not pools:
                return False
            if any(type(item.get('num_threads')) is not int or item['num_threads'] != 1 for item in pools):
                return False
        return True
    guard = npu.get('profile_guard') or {}
    calls = (guard.get('runtime_profile_checks') or {}).get('concatenate') or []
    return {
        'worker_schema': all(type(worker.get('schema_version')) is int and worker['schema_version'] == 1 for worker in (cpu, npu)),
        'worker_status': cpu.get('status') == npu.get('status') == 'passed',
        'separate_processes': type(cpu.get('pid')) is int and type(npu.get('pid')) is int and cpu['pid'] != npu['pid'],
        'worker_roles': cpu.get('role') == 'cpu' and npu.get('role') == 'npu',
        'source_generated_binding': all(worker.get('binding') == binding for worker in (cpu, npu)),
        'same_single_core': all(worker.get('affinity', {}).get(phase) == [core]
                               for worker in (cpu, npu) for phase in ('before', 'after')),
        'thread_environment_one': all(worker.get('thread_environment', {}).get(key) == '1'
                                      for worker in (cpu, npu) for key in THREAD_KEYS),
        'threadpools_one': threadpools_ok(cpu) and threadpools_ok(npu),
        'sample_counts_and_statistics': bool(timing_ok),
        'single_npu': npu.get('device_before') == npu.get('device_after') == {'device_index': 0, 'visible_device_count': 1},
        'runtime_identity_stable': npu.get('runtime_identity_before') == npu.get('runtime_identity_after') == runtime_identity,
        'runtime_profile_guard': all(guard.get(name) is True for name in (
            'strict_npu', 'ok', 'preflight_ok', 'preflight_strict_npu',
            'runtime_profile_ok', 'runtime_target_integrity_ok'))
            and guard.get('profile_status') == 'matched'
            and guard.get('required_ops') == guard.get('profile_scoped') == ['concatenate']
            and all(guard.get(name) == [] for name in ('missing', 'non_strict', 'runtime_target_integrity_errors')),
        'guard_qualification_binding': isinstance(qualification_guard, dict) and all(
            name in qualification_guard and guard.get(name) == qualification_guard[name]
            for name in ('schema_version', 'backend', 'backend_version', 'audit_id', 'environment', 'capabilities')),
        'every_call_profile_matches': len(calls) == 2 * iterations and all(
            call.get('matched') is True and call.get('reasons') == [] and call.get('profile') == profile for call in calls),
        'runtime_calls_exact': npu.get('runtime_call_counts') == {
            'resident': {'concatenate': iterations}, 'warm_e2e': {'concatenate': iterations}},
        'synchronization_exact': npu.get('synchronization') == {
            'api': 'aclrtSynchronizeDevice', 'before_and_after_each_call': True,
            'setup_calls': 1, 'observed_calls': 1 + 4 * iterations,
            'expected_calls': 1 + 4 * iterations},
        'transfer_boundary': npu.get('transfers') == {
            'resident_setup_h2d_arrays': 2, 'resident_measured_h2d_arrays': 0,
            'resident_measured_d2h_arrays': 0, 'resident_final_d2h_arrays': 1,
            'warm_e2e_h2d_arrays_per_call': 2, 'warm_e2e_d2h_arrays_per_call': 1},
        'input_identity': bool(cpu.get('inputs')) and cpu.get('inputs') == npu.get('inputs'),
    }


def validate_receipt(root, receipt):
    """Recheck the official scope, files, outputs, raw timings and all claims."""
    if (receipt.get('receipt_type') != RECEIPT_TYPE or type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1
            or receipt.get('producer') != PRODUCER or receipt.get('status') != 'passed'
            or receipt.get('sample_id') != SAMPLE_ID):
        raise ValueError('scoped performance schema/status is invalid')
    scope = load_scope(receipt.get('scoped_qualification_ref'))
    files = scope_files(scope)
    expected_binding = {
        'source_sha256': digest(files['original.py']),
        'generated_sha256': digest(files['migrated.py']),
        'migration_report_sha256': digest(files['mainline-result.json']),
        'scoped_qualification_sha256': receipt['scoped_qualification_ref']['receipt_sha256'],
        'producer_sha256': receipt.get('framework_identity', {}).get('runner_sha256'),
    }
    binding = receipt.get('binding') or {}
    for field, name in [('source', 'original.py'), ('migrated', 'migrated.py'), ('migration_report', 'mainline-result.json')]:
        entry = binding.get(field) or {}
        path = regular_file(root, entry.get('path'))
        if digest(path) != entry.get('sha256') or digest(path) != digest(files[name]):
            raise ValueError('scoped performance source binding mismatch: ' + field)
    runner = regular_file(root, receipt['framework_identity']['runner_path'])
    if digest(runner) != expected_binding['producer_sha256']:
        raise ValueError('performance producer hash mismatch')
    workers = receipt.get('workers') or {}
    for role in ('cpu', 'npu'):
        entry = (receipt.get('worker_artifacts') or {}).get(role) or {}
        path = regular_file(root, entry.get('path'))
        if digest(path) != entry.get('sha256') or read_json(path) != workers.get(role):
            raise ValueError('performance worker evidence mismatch: ' + role)
    protocol = receipt.get('protocol') or {}
    if (protocol.get('scope') != SCOPE or protocol.get('includes_runtime_guard') is not True
            or type(protocol.get('warmup')) is not int or not 1 <= protocol['warmup'] <= 1000
            or type(protocol.get('samples')) is not int or not 7 <= protocol['samples'] <= 10000
            or type(protocol.get('core')) is not int or protocol['core'] < 0
            or type(protocol.get('cpu_threads_requested')) is not int or protocol['cpu_threads_requested'] != 1
            or protocol.get('cold_start_included') is not False):
        raise ValueError('performance protocol is invalid')
    if receipt.get('evidence_boundary') != {
            'steady_state': True, 'device_resident': True, 'end_to_end': False,
            'kernel_only': False, 'includes_runtime_guard': True,
            'warm_e2e_measured_separately': True, 'global_runtime_audit_promoted': False,
            'workload': 'unchanged_original_sample'}:
        raise ValueError('performance evidence boundary is invalid')
    runtime = read_json(files['runtime-before.json'])
    profiles = read_json(files['profiles.json'])['profiles']['callsites']
    if len(profiles) != 1:
        raise ValueError('I006 must have exactly one scoped callsite')
    qualification_guard = read_json(files['validation-evidence.json'])['backend_provenance']
    gates = worker_gates(workers['cpu'], workers['npu'], protocol, profiles[0]['profile'], runtime, expected_binding, qualification_guard)
    import numpy as np
    arrays = {}
    for name in ('cpu', 'resident', 'warm_e2e'):
        entry = receipt.get('outputs', {}).get(name) or {}
        path = regular_file(root, entry.get('path'))
        if digest(path) != entry.get('sha256'):
            raise ValueError('performance output hash mismatch: ' + name)
        value = np.load(path, allow_pickle=False)
        if value.shape != (3, 6, 5) or value.dtype != np.float32:
            raise ValueError('performance output is outside original I006 shape/dtype')
        arrays[name] = value
    comparisons = {name: {'consistent': bool(np.array_equal(arrays['cpu'], arrays[name])),
                          'exact_equal': bool(np.array_equal(arrays['cpu'], arrays[name])),
                          'shape': [3, 6, 5], 'dtype': 'float32'} for name in ('resident', 'warm_e2e')}
    gates['numerically_consistent'] = all(item['consistent'] for item in comparisons.values())
    if receipt.get('gates') != gates or not all(gates.values()):
        raise ValueError('scoped performance gates failed: ' + ','.join(name for name, passed in gates.items() if not passed))
    if receipt.get('comparisons') != comparisons:
        raise ValueError('scoped performance comparison claims mismatch')
    metrics = ratio_metrics(workers['cpu']['timings']['cpu'], workers['npu']['timings']['resident'], SOURCE_SHA256)
    e2e = ratio_metrics(workers['cpu']['timings']['cpu'], workers['npu']['timings']['warm_e2e'], SOURCE_SHA256)
    performance = {'performance_valid': True, 'issues': [], **metrics,
                   'warm_end_to_end': {'directly_sampled': True, **e2e}}
    if receipt.get('performance') != performance:
        raise ValueError('performance metrics do not match raw samples')
    return {'source_sha256': SOURCE_SHA256,
            'migrated_sha256': expected_binding['generated_sha256'],
            'migration_report_sha256': expected_binding['migration_report_sha256'],
            'runtime_identity_sha256': runtime['identity_sha256'],
            'runtime_identity_status': runtime['status'],
            'scope': SCOPE, 'metrics': metrics,
            'scoped_qualification_ref': receipt['scoped_qualification_ref'],
            'identity_scope': 'official_overlay_callsite',
            'scoped_qualification_receipt_sha256': receipt['scoped_qualification_ref']['receipt_sha256']}
