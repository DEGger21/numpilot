#!/usr/bin/env python3
"""Measure the original I006 NumPilot artifact on one CPU core / one NPU.

Requires a formal scoped-runtime-qualification-ref from the benchmark suite.
The parent orchestrates separate workers; it never imports AsNumpy. The two
NPU scopes run the identical NumPilot-generated forward, without rewriting it:
resident inputs, and warm host inputs through H2D/forward/D2H. Strict runtime
guard overhead is included in both scopes. Ratios below one remain valid data.

Example after loading CANN (same interpreter as qualification):
  python tools/qualified_i006_performance.py --scoped-qualification-ref ref.json \
      --runtime-site /abs/integration/runtime/site --out /abs/new-performance
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write('\n')
    Path(path).chmod(0o600)


def _threadpools():
    from threadpoolctl import threadpool_info
    return [{key: item.get(key) for key in ('user_api', 'internal_api', 'prefix', 'num_threads')}
            for item in threadpool_info() if item.get('user_api') in ('blas', 'openmp')]


def _arrays(inputs):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}
            for name, value in sorted(inputs.items())}


def measure(forward, warmup, samples, synchronize=None):
    from numpilot.scoped_performance import timing_summary
    durations = []
    output = None
    was_enabled = gc.isenabled()
    try:
        if was_enabled:
            gc.disable()
        for _ in range(warmup + samples):
            output = None  # Release prior output outside the measured interval.
            if synchronize is not None:
                synchronize()
            start = time.perf_counter_ns()
            output = forward()
            if synchronize is not None:
                synchronize()
            durations.append(time.perf_counter_ns() - start)
    finally:
        if was_enabled:
            gc.enable()
    return timing_summary(durations[:warmup], durations[warmup:]), output


class _Device:
    def __init__(self):
        self.library = ctypes.CDLL('libascendcl.so')
        self.sync = self.library.aclrtSynchronizeDevice
        self.sync.argtypes = []
        self.sync.restype = ctypes.c_int
        self.get_device = self.library.aclrtGetDevice
        self.get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.get_device.restype = ctypes.c_int
        self.get_count = self.library.aclrtGetDeviceCount
        self.get_count.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.get_count.restype = ctypes.c_int
        self.calls = 0

    def identity(self):
        device, count = ctypes.c_int(-1), ctypes.c_uint32(0)
        if self.get_device(ctypes.byref(device)) or self.get_count(ctypes.byref(count)):
            raise RuntimeError('ACL device identity probe failed')
        result = {'device_index': device.value, 'visible_device_count': count.value}
        if result != {'device_index': 0, 'visible_device_count': 1}:
            raise RuntimeError('benchmark requires exactly one visible NPU device 0')
        return result

    def __call__(self):
        if self.sync() != 0:
            raise RuntimeError('aclrtSynchronizeDevice failed')
        self.calls += 1


def _namespace(path):
    result = {'__name__': '<numpilot-performance>', '__file__': str(path)}
    exec(compile(Path(path).read_text(encoding='utf-8'), str(path), 'exec'), result)
    return result


def worker(config, role, out):
    # Set CPU affinity before numerical imports or dynamic BLAS initialization.
    core = config['protocol']['core']
    os.sched_setaffinity(0, {core})
    sys.path.insert(0, config['repo_root'])
    from numpilot import scoped_performance as contract
    if _sha(__file__) != config['binding']['producer_sha256']:
        raise ValueError('worker code differs from frozen producer')
    for name, digest_name in [('original.py', 'source_sha256'), ('migrated.py', 'generated_sha256'),
                               ('mainline-result.json', 'migration_report_sha256')]:
        if _sha(Path(config['out']) / name) != config['binding'][digest_name]:
            raise ValueError('source/artifact changed before worker launch')
    import numpy as np
    original = _namespace(Path(config['out']) / 'original.py')
    inputs = original['build_inputs']()
    before_inputs = _arrays(inputs)
    record = {'schema_version': 1, 'role': role, 'status': 'passed', 'pid': os.getpid(),
              'binding': config['binding'], 'affinity': {'before': sorted(os.sched_getaffinity(0))},
              'thread_environment': {key: os.environ.get(key) for key in contract.THREAD_KEYS},
              'threadpools': {'before': _threadpools()}, 'inputs': before_inputs, 'timings': {}}
    warmup, samples = config['protocol']['warmup'], config['protocol']['samples']
    if role == 'cpu':
        timing, output = measure(lambda: original['forward'](**inputs), warmup, samples)
        record['timings']['cpu'] = timing
        np.save(out / 'cpu.npy', output['out'], allow_pickle=False)
    elif role == 'npu':
        scope = contract.load_scope(config['scoped_qualification_ref'])
        from tools.qualified_i006_migration import select_runtime_site
        select_runtime_site(Path(config['runtime_site']))
        from numpilot.benchmark_overlay import load_benchmark_overlay
        from numpilot.module_injection import temporary_module, traced_module
        scope_paths = contract.scope_files(scope)
        overlay = load_benchmark_overlay(str(scope_paths['capability-overlay.json']), scope['official_evidence_root'])
        backend = overlay.resolver
        identity = backend.runtime_identity().to_dict()
        if identity != config['runtime_identity']:
            raise ValueError('live runtime differs from the formal scoped migration')
        module = backend.get_module()
        # Logging configuration is outside timing; no compute target is replaced.
        from loguru import logger
        logger.disable('asnumpy')
        guard = backend.prepare_profile_guard(['concatenate'])
        proxy, counts = traced_module(module, ['concatenate'], before_call=guard.validate_call,
                                      targets=guard.runtime_targets, enforce_target_identity=True)
        transfers = {'h2d': 0, 'd2h': 0}
        def upload(value):
            transfers['h2d'] += 1
            return backend.to_device(value)
        def download(value):
            if set(value) != {'out'} or not callable(getattr(value['out'], 'to_numpy', None)):
                raise ValueError('generated result is not the expected device array')
            transfers['d2h'] += 1
            return value['out'].to_numpy()
        # Generated ap.ndarray.from_numpy is observed through the normal module
        # proxy, leaving the actual AsNumpy module and compute target untouched.
        proxy.ndarray = SimpleNamespace(from_numpy=upload)
        device = _Device()
        record['device_before'] = device.identity()
        record['runtime_identity_before'] = identity
        with temporary_module('asnumpy', proxy):
            generated = _namespace(Path(config['out']) / 'migrated.py')
            resident = {name: upload(value) for name, value in inputs.items()}
            setup_h2d = transfers['h2d']
            device()
            start_transfers = dict(transfers)
            timing, output = measure(lambda: generated['forward'](**resident), warmup, samples, device)
            record['timings']['resident'] = timing
            resident_counts = dict(counts)
            resident_measured = {key: transfers[key] - start_transfers[key] for key in transfers}
            final = download(output)
            np.save(out / 'resident.npy', final, allow_pickle=False)
            resident_d2h = transfers['d2h'] - start_transfers['d2h']
            start_transfers = dict(transfers)
            def end_to_end():
                return download(generated['forward'](**inputs))
            timing, final = measure(end_to_end, warmup, samples, device)
            record['timings']['warm_e2e'] = timing
            np.save(out / 'warm_e2e.npy', final, allow_pickle=False)
        iterations = warmup + samples
        record['runtime_call_counts'] = {'resident': resident_counts,
            'warm_e2e': {key: counts[key] - resident_counts[key] for key in counts}}
        record['transfers'] = {
            'resident_setup_h2d_arrays': setup_h2d,
            'resident_measured_h2d_arrays': resident_measured['h2d'],
            'resident_measured_d2h_arrays': resident_measured['d2h'],
            'resident_final_d2h_arrays': resident_d2h,
            'warm_e2e_h2d_arrays_per_call': (transfers['h2d'] - start_transfers['h2d']) / iterations,
            'warm_e2e_d2h_arrays_per_call': (transfers['d2h'] - start_transfers['d2h']) / iterations,
        }
        record['profile_guard'] = guard.to_dict()
        record['runtime_identity_after'] = backend.runtime_identity().to_dict()
        record['device_after'] = device.identity()
        record['synchronization'] = {'api': 'aclrtSynchronizeDevice', 'before_and_after_each_call': True,
            'setup_calls': 1, 'observed_calls': device.calls, 'expected_calls': 1 + 4 * iterations}
    else:
        raise ValueError('unknown worker role')
    if _arrays(inputs) != before_inputs:
        raise ValueError('performance execution changed the canonical inputs')
    record['affinity']['after'] = sorted(os.sched_getaffinity(0))
    record['threadpools']['after'] = _threadpools()
    _write(out / 'worker.json', record)


def _environment(repo):
    from numpilot.scoped_performance import THREAD_KEYS
    env = {key: value for key, value in os.environ.items()
           if not any(marker in key.upper() for marker in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY', 'PRIVATE', 'ACCESS_KEY'))}
    env.update({key: '1' for key in THREAD_KEYS})
    env.update({'PYTHONHASHSEED': '0', 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(repo)})
    return env


def run(args):
    sys.path.insert(0, str(REPO_ROOT))
    from numpilot import scoped_performance as contract
    ref_path = Path(args.scoped_qualification_ref).expanduser().resolve(strict=True)
    ref = _json(ref_path)
    scope = contract.load_scope(ref)
    files = contract.scope_files(scope)
    runtime = _json(files['runtime-before.json'])
    profile = _json(files['profiles.json'])['profiles']['callsites'][0]['profile']
    qualification_guard = _json(files['validation-evidence.json'])['backend_provenance']
    site = Path(args.runtime_site).expanduser().resolve(strict=True)
    if not Path(runtime['module_path']).resolve(strict=True).is_relative_to(site):
        raise ValueError('runtime-site does not match the formal qualification')
    affinity = sorted(os.sched_getaffinity(0))
    core = affinity[0] if args.core is None else args.core
    if core not in affinity:
        raise ValueError('requested CPU core is outside current affinity')
    out = Path(args.out).expanduser().absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    protocol = {'scope': contract.SCOPE, 'core': core, 'cpu_threads_requested': 1,
                'warmup': args.warmup, 'samples': args.samples, 'clock': 'time.perf_counter_ns',
                'includes_runtime_guard': True, 'cold_start_included': False,
                'output_allocation_included': True, 'output_destruction_included': False}
    binding = {'source_sha256': _sha(files['original.py']), 'generated_sha256': _sha(files['migrated.py']),
               'migration_report_sha256': _sha(files['mainline-result.json']),
               'scoped_qualification_sha256': ref['receipt_sha256'], 'producer_sha256': _sha(__file__)}
    for name in ('original.py', 'migrated.py', 'mainline-result.json'):
        shutil.copyfile(files[name], out / name)
    shutil.copyfile(__file__, out / 'runner.py')
    _write(out / 'scoped-qualification-ref.json', ref)
    config = {'repo_root': str(REPO_ROOT), 'out': str(out), 'protocol': protocol,
              'runtime_site': str(site), 'runtime_identity': runtime, 'profile': profile,
              'binding': binding, 'scoped_qualification_ref': ref}
    _write(out / 'config.json', config)
    receipt = {'schema_version': 1, 'receipt_type': contract.RECEIPT_TYPE,
               'producer': contract.PRODUCER, 'sample_id': contract.SAMPLE_ID, 'status': 'failed',
               'framework_identity': {'runner_path': 'runner.py', 'runner_sha256': binding['producer_sha256']},
               'scoped_qualification_ref': ref, 'protocol': protocol,
               'binding': {field: {'path': name, 'sha256': _sha(out / name)} for field, name in (
                   ('source', 'original.py'), ('migrated', 'migrated.py'), ('migration_report', 'mainline-result.json'))},
               'evidence_boundary': {'steady_state': True, 'device_resident': True, 'end_to_end': False,
                   'kernel_only': False, 'includes_runtime_guard': True, 'warm_e2e_measured_separately': True,
                   'global_runtime_audit_promoted': False, 'workload': 'unchanged_original_sample'},
               'workers': {}, 'worker_artifacts': {}, 'outputs': {}, 'gates': {}, 'comparisons': {},
               'performance': {'performance_valid': False, 'issues': [], 'median_speedup': None}}
    try:
        for role in ('cpu', 'npu'):
            work = out / role
            work.mkdir(mode=0o700)
            command = [sys.executable, str(out / 'runner.py'), '--worker', role,
                       '--config', str(out / 'config.json'), '--worker-out', str(work)]
            _write(work / 'command.json', {'argv': command})
            with (work / 'stdout.txt').open('x') as stdout, (work / 'stderr.txt').open('x') as stderr:
                completed = subprocess.run(command, env=_environment(REPO_ROOT), cwd=str(REPO_ROOT),
                                           stdout=stdout, stderr=stderr, timeout=args.timeout)
            if completed.returncode:
                raise RuntimeError(role + ' worker failed; inspect preserved stderr.txt')
            record = _json(work / 'worker.json')
            receipt['workers'][role] = record
            receipt['worker_artifacts'][role] = {'path': role + '/worker.json', 'sha256': _sha(work / 'worker.json')}
        import numpy as np
        arrays = {}
        for name, role in [('cpu', 'cpu'), ('resident', 'npu'), ('warm_e2e', 'npu')]:
            path = out / role / (name + '.npy')
            receipt['outputs'][name] = {'path': str(path.relative_to(out)), 'sha256': _sha(path)}
            arrays[name] = np.load(path, allow_pickle=False)
        comparisons = {name: {'consistent': bool(np.array_equal(arrays['cpu'], arrays[name])),
                              'exact_equal': bool(np.array_equal(arrays['cpu'], arrays[name])),
                              'shape': list(arrays[name].shape), 'dtype': str(arrays[name].dtype)}
                       for name in ('resident', 'warm_e2e')}
        cpu, npu = receipt['workers']['cpu'], receipt['workers']['npu']
        gates = contract.worker_gates(cpu, npu, protocol, profile, runtime, binding, qualification_guard)
        gates['numerically_consistent'] = all(item['consistent'] for item in comparisons.values())
        receipt['gates'], receipt['comparisons'] = gates, comparisons
        valid = all(gates.values())
        metrics = contract.ratio_metrics(cpu['timings']['cpu'], npu['timings']['resident'], contract.SOURCE_SHA256)
        e2e = contract.ratio_metrics(cpu['timings']['cpu'], npu['timings']['warm_e2e'], contract.SOURCE_SHA256)
        receipt['performance'] = {'performance_valid': valid,
            'issues': [name for name, passed in gates.items() if not passed], **metrics,
            'warm_end_to_end': {'directly_sampled': True, **e2e}}
        if not valid:
            raise ValueError('performance gates failed')
        receipt['status'] = 'passed'
        contract.validate_receipt(out, receipt)
    except Exception as exc:
        receipt['status'] = 'failed'
        receipt['performance']['performance_valid'] = False
        receipt['performance']['issues'].append(type(exc).__name__ + ': ' + str(exc))
        for metrics in (receipt['performance'], receipt['performance'].get('warm_end_to_end', {})):
            if metrics.get('median_speedup') is not None:
                metrics['diagnostic_cpu_over_npu'] = metrics['median_speedup']
                metrics['median_speedup'] = None
                metrics['accelerated'] = None
    _write(out / 'receipt.json', receipt)
    _write(out / 'manifest.json', {'schema': 'numpilot.performance-evidence-manifest', 'schema_version': 1,
        'entries': {contract.SAMPLE_ID: {'path': 'receipt.json', 'sha256': _sha(out / 'receipt.json'),
                                       'selector': {'workload': 'unchanged_original_sample'}}}})
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoped-qualification-ref')
    parser.add_argument('--runtime-site')
    parser.add_argument('--out')
    parser.add_argument('--core', type=int)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--worker', choices=('cpu', 'npu'), help=argparse.SUPPRESS)
    parser.add_argument('--config', help=argparse.SUPPRESS)
    parser.add_argument('--worker-out', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.config or not args.worker_out:
            parser.error('worker requires config and output')
        worker(_json(args.config), args.worker, Path(args.worker_out))
        return 0
    if not args.scoped_qualification_ref or not args.runtime_site or not args.out:
        parser.error('--scoped-qualification-ref, --runtime-site and --out are required')
    if not 1 <= args.warmup <= 1000 or not 7 <= args.samples <= 10000 or not 0 < args.timeout <= 3600:
        parser.error('warmup=1..1000, samples=7..10000 and timeout=(0,3600] are required')
    result = run(args)
    print(json.dumps({'status': result['status'], 'performance': result['performance'],
                      'receipt': str(Path(args.out).absolute() / 'receipt.json')}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
