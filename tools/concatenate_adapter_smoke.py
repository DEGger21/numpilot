#!/usr/bin/env python3
"""Direct I006 adapter correctness and timing diagnostics, not migration evidence.

Example (after loading the CANN environment):
  python tools/concatenate_adapter_smoke.py --runtime-site /abs/runtime/site \
      --out /abs/new-run --sizes base,medium,large

The original benchmark file is never edited. Its original build_inputs and
forward are executed; the NPU forward is a new function with the same bytecode
and only its `np` global rebound to AsNumpy. This is explicitly a direct adapter
smoke, not a NumPilot migration or a performance qualification receipt.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import sys
import time
import types

SOURCE_RELATIVE = 'benchmark/cases/I_gap_model/I006_concatenate_axis/program.py'
SOURCE_SHA256 = '17909c8f44bfc183a61e068c96e5df5d5f5e3678e8a8e854f3cc67246fd3823b'
SCALES = {'base': None, 'medium': (8, 128, 64, 128), 'large': (64, 1024, 512, 512)}
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write(path, value):
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write('\n')


def _load_case(source):
    if _sha(source) != SOURCE_SHA256:
        raise ValueError('I006 source hash changed; review the benchmark before running')
    namespace = runpy.run_path(str(source))
    return namespace['build_inputs'], namespace['forward']


def _rebind_forward(forward, module):
    return types.FunctionType(forward.__code__, {**forward.__globals__, 'np': module},
                              forward.__name__, forward.__defaults__, forward.__closure__)


def _inputs(size, builder):
    if size == 'base':
        return builder()
    import numpy as np
    batch, left_width, right_width, features = SCALES[size]
    rng = np.random.default_rng(906)
    return {'x': rng.standard_normal((batch, left_width, features), dtype=np.float32),
            'y': rng.standard_normal((batch, right_width, features), dtype=np.float32)}


def _array_record(array):
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'nbytes': int(array.nbytes),
            'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def _check(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError('output shape/dtype mismatch')
    np.testing.assert_array_equal(actual, expected)
    return {'exact_equal': True, 'max_abs_error': 0.0, **_array_record(actual)}


def _samples(operation, warmup, repeats, sync=None):
    for _ in range(warmup):
        value = operation()
        if sync:
            sync()
        del value
    timings = []
    for _ in range(repeats):
        if sync:
            sync()
        started = time.perf_counter_ns()
        value = operation()
        if sync:
            sync()
        timings.append((time.perf_counter_ns() - started) / 1e6)
        del value  # Allocation is timed; result destruction is outside each interval.
    return {'samples_ms': timings, 'median_ms': statistics.median(timings),
            'min_ms': min(timings), 'max_ms': max(timings)}


class _Sync:
    def __init__(self):
        self.library = ctypes.CDLL('libascendcl.so')
        self.call = self.library.aclrtSynchronizeDevice
        self.call.argtypes = []
        self.call.restype = ctypes.c_int
        get_device = self.library.aclrtGetDevice
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_device.restype = ctypes.c_int
        device = ctypes.c_int(-1)
        if get_device(ctypes.byref(device)) != 0 or device.value != 0:
            raise RuntimeError('expected exactly logical NPU device 0')
        get_count = self.library.aclrtGetDeviceCount
        get_count.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        get_count.restype = ctypes.c_int
        count = ctypes.c_uint32(0)
        if get_count(ctypes.byref(count)) != 0 or count.value != 1:
            raise RuntimeError(f'expected one visible NPU, observed {count.value}')
        self.device = device.value
        self.count = count.value

    def __call__(self):
        if self.call() != 0:
            raise RuntimeError('aclrtSynchronizeDevice failed')


def _runtime_identity(ap, runtime_site):
    import importlib
    core = importlib.import_module('asnumpy._core')
    array = importlib.import_module('asnumpy.array')
    identity = {}
    for name, module in [('module', ap), ('core', core), ('array', array)]:
        path = Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to(runtime_site):
            raise RuntimeError(f'{name} import escaped isolated --runtime-site: {path}')
        identity[name] = {'path': str(path), 'sha256': _sha(path)}
    return identity


def _legacy_regressions(ap):
    import numpy as np
    import importlib
    core = importlib.import_module('asnumpy._core')
    host = np.arange(6, dtype=np.int32).reshape(2, 3)
    a, b = (ap.ndarray.from_numpy(host), ap.ndarray.from_numpy(host + 7))
    results = {}
    for name in ('hstack', 'vstack'):
        expected = getattr(np, name)((host, host + 7))
        actual = getattr(ap, name)((a, b)).to_numpy()
        results[name + '_int32'] = _check(actual, expected)
    actual = core.array.concatenate([a, b], 1).to_numpy()
    results['legacy_native_concatenate_int32'] = _check(actual, np.concatenate((host, host + 7), axis=1))
    return results


def _worker(args):
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [args.cpu_core] or any(os.environ.get(name) != '1' for name in THREAD_VARS):
        raise RuntimeError('worker must start pinned to one CPU with thread limits set')
    import numpy as np
    builder, forward = _load_case(args.source)
    result = {'schema': 'numpilot.concatenate-direct-smoke.worker.v1', 'worker': args.worker,
              'source_sha256': _sha(args.source), 'affinity_before': affinity,
              'thread_limits': {name: os.environ[name] for name in THREAD_VARS}, 'cases': {}}
    try:
        from threadpoolctl import threadpool_info
        result['threadpools'] = threadpool_info()
        if any(item.get('num_threads', 1) != 1 for item in result['threadpools']):
            raise RuntimeError('a numerical threadpool has more than one thread')
    except ImportError:
        result['threadpools'] = []
        result['threadpool_observation'] = 'unavailable; affinity enforced by taskset'
    ap = sync = None
    native_calls = []
    if args.worker == 'npu':
        runtime_site = Path(args.runtime_site).resolve(strict=True)
        sys.path.insert(0, str(runtime_site))
        import_started = time.perf_counter_ns()
        import asnumpy as ap
        import importlib
        importlib.import_module('asnumpy._core')
        importlib.import_module('asnumpy.array')
        result['runtime_import_ms'] = (time.perf_counter_ns() - import_started) / 1e6
        from loguru import logger
        logger.remove()
        result['runtime_before'] = _runtime_identity(ap, runtime_site)
        sync = _Sync()
        result['device'] = {'logical_device': sync.device,
                            'visible_device_count': sync.count,
                            'inherited_ascend_rt_visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                            'inherited_ascend_visible_devices': os.environ.get('ASCEND_VISIBLE_DEVICES'),
                            'synchronize_api': 'aclrtSynchronizeDevice'}
        device_forward = _rebind_forward(forward, ap)
        import importlib
        array_module = importlib.import_module('asnumpy.array')
        native = importlib.import_module('asnumpy._core').array.concatenate_aclnn
        if array_module._concatenate is not native:
            raise RuntimeError('public adapter native alias does not identify concatenate_aclnn')
        legacy = array_module._concatenate_legacy

        def counted_native(arrays, axis):
            native_calls.append((len(arrays), axis))
            return native(arrays, axis)

        def forbidden_legacy(*unused, **unused_keywords):
            raise AssertionError('public concatenate attempted legacy D2D route')

        array_module._concatenate = counted_native
        array_module._concatenate_legacy = forbidden_legacy
    for size in args.sizes.split(','):
        host = _inputs(size, builder)
        expected = forward(**host)['out']
        case = {'inputs': {k: _array_record(v) for k, v in host.items()},
                'oracle': _array_record(expected)}
        if ap is None:
            case['cpu'] = _samples(lambda: forward(**host)['out'], args.warmup, args.repeats)
            case['correctness'] = _check(forward(**host)['out'], expected)
        else:
            def e2e():
                upload = {key: ap.ndarray.from_numpy(value) for key, value in host.items()}
                return device_forward(**upload)['out'].to_numpy()
            started = time.perf_counter_ns()
            first = e2e()
            sync()
            case['first_e2e_ms'] = (time.perf_counter_ns() - started) / 1e6
            case['cold_boundary'] = ('first scale only: runtime_import_ms + first_e2e_ms; '
                                     'excludes interpreter/input construction, later scales already warm')
            case['correctness'] = _check(first, expected)
            device = {key: ap.ndarray.from_numpy(value) for key, value in host.items()}
            case['resident'] = _samples(lambda: device_forward(**device)['out'],
                                        args.warmup, args.repeats, sync)
            case['resident_correctness'] = _check(device_forward(**device)['out'].to_numpy(), expected)
            case['warm_e2e'] = _samples(e2e, args.warmup, args.repeats, sync)
            case['warm_e2e_correctness'] = _check(e2e(), expected)
            case['inputs_survive_repeated_calls'] = {
                key: _check(value.to_numpy(), host[key]) for key, value in device.items()}
            del device, first
        result['cases'][size] = case
        print(f'{args.worker} {size}: correctness passed', flush=True)
        del host, expected
    if ap is not None:
        expected_calls = len(args.sizes.split(',')) * (2 * (args.warmup + args.repeats) + 3)
        if len(native_calls) != expected_calls or any(call != (2, 1) for call in native_calls):
            raise RuntimeError(f'ACLNN binding call count mismatch: {len(native_calls)} / {expected_calls}')
        result['route'] = {'native_binding': 'asnumpy._core.array.concatenate_aclnn',
                           'native_call_count': len(native_calls),
                           'legacy_route_forbidden_during_public_smoke': True}
        array_module._concatenate = native
        array_module._concatenate_legacy = legacy
        result['legacy_regressions'] = _legacy_regressions(ap)
        result['runtime_after'] = _runtime_identity(ap, runtime_site)
        if result['runtime_before'] != result['runtime_after']:
            raise RuntimeError('runtime identity drifted')
    result['affinity_after'] = sorted(os.sched_getaffinity(0))
    if result['affinity_after'] != affinity:
        raise RuntimeError('worker affinity drifted')
    result['status'] = 'passed'
    return result


def _run(args):
    source = Path(args.source).resolve(strict=True)
    _load_case(source)
    runtime_site = Path(args.runtime_site).resolve(strict=True)
    if not (runtime_site / 'asnumpy').is_dir():
        raise ValueError('--runtime-site must contain the isolated asnumpy package')
    out = Path(args.out).absolute()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    shutil.copyfile(source, out / 'source.py')
    env = os.environ.copy()
    env.update({name: '1' for name in THREAD_VARS})
    # Preserve the container's physical-device assignment; worker asserts one visible logical device.
    python_site = Path(sys.executable).parent.parent / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    if not python_site.is_dir():
        raise RuntimeError('use a virtual-environment Python with a site-packages directory')
    env['PYTHONPATH'] = os.pathsep.join((str(runtime_site), str(python_site)))
    env.pop('PYTHONHOME', None)
    workers = {}
    for role in ('cpu', 'npu'):
        command = ['taskset', '--cpu-list', str(args.cpu_core), sys.executable, '-S',
                   str(Path(__file__).resolve()), '--worker', role, '--source', str(out / 'source.py'),
                   '--runtime-site', str(runtime_site), '--out', str(out / (role + '.json')),
                   '--sizes', args.sizes, '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                   '--cpu-core', str(args.cpu_core)]
        with (out / (role + '.log')).open('w', encoding='utf-8') as stream:
            completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'{role} worker failed ({completed.returncode}); see {out / (role + ".log")}')
        workers[role] = json.loads((out / (role + '.json')).read_text())
    cases = {}
    for size in args.sizes.split(','):
        cpu, npu = workers['cpu']['cases'][size], workers['npu']['cases'][size]
        if cpu['inputs'] != npu['inputs'] or cpu['oracle'] != npu['oracle']:
            raise RuntimeError('CPU/NPU inputs or oracle differ')
        cases[size] = {'correctness': npu['correctness'],
                       'cpu_median_ms': cpu['cpu']['median_ms'],
                       'resident_median_ms': npu['resident']['median_ms'],
                       'warm_e2e_median_ms': npu['warm_e2e']['median_ms'],
                       'diagnostic_resident_ratio': cpu['cpu']['median_ms'] / npu['resident']['median_ms'],
                       'diagnostic_warm_e2e_ratio': cpu['cpu']['median_ms'] / npu['warm_e2e']['median_ms']}
    report = {'schema': 'numpilot.concatenate-direct-smoke.v1', 'status': 'passed',
              'source': {'path': str(source), 'sha256': SOURCE_SHA256, 'snapshot': 'source.py'},
              'harness_sha256': _sha(__file__), 'cases': cases,
              'transformation': {'kind': 'direct adapter smoke; not NumPilot migration',
                                 'original_build_inputs': True, 'original_forward_bytecode': True,
                                 'rebound_global': 'np -> asnumpy only for device forward',
                                 'scale_policy': 'base original; others deterministic seed 906 rank3 extensions'},
              'timing': {'warmup': args.warmup, 'repeats': args.repeats,
                         'resident': 'device input upload excluded; allocation/forward/terminal sync included',
                         'warm_e2e': 'upload/forward/download/temporary device cleanup/terminal sync included',
                         'cpu': 'host allocation and original forward; output destruction excluded',
                         'cold': 'runtime import and first base call separately recorded; not process cold-start'},
              'evidence_boundary': {'performance_qualified': False, 'migration_qualified': False,
                                    'strict_npu_qualified': False, 'diagnostic_only': True,
                                    'llm_called': False},
              'workers': {role: {'path': role + '.json', 'sha256': _sha(out / (role + '.json'))}
                          for role in workers}}
    _write(out / 'receipt.json', report)
    print(f'status=passed diagnostic_only=true receipt={out / "receipt.json"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-site', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sizes', default='base,medium', help='comma-separated base,medium,large')
    parser.add_argument('--source', default=str(Path(__file__).resolve().parents[1] / SOURCE_RELATIVE))
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--cpu-core', type=int)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--worker', choices=('cpu', 'npu'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    names = args.sizes.split(',')
    if not names or len(set(names)) != len(names) or any(name not in SCALES for name in names):
        parser.error('--sizes must select unique base,medium,large names')
    if not 0 <= args.warmup <= 100 or not 3 <= args.repeats <= 1000 or args.timeout <= 0:
        parser.error('warmup/repeats/timeout outside supported range')
    available = os.sched_getaffinity(0)
    if args.cpu_core is None:
        args.cpu_core = min(available)
    if args.cpu_core not in available:
        parser.error('--cpu-core is outside the allowed affinity mask')
    try:
        if args.worker:
            _write(Path(args.out), _worker(args))
        else:
            _run(args)
    except Exception as exc:
        failure = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                   'performance_qualified': False, 'migration_qualified': False, 'llm_called': False}
        target = Path(args.out) if args.worker else Path(args.out) / 'failure.json'
        if target.parent.is_dir() and not target.exists():
            _write(target, failure)
        print(json.dumps(failure), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
