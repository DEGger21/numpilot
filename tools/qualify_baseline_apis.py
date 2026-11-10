#!/usr/bin/env python3
"""Qualify unchanged baseline APIs with independent NumPy and per-API msprof runs.

This producer never calls a model or modifies an AsNumpy installation. Failed
APIs remain diagnostic evidence and are excluded from the published certificate.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import zipfile

REPO = Path(__file__).resolve().parents[1]
APIS = ('absolute', 'add', 'divide', 'full', 'maximum', 'mean', 'minimum',
        'multiply', 'not_equal', 'square', 'tanh')
SAMPLES = ('G_partial_dedup/G001_concat_three_views',
           'D1_partial_low_kernel/D1005_feature_join', 'I_gap_model/I003_nan_classification')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, allow_identical=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if allow_identical and path.is_file() and not path.is_symlink():
        if path.read_text() == json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n':
            return
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    path.chmod(0o600)


def ref(root, path):
    return {'path': str(Path(path).relative_to(root)), 'sha256': sha(path)}


def save_array(root, name, array):
    import numpy as np
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('xb') as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    return ref(root, path)


def encode(root, name, value):
    import numpy as np
    if type(value) is np.ndarray:
        return {'kind': 'array', **save_array(root, name + '.npy', value)}
    if isinstance(value, np.dtype) or value is np.float32:
        if np.dtype(value) != np.dtype('float32'):
            raise ValueError('unsupported dtype metadata')
        return {'kind': 'dtype', 'value': 'float32'}
    if isinstance(value, (tuple, list)):
        if not value or any(type(x) is not int or x <= 0 for x in value):
            raise ValueError('unsupported shape metadata')
        return {'kind': 'shape', 'value': list(value)}
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if type(value) in (int, float, bool) and math.isfinite(value):
        return {'kind': 'scalar', 'value': value}
    raise ValueError('unsupported argument metadata: ' + type(value).__name__)


def decode(root, item):
    import numpy as np
    kind = item['kind']
    if kind == 'array':
        path = root / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('input hash mismatch')
        return np.load(path, allow_pickle=False)
    if kind == 'shape':
        return tuple(item['value'])
    if kind == 'dtype':
        return np.dtype(item['value'])
    if kind == 'scalar':
        return item['value']
    raise ValueError('unknown argument metadata')


def compare(expected, actual):
    import numpy as np
    assert expected.shape == actual.shape and expected.dtype == actual.dtype, 'shape/dtype mismatch'
    for op in (np.isnan, np.isposinf, np.isneginf):
        assert np.array_equal(op(expected), op(actual)), 'nonfinite mismatch'
    zero = expected == 0
    assert np.all(actual[zero] == 0), 'zero value mismatch'
    assert np.array_equal(np.signbit(expected[zero]), np.signbit(actual[zero])), 'signed zero mismatch'
    if expected.dtype == np.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)


def prepare_cases(root):
    import numpy as np
    from numpilot.benchmark_lowering import lower_benchmark_program
    from numpilot.capabilities import profile_api_call
    cases = []
    first_profiles = {}
    originals = {name: getattr(np, name) for name in APIS}

    def record(name, args, kwargs, output, label):
        profile = profile_api_call(name, args, kwargs)
        if profile.unprofiled_operands or (not profile.inputs and profile.creation_output is None):
            raise ValueError('unprofiled baseline call: ' + name)
        index = len(cases)
        stem = 'data/%04d' % index
        case = {'case_id': '%04d:%s:%s' % (index, label, name), 'api': name,
                'profile': profile.to_dict(),
                'arguments': [encode(root, stem + '-arg%d' % n, value) for n, value in enumerate(args)],
                'keywords': {key: encode(root, stem + '-' + key, value) for key, value in kwargs.items()},
                'expected': save_array(root, stem + '-expected.npy', np.asarray(output))}
        cases.append(case)
        fingerprint = (name, json.dumps(profile.to_dict(), sort_keys=True))
        first_profiles.setdefault(fingerprint, (name, args, kwargs))

    def wrapper(name):
        def call(*args, **kwargs):
            result = originals[name](*args, **kwargs)
            caller = sys._getframe(1)
            if caller.f_code.co_filename.startswith('baseline-case:') and caller.f_code.co_name == 'forward':
                copied = tuple(x.copy() if type(x) is np.ndarray else x for x in args)
                record(name, copied, dict(kwargs), result, caller.f_code.co_filename)
            return result
        return call

    sources = []
    try:
        for name in APIS:
            setattr(np, name, wrapper(name))
        for sample in SAMPLES:
            path = REPO / 'benchmark/cases' / sample / 'program.py'
            source = path.read_text()
            namespace = {'__name__': 'baseline_profile_capture'}
            exec(compile(lower_benchmark_program(source).source, 'baseline-case:' + sample, 'exec'), namespace)
            namespace['forward'](**namespace['build_inputs']())
            sources.append({'sample_id': sample, 'sha256': sha(path)})
    finally:
        for name, value in originals.items():
            setattr(np, name, value)

    # Test nonfinite values and zero signs at every exact tensor call profile.
    # Creation metadata is already an exact value/shape/dtype contract.
    for name, args, kwargs in list(first_profiles.values()):
        if name == 'full':
            continue
        for variant in ('nonfinite', 'zero_signs'):
            changed = []
            for index, value in enumerate(args):
                if type(value) is not np.ndarray:
                    changed.append(value)
                    continue
                palette = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan, .5], dtype=np.float32)
                if variant == 'zero_signs':
                    palette = np.array([0., -0.], dtype=np.float32)
                    if name == 'divide' and index == 1:
                        palette = np.array([1., 1.], dtype=np.float32)
                elif index:
                    palette = np.roll(palette, -1)
                changed.append(np.resize(palette, value.shape).copy())
            with np.errstate(all='ignore'):
                output = originals[name](*changed, **kwargs)
            record(name, tuple(changed), kwargs, output, variant)
    if {case['api'] for case in cases} != set(APIS):
        raise ValueError('source workload did not cover all intended APIs')
    return cases, sources


def runtime_files(site, wheel):
    with zipfile.ZipFile(wheel) as archive:
        expected = {name: hashlib.sha256(archive.read(name)).hexdigest()
                    for name in archive.namelist() if name.startswith('asnumpy/')
                    and (name.endswith('.py') or name.endswith('.so'))}
    for name, digest in expected.items():
        path = site / name
        if path.is_symlink() or sha(path) != digest:
            raise ValueError('live wheel dependency mismatch: ' + name)
    return expected


def install_probe(ap, name):
    module_name = 'statistics' if name == 'mean' else 'array' if name == 'full' else 'logic' if name == 'not_equal' else 'math'
    public = importlib.import_module('asnumpy.' + module_name)
    core = importlib.import_module('asnumpy._core')
    native = getattr(getattr(core, module_name), name)
    if getattr(public, '_' + name) is not native:
        raise ValueError('unexpected native alias: ' + name)
    counts = {'native': 0, 'fallback': 0}
    def counted(*args, **kwargs):
        counts['native'] += 1
        return native(*args, **kwargs)
    setattr(public, '_' + name, counted)
    if name == 'add':
        loops = public.add._ops.ops
        if not loops or any(op.routine is not native for op in loops):
            raise ValueError('add dtype loop is not the audited native dispatch')
        for op in loops:
            op.routine = counted
        def forbidden(*args, **kwargs):
            counts['fallback'] += 1
            raise ValueError('unexpected add fallback loop')
        public.add._fallback = forbidden
    return counts, '_core.' + module_name + '.' + name


def worker(args):
    import numpy as np
    from loguru import logger
    logger.remove()
    from numpilot.backends.npu import NpuBackend
    from numpilot.capabilities import profile_api_call
    root = Path(args.out).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    site = Path(plan['runtime_site']).resolve()
    import asnumpy as ap
    if Path(ap.__file__).resolve().parent.parent != site:
        raise ValueError('wrong runtime installation')
    backend = NpuBackend()
    before = backend.runtime_identity().to_dict()
    hashes = runtime_files(site, Path(plan['wheel']))
    library = ctypes.CDLL('libascendcl.so')
    get_count = library.aclrtGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    get_count.restype = ctypes.c_int
    count = ctypes.c_uint32()
    assert get_count(ctypes.byref(count)) == 0 and count.value == 1
    synchronize = library.aclrtSynchronizeDevice
    synchronize.argtypes, synchronize.restype = [], ctypes.c_int
    counts, target = install_probe(ap, args.worker_api)
    cases, execution, unit, failures = [], [], [], []
    for original in plan['cases']:
        if original['api'] != args.worker_api:
            continue
        case = dict(original)
        host_args = tuple(decode(root, x) for x in case['arguments'])
        kwargs = {key: decode(root, value) for key, value in case['keywords'].items()}
        device_args = tuple(ap.ndarray.from_numpy(x) if type(x) is np.ndarray else x for x in host_args)
        profile = profile_api_call(args.worker_api, device_args, kwargs,
            tensor_profile_extractor=backend._extract_tensor_profile).to_dict()
        if profile != case['profile']:
            raise ValueError('live operand profile drift')
        previous = dict(counts)
        try:
            result = getattr(ap, args.worker_api)(*device_args, **kwargs)
            assert isinstance(result, ap.ndarray), 'API returned a host result'
            assert synchronize() == 0
            actual = result.to_numpy()
            case['actual'] = save_array(root, 'actual/' + case['case_id'].split(':')[0] + '.npy', actual)
            expected = np.load(root / case['expected']['path'], allow_pickle=False)
            compare(expected, actual)
            assert counts['native'] - previous['native'] == 1 and counts['fallback'] == 0
            unit.append({'case_id': case['case_id'], 'status': 'passed'})
        except Exception as exc:
            failures.append({'case_id': case['case_id'], 'reason': str(exc)[:1500]})
            unit.append({'case_id': case['case_id'], 'status': 'failed'})
        cases.append(case)
        execution.append({'case_id': case['case_id'], 'api': args.worker_api, 'profile': profile,
            'native_calls': counts['native'] - previous['native'], 'host_fallback_calls': counts['fallback'] - previous['fallback'],
            'target': target})
    after = backend.runtime_identity().to_dict()
    assert before == after and runtime_files(site, Path(plan['wheel'])) == hashes
    common = {'schema_version': 1, 'transaction_sha256': plan['transaction_sha256'], 'worker_sha256': sha(__file__)}
    folder = root / 'apis' / args.worker_api
    write(folder / 'numerical.json', {'schema': 'numpilot.baseline-api-numerical', **common,
          'runtime_identity_sha256': before['identity_sha256'], 'cases': cases})
    write(folder / 'unit-tests.json', {'schema': 'numpilot.baseline-api-unit-tests', **common,
          'status': 'failed' if failures else 'passed', 'cases': unit})
    write(folder / 'execution.json', {'schema': 'numpilot.baseline-api-execution', **common,
          'api': args.worker_api,
          'status': 'failed' if failures else 'passed', 'runtime_before': before, 'runtime_after': after,
          'runtime_files_sha256': hashes, 'cases': execution,
          'numerical_sha256': sha(folder / 'numerical.json'), 'unit_test_sha256': sha(folder / 'unit-tests.json'),
          'numerical': ref(root, folder / 'numerical.json'), 'unit_tests': ref(root, folder / 'unit-tests.json')})
    write(folder / 'diagnostics.json', {'api': args.worker_api, 'failures': failures, 'device_count': count.value})


def collect(args):
    root = Path(args.out).absolute()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    integration = Path(args.integration_dir).resolve(strict=True)
    transaction = json.loads((integration / 'integration-transaction.json').read_text())
    site = integration / transaction['runtime']['site_relative']
    wheel = integration / transaction['build']['wheel_relative']
    shutil.copyfile(__file__, root / 'worker.py')
    shutil.copytree(args.official_evidence_root, root / 'official')
    shutil.copyfile(args.official_base_manifest, root / 'official-base-manifest.json')
    component = json.loads((REPO / 'components/asnumpy/manifest.json').read_text())['source_bundle']
    for original, name in [('source-manifest.json', 'baseline-source-manifest.json'),
                           (component['filename'], 'baseline-source.tar.gz')]:
        shutil.copyfile(REPO / 'components/asnumpy' / original, root / name)
    manifest = json.loads((root / 'baseline-source-manifest.json').read_text())['files']
    names = set(manifest) | set(transaction['source']['overlay_sha256'])
    if (integration / 'source/src/asnumpy/_version.py').exists():
        names.add('src/asnumpy/_version.py')
    for name in names:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(integration / 'source' / name, destination)
    cases, sources = prepare_cases(root)
    write(root / 'plan.json', {'schema_version': 1, 'transaction_sha256': transaction['transaction_sha256'],
          'runtime_site': str(site), 'wheel': str(wheel), 'cases': cases, 'sources': sources})
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(site), sysconfig.get_path('purelib'), str(REPO)))
    env['PYTHONNOUSERSITE'] = '1'
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    msprof = shutil.which('msprof')
    if not msprof:
        raise ValueError('msprof is unavailable')
    for api in APIS:
        launch = root / ('launch-' + api + '.sh')
        app = [sys.executable, '-S', str(root / 'worker.py'), '--out', str(root), '--worker-api', api]
        launch.write_text('#!/usr/bin/env bash\nset -e\nexec ' + shlex.join(app) + '\n')
        launch.chmod(0o700)
        command = [msprof, '--application=' + str(launch), '--output=' + str(root / 'traces' / api),
                   '--task-time=on', '--runtime-api=on', '--aic-metrics=PipeUtilization']
        with (root / (api + '-profiler.log')).open('x') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if completed.returncode:
            raise RuntimeError('profiler failed for ' + api)
        execution_path = root / 'apis' / api / 'execution.json'
        if not execution_path.is_file():
            raise RuntimeError('worker did not finish for ' + api + '; inspect ' + str(root / (api + '-profiler.log')))
        execution = json.loads(execution_path.read_text())
        print(api + ': ' + execution['status'], flush=True)
    publish(root)


def publish(root):
    from numpilot.baseline_api_qualification import (SCHEMA, PRODUCER,
        BaselineApiCertificate, compute_baseline_qualification_id, validate_baseline_api_certificate)
    all_cases, all_execution, all_unit, traces, workers, excluded = [], [], [], [], [], []
    runtime = None
    hashes = None
    for api in APIS:
        folder = root / 'apis' / api
        execution = json.loads((folder / 'execution.json').read_text())
        if execution['status'] != 'passed':
            excluded.append(api)
            continue
        if runtime is not None and (runtime != execution['runtime_before'] or hashes != execution['runtime_files_sha256']):
            raise ValueError('cross-process runtime drift')
        runtime, hashes = execution['runtime_before'], execution['runtime_files_sha256']
        all_cases.extend(json.loads((folder / 'numerical.json').read_text())['cases'])
        all_unit.extend(json.loads((folder / 'unit-tests.json').read_text())['cases'])
        all_execution.extend(execution['cases'])
        worker_ref = ref(root, folder / 'execution.json')
        workers.append({'api': api, **worker_ref})
        paths = sorted((root / 'traces' / api).rglob('op_summary*.csv'))
        if not paths:
            raise ValueError('no device CSV for ' + api)
        traces.append({'api': api, 'execution': worker_ref, 'files': [ref(root, p) for p in paths]})
    if not all_cases:
        raise ValueError('no APIs passed independent qualification probes')
    plan = json.loads((root / 'plan.json').read_text())
    common = {'schema_version': 1, 'transaction_sha256': plan['transaction_sha256'], 'worker_sha256': sha(root / 'worker.py')}
    write(root / 'numerical.json', {'schema': 'numpilot.baseline-api-numerical', **common,
          'runtime_identity_sha256': runtime['identity_sha256'], 'cases': all_cases}, allow_identical=True)
    write(root / 'unit-tests.json', {'schema': 'numpilot.baseline-api-unit-tests', **common,
          'status': 'passed', 'cases': all_unit}, allow_identical=True)
    write(root / 'execution.json', {'schema': 'numpilot.baseline-api-execution', **common,
          'status': 'passed', 'runtime_before': runtime, 'runtime_after': runtime,
          'runtime_files_sha256': hashes, 'cases': all_execution, 'workers': workers,
          'numerical_sha256': sha(root / 'numerical.json'), 'unit_test_sha256': sha(root / 'unit-tests.json')}, allow_identical=True)
    entries = []
    for api in APIS:
        profiles = {json.dumps(c['profile'], sort_keys=True): c['profile'] for c in all_cases if c['api'] == api}
        if not profiles:
            continue
        signatures = {(p['positional_args'], tuple(p['keyword_args'])) for p in profiles.values()}
        entries.append({'registry_key': api, 'qualified_api': 'numpy.' + api, 'asnumpy_api': 'asnumpy.' + api,
            'constraints': ['dtype=float32', 'rank>=1', 'rank<=3', 'contiguous=true'],
            'signature_variants': [{'positional_args': n, 'keyword_args': list(keys)} for n, keys in sorted(signatures)],
            'profiles': [profiles[key] for key in sorted(profiles)]})
    certificate = {'schema': SCHEMA, 'schema_version': 1, 'producer': PRODUCER,
        'runtime': runtime, 'entries': entries,
        'source': {'baseline_manifest': ref(root, root / 'baseline-source-manifest.json'),
                   'baseline_archive': ref(root, root / 'baseline-source.tar.gz'), 'candidate_tree': 'source',
                   'official_overlay': ref(root, root / 'official/capability-overlay.json'),
                   'official_base_manifest': ref(root, root / 'official-base-manifest.json'),
                   'official_evidence_root': 'official'},
        'evidence': {'worker': ref(root, root / 'worker.py'), 'execution': ref(root, root / 'execution.json'),
                     'numerical': ref(root, root / 'numerical.json'), 'unit_tests': ref(root, root / 'unit-tests.json'), 'traces': traces}}
    certificate['qualification_id'] = compute_baseline_qualification_id(certificate)
    parsed = BaselineApiCertificate.from_dict(certificate)
    validate_baseline_api_certificate(parsed, evidence_root=str(root), current_identity=runtime)
    write(root / 'baseline-api-certificate.json', certificate)
    write(root / 'production-receipt.json', {'status': 'published', 'qualification_id': certificate['qualification_id'],
        'certificate': ref(root, root / 'baseline-api-certificate.json'), 'qualified_apis': [e['registry_key'] for e in entries],
        'excluded_apis': excluded, 'global_runtime_status': runtime['status'], 'model_called': False})
    print(json.dumps({'certificate': str(root / 'baseline-api-certificate.json'), 'excluded': excluded}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration-dir')
    parser.add_argument('--official-evidence-root')
    parser.add_argument('--official-base-manifest')
    parser.add_argument('--out', required=True)
    parser.add_argument('--worker-api', choices=APIS, help=argparse.SUPPRESS)
    parser.add_argument('--publish-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_api:
        worker(args)
    elif args.publish_only:
        publish(Path(args.out).resolve())
    else:
        if not all((args.integration_dir, args.official_evidence_root, args.official_base_manifest)):
            parser.error('all integration/official evidence paths are required')
        collect(args)


if __name__ == '__main__':
    main()
