#!/usr/bin/env python3
"""Collect real msprof kernel rows for one isolated concatenate adapter.

Profiling establishes observed device execution, not performance. Raw output,
worker identity and source hashes remain beside the manifest for independent
qualification validation. Never infer a kernel count from Python call counts.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    Path(path).chmod(0o600)


def summarize_trace(path):
    """Count measured Cat tasks, never a coincidental substring such as Scatter."""
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'OP Type', 'Op Name', 'Task Type', 'Task Duration(us)'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('op_summary CSV lacks required device task columns')
        rows = list(reader)
    types = {}
    cat_rows = 0
    for row in rows:
        kind = row['OP Type']
        types[kind] = types.get(kind, 0) + 1
        if (kind == 'ConcatD' and row['Op Name'] == 'aclnnCat_ConcatD_ConcatD'
                and row['Task Type'] in {'AI_VECTOR_CORE', 'AI_CORE'}):
            duration = float(row['Task Duration(us)'])
            if not 0 < duration < float('inf'):
                raise ValueError('Cat task duration must be finite and positive')
            cat_rows += 1
    return len(rows), types, cat_rows


def runtime_binding(runtime_site, transaction):
    import asnumpy as ap
    core = importlib.import_module('asnumpy._core')
    arrays = importlib.import_module('asnumpy.array')
    files = {}
    for key, module in [('module', ap), ('core', core), ('array', arrays)]:
        path = Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to(runtime_site):
            raise RuntimeError('module escaped isolated runtime: ' + key)
        files[key] = {'path': str(path), 'sha256': digest(path)}
    expected = transaction['source']['overlay_sha256']
    if (files['core']['sha256'] != transaction['runtime']['core_sha256'] or
        files['array']['sha256'] != expected['src/asnumpy/array.py'] or
        files['module']['sha256'] != expected['src/asnumpy/__init__.py']):
        raise RuntimeError('live runtime does not match the integrated transaction')
    return files, ap, core, arrays


def worker(args):
    import numpy as np
    from loguru import logger
    logger.remove()
    site = Path(args.runtime_site).resolve(strict=True)
    transaction = json.loads(Path(args.transaction).read_text())
    before, ap, core, arrays = runtime_binding(site, transaction)
    native = core.array.concatenate_aclnn
    if arrays._concatenate is not native:
        raise RuntimeError('public wrapper does not select concatenate_aclnn')
    observed = {'native_calls': 0, 'legacy_calls': 0}

    def native_call(*a, **kw):
        observed['native_calls'] += 1
        return native(*a, **kw)

    def forbidden(*a, **kw):
        observed['legacy_calls'] += 1
        raise RuntimeError('legacy concatenation route entered')

    arrays._concatenate = native_call
    arrays._concatenate_legacy = forbidden
    library = ctypes.CDLL('libascendcl.so')
    synchronize = library.aclrtSynchronizeDevice
    synchronize.argtypes = []
    synchronize.restype = ctypes.c_int
    cases = []
    for rank, shapes in [(2, ((16, 12), (16, 7))), (3, ((3, 4, 5), (3, 2, 5)))]:
        host = [np.arange(np.prod(shape), dtype=np.float32).reshape(shape) for shape in shapes]
        device = [ap.ndarray.from_numpy(value) for value in host]
        expected = np.concatenate(host, axis=1)
        for repeat in range(2):
            actual = ap.concatenate(device, axis=1)
            if synchronize() != 0:
                raise RuntimeError('device synchronization failed')
            actual_host = actual.to_numpy()
            np.testing.assert_array_equal(actual_host, expected)
            cases.append({'rank': rank, 'repeat': repeat, 'axis': 1,
                          'shape': list(actual_host.shape), 'dtype': str(actual_host.dtype),
                          'exact_equal': True})
    after, *_ = runtime_binding(site, transaction)
    if before != after or observed != {'native_calls': 4, 'legacy_calls': 0}:
        raise RuntimeError('runtime identity or dispatch drift')
    write_json(Path(args.out), {'schema': 'numpilot.cat-profile-worker.v1', 'status': 'passed',
        'transaction_sha256': digest(args.transaction), 'worker_sha256': digest(__file__),
        'runtime': before, 'dispatch': observed, 'cases': cases})


def collect(args):
    output = Path(args.out).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    runtime_site = Path(args.runtime_site).resolve(strict=True)
    transaction = Path(args.transaction).resolve(strict=True)
    profiler = shutil.which('msprof')
    if not profiler:
        raise RuntimeError('msprof is unavailable')
    import sysconfig
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(runtime_site), sysconfig.get_path('purelib')))
    env['PYTHONNOUSERSITE'] = '1'
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    worker_script = output / 'worker.py'
    shutil.copyfile(__file__, worker_script)
    app = [sys.executable, '-S', str(worker_script), '--worker',
           '--runtime-site', str(runtime_site), '--transaction', str(transaction),
           '--out', str(output / 'worker.json')]
    launch = output / 'launch.sh'
    launch.write_text('#!/usr/bin/env bash\nset -e\nexec ' + shlex.join(app) + '\n')
    launch.chmod(0o700)
    command = [profiler, '--application=' + str(launch), '--output=' + str(output / 'raw'),
               '--task-time=on', '--runtime-api=on', '--aic-metrics=PipeUtilization']
    with (output / 'profiler.log').open('x') as stream:
        completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=300, check=False)
    if completed.returncode:
        raise RuntimeError('msprof failed; inspect profiler.log')
    info = json.loads((output / 'worker.json').read_text())
    if info.get('status') != 'passed' or info['worker_sha256'] != digest(__file__):
        raise RuntimeError('profile worker did not pass')
    csvs = sorted((output / 'raw').rglob('op_summary*.csv'))
    if not csvs:
        raise RuntimeError('msprof produced no op_summary CSV')
    files = []
    types = {}
    cat_rows = 0
    for path in csvs:
        count, counts, observed = summarize_trace(path)
        for key, value in counts.items():
            types[key] = types.get(key, 0) + value
        cat_rows += observed
        files.append({'path': str(path.relative_to(output)), 'sha256': digest(path), 'rows': count})
    if not cat_rows:
        raise RuntimeError('msprof produced no measured aclnnCat device tasks')
    manifest = {'schema': 'numpilot.concatenate-kernel-profile.v1', 'status': 'passed',
                'api': 'numpy.concatenate', 'transaction_sha256': json.loads(transaction.read_text())['transaction_sha256'],
                'transaction_file_sha256': digest(transaction),
                'runtime_core_sha256': info['runtime']['core']['sha256'],
                'runtime_module_sha256': info['runtime']['module']['sha256'],
                'runtime_array_sha256': info['runtime']['array']['sha256'],
                'worker': {'path': 'worker.py', 'sha256': digest(worker_script)},
                'execution': {'path': 'worker.json', 'sha256': digest(output / 'worker.json')},
                'native_calls': info['dispatch']['native_calls'],
                'tested_ranks': sorted({case['rank'] for case in info['cases']}),
                'traces': files, 'op_type_counts': types,
                'cat_kernel_rows': cat_rows,
                'profiler_command': command, 'performance_claim': False,
                'boundary': 'Observed profiler rows from a separate rank-2/rank-3 execution; not inferred from Python calls.'}
    write_json(output / 'kernel-profile.json', manifest)
    print(json.dumps({'profile': str(output / 'kernel-profile.json'), 'op_type_counts': types}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-site', required=True)
    parser.add_argument('--transaction', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        collect(args)


if __name__ == '__main__':
    main()
