#!/usr/bin/env python3
"""Complete Vitamin C grid-density chain on Ascend NPU via AsNumpy.

The migrated chain keeps the scientific formula unchanged:

    projected      = dm_t @ ao_t
    rho            = sum(ao_t * projected, axis=0, dtype=float32)
    weighted       = rho * weights
    electron_count = sum(weighted, dtype=float32)

``matmul``, ``multiply`` and ``sum`` are dispatched through the audited
AsNumpy runtime; ``dot`` is deliberately not used because its capability entry
is a conditional CPU fallback.  Every device call is pre-validated by the
NumPilot ``RuntimeProfileGuard`` bound to this exact runtime snapshot.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--cpu-fullchain', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 3:
        raise ValueError('warmup/samples are too small')
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.environ.get('ASCEND_VISIBLE_DEVICES') != '6':
        raise RuntimeError('expected physical NPU 6')
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise RuntimeError('OPENBLAS_NUM_THREADS must be 1')
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('run under taskset with one logical CPU')

    import numpy as np
    import asnumpy as ap
    from numpilot.backends.npu import NpuBackend

    baseline_json = json.loads((args.baseline / 'numpy_baseline.json').read_text())
    cpu_json = json.loads((args.cpu_fullchain / 'numpy_fullchain_receipt.json').read_text())
    baseline_inputs = args.baseline / 'scf_grid_inputs.npz'
    if digest(baseline_inputs) != baseline_json['outputs']['inputs_sha256']:
        raise RuntimeError('baseline inputs hash does not match its receipt')
    if digest(baseline_inputs) != cpu_json['baseline_inputs_sha256']:
        raise RuntimeError('CPU full-chain reference used different inputs')
    inputs = np.load(baseline_inputs, allow_pickle=False)
    dm_t = np.ascontiguousarray(inputs['dm_t'])
    ao_t = np.ascontiguousarray(inputs['ao_t'])
    weights = np.ascontiguousarray(inputs['weights'])
    assert dm_t.dtype == ao_t.dtype == weights.dtype == np.float32
    assert dm_t.flags.c_contiguous and ao_t.flags.c_contiguous and weights.flags.c_contiguous

    expected_projected = np.load(args.cpu_fullchain / 'numpy_fullchain_projected.npy',
                                 allow_pickle=False)
    expected_rho = np.load(args.cpu_fullchain / 'numpy_fullchain_rho.npy',
                           allow_pickle=False)
    expected_weighted = np.load(args.cpu_fullchain / 'numpy_fullchain_weighted.npy',
                                allow_pickle=False)
    expected_electron_count = float(cpu_json['outputs']['electron_count_float32'])
    for name, value in (('dm_t', dm_t), ('ao_t', ao_t), ('weights', weights)):
        assert tuple(value.shape) == tuple(cpu_json['shapes'][name])
    assert tuple(expected_projected.shape) == tuple(cpu_json['shapes']['projected'])
    assert tuple(expected_rho.shape) == tuple(cpu_json['shapes']['rho'])
    assert tuple(expected_weighted.shape) == tuple(cpu_json['shapes']['weighted'])

    backend = NpuBackend()
    identity = backend.runtime_identity().to_dict()
    if identity.get('status') != 'matched':
        raise RuntimeError('AsNumpy runtime identity is not matched')
    guard = backend.prepare_profile_guard(['matmul', 'multiply', 'sum'])

    acl = ctypes.CDLL('libascendcl.so')
    get_count = acl.aclrtGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    get_count.restype = ctypes.c_int
    visible = ctypes.c_uint32()
    if get_count(ctypes.byref(visible)) != 0 or visible.value != 1:
        raise RuntimeError('expected one visible NPU')
    synchronize = acl.aclrtSynchronizeDevice
    synchronize.argtypes = []
    synchronize.restype = ctypes.c_int

    counters = {}

    def counts(scope):
        return counters.setdefault(scope, {
            'h2d': 0, 'd2h': 0, 'd2h_scalar_internal': 0, 'sync': 0,
            'matmul': 0, 'multiply': 0, 'sum': 0,
        })

    def sync(scope):
        if synchronize() != 0:
            raise RuntimeError('aclrtSynchronizeDevice failed')
        counts(scope)['sync'] += 1

    def upload(value, scope):
        counts(scope)['h2d'] += 1
        return ap.ndarray.from_numpy(value)

    def matmul(lhs, rhs, scope):
        guard.validate_call('matmul', (lhs, rhs), {})
        counts(scope)['matmul'] += 1
        return ap.matmul(lhs, rhs)

    def multiply(x1, x2, scope):
        guard.validate_call('multiply', (x1, x2), {})
        counts(scope)['multiply'] += 1
        return ap.multiply(x1, x2)

    def sum_axis0(value, scope):
        guard.validate_call('sum', (value,), {'axis': 0, 'dtype': np.float32})
        counts(scope)['sum'] += 1
        return ap.sum(value, axis=0, dtype=np.float32)

    def sum_scalar(value, scope):
        guard.validate_call('sum', (value,), {'dtype': np.float32})
        counts(scope)['sum'] += 1
        # ap.sum(axis=None) returns a host Python float; the library performs
        # the scalar device-to-host copy internally.
        counts(scope)['d2h_scalar_internal'] += 1
        return float(ap.sum(value, dtype=np.float32))

    def download(value, scope):
        counts(scope)['d2h'] += 1
        return value.to_numpy()

    def run_matmul_only(lhs, rhs, scope):
        projected = matmul(lhs, rhs, scope)
        host = download(projected, scope)
        sync(scope)
        return host

    def run_through_rho(lhs, rhs, scope):
        projected = matmul(lhs, rhs, scope)
        product = multiply(rhs, projected, scope)
        rho = sum_axis0(product, scope)
        projected_host = download(projected, scope)
        rho_host = download(rho, scope)
        sync(scope)
        return projected_host, rho_host

    def run_full_chain(lhs, rhs, weight, scope):
        projected = matmul(lhs, rhs, scope)
        product = multiply(rhs, projected, scope)
        rho = sum_axis0(product, scope)
        weighted = multiply(rho, weight, scope)
        electron_count = sum_scalar(weighted, scope)
        projected_host = download(projected, scope)
        rho_host = download(rho, scope)
        sync(scope)
        return projected_host, rho_host, electron_count

    def measure(scope, function):
        for _ in range(args.warmup):
            function()
        samples = []
        result = None
        for _ in range(args.samples):
            sync(scope)
            start = time.perf_counter_ns()
            result = function()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        ordered = sorted(samples)
        return {
            'samples_ms': samples,
            'standard_median_ms': statistics.median(samples),
            'upper_median_ms': ordered[len(ordered) // 2],
        }, result

    # Resident-input scopes: the three inputs are uploaded once before timing.
    sync('setup')
    resident_dm = upload(dm_t, 'setup')
    resident_ao = upload(ao_t, 'setup')
    resident_weights = upload(weights, 'setup')
    sync('setup')

    matmul_timing, matmul_host = measure(
        'resident_matmul_only',
        lambda: run_matmul_only(resident_dm, resident_ao, 'resident_matmul_only'),
    )
    rho_timing, rho_result = measure(
        'resident_through_rho',
        lambda: run_through_rho(resident_dm, resident_ao, 'resident_through_rho'),
    )
    resident_timing, resident_result = measure(
        'resident_full_chain',
        lambda: run_full_chain(resident_dm, resident_ao, resident_weights,
                               'resident_full_chain'),
    )

    # Warm end-to-end scope: uploads, all device operators, downloads, sync.
    def warm_iteration():
        scope = 'warm_e2e_full_chain'
        lhs = upload(dm_t, scope)
        rhs = upload(ao_t, scope)
        weight = upload(weights, scope)
        projected = matmul(lhs, rhs, scope)
        product = multiply(rhs, projected, scope)
        rho = sum_axis0(product, scope)
        weighted = multiply(rho, weight, scope)
        electron_count = sum_scalar(weighted, scope)
        projected_host = download(projected, scope)
        rho_host = download(rho, scope)
        sync(scope)
        return projected_host, rho_host, electron_count

    e2e_timing, e2e_result = measure('warm_e2e_full_chain', warm_iteration)

    resident_projected, resident_rho, resident_electron = resident_result
    e2e_projected, e2e_rho, e2e_electron = e2e_result
    rho_projected, rho_value = rho_result

    def compare(actual, expected, atol, rtol):
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        return {
            'consistent': bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            'atol': atol,
            'rtol': rtol,
            'max_abs': float(np.max(np.abs(delta))),
            'relative_frobenius': float(np.linalg.norm(delta) / np.linalg.norm(expected)),
            'dtype': str(actual.dtype),
            'shape': list(actual.shape),
        }

    projected_comparison = compare(resident_projected, expected_projected, 1e-5, 1e-4)
    rho_comparison = compare(resident_rho, expected_rho, 1e-4, 1e-4)
    e2e_projected_comparison = compare(e2e_projected, expected_projected, 1e-5, 1e-4)
    e2e_rho_comparison = compare(e2e_rho, expected_rho, 1e-4, 1e-4)
    electron_delta = float(resident_electron) - expected_electron_count
    e2e_electron_delta = float(e2e_electron) - expected_electron_count
    if not projected_comparison['consistent'] or not rho_comparison['consistent']:
        raise RuntimeError('resident NPU chain does not match the NumPy reference')
    if not e2e_projected_comparison['consistent'] or not e2e_rho_comparison['consistent']:
        raise RuntimeError('warm E2E NPU chain does not match the NumPy reference')
    if abs(electron_delta) > 1e-3 or abs(e2e_electron_delta) > 1e-3:
        raise RuntimeError('NPU electron count exceeds the 1e-3 acceptance bound')
    if not np.isfinite(resident_projected).all() or not np.isfinite(resident_rho).all():
        raise RuntimeError('NPU output contains non-finite values')

    np.save(args.out / 'asnumpy_fullchain_projected.npy', resident_projected,
            allow_pickle=False)
    np.save(args.out / 'asnumpy_fullchain_rho.npy', resident_rho, allow_pickle=False)
    np.save(args.out / 'asnumpy_fullchain_electron_count.npy',
            np.asarray(resident_electron, dtype=np.float32), allow_pickle=False)

    cpu_timing = cpu_json['timing']
    cp = cpu_timing['full_chain']['standard_median_ms']
    cp_mm = cpu_timing['matmul_only']['standard_median_ms']
    cp_rho = cpu_timing['matmul_through_rho']['standard_median_ms']
    totals = {}
    for scope, values in counters.items():
        for key, value in values.items():
            totals[key] = totals.get(key, 0) + value

    receipt = {
        'schema': 'numpilot.vitamin-c-real-asnumpy-fullchain.v1',
        'status': 'passed',
        'model_called': False,
        'scope': 'complete grid-density chain on saved real SCF/DFT-grid inputs; SCF is not rerun',
        'baseline_json_sha256': digest(args.baseline / 'numpy_baseline.json'),
        'baseline_inputs_sha256': digest(baseline_inputs),
        'cpu_fullchain_json_sha256': digest(args.cpu_fullchain / 'numpy_fullchain_receipt.json'),
        'source_xyz_sha256': baseline_json['source_xyz_sha256'],
        'runtime': identity,
        'environment': {
            'visible_device_environment': os.environ.get('ASCEND_VISIBLE_DEVICES'),
            'device_count': visible.value,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
        },
        'shapes': {
            'dm_t': list(dm_t.shape), 'ao_t': list(ao_t.shape),
            'weights': list(weights.shape), 'projected': list(resident_projected.shape),
            'rho': list(resident_rho.shape),
        },
        'dtypes': {
            'dm_t': str(dm_t.dtype), 'ao_t': str(ao_t.dtype),
            'weights': str(weights.dtype), 'projected': str(resident_projected.dtype),
            'rho': str(resident_rho.dtype), 'electron_count': 'float32',
        },
        'chain': {
            'projected': 'ap.matmul(dm_device, ao_device)',
            'rho': 'ap.sum(ap.multiply(ao_device, projected), axis=0, dtype=float32)',
            'weighted': 'ap.multiply(rho, weights_device)',
            'electron_count': 'ap.sum(weighted, dtype=float32)',
            'dot_used_on_device': False,
            'dot_reason': 'conditional CPU fallback per the audited capability catalog',
        },
        'guard': guard.to_dict(),
        'dispatch': {
            'per_scope': counters,
            'totals': totals,
            'notes': [
                'd2h_scalar_internal counts the host float returned by ap.sum(axis=None); the library performs that copy internally',
                'setup counts the one-time upload of the three resident inputs',
            ],
        },
        'comparison': {
            'resident_projected': projected_comparison,
            'resident_rho': rho_comparison,
            'warm_e2e_projected': e2e_projected_comparison,
            'warm_e2e_rho': e2e_rho_comparison,
            'electron_count': {
                'npu_resident_float32': float(resident_electron),
                'npu_warm_e2e_float32': float(e2e_electron),
                'cpu_float32': expected_electron_count,
                'delta_resident': electron_delta,
                'delta_warm_e2e': e2e_electron_delta,
                'acceptance_abs': 1e-3,
            },
        },
        'timing': {
            'warmup': args.warmup,
            'samples': args.samples,
            'cpu': {
                'matmul_only_standard_median_ms': cp_mm,
                'matmul_through_rho_standard_median_ms': cp_rho,
                'full_chain_standard_median_ms': cp,
                'scope': cpu_timing['cpu_scope'],
            },
            'npu': {
                'resident_matmul_only': matmul_timing,
                'resident_matmul_through_rho': rho_timing,
                'resident_full_chain': resident_timing,
                'warm_e2e_full_chain': e2e_timing,
                'resident_scope': 'inputs resident; device chain + explicit D2H + sync',
                'e2e_scope': 'H2D of dm/ao/weights + device chain + D2H + sync; SCF/grid construction excluded',
            },
            'speedups': {
                'resident_matmul_only': cp_mm / matmul_timing['standard_median_ms'],
                'resident_matmul_through_rho': cp_rho / rho_timing['standard_median_ms'],
                'resident_full_chain': cp / resident_timing['standard_median_ms'],
                'warm_e2e_full_chain': cp / e2e_timing['standard_median_ms'],
            },
        },
        'outputs': {
            'projected_sha256': digest(args.out / 'asnumpy_fullchain_projected.npy'),
            'rho_sha256': digest(args.out / 'asnumpy_fullchain_rho.npy'),
            'electron_count_sha256': digest(args.out / 'asnumpy_fullchain_electron_count.npy'),
        },
        'boundary': {
            'actual_scf_and_grid_inputs': True,
            'scf_rerun_in_this_script': False,
            'full_scf_accelerated': False,
            'chemical_accuracy_claim': False,
            'accelerated_stage': 'DM/AO grid contraction, elementwise product, axis-0 reduction, weighted reduction',
            'kernel_input_float32_cast': True,
        },
    }
    (args.out / 'asnumpy_fullchain_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    )
    print(json.dumps({
        'status': receipt['status'],
        'comparison': {
            'projected_max_abs': projected_comparison['max_abs'],
            'rho_max_abs': rho_comparison['max_abs'],
            'electron_count': float(resident_electron),
            'electron_delta': electron_delta,
        },
        'resident_full_chain_median_ms': resident_timing['standard_median_ms'],
        'resident_full_chain_speedup': receipt['timing']['speedups']['resident_full_chain'],
        'warm_e2e_median_ms': e2e_timing['standard_median_ms'],
        'warm_e2e_speedup': receipt['timing']['speedups']['warm_e2e_full_chain'],
        'dispatch_totals': totals,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
