#!/usr/bin/env python3
"""One-core NumPy reference for the complete Vitamin C grid-density chain.

This script does not rerun SCF.  It reloads the SCF/DFT-grid inputs already
materialized by ``numpy_pipeline.py`` and measures the complete chain

    projected     = dm_t @ ao_t
    rho           = sum(ao_t * projected, axis=0, dtype=float32)
    weighted      = rho * weights
    electron_count= sum(weighted, dtype=float32)

on exactly one logical CPU core with one BLAS thread.  The electron count uses
an explicit multiply+sum (the form used by the AsNumpy side) and also records
the legacy ``dot`` value for continuity with the earlier baseline receipt.
"""
from __future__ import annotations

import argparse
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


def timed_scope(function, warmup, samples):
    for _ in range(warmup):
        function()
    values = []
    result = None
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(values)
    return {
        'samples_ms': values,
        'standard_median_ms': statistics.median(values),
        'upper_median_ms': ordered[len(ordered) // 2],
    }, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 3:
        raise ValueError('warmup/samples are too small')
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise RuntimeError('OPENBLAS_NUM_THREADS must be set before NumPy import')

    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) != 1:
        raise RuntimeError('run under taskset with exactly one logical CPU')

    baseline_json = json.loads((args.baseline / 'numpy_baseline.json').read_text())
    baseline_inputs = args.baseline / 'scf_grid_inputs.npz'
    if digest(baseline_inputs) != baseline_json['outputs']['inputs_sha256']:
        raise RuntimeError('baseline inputs hash does not match its receipt')
    inputs = np.load(baseline_inputs, allow_pickle=False)
    dm_t = np.ascontiguousarray(inputs['dm_t'])
    ao_t = np.ascontiguousarray(inputs['ao_t'])
    weights = np.ascontiguousarray(inputs['weights'])
    if dm_t.dtype != np.float32 or ao_t.dtype != np.float32 or weights.dtype != np.float32:
        raise RuntimeError('expected float32 SCF/grid inputs')
    assert tuple(dm_t.shape) == tuple(baseline_json['shapes']['dm_t'])
    assert tuple(ao_t.shape) == tuple(baseline_json['shapes']['ao_t'])
    assert tuple(weights.shape) == tuple(baseline_json['shapes']['weights'])

    established_projected = np.load(args.baseline / 'numpy_projected.npy', allow_pickle=False)
    established_rho = np.load(args.baseline / 'numpy_rho.npy', allow_pickle=False)
    established_electron_count = float(baseline_json['outputs']['electron_count_float32'])

    pools = threadpool_info()
    if not pools or any(p.get('num_threads') != 1 for p in pools):
        raise RuntimeError('loaded numerical threadpool is not limited to one')

    def matmul_only():
        return np.matmul(dm_t, ao_t)

    def through_rho():
        projected = np.matmul(dm_t, ao_t)
        rho = np.sum(np.multiply(ao_t, projected), axis=0, dtype=np.float32)
        return projected, rho

    def full_chain():
        projected = np.matmul(dm_t, ao_t)
        rho = np.sum(np.multiply(ao_t, projected), axis=0, dtype=np.float32)
        weighted = np.multiply(rho, weights)
        electron_count = np.sum(weighted, dtype=np.float32)
        return projected, rho, weighted, electron_count

    with threadpool_limits(limits=1):
        matmul_timing, _ = timed_scope(matmul_only, args.warmup, args.samples)
        rho_timing, _ = timed_scope(through_rho, args.warmup, args.samples)
        chain_timing, chain_result = timed_scope(full_chain, args.warmup, args.samples)
        projected, rho, weighted, electron_count = chain_result
        dot_electron_count = float(np.dot(rho, weights))

    # The full-chain CPU reference must reproduce the established baseline
    # bit-for-bit; otherwise the comparison identities are not comparable.
    projected_bitwise = bool(np.array_equal(projected, established_projected))
    rho_bitwise = bool(np.array_equal(rho, established_rho))
    if not projected_bitwise or not rho_bitwise:
        raise RuntimeError('full-chain NumPy reference drifted from the established baseline')

    np.save(args.out / 'numpy_fullchain_projected.npy', projected, allow_pickle=False)
    np.save(args.out / 'numpy_fullchain_rho.npy', rho, allow_pickle=False)
    np.save(args.out / 'numpy_fullchain_weighted.npy', weighted, allow_pickle=False)
    np.save(args.out / 'numpy_fullchain_electron_count.npy',
            np.asarray(electron_count, dtype=np.float32), allow_pickle=False)

    receipt = {
        'schema': 'numpilot.vitamin-c-real-numpy-fullchain.v1',
        'status': 'passed',
        'model_called': False,
        'scope': 'complete grid-density chain on saved real SCF/DFT-grid inputs; SCF is not rerun',
        'source_xyz_sha256': baseline_json['source_xyz_sha256'],
        'baseline_json_sha256': digest(args.baseline / 'numpy_baseline.json'),
        'baseline_inputs_sha256': digest(baseline_inputs),
        'baseline_projected_sha256': digest(args.baseline / 'numpy_projected.npy'),
        'baseline_rho_sha256': digest(args.baseline / 'numpy_rho.npy'),
        'molecule': baseline_json.get('molecule'),
        'atom_count': baseline_json.get('atom_count'),
        'method': baseline_json.get('method'),
        'shapes': {
            'dm_t': list(dm_t.shape),
            'ao_t': list(ao_t.shape),
            'weights': list(weights.shape),
            'projected': list(projected.shape),
            'rho': list(rho.shape),
            'weighted': list(weighted.shape),
        },
        'dtypes': {
            'dm_t': str(dm_t.dtype), 'ao_t': str(ao_t.dtype),
            'weights': str(weights.dtype), 'projected': str(projected.dtype),
            'rho': str(rho.dtype), 'weighted': str(weighted.dtype),
        },
        'chain': {
            'projected': 'np.matmul(dm_t, ao_t)',
            'rho': 'np.sum(np.multiply(ao_t, projected), axis=0, dtype=np.float32)',
            'weighted': 'np.multiply(rho, weights)',
            'electron_count': 'np.sum(weighted, dtype=np.float32)',
            'legacy_electron_count': 'np.dot(rho, weights)',
        },
        'timing': {
            'warmup': args.warmup,
            'samples': args.samples,
            'matmul_only': matmul_timing,
            'matmul_through_rho': rho_timing,
            'full_chain': chain_timing,
            'cpu_scope': 'taskset one logical CPU, OPENBLAS_NUM_THREADS=1, float32',
        },
        'outputs': {
            'electron_count_float32': float(electron_count),
            'electron_count_dot_float32': dot_electron_count,
            'electron_count_sum_minus_dot': float(electron_count) - dot_electron_count,
            'established_electron_count_float32': established_electron_count,
            'rho_min': float(np.min(rho)),
            'rho_max': float(np.max(rho)),
            'projected_sha256': digest(args.out / 'numpy_fullchain_projected.npy'),
            'rho_sha256': digest(args.out / 'numpy_fullchain_rho.npy'),
            'weighted_sha256': digest(args.out / 'numpy_fullchain_weighted.npy'),
            'electron_count_sha256': digest(args.out / 'numpy_fullchain_electron_count.npy'),
        },
        'comparison_to_established_baseline': {
            'projected_bitwise_equal': projected_bitwise,
            'rho_bitwise_equal': rho_bitwise,
            'electron_count_dot_delta_vs_established': dot_electron_count - established_electron_count,
        },
        'cpu': {
            'affinity': allowed,
            'blas_threads': 1,
            'threadpools': pools,
        },
        'boundary': {
            'actual_scf_and_grid_inputs': True,
            'scf_rerun_in_this_script': False,
            'full_scf_accelerated': False,
            'chemical_accuracy_claim': False,
        },
    }
    (args.out / 'numpy_fullchain_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    )
    print(json.dumps({
        'status': receipt['status'],
        'full_chain_median_ms': chain_timing['standard_median_ms'],
        'matmul_only_median_ms': matmul_timing['standard_median_ms'],
        'through_rho_median_ms': rho_timing['standard_median_ms'],
        'electron_count_float32': float(electron_count),
        'electron_count_dot_float32': dot_electron_count,
        'projected_bitwise_equal': projected_bitwise,
        'rho_bitwise_equal': rho_bitwise,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
