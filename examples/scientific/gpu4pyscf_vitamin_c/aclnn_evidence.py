#!/usr/bin/env python3
"""Corroborate device dispatch from the AsNumpy ACLNN log lines.

The AsNumpy wrapper prints one ``aclnn<Op> completed`` line per ACLNN launch.
This parser counts those lines and checks them against the logical dispatch
counters recorded in the write-once AsNumpy receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


PATTERNS = {
    'matmul': re.compile(r'\(Matmul\) aclnnMatmul completed'),
    'multiply': re.compile(r'\(Multiply\) aclnnMul completed'),
    'reducesum': re.compile(r'\(Sum\) aclnnReduceSum completed'),
    'flatten': re.compile(r'\(Sum\) aclnnFlatten completed'),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('refusing to overwrite %s' % args.out)

    receipt = json.loads(args.receipt.read_text())
    totals = receipt['dispatch']['totals']
    expected = {
        'matmul': int(totals['matmul']),
        'multiply': int(totals['multiply']),
        'reducesum': int(totals['sum']),
        'flatten': int(totals['d2h_scalar_internal']),
    }
    observed = {name: 0 for name in PATTERNS}
    other_lines = []
    for line in args.log.read_text(errors='replace').splitlines():
        matched = False
        for name, pattern in PATTERNS.items():
            if pattern.search(line):
                observed[name] += 1
                matched = True
                break
        if not matched and ('aclnn' in line or line.startswith('{')):
            other_lines.append(line)

    consistent = observed == expected
    report = {
        'schema': 'numpilot.vitamin-c-aclnn-dispatch-evidence.v1',
        'status': 'passed' if consistent else 'failed',
        'model_called': False,
        'log_path': str(args.log),
        'log_sha256': digest(args.log),
        'receipt_path': str(args.receipt),
        'receipt_sha256': digest(args.receipt),
        'expected_from_receipt': expected,
        'observed_log_lines': observed,
        'consistent': consistent,
        'non_operator_aclnn_or_json_lines': other_lines,
        'note': 'one ReduceSum line per logical sum call; Flatten precedes each axis=None scalar sum',
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'expected': expected,
                      'observed': observed}, ensure_ascii=False))
    if not consistent:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
