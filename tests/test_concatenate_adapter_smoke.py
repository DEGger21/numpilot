"""CPU-only checks for the direct adapter experiment boundary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import concatenate_adapter_smoke as smoke


SOURCE = Path(__file__).parents[1] / smoke.SOURCE_RELATIVE


def test_i006_rebinding_preserves_source_builder_and_numpy_forward():
    before = SOURCE.read_bytes()
    builder, forward = smoke._load_case(SOURCE)
    calls = []

    def cat(arrays, axis):
        calls.append(axis)
        return np.concatenate(arrays, axis=axis)

    rebound = smoke._rebind_forward(forward, SimpleNamespace(concatenate=cat))
    inputs = builder()
    expected = forward(**inputs)['out']
    actual = rebound(**inputs)['out']
    assert rebound.__code__ is forward.__code__
    assert forward.__globals__['np'] is np
    assert calls == [1]
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (3, 6, 5)
    assert actual.dtype == np.float32
    assert SOURCE.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == smoke.SOURCE_SHA256


def test_changed_i006_source_is_rejected_before_execution(tmp_path):
    source = tmp_path / 'program.py'
    marker = tmp_path / 'executed'
    source.write_text(f'open({str(marker)!r}, "w").close()\n')
    with pytest.raises(ValueError, match='source hash changed'):
        smoke._load_case(source)
    assert not marker.exists()


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity') or not shutil.which('taskset'),
                    reason='single-core subprocess requires Linux taskset')
def test_cpu_worker_is_single_core_without_npu_import(tmp_path):
    core = min(os.sched_getaffinity(0))
    receipt = tmp_path / 'cpu.json'
    env = os.environ.copy()
    env.update({name: '1' for name in smoke.THREAD_VARS})
    env.pop('PYTHONPATH', None)
    command = ['taskset', '--cpu-list', str(core), sys.executable, str(Path(smoke.__file__)),
               '--worker', 'cpu', '--runtime-site', '/does/not/exist',
               '--source', str(SOURCE), '--out', str(receipt), '--sizes', 'base',
               '--repeats', '3', '--warmup', '1', '--cpu-core', str(core)]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads(receipt.read_text())
    assert data['status'] == 'passed'
    assert data['affinity_before'] == data['affinity_after'] == [core]
    assert data['cases']['base']['correctness']['exact_equal'] is True
    assert len(data['cases']['base']['cpu']['samples_ms']) == 3
    assert 'runtime_before' not in data
