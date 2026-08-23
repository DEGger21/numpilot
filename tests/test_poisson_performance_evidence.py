import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np

from tools.poisson_performance_evidence import (
    _build_scaled_inputs,
    _load_profiler_audit,
    _profile_identity,
    _timing_samples,
)
from tools.poisson_performance_profiler_audit import audit


def test_scaled_inputs_match_original_size_contract_and_are_deterministic():
    first = _build_scaled_inputs(48, 48, 7)
    second = _build_scaled_inputs(48, 48, 7)

    assert len(first) == 30
    assert first[0].shape == (48 * 48,)
    assert first[3].shape == (4 * 48,)
    assert first[6].shape == (7 * 7,)
    assert first[15].shape == (48 * 48, 7 * 7)
    assert first[16].shape == (4 * 48, 7 * 7)
    assert all(item.dtype == np.float32 and item.flags.c_contiguous for item in first)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_timing_samples_keep_warmup_separate_and_reset_parameter_state():
    canonical = [np.array([np.float32(index)]) for index in range(30)]
    observed_parameter_starts = []
    synchronizations = []

    def forward(*inputs):
        observed_parameter_starts.append(float(inputs[6][0]))
        parameters = tuple(
            np.add(inputs[index], np.float32(1.0)) for index in range(6, 10)
        )
        return (np.array([np.float32(1.0)]), *parameters)

    timing = _timing_samples(
        forward,
        canonical,
        iterations=2,
        warmup=1,
        repeats=3,
        synchronize=lambda: synchronizations.append(True),
        completion_scope="test completion",
    )

    assert len(timing["warmup_samples_ms"]) == 1
    assert len(timing["samples_ms"]) == 3
    assert observed_parameter_starts == [6.0, 7.0] * 4
    assert len(synchronizations) == 4
    assert canonical[6][0] == np.float32(6.0)


def _npu_identity():
    return {
        "source_sha256": "s" * 64,
        "migrated_sha256": "m" * 64,
        "overlay_sha256": "o" * 64,
        "capability_audit_id": "audit-v1",
        "runner_sha256": "r" * 64,
        "scale": {
            "interior_axis": 48,
            "interior_points": 2304,
            "boundary_axis": 48,
            "boundary_points": 192,
            "centre_axis": 7,
            "rbf_count": 49,
        },
    }


def test_profiler_audit_is_required_and_must_match_identity(tmp_path):
    npu = _npu_identity()
    missing, issues = _load_profiler_audit(None, npu)
    assert missing is None
    assert issues == ["missing passing profiler audit for this scale"]

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({
        "status": "pass",
        "profile_identity": {**_profile_identity(npu), "overlay_sha256": "x" * 64},
    }))
    _, issues = _load_profiler_audit(str(audit_path), npu)
    assert issues == ["profiler audit identity or scale does not match benchmark"]


def test_profiler_audit_matches_kernel_and_boundary_transfer_accounting(tmp_path):
    runtime_db = tmp_path / "runtime.db"
    with sqlite3.connect(runtime_db) as connection:
        connection.execute(
            "CREATE TABLE MemcpyInfo (data_size INTEGER, memcpy_direction INTEGER)"
        )
        connection.executemany(
            "INSERT INTO MemcpyInfo VALUES (?, ?)", [(4, 0), (8, 0)]
        )

    task_db = tmp_path / "rts_track.db"
    with sqlite3.connect(task_db) as connection:
        connection.execute(
            "CREATE TABLE TaskTrack (task_type TEXT, kernel_name TEXT)"
        )
        connection.executemany(
            "INSERT INTO TaskTrack VALUES (?, ?)",
            [("KERNEL_AICORE", "Add_first"), ("KERNEL_AICORE", "Add_second")],
        )

    npu = {
        **_npu_identity(),
        "strict_npu": True,
        "runtime_profile_ok": True,
        "runtime_target_integrity_ok": True,
        "runtime_call_counts": {"add": 2},
        "total_runtime_calls": 2,
        "transfers": {
            "h2d_count": 1,
            "h2d_bytes": 4,
            "d2h_count": 1,
            "d2h_bytes": 8,
        },
    }
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({
        "status": "pass",
        "workload": {"iterations_per_timing_sample": 1},
        "npu": npu,
    }))

    receipt = audit(
        run_evidence=run_path, runtime_db=runtime_db, task_db=task_db
    )

    assert receipt["status"] == "pass"
    assert all(receipt["checks"].values())
    assert receipt["profile_identity"] == _profile_identity(npu)
    assert receipt["memcpy"]["bytes"] == 12
