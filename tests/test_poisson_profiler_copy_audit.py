import json
import sqlite3

from tools.poisson_profiler_copy_audit import audit


def _runtime_db(path, copies):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE MemcpyInfo("
            "data_size INTEGER, memcpy_direction INTEGER)"
        )
        connection.executemany(
            "INSERT INTO MemcpyInfo VALUES (?, 0)",
            [(size,) for size in copies],
        )


def _task_db(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE TaskTrack(task_type TEXT, kernel_name TEXT)")
        connection.executemany(
            "INSERT INTO TaskTrack VALUES ('KERNEL_AIVEC', ?)",
            [("Add_hash",), ("Add_hash",), ("Mul_hash",)],
        )


def test_copy_audit_closes_source_count_kernel_and_numerical_gates(tmp_path):
    before_db = tmp_path / "before.db"
    after_db = tmp_path / "after.db"
    task_db = tmp_path / "tasks.db"
    # calls=3, h2d=2, d2h=1 => before=8, after=3, eliminated=5.
    _runtime_db(before_db, [4] * 8)
    _runtime_db(after_db, [4] * 3)
    _task_db(task_db)

    evidence = tmp_path / "iteration.json"
    evidence.write_text(json.dumps({
        "status": "pass",
        "strict_npu": True,
        "iterations": 1,
        "total_runtime_calls": 3,
        "runtime_call_counts": {"add": 2, "multiply": 1},
        "runtime_profile_ok": True,
        "runtime_target_integrity_ok": True,
        "transfers": {"h2d": 2, "d2h": 1},
        "comparison": {"consistent": True, "max_abs_diff": 0.0},
        "capability_audit_id": "audit",
    }), encoding="utf-8")
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({
        "status": "pass",
        "all_probe_calls_found_exactly_once": True,
        "direction_field_discriminates": False,
        "conclusion": "field is not sufficient",
    }), encoding="utf-8")
    before_utils = tmp_path / "before_utils.py"
    before_utils.write_text("super().__init__(shape_or_array)\n", encoding="utf-8")
    after_utils = tmp_path / "after_utils.py"
    after_utils.write_text(
        "super().__init__(shape_or_array, _move=True)\n", encoding="utf-8"
    )
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")

    result = audit(
        before_runtime_db=before_db,
        after_runtime_db=after_db,
        after_task_db=task_db,
        iteration_evidence=evidence,
        direction_probe_analysis=probe,
        before_utils=before_utils,
        after_utils=after_utils,
        wheel=wheel,
    )

    assert result["status"] == "pass"
    assert result["before"]["rows"] == 8
    assert result["after"]["rows"] == 3
    assert result["eliminated"]["copies"] == 5
    assert all(result["checks"].values())
