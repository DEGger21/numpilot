import json
from pathlib import Path

from tools.render_migration_report import build_run_report


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_run_report_includes_scale_and_resident_speedup(tmp_path):
    run = tmp_path / "run"
    _write(run / "qualified-analysis.json", {
        "source_path": "demo.py", "source_sha256": "a" * 64,
        "decision": "DIRECT", "calls": [{"source_api": "numpy.sin"}],
        "qualified_apis": ["sin"], "runtime_identity_status": "matched",
        "runtime_audit_id": "audit-1",
    })
    _write(run / "migration" / "script-migration-report.json", {
        "status": "passed", "configured_model": "demo-model",
        "migrated_sha256": "b" * 64, "fallback_accepted": False,
        "analysis": {"counts": {"strict_npu": 1}, "scan": {"rewrite_count": 1}},
    })
    _write(run / "script-verification-report.json", {
        "status": "passed", "strict_npu_verified": True,
        "numerically_consistent": True,
        "comparisons": {"result": {"consistent": True, "max_abs_diff": 1e-6, "max_rel_diff": 2e-6}},
    })
    _write(run / "script-benchmark-report.json", {
        "status": "passed",
        "performance": {"performance_valid": True, "cpu_median_ms": 100.0,
                         "npu_median_ms": 40.0, "median_speedup": 2.5,
                         "conservative_cpu_p10_over_npu_p90": 2.1},
        "workers": {"npu": {"outputs": [{"elements": 1_000_000,
                                             "shape": [1000, 1000], "dtype": "<f4",
                                             "data_bytes": 4_000_000}],
                              "affinity": {"requested_core": 0},
                              "timing": {"warmup_ns": [1, 2], "samples_ns": [3] * 7},
                              "runtime_identity_after": {"asnumpy_version": "0.3.1", "asnumpy_commit": "abc", "cann_version": "9.0", "soc_model": "910B", "status": "matched"}}},
    })
    _write(run / "workflow-report.json", {"status": "passed", "stages": [{"name": "04-benchmark", "status": "passed", "elapsed_seconds": 3.2}]})

    report = build_run_report(run)
    assert "1,000,000 个元素" in report
    assert "dtype=`<f4`" in report
    assert "2.5×" in report
    assert "2.1×" in report
    assert "warmup=2" in report
    assert "samples=7" in report
    assert "core=0" in report
    assert "迁移后的 AsNumpy 程序" in report
    assert "steady-state device-resident" in report
    assert "AsNumpy" in report


def test_build_run_report_handles_missing_optional_receipts(tmp_path):
    run = tmp_path / "run"
    _write(run / "qualified-analysis.json", {"decision": "BLOCKED", "calls": []})
    report = build_run_report(run)
    assert "未执行 benchmark" in report
    assert "未找到 script-migration-report.json" in report
    assert "未发布 NPU 加速比" not in report


def test_build_run_report_distinguishes_user_timing_from_stage_sum(tmp_path):
    run = tmp_path / "run"
    _write(run / "qualified-analysis.json", {"decision": "DIRECT", "calls": []})
    _write(run / "workflow-report.json", {
        "status": "passed",
        "stages": [
            {"name": "01-analyze", "status": "passed", "elapsed_seconds": 1.25},
            {"name": "02-migrate", "status": "passed", "elapsed_seconds": 2.75},
        ],
    })
    _write(tmp_path / "timing.json", {
        "started_at_utc": "2026-09-03T10:00:00Z",
        "finished_at_utc": "2026-09-03T10:00:12Z",
    })
    report = build_run_report(run)
    assert "用户记录的终端主链墙钟耗时: **12.000 s**" in report
    assert "框架阶段耗时之和: **4.000 s**" in report
    assert "不等同于用户终端主链墙钟时间" in report


def test_build_run_report_accepts_elapsed_ms_timing(tmp_path):
    run = tmp_path / "run"
    _write(run / "qualified-analysis.json", {"decision": "DIRECT", "calls": []})
    _write(tmp_path / "timing.json", {"elapsed_ms": 1250})
    report = build_run_report(run)
    assert "用户记录的终端主链墙钟耗时: **1.250 s**" in report


def test_build_run_report_does_not_claim_advantage_below_threshold(tmp_path):
    run = tmp_path / "run"
    _write(run / "qualified-analysis.json", {"decision": "DIRECT", "calls": []})
    _write(run / "script-benchmark-report.json", {
        "status": "passed",
        "performance": {
            "performance_valid": True,
            "cpu_median_ms": 10,
            "npu_median_ms": 12,
            "median_speedup": 0.83,
        },
        "workers": {
            "npu": {"outputs": [{"elements": 4, "shape": [4], "dtype": "<f4", "data_bytes": 16}]}
        },
    })
    report = build_run_report(run)
    assert "不发布 NPU 加速声明" in report
    assert "resident 加速比:" not in report
