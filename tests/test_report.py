# tests/test_report.py
"""迁移报告渲染与落盘。"""
import os

from numpilot.report import build_report_md, save_migration_report


def _info():
    return {
        "net": "mlp_forward", "source_path": "examples/mlp_forward.py",
        "engine": "ast", "backend": "npu", "status": "pass",
        "n_calls": 15, "n_ops": 10,
        "rewrites": {"matmul": 2, "relu": 1, "add": 2},
        "total_rewrites": 5, "extension_apis": {"relu": 1},
        "npu_ops": ["add", "matmul"], "cpu_ops": ["zeros"],
        "backend_version": "0.3.1", "backend_module": "/x/asnumpy/__init__.py",
        "numerically_consistent": True, "max_rel_diff": 1.49e-7, "tol": 1e-4,
        "t_cpu_ms": 1500.0, "t_npu_ms": 180.0, "speedup": 8.33, "n_elements": 640,
        "performance_valid": True, "cpu_threads": 1,
        "migrated_path": "out_wizard/migrated.py",
    }


def test_build_report_has_sections():
    md = build_report_md(_info())
    for h in ["# NumPilot 迁移报告", "算子替换", "已有 AsNumpy 扩展 API",
              "执行证据", "语义校验", "整网性能对比", "加速比"]:
        assert h in md
    assert "relu" in md and "8.3" in md


def test_build_report_unsupported():
    md = build_report_md({"net": "x", "status": "unsupported", "unsupported": ["fft"]})
    assert "缺口算子" in md and "fft" in md


def test_save_report_timestamped(tmp_path):
    path = save_migration_report(_info(), reports_dir=str(tmp_path))
    assert os.path.isfile(path)
    assert os.path.basename(path).startswith("migration_mlp_forward_")
    assert path.endswith(".md")
    assert "整网性能对比" in open(path, encoding="utf-8").read()


def test_report_suppresses_speedup_without_strict_provenance():
    info = _info()
    info["performance_valid"] = False
    info["performance_issues"] = ["matmul 使用 CPU fallback"]
    md = build_report_md(info)
    assert "未发布 NPU 加速比" in md
    assert "8.3300×" not in md
    assert "matmul 使用 CPU fallback" in md


def test_benchmark_report_marks_fallback_as_unqualified():
    from numpilot.report import build_benchmark_report_md

    result = {
        "net": "x", "file": "x.py", "status": "pass",
        "backend_version": "0.3.1", "allow_cpu_fallback": True,
        "fallback_override_used": True, "performance_valid": False,
        "speedup": None, "t_cpu_ms": 10.0, "t_npu_ms": 1.0,
        "artifact_inputs_executed": True,
        "timing_scope": {
            "description": "forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步",
            "kernel_only": False, "end_to_end": False, "includes_h2d": False,
            "includes_d2h_sync_each_iteration": True,
        },
        "provenance": {
            "ok": False, "required_ops": ["matmul"],
            "issues": ["已知非严格 NPU/CPU fallback: matmul"],
        },
        "comparison": {
            "consistent": True, "max_abs_diff": 0.0, "max_rel_diff": 0.0,
            "atol": 1e-5, "rtol": 1e-4,
        },
    }
    md = build_benchmark_report_md(result)
    assert "未发布 NPU 加速比" in md
    assert "10.0000×" not in md
    assert "CPU fallback" in md
    assert "自身 build_inputs 已执行:True" in md
    assert "kernel-only=False" in md
    assert "每轮含 D2H 同步=True" in md
