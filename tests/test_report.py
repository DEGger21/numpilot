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
        "total_rewrites": 5, "custom": {"relu": 1},
        "npu_ops": ["add", "matmul"], "cpu_ops": ["zeros"],
        "backend_version": "0.3.1", "backend_module": "/x/asnumpy/__init__.py",
        "numerically_consistent": True, "max_rel_diff": 1.49e-7, "tol": 1e-4,
        "t_cpu_ms": 1500.0, "t_npu_ms": 180.0, "speedup": 8.33, "n_elements": 640,
        "migrated_path": "out_wizard/migrated.py",
    }


def test_build_report_has_sections():
    md = build_report_md(_info())
    for h in ["# NumPilot 迁移报告", "算子替换", "自定义 NPU 算子",
              "执行证据", "数值校验", "整网性能对比", "加速比"]:
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
