import json
import os
from pathlib import Path
import stat
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "run_demo.sh"


def test_competition_demo_is_executable_and_valid_bash():
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_competition_demo_rejects_unknown_arguments():
    result = subprocess.run(
        [str(SCRIPT), "--unknown"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "未知参数" in result.stderr


def test_competition_demo_runs_one_ordinary_glm_chain(monkeypatch, tmp_path):
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n"
        "if [ \"${1:-}\" = \"tools/render_demo_summary.py\" ]; then\n"
        "  exec \"$NUMPILOT_TEST_REAL_PY\" \"$@\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NUMPILOT_TEST_REAL_PY": sys.executable,
        "NUMPILOT_DEMO_OUT": str(tmp_path / "run"),
    })

    result = subprocess.run(
        [str(SCRIPT), "--quick", "--core", "2"],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "-m numpilot configure-llm --check" in calls
    assert "-m numpilot doctor --backend npu" in calls
    assert "-m numpilot analyze examples/poisson_source_smooth_l1_map.py --require-direct" in calls
    assert "--mode script --engine llm --backend npu --fallback-policy reject" in calls
    assert "-m numpilot verify-script" in calls
    assert "-m numpilot benchmark-script" in calls
    assert "--warmup 2 --samples 7 --core 2" in calls
    assert "bench-cpu" not in calls
    assert "bench-npu" not in calls
    assert "--engine ast" not in calls
    assert "--mode contract" not in calls
    assert "tools/render_demo_summary.py --analysis" in calls
    assert "一屏回执摘要" in result.stdout
    assert "UNAVAILABLE" in result.stdout


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_demo_summary_labels_resident_and_component_scopes_truthfully(tmp_path):
    analysis = _write_json(tmp_path / "analysis.json", {
        "decision": "DIRECT",
        "runtime_identity_status": "matched",
        "runtime_identity_sha256": "a" * 64,
        "runtime_audit_id": "asnumpy-test-cann-test",
        "calls": [
            {"decision": "DIRECT", "performance_eligible": True},
            {"decision": "DIRECT", "performance_eligible": True},
        ],
    })
    migration = _write_json(tmp_path / "migration.json", {
        "actual_remote_call": True,
        "ast_audit": "passed",
        "strict_npu": True,
        "source_sha256": "b" * 64,
        "migrated_sha256": "c" * 64,
        "llm_receipt": {
            "http_status": 200,
            "tls_verification": "enabled",
            "response_model_identity": "exact",
            "response_model": "glm-test",
        },
    })
    verification = _write_json(tmp_path / "verification.json", {
        "status": "passed",
        "numerically_consistent": True,
        "comparisons": {"result": {
            "consistent": True,
            "max_abs_diff": 1e-6,
            "max_rel_diff": 2e-6,
            "n_elements": 100,
        }},
    })
    benchmark = _write_json(tmp_path / "benchmark.json", {
        "status": "passed",
        "gates": {f"gate_{index}": True for index in range(16)},
        "protocol": {"samples": 20, "core": 0, "cpu_threads_requested": 1},
        "performance": {
            "performance_valid": True,
            "cpu_median_ms": 10.0,
            "npu_median_ms": 4.0,
            "median_speedup": 2.5,
            "conservative_cpu_p10_over_npu_p90": 2.25,
            "median_speedup_bootstrap_95ci": [2.4, 2.6],
            "warm_runtime_single_pass_component_sum": {
                "cpu_ms": 12.0,
                "npu_ms": 15.0,
                "diagnostic_cpu_over_npu": 0.8,
                "directly_sampled_end_to_end": False,
            },
        },
    })

    result = subprocess.run(
        [
            sys.executable,
            "tools/render_demo_summary.py",
            "--analysis", str(analysis),
            "--migration", str(migration),
            "--verification", str(verification),
            "--benchmark", str(benchmark),
        ],
        cwd=str(SCRIPT.parent),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[真实 GLM] PASS · model=glm-test" in result.stdout
    assert "[strict-NPU 门禁] PASS · 16/16" in result.stdout
    assert "[设备驻留 headline] PASS" in result.stdout
    assert "median=2.500×" in result.stdout
    assert "bootstrap95%=[2.400, 2.600]" in result.stdout
    assert "[一次组件和（诊断；非直接 E2E）]" in result.stdout
    assert "CPU/NPU=0.800×" in result.stdout
    assert "[证据路径]" in result.stdout
    assert "[回执 SHA256 前 12 位]" in result.stdout
