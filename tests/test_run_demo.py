import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from tools.render_demo_summary import (
    MAX_RECEIPT_BYTES,
    _load_object,
    render_summary,
    render_summary_tty,
)


def test_summary_numeric_formatters_fail_closed_on_overflow():
    huge = 10 ** 10000
    from tools import render_demo_summary as summary

    assert summary._number(huge) == "UNAVAILABLE"
    assert summary._scientific(huge) == "UNAVAILABLE"
    assert summary._display_text(huge) == "UNAVAILABLE"


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


def test_competition_demo_requires_llm_config_path():
    result = subprocess.run(
        [str(SCRIPT), "--llm-config"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--llm-config" in result.stderr


def test_competition_demo_rejects_explicit_empty_output_path():
    result = subprocess.run(
        [str(SCRIPT), "--out", ""],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--out 目录不能为空" in result.stderr


def test_competition_demo_rejects_reusing_existing_run_directory(tmp_path):
    run_root = tmp_path / "already-used"
    run_root.mkdir()
    (run_root / "workflow-report.json").write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_DEMO_OUT": str(run_root),
    })

    result = subprocess.run(
        [str(SCRIPT), "--quick"],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "全新的路径" in result.stderr
    assert (run_root / "workflow-report.json").read_text(encoding="utf-8") == "{}\n"


def test_competition_demo_rejects_symlinked_output_ancestor(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    result = subprocess.run(
        [str(SCRIPT), "--out", str(linked / "new-run"), "--quick"],
        cwd=str(SCRIPT.parent),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "符号链接" in result.stderr
    assert not (outside / "new-run").exists()


def test_competition_demo_rejects_dotdot_symlinked_output_ancestor(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    # Keep the lexical ``..``: normalizing first would hide the symlink and
    # allow the shell's subsequent mkdir to operate in the wrong tree.
    result = subprocess.run(
        [str(SCRIPT), "--out", str(linked / ".." / "new-run"), "--quick"],
        cwd=str(SCRIPT.parent),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "符号链接" in result.stderr
    assert not (outside / "new-run").exists()


def test_competition_demo_resolves_relative_output_from_invocation_dir(tmp_path):
    fake_python = tmp_path / "python"
    log = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NO_COLOR": "1",
    })

    result = subprocess.run(
        [str(SCRIPT), "--out", "relative-run", "--quick"],
        cwd=str(tmp_path),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert str(tmp_path / "relative-run") in result.stdout
    assert (tmp_path / "relative-run").is_dir()


def test_competition_demo_accepts_equals_form_for_llm_config(tmp_path):
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"${NUMPILOT_LLM_CONFIG_FILE-}\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    config_path = tmp_path / "runtime" / "llm.json"
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NUMPILOT_DEMO_OUT": str(tmp_path / "run"),
    })

    result = subprocess.run(
        [str(SCRIPT), "--llm-config=%s" % config_path, "--quick"],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines()
    assert all(
        line == str(config_path)
        for line in log.read_text(encoding="utf-8").splitlines()
    )


def test_competition_demo_exports_explicit_llm_config_path(monkeypatch, tmp_path):
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'config=%s call=%s\\n' \"${NUMPILOT_LLM_CONFIG_FILE-}\" \"$*\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    config_path = tmp_path / "private" / "llm.json"
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NUMPILOT_DEMO_OUT": str(tmp_path / "run"),
        # An inherited value must be replaceable by the explicit flag.
        "NUMPILOT_LLM_CONFIG_FILE": str(tmp_path / "inherited.json"),
    })

    result = subprocess.run(
        [str(SCRIPT), "--llm-config", str(config_path), "--quick"],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert calls
    assert all(
        line.startswith("config=%s " % config_path)
        for line in calls.splitlines()
    )


def test_competition_demo_resolves_relative_llm_config_from_invocation_dir(
    tmp_path,
):
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"${NUMPILOT_LLM_CONFIG_FILE-}\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    relative_path = Path("private") / "llm.json"
    expected_path = tmp_path / relative_path
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NUMPILOT_DEMO_OUT": str(tmp_path / "run"),
    })

    result = subprocess.run(
        [str(SCRIPT), "--llm-config", str(relative_path), "--quick"],
        cwd=str(tmp_path),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines()
    assert all(
        line == str(expected_path)
        for line in log.read_text(encoding="utf-8").splitlines()
    )


def test_competition_demo_resolves_explicit_source_from_invocation_dir(tmp_path):
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    source = tmp_path / "inputs" / "ordinary.py"
    source.parent.mkdir()
    source.write_text("result = 1\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NUMPILOT_DEMO_OUT": str(tmp_path / "run"),
        "NO_COLOR": "1",
    })

    result = subprocess.run(
        [str(SCRIPT), "inputs/ordinary.py", "--quick"],
        cwd=str(tmp_path),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    # Every stage must receive the absolute source resolved from the caller's
    # working directory.  A substring-only negative assertion would also
    # match the ``inputs/ordinary.py`` suffix inside that absolute path.
    assert f"analyze {source} --require-direct" in calls
    assert f"migrate {source} --mode script" in calls
    assert f"verify-script {source} --migration-report" in calls
    assert f"benchmark-script {source} --migration-report" in calls


def test_competition_demo_expands_leading_tilde_paths_without_eval(tmp_path):
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'config=%s args=%s\\n' \"${NUMPILOT_LLM_CONFIG_FILE-}\" \"$*\" >> \"$NUMPILOT_TEST_CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    environment = os.environ.copy()
    environment.update({
        "HOME": str(tmp_path),
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_TEST_CALL_LOG": str(log),
        "NO_COLOR": "1",
    })

    result = subprocess.run(
        [
            str(SCRIPT),
            "--out",
            "~/tilde-run",
            "--llm-config",
            "~/private/llm.json",
            "--quick",
        ],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert all(
        line.startswith("config=%s " % (tmp_path / "private/llm.json"))
        for line in calls.splitlines()
    )
    assert str(tmp_path / "tilde-run") in result.stdout
    assert not (Path(SCRIPT.parent) / "~").exists()


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
    assert "\x1b" not in result.stdout + result.stderr


def test_competition_demo_failure_reports_stage_and_preserves_run_path(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        "  *' configure-llm --check '*) exit 0 ;;\n"
        "  *' doctor --backend npu '*) exit 7 ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    run_root = tmp_path / "failed-run"
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_DEMO_OUT": str(run_root),
        "NO_COLOR": "1",
    })

    result = subprocess.run(
        [str(SCRIPT), "--quick"],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert "FAIL" in result.stderr
    assert "stage=1/5 环境与真实 NPU runtime 自检" in result.stderr
    assert "exit=7" in result.stderr
    assert str(run_root) in result.stderr
    assert run_root.is_dir()
    assert stat.S_IMODE(run_root.stat().st_mode) == 0o700
    assert "\x1b" not in result.stdout + result.stderr


def test_competition_demo_config_failure_keeps_exit_code_and_run_path(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        "  *' configure-llm --check '*) exit 4 ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    run_root = tmp_path / "config-failed-run"
    environment = os.environ.copy()
    environment.update({
        "NUMPILOT_PY": str(fake_python),
        "NUMPILOT_DEMO_OUT": str(run_root),
        "NO_COLOR": "1",
    })

    result = subprocess.run(
        [str(SCRIPT), "--quick"],
        cwd=str(SCRIPT.parent),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    assert "FAIL" in result.stderr
    assert "stage=configured model 本地检查" in result.stderr
    assert "exit=4" in result.stderr
    assert str(run_root) in result.stderr
    assert run_root.is_dir()
    assert stat.S_IMODE(run_root.stat().st_mode) == 0o700
    assert "\x1b" not in result.stdout + result.stderr


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_demo_summary_loader_rejects_symlink_and_oversized_receipts(tmp_path):
    target = _write_json(tmp_path / "target.json", {"status": "passed"})
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    assert _load_object(linked) == {}

    # O_NOFOLLOW protects only the final component; a symlinked parent must
    # also be rejected so a caller cannot redirect an apparently local receipt
    # outside the intended run directory.
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = _write_json(outside / "receipt.json", {"secret": True})
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    assert _load_object(linked_parent / outside_target.name) == {}

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b"x" * MAX_RECEIPT_BYTES + b"}")
    assert _load_object(oversized) == {}

    # A normal small receipt continues to load as before.
    assert _load_object(target) == {"status": "passed"}


def test_demo_summary_loader_rejects_dotdot_path_that_crosses_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = _write_json(outside / "receipt.json", {"secret": True})
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    # ``abspath`` would normalize this to ``tmp_path/receipt.json``; the
    # actual filesystem walk follows ``linked-parent`` first, so reject it
    # before attempting to open the caller-supplied receipt.
    assert _load_object(linked_parent / ".." / target.name) == {}


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
        "protocol": {
            "warmup": 3,
            "samples": 20,
            "core": 0,
            "cpu_threads_requested": 1,
        },
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
    assert "warmup=3 · samples=20" in result.stdout
    assert "[设备驻留 headline] PASS" in result.stdout
    assert "median=2.500×" in result.stdout
    assert "bootstrap95%=[2.400, 2.600]" in result.stdout
    assert "[一次组件和（诊断；非直接 E2E）]" in result.stdout
    assert "CPU/NPU=0.800×" in result.stdout
    assert "[证据路径]" in result.stdout
    assert "[回执 SHA256 前 12 位]" in result.stdout
    # Older/minimal receipts do not carry all cross-stage declarations.  The
    # summary must make that limitation explicit without changing the legacy
    # component/headline labels above.
    assert "[跨回执绑定] UNKNOWN（不可发布）" in result.stdout


def _write_cross_receipt_set(root, *, mismatch=False):
    source_sha = "a" * 64
    migrated_sha = "b" * 64
    report_sha = "c" * 64
    benchmark_migrated = "d" * 64 if mismatch else migrated_sha
    analysis = _write_json(root / "analysis.json", {
        "source_sha256": source_sha,
    })
    migration = _write_json(root / "migration.json", {
        "source_sha256": source_sha,
        "migrated_sha256": migrated_sha,
    })
    verification = _write_json(root / "verification.json", {
        "source": {"sha256": source_sha},
        "migration": {
            "migrated_sha256": migrated_sha,
            "report_sha256": report_sha,
        },
    })
    benchmark = _write_json(root / "benchmark.json", {
        "status": "passed",
        "gates": {"resident": True},
        "performance": {"performance_valid": True},
        "binding": {
            "source": {"sha256": source_sha},
            "migrated": {"sha256": benchmark_migrated},
            "migration_report": {"sha256": report_sha},
        },
    })
    return analysis, migration, verification, benchmark


def _render_summary_process(paths):
    analysis, migration, verification, benchmark = paths
    return subprocess.run(
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


def test_demo_summary_cross_receipt_binding_rejects_mismatch(tmp_path):
    result = _render_summary_process(
        _write_cross_receipt_set(tmp_path, mismatch=True)
    )

    assert result.returncode == 0, result.stderr
    assert "[跨回执绑定] FAIL（不可发布）" in result.stdout
    assert "migrated=FAIL(mismatch=M,V,B)" in result.stdout
    # The existing headline still describes only the benchmark performance
    # gate; publication must additionally respect the binding line.
    assert "[设备驻留 headline] PASS" in result.stdout


def test_demo_summary_cross_receipt_binding_passes_when_complete(tmp_path):
    result = _render_summary_process(_write_cross_receipt_set(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "[跨回执绑定] PASS" in result.stdout
    assert "source=PASS(aaaaaaaaaaaa)" in result.stdout
    assert "migrated=PASS(bbbbbbbbbbbb)" in result.stdout
    assert "report=PASS(cccccccccccc)" in result.stdout


def test_demo_summary_tty_is_a_visual_read_only_card(monkeypatch, tmp_path):
    """TTY mode groups the same truthful lines without changing the plain API."""

    paths = _write_cross_receipt_set(tmp_path)

    class TtyBuffer(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("NUMPILOT_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = TtyBuffer()

    assert render_summary_tty(*paths, stream=output)
    rendered = output.getvalue()
    assert "证据回放" in rendered
    assert "阶段门禁" in rendered
    assert "性能口径" in rendered
    assert "可追溯回执" in rendered
    # Rich control sequences are expected in a real TTY, but untrusted receipt
    # fields must still be represented as text rather than interpreted markup.
    assert "PASS" in rendered


def test_demo_summary_sanitizes_untrusted_metadata_and_secrets(monkeypatch, tmp_path):
    """Receipt fields must not become terminal injection or credential leaks."""

    token = "summary-secret-token"
    monkeypatch.setenv("NUMPILOT_LLM_AUTH_TOKEN", token)
    analysis = _write_json(tmp_path / "analysis.json", {
        "runtime_identity_status": "matched\n\x1b[31mFAKE\x1b[0m",
        "runtime_identity_sha256": "a" * 64,
        "runtime_audit_id": "audit\r\nforged=PASS",
    })
    migration = _write_json(tmp_path / "migration.json", {
        "response_model": token + "\nforged-model",
        "actual_remote_call": True,
        "llm_receipt": {
            "response_model": token,
            "response_model_identity": "Bearer leaked-token",
            "http_status": "200\nforged",
        },
    })
    verification = _write_json(tmp_path / "verification.json", {
        "comparisons": {"result": {"n_elements": "100\nforged"}},
    })
    benchmark = _write_json(tmp_path / "benchmark.json", {
        "protocol": {"samples": "20\x1b[2J", "core": 0},
    })

    rendered = render_summary(analysis, migration, verification, benchmark)

    assert token not in rendered
    assert "leaked-token" not in rendered
    assert "\x1b" not in rendered
    assert "audit\r\nforged=PASS" not in rendered
    assert "audit�forged=PASS" in rendered
    assert "n_elements" not in rendered
