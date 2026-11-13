# tests/test_wizard.py
"""交互式向导:helper 单测 + 非交互端到端(shim, CI 安全)。"""
import pytest

from numpilot.llm_config import LlmConfigError, save_llm_config
from numpilot.wizard import (_ask_engine, agent_roster, custom_op_breakdown,
                             run_wizard, validate_task_file)


def test_validate_ok():
    ok, p, msg = validate_task_file("examples/mlp_forward.py")
    assert ok and p.endswith("mlp_forward.py") and msg == ""


def test_validate_missing():
    ok, _, msg = validate_task_file("examples/does_not_exist.py")
    assert not ok and "不存在" in msg


def test_validate_no_required_funcs(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("import numpy as np\nx = 1\n")
    ok, _, msg = validate_task_file(str(f))
    assert not ok and "build_inputs" in msg


def test_custom_op_breakdown():
    assert custom_op_breakdown({"matmul": 2, "relu": 3, "softmax": 1}) == {"relu": 3, "softmax": 1}
    assert custom_op_breakdown({"matmul": 2, "add": 1}) == {}


def test_agent_roster_migration_model_follows_engine(monkeypatch):
    monkeypatch.setenv("NUMPILOT_LLM_MODEL", "glm-5.1")
    ast_roster = agent_roster("ast")
    llm_roster = agent_roster("llm")
    assert len(ast_roster) == 6
    # Code Migrator and Migration Reporter use the configured model under llm.
    ast_mig = [m for n, r, m in ast_roster if "Code Migrator" in n][0]
    llm_mig = [m for n, r, m in llm_roster if "Code Migrator" in n][0]
    assert "确定性" in ast_mig and "glm-5.1" in llm_mig
    ast_reporter = [m for n, r, m in ast_roster if "Migration Reporter" in n][0]
    llm_reporter = [m for n, r, m in llm_roster if "Migration Reporter" in n][0]
    assert "确定性" in ast_reporter and "glm-5.1" in llm_reporter
    assert all("无模型" not in m for n, r, m in ast_roster)


def test_ast_engine_never_reads_llm_config(monkeypatch):
    from numpilot import wizard
    monkeypatch.setattr(
        wizard,
        "load_llm_config",
        lambda: (_ for _ in ()).throw(AssertionError("AST 不应读取 LLM 配置")),
    )
    assert _ask_engine("ast") == "ast"


def test_noninteractive_llm_uses_framework_config(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", str(tmp_path / "config"))
    for key in ("NUMPILOT_LLM_BASE_URL", "NUMPILOT_LLM_AUTH_TOKEN", "NUMPILOT_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    save_llm_config("https://gateway.example/v1", "wizard-token", "wizard-model")
    assert _ask_engine("llm") == "llm"


def test_noninteractive_llm_missing_config_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMPILOT_CONFIG_HOME", str(tmp_path / "config"))
    for key in ("NUMPILOT_LLM_BASE_URL", "NUMPILOT_LLM_AUTH_TOKEN", "NUMPILOT_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(LlmConfigError):
        _ask_engine("llm")


def test_wizard_noninteractive_shim_e2e(tmp_path):
    # 非交互(带 file+engine)走 shim 后端, 不需 NPU/网络; 子进程 bench-cpu 为纯 numpy。
    rc = run_wizard(
        file="examples/mlp_forward.py",
        engine="ast",
        backend="shim",
        out_dir=str(tmp_path / "wizard"),
        reports_dir=str(tmp_path / "reports"),
    )
    assert rc == 0


def test_wizard_unsupported_returns_1(tmp_path):
    rc = run_wizard(
        file="examples/unsupported_demo.py",
        engine="ast",
        backend="shim",
        reports_dir=str(tmp_path / "reports"),
    )
    assert rc == 1


def test_wizard_startup_check_aborts_when_asnumpy_missing(monkeypatch):
    # 启动即自检:npu 后端 asnumpy 缺失 → 非交互直接终止(rc=3), 不进入选文件/迁移
    from numpilot import wizard
    from numpilot.env_check import ERROR, EnvReport

    def fake_check(backend="npu"):
        rep = EnvReport(backend=backend)
        rep.add("numpy", "ok", "x")
        rep.add("asnumpy", ERROR, "未安装(模拟)")
        return rep

    monkeypatch.setattr(wizard.Env, "check", staticmethod(fake_check))
    rc = run_wizard(file="examples/mlp_forward.py", engine="ast", backend="npu")
    assert rc == 3
