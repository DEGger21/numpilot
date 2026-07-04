# tests/test_wizard.py
"""交互式向导:helper 单测 + 非交互端到端(shim, CI 安全)。"""
import os

from numpilot.wizard import (agent_roster, custom_op_breakdown, load_env_llm,
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


def test_load_env_llm(tmp_path):
    f = tmp_path / ".env.llm"
    f.write_text('export NUMPILOT_TEST_FOO="bar"\n# 注释行\nexport NUMPILOT_TEST_BAZ="q x"\n')
    loaded = load_env_llm(str(f))
    assert {"NUMPILOT_TEST_FOO", "NUMPILOT_TEST_BAZ"} <= set(loaded)
    assert os.environ["NUMPILOT_TEST_FOO"] == "bar"
    del os.environ["NUMPILOT_TEST_FOO"], os.environ["NUMPILOT_TEST_BAZ"]


def test_load_env_llm_missing_file():
    assert load_env_llm("/no/such/.env.llm") == []


def test_custom_op_breakdown():
    assert custom_op_breakdown({"matmul": 2, "relu": 3, "softmax": 1}) == {"relu": 3, "softmax": 1}
    assert custom_op_breakdown({"matmul": 2, "add": 1}) == {}


def test_agent_roster_migration_model_follows_engine():
    ast_roster = agent_roster("ast")
    llm_roster = agent_roster("llm")
    assert len(ast_roster) == 5
    # 迁移智能体是唯一(可选)用大模型的;其余无模型
    ast_mig = [m for n, r, m in ast_roster if "Migration" in n][0]
    llm_mig = [m for n, r, m in llm_roster if "Migration" in n][0]
    assert "无模型" in ast_mig and "glm-5.1" in llm_mig
    assert all("无模型" in m for n, r, m in ast_roster if "Migration" not in n)


def test_wizard_noninteractive_shim_e2e():
    # 非交互(带 file+engine)走 shim 后端, 不需 NPU/网络; 子进程 bench-cpu 为纯 numpy。
    rc = run_wizard(file="examples/mlp_forward.py", engine="ast", backend="shim")
    assert rc == 0


def test_wizard_unsupported_returns_1():
    rc = run_wizard(file="examples/unsupported_demo.py", engine="ast", backend="shim")
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
