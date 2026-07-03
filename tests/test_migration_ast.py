from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.llm import LlmEngine
from numpilot.contracts import OffloadPlan

SRC = "import numpy as np\ndef f(x, w):\n    return np.matmul(x, w)\n"

def test_ast_rewrites_npu_op_to_ap():
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    out = AstRewriteEngine().rewrite(SRC, plan)
    assert "import asnumpy as ap" in out
    assert "ap.matmul(" in out
    assert "np.matmul(" not in out

def test_ast_keeps_cpu_op_as_np():
    plan = OffloadPlan(placement={"matmul": "CPU"}, threshold_elements=1_000_000)
    out = AstRewriteEngine().rewrite(SRC, plan)
    assert "np.matmul(" in out
    assert "ap.matmul(" not in out

def test_engine_name():
    assert AstRewriteEngine().name == "ast"

def test_llm_falls_back_to_ast_when_unconfigured(monkeypatch):
    for k in ["NUMPILOT_LLM_BASE_URL", "NUMPILOT_LLM_AUTH_TOKEN", "NUMPILOT_LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    out = LlmEngine().rewrite(SRC, plan)
    assert "ap.matmul(" in out          # 未配置 → 回退 AST 引擎, 结果等价
