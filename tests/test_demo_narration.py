# tests/test_demo_narration.py
"""演示化输出的回归:替换计数、执行证据、终端渲染。"""
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.shim import ShimBackend
from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import count_ap_calls
from numpilot.__main__ import _render_terminal
from numpilot.orchestrator import Orchestrator

SRC = ("import numpy as np\n"
       "def build_inputs():\n"
       "    return (np.zeros((2, 2), dtype=np.float32),)\n"
       "def forward(x):\n"
       "    return np.maximum(np.matmul(x, x), 0)\n")


def test_count_ap_calls_counts_per_op():
    migrated = ("import numpy as np\nimport asnumpy as ap\n"
                "def f(x):\n    return ap.relu(ap.matmul(ap.matmul(x, x), x))\n")
    assert count_ap_calls(migrated) == {"relu": 1, "matmul": 2}


def test_count_ap_calls_ignores_non_ap_and_bad_syntax():
    assert count_ap_calls("import numpy as np\nx = np.matmul(a, b)\n") == {}
    assert count_ap_calls("def broken(:\n") == {}


def test_migration_result_records_rewrites():
    plan = OffloadPlan(placement={"matmul": "NPU", "maximum": "NPU"},
                       threshold_elements=1_000_000)
    res = MigrationAgent().migrate(SRC, plan)
    # maximum(x,0) 被改写为 relu;matmul 直接改写
    assert res.rewrites.get("matmul") == 1
    assert res.rewrites.get("relu") == 1
    assert res.total_rewrites >= 2


def test_verification_fills_shim_evidence():
    plan = OffloadPlan(placement={"matmul": "NPU", "maximum": "NPU"},
                       threshold_elements=1_000_000)
    migrated = MigrationAgent().migrate(SRC, plan).migrated_source
    rep = VerificationAgent(ShimBackend()).verify(SRC, migrated)
    # shim 用 types.ModuleType 建的模块无 __file__/__version__ → 天然标记"仿真"
    assert rep.backend == "shim"
    assert rep.backend_version is None


def test_render_terminal_pass_has_all_sections():
    res = Orchestrator(backend=ShimBackend()).run("examples/mlp_forward.py")
    text = _render_terminal(res)
    for tag in ["[分析]", "[迁移]", "[卸载]", "[执行]", "[校验]", "[性能]"]:
        assert tag in text
    assert "np.* → ap.*" in text


def test_render_terminal_unsupported_shows_gap():
    res = Orchestrator(backend=ShimBackend()).run("examples/unsupported_demo.py")
    text = _render_terminal(res)
    assert "[缺口]" in text and "fft" in text
