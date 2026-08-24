import ast
import sys

import numpy as np
import pytest

from numpilot.agents.migration import MigrationAgent
from numpilot.backends.shim import ShimBackend
from numpilot.engines.ast_rewrite import AstRewriteEngine, audit_forward_surface
from numpilot.engines.llm import LlmEngine, LlmError
from numpilot.contracts import OffloadPlan

SRC = "import numpy as np\ndef forward(x, w):\n    return np.matmul(x, w)\n"

def test_ast_rewrites_npu_op_to_ap():
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    out = AstRewriteEngine().rewrite(SRC, plan)
    assert "import asnumpy as ap" in out
    assert "ap.matmul(" in out
    assert "np.matmul(" not in out
    assert "if isinstance(x, np.ndarray):" in out
    assert "x = ap.ndarray.from_numpy(x)" in out
    assert "if isinstance(w, np.ndarray):" in out

def test_ast_keeps_cpu_op_as_np():
    plan = OffloadPlan(placement={"matmul": "CPU"}, threshold_elements=1_000_000)
    out = AstRewriteEngine().rewrite(SRC, plan)
    assert "np.matmul(" in out
    assert "ap.matmul(" not in out


def test_ast_only_rewrites_forward_boundary():
    src = """\
import numpy as np
top_level = np.zeros((2, 2), dtype=np.float32)

def build_inputs():
    return np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)

def helper(x):
    return np.exp(x)

def forward(x, w):
    return np.matmul(x, w)
"""
    plan = OffloadPlan(
        placement={"zeros": "NPU", "ones": "NPU", "exp": "NPU", "matmul": "NPU"},
        threshold_elements=1_000_000,
    )
    out = AstRewriteEngine().rewrite(src, plan)
    tree = ast.parse(out)
    by_name = {node.name: node for node in tree.body if hasattr(node, "name")}

    assert "np.zeros(" in ast.unparse(tree.body[2].value)
    assert "np.ones(" in ast.unparse(by_name["build_inputs"])
    assert "np.exp(" in ast.unparse(by_name["helper"])
    assert "ap.matmul(" in ast.unparse(by_name["forward"])


def test_migrated_artifact_runs_with_its_own_build_inputs(monkeypatch):
    src = """\
import numpy as np

def build_inputs():
    return np.ones((2, 3), dtype=np.float32), np.ones((3, 2), dtype=np.float32)

def forward(x, w):
    return np.matmul(x, w)
"""
    plan = OffloadPlan(placement={"ones": "NPU", "matmul": "NPU"}, threshold_elements=1)
    out = AstRewriteEngine().rewrite(src, plan)
    shim = ShimBackend().get_module()
    monkeypatch.setitem(sys.modules, "asnumpy", shim)
    ns = {"__name__": "migrated"}
    exec(compile(out, "<migrated>", "exec"), ns)

    result = ns["forward"](*ns["build_inputs"]())
    np.testing.assert_array_equal(result.to_numpy(), np.full((2, 2), 3.0, dtype=np.float32))

    # 已上传输入不会被二次包装,仍可直接驱动同一个 forward。
    dev_inputs = tuple(shim.ndarray.from_numpy(x) for x in ns["build_inputs"]())
    result = ns["forward"](*dev_inputs)
    np.testing.assert_array_equal(result.to_numpy(), np.full((2, 2), 3.0, dtype=np.float32))


def test_ast_and_expected_calls_do_not_rewrite_shadowed_numpy_alias():
    src = """\
import math
import numpy as np
import numpy as n

def forward(x, np):
    return n.add(np.sqrt(x), n.sqrt(x))
"""
    plan = OffloadPlan(
        placement={"add": "NPU", "sqrt": "NPU"},
        threshold_elements=1,
    )

    migration = MigrationAgent(AstRewriteEngine()).migrate(src, plan)

    assert "ap.add(np.sqrt(x), ap.sqrt(x))" in migration.migrated_source
    assert migration.expected_rewrites == {"add": 1, "sqrt": 1}
    assert migration.rewrites == migration.expected_rewrites


def test_forward_surface_requires_every_return_leaf_to_be_device_derived():
    source = """\
import numpy as np
import asnumpy as ap
def forward(x):
    value = ap.add(x, x)
    return (value, np.pi)
"""

    audit = audit_forward_surface(source)

    assert not audit.ok
    assert any("每个返回值叶子" in item for item in audit.violations)


def test_forward_surface_rejects_unknown_attributes_and_destructuring():
    unknown = """\
import asnumpy as ap
def forward(x):
    return (ap.add(x, x), ap.__dict__)
"""
    destructuring = """\
import asnumpy as ap
def forward(x):
    a, b = x
    return ap.add(a, b)
"""

    assert any(
        ".__dict__" in item
        for item in audit_forward_surface(unknown).violations
    )
    assert any(
        "单一名称赋值" in item
        for item in audit_forward_surface(destructuring).violations
    )


def test_engine_name():
    assert AstRewriteEngine().name == "ast"

def test_llm_falls_back_to_ast_when_unconfigured(monkeypatch):
    for k in ["NUMPILOT_LLM_BASE_URL", "NUMPILOT_LLM_AUTH_TOKEN", "NUMPILOT_LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    out = LlmEngine().rewrite(SRC, plan)                 # 默认非 strict
    assert "ap.matmul(" in out          # 未配置 → 回退 AST 引擎, 结果等价


def test_llm_strict_raises_when_unconfigured(monkeypatch):
    for k in ["NUMPILOT_LLM_BASE_URL", "NUMPILOT_LLM_AUTH_TOKEN", "NUMPILOT_LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    with pytest.raises(LlmError):       # strict → 未配置直接终止, 不回退
        LlmEngine(strict=True).rewrite(SRC, plan)
