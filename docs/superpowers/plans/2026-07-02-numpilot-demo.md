# NumPilot 最简 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地一个独立的 Python 多智能体程序,把只用 AsNumpy 已覆盖算子的 NumPy 程序自动迁移为 AsNumpy(NPU)版本,并做数值一致性校验与加速比评测。

**Architecture:** 编排器(Orchestrator)统筹 4 个契约化智能体(Analysis→Decision→Migration→Verification);迁移引擎用策略模式(默认 AST 改写,LLM 可插拔);执行后端用端口-适配器(默认 numpy-shim,真实 NPU 可换)。

**Tech Stack:** Python 3.9+、标准库 `ast`、`numpy`、`pytest`。默认路径零外部依赖(仅 numpy)。

## Global Constraints

- Python >= 3.9;运行时依赖仅 `numpy`(测试另需 `pytest`)。
- 文档/注释中文为主,代码/路径/标识符英文。
- 覆盖集从真实 `asnumpy` 的 `_LAZY_MAPPING` 键派生的快照,存 `numpilot/coverage.py`。
- 默认路径**零外部依赖、零 LLM key**:`python -m numpilot migrate examples/mlp_forward.py` 一键跑通。
- 迁移的本质:`np.<covered>` → `ap.<covered>`,输入用 `ap.ndarray.from_numpy(x)` 上 NPU,最终结果 `.to_numpy()` 回 CPU。
- 决策阈值默认 `1_000_000` 元素上 NPU,可配。
- 容差按 dtype:fp32→`1e-4`,fp16→`1e-2`,整型→精确相等。
- 每个 Agent 只依赖 `contracts.py` 中的 dataclass 输入/输出,不跨层耦合。
- 工作根目录:`/home/lixing/numpilot`。git 已初始化。

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `numpilot/contracts.py` | 全部 dataclass 契约(`OpUse/AnalysisResult/OffloadPlan/MigrationResult/VerificationReport`) |
| `numpilot/coverage.py` | AsNumpy 覆盖集快照 + `is_supported()` |
| `numpilot/agents/analysis.py` | AST 解析,产出 `AnalysisResult` |
| `numpilot/agents/decision.py` | 规则成本模型,产出 `OffloadPlan` |
| `numpilot/engines/base.py` | `MigrationEngine` 抽象接口 |
| `numpilot/engines/ast_rewrite.py` | 默认 AST 改写引擎 |
| `numpilot/engines/llm.py` | provider 无关 LLM stub(回退 AST) |
| `numpilot/agents/migration.py` | 调用引擎,产出 `MigrationResult` |
| `numpilot/backends/base.py` | `ExecutionBackend` 端口 + `select_backend()` |
| `numpilot/backends/shim.py` | numpy 支撑的假 asnumpy |
| `numpilot/backends/npu.py` | 真实 asnumpy 适配器 |
| `numpilot/agents/verification.py` | 跑双端、数值对比、测速,产出 `VerificationReport` |
| `numpilot/orchestrator.py` | 编排 4 Agent,汇总报告 |
| `numpilot/__main__.py` | CLI 入口 |
| `examples/mlp_forward.py` | 待迁移的 numpy MLP |
| `examples/unsupported_demo.py` | 负例(含缺口算子) |
| `docs/architecture.md` / `quality-scenarios.md` / `adr/*` | 课程架构文档 |

---

## Task 1: 项目骨架 + 覆盖集

**Files:**
- Create: `numpilot/__init__.py`, `numpilot/coverage.py`, `pyproject.toml`, `tests/__init__.py`, `tests/test_coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Produces: `numpilot.coverage.SUPPORTED_OPS: frozenset[str]`;`numpilot.coverage.is_supported(name: str) -> bool`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_coverage.py
from numpilot.coverage import SUPPORTED_OPS, is_supported

def test_core_ops_supported():
    for name in ["add", "subtract", "multiply", "divide", "matmul",
                 "exp", "sqrt", "sum", "mean", "maximum", "relu", "softmax"]:
        assert is_supported(name), name

def test_unsupported_op():
    assert not is_supported("fft")          # 不在覆盖集
    assert not is_supported("convolve")

def test_supported_ops_is_frozenset():
    assert isinstance(SUPPORTED_OPS, frozenset)
    assert "matmul" in SUPPORTED_OPS
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_coverage.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'numpilot'`)

- [ ] **Step 3: 写 pyproject.toml**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "numpilot"
version = "0.1.0"
description = "面向科学智能的昇腾 NPU 代码自动迁移智能体(最简 demo)"
requires-python = ">=3.9"
dependencies = ["numpy"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["numpilot*"]
```

- [ ] **Step 4: 写 coverage.py**

```python
# numpilot/coverage.py
"""AsNumpy 覆盖集快照。

来源:真实 asnumpy/src/asnumpy/__init__.py 的 _LAZY_MAPPING 键。
再生成:python -c "import asnumpy,inspect; ..." 或手工比对上游。
本 demo 用静态快照,避免运行时依赖 asnumpy。
"""

# 从 _LAZY_MAPPING 派生的算子名(去掉 dtype 别名与模块名如 linalg/random/custom_ops)
SUPPORTED_OPS = frozenset({
    # array 创建
    "empty", "empty_like", "eye", "full", "full_like", "identity",
    "linspace", "ones", "ones_like", "zeros", "zeros_like",
    # linalg.direct
    "dot", "einsum", "inner", "matmul", "outer", "vdot",
    # logic
    "all", "any", "equal", "greater", "greater_equal", "isfinite",
    "isinf", "isneginf", "isposinf", "less", "less_equal",
    "logical_and", "logical_not", "logical_or", "logical_xor", "not_equal",
    # math
    "absolute", "add", "amax", "amin", "around", "arccos", "arccosh",
    "arcsin", "arcsinh", "arctan", "arctan2", "arctanh", "ceil", "clip",
    "copysign", "cos", "cosh", "cross", "cumprod", "cumsum", "deg2rad",
    "degrees", "divide", "divmod", "exp", "exp2", "expm1", "fabs", "fix",
    "floor", "floor_divide", "fmax", "fmin", "fmod", "float_power", "gelu",
    "gcd", "heaviside", "hypot", "lcm", "ldexp", "log", "log10", "log1p",
    "log2", "logaddexp", "logaddexp2", "max", "maximum", "min", "minimum",
    "mod", "modf", "multiply", "nan_to_num", "nancumprod", "nancumsum",
    "nanmax", "nanprod", "nansum", "negative", "power", "positive", "prod",
    "rad2deg", "radians", "real", "reciprocal", "relu", "remainder", "rint",
    "round_", "sign", "signbit", "sin", "sinc", "sinh", "sqrt", "square",
    "subtract", "sum", "tan", "tanh", "true_divide", "trunc",
    # sorting / statistics / nn
    "sort", "mean", "softmax",
    # custom_ops
    "angle",
})


def is_supported(name: str) -> bool:
    """判断某 numpy 函数名是否被 AsNumpy 覆盖。"""
    return name in SUPPORTED_OPS
```

同时创建空文件 `numpilot/__init__.py` 与 `tests/__init__.py`。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_coverage.py -v`
Expected: PASS(3 passed)

- [ ] **Step 6: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/ tests/ pyproject.toml
git commit -m "feat: 项目骨架 + AsNumpy 覆盖集快照"
```

---

## Task 2: 契约层(contracts.py)

**Files:**
- Create: `numpilot/contracts.py`, `tests/test_contracts.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces:
  - `OpUse(name: str, lineno: int, supported: bool, est_elements: Optional[int])`
  - `AnalysisResult(source_path: str, op_uses: list[OpUse], unsupported: list[str], migratable: bool)`
  - `OffloadPlan(placement: dict[str, str], threshold_elements: int)`
  - `MigrationResult(engine: str, migrated_source: str)`
  - `VerificationReport(numerically_consistent: bool, max_rel_diff: float, tol: float, speedup: Optional[float], per_op: list[dict], status: str)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_contracts.py
from numpilot.contracts import (
    OpUse, AnalysisResult, OffloadPlan, MigrationResult, VerificationReport,
)

def test_opuse_fields():
    op = OpUse(name="matmul", lineno=3, supported=True, est_elements=None)
    assert op.name == "matmul" and op.supported

def test_analysis_result_migratable():
    r = AnalysisResult(source_path="a.py", op_uses=[], unsupported=[], migratable=True)
    assert r.migratable and r.unsupported == []

def test_report_status():
    rep = VerificationReport(numerically_consistent=True, max_rel_diff=0.0,
                             tol=1e-4, speedup=2.0, per_op=[], status="pass")
    assert rep.status == "pass"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_contracts.py -v`
Expected: FAIL(`ModuleNotFoundError` / ImportError)

- [ ] **Step 3: 写 contracts.py**

```python
# numpilot/contracts.py
"""Agent 间传递的数据契约。每个 Agent 只认这里的输入/输出类型。"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpUse:
    name: str                       # numpy 函数名, e.g. "matmul"
    lineno: int
    supported: bool                 # 是否在覆盖集
    est_elements: Optional[int] = None  # 估计张量规模(决策用), 未知为 None


@dataclass
class AnalysisResult:
    source_path: str
    op_uses: list                   # list[OpUse]
    unsupported: list               # list[str], 去重的缺口算子名
    migratable: bool                # unsupported 为空


@dataclass
class OffloadPlan:
    placement: dict                 # op name -> "NPU" | "CPU"
    threshold_elements: int


@dataclass
class MigrationResult:
    engine: str                     # "ast" | "llm"
    migrated_source: str


@dataclass
class VerificationReport:
    numerically_consistent: bool
    max_rel_diff: float
    tol: float
    speedup: Optional[float]
    per_op: list = field(default_factory=list)
    status: str = "pass"            # "pass" | "fail" | "unsupported"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_contracts.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/contracts.py tests/test_contracts.py
git commit -m "feat: Agent 间 dataclass 契约层"
```

---

## Task 3: 分析智能体(Analysis Agent)

**Files:**
- Create: `numpilot/agents/__init__.py`, `numpilot/agents/analysis.py`, `tests/test_analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `numpilot.coverage.is_supported`;`numpilot.contracts.{OpUse,AnalysisResult}`
- Produces: `AnalysisAgent().analyze(source: str, source_path: str = "<mem>") -> AnalysisResult`
  - 识别形如 `np.<name>(...)` / `numpy.<name>(...)` 的调用;`<name>` 是被调用的 numpy 函数名。
  - `est_elements`:若实参能静态推断出 `np.zeros((a,b))` / `np.ones([...])` 之类的 shape 常量,估其元素数;否则 None。demo 只需支持从 `np.zeros/ones/full/empty((...))` 的第一个 shape 常量估算。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_analysis.py
from numpilot.agents.analysis import AnalysisAgent

SRC_OK = """
import numpy as np
def f(x, w):
    h = np.matmul(x, w)
    return np.maximum(h, 0)
"""

SRC_GAP = """
import numpy as np
def g(x):
    return np.fft.fft(x)
"""

def test_extracts_op_uses():
    r = AnalysisAgent().analyze(SRC_OK)
    names = {op.name for op in r.op_uses}
    assert "matmul" in names and "maximum" in names
    assert r.migratable and r.unsupported == []

def test_detects_unsupported():
    r = AnalysisAgent().analyze(SRC_GAP)
    assert "fft" in r.unsupported
    assert not r.migratable

def test_est_elements_from_zeros():
    src = "import numpy as np\nx = np.zeros((100, 200))\n"
    r = AnalysisAgent().analyze(src)
    z = [op for op in r.op_uses if op.name == "zeros"][0]
    assert z.est_elements == 20000
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_analysis.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 analysis.py**

```python
# numpilot/agents/analysis.py
"""分析智能体:AST 静态解析,提取 numpy 算子清单并比对覆盖集。"""
import ast
from math import prod

from numpilot.contracts import AnalysisResult, OpUse
from numpilot.coverage import is_supported


class _OpVisitor(ast.NodeVisitor):
    def __init__(self, np_aliases):
        self.np_aliases = np_aliases      # {"np", "numpy"}
        self.op_uses = []

    def visit_Call(self, node):
        name = self._numpy_func_name(node.func)
        if name is not None:
            self.op_uses.append(
                OpUse(
                    name=name,
                    lineno=getattr(node, "lineno", 0),
                    supported=is_supported(name),
                    est_elements=self._est_elements(name, node),
                )
            )
        self.generic_visit(node)

    def _numpy_func_name(self, func):
        # np.matmul(...) -> "matmul"; np.fft.fft(...) -> "fft"(取最后一段, 命名空间不在覆盖集自然判缺口)
        if isinstance(func, ast.Attribute):
            root = func
            while isinstance(root.value, ast.Attribute):
                root = root.value
            if isinstance(root.value, ast.Name) and root.value.id in self.np_aliases:
                return func.attr
        return None

    def _est_elements(self, name, node):
        if name not in ("zeros", "ones", "full", "empty"):
            return None
        if not node.args:
            return None
        shape_node = node.args[0]
        dims = self._const_shape(shape_node)
        return prod(dims) if dims else None

    @staticmethod
    def _const_shape(shape_node):
        if isinstance(shape_node, ast.Constant) and isinstance(shape_node.value, int):
            return [shape_node.value]
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            dims = []
            for el in shape_node.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, int):
                    dims.append(el.value)
                else:
                    return None
            return dims
        return None


class AnalysisAgent:
    def analyze(self, source: str, source_path: str = "<mem>") -> AnalysisResult:
        tree = ast.parse(source)
        np_aliases = self._collect_numpy_aliases(tree)
        visitor = _OpVisitor(np_aliases)
        visitor.visit(tree)
        unsupported = sorted({op.name for op in visitor.op_uses if not op.supported})
        return AnalysisResult(
            source_path=source_path,
            op_uses=visitor.op_uses,
            unsupported=unsupported,
            migratable=(len(unsupported) == 0 and len(visitor.op_uses) > 0),
        )

    @staticmethod
    def _collect_numpy_aliases(tree):
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "numpy":
                        aliases.add(a.asname or "numpy")
        return aliases or {"np", "numpy"}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_analysis.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/agents/ tests/test_analysis.py
git commit -m "feat: 分析智能体(AST 算子提取 + 覆盖集比对)"
```

---

## Task 4: 决策智能体(Decision Agent)

**Files:**
- Create: `numpilot/agents/decision.py`, `tests/test_decision.py`
- Test: `tests/test_decision.py`

**Interfaces:**
- Consumes: `AnalysisResult`;`OffloadPlan`
- Produces: `DecisionAgent(threshold_elements: int = 1_000_000).plan(analysis: AnalysisResult) -> OffloadPlan`
  - 规则:某算子任一调用点 `est_elements >= threshold` → 该算子 `"NPU"`;若所有调用点 `est_elements` 均已知且 `< threshold` → `"CPU"`;若规模未知(None)→ 默认 `"NPU"`(demo 保守偏向 NPU 以展示迁移)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_decision.py
from numpilot.agents.decision import DecisionAgent
from numpilot.contracts import AnalysisResult, OpUse

def _analysis(op_uses):
    return AnalysisResult("m", op_uses, [], True)

def test_large_op_goes_npu():
    a = _analysis([OpUse("matmul", 1, True, 4_000_000)])
    plan = DecisionAgent(threshold_elements=1_000_000).plan(a)
    assert plan.placement["matmul"] == "NPU"

def test_small_known_op_goes_cpu():
    a = _analysis([OpUse("add", 1, True, 100)])
    plan = DecisionAgent(threshold_elements=1_000_000).plan(a)
    assert plan.placement["add"] == "CPU"

def test_unknown_size_defaults_npu():
    a = _analysis([OpUse("exp", 1, True, None)])
    plan = DecisionAgent().plan(a)
    assert plan.placement["exp"] == "NPU"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_decision.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 decision.py**

```python
# numpilot/agents/decision.py
"""决策智能体:规则版成本模型,按张量规模阈值决定 CPU/NPU 卸载。"""
from numpilot.contracts import AnalysisResult, OffloadPlan


class DecisionAgent:
    def __init__(self, threshold_elements: int = 1_000_000):
        self.threshold = threshold_elements

    def plan(self, analysis: AnalysisResult) -> OffloadPlan:
        by_op = {}
        for op in analysis.op_uses:
            by_op.setdefault(op.name, []).append(op.est_elements)

        placement = {}
        for name, sizes in by_op.items():
            if any(s is None for s in sizes):
                placement[name] = "NPU"                 # 规模未知, 保守上 NPU
            elif any(s >= self.threshold for s in sizes):
                placement[name] = "NPU"
            else:
                placement[name] = "CPU"
        return OffloadPlan(placement=placement, threshold_elements=self.threshold)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_decision.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/agents/decision.py tests/test_decision.py
git commit -m "feat: 决策智能体(规则成本模型 CPU/NPU 卸载)"
```

---

## Task 5: 迁移引擎(策略接口 + AST 改写 + LLM stub)

**Files:**
- Create: `numpilot/engines/__init__.py`, `numpilot/engines/base.py`, `numpilot/engines/ast_rewrite.py`, `numpilot/engines/llm.py`, `tests/test_migration_ast.py`
- Test: `tests/test_migration_ast.py`

**Interfaces:**
- Consumes: `OffloadPlan`
- Produces:
  - `numpilot.engines.base.MigrationEngine`(抽象):`name: str`;`rewrite(source: str, plan: OffloadPlan) -> str`
  - `numpilot.engines.ast_rewrite.AstRewriteEngine`(`name="ast"`)
  - `numpilot.engines.llm.LlmEngine`(`name="llm"`,未配置时回退 AST)
- AST 改写规则(demo 目标产物,基于覆盖集泛化):
  1. `import numpy as np` 之后追加 `import asnumpy as ap`。
  2. 对 `placement[op] == "NPU"` 的算子调用 `np.<op>(...)` → `ap.<op>(...)`。`"CPU"` 的保持 `np.<op>`。
  3. 数据搬运在 demo 中由 example 程序的 driver 负责(见 Task 8/9),引擎只做算子命名空间改写,保持最小侵入且语义等价。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_migration_ast.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_migration_ast.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 base.py**

```python
# numpilot/engines/base.py
"""迁移引擎策略接口。"""
from abc import ABC, abstractmethod

from numpilot.contracts import OffloadPlan


class MigrationEngine(ABC):
    name: str = "base"

    @abstractmethod
    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        """把 numpy 源码按卸载方案改写为 asnumpy 版本, 返回新源码。"""
        raise NotImplementedError
```

- [ ] **Step 4: 写 ast_rewrite.py**

```python
# numpilot/engines/ast_rewrite.py
"""默认迁移引擎:确定性 AST 改写。基于覆盖集与卸载方案泛化改写。"""
import ast

from numpilot.contracts import OffloadPlan
from numpilot.engines.base import MigrationEngine


class _Rewriter(ast.NodeTransformer):
    def __init__(self, np_aliases, npu_ops):
        self.np_aliases = np_aliases        # {"np"}
        self.npu_ops = npu_ops              # set[str], 需改到 ap 的算子
        self.used_ap = False

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.np_aliases
            and node.attr in self.npu_ops
        ):
            self.used_ap = True
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="ap", ctx=ast.Load()),
                              attr=node.attr, ctx=node.ctx),
                node,
            )
        return node


class AstRewriteEngine(MigrationEngine):
    name = "ast"

    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        tree = ast.parse(source)
        np_aliases = self._np_aliases(tree)
        npu_ops = {op for op, place in plan.placement.items() if place == "NPU"}
        rewriter = _Rewriter(np_aliases, npu_ops)
        new_tree = rewriter.visit(tree)
        ast.fix_missing_locations(new_tree)
        if rewriter.used_ap:
            self._insert_ap_import(new_tree)
        return ast.unparse(new_tree)

    @staticmethod
    def _np_aliases(tree):
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "numpy":
                        aliases.add(a.asname or "numpy")
        return aliases or {"np"}

    @staticmethod
    def _insert_ap_import(tree):
        imp = ast.Import(names=[ast.alias(name="asnumpy", asname="ap")])
        last_import_idx = -1
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import_idx = i
        tree.body.insert(last_import_idx + 1, imp)
        ast.fix_missing_locations(tree)
```

- [ ] **Step 5: 写 llm.py**

```python
# numpilot/engines/llm.py
"""可插拔 LLM 迁移引擎(provider 无关, OpenAI 兼容协议)。

启用需以下环境变量(任一缺失即回退 AST 引擎):
  NUMPILOT_LLM_BASE_URL    OpenAI 兼容 endpoint, 例: https://.../v1
  NUMPILOT_LLM_AUTH_TOKEN  鉴权 token(以 Authorization: Bearer 发送)
  NUMPILOT_LLM_MODEL       模型名
  NUMPILOT_LLM_PROVIDER    (可选)仅作标签, 例: pangu | glm | openai | deepseek

只用标准库 urllib 调用, 不引入额外依赖。任何异常(网络/解析/鉴权)
都回退确定性 AST 引擎, 保证 demo 始终可跑。凭据仅从环境变量读取,
不落盘、不入库。
"""
import json
import os
import re
import urllib.request

from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.base import MigrationEngine

_REQUIRED_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

_SYSTEM_PROMPT = (
    "你是 NumPilot 的代码迁移智能体。把用户给的 NumPy 程序迁移为 AsNumpy(NPU)版本:\n"
    "1) 在 import numpy 之后加 `import asnumpy as ap`;\n"
    "2) 把下列需上 NPU 的算子调用 np.<op>(...) 改为 ap.<op>(...);其余保持不变;\n"
    "3) 不改变计算逻辑与函数签名, 只做命名空间替换与必要的 import。\n"
    "只输出完整的迁移后 Python 源码, 用 ```python 代码块包裹, 不要解释。"
)


class LlmEngine(MigrationEngine):
    name = "llm"

    def is_configured(self) -> bool:
        return all(os.getenv(k) for k in _REQUIRED_ENV)

    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        if not self.is_configured():
            return AstRewriteEngine().rewrite(source, plan)
        try:
            npu_ops = sorted(op for op, p in plan.placement.items() if p == "NPU")
            user_msg = f"需上 NPU 的算子: {npu_ops}\n\n源码:\n```python\n{source}\n```"
            content = self._chat(_SYSTEM_PROMPT, user_msg)
            code = self._extract_code(content)
            if code and "ap." in code:
                return code
        except Exception:
            pass
        # 任何失败都回退 AST, 保证可跑
        return AstRewriteEngine().rewrite(source, plan)

    @staticmethod
    def _chat(system: str, user: str) -> str:
        base = os.environ["NUMPILOT_LLM_BASE_URL"].rstrip("/")
        url = base + "/chat/completions"
        body = json.dumps({
            "model": os.environ["NUMPILOT_LLM_MODEL"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + os.environ["NUMPILOT_LLM_AUTH_TOKEN"])
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
        return (m.group(1) if m else content).strip()
```

同时创建空 `numpilot/engines/__init__.py`。

- [ ] **Step 6: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_migration_ast.py -v`
Expected: PASS(4 passed)

- [ ] **Step 7: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/engines/ tests/test_migration_ast.py
git commit -m "feat: 迁移引擎(策略接口 + AST 改写 + LLM 可插拔 stub)"
```

---

## Task 6: 迁移智能体(Migration Agent)

**Files:**
- Create: `numpilot/agents/migration.py`, `tests/test_migration_agent.py`
- Test: `tests/test_migration_agent.py`

**Interfaces:**
- Consumes: `MigrationEngine`;`OffloadPlan`;`MigrationResult`
- Produces: `MigrationAgent(engine: MigrationEngine = AstRewriteEngine()).migrate(source: str, plan: OffloadPlan) -> MigrationResult`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_migration_agent.py
from numpilot.agents.migration import MigrationAgent
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.contracts import OffloadPlan

SRC = "import numpy as np\ndef f(x, w):\n    return np.matmul(x, w)\n"

def test_migrate_returns_result_with_engine_name():
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    res = MigrationAgent(AstRewriteEngine()).migrate(SRC, plan)
    assert res.engine == "ast"
    assert "ap.matmul(" in res.migrated_source
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_migration_agent.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 migration.py**

```python
# numpilot/agents/migration.py
"""迁移智能体:委托 MigrationEngine 策略完成 np->ap 改写。"""
from numpilot.contracts import MigrationResult, OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.base import MigrationEngine


class MigrationAgent:
    def __init__(self, engine: MigrationEngine = None):
        self.engine = engine or AstRewriteEngine()

    def migrate(self, source: str, plan: OffloadPlan) -> MigrationResult:
        migrated = self.engine.rewrite(source, plan)
        return MigrationResult(engine=self.engine.name, migrated_source=migrated)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_migration_agent.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/agents/migration.py tests/test_migration_agent.py
git commit -m "feat: 迁移智能体(委托策略引擎)"
```

---

## Task 7: 执行后端(端口 + shim + npu 适配器)

**Files:**
- Create: `numpilot/backends/__init__.py`, `numpilot/backends/base.py`, `numpilot/backends/shim.py`, `numpilot/backends/npu.py`, `tests/test_backends.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces:
  - `numpilot.backends.base.ExecutionBackend`(抽象):`name: str`;`get_module() -> module`(返回可当 `asnumpy` 用的模块对象);`to_device(np_array) -> device_array`;`to_host(device_array) -> np_array`。
  - `numpilot.backends.shim.ShimBackend`(`name="shim"`):`get_module()` 返回一个动态构造的假 `asnumpy` 模块,内含 `add/subtract/multiply/divide/matmul/dot/exp/sqrt/square/sum/mean/max/maximum/minimum/tanh/relu/softmax` 等常用算子(转发 numpy)以及 `ndarray` 命名空间(`ndarray.from_numpy` / `.to_numpy()`)。
  - `numpilot.backends.npu.NpuBackend`(`name="npu"`):`get_module()` 返回真实 `import asnumpy`(不可用时抛清晰错误)。
  - `numpilot.backends.base.select_backend(name: str | None = None) -> ExecutionBackend`:读参数或 `NUMPILOT_BACKEND` 环境变量,默认 `"shim"`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backends.py
import numpy as np
from numpilot.backends.base import select_backend
from numpilot.backends.shim import ShimBackend

def test_default_backend_is_shim():
    b = select_backend()
    assert b.name == "shim"

def test_shim_module_matmul_matches_numpy():
    ap = ShimBackend().get_module()
    x = np.ones((4, 4), dtype=np.float32)
    xd = ap.ndarray.from_numpy(x)
    out = ap.matmul(xd, xd).to_numpy()
    assert np.allclose(out, np.matmul(x, x))

def test_shim_relu_and_softmax():
    ap = ShimBackend().get_module()
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    xd = ap.ndarray.from_numpy(x)
    assert np.allclose(ap.relu(xd).to_numpy(), np.maximum(x, 0))
    sm = ap.softmax(ap.ndarray.from_numpy(x)).to_numpy()
    assert np.allclose(sm.sum(), 1.0, atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_backends.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 base.py**

```python
# numpilot/backends/base.py
"""执行后端端口:让验证阶段在不同后端(shim / 真实 NPU)运行迁移代码。"""
import os
from abc import ABC, abstractmethod


class ExecutionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def get_module(self):
        """返回一个可作为 `asnumpy` 使用的模块对象。"""

    @abstractmethod
    def to_device(self, np_array):
        ...

    @abstractmethod
    def to_host(self, device_array):
        ...


def select_backend(name=None):
    name = name or os.getenv("NUMPILOT_BACKEND", "shim")
    if name == "shim":
        from numpilot.backends.shim import ShimBackend
        return ShimBackend()
    if name == "npu":
        from numpilot.backends.npu import NpuBackend
        return NpuBackend()
    raise ValueError(f"未知后端: {name!r}(可选 shim | npu)")
```

- [ ] **Step 4: 写 shim.py**

```python
# numpilot/backends/shim.py
"""numpy 支撑的假 asnumpy 后端。让迁移后的 `import asnumpy as ap` 代码
在任意机器离线跑通, 数值与 numpy 恒等, 便于复现与 CI。"""
import types

import numpy as np

from numpilot.backends.base import ExecutionBackend


class _NDArray:
    """对标 asnumpy.NPUArray 的最小包装, 内部用 numpy 承载。"""
    def __init__(self, data):
        self._data = np.asarray(data)

    @staticmethod
    def from_numpy(arr):
        return _NDArray(arr)

    def to_numpy(self):
        return np.asarray(self._data)


def _unwrap(x):
    return x._data if isinstance(x, _NDArray) else x


def _wrap(x):
    return _NDArray(x)


def _make_module():
    m = types.ModuleType("asnumpy")

    def _u(name, npfunc):
        setattr(m, name, lambda a, *args, **kw: _wrap(npfunc(_unwrap(a), *args, **kw)))

    def _b(name, npfunc):
        setattr(m, name, lambda a, b, *args, **kw: _wrap(npfunc(_unwrap(a), _unwrap(b), *args, **kw)))

    # 二元 / 通用
    for name in ["add", "subtract", "multiply", "divide", "matmul", "dot",
                 "maximum", "minimum", "power"]:
        _b(name, getattr(np, name))
    # 一元
    for name in ["exp", "sqrt", "square", "tanh", "sin", "cos", "log", "abs"]:
        _u(name, getattr(np, name))
    # 归约(带 axis 透传)
    for name in ["sum", "mean", "max", "min"]:
        _u(name, getattr(np, name))

    # 非 numpy 原生算子
    m.relu = lambda a: _wrap(np.maximum(_unwrap(a), 0))

    def _softmax(a, axis=-1):
        x = _unwrap(a)
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return _wrap(e / np.sum(e, axis=axis, keepdims=True))
    m.softmax = _softmax

    # 创建
    m.zeros = lambda *a, **k: _wrap(np.zeros(*a, **k))
    m.ones = lambda *a, **k: _wrap(np.ones(*a, **k))

    m.ndarray = _NDArray
    return m


class ShimBackend(ExecutionBackend):
    name = "shim"

    def __init__(self):
        self._module = _make_module()

    def get_module(self):
        return self._module

    def to_device(self, np_array):
        return _NDArray.from_numpy(np_array)

    def to_host(self, device_array):
        return device_array.to_numpy()
```

- [ ] **Step 5: 写 npu.py**

```python
# numpilot/backends/npu.py
"""真实 asnumpy(NPU)后端适配器。需已 build/install asnumpy。"""
from numpilot.backends.base import ExecutionBackend


class NpuBackend(ExecutionBackend):
    name = "npu"

    def get_module(self):
        try:
            import asnumpy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "未找到 asnumpy。请先在 CANN 环境下 build/install asnumpy, "
                "或改用默认 shim 后端(NUMPILOT_BACKEND=shim)。"
            ) from e
        return asnumpy

    def to_device(self, np_array):
        return self.get_module().ndarray.from_numpy(np_array)

    def to_host(self, device_array):
        return device_array.to_numpy()
```

同时创建空 `numpilot/backends/__init__.py`。

- [ ] **Step 6: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_backends.py -v`
Expected: PASS(3 passed)

- [ ] **Step 7: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/backends/ tests/test_backends.py
git commit -m "feat: 执行后端(端口-适配器: shim 默认 + 真实 NPU)"
```

---

## Task 8: 示例程序(MLP 前向 + 负例)

**Files:**
- Create: `examples/mlp_forward.py`, `examples/unsupported_demo.py`
- Test:(端到端在 Task 10 覆盖;本任务只需能被 numpy 直接跑)

**Interfaces:**
- `examples/mlp_forward.py` 提供:
  - `build_inputs() -> tuple[np.ndarray, ...]`:构造固定随机种子的 `x, W1, b1, W2, b2`(fp32,规模适中,如 batch=64、784→256→10)。
  - `forward(x, W1, b1, W2, b2, *, xp=np) -> np.ndarray`:仅用覆盖集算子(`matmul/add/maximum/exp/sum` 等),`xp` 参数允许注入 `np` 或迁移后的 `ap`(依赖注入,便于验证阶段双端复用同一份逻辑)。
- `examples/unsupported_demo.py`:用一个缺口算子(如 `np.fft.fft`),用于负例。

- [ ] **Step 1: 写 mlp_forward.py**

```python
# examples/mlp_forward.py
"""待迁移示例:小型 MLP 前向推理。仅使用 AsNumpy 覆盖集算子。

forward() 接受 xp 参数(numpy 或迁移后的 asnumpy 别名), 以便验证阶段
在同一份计算逻辑上对比双端结果。真实迁移由 NumPilot 的 AST 引擎完成,
本文件的 xp 注入只是为验证提供等价参照。
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 784)).astype(np.float32)
    W1 = rng.standard_normal((784, 256)).astype(np.float32) * 0.02
    b1 = np.zeros((256,), dtype=np.float32)
    W2 = rng.standard_normal((256, 10)).astype(np.float32) * 0.02
    b2 = np.zeros((10,), dtype=np.float32)
    return x, W1, b1, W2, b2


def forward(x, W1, b1, W2, b2, *, xp=np):
    h = xp.add(xp.matmul(x, W1), b1)
    h = xp.maximum(h, 0)                     # relu
    logits = xp.add(xp.matmul(h, W2), b2)
    e = xp.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


if __name__ == "__main__":
    out = forward(*build_inputs())
    print("output shape:", out.shape, "row sums ~1:", np.allclose(out.sum(-1), 1, atol=1e-4))
```

- [ ] **Step 2: 跑一下确认 numpy 侧可用**

Run: `cd /home/lixing/numpilot && python examples/mlp_forward.py`
Expected: 打印 `output shape: (64, 10) row sums ~1: True`

- [ ] **Step 3: 写 unsupported_demo.py**

```python
# examples/unsupported_demo.py
"""负例:使用缺口算子(np.fft.fft), NumPilot 应标记 unsupported 并终止迁移。"""
import numpy as np


def spectrum(x):
    return np.fft.fft(x)


if __name__ == "__main__":
    print(spectrum(np.ones(8)))
```

- [ ] **Step 4: 提交**

```bash
cd /home/lixing/numpilot
git add examples/
git commit -m "feat: 示例程序(MLP 前向 + 缺口算子负例)"
```

---

## Task 9: 验证智能体(Verification Agent)

**Files:**
- Create: `numpilot/agents/verification.py`, `tests/test_verification.py`
- Test: `tests/test_verification.py`

**Interfaces:**
- Consumes: `ExecutionBackend`;`VerificationReport`
- Produces: `VerificationAgent(backend: ExecutionBackend = ShimBackend()).verify(orig_source: str, migrated_source: str, dtype=np.float32) -> VerificationReport`
  - 执行策略(demo 稳健做法):在受控命名空间 `exec` 原始源码与迁移源码,各取其 `forward` 与 `build_inputs`。迁移源码的 `import asnumpy as ap` 由后端注入的假模块满足(通过 `sys.modules` 临时挂载 backend.get_module())。
  - 数值对比:两端 `forward` 结果转 numpy 后算 `max_rel_diff`;与 dtype 容差比较得 `numerically_consistent`。
  - 测速:分别计时(warmup + N 次取中位),`speedup = t_orig / t_migrated`(shim 后端下约等于 1,仅演示链路)。
  - `status`:数值一致→`"pass"`,否则→`"fail"`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_verification.py
import numpy as np
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.shim import ShimBackend

ORIG = open("examples/mlp_forward.py").read()

def test_migrated_matches_original_under_shim():
    # 迁移版:把 np.<op> 换成 ap.<op>(与 AST 引擎产物一致的最小形态)
    migrated = ORIG.replace("import numpy as np\n",
                            "import numpy as np\nimport asnumpy as ap\n")
    for op in ["add", "matmul", "maximum", "exp"]:
        migrated = migrated.replace(f"xp.{op}", f"ap.{op}")
    # forward 默认 xp=np; 这里把 ap 传进去以驱动后端
    rep = VerificationAgent(ShimBackend()).verify(ORIG, migrated)
    assert rep.status == "pass"
    assert rep.numerically_consistent
    assert rep.max_rel_diff < rep.tol
```

> 说明:该测试聚焦"验证智能体在 shim 后端下能判定数值一致"。实现里 `verify` 通过把 backend 模块挂到 `sys.modules["asnumpy"]`,再 `exec` 迁移源码取其 `forward`,并以 `xp=ap` 调用。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_verification.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 verification.py**

```python
# numpilot/agents/verification.py
"""验证智能体:双端执行 + 数值对比 + 测速, 产出 VerificationReport。"""
import sys
import time

import numpy as np

from numpilot.backends.base import ExecutionBackend
from numpilot.backends.shim import ShimBackend
from numpilot.contracts import VerificationReport

_TOL = {"float32": 1e-4, "float16": 1e-2}


def _tol_for(dtype) -> float:
    return _TOL.get(np.dtype(dtype).name, 0.0)


def _load_namespace(source: str) -> dict:
    ns = {}
    exec(compile(source, "<migrated>", "exec"), ns)
    return ns


def _time_call(fn, *args, warmup=3, iters=10, **kw):
    for _ in range(warmup):
        fn(*args, **kw)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


class VerificationAgent:
    def __init__(self, backend: ExecutionBackend = None):
        self.backend = backend or ShimBackend()

    def verify(self, orig_source: str, migrated_source: str,
               dtype=np.float32) -> VerificationReport:
        tol = _tol_for(dtype)

        # 原始 numpy 侧
        orig_ns = _load_namespace(orig_source)
        inputs = orig_ns["build_inputs"]()
        out_orig = np.asarray(orig_ns["forward"](*inputs, xp=np))
        t_orig = _time_call(orig_ns["forward"], *inputs, xp=np)

        # 迁移侧:把后端模块挂成 asnumpy 供迁移源码 import
        ap_module = self.backend.get_module()
        saved = sys.modules.get("asnumpy")
        sys.modules["asnumpy"] = ap_module
        try:
            mig_ns = _load_namespace(migrated_source)
            out_mig_raw = mig_ns["forward"](*inputs, xp=ap_module)
            out_mig = np.asarray(
                out_mig_raw.to_numpy() if hasattr(out_mig_raw, "to_numpy") else out_mig_raw
            )
            t_mig = _time_call(mig_ns["forward"], *inputs, xp=ap_module)
        finally:
            if saved is not None:
                sys.modules["asnumpy"] = saved
            else:
                sys.modules.pop("asnumpy", None)

        denom = np.max(np.abs(out_orig))
        max_rel = float(np.max(np.abs(out_orig - out_mig)) / denom) if denom > 0 else 0.0
        consistent = max_rel <= tol
        speedup = float(t_orig / t_mig) if t_mig > 0 else None

        return VerificationReport(
            numerically_consistent=consistent,
            max_rel_diff=max_rel,
            tol=tol,
            speedup=speedup,
            per_op=[],
            status="pass" if consistent else "fail",
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_verification.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/agents/verification.py tests/test_verification.py
git commit -m "feat: 验证智能体(双端执行 + 数值对比 + 测速)"
```

---

## Task 10: 编排器 + CLI + 端到端

**Files:**
- Create: `numpilot/orchestrator.py`, `numpilot/__main__.py`, `tests/test_e2e.py`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Consumes: 全部 Agent + `select_backend`
- Produces:
  - `numpilot.orchestrator.Orchestrator(backend=None, engine=None, threshold=1_000_000).run(source_path: str, out_dir: str | None = None) -> dict`
    - 返回 dict 含 `status`、`report`(`VerificationReport` 或 unsupported 说明)、`analysis`、`plan`、`migrated_path`。
    - 缺口算子:`analysis.migratable is False` → 直接返回 `status="unsupported"`,不迁移不验证。
    - 正常:分析→决策→迁移→(落盘 migrated 文件)→验证,写 `report.md` + `report.json`(若给 out_dir)。
  - `numpilot/__main__.py`:`python -m numpilot migrate <file> [--backend shim|npu] [--engine ast|llm] [--out DIR] [--threshold N]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_e2e.py
from numpilot.orchestrator import Orchestrator

def test_e2e_mlp_migrates_and_verifies():
    res = Orchestrator().run("examples/mlp_forward.py")
    assert res["status"] == "pass"
    assert res["report"].numerically_consistent
    assert "ap.matmul(" in res["migrated_source"]

def test_e2e_unsupported_stops_early():
    res = Orchestrator().run("examples/unsupported_demo.py")
    assert res["status"] == "unsupported"
    assert "fft" in res["unsupported"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_e2e.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 orchestrator.py**

```python
# numpilot/orchestrator.py
"""编排智能体:统筹分析→决策→迁移→验证, 汇总迁移报告。"""
import json
import os

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.base import select_backend
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.llm import LlmEngine


def _make_engine(name):
    if name in (None, "ast"):
        return AstRewriteEngine()
    if name == "llm":
        return LlmEngine()
    raise ValueError(f"未知引擎: {name!r}(可选 ast | llm)")


class Orchestrator:
    def __init__(self, backend=None, engine=None, threshold=1_000_000):
        self.backend = backend if backend is not None else select_backend()
        self.engine = engine if engine is not None else AstRewriteEngine()
        self.threshold = threshold

    def run(self, source_path: str, out_dir=None) -> dict:
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        analysis = AnalysisAgent().analyze(source, source_path)
        if not analysis.migratable:
            result = {
                "status": "unsupported",
                "unsupported": analysis.unsupported,
                "analysis": analysis,
                "source_path": source_path,
            }
            self._maybe_write(out_dir, result)
            return result

        plan = DecisionAgent(self.threshold).plan(analysis)
        migration = MigrationAgent(self.engine).migrate(source, plan)

        migrated_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            migrated_path = os.path.join(out_dir, "migrated.py")
            with open(migrated_path, "w", encoding="utf-8") as f:
                f.write(migration.migrated_source)

        report = VerificationAgent(self.backend).verify(source, migration.migrated_source)
        result = {
            "status": report.status,
            "report": report,
            "analysis": analysis,
            "plan": plan,
            "migrated_source": migration.migrated_source,
            "migrated_path": migrated_path,
            "source_path": source_path,
        }
        self._maybe_write(out_dir, result)
        return result

    @staticmethod
    def _maybe_write(out_dir, result):
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        rep = result.get("report")
        summary = {
            "status": result["status"],
            "source_path": result["source_path"],
            "unsupported": result.get("unsupported", []),
            "numerically_consistent": getattr(rep, "numerically_consistent", None),
            "max_rel_diff": getattr(rep, "max_rel_diff", None),
            "tol": getattr(rep, "tol", None),
            "speedup": getattr(rep, "speedup", None),
        }
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(_render_md(summary, result))


def _render_md(summary, result) -> str:
    lines = ["# NumPilot 迁移报告", ""]
    lines.append(f"- 源文件:`{summary['source_path']}`")
    lines.append(f"- 状态:**{summary['status']}**")
    if summary["status"] == "unsupported":
        lines.append(f"- 缺口算子(未迁移):{', '.join(summary['unsupported'])}")
        return "\n".join(lines) + "\n"
    lines.append(f"- 数值一致:{summary['numerically_consistent']}(max_rel_diff={summary['max_rel_diff']:.2e}, tol={summary['tol']:.0e})")
    if summary["speedup"] is not None:
        lines.append(f"- 加速比(当前后端):{summary['speedup']:.4f}x")
    plan = result.get("plan")
    if plan is not None:
        lines.append("")
        lines.append("## 卸载方案")
        for op, place in sorted(plan.placement.items()):
            lines.append(f"- `{op}` → {place}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: 写 __main__.py**

```python
# numpilot/__main__.py
"""CLI: python -m numpilot migrate <file> [选项]"""
import argparse
import sys

from numpilot.backends.base import select_backend
from numpilot.orchestrator import Orchestrator, _make_engine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="numpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    mig = sub.add_parser("migrate", help="迁移一个 numpy 程序到 AsNumpy(NPU)")
    mig.add_argument("file")
    mig.add_argument("--backend", choices=["shim", "npu"], default=None)
    mig.add_argument("--engine", choices=["ast", "llm"], default=None)
    mig.add_argument("--out", default=None, help="报告与迁移代码输出目录")
    mig.add_argument("--threshold", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    orch = Orchestrator(
        backend=select_backend(args.backend),
        engine=_make_engine(args.engine),
        threshold=args.threshold,
    )
    res = orch.run(args.file, out_dir=args.out)

    print(f"状态: {res['status']}")
    if res["status"] == "unsupported":
        print("缺口算子:", ", ".join(res["unsupported"]))
        return 1
    rep = res["report"]
    print(f"数值一致: {rep.numerically_consistent}  max_rel_diff={rep.max_rel_diff:.2e}  tol={rep.tol:.0e}")
    if rep.speedup is not None:
        print(f"加速比(当前后端): {rep.speedup:.4f}x")
    if res.get("migrated_path"):
        print("迁移代码:", res["migrated_path"])
    return 0 if res["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /home/lixing/numpilot && python -m pytest tests/test_e2e.py -v`
Expected: PASS(2 passed)

- [ ] **Step 6: 跑一次真实 CLI 冒烟**

Run: `cd /home/lixing/numpilot && python -m numpilot migrate examples/mlp_forward.py --out /tmp/numpilot_out && echo "---" && cat /tmp/numpilot_out/report.md`
Expected: 状态 pass、数值一致 True、report.md 含卸载方案

- [ ] **Step 7: 全量测试**

Run: `cd /home/lixing/numpilot && python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 8: 提交**

```bash
cd /home/lixing/numpilot
git add numpilot/orchestrator.py numpilot/__main__.py tests/test_e2e.py
git commit -m "feat: 编排器 + CLI + 端到端流水线"
```

---

## Task 11: 课程架构文档 + README

**Files:**
- Create: `README.md`, `docs/architecture.md`, `docs/quality-scenarios.md`, `docs/adr/ADR-001..004*.md`
- Test:(文档任务,无自动化测试;人工核对)

- [ ] **Step 1: 写 README.md**

内容涵盖:项目简介、与 NumPilot PDF 的关系、一键运行(`python -m numpilot migrate examples/mlp_forward.py`)、架构一图、后端/引擎切换说明、**启用真·LLM 需提供的 3 个环境变量**(`NUMPILOT_LLM_BASE_URL` / `NUMPILOT_LLM_AUTH_TOKEN` / `NUMPILOT_LLM_MODEL`,`PROVIDER` 可选;附 `--engine llm` 用法与盘古/GLM 示例)、目录导航、测试命令。README 须提示凭据仅经环境变量注入、切勿入库。

- [ ] **Step 2: 写 docs/architecture.md**

内容:**4+1 视图**(逻辑视图=组件与契约、进程视图=流水线时序、开发视图=包结构、物理视图=shim/NPU 部署、场景视图=迁移用例走查)+ **C4 三层**(System Context / Container / Component,用 mermaid 或 ASCII)。

- [ ] **Step 3: 写 docs/quality-scenarios.md**

用标准质量属性场景模板(刺激源/刺激/制品/环境/响应/度量)写:可修改性(换 LLM 引擎)、可移植性(shim↔NPU)、正确性(数值校验)、可测试性(契约边界)。

- [ ] **Step 4: 写 4 篇 ADR**

`ADR-001` 策略模式迁移引擎、`ADR-002` 端口-适配器后端、`ADR-003` 契约先行 Agent、`ADR-004` 裁掉缺口算子生成。每篇用标准 ADR 模板(背景/决策/后果/备选)。

- [ ] **Step 5: 提交**

```bash
cd /home/lixing/numpilot
git add README.md docs/architecture.md docs/quality-scenarios.md docs/adr/
git commit -m "docs: 课程架构文档(4+1 + C4 + ADR + 质量属性场景)+ README"
```

---

## Self-Review

**1. Spec coverage:**
- 覆盖集(spec §3)→ Task 1 ✓
- 5 智能体(spec §4.1)→ Analysis T3 / Decision T4 / Migration T6 / Verification T9 / Orchestrator T10 ✓
- 策略引擎(spec §4.2)→ Task 5 ✓;端口后端 → Task 7 ✓
- 契约(spec §5)→ Task 2 ✓
- 错误处理:缺口终止 → T10 e2e ✓;数值不一致 fail → T9 ✓;LLM 回退 → T5 ✓
- 测试策略(spec §7):单元 T1-T9 ✓、e2e T10 ✓、负例 T10 ✓
- 目录结构(spec §8)→ 全任务累计产出 ✓
- 课程交付物(spec §9)→ Task 11 ✓
- ADR(spec §10)→ Task 11 ✓

**2. Placeholder scan:** 无 TBD/TODO 式空洞步骤(llm.py 的 `TODO(接入)` 是有意的真实占位,标注清晰且默认回退可跑)。

**3. Type consistency:** `AnalysisResult/OffloadPlan/MigrationResult/VerificationReport` 字段在 T2 定义,T3/T4/T6/T9/T10 使用一致;`MigrationEngine.rewrite(source, plan)`、`ExecutionBackend.get_module/to_device/to_host`、`select_backend`、`_make_engine` 签名跨任务一致;`forward(..., xp=...)` / `build_inputs()` 在 T8 定义,T9/T10 使用一致。

无遗留问题。
