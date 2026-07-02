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
