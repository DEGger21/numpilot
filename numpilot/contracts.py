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
    rewrites: dict = field(default_factory=dict)  # 迁移后 ap.<op> 调用计数 {op: n}

    @property
    def total_rewrites(self) -> int:
        return sum(self.rewrites.values())


@dataclass
class VerificationReport:
    numerically_consistent: bool
    max_rel_diff: float
    tol: float
    speedup: Optional[float]
    per_op: list = field(default_factory=list)
    status: str = "pass"            # "pass" | "fail" | "unsupported"
    backend: str = "shim"           # 迁移侧执行后端
    t_orig_ms: Optional[float] = None   # 原始 numpy@CPU 单次耗时(ms)
    t_mig_ms: Optional[float] = None    # 迁移 asnumpy@后端 单次耗时(ms)
    n_elements: Optional[int] = None    # 主要输出规模(元素数), 供报告展示
    backend_module: Optional[str] = None    # 迁移侧实际命中的 asnumpy 模块路径(__file__), 证明真机
    backend_version: Optional[str] = None   # 该模块 __version__, shim 仿真时为 None
    cpu_threads: Optional[int] = None        # numpy@CPU 基线实际使用的 BLAS 线程数(公平对比口径)
