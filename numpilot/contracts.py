"""Agent 间传递的数据契约。每个 Agent 只认这里的输入/输出类型。"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpUse:
    name: str                       # numpy 函数名, e.g. "matmul"
    lineno: int
    supported: bool                 # 是否在覆盖集
    est_elements: Optional[int] = None  # 估计张量规模(决策用), 未知为 None
    reason: Optional[str] = None        # capability/signature 不满足时的原因
    col_offset: int = 0
    qualified_name: Optional[str] = None  # e.g. numpy.fft.fft
    call_expr: Optional[str] = None
    positional_args: int = 0
    keyword_args: tuple = ()
    kind: str = "numpy_api"          # "numpy_api" | "domain_operator"
    target_api: Optional[str] = None  # e.g. asnumpy.smooth_l1_loss
    requires_qualification: bool = False
    operator_spec: Optional[dict] = None


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
    strategy: Optional[str] = None

    def __post_init__(self):
        invalid = {
            target for target in self.placement.values() if target not in {"CPU", "NPU"}
        }
        if invalid:
            raise ValueError(f"非法 placement: {sorted(invalid)}")
        if self.strategy is None:
            self.strategy = (
                "whole_forward_npu"
                if self.placement and set(self.placement.values()) == {"NPU"}
                else "legacy_mixed"
            )
        if self.strategy == "whole_forward_npu" and any(
            target != "NPU" for target in self.placement.values()
        ):
            raise ValueError("whole_forward_npu 不能包含 CPU placement")


@dataclass
class MigrationResult:
    engine: str                     # "ast" | "llm"
    migrated_source: str
    rewrites: dict = field(default_factory=dict)  # 迁移后 ap.<op> 调用计数 {op: n}
    expected_rewrites: dict = field(default_factory=dict)  # 原始 forward + plan 推导的期望清单

    @property
    def total_rewrites(self) -> int:
        return sum(self.rewrites.values())

    @property
    def rewrite_integrity(self) -> bool:
        return self.rewrites == self.expected_rewrites


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
    max_abs_diff: float = 0.0
    atol: float = 0.0
    rtol: float = 0.0
    failure_path: Optional[str] = None
    failure_index: Optional[str] = None
    failure_reason: Optional[str] = None
    exception_phase: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    traceback: Optional[str] = None
    artifact_inputs_executed: bool = False
    backend_provenance: dict = field(default_factory=dict)
    input_consistent: Optional[bool] = None
    kernel_consistent: Optional[bool] = None
    artifact_consistent: Optional[bool] = None
    comparisons: dict = field(default_factory=dict)
