"""决策智能体:单设备边界 MVP,默认将完整 forward 放到 NPU。"""
from numpilot.contracts import AnalysisResult, OffloadPlan


class DecisionAgent:
    """为可迁移 forward 生成统一 NPU 放置,避免逐算子跨设备往返。"""

    def __init__(self, threshold_elements: int = 1_000_000):
        """初始化决策智能体。

        Args:
            threshold_elements: 保留在计划中的兼容字段;单设备 MVP 不按它拆分 forward。
        """
        self.threshold = threshold_elements

    def plan(self, analysis: AnalysisResult) -> OffloadPlan:
        """根据分析结果生成卸载计划。

        规则:
        - 分析阶段确认可迁移后,forward 涉及的算子全部放到 NPU。
        - 不在同一个 forward 内按规模混合 CPU/NPU,避免隐式 H2D/D2H 往返。
        - unsupported 仍由编排器在调用本方法前提前终止。

        Args:
            analysis: 分析结果,包含操作使用列表

        Returns:
            卸载计划,包含每个操作的目标位置(CPU/NPU)
        """
        placement = {op.name: "NPU" for op in analysis.op_uses}

        return OffloadPlan(placement=placement, threshold_elements=self.threshold)
