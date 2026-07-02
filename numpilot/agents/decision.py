"""决策智能体:规则版成本模型,按张量规模阈值决定 CPU/NPU 卸载。"""
from numpilot.contracts import AnalysisResult, OffloadPlan


class DecisionAgent:
    """规则基成本模型:按操作数规模决定卸载位置。"""

    def __init__(self, threshold_elements: int = 1_000_000):
        """初始化决策智能体。

        Args:
            threshold_elements: 决策阈值,超过此值的操作卸载到NPU。
        """
        self.threshold = threshold_elements

    def plan(self, analysis: AnalysisResult) -> OffloadPlan:
        """根据分析结果生成卸载计划。

        规则:
        - 若某操作任一调用点 est_elements 为 None → "NPU" (规模未知,保守上NPU)
        - 若某操作任一调用点 est_elements >= threshold → "NPU"
        - 否则 → "CPU"

        Args:
            analysis: 分析结果,包含操作使用列表

        Returns:
            卸载计划,包含每个操作的目标位置(CPU/NPU)
        """
        # 将操作按名称分组,收集所有调用点的规模
        by_op = {}
        for op in analysis.op_uses:
            by_op.setdefault(op.name, []).append(op.est_elements)

        # 为每个操作决策位置
        placement = {}
        for name, sizes in by_op.items():
            if any(s is None for s in sizes):
                # 规模未知,保守上NPU
                placement[name] = "NPU"
            elif any(s >= self.threshold for s in sizes):
                # 任一调用点超过阈值,上NPU
                placement[name] = "NPU"
            else:
                # 所有调用点都已知且都小于阈值,留在CPU
                placement[name] = "CPU"

        return OffloadPlan(placement=placement, threshold_elements=self.threshold)
