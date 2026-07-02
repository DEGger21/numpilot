"""决策智能体测试。"""
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
