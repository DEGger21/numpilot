"""Gap Filler 的缺口分派与 Diagnostics 的结构化诊断。"""
from types import SimpleNamespace

from numpilot.agents.diagnostics import DiagnosticsAgent
from numpilot.agents.gap_filler import GapFillerAgent
from numpilot.agents.planner import (ROUTE_DIRECT, ROUTE_FILL_GAPS,
                                     PlannerAgent)
from numpilot.contracts import AnalysisResult, OpUse


def _analysis(*uses, migratable=False):
    unsupported = list(dict.fromkeys(use.name for use in uses if not use.supported))
    return AnalysisResult(
        source_path="case.py",
        op_uses=list(uses),
        unsupported=unsupported,
        migratable=migratable,
    )


def test_python_operators_are_deduplicated_into_one_lowering_task():
    plan = GapFillerAgent().plan(
        _analysis(
            OpUse(name="python_operator.Lt", lineno=7, supported=False),
            OpUse(name="python_operator.Lt", lineno=9, supported=False),
            OpUse(name="python_operator.Eq", lineno=11, supported=False),
        )
    )
    assert [task.name for task in plan.tasks] == [
        "python_operator.Eq",
        "python_operator.Lt",
    ]
    lt = plan.tasks[1]
    assert lt.category == "dialect_operator"
    assert lt.route == "code_migrator"
    assert lt.call_sites == (7, 9)
    assert plan.ready is False


def test_rejected_keyword_needs_consent_not_silent_rewrite():
    plan = GapFillerAgent().plan(
        _analysis(
            OpUse(
                name="add",
                lineno=4,
                supported=False,
                reason="AsNumpy 未验证 NumPy 关键字: out",
            )
        )
    )
    task = plan.tasks[0]
    assert (task.category, task.route, task.status) == (
        "keyword_contract",
        "consent_gate",
        "consent_required",
    )


def test_official_alias_is_routed_to_the_migrator():
    plan = GapFillerAgent().plan(
        _analysis(OpUse(name="abs", lineno=2, supported=False))
    )
    task = plan.tasks[0]
    assert task.category == "alias"
    assert task.route == "code_migrator"
    assert task.target == "absolute"


def test_known_contract_divergence_is_surfaced_without_blocking():
    plan = GapFillerAgent().plan(
        _analysis(OpUse(name="fmax", lineno=3, supported=True), migratable=True)
    )
    assert plan.ready is True
    assert plan.tasks[0].category == "semantic_divergence"
    assert plan.tasks[0].status == "review_required"
    assert "NaN" in plan.tasks[0].reason


def test_planner_routes_direct_when_no_gap_blocks():
    analysis = _analysis(
        OpUse(name="sqrt", lineno=3, supported=True), migratable=True
    )
    plan = PlannerAgent().plan(analysis, GapFillerAgent().plan(analysis))
    assert plan.route == ROUTE_DIRECT
    assert plan.placement == {"sqrt": "NPU"}


def test_planner_routes_to_gap_filling_when_a_gap_blocks():
    analysis = _analysis(OpUse(name="fft", lineno=3, supported=False))
    plan = PlannerAgent().plan(analysis, GapFillerAgent().plan(analysis))
    assert plan.route == ROUTE_FILL_GAPS
    assert [task["name"] for task in plan.gap_tasks] == ["fft"]


def test_planner_fails_closed_when_analysis_is_not_migratable():
    analysis = _analysis(
        OpUse(name="sqrt", lineno=3, supported=True), migratable=False
    )
    plan = PlannerAgent().plan(analysis, GapFillerAgent().plan(analysis))
    assert plan.route == ROUTE_FILL_GAPS


def test_diagnostics_maps_runtime_phase_to_code_migrator():
    report = SimpleNamespace(
        status="fail",
        exception_phase="migrated.forward",
        exception_type="TypeError",
        exception_message="boom",
    )
    diagnosis = DiagnosticsAgent().diagnose(report)
    assert diagnosis.category == "runtime_error"
    assert diagnosis.suggested_role == "code_migrator"


def test_diagnostics_maps_capability_gate_to_gap_filler():
    report = SimpleNamespace(
        status="fail",
        exception_phase="migrated.capabilities",
        exception_type="RuntimeError",
        exception_message="audit",
    )
    assert DiagnosticsAgent().diagnose(report).suggested_role == "gap_filler"


def test_diagnostics_reports_input_divergence_back_to_planner():
    report = SimpleNamespace(
        status="fail",
        exception_phase=None,
        failure_path="inputs:out",
        failure_reason="value mismatch",
        failure_index=None,
        max_abs_diff=1.0,
        max_rel_diff=1.0,
        comparisons={},
    )
    diagnosis = DiagnosticsAgent().diagnose(report)
    assert (diagnosis.category, diagnosis.suggested_role) == (
        "input_divergence",
        "planner",
    )


def test_diagnostics_is_silent_on_pass_and_absent_report():
    agent = DiagnosticsAgent()
    assert agent.diagnose(SimpleNamespace(status="pass")) is None
    assert agent.diagnose(None) is None
