"""诊断智能体 Diagnostics Agent：把一次失败定位成可路由的结构化结论。

The verifier reports *what* failed (phase, path, reason).  This role turns
that into *where the fix belongs* without raising confidence: it never marks
a failed run as passing and never invents evidence.  Its output is a
:class:`Diagnosis` consumed by the orchestrator record and the report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnosis:
    """A structured, conservative failure diagnosis."""

    category: str
    stage: str
    suggested_role: str
    summary: str
    failure_path: str | None = None
    exception_phase: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    evidence: dict | None = None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "stage": self.stage,
            "suggested_role": self.suggested_role,
            "summary": self.summary,
            "failure_path": self.failure_path,
            "exception_phase": self.exception_phase,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "evidence": self.evidence or {},
        }


def _phase_route(phase):
    if not phase:
        return None
    if phase.startswith("original."):
        return (
            "source_contract",
            "understand",
            "planner",
            "原始程序不满足执行契约，迁移无法开始",
        )
    if phase == "migrated.surface":
        return (
            "rewrite_surface",
            "migrate",
            "code_migrator",
            "迁移产物不满足 strict forward 执行表面",
        )
    if phase == "migrated.capabilities":
        return (
            "capability_gate",
            "validate",
            "gap_filler",
            "算子未通过能力/来源门禁",
        )
    if phase in ("migrated.load", "migrated.build_inputs"):
        return (
            "artifact_contract",
            "migrate",
            "code_migrator",
            "迁移产物的 build_inputs/模块加载失败",
        )
    if phase in ("migrated.forward", "migrated.runtime_trace"):
        return (
            "runtime_error",
            "validate",
            "code_migrator",
            "迁移产物运行时异常",
        )
    if phase == "compare":
        return (
            "comparison_error",
            "validate",
            "diagnostics_agent",
            "比较阶段异常",
        )
    return ("runtime_error", "validate", "code_migrator", "迁移侧执行失败")


def _divergence_category(reason):
    text = (reason or "").lower()
    if "shape" in text or "形状" in text:
        return "shape_divergence"
    if "dtype" in text or "类型" in text:
        return "dtype_divergence"
    if "nan" in text:
        return "nan_divergence"
    return "numeric_divergence"


class DiagnosticsAgent:
    """Locate the failure and name the role that must act next."""

    def diagnose(self, report) -> Diagnosis | None:
        if report is None:
            return None
        if getattr(report, "status", None) == "pass":
            return None

        phase = getattr(report, "exception_phase", None)
        if phase:
            category, stage, role, summary = _phase_route(phase)
            return Diagnosis(
                category=category,
                stage=stage,
                suggested_role=role,
                summary=f"{summary}（{phase}）",
                exception_phase=phase,
                exception_type=getattr(report, "exception_type", None),
                exception_message=getattr(report, "exception_message", None),
                evidence={
                    "phase": phase,
                    "type": getattr(report, "exception_type", None),
                },
            )

        path = getattr(report, "failure_path", None)
        if path and path.startswith("inputs:"):
            return Diagnosis(
                category="input_divergence",
                stage="validate",
                suggested_role="planner",
                summary="两侧 build_inputs 生成的输入不一致，先修输入契约再谈数值",
                failure_path=path,
                evidence={"comparisons": getattr(report, "comparisons", {})},
            )
        if path and path.startswith("artifact:"):
            return Diagnosis(
                category="artifact_divergence",
                stage="validate",
                suggested_role="code_migrator",
                summary="迁移产物自身 build_inputs + forward 结果不一致",
                failure_path=path,
                evidence={"comparisons": getattr(report, "comparisons", {})},
            )

        reason = getattr(report, "failure_reason", None)
        category = _divergence_category(reason)
        return Diagnosis(
            category=category,
            stage="validate",
            suggested_role="code_migrator",
            summary=f"同一规范输入下数值不一致（{category}）",
            failure_path=path,
            evidence={
                "reason": reason,
                "failure_index": getattr(report, "failure_index", None),
                "max_abs_diff": getattr(report, "max_abs_diff", None),
                "max_rel_diff": getattr(report, "max_rel_diff", None),
            },
        )
