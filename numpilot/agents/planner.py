"""规划智能体 Planner：把分析结果与缺口清单变成可执行的迁移计划。

The architecture's Migration Plan answers one question before any code is
written: **Direct Migration, or Fill API Gaps?**  This module is the single
place that decides the route, so the orchestrator never has to infer it from
a list of names.

``direct_migration``
    No task requires a *missing API*.  Placement is the whole ``forward`` on
    NPU, delegated to :class:`~numpilot.agents.decision.DecisionAgent`.  Tasks
    routed to the Code Migrator (official aliases, Python dunders, scalar
    bindings, whole-program lowering) stay on this route and are handed to the
    migrator as ``translation_tasks``.
``fill_api_gaps``
    At least one task genuinely needs a capability the runtime does not have
    (``gap_filler``) or an explicit consent decision (``consent_gate``).  The
    plan carries those tasks so the Gap Filler and the reporter can act on
    them.

When the caller supplies the source, the plan also carries a deterministic
:mod:`numpilot.translation_taskbook` entry.  The taskbook is the per-callsite
hand-off to the Code Migrator; the plan remains data only and never generates
or applies code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpilot.agents.decision import DecisionAgent
from numpilot.gap_requirements import build_planning_requirements
from numpilot.translation_taskbook import build_translation_taskbook

ROUTE_DIRECT = "direct_migration"
ROUTE_FILL_GAPS = "fill_api_gaps"


@dataclass(frozen=True)
class MigrationPlan:
    """The planner's route decision and placement for one program."""

    route: str
    placement: dict
    threshold_elements: int
    strategy: str
    gap_tasks: tuple = field(default_factory=tuple)
    planning_requirements: tuple = field(default_factory=tuple)
    translation_tasks: tuple = field(default_factory=tuple)
    translation_taskbook: dict = field(default_factory=dict)

    @property
    def direct(self) -> bool:
        return self.route == ROUTE_DIRECT

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "strategy": self.strategy,
            "threshold_elements": self.threshold_elements,
            "placement": dict(self.placement),
            "gap_tasks": list(self.gap_tasks),
            "planning_requirements": list(self.planning_requirements),
            "translation_tasks": list(self.translation_tasks),
            "translation_taskbook_id": self.translation_taskbook.get("taskbook_id"),
        }


class PlannerAgent:
    """Choose the route, the placement and the translation hand-off."""

    def __init__(self, threshold_elements: int = 1_000_000, decision=None):
        self.decision = decision or DecisionAgent(threshold_elements)

    def plan(
        self,
        analysis,
        gap_plan,
        *,
        source=None,
        prepared_source=None,
        preparation=None,
        source_path=None,
    ) -> MigrationPlan:
        offload = self.decision.plan(analysis)
        blocking = gap_plan.blocking
        translation = tuple(
            task for task in blocking if task.route == "code_migrator"
        )
        capability = tuple(
            task for task in blocking if task.route != "code_migrator"
        )
        # A missing API or a consent gate must stop before code generation.
        # A task routed to the Code Migrator is work for this route, not a
        # reason to skip the migrator.  ``not analysis.migratable`` with no
        # routed task at all remains fail-closed.
        needs_gap_filling = bool(capability) or (
            not analysis.migratable and not translation
        )
        route = ROUTE_FILL_GAPS if needs_gap_filling else ROUTE_DIRECT
        taskbook = {}
        if source is not None:
            taskbook = build_translation_taskbook(
                analysis=analysis,
                route=route,
                source=source,
                prepared_source=prepared_source,
                preparation=preparation,
                gap_plan=gap_plan,
                source_path=source_path,
            )
        return MigrationPlan(
            route=route,
            placement=dict(offload.placement),
            threshold_elements=offload.threshold_elements,
            strategy=offload.strategy,
            gap_tasks=tuple(task.to_dict() for task in blocking),
            planning_requirements=build_planning_requirements(analysis, gap_plan),
            translation_tasks=tuple(task.to_dict() for task in translation),
            translation_taskbook=taskbook,
        )
