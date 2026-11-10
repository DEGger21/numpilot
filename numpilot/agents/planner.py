"""规划智能体 Planner：把分析结果与缺口清单变成可执行的迁移计划。

The architecture's Migration Plan answers one question before any code is
written: **Direct Migration, or Fill API Gaps?**  This module is the single
place that decides the route, so the orchestrator never has to infer it from
a list of names.

``direct_migration``
    No blocking gap task exists.  Placement is the whole ``forward`` on NPU,
    delegated to :class:`~numpilot.agents.decision.DecisionAgent`.
``fill_api_gaps``
    At least one gap must be supplied before code can be written.  The plan
    carries those tasks so the Gap Filler and the reporter can act on them.

The plan is data only; it does not generate or apply any code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpilot.agents.decision import DecisionAgent
from numpilot.gap_requirements import build_planning_requirements

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
        }


class PlannerAgent:
    """Choose the route and the placement; never write code."""

    def __init__(self, threshold_elements: int = 1_000_000, decision=None):
        self.decision = decision or DecisionAgent(threshold_elements)

    def plan(self, analysis, gap_plan) -> MigrationPlan:
        offload = self.decision.plan(analysis)
        blocking = gap_plan.blocking
        # ``not analysis.migratable`` is fail-closed even when the gap
        # inventory is empty: the analyzer rejected the program, so the
        # planner must not silently choose the direct route.
        needs_gap_filling = not analysis.migratable or bool(blocking)
        return MigrationPlan(
            route=ROUTE_FILL_GAPS if needs_gap_filling else ROUTE_DIRECT,
            placement=dict(offload.placement),
            threshold_elements=offload.threshold_elements,
            strategy=offload.strategy,
            gap_tasks=tuple(task.to_dict() for task in blocking),
            planning_requirements=build_planning_requirements(analysis, gap_plan),
        )
