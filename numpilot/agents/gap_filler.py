"""缺口智能体 API Gap Filler：把静态分析的缺口整理成可执行任务。

This role owns the *inventory* of gaps, not their invention.  Every unique
gap becomes exactly one :class:`GapTask` so the same missing API is never
reported twice, and each task carries the route the mainline will take:

``code_migrator``
    Deterministic lowering exists or can be expressed with existing calls
    (Python operators, scalar materialisation, mechanical aliases).
``consent_gate``
    The NumPy call uses a keyword contract AsNumpy rejects (``out=``,
    ``where=``, ``casting=``).  Rewriting it silently would change semantics,
    so it needs an explicit, recorded decision instead.
``gap_filler``
    The API is genuinely absent.  It stays blocked until a capability is
    supplied and passes the API-correctness gate.

No candidates are generated yet.  The task says what would be needed and
which gate must accept it; producing an artifact without that gate would be
exactly the "name exists therefore capability exists" error the project
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Mechanical renames where NumPy's documented alias has the same contract.
_OFFICIAL_ALIASES = {
    "abs": "absolute",
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atan2": "arctan2",
    "atanh": "arctanh",
    "fabs": "absolute",
    "pow": "power",
    "round_": "around",
    "round": "around",
}

# Known contract divergences that survive name coverage.  These are warnings,
# not blockers: they must be exercised by the API-correctness gate.
_CONTRACT_NOTES = {
    "fmax": "底层为 aclnnMaximum，NaN 传播；NumPy fmax 忽略 NaN",
    "copysign": "丢失有符号零；NumPy 保留 -0.0",
}

_ROUTE_BY_PREFIX = (
    ("python_operator.", "dialect_operator", "code_migrator"),
    ("python_structure.", "dialect_structure", "code_migrator"),
)

_BLOCKING = {"lowering_required", "consent_required", "gap_required"}


@dataclass(frozen=True)
class GapTask:
    """One unique gap with its route and evidence."""

    name: str
    category: str
    route: str
    status: str
    call_sites: tuple = ()
    reason: str = ""
    target: str | None = None

    @property
    def blocking(self) -> bool:
        return self.status in _BLOCKING

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "route": self.route,
            "status": self.status,
            "call_sites": list(self.call_sites),
            "reason": self.reason,
            "target": self.target,
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class GapPlan:
    """The deduplicated gap inventory for one analysis."""

    tasks: tuple = field(default_factory=tuple)

    @property
    def blocking(self) -> tuple:
        return tuple(task for task in self.tasks if task.blocking)

    @property
    def ready(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "n_tasks": len(self.tasks),
            "n_blocking": len(self.blocking),
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _classify(use):
    name = use.name
    for prefix, category, route in _ROUTE_BY_PREFIX:
        if name.startswith(prefix):
            return category, route
    reason = use.reason or ""
    if "未验证 NumPy 关键字" in reason:
        return "keyword_contract", "consent_gate"
    if "绑定" in reason and "数组" in reason:
        return "binding_contract", "code_migrator"
    if "axis" in reason:
        return "axis_contract", "gap_filler"
    return "api_gap", "gap_filler"


def _status_for(route):
    if route == "consent_gate":
        return "consent_required"
    if route == "gap_filler":
        return "gap_required"
    return "lowering_required"


class GapFillerAgent:
    """Turn ``AnalysisResult.op_uses`` into a deduplicated, routed plan."""

    def plan(self, analysis) -> GapPlan:
        grouped = {}
        for use in getattr(analysis, "op_uses", ()) or ():
            if use.supported:
                note = _CONTRACT_NOTES.get(use.name)
                if note:
                    key = ("semantic_divergence", use.name)
                    existing = grouped.get(key)
                    sites = tuple(sorted(set((existing.call_sites if existing else ()) + (use.lineno,))))
                    grouped[key] = GapTask(
                        name=use.name,
                        category="semantic_divergence",
                        route="api_correctness_gate",
                        status="review_required",
                        call_sites=sites,
                        reason=note,
                        target=use.name,
                    )
                continue

            category, route = _classify(use)
            if category == "api_gap" and use.name in _OFFICIAL_ALIASES:
                # A documented alias is a mechanical rename, not a gap.
                category, route = "alias", "code_migrator"
            key = (category, use.name)
            existing = grouped.get(key)
            sites = tuple(sorted(set((existing.call_sites if existing else ()) + (use.lineno,))))
            target = None
            if category in ("api_gap", "alias"):
                target = _OFFICIAL_ALIASES.get(use.name, use.name)
            grouped[key] = GapTask(
                name=use.name,
                category=category,
                route=route,
                status=_status_for(route),
                call_sites=sites,
                reason=use.reason or "",
                target=target,
            )

        tasks = tuple(
            sorted(
                grouped.values(),
                key=lambda task: (task.category, task.name, task.call_sites),
            )
        )
        return GapPlan(tasks=tasks)
