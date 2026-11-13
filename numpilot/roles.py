"""六角色主链契约：Understand → Migrate → Validate。

The architecture defines six model-facing roles.  This module is the single
source of truth for their names, stage, duty, and current model policy.  The
orchestrator executes them in :data:`ROLE_ORDER` and records one
:class:`RoleRecord` per role; the deterministic gates (API correctness, full
program validation, ValidationReceipt) are backbone and are deliberately not
roles.

Keeping the roster here means the CLI, wizard, reports, and tests cannot drift
into different agent lists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class RoleStage(str, Enum):
    """The three pipeline stages from the architecture contract."""

    UNDERSTAND = "understand"
    MIGRATE = "migrate"
    VALIDATE = "validate"


class Role(str, Enum):
    """The six model-facing roles."""

    ANALYZER = "analyzer"
    PLANNER = "planner"
    GAP_FILLER = "gap_filler"
    CODE_MIGRATOR = "code_migrator"
    DIAGNOSTICS = "diagnostics"
    REPORTER = "reporter"


@dataclass(frozen=True)
class RoleDefinition:
    """A stable, human-readable role contract."""

    role: Role
    display_name: str
    duty: str
    stage: RoleStage
    uses_configured_model: bool = False

    @property
    def model_label(self) -> str:
        return "确定性规则引擎"


ROLE_DEFINITIONS = (
    RoleDefinition(
        Role.ANALYZER,
        "① 分析智能体 Analyzer",
        "AST 解析 forward · 提取调用契约 · 比对能力注册表",
        RoleStage.UNDERSTAND,
    ),
    RoleDefinition(
        Role.PLANNER,
        "② 规划智能体 Planner",
        "确定主机/设备边界 · 划分 NPU 岛 · 生成改写计划",
        RoleStage.UNDERSTAND,
    ),
    RoleDefinition(
        Role.GAP_FILLER,
        "③ 缺口智能体 API Gap Filler",
        "每个唯一缺口一个任务 · 官方别名/组合/生成候选",
        RoleStage.MIGRATE,
    ),
    RoleDefinition(
        Role.CODE_MIGRATOR,
        "④ 迁移智能体 Code Migrator",
        "np.* → ap.* 改写 · 边界插入 · 方言下降",
        RoleStage.MIGRATE,
        uses_configured_model=True,
    ),
    RoleDefinition(
        Role.DIAGNOSTICS,
        "⑤ 诊断智能体 Diagnostics Agent",
        "结构化定位失败 · 判定回退角色 · 不升级结论",
        RoleStage.VALIDATE,
    ),
    RoleDefinition(
        Role.REPORTER,
        "⑥ 报告智能体 Migration Reporter",
        "汇总证据 · 写报告 · 引用回执而非替代回执",
        RoleStage.VALIDATE,
        uses_configured_model=True,
    ),
)

ROLE_ORDER = tuple(definition.role for definition in ROLE_DEFINITIONS)
_ROLE_BY_NAME = {definition.role: definition for definition in ROLE_DEFINITIONS}


def role_definition(role) -> RoleDefinition:
    return _ROLE_BY_NAME[Role(role)]


def model_label(role) -> str:
    """Return the execution lane this role can use in the current product.

    The Code Migrator and Migration Reporter can run on the configured model
    when the caller selects the LLM engine.  The remaining roles are
    deterministic gates; their labels name that lane without advertising a
    "no model" state in the user-facing roster.
    """
    definition = role_definition(role)
    return "大模型" if definition.uses_configured_model else definition.model_label


def role_model_label(role, engine_name) -> str:
    definition = role_definition(role)
    configured = os.getenv("NUMPILOT_LLM_MODEL", "已配置 LLM")
    if definition.role in {Role.CODE_MIGRATOR, Role.REPORTER}:
        if engine_name == "llm":
            return f"{configured} 大模型"
        return (
            "AST 确定性改写（规则引擎）"
            if definition.role is Role.CODE_MIGRATOR
            else "确定性报告模板"
        )
    return definition.model_label


@dataclass(frozen=True)
class RoleRecord:
    """One role's execution record inside a pipeline run."""

    role: Role
    stage: RoleStage
    status: str  # "ok" | "blocked" | "failed" | "skipped"
    summary: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self, engine_name=None) -> dict:
        definition = role_definition(self.role)
        return {
            "role": self.role.value,
            "name": definition.display_name,
            "stage": self.stage.value,
            "status": self.status,
            "summary": self.summary,
            "model": role_model_label(self.role, engine_name),
            "evidence": self.evidence,
        }


def role_roster(engine_name) -> list:
    """Return the six-role roster as ``(name, duty, model)`` tuples."""
    return [
        (
            definition.display_name,
            definition.duty,
            role_model_label(definition.role, engine_name),
        )
        for definition in ROLE_DEFINITIONS
    ]


def records_to_dicts(records, engine_name=None) -> list:
    """Serialise records in canonical role order, skipping roles not run."""
    by_role = {record.role: record for record in records}
    return [
        by_role[role].to_dict(engine_name)
        for role in ROLE_ORDER
        if role in by_role
    ]
