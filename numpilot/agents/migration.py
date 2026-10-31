"""迁移智能体:委托 MigrationEngine 策略完成 np->ap 改写。"""
from numpilot.contracts import MigrationResult, OffloadPlan
from numpilot.engines.ast_rewrite import (
    AstRewriteEngine,
    count_ap_calls,
    expected_forward_ap_calls,
)
from numpilot.engines.base import MigrationEngine


class MigrationAgent:
    def __init__(self, engine: MigrationEngine = None):
        self.engine = engine or AstRewriteEngine()

    def _result(self, source, plan, migrated) -> MigrationResult:
        return MigrationResult(
            engine=self.engine.name,
            migrated_source=migrated,
            rewrites=count_ap_calls(migrated),
            expected_rewrites=expected_forward_ap_calls(source, plan),
            model_receipt=dict(getattr(self.engine, "last_receipt", {}) or {}),
            model_identity=dict(getattr(self.engine, "last_identity", {}) or {}),
        )

    def migrate(self, source: str, plan: OffloadPlan) -> MigrationResult:
        return self._result(source, plan, self.engine.rewrite(source, plan))

    @property
    def supports_repair(self) -> bool:
        """Whether the engine can consume a diagnosis and produce a candidate."""
        return callable(getattr(self.engine, "repair", None))

    def repair(self, source, plan, previous_source, diagnosis) -> MigrationResult:
        """Apply one bounded repair attempt; raises when the engine has none.

        The engine owns the strategy; this agent only rebuilds the deterministic
        accounting (rewrites, expected rewrites, receipts) around the result so
        a repaired candidate is judged by exactly the same integrity rules.
        """
        repair = getattr(self.engine, "repair", None)
        if not callable(repair):
            raise RuntimeError("迁移引擎未实现 repair，不能执行有界修复")
        repaired = repair(source, plan, previous_source, diagnosis)
        return self._result(source, plan, repaired)
