"""迁移智能体:委托 MigrationEngine 策略完成 np->ap 改写。"""
from numpilot.contracts import MigrationResult, OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine, count_ap_calls
from numpilot.engines.base import MigrationEngine


class MigrationAgent:
    def __init__(self, engine: MigrationEngine = None):
        self.engine = engine or AstRewriteEngine()

    def migrate(self, source: str, plan: OffloadPlan) -> MigrationResult:
        migrated = self.engine.rewrite(source, plan)
        return MigrationResult(
            engine=self.engine.name,
            migrated_source=migrated,
            rewrites=count_ap_calls(migrated),
        )
