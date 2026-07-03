"""编排智能体:统筹分析→决策→迁移→验证, 汇总迁移报告。"""
import json
import os

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.base import select_backend
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.llm import LlmEngine


def _make_engine(name):
    if name in (None, "ast"):
        return AstRewriteEngine()
    if name == "llm":
        return LlmEngine()
    raise ValueError(f"未知引擎: {name!r}(可选 ast | llm)")


class Orchestrator:
    def __init__(self, backend=None, engine=None, threshold=1_000_000):
        self.backend = backend if backend is not None else select_backend()
        self.engine = engine if engine is not None else AstRewriteEngine()
        self.threshold = threshold

    def run(self, source_path: str, out_dir=None) -> dict:
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        analysis = AnalysisAgent().analyze(source, source_path)
        if not analysis.migratable:
            result = {
                "status": "unsupported",
                "unsupported": analysis.unsupported,
                "analysis": analysis,
                "source_path": source_path,
            }
            self._maybe_write(out_dir, result)
            return result

        plan = DecisionAgent(self.threshold).plan(analysis)
        migration = MigrationAgent(self.engine).migrate(source, plan)

        migrated_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            migrated_path = os.path.join(out_dir, "migrated.py")
            with open(migrated_path, "w", encoding="utf-8") as f:
                f.write(migration.migrated_source)

        report = VerificationAgent(self.backend).verify(source, migration.migrated_source)
        result = {
            "status": report.status,
            "report": report,
            "analysis": analysis,
            "plan": plan,
            "migrated_source": migration.migrated_source,
            "migrated_path": migrated_path,
            "source_path": source_path,
        }
        self._maybe_write(out_dir, result)
        return result

    @staticmethod
    def _maybe_write(out_dir, result):
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        rep = result.get("report")
        summary = {
            "status": result["status"],
            "source_path": result["source_path"],
            "unsupported": result.get("unsupported", []),
            "numerically_consistent": getattr(rep, "numerically_consistent", None),
            "max_rel_diff": getattr(rep, "max_rel_diff", None),
            "tol": getattr(rep, "tol", None),
            "speedup": getattr(rep, "speedup", None),
        }
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(_render_md(summary, result))


def _render_md(summary, result) -> str:
    lines = ["# NumPilot 迁移报告", ""]
    lines.append(f"- 源文件:`{summary['source_path']}`")
    lines.append(f"- 状态:**{summary['status']}**")
    if summary["status"] == "unsupported":
        lines.append(f"- 缺口算子(未迁移):{', '.join(summary['unsupported'])}")
        return "\n".join(lines) + "\n"
    lines.append(f"- 数值一致:{summary['numerically_consistent']}(max_rel_diff={summary['max_rel_diff']:.2e}, tol={summary['tol']:.0e})")
    if summary["speedup"] is not None:
        lines.append(f"- 加速比(当前后端):{summary['speedup']:.4f}x")
    plan = result.get("plan")
    if plan is not None:
        lines.append("")
        lines.append("## 卸载方案")
        for op, place in sorted(plan.placement.items()):
            lines.append(f"- `{op}` → {place}")
    return "\n".join(lines) + "\n"
