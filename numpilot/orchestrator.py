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
        # CLI 显式选 llm → strict:失败直接终止, 不回退 AST(用智能体迁移, 失败即暴露)
        return LlmEngine(strict=True)
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

        report = VerificationAgent(self.backend).verify(
            source, migration.migrated_source, measure=False)
        result = {
            "status": report.status,
            "report": report,
            "analysis": analysis,
            "plan": plan,
            "engine": migration.engine,
            "rewrites": migration.rewrites,
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
        rewrites = result.get("rewrites", {}) or {}
        summary = {
            "status": result["status"],
            "source_path": result["source_path"],
            "unsupported": result.get("unsupported", []),
            "engine": result.get("engine"),
            "rewrites": rewrites,
            "total_rewrites": sum(rewrites.values()),
            "numerically_consistent": getattr(rep, "numerically_consistent", None),
            "max_rel_diff": getattr(rep, "max_rel_diff", None),
            "tol": getattr(rep, "tol", None),
            "speedup": getattr(rep, "speedup", None),
            "backend": getattr(rep, "backend", None),
            "backend_module": getattr(rep, "backend_module", None),
            "backend_version": getattr(rep, "backend_version", None),
            "cpu_threads": getattr(rep, "cpu_threads", None),
            "t_orig_ms": getattr(rep, "t_orig_ms", None),
            "t_mig_ms": getattr(rep, "t_mig_ms", None),
            "n_elements": getattr(rep, "n_elements", None),
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

    # 替换算子清单(迁移后真实调用的 ap.* 算子)
    rewrites = summary.get("rewrites") or {}
    if rewrites:
        lines.append("")
        lines.append(f"## 替换算子(np.* → ap.*, 引擎={summary.get('engine')})")
        lines.append(f"- 共替换 **{summary.get('total_rewrites', 0)}** 处调用")
        for op, n in sorted(rewrites.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `ap.{op}` ×{n}")

    # 执行证据:证明迁移侧真的用了 asnumpy(而非 numpy)
    backend = summary.get("backend") or "?"
    lines.append("")
    lines.append("## 执行证据")
    lines.append(f"- 后端:**{backend}**")
    mod, ver = summary.get("backend_module"), summary.get("backend_version")
    if mod:
        lines.append(f"- asnumpy 模块:`{mod}`")
    if ver:
        lines.append(f"- asnumpy 版本:`{ver}`(真机 wheel)")
    if not mod and not ver:
        lines.append("- 仿真后端(shim):离线用 numpy 承载 asnumpy 语义,数值恒等,非真机")

    # 性能对比交给两个独立 bench 程序(公平口径:numpy 单核 vs NPU 满线程)
    src = summary.get("source_path", "<file>")
    lines.append("")
    lines.append("## 性能对比")
    lines.append("本机 numpy=scipy-openblas 默认吃满全核(640),与单卡 NPU 不对等。"
                 "故性能对比拆成两个独立进程,避免线程设置互相污染:")
    lines.append(f"1. `python -m numpilot bench-cpu {src} --out out_demo`(numpy 钉 1 核)")
    lines.append(f"2. `python -m numpilot bench-npu {src} --out out_demo`(NPU 满线程,读 1 的基线算加速比)")

    plan = result.get("plan")
    if plan is not None:
        lines.append("")
        lines.append("## 卸载方案")
        for op, place in sorted(plan.placement.items()):
            lines.append(f"- `{op}` → {place}")
    return "\n".join(lines) + "\n"
