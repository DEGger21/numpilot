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
    def __init__(
        self,
        backend=None,
        engine=None,
        threshold=1_000_000,
        analysis_profiles=None,
    ):
        self.backend = backend if backend is not None else select_backend()
        self.engine = engine if engine is not None else AstRewriteEngine()
        self.threshold = threshold
        self.analysis_profiles = analysis_profiles

    def run(self, source_path: str, out_dir=None) -> dict:
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        profile_resolver = (
            self.backend
            if callable(
                getattr(self.backend, "resolve_profile_capability", None)
            )
            else None
        )
        analysis = AnalysisAgent(profile_resolver).analyze(
            source,
            source_path,
            profiles=self.analysis_profiles,
        )
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
            source,
            migration.migrated_source,
            measure=False,
            required_ops=migration.expected_rewrites,
        )
        result = {
            "status": report.status,
            "report": report,
            "analysis": analysis,
            "plan": plan,
            "engine": migration.engine,
            "rewrites": migration.rewrites,
            "expected_rewrites": migration.expected_rewrites,
            "rewrite_integrity": migration.rewrite_integrity,
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
        expected_rewrites = result.get("expected_rewrites", {}) or {}
        summary = {
            "status": result["status"],
            "source_path": result["source_path"],
            "unsupported": result.get("unsupported", []),
            "engine": result.get("engine"),
            "rewrites": rewrites,
            "expected_rewrites": expected_rewrites,
            "rewrite_integrity": result.get("rewrite_integrity"),
            "total_rewrites": sum(rewrites.values()),
            "numerically_consistent": getattr(rep, "numerically_consistent", None),
            "max_rel_diff": getattr(rep, "max_rel_diff", None),
            "max_abs_diff": getattr(rep, "max_abs_diff", None),
            "tol": getattr(rep, "tol", None),
            "atol": getattr(rep, "atol", None),
            "rtol": getattr(rep, "rtol", None),
            "failure_path": getattr(rep, "failure_path", None),
            "failure_index": getattr(rep, "failure_index", None),
            "failure_reason": getattr(rep, "failure_reason", None),
            "exception_phase": getattr(rep, "exception_phase", None),
            "exception_type": getattr(rep, "exception_type", None),
            "exception_message": getattr(rep, "exception_message", None),
            "artifact_inputs_executed": getattr(rep, "artifact_inputs_executed", False),
            "input_consistent": getattr(rep, "input_consistent", None),
            "kernel_consistent": getattr(rep, "kernel_consistent", None),
            "artifact_consistent": getattr(rep, "artifact_consistent", None),
            "comparisons": getattr(rep, "comparisons", {}),
            "backend_provenance": getattr(rep, "backend_provenance", {}),
            "speedup": getattr(rep, "speedup", None),
            "backend": getattr(rep, "backend", None),
            "backend_module": getattr(rep, "backend_module", None),
            "backend_version": getattr(rep, "backend_version", None),
            "cpu_threads": getattr(rep, "cpu_threads", None),
            "t_orig_ms": getattr(rep, "t_orig_ms", None),
            "t_mig_ms": getattr(rep, "t_mig_ms", None),
            "n_elements": getattr(rep, "n_elements", None),
            "strategy": getattr(result.get("plan"), "strategy", None),
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
    lines.append(f"- 数值一致:{summary['numerically_consistent']}")
    if summary.get("exception_type"):
        lines.append(
            f"- 执行异常:`{summary['exception_phase']}` / "
            f"`{summary['exception_type']}: {summary['exception_message']}`"
        )
    else:
        lines.append(
            "- 差值:"
            f"max_abs={summary['max_abs_diff']:.2e}, "
            f"max_rel={summary['max_rel_diff']:.2e}, "
            f"atol={summary['atol']:.0e}, rtol={summary['rtol']:.0e}"
        )
        if summary.get("failure_reason"):
            location = summary.get("failure_path") or "?"
            if summary.get("failure_index"):
                location += summary["failure_index"]
            lines.append(f"- 首个失败:`{location}`，{summary['failure_reason']}")
    lines.append(
        "- 迁移产物自身 build_inputs 已执行:"
        + ("是" if summary.get("artifact_inputs_executed") else "否")
    )
    comparisons = summary.get("comparisons") or {}
    if comparisons:
        lines.append("- 三层门禁:")
        for key, label in (
            ("inputs", "输入生成"),
            ("kernel", "同一规范输入 forward"),
            ("artifact", "产物 build_inputs + forward"),
        ):
            if key in comparisons:
                lines.append(f"  - {label}:{comparisons[key].get('consistent')}")

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
    provenance = summary.get("backend_provenance") or {}
    if provenance:
        lines.append(f"- capability audit:`{provenance.get('audit_id', '?')}`")
        if provenance.get("strict_npu") is True:
            lines.append(f"- strict-NPU 门禁:{provenance.get('ok')}")
        else:
            lines.append("- strict-NPU 门禁:未执行（shim/非 strict）")
        if provenance.get("missing"):
            lines.append(f"- 缺失 API:{', '.join(provenance['missing'])}")
        if provenance.get("non_strict"):
            lines.append(f"- 非严格 NPU:{', '.join(provenance['non_strict'])}")

    # 性能对比交给两个独立 bench 程序，并由 provenance gate 决定是否可发布。
    src = summary.get("source_path", "<file>")
    lines.append("")
    lines.append("## 性能对比")
    lines.append("性能对比拆成两个独立进程，记录实际 BLAS 线程并避免设置互相污染。"
                 "只有 strict-NPU provenance 通过时才发布加速比:")
    lines.append(f"1. `python -m numpilot bench-cpu {src} --out out_demo`")
    lines.append(f"2. `python -m numpilot bench-npu {src} --out out_demo`")

    plan = result.get("plan")
    if plan is not None:
        lines.append("")
        lines.append("## 卸载方案")
        lines.append(f"- 策略:`{getattr(plan, 'strategy', 'unknown')}`")
        for op, place in sorted(plan.placement.items()):
            lines.append(f"- `{op}` → {place}")
    return "\n".join(lines) + "\n"
