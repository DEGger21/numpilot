"""编排智能体:按六角色主链统筹 Understand → Migrate → Validate。

Flow (matches ``docs/assets/architecture/numpilot-architecture-en.png``):

1. **Understanding** — the Analyzer parses the AST and writes a Program IR/DAG
   plus a capability verdict; the Planner turns that into a Migration Plan and
   answers "Direct Migration, or Fill API Gaps?".
2. **Migration** — on the gap route the API Gap Filler inventories and routes
   every unique gap; it stops fail-closed when an artifact would be required.
   On the direct route the Code Migrator rewrites the program.  Failures after
   full integration come back to the Code Migrator through a bounded repair
   loop (at most :data:`MAX_REPAIR_ITERATIONS` attempts).
3. **Validation** — the deterministic verifier is the backbone, not a role.
   The Diagnostics Agent locates a failure and names the role that must act;
   the Migration Reporter collects the evidence, artifacts and output program.

The orchestrator is the coordinator, not one of the six roles.
"""
import hashlib
import json
import os

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.diagnostics import DiagnosticsAgent
from numpilot.agents.gap_filler import GapFillerAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.planner import ROUTE_FILL_GAPS, PlannerAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.base import select_backend
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.llm import LlmEngine
from numpilot.paths import default_benchmark_dir
from numpilot.program_ir import build_program_ir
from numpilot.roles import Role, RoleRecord, RoleStage, records_to_dicts

MAX_REPAIR_ITERATIONS = 3


def _make_engine(name):
    if name in (None, "ast"):
        return AstRewriteEngine()
    if name == "llm":
        # CLI 显式选 llm → strict:失败直接终止, 不回退 AST(用智能体迁移, 失败即暴露)
        return LlmEngine(strict=True)
    raise ValueError(f"未知引擎: {name!r}(可选 ast | llm)")


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _analyzer_record(analysis, program_ir):
    return RoleRecord(
        Role.ANALYZER,
        RoleStage.UNDERSTAND,
        "ok" if analysis.migratable else "blocked",
        (
            f"IR {program_ir.n_nodes} 节点/{program_ir.n_edges} 边；"
            f"{len(analysis.op_uses)} 个调用点，缺口 {len(analysis.unsupported)} 个"
        ),
        {
            "n_calls": len(analysis.op_uses),
            "unsupported": list(analysis.unsupported),
            "program_ir": program_ir.to_dict(),
        },
    )


def _planner_record(plan):
    return RoleRecord(
        Role.PLANNER,
        RoleStage.UNDERSTAND,
        "ok" if plan.direct else "blocked",
        (
            "直迁（Direct Migration）"
            if plan.direct
            else f"需要补缺口（Fill API Gaps）：{len(plan.gap_tasks)} 个任务"
        ),
        plan.to_dict(),
    )


def _gap_record(gap_plan):
    blocking = gap_plan.blocking
    review = [task for task in gap_plan.tasks if task.status == "review_required"]
    if blocking:
        summary = f"{len(gap_plan.tasks)} 个任务，其中 {len(blocking)} 个阻断"
    elif review:
        summary = f"{len(gap_plan.tasks)} 个任务全部可路由；{len(review)} 个契约偏差需门禁复核"
    else:
        summary = f"{len(gap_plan.tasks)} 个任务，无阻断"
    return RoleRecord(
        Role.GAP_FILLER,
        RoleStage.MIGRATE,
        "blocked" if blocking else "ok",
        summary,
        gap_plan.to_dict(),
    )


def _migrator_record(migration, repair_attempts):
    evidence = {
        "engine": migration.engine,
        "rewrites": dict(migration.rewrites),
        "expected_rewrites": dict(migration.expected_rewrites),
        "rewrite_integrity": migration.rewrite_integrity,
        "model_identity": migration.model_identity or {},
        "repair_attempts": repair_attempts,
    }
    return RoleRecord(
        Role.CODE_MIGRATOR,
        RoleStage.MIGRATE,
        "ok" if migration.rewrite_integrity else "failed",
        (
            f"{sum(migration.rewrites.values())} 处改写，"
            f"完整性={'通过' if migration.rewrite_integrity else '失败'}"
            + (f"，有界修复 {repair_attempts} 次" if repair_attempts else "")
        ),
        evidence,
    )


def _validation_evidence(report, source, migration):
    return {
        "status": report.status,
        "numerically_consistent": report.numerically_consistent,
        "input_consistent": report.input_consistent,
        "kernel_consistent": report.kernel_consistent,
        "artifact_consistent": report.artifact_consistent,
        "artifact_inputs_executed": report.artifact_inputs_executed,
        "backend": report.backend,
        "backend_version": report.backend_version,
        "backend_provenance": report.backend_provenance or {},
        "comparisons": report.comparisons or {},
        "source_sha256": _sha256(source),
        "migrated_sha256": _sha256(migration.migrated_source),
    }


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
        return self.run_source(source, source_path, out_dir)

    def run_source(
        self,
        source,
        source_path,
        out_dir=None,
        on_record=None,
        verification_source=None,
        strict_input_types=False,
    ) -> dict:
        """Run the six-role mainline on in-memory source.

        ``on_record`` is an optional observer called with each
        :class:`~numpilot.roles.RoleRecord` as soon as it is produced; the
        wizard uses it for progress, the CLI ignores it.

        ``verification_source`` separates the program used for analysis and
        migration from the NumPy oracle used for numerical verification.  The
        benchmark adapter lowers documented equivalent spellings for analysis
        while keeping the original program as the oracle.  ``source`` remains
        the analysis source; ``verification_source`` defaults to it.
        """
        oracle_source = (
            source if verification_source is None else verification_source
        )
        records = []

        def emit(record):
            records.append(record)
            if on_record is not None:
                on_record(record)

        # ---------------------------------------------------------- Understand
        profile_resolver = (
            self.backend
            if callable(getattr(self.backend, "resolve_profile_capability", None))
            else None
        )
        analysis = AnalysisAgent(profile_resolver).analyze(
            source,
            source_path,
            profiles=self.analysis_profiles,
            strict_input_types=strict_input_types,
        )
        program_ir = build_program_ir(source, source_path)
        gap_plan = GapFillerAgent().plan(analysis)
        plan = PlannerAgent(self.threshold).plan(analysis, gap_plan)
        emit(_analyzer_record(analysis, program_ir))
        emit(_planner_record(plan))

        # ------------------------------------------------------------ Migrate
        if plan.route == ROUTE_FILL_GAPS:
            # The Gap Filler owns these tasks.  No accepted artifact exists
            # yet, so the mainline stops here instead of rewriting around a
            # gap; a silent substitute would be exactly what the project bans.
            emit(_gap_record(gap_plan))
            reporter = RoleRecord(
                Role.REPORTER,
                RoleStage.VALIDATE,
                "ok",
                "已写出缺口报告" if out_dir else "未落盘（无 out_dir）",
                {"unsupported": list(analysis.unsupported)},
            )
            emit(reporter)
            result = {
                "status": "unsupported",
                "route": plan.route,
                "unsupported": analysis.unsupported,
                "analysis": analysis,
                "plan": plan,
                "program_ir": program_ir.to_dict(),
                "gap_plan": gap_plan.to_dict(),
                "source_path": source_path,
                "roles": records_to_dicts(records),
                "diagnosis": None,
            }
            self._maybe_write(out_dir, result)
            return result

        emit(_gap_record(gap_plan))
        migrator = MigrationAgent(self.engine)
        verifier = VerificationAgent(self.backend)
        migration = migrator.migrate(source, plan)
        report = verifier.verify(
            oracle_source,
            migration.migrated_source,
            measure=False,
            required_ops=migration.expected_rewrites,
        )
        diagnostics = DiagnosticsAgent()
        diagnosis = diagnostics.diagnose(report)

        # Bounded repair: only engines that implement ``repair`` can consume a
        # diagnosis; the deterministic AST engine has exactly one attempt.
        repair_attempts = 0
        while (
            report.status != "pass"
            and diagnosis is not None
            and diagnosis.suggested_role == "code_migrator"
            and migrator.supports_repair
            and repair_attempts < MAX_REPAIR_ITERATIONS
        ):
            repair_attempts += 1
            migration = migrator.repair(
                source, plan, migration.migrated_source, diagnosis
            )
            report = verifier.verify(
                oracle_source,
                migration.migrated_source,
                measure=False,
                required_ops=migration.expected_rewrites,
            )
            diagnosis = diagnostics.diagnose(report)

        emit(_migrator_record(migration, repair_attempts))

        # ----------------------------------------------------------- Validate
        if diagnosis is None:
            emit(RoleRecord(
                Role.DIAGNOSTICS,
                RoleStage.VALIDATE,
                "skipped",
                "验证通过，无需诊断",
                {"repair_attempts": repair_attempts},
            ))
        else:
            emit(RoleRecord(
                Role.DIAGNOSTICS,
                RoleStage.VALIDATE,
                "ok",
                diagnosis.summary,
                diagnosis.to_dict(),
            ))

        migrated_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            migrated_path = os.path.join(out_dir, "migrated.py")
            with open(migrated_path, "w", encoding="utf-8") as f:
                f.write(migration.migrated_source)
        evidence = _validation_evidence(report, oracle_source, migration)
        emit(RoleRecord(
            Role.REPORTER,
            RoleStage.VALIDATE,
            "ok",
            "已写出报告、验证证据与迁移产物" if out_dir else "未落盘（无 out_dir）",
            {"status": report.status},
        ))

        result = {
            "status": report.status,
            "route": plan.route,
            "report": report,
            "analysis": analysis,
            "plan": plan,
            "program_ir": program_ir.to_dict(),
            "gap_plan": gap_plan.to_dict(),
            "engine": migration.engine,
            "rewrites": migration.rewrites,
            "expected_rewrites": migration.expected_rewrites,
            "rewrite_integrity": migration.rewrite_integrity,
            "model_receipt": migration.model_receipt,
            "model_identity": migration.model_identity,
            "migrated_source": migration.migrated_source,
            "migrated_path": migrated_path,
            "source_path": source_path,
            "source_sha256": _sha256(oracle_source),
            "analysis_source_sha256": _sha256(source),
            "repair_attempts": repair_attempts,
            "validation_evidence": evidence,
            "roles": records_to_dicts(records, migration.engine),
            "diagnosis": diagnosis.to_dict() if diagnosis is not None else None,
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
            "route": result.get("route"),
            "source_path": result["source_path"],
            "unsupported": result.get("unsupported", []),
            "engine": result.get("engine"),
            "rewrites": rewrites,
            "expected_rewrites": expected_rewrites,
            "rewrite_integrity": result.get("rewrite_integrity"),
            "model_receipt": result.get("model_receipt", {}),
            "model_identity": result.get("model_identity", {}),
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
            "program_ir": result.get("program_ir"),
            "gap_plan": result.get("gap_plan"),
            "diagnosis": result.get("diagnosis"),
            "repair_attempts": result.get("repair_attempts", 0),
            "validation_evidence": result.get("validation_evidence"),
            "roles": result.get("roles", []),
        }
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(_render_md(summary, result))


def _render_roles(lines, roles):
    if not roles:
        return
    lines.append("")
    lines.append("## 六角色执行记录")
    for record in roles:
        lines.append(
            f"- {record['name']}（{record['stage']}，{record['status']}）："
            f"{record['summary']} · 模型={record['model']}"
        )


def _render_md(summary, result) -> str:
    lines = ["# NumPilot 迁移报告", ""]
    lines.append(f"- 源文件:`{summary['source_path']}`")
    lines.append(f"- 状态:**{summary['status']}**")
    lines.append(f"- 路线:`{summary.get('route') or '?'}`")
    roles = summary.get("roles") or []
    if summary["status"] == "unsupported":
        lines.append(f"- 缺口算子(未迁移):{', '.join(summary['unsupported'])}")
        _render_roles(lines, roles)
        gap_plan = summary.get("gap_plan") or {}
        tasks = gap_plan.get("tasks") or []
        if tasks:
            lines.append("")
            lines.append("## 缺口任务")
            for task in tasks:
                lines.append(
                    f"- `{task['name']}`（{task['category']} → {task['route']}，"
                    f"{task['status']}）@L{', '.join(str(n) for n in task['call_sites'])}"
                )
                if task.get("reason"):
                    lines.append(f"  - {task['reason']}")
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

    _render_roles(lines, roles)
    diagnosis = summary.get("diagnosis") or {}
    if diagnosis:
        lines.append("")
        lines.append("## 诊断")
        lines.append(
            f"- 分类:`{diagnosis.get('category')}` · 回退角色:"
            f"`{diagnosis.get('suggested_role')}`"
        )
        lines.append(f"- 结论:{diagnosis.get('summary')}")

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
    bench_dir = default_benchmark_dir()
    lines.append("")
    lines.append("## 性能对比")
    lines.append("性能对比拆成两个独立进程，记录实际 BLAS 线程并避免设置互相污染。"
                 "只有 strict-NPU provenance 通过时才发布加速比:")
    lines.append(f"1. `python -m numpilot bench-cpu {src} --out {bench_dir}`")
    lines.append(f"2. `python -m numpilot bench-npu {src} --out {bench_dir}`")

    plan = result.get("plan")
    if plan is not None:
        lines.append("")
        lines.append("## 卸载方案")
        lines.append(f"- 策略:`{getattr(plan, 'strategy', 'unknown')}`")
        for op, place in sorted(plan.placement.items()):
            lines.append(f"- `{op}` → {place}")
    return "\n".join(lines) + "\n"
