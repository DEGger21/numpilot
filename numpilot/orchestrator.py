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
from numpilot.agents.gap_filler import GapFillerAgent, collect_gap_inventory
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.planner import ROUTE_FILL_GAPS, PlannerAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.base import select_backend
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.engines.llm import LlmEngine
from numpilot.paths import default_benchmark_dir
from numpilot.program_ir import build_program_ir
from numpilot.program_preparation import prepare_program
from numpilot.progress import ProgressEmitter, notify_observer
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

    def run(
        self,
        source_path: str,
        out_dir=None,
        *,
        on_record=None,
        on_event=None,
        strict_input_types=False,
    ) -> dict:
        with open(source_path, encoding="utf-8") as f:
            source = f.read()
        return self.run_source(
            source,
            source_path,
            out_dir,
            on_record=on_record,
            on_event=on_event,
            strict_input_types=strict_input_types,
        )

    def run_source(
        self,
        source,
        source_path,
        out_dir=None,
        on_record=None,
        verification_source=None,
        strict_input_types=False,
        on_event=None,
        prepare_source=None,
    ) -> dict:
        """Run the six-role mainline on in-memory source.

        ``on_record`` is an optional observer called with each
        :class:`~numpilot.roles.RoleRecord` as soon as it is produced; the
        existing final role records remain unchanged. ``on_event`` observes
        immutable lifecycle events before and after actual operations, including
        the deterministic verification gate (which is not a seventh role).
        Observer display failures do not change migration or validation results.

        ``verification_source`` separates the program used for analysis and
        migration from the NumPy oracle used for numerical verification.  The
        benchmark adapter lowers documented equivalent spellings for analysis
        while keeping the original program as the oracle.  ``source`` remains
        the analysis source; ``verification_source`` defaults to it.

        ``prepare_source`` runs deterministic whole-program preparation
        (helper inlining, static control-flow unrolling, destructuring and
        documented equivalent spellings) before analysis.  The original
        ``source`` stays the oracle.  ``None`` means "prepare unless the caller
        already supplied analysis profiles", because profiles are bound to the
        exact analyzed source positions.  Callers that already prepared a
        source pass ``False`` so the same transformation is not applied twice.
        """
        if prepare_source is None:
            prepare_source = self.analysis_profiles is None
        oracle_source = (
            source if verification_source is None else verification_source
        )
        preparation = prepare_program(source) if prepare_source else None
        analysis_source = (
            preparation.analysis_source if preparation is not None else source
        )
        records = []
        progress = ProgressEmitter(on_event)
        engine_mode = "llm" if getattr(self.engine, "name", None) == "llm" else "static"

        def emit(record):
            records.append(record)
            notify_observer(on_record, record)

        # ---------------------------------------------------------- Understand
        profile_resolver = (
            self.backend
            if callable(getattr(self.backend, "resolve_profile_capability", None))
            else None
        )
        with progress.operation(Role.ANALYZER, "分析源码与 API 调用契约") as activity:
            analysis = AnalysisAgent(profile_resolver).analyze(
                analysis_source, source_path, profiles=self.analysis_profiles,
                strict_input_types=strict_input_types,
            )
            program_ir = build_program_ir(analysis_source, source_path)
            gap_inventory = collect_gap_inventory(analysis)
            activity.finish("completed" if analysis.migratable else "blocked",
                            "分析完成" if analysis.migratable else "分析完成，存在待处理的 API 缺口")
        with progress.operation(Role.PLANNER, "制定迁移路线与逐调用点任务书") as activity:
            plan = PlannerAgent(self.threshold).plan(
                analysis,
                gap_inventory,
                source=source,
                prepared_source=analysis_source,
                preparation=preparation,
                source_path=source_path,
            )
            activity.finish("completed" if plan.direct else "blocked",
                            "迁移计划与任务书已生成" if plan.direct else "计划要求先补全 API 缺口")
        # Recheck the analysis inventory after planning. This small deterministic
        # pass is actual Gap Filler work, not a reordered progress notification.
        with progress.operation(Role.GAP_FILLER, "核对规划任务、缺口清单与可用接入路径") as activity:
            gap_plan = GapFillerAgent().plan(analysis)
            if gap_plan != gap_inventory:
                raise RuntimeError("gap inventory changed after planning")
            activity.finish("blocked" if gap_plan.blocking else "completed",
                            "缺口需要补全，当前迁移暂停" if gap_plan.blocking else "缺口路径核对完成")
        emit(_analyzer_record(analysis, program_ir))
        emit(_planner_record(plan))

        # ------------------------------------------------------------ Migrate
        if plan.route == ROUTE_FILL_GAPS:
            # The Gap Filler owns these tasks.  No accepted artifact exists
            # yet, so the mainline stops here instead of rewriting around a
            # gap; a silent substitute would be exactly what the project bans.
            emit(_gap_record(gap_plan))
            progress.emit(Role.CODE_MIGRATOR, "skipped", "缺口未补全，尚未改写代码", execution_mode=engine_mode)
            progress.emit(None, "skipped", "尚无可验证的迁移产物")
            progress.emit(Role.DIAGNOSTICS, "skipped", "缺口已明确，尚未进入运行诊断")
            reporter = RoleRecord(
                Role.REPORTER,
                RoleStage.VALIDATE,
                "ok",
                "已写出缺口报告" if out_dir else "未落盘（无 out_dir）",
                {"unsupported": list(analysis.unsupported)},
            )
            result = {
                "status": "unsupported",
                "route": plan.route,
                "unsupported": analysis.unsupported,
                "analysis": analysis,
                "plan": plan,
                "program_ir": program_ir.to_dict(),
                "gap_plan": gap_plan.to_dict(),
                "source_path": source_path,
                "source_sha256": _sha256(oracle_source),
                "analysis_source_sha256": _sha256(analysis_source),
                "preparation": preparation.to_dict() if preparation is not None else None,
                "roles": records_to_dicts(records + [reporter]),
                "diagnosis": None,
            }
            with progress.operation(Role.REPORTER, "汇总缺口报告与执行记录") as activity:
                if out_dir:
                    from numpilot.gap_requirements import build_planning_taskbook, persist_gap_taskbook
                    from numpilot.translation_taskbook import persist_translation_taskbook

                    taskbook = build_planning_taskbook(analysis, gap_plan, source=analysis_source)
                    result["gap_taskbook"] = persist_gap_taskbook(out_dir, taskbook)
                    if plan.translation_taskbook:
                        result["translation_taskbook"] = persist_translation_taskbook(
                            out_dir, plan.translation_taskbook
                        )
                self._maybe_write(out_dir, result)
                emit(reporter)
                activity.finish(message="缺口报告已写入文件" if out_dir else "缺口报告已汇总，未指定输出目录")
            return result

        emit(_gap_record(gap_plan))
        with progress.operation(Role.CODE_MIGRATOR, "生成迁移代码并核对改写完整性", execution_mode=engine_mode) as activity:
            migrator = MigrationAgent(self.engine)
            migration = migrator.migrate(analysis_source, plan)
            activity.finish("completed" if migration.rewrite_integrity else "failed",
                            "迁移代码已生成" if migration.rewrite_integrity else "迁移改写完整性未通过")

        verifier = None
        diagnostics = None

        def verify_candidate(candidate, attempt):
            nonlocal verifier
            with progress.operation(None, "执行输入、数值与能力门禁", attempt=attempt) as activity:
                if verifier is None:
                    verifier = VerificationAgent(self.backend)
                report = verifier.verify(
                    oracle_source, candidate.migrated_source, measure=False,
                    required_ops=candidate.expected_rewrites,
                )
                activity.finish("completed" if report.status == "pass" else "failed",
                                "确定性验证通过" if report.status == "pass" else "确定性验证未通过")
                return report

        def diagnose_report(report, attempt):
            nonlocal diagnostics
            with progress.operation(Role.DIAGNOSTICS, "检查验证结果与失败位置", attempt=attempt) as activity:
                if diagnostics is None:
                    diagnostics = DiagnosticsAgent()
                diagnosis = diagnostics.diagnose(report)
                activity.finish("skipped" if diagnosis is None else "completed",
                                "验证通过，无需诊断" if diagnosis is None else "诊断完成，已确定后续处理角色")
                return diagnosis

        report = verify_candidate(migration, 1)
        diagnosis = diagnose_report(report, 1)

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
            with progress.operation(Role.CODE_MIGRATOR, "根据诊断执行有界代码修复",
                                    execution_mode=engine_mode, attempt=repair_attempts + 1) as activity:
                migration = migrator.repair(
                    analysis_source, plan, migration.migrated_source, diagnosis
                )
                activity.finish("completed" if migration.rewrite_integrity else "failed",
                                "修复代码已生成，等待重新验证" if migration.rewrite_integrity else "修复改写完整性未通过")
            report = verify_candidate(migration, repair_attempts + 1)
            diagnosis = diagnose_report(report, repair_attempts + 1)

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

        reporter = RoleRecord(
            Role.REPORTER,
            RoleStage.VALIDATE,
            "ok",
            "已写出报告、验证证据与迁移产物" if out_dir else "未落盘（无 out_dir）",
            {"status": report.status},
        )

        with progress.operation(Role.REPORTER, "写入迁移代码、验证证据与报告") as activity:
            migrated_path = None
            if out_dir:
                from numpilot.translation_taskbook import persist_translation_taskbook

                os.makedirs(out_dir, exist_ok=True)
                migrated_path = os.path.join(out_dir, "migrated.py")
                with open(migrated_path, "w", encoding="utf-8") as f:
                    f.write(migration.migrated_source)
                if plan.translation_taskbook:
                    result_taskbook = persist_translation_taskbook(
                        out_dir, plan.translation_taskbook
                    )
                else:
                    result_taskbook = None
            else:
                result_taskbook = None
            evidence = _validation_evidence(report, oracle_source, migration)
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
                "analysis_source_sha256": _sha256(analysis_source),
                "preparation": preparation.to_dict() if preparation is not None else None,
                "translation_taskbook": result_taskbook,
                "repair_attempts": repair_attempts,
                "validation_evidence": evidence,
                "roles": records_to_dicts(records + [reporter], migration.engine),
                "diagnosis": diagnosis.to_dict() if diagnosis is not None else None,
            }
            self._maybe_write(out_dir, result)
            emit(reporter)
            activity.finish(message="迁移产物与报告已写入文件" if out_dir else "执行报告已汇总，未指定输出目录")
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
            "source_sha256": result.get("source_sha256"),
            "analysis_source_sha256": result.get("analysis_source_sha256"),
            "preparation": result.get("preparation"),
            "translation_taskbook": result.get("translation_taskbook"),
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
            "planning_requirements": getattr(result.get("plan"), "planning_requirements", ()),
            "gap_taskbook": result.get("gap_taskbook"),
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
    translation = summary.get("translation_taskbook") or {}
    if translation:
        lines.append(
            f"- 迁移任务书:`{translation['taskbook_id']}` "
            f"(`{translation['json']['path']}`)"
        )
    roles = summary.get("roles") or []
    if summary["status"] == "unsupported":
        lines.append(f"- 缺口算子(未迁移):{', '.join(summary['unsupported'])}")
        _render_roles(lines, roles)
        gap_plan = summary.get("gap_plan") or {}
        taskbook = summary.get("gap_taskbook") or {}
        if taskbook:
            lines.append("")
            lines.append("## 缺口任务书")
            lines.append(f"- JSON: `{taskbook['json']['path']}`")
            lines.append(f"- Markdown: `{taskbook['markdown']['path']}`")
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
