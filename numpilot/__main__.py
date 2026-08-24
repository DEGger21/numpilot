"""CLI: python -m numpilot <migrate|check-suite|complete-gaps|setup-skills|bench-cpu|bench-npu|doctor|configure-llm|paths> ...

演示三步:
  1) bench-cpu <file>   纯 NumPy 基线，默认请求 BLAS 1 线程并记录实际线程池
  2) bench-npu <file>   审计迁移算子执行来源；仅严格 NPU 结果可与 1) 计算加速比
  migrate <file>        迁移 + 正确性 + "真用了 asnumpy" 叙事(不做性能对比)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


def parse_cpu_threads(val):
    """1(默认, 单核) / 0|all|max|full(全核, 不限制)→ int 或 None。"""
    if val is None:
        return 1
    if str(val).lower() in ("0", "all", "max", "full"):
        return None
    n = int(val)
    return n if n > 0 else None


def _early_pin_for_bench_cpu():
    """bench-cpu 是纯 numpy 进程(不碰 NPU/CANN),可安全地全局钉核。

    必须在 import numpy 之前设置线程环境变量；它们只是请求值，实际生效线程数
    由 benchmark 在 NumPy 加载后通过 threadpoolctl 观测并写入证据。
    bench-npu / migrate 不走这里，故只对 bench-cpu 生效。bench-npu 会另行记录
    自身进程中真实可观测的 BLAS 线程池，避免把“未限制”误写成“全线程”。
    """
    argv = sys.argv[1:]
    if not argv or argv[0] != "bench-cpu":
        return
    n = "1"
    if "--cpu-threads" in argv:
        i = argv.index("--cpu-threads")
        n = argv[i + 1] if i + 1 < len(argv) else "1"
    if parse_cpu_threads(n) is None:
        return                            # 0/all → 不限制
    n = str(parse_cpu_threads(n))
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = n


_early_pin_for_bench_cpu()                # 在下面 import numpy(经 numpilot 传递)前完成

from numpilot.backends.base import select_backend            # noqa: E402
from numpilot.asnumpy_skill import (                         # noqa: E402
    ASNUMPY_INTEGRATION_SKILL_FILES,
    AsnumpyIntegrationSkillError,
    load_asnumpy_integration_skill,
)
from numpilot.bench import run_cpu_baseline, run_npu_compare  # noqa: E402
from numpilot.capabilities import load_analysis_profile_set    # noqa: E402
from numpilot.cannbot_skills import (                          # noqa: E402
    CANNBOT_GENERATION_SKILL_FILES,
    CANNBOT_PRECISION_SKILL_FILES,
    CANNBOT_REPAIR_SKILL_FILES,
    CANNBOT_RUNTIME_SKILL_FILES,
    CannbotSkillError,
    resolve_skill_bundle,
)
from numpilot.coverage import is_custom                       # noqa: E402
from numpilot.engines.llm import (LlmEngine, LlmError,        # noqa: E402
                                  LlmHttpError, LlmResponseError,
                                  LlmTransportError)
from numpilot.env_check import Env                            # noqa: E402
from numpilot.llm_config import (LlmConfigError,               # noqa: E402
                                 configure_llm_interactive,
                                 init_llm_config_template,
                                 load_llm_config)
from numpilot.orchestrator import Orchestrator, _make_engine  # noqa: E402
from numpilot.paths import (default_benchmark_dir, default_gap_dir,  # noqa: E402
                            path_summary)
from numpilot.report import save_migration_report            # noqa: E402


class RuntimeActivationConfigError(ValueError):
    """Raised for an incomplete or unsafe explicit qualification activation."""


def _qualified_runtime_inputs(args, backend_name):
    """Build explicit overlay/profile inputs without environment discovery."""

    overlay_path = getattr(args, "capability_overlay", None)
    evidence_root = getattr(args, "qualification_evidence_root", None)
    profiles_path = getattr(args, "analysis_profiles", None)
    if bool(overlay_path) != bool(evidence_root):
        raise RuntimeActivationConfigError(
            "--capability-overlay 与 --qualification-evidence-root 必须成对提供"
        )
    if backend_name != "npu" and any(
        (overlay_path, evidence_root, profiles_path)
    ):
        raise RuntimeActivationConfigError(
            "qualification overlay/profile 只允许用于 --backend npu"
        )
    profiles = None
    if profiles_path:
        try:
            profiles = load_analysis_profile_set(profiles_path)
        except (OSError, ValueError) as exc:
            raise RuntimeActivationConfigError(
                "analysis profile 加载失败: %s" % exc
            ) from exc
    if backend_name != "npu":
        return select_backend(backend_name), profiles
    if not overlay_path:
        return select_backend("npu"), profiles

    from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig

    return (
        NpuBackend(
            overlay_config=OverlayRuntimeConfig(
                overlay_path=overlay_path,
                evidence_root=evidence_root,
            )
        ),
        profiles,
    )


def _net_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _render_terminal(res):
    """把迁移结果渲染成分段叙事:分析→迁移→卸载→执行→校验→性能→产物。"""
    analysis = res.get("analysis")
    net = _net_name(res.get("source_path", "<?>"))
    out = []
    out.append(f"── NumPilot 迁移 · {net} " + "─" * max(4, 34 - len(net)))

    # [分析] NumPy 调用点 + 显式 OperatorSpec 调用点 / 去重算子数
    if analysis is not None:
        ops = analysis.op_uses
        distinct = sorted({o.name for o in ops})
        domain_count = sum(o.kind == "domain_operator" for o in ops)
        numpy_count = len(ops) - domain_count
        call_summary = f"{numpy_count} 处 NumPy 调用"
        if domain_count:
            call_summary += f" + {domain_count} 处 OperatorSpec 调用"
        if res["status"] == "unsupported":
            out.append(f"[分析] {call_summary} · {len(distinct)} 个算子 · "
                       f"命中 {len(res['unsupported'])} 个阻断项 ✗")
            out.append("[缺口] " + ", ".join(res["unsupported"]) + "  (无法迁移)")
            return "\n".join(out)
        suffix = (
            "NumPy API 已覆盖；领域算子已显式映射（strict-NPU 仍需资格化） ✓"
            if domain_count else "全部在 asnumpy 覆盖集 ✓"
        )
        out.append(f"[分析] {call_summary} · {len(distinct)} 个算子 · {suffix}")

    # [迁移] 引擎 + 替换计数
    rewrites = res.get("rewrites") or {}
    total = sum(rewrites.values())
    domain_count = sum(
        op.kind == "domain_operator"
        for op in (analysis.op_uses if analysis is not None else [])
    )
    mapping = "np.*/OperatorSpec helper → ap.*" if domain_count else "np.* → ap.*"
    out.append(f"[迁移] 引擎={res.get('engine')} · 替换 {total} 处 {mapping}")
    if rewrites:
        items = "  ".join(f"{op}×{n}" for op, n in
                          sorted(rewrites.items(), key=lambda kv: (-kv[1], kv[0])))
        out.append("       " + items)

    # [卸载] CPU/NPU 分布
    plan = res.get("plan")
    if plan is not None and plan.placement:
        npu = sorted(o for o, p in plan.placement.items() if p == "NPU")
        cpu = sorted(o for o, p in plan.placement.items() if p == "CPU")
        out.append(
            f"[卸载] 策略={plan.strategy} · NPU: {', '.join(npu) or '—'}   "
            f"CPU: {', '.join(cpu) or '—'}"
        )

    rep = res["report"]

    # [执行] 模块命中不等于 strict NPU；必须同时展示 capability provenance。
    if rep.backend_version:
        loc = rep.backend_module or "?"
        out.append(f"[执行] 后端={rep.backend} · 已加载 asnumpy {rep.backend_version}")
        out.append(f"       {loc}")
    else:
        out.append(f"[执行] 后端={rep.backend} · shim 仿真(numpy 承载 asnumpy 语义,数值恒等)")
    provenance = rep.backend_provenance or {}
    if provenance:
        if provenance.get("strict_npu"):
            gate = "通过 ✓" if provenance.get("ok") else "失败 ✗"
        else:
            gate = "未执行(shim/非 strict)"
        out.append(
            f"       capability audit={provenance.get('audit_id', '?')} · "
            f"strict-NPU={gate}"
        )

    # [校验] 数值一致性
    mark = "✓" if rep.numerically_consistent else "✗"
    if rep.exception_type:
        out.append(
            f"[校验] 失败 ✗  {rep.exception_phase}: "
            f"{rep.exception_type}: {rep.exception_message}"
        )
    else:
        out.append(
            f"[校验] 数值一致 {mark}  max_abs={rep.max_abs_diff:.2e}  "
            f"max_norm={rep.max_rel_diff:.2e}  "
            f"(atol={rep.atol:.0e}, rtol={rep.rtol:.0e})"
        )
        out.append(
            "       同输入计算主体=%s · 产物输入=%s · 产物端到端=%s"
            % (rep.kernel_consistent, rep.input_consistent, rep.artifact_consistent)
        )

    # [产物]
    arts = [res["migrated_path"]] if res.get("migrated_path") else []
    if arts:
        out.append("[产物] " + " · ".join(arts))

    # [性能] 交给两个独立 bench 程序(公平口径:numpy 单核 vs NPU 满线程)
    src = res.get("source_path", "<file>")
    bench_dir = default_benchmark_dir()
    out.append("[性能] 本命令不发布加速比；请运行独立、fail-closed 的两步评测:")
    out.append(f"       ① python -m numpilot bench-cpu {src} --out {bench_dir}")
    out.append(f"       ② python -m numpilot bench-npu {src} --out {bench_dir}")
    return "\n".join(out)


def _resolve_backend_name(args):
    return args.backend or os.getenv("NUMPILOT_BACKEND", "shim")


def _cmd_doctor(args):
    rep = Env.check(backend=_resolve_backend_name(args))
    print(rep.render())
    return 1 if rep.errors else 0


def _cmd_paths(args):
    paths = path_summary()
    if args.get:
        print(paths[args.get])
    elif args.json:
        print(json.dumps(paths, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("NumPilot 路径（本命令不会创建目录）:")
        for name, path in paths.items():
            print(f"  {name:14} {path}")
    return 0


def _cmd_check_suite(args):
    """Statically admit a proposed migration suite without executing it."""

    from numpilot.suite_check import (
        SuiteCheckError,
        check_suite,
        write_suite_report,
    )

    try:
        report = check_suite(
            args.paths,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
            expected_direct=args.expect_direct,
            expected_single_gap=args.expect_single_gap,
        )
        if args.out:
            write_suite_report(args.out, report)
    except SuiteCheckError as exc:
        print("[测试集准入] 失败: %s" % exc, file=sys.stderr)
        return 2

    counts = report["counts"]
    print(
        "[测试集准入] status=%s total=%d direct=%d single_gap=%d blocked=%d"
        % (
            report["status"],
            len(report["cases"]),
            counts["direct_migration_candidate"],
            counts["single_gap_candidate"],
            counts["blocked"],
        )
    )
    for case in report["cases"]:
        suffix = ""
        if case["category"] == "single_gap_candidate":
            suffix = " gap=" + case["generatable_gaps"][0]["qualified_api"]
        print(
            "[%s] %s%s"
            % (case["category"], case["path"], suffix)
        )
        for error in case.get("errors", ()):
            print("  - %s" % error, file=sys.stderr)
    for error in report["expectation_errors"]:
        print("[测试集期望] %s" % error, file=sys.stderr)
    if args.out:
        print("[测试集准入] report=%s" % Path(args.out).absolute())
    print(
        "[测试集准入] execution_performed=no runtime_capability_checked=no；"
        "通过仅表示静态候选合格，真机仍须精确 wheel provenance 与数值门禁。"
    )
    return 0 if report["status"] == "pass" else 1


def _load_llm_or_report():
    """Load framework-managed credentials; never prompt from non-wizard commands."""
    try:
        status = load_llm_config()
    except LlmConfigError as exc:
        print(f"[LLM] 本机配置无效: {exc}", file=sys.stderr)
        print("[LLM] 请运行交互配置，或用 configure-llm --init 生成文件后手填。", file=sys.stderr)
        return None
    if not status.configured:
        print(f"[LLM] {status.summary()}", file=sys.stderr)
        print("[LLM] 请运行交互配置，或用 configure-llm --init 生成文件后手填。", file=sys.stderr)
        return None
    return status


_LLM_READY_MARKER = "NUMPILOT_LLM_READY"
_LLM_PROBE_SYSTEM = (
    "You are a connectivity probe. Return one short acknowledgement."
)
_LLM_PROBE_QUESTION = "Reply with NUMPILOT_LLM_READY."


def _run_llm_readiness_gate(status, *, require_marker):
    """Make and verify one small remote call before operator generation."""

    content, receipt = LlmEngine._chat_with_receipt(
        _LLM_PROBE_SYSTEM,
        _LLM_PROBE_QUESTION,
    )
    expected_hash = (
        hashlib.sha256(content.encode("utf-8")).hexdigest()
        if isinstance(content, str) and content.strip()
        else None
    )
    http_status = receipt.get("http_status") if isinstance(receipt, dict) else None
    receipt_is_valid = (
        expected_hash is not None
        and receipt.get("schema_version") == 1
        and isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 200 <= http_status <= 299
        and receipt.get("requested_model") == status.model
        and receipt.get("content_sha256") == expected_hash
        and receipt.get("tls_verification") in {"enabled", "disabled"}
    ) if isinstance(receipt, dict) else False
    marker_present = (
        isinstance(content, str) and _LLM_READY_MARKER in content
    )
    if not receipt_is_valid or (require_marker and not marker_present):
        raise LlmResponseError("readiness_gate_validation_failed", receipt or {})
    safe_receipt_fields = (
        "schema_version", "utc_start", "utc_end", "http_status",
        "requested_model", "response_model", "response_id",
        "response_created", "request_body_sha256", "response_body_sha256",
        "content_sha256", "request_id", "usage", "tls_verification",
    )
    return {
        "schema_version": 1,
        "gate": "llm_readiness",
        "status": "pass",
        "actual_remote_call": True,
        "question_sha256": hashlib.sha256(
            _LLM_PROBE_QUESTION.encode("utf-8")
        ).hexdigest(),
        "answer_marker": _LLM_READY_MARKER,
        "answer_marker_present": marker_present,
        "receipt": {
            name: receipt.get(name)
            for name in safe_receipt_fields
            if name in receipt
        },
    }


def _run_cannbot_skill_gate(bundle, required_files=None):
    """Prove that the external CANNBot workflow files were read."""

    required = tuple(required_files or CANNBOT_GENERATION_SKILL_FILES)
    if not required or len(set(required)) != len(required):
        raise CannbotSkillError("CANNBot gate has an invalid required file set")
    manifest = bundle.manifest()
    manifest_files = manifest.get("files") or {}
    verified_hashes = {}
    for relative in required:
        content = bundle.files.get(relative)
        if not isinstance(content, str) or not content.strip():
            raise CannbotSkillError(
                "required CANNBot workflow file is empty or unreadable: %s"
                % relative
            )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if manifest_files.get(relative) != content_hash:
            raise CannbotSkillError(
                "required CANNBot workflow file hash mismatch: %s" % relative
            )
        verified_hashes[relative] = content_hash
    rendered = bundle.prompt_context(required)
    if (
        bundle.bundle_sha256 not in rendered
        or any(bundle.files[path] not in rendered for path in required)
    ):
        raise CannbotSkillError(
            "required CANNBot workflow was not embedded in prompt context"
        )
    return {
        "schema_version": 1,
        "gate": "cannbot_skill_readiness",
        "status": "pass",
        "repository": manifest.get("repository"),
        "commit": bundle.commit,
        "skill_name": bundle.skill_name,
        "primary_file": required[0],
        "primary_file_sha256": verified_hashes[required[0]],
        "required_skill_files": list(required),
        "required_skill_files_sha256": verified_hashes,
        "bundle_sha256": bundle.bundle_sha256,
        "verified_file_count": len(bundle.files),
        "prompt_context_sha256": hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest(),
        "content_embedded_for_remote_model": True,
    }


def _run_asnumpy_skill_gate(bundle, required_files=None):
    """Prove that NumPilot's repository-owned integration skill was read."""

    required = tuple(required_files or ASNUMPY_INTEGRATION_SKILL_FILES)
    if not required or len(set(required)) != len(required):
        raise AsnumpyIntegrationSkillError(
            "AsNumpy integration gate has an invalid required file set"
        )
    manifest = bundle.manifest()
    manifest_files = manifest.get("files") or {}
    verified_hashes = {}
    for relative in required:
        content = bundle.files.get(relative)
        if not isinstance(content, str) or not content.strip():
            raise AsnumpyIntegrationSkillError(
                "required AsNumpy integration skill file is unreadable: %s"
                % relative
            )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if manifest_files.get(relative) != digest:
            raise AsnumpyIntegrationSkillError(
                "required AsNumpy integration skill hash mismatch: %s" % relative
            )
        verified_hashes[relative] = digest
    rendered = bundle.prompt_context(required)
    if (
        bundle.bundle_sha256 not in rendered
        or any(bundle.files[path] not in rendered for path in required)
    ):
        raise AsnumpyIntegrationSkillError(
            "AsNumpy integration skill was not embedded in prompt context"
        )
    return {
        "schema_version": 1,
        "gate": "asnumpy_integration_skill_readiness",
        "status": "pass",
        "source": manifest.get("source"),
        "skill_name": bundle.skill_name,
        "required_skill_files": list(required),
        "required_skill_files_sha256": verified_hashes,
        "bundle_sha256": bundle.bundle_sha256,
        "verified_file_count": len(bundle.files),
        "prompt_context_sha256": hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest(),
        "content_embedded_for_remote_model": True,
    }


def _write_generation_preflight(
    out_dir,
    llm_gate,
    skill_gate,
    timeout_seconds,
    operator_goal=None,
):
    """Atomically save both successful gates without secrets or response text."""

    root = Path(out_dir).absolute()
    if os.path.lexists(root) and (root.is_symlink() or not root.is_dir()):
        raise OSError("generation output is not a real directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    skill_gate_name = skill_gate.get("gate")
    if skill_gate_name not in (
        "cannbot_skill_readiness",
        "asnumpy_integration_skill_readiness",
    ):
        raise OSError("generation skill gate identity is invalid")
    preflight = {
            "schema_version": 1,
            "producer": "numpilot-generation-preflight-v1",
            "status": "pass",
            "required_order": ["llm_readiness", skill_gate_name],
            "client_timeout_seconds": timeout_seconds,
            "gates": {
                "llm_readiness": llm_gate,
                skill_gate_name: skill_gate,
            },
        }
    if operator_goal is not None:
        preflight["operator_goal"] = operator_goal
        preflight["performance_optimization"] = (
            "deferred_until_accuracy_and_user_confirmation"
            if operator_goal == "performance"
            else "skipped_accuracy_first"
        )
    payload = (
        json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=".generation-preflight-", suffix=".tmp", dir=str(root)
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / "generation-preflight.json")
        temporary = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _create_output_stage(out_dir, label):
    """Create a private sibling stage without claiming the final output path."""

    final = Path(out_dir).absolute()
    if os.path.lexists(final):
        raise OSError("%s output already exists" % label)
    final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", final.name)[:80] or "output"
    return Path(tempfile.mkdtemp(
        prefix=".%s-%s-stage-" % (safe_name, label),
        dir=str(final.parent),
    ))


def _publish_output_stage(stage, out_dir, label):
    """Publish validated staged entries without replacing any existing path."""

    stage = Path(stage)
    final = Path(out_dir).absolute()
    # mkdir is the atomic no-replace reservation.  We intentionally do not use
    # os.replace here: on POSIX it can replace an existing empty user directory.
    os.mkdir(final, mode=0o700)
    try:
        for source in sorted(stage.iterdir(), key=lambda item: item.name):
            os.rename(source, final / source.name)
        os.rmdir(stage)
    except Exception as exc:
        # The directory was exclusively created by this process, but another
        # process may have written into it since then.  Preserve it rather than
        # recursively deleting data we cannot prove we own.
        raise OSError(
            "%s publication was interrupted; retained partial output at %s"
            % (label, final)
        ) from exc


def _resolve_operator_goal(args):
    """Choose the implemented accuracy lane or the reserved performance lane."""

    goal = getattr(args, "operator_goal", None)
    if goal is None and sys.stdin.isatty() and sys.stdout.isatty():
        print("[算子目标] 请选择补全目标：")
        print("  1) 精度正确（推荐）：通过编译、真机运行和数值精度后即可使用")
        print("  2) 高性能（尚未开放）：当前会安全拒绝且不会联网或写输出")
        answer = input("输入 1 或 2 [默认 1]: ").strip()
        goal = "performance" if answer == "2" else "accuracy"
    elif goal is None:
        goal = "accuracy"
        print(
            "[算子目标] 非交互运行，默认 accuracy：先保证精度正确，"
            "跳过性能优化。"
        )
    if goal == "performance":
        print(
            "[算子目标] performance 尚未开放：当前版本不会启动模型、"
            "读取 Skill 或写入输出。",
            file=sys.stderr,
        )
    else:
        print(
            "[算子目标] accuracy：本轮以语义、编译、真机运行和数值精度正确为准；"
            "不执行性能优化。"
        )
    return goal


def _cmd_configure_llm(args):
    """Framework-owned setup; token is hidden or read from stdin, never argv."""
    if args.init:
        conflicting = (
            any(value is not None for value in (args.base_url, args.model, args.provider))
            or args.token_stdin
        )
        if conflicting:
            print(
                "[LLM] --init 只生成空白模板，不能与 --base-url、--model、"
                "--provider 或 --token-stdin 同时使用。",
                file=sys.stderr,
            )
            return 2
        try:
            path, created = init_llm_config_template()
        except LlmConfigError as exc:
            print(f"[LLM] 模板生成失败: {exc}", file=sys.stderr)
            return 2
        if created:
            print(f"LLM 配置模板已生成: {path} (权限 0600)")
            print("请打开该 JSON 文件，填写 base_url、auth_token 和 model。")
        else:
            print(f"LLM 配置目标已存在，未覆盖: {path}")
        print("填写后运行: python -m numpilot configure-llm --check")
        print("确认真实调用: python -m numpilot configure-llm --probe")
        return 0

    if args.check:
        status = _load_llm_or_report()
        if status is None:
            return 1
        print(status.summary())
        print(f"配置路径: {status.path}")
        return 0

    if args.probe:
        status = _load_llm_or_report()
        if status is None:
            print(
                "[LLM probe] 失败: 本地配置未通过，未发起远端请求。",
                file=sys.stderr,
            )
            return 1
        try:
            gate = _run_llm_readiness_gate(status, require_marker=False)
        except LlmResponseError as exc:
            if "readiness_gate_validation_failed" in str(exc):
                print(
                    "[LLM probe] 失败: 远端响应与本地回执校验不一致，"
                    "不能证明真实模型调用成功。",
                    file=sys.stderr,
                )
            else:
                print(
                    "[LLM probe] 失败: 未成功获得可验证的远端模型响应: %s"
                    % exc,
                    file=sys.stderr,
                )
            return 4
        except (LlmHttpError, LlmTransportError) as exc:
            print(
                "[LLM probe] 失败: 未成功获得可验证的远端模型响应: %s"
                % exc,
                file=sys.stderr,
            )
            return 4
        except Exception as exc:
            print(
                "[LLM probe] 失败: 本地请求准备或未知错误: %s"
                % type(exc).__name__,
                file=sys.stderr,
            )
            return 4
        receipt = gate["receipt"]
        usage = receipt.get("usage") or {}
        print(
            "[LLM probe] 成功: actual_remote_call=yes http_status=%s "
            "tls_verification=%s"
            % (receipt.get("http_status"), receipt.get("tls_verification"))
        )
        print(
            "[LLM probe] requested_model=%s response_model=%s "
            "total_tokens=%s content_sha256=%s"
            % (
                receipt.get("requested_model"),
                receipt.get("response_model") or "unknown",
                usage.get("total_tokens", "unknown"),
                receipt["content_sha256"],
            )
        )
        return 0

    token = None
    if args.token_stdin:
        if not args.base_url or not args.model:
            print("[LLM] --token-stdin 必须同时提供 --base-url 和 --model", file=sys.stderr)
            return 2
        token = sys.stdin.readline().rstrip("\r\n")
    elif not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "[LLM] 当前不是交互终端，拒绝等待凭据输入。CI 请使用 NUMPILOT_LLM_* "
            "环境变量，或运行 configure-llm --init 生成文件后手填。",
            file=sys.stderr,
        )
        return 2

    try:
        status = configure_llm_interactive(
            base_url=args.base_url,
            model=args.model,
            provider=args.provider,
            auth_token=token,
        )
    except (EOFError, KeyboardInterrupt):
        print("\n[LLM] 已取消，未写入配置。", file=sys.stderr)
        return 130
    except LlmConfigError as exc:
        print(f"[LLM] 配置失败: {exc}", file=sys.stderr)
        return 2
    print(status.summary())
    print(f"配置已安全保存: {status.path} (权限 0600)")
    return 0


def _cmd_setup_skills(args):
    """Fetch or verify the pinned official CANNBot registry-invoke skill."""

    skill_root = args.root or "auto"
    bundle = resolve_skill_bundle(skill_root, allow_fetch=not args.check)
    action = "verified" if args.check or args.root else "ready"
    print(
        "[CANNBot] %s skill=%s commit=%s"
        % (action, bundle.skill_name, bundle.commit)
    )
    print(
        "[CANNBot] files=%d bytes=%d bundle_sha256=%s"
        % (len(bundle.files), bundle.total_bytes, bundle.bundle_sha256)
    )
    print("[CANNBot] checkout=%s" % bundle.root)
    return 0


def _cmd_complete_gaps(args):
    """Prepare gap specs or ask the configured LLM for an unverified draft."""
    from numpilot.gaps.pipeline import GapCompletionError, GapCompletionPipeline
    from numpilot.gaps.provider import LlmDraftProvider

    configuration_status = None
    llm_gate = None
    timeout_seconds = None
    operator_goal = None
    if args.provider == "llm":
        operator_goal = _resolve_operator_goal(args)
        if operator_goal == "performance":
            return 2
        raw_timeout = (
            str(args.llm_timeout)
            if args.llm_timeout is not None
            else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
        )
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError:
            print("[生成门禁 1/2] NUMPILOT_LLM_TIMEOUT 必须是秒数", file=sys.stderr)
            return 2
        if not 10 <= timeout_seconds <= 3600:
            print("[生成门禁 1/2] LLM timeout 必须在 10～3600 秒之间", file=sys.stderr)
            return 2
        os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
        configuration_status = _load_llm_or_report()
        if configuration_status is None:
            print("[生成门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
            return 4
        try:
            llm_gate = _run_llm_readiness_gate(
                configuration_status, require_marker=True
            )
        except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
            print(
                "[生成门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
                file=sys.stderr,
            )
            print(
                "[算子生成] actual_remote_generation_call=no · 未读取 Skill，"
                "未开始算子生成",
                file=sys.stderr,
            )
            return 4
        except Exception as exc:
            print(
                "[生成门禁 1/2] fail · 简单问答本地错误: %s"
                % type(exc).__name__,
                file=sys.stderr,
            )
            return 4
        probe_receipt = llm_gate["receipt"]
        print(
            "[生成门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · "
            "timeout=%ss"
            % (
                probe_receipt.get("http_status"),
                probe_receipt.get("response_model") or configuration_status.model,
                timeout_seconds,
            )
        )

    # This checkout is independent from the NumPilot source tree.  Auto mode
    # fetches the pinned official commit only when the user cache is absent.
    bundle = resolve_skill_bundle(
        args.skill_root,
        allow_fetch=not args.offline_skills,
    )
    skill_gate = _run_cannbot_skill_gate(
        bundle, required_files=CANNBOT_GENERATION_SKILL_FILES
    )
    if args.provider == "llm":
        print(
            "[生成门禁 2/2] pass · 已读取 CANNBot Skill · commit=%s · "
            "files=%s · bundle_sha256=%s"
            % (bundle.commit, len(bundle.files), bundle.bundle_sha256)
        )
        try:
            _write_generation_preflight(
                args.out,
                llm_gate,
                skill_gate,
                timeout_seconds,
                operator_goal=operator_goal,
            )
        except OSError as exc:
            print(
                "[生成门禁] 无法落盘门禁收据: %s" % type(exc).__name__,
                file=sys.stderr,
            )
            return 2
    print(
        "[CANNBot] 使用官方 skill=%s commit=%s bundle_sha256=%s"
        % (bundle.skill_name, bundle.commit, bundle.bundle_sha256)
    )
    provider = None
    if args.provider == "llm":
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
            operator_goal=operator_goal,
        )

    pipeline = GapCompletionPipeline(
        provider=provider,
        skill_root=bundle.root,
        asnumpy_reference=args.asnumpy_reference,
        skill_bundle=bundle,
    )
    try:
        result = pipeline.run(
            args.file,
            args.out,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
        )
        if provider is not None and result.drafts:
            report_path = os.path.join(result.out_dir, "gap-report.json")
            if os.path.islink(report_path) or not os.path.isfile(report_path):
                raise GapCompletionError("gap report is not a regular file")
            with open(report_path, "rb") as handle:
                report_payload = handle.read(8 * 1024 * 1024 + 1)
            if len(report_payload) > 8 * 1024 * 1024:
                raise GapCompletionError("gap report exceeds the size limit")
            report = json.loads(report_payload.decode("utf-8"))
            actions = {
                draft.gap_id: provider.action_for(draft)
                for draft in result.drafts
            }
            report_drafts = report.get("drafts")
            if not isinstance(report_drafts, list):
                raise GapCompletionError("gap report draft list is invalid")
            seen = set()
            for draft_data in report_drafts:
                if not isinstance(draft_data, dict):
                    raise GapCompletionError("gap report draft entry is invalid")
                gap_id = draft_data.get("gap_id")
                if gap_id not in actions:
                    raise GapCompletionError("gap report draft identity mismatch")
                action = actions[gap_id]
                draft_data["current_run_action"] = action
                draft_data["current_run_actual_remote_call"] = (
                    action == "fresh_remote"
                )
                seen.add(gap_id)
            if seen != set(actions):
                raise GapCompletionError("gap report is missing a generated draft")
            report["current_run_actions"] = dict(sorted(actions.items()))
            report["current_run_actual_remote_call_count"] = sum(
                action == "fresh_remote" for action in actions.values()
            )
            payload = (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            fd = -1
            temp_name = ""
            try:
                fd, temp_name = tempfile.mkstemp(
                    prefix=".gap-report-", suffix=".tmp", dir=result.out_dir
                )
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    fd = -1
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, report_path)
                temp_name = ""
                directory_fd = os.open(result.out_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if fd >= 0:
                    os.close(fd)
                if temp_name:
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass
    except GapCompletionError as exc:
        print(f"[算子补全] 失败: {exc}", file=sys.stderr)
        if args.provider == "llm":
            print(
                "[算子生成] verified_new_remote_artifact=no；本次没有获得经远端回执"
                "绑定且通过本地校验的新模型产物（请求可能未发出、远端失败，或响应"
                "校验失败），并且未回退到 AST。",
                file=sys.stderr,
            )
        return 2

    print(f"[算子补全] status={result.status}")
    print(f"[算子补全] report={os.path.join(result.out_dir, 'gap-report.json')}")
    for spec in result.specs:
        workspace = os.path.join(result.out_dir, "gaps", spec.gap_id)
        print(f"[缺口] {spec.qualified_api} -> {workspace}")
    for item in result.blocked:
        print(
            "[阻断] {name}: {reason}".format(
                name=item.get("qualified_api") or item.get("name", "?"),
                reason=item.get("reason", "unknown"),
            ),
            file=sys.stderr,
        )
    if result.drafts:
        print(
            "[安全门禁] 已生成 Ascend C 草稿，但尚未编译、单测、数值校验或激活；"
            "当前不会注册到 AsNumpy。"
        )
        for draft in result.drafts:
            evidence = draft.generation
            hashes = evidence.get("response_sha256") or []
            action = provider.action_for(draft) if provider is not None else "unknown"
            print(
                "[模型证据] action={action} mode={mode} provider={provider} model={model} "
                "attempts={attempts} fallback={fallback} response_sha256={digest}".format(
                    action=action,
                    mode=evidence.get("mode", "?"),
                    provider=evidence.get("provider", "?"),
                    model=evidence.get("model", "?"),
                    attempts=evidence.get("attempt_count", "?"),
                    fallback=evidence.get("fallback", "?"),
                    digest=hashes[-1] if hashes else "?",
                )
            )
            if action == "fresh_remote" and evidence.get("mode") == "remote_llm":
                receipts = evidence.get("remote_receipts") or []
                statuses = ",".join(
                    str(receipt.get("http_status", "?")) for receipt in receipts
                ) or "?"
                print(
                    "[算子生成] actual_remote_generation_call=yes "
                    "verified_new_remote_artifact=yes receipts={count} "
                    "http_statuses={statuses} content_hash_bound=yes".format(
                        count=len(receipts),
                        statuses=statuses,
                    )
                )
            elif action == "reused_existing":
                print(
                    "[算子生成] actual_remote_generation_call=no "
                    "action=reused_existing；"
                    "本次只复验并复用已有产物，没有重新请求大模型。"
                )
            else:
                print(
                    "[算子生成] actual_remote_generation_call=no action={action}；"
                    "本次没有发起真实远端模型调用。".format(action=action)
                )
    elif provider is not None:
        print(
            "[算子生成] actual_remote_generation_call=no；"
            "本次没有可生成的合格缺口，"
            "因此没有请求大模型。"
        )
    return 1 if result.blocked else 0


def _cmd_repair_draft(args):
    """Repair selected signed-draft files into a new unverified candidate."""

    from pathlib import Path

    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
    )

    def read_bounded(path, label, limit):
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise DraftGenerationError("%s must be a regular file" % label)
        payload = target.read_bytes()
        if len(payload) > limit:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        return payload.decode("utf-8")

    raw_timeout = (
        str(args.llm_timeout)
        if args.llm_timeout is not None
        else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        print("[修复门禁 1/2] NUMPILOT_LLM_TIMEOUT 必须是秒数", file=sys.stderr)
        return 2
    if not 10 <= timeout_seconds <= 3600:
        print("[修复门禁 1/2] LLM timeout 必须在 10～3600 秒之间", file=sys.stderr)
        return 2
    os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
    configuration_status = _load_llm_or_report()
    if configuration_status is None:
        print("[修复门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
        return 4
    try:
        llm_gate = _run_llm_readiness_gate(
            configuration_status, require_marker=True
        )
    except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
        print(
            "[修复门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
            file=sys.stderr,
        )
        print(
            "[算子修复] actual_remote_repair_call=no · 未读取 Skill，"
            "未开始模型修复",
            file=sys.stderr,
        )
        return 4
    probe_receipt = llm_gate["receipt"]
    print(
        "[修复门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · "
        "timeout=%ss"
        % (
            probe_receipt.get("http_status"),
            probe_receipt.get("response_model") or configuration_status.model,
            timeout_seconds,
        )
    )

    # Repair is code generation too: it may not read the external Skill until
    # the model-readiness call above has passed, and it may not contact the
    # model for repair until both gates have passed.
    bundle = resolve_skill_bundle(
        args.skill_root,
        allow_fetch=not args.offline_skills,
    )
    repair_skill_files = {
        "compile": CANNBOT_REPAIR_SKILL_FILES,
        "precision": CANNBOT_PRECISION_SKILL_FILES,
        "runtime": CANNBOT_RUNTIME_SKILL_FILES,
    }[args.repair_profile]
    skill_gate = _run_cannbot_skill_gate(
        bundle, required_files=repair_skill_files
    )
    print(
        "[修复门禁 2/2] pass · 已读取 CANNBot Skill · commit=%s · "
        "files=%s · bundle_sha256=%s"
        % (bundle.commit, len(bundle.files), bundle.bundle_sha256)
    )
    preflight_dir = Path(str(Path(args.out).absolute()) + ".preflight")
    try:
        _write_generation_preflight(
            preflight_dir, llm_gate, skill_gate, timeout_seconds
        )
    except OSError as exc:
        print(
            "[修复门禁] 无法落盘门禁收据: %s" % type(exc).__name__,
            file=sys.stderr,
        )
        return 2
    try:
        spec_data = json.loads(read_bounded(
            args.spec, "gap spec", 2 * 1024 * 1024
        ))
        if not isinstance(spec_data, dict):
            raise DraftGenerationError("gap spec must be a JSON object")
        spec = GapSpec.from_dict(spec_data)
        request = read_bounded(
            args.generation_request,
            "generation request",
            2 * 1024 * 1024,
        )
        diagnostic = read_bounded(
            args.diagnostic_file,
            "repair diagnostic",
            64 * 1024,
        )
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
            repair_skill_files=repair_skill_files,
        )
        report = provider.repair_to_candidate(
            spec,
            Path(args.draft),
            Path(args.out),
            paths=tuple(args.path),
            diagnostic=diagnostic,
            generation_request=request,
            base_candidate_dir=(
                Path(args.base_candidate) if args.base_candidate else None
            ),
        )
    except DraftRepairError as exc:
        print("[LLM repair] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=%s verified_new_remote_repair=no "
            "successful_http_receipts=%d；远端修复响应未通过本地校验。"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[LLM repair] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] verified_new_remote_repair=no；未生成可发布修复副本。",
            file=sys.stderr,
        )
        return 2

    print(
        "[LLM repair] status=%s source_artifact_id=%s out=%s"
        % (report["status"], report["source_artifact_id"], args.out)
    )
    print("[修复门禁] receipt=%s" % (preflight_dir / "generation-preflight.json"))
    print(
        "[LLM调用] actual_remote_call=yes verified_new_remote_repair=yes "
        "attempts=%d receipts=%d content_hash_bound=yes"
        % (report["attempt_count"], len(report["remote_receipts"]))
    )
    return 0


def _cmd_seal_repair(args):
    """Seal a verified repair chain without making another model call."""

    from pathlib import Path

    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.provider import DraftGenerationError, LlmDraftProvider

    try:
        spec_path = Path(args.spec)
        if spec_path.is_symlink() or not spec_path.is_file():
            raise DraftGenerationError("gap spec must be a regular file")
        payload = spec_path.read_bytes()
        if len(payload) > 2 * 1024 * 1024:
            raise DraftGenerationError("gap spec exceeds the size limit")
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise DraftGenerationError("gap spec must be a JSON object")
        # Sealing is deliberately offline.  The original repair report is
        # bound to a verified CANNBot bundle, so reloading that same pinned
        # bundle is part of verification but can never trigger a fetch here.
        bundle = resolve_skill_bundle(args.skill_root, allow_fetch=False)
        draft = LlmDraftProvider(skill_bundle=bundle).seal_repair_candidate(
            GapSpec.from_dict(data),
            Path(args.draft),
            Path(args.candidate),
            Path(args.out),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            CannbotSkillError, DraftGenerationError) as exc:
        print("[seal repair] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=no sealed_artifact=no",
            file=sys.stderr,
        )
        return 2
    print(
        "[seal repair] status=%s artifact_id=%s files=%d out=%s"
        % (draft.status, draft.artifact_id, len(draft.files), draft.workspace)
    )
    print(
        "[LLM调用] actual_remote_call=no action=sealed_repair_chain "
        "prior_receipts=%d"
        % len(draft.generation["remote_receipts"])
    )
    return 0


def _cmd_generate_asnumpy_adapter(args):
    """Generate an isolated AsNumpy manifest/test overlay with the remote model."""

    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
        _load_existing_draft,
    )

    def read_bounded(path, label, limit):
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise DraftGenerationError("%s must be a regular file" % label)
        payload = target.read_bytes()
        if len(payload) > limit:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        return payload.decode("utf-8")

    out_root = Path(args.out).absolute()
    if os.path.lexists(out_root):
        print("[AsNumpy 接入] 输出目录必须不存在", file=sys.stderr)
        return 2
    raw_timeout = (
        str(args.llm_timeout)
        if args.llm_timeout is not None
        else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        print("[接入门禁 1/2] NUMPILOT_LLM_TIMEOUT 必须是秒数", file=sys.stderr)
        return 2
    if not 10 <= timeout_seconds <= 3600:
        print("[接入门禁 1/2] LLM timeout 必须在 10～3600 秒之间", file=sys.stderr)
        return 2
    os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
    configuration_status = _load_llm_or_report()
    if configuration_status is None:
        print("[接入门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
        return 4
    try:
        llm_gate = _run_llm_readiness_gate(
            configuration_status, require_marker=True
        )
    except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
        print(
            "[接入门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
            file=sys.stderr,
        )
        print(
            "[AsNumpy 接入生成] actual_remote_call=no · 未读取 Skill，未生成 overlay",
            file=sys.stderr,
        )
        return 4
    probe_receipt = llm_gate["receipt"]
    print(
        "[接入门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · timeout=%ss"
        % (
            probe_receipt.get("http_status"),
            probe_receipt.get("response_model") or configuration_status.model,
            timeout_seconds,
        )
    )

    bundle = load_asnumpy_integration_skill(
        Path(args.asnumpy_skill_root) if args.asnumpy_skill_root else None
    )
    skill_gate = _run_asnumpy_skill_gate(
        bundle, required_files=ASNUMPY_INTEGRATION_SKILL_FILES
    )
    print(
        "[接入门禁 2/2] pass · 已读取 NumPilot AsNumpy 案例 Skill · "
        "files=%s · bundle_sha256=%s"
        % (len(bundle.files), bundle.bundle_sha256)
    )
    staging_root = None
    try:
        staging_root = _create_output_stage(out_root, "asnumpy-generate")
        _write_generation_preflight(
            staging_root, llm_gate, skill_gate, timeout_seconds
        )
        spec_data = json.loads(read_bounded(
            args.spec, "gap spec", 2 * 1024 * 1024
        ))
        if not isinstance(spec_data, dict):
            raise DraftGenerationError("gap spec must be a JSON object")
        spec = GapSpec.from_dict(spec_data)
        signed = _load_existing_draft(Path(args.artifact), spec)
        artifact_manifest = json.loads(read_bounded(
            Path(args.artifact) / "operator-artifact.json",
            "operator artifact manifest",
            2 * 1024 * 1024,
        ))
        adapter_paths = sorted(
            path for path in signed.files
            if path.startswith("asnumpy_adapter/") and path.endswith(".json")
        )
        if len(adapter_paths) != 1:
            raise DraftGenerationError(
                "signed artifact must contain exactly one AsNumpy adapter JSON"
            )
        adapter_path = Path(args.artifact) / adapter_paths[0]
        adapter_contract = read_bounded(
            adapter_path, "AsNumpy adapter contract", 256 * 1024
        )
        if hashlib.sha256(adapter_contract.encode("utf-8")).hexdigest() != (
            signed.files[adapter_paths[0]]
        ):
            raise DraftGenerationError("AsNumpy adapter contract hash mismatch")
        adapter_data = json.loads(adapter_contract)
        api_name = adapter_data.get("api_name") if isinstance(adapter_data, dict) else None
        if (
            not isinstance(api_name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", api_name)
            or api_name != spec.api_name
        ):
            raise DraftGenerationError("AsNumpy adapter API identity mismatch")

        reference_root = Path(args.asnumpy_reference).absolute()
        if reference_root.is_symlink() or not reference_root.is_dir():
            raise DraftGenerationError("AsNumpy reference must be a real directory")
        reference_paths = (
            "scripts/gen_custom_ops.py",
            "csrc/custom_ops/registry/angle_v2/op.yaml",
            "tests/asnumpy_tests/custom_ops_tests/test_angle_v2.py",
            "CMakeLists.txt",
            "cmake/CustomOps.cmake",
            "csrc/custom_ops/CMakeLists.txt",
            "include/asnumpy/custom_ops/loader.hpp",
            "csrc/custom_ops/loader.cpp",
        )
        context = {
            relative: read_bounded(
                reference_root / relative,
                "AsNumpy reference %s" % relative,
                96 * 1024,
            )
            for relative in reference_paths
        }
        operator_reference_paths = (
            "CMakeLists.txt",
            "op_api/aclnn_%s.h" % api_name,
            "op_api/%s.cpp" % api_name,
        )
        for relative in operator_reference_paths:
            if relative not in signed.files:
                continue
            content = read_bounded(
                Path(args.artifact) / relative,
                "signed operator reference %s" % relative,
                96 * 1024,
            )
            if hashlib.sha256(content.encode("utf-8")).hexdigest() != (
                signed.files[relative]
            ):
                raise DraftGenerationError(
                    "signed operator reference hash mismatch: %s" % relative
                )
            context["SIGNED_OPERATOR/%s" % relative] = content
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
        )
        report = provider.generate_asnumpy_adapter_candidate(
            staging_root,
            operator_artifact=artifact_manifest,
            adapter_contract=adapter_contract,
            asnumpy_context=context,
            expected_paths=(
                "csrc/custom_ops/registry/%s/op.yaml" % api_name,
                "tests/asnumpy_tests/custom_ops_tests/test_%s.py" % api_name,
            ),
            skill_files=ASNUMPY_INTEGRATION_SKILL_FILES,
        )
        _publish_output_stage(staging_root, out_root, "AsNumpy adapter")
        staging_root = None
    except DraftRepairError as exc:
        print("[AsNumpy 接入生成] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=%s verified_new_adapter=no receipts=%d"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[AsNumpy 接入生成] 失败: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)

    print(
        "[AsNumpy 接入生成] status=%s adapter_artifact_id=%s out=%s"
        % (report["status"], report["adapter_artifact_id"], out_root / "candidate")
    )
    print(
        "[LLM调用] actual_remote_call=yes verified_new_adapter=yes "
        "attempts=%d receipts=%d content_hash_bound=yes"
        % (report["attempt_count"], len(report["remote_receipts"]))
    )
    return 0


def _cmd_repair_asnumpy_adapter(args):
    """Return a raw AsNumpy diagnostic to the model for one-file repair."""

    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
    )

    def read_bounded(path, label, limit):
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise DraftGenerationError("%s must be a regular file" % label)
        payload = target.read_bytes()
        if len(payload) > limit:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        return payload.decode("utf-8")

    out_root = Path(args.out).absolute()
    if os.path.lexists(out_root):
        print("[AsNumpy 接入修复] 输出目录必须不存在", file=sys.stderr)
        return 2
    raw_timeout = (
        str(args.llm_timeout)
        if args.llm_timeout is not None
        else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        print("[接入修复门禁 1/2] NUMPILOT_LLM_TIMEOUT 必须是秒数", file=sys.stderr)
        return 2
    if not 10 <= timeout_seconds <= 3600:
        print(
            "[接入修复门禁 1/2] LLM timeout 必须在 10～3600 秒之间",
            file=sys.stderr,
        )
        return 2
    os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
    configuration_status = _load_llm_or_report()
    if configuration_status is None:
        print("[接入修复门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
        return 4
    try:
        llm_gate = _run_llm_readiness_gate(
            configuration_status, require_marker=True
        )
    except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
        print(
            "[接入修复门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
            file=sys.stderr,
        )
        print(
            "[AsNumpy 接入修复] actual_remote_call=no · 未读取 Skill，未开始修复",
            file=sys.stderr,
        )
        return 4
    receipt = llm_gate["receipt"]
    print(
        "[接入修复门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · "
        "timeout=%ss"
        % (
            receipt.get("http_status"),
            receipt.get("response_model") or configuration_status.model,
            timeout_seconds,
        )
    )

    profile_files = ASNUMPY_INTEGRATION_SKILL_FILES
    bundle = load_asnumpy_integration_skill(
        Path(args.asnumpy_skill_root) if args.asnumpy_skill_root else None
    )
    skill_gate = _run_asnumpy_skill_gate(
        bundle, required_files=profile_files
    )
    print(
        "[接入修复门禁 2/2] pass · 已读取 NumPilot AsNumpy 案例 Skill · "
        "profile=%s · files=%s · bundle_sha256=%s"
        % (args.repair_profile, len(bundle.files), bundle.bundle_sha256)
    )
    staging_root = None
    try:
        staging_root = _create_output_stage(out_root, "asnumpy-repair")
        _write_generation_preflight(
            staging_root, llm_gate, skill_gate, timeout_seconds
        )
        diagnostic = read_bounded(
            args.diagnostic_file, "AsNumpy repair diagnostic", 64 * 1024
        )
        reference_root = Path(args.asnumpy_reference).absolute()
        if reference_root.is_symlink() or not reference_root.is_dir():
            raise DraftGenerationError("AsNumpy reference must be a real directory")
        reference_paths = (
            "CMakeLists.txt",
            "cmake/CustomOps.cmake",
            "scripts/gen_custom_ops.py",
        )
        context = {
            relative: read_bounded(
                reference_root / relative,
                "AsNumpy reference %s" % relative,
                96 * 1024,
            )
            for relative in reference_paths
        }
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
        )
        report = provider.repair_asnumpy_adapter_candidate(
            staging_root,
            base_adapter_root=Path(args.base_adapter),
            diagnostic=diagnostic,
            path=args.path,
            asnumpy_context=context,
            skill_files=profile_files,
            repair_profile=args.repair_profile,
        )
        _publish_output_stage(staging_root, out_root, "AsNumpy adapter repair")
        staging_root = None
    except DraftRepairError as exc:
        print("[AsNumpy 接入修复] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=%s verified_new_adapter_repair=no "
            "receipts=%d"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[AsNumpy 接入修复] 失败: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
    print(
        "[AsNumpy 接入修复] status=%s adapter_artifact_id=%s "
        "parent_adapter_artifact_id=%s path=%s out=%s"
        % (
            report["status"],
            report["adapter_artifact_id"],
            report["parent_adapter_artifact_id"],
            args.path,
            out_root / "candidate",
        )
    )
    print(
        "[LLM调用] actual_remote_call=yes verified_new_adapter_repair=yes "
        "attempts=%d receipts=%d content_hash_bound=yes"
        % (report["attempt_count"], len(report["remote_receipts"]))
    )
    return 0


def _preflight(backend_name, skip):
    """migrate 前置自检。npu 后端缺硬依赖 → 拦截退出;shim 仅软告警。"""
    if skip:
        print("[env-check] 已通过 --skip-env-check 跳过环境自检")
        return
    rep = Env.check(backend=backend_name)
    if rep.errors:
        print(rep.render(), file=sys.stderr)
        print(
            "\n[env-check] 环境不满足 backend=%s 的硬依赖,已拦截。\n"
            "  修复以上 ✗ 项后重试,或加 --skip-env-check 强制继续(后果自负)。"
            % backend_name,
            file=sys.stderr,
        )
        sys.exit(3)
    if rep.warnings:
        print("[env-check] %d 个警告(不阻断):%s"
              % (len(rep.warnings), "; ".join(w.name for w in rep.warnings)))


def _cmd_migrate(args):
    backend_name = _resolve_backend_name(args)
    backend, analysis_profiles = _qualified_runtime_inputs(args, backend_name)
    _preflight(backend_name, args.skip_env_check)

    orch = Orchestrator(
        backend=backend,
        engine=_make_engine(args.engine),
        threshold=args.threshold,
        analysis_profiles=analysis_profiles,
    )
    res = orch.run(args.file, out_dir=args.out)

    print(_render_terminal(res))
    try:
        rp = save_migration_report(
            _migrate_report_info(res), reports_dir=args.reports_dir
        )
        print(f"[报告] 迁移报告已落盘:{rp}")
    except Exception as e:
        print(f"[报告] 落盘失败(不影响迁移):{e}")
    if res["status"] == "unsupported":
        return 1
    return 0 if res["status"] == "pass" else 2


def _migrate_report_info(res):
    """把 migrate 结果规范化为 report.build_report_md 的 info dict(无性能字段)。"""
    from numpilot.wizard import agent_roster
    net = _net_name(res.get("source_path", "task"))
    info = {"net": net, "source_path": res.get("source_path"),
            "engine": res.get("engine"), "status": res["status"],
            "agents": agent_roster(res.get("engine"))}
    a = res.get("analysis")
    if res["status"] == "unsupported":
        info["unsupported"] = res.get("unsupported", [])
        info["unsupported_details"] = [
            {"name": op.name, "lineno": op.lineno, "reason": op.reason}
            for op in (a.op_uses if a else []) if not op.supported
        ]
        return info
    rep = res["report"]
    plan = res.get("plan")
    rewrites = res.get("rewrites") or {}
    info["backend"] = rep.backend
    info["n_calls"] = len(a.op_uses) if a else None
    info["n_numpy_calls"] = (
        sum(op.kind == "numpy_api" for op in a.op_uses) if a else None
    )
    info["n_operator_calls"] = (
        sum(op.kind == "domain_operator" for op in a.op_uses) if a else None
    )
    info["n_ops"] = len({o.name for o in a.op_uses}) if a else None
    info["rewrites"] = rewrites
    info["total_rewrites"] = sum(rewrites.values())
    info["extension_apis"] = {op: n for op, n in rewrites.items() if is_custom(op)}
    if a:
        info["operator_specs"] = sorted({
            op.name for op in a.op_uses if op.kind == "domain_operator"
        })
        info["extension_apis"].update({
            op: rewrites.get(op, 0) for op in info["operator_specs"]
        })
    if plan:
        info["strategy"] = plan.strategy
        info["npu_ops"] = sorted(o for o, p in plan.placement.items() if p == "NPU")
        info["cpu_ops"] = sorted(o for o, p in plan.placement.items() if p == "CPU")
    info["backend_version"] = rep.backend_version
    info["backend_module"] = rep.backend_module
    info["numerically_consistent"] = rep.numerically_consistent
    info["max_rel_diff"] = rep.max_rel_diff
    info["max_abs_diff"] = rep.max_abs_diff
    info["tol"] = rep.tol
    info["atol"] = rep.atol
    info["rtol"] = rep.rtol
    info["failure_path"] = rep.failure_path
    info["failure_index"] = rep.failure_index
    info["failure_reason"] = rep.failure_reason
    info["exception_phase"] = rep.exception_phase
    info["exception_type"] = rep.exception_type
    info["exception_message"] = rep.exception_message
    info["artifact_inputs_executed"] = rep.artifact_inputs_executed
    info["input_consistent"] = rep.input_consistent
    info["kernel_consistent"] = rep.kernel_consistent
    info["artifact_consistent"] = rep.artifact_consistent
    info["comparisons"] = rep.comparisons
    info["backend_provenance"] = rep.backend_provenance
    info["performance_valid"] = False
    info["migrated_path"] = res.get("migrated_path")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(prog="numpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    mig = sub.add_parser("migrate", help="迁移一个 numpy 程序到 AsNumpy(NPU)")
    mig.add_argument("file")
    mig.add_argument("--backend", choices=["shim", "npu"], default=None)
    mig.add_argument("--engine", choices=["ast", "llm"], default=None)
    mig.add_argument("--out", default=None, help="结构化报告与迁移代码输出目录")
    mig.add_argument(
        "--reports-dir",
        default=None,
        help="时间戳报告目录；默认使用仓库外用户状态目录",
    )
    mig.add_argument("--threshold", type=int, default=1_000_000)
    mig.add_argument("--skip-env-check", action="store_true",
                     help="跳过 migrate 前置环境自检(缺依赖也强制继续)")
    mig.add_argument(
        "--capability-overlay",
        default=None,
        help="显式启用的 qualification overlay JSON（仅 NPU；不自动发现）",
    )
    mig.add_argument(
        "--qualification-evidence-root",
        default=None,
        help="overlay 证据根目录；必须与 --capability-overlay 成对提供",
    )
    mig.add_argument(
        "--analysis-profiles",
        default=None,
        help="与源 SHA/callsite 精确绑定的 runtime profile JSON",
    )

    bc = sub.add_parser("bench-cpu", help="① 纯 numpy 基线, 进程内钉 1 核, 存基线供对比")
    bc.add_argument("file")
    bc.add_argument(
        "--out",
        default=str(default_benchmark_dir()),
        help="基线文件输出目录（默认 %(default)s）",
    )
    bc.add_argument("--cpu-threads", default="1",
                    help="请求的 NumPy BLAS 线程数(默认 1；0/all 表示不主动限制)")

    bn = sub.add_parser(
        "bench-npu", help="② 审计并评测 asnumpy；仅严格 NPU 路径发布加速比")
    bn.add_argument("file")
    bn.add_argument(
        "--out",
        default=str(default_benchmark_dir()),
        help="读取 ① 基线 / 写迁移代码的目录（默认 %(default)s）",
    )
    bn.add_argument("--engine", choices=["ast", "llm"], default=None)
    bn.add_argument("--threshold", type=int, default=1_000_000)
    bn.add_argument("--skip-env-check", action="store_true")
    bn.add_argument(
        "--allow-cpu-fallback", action="store_true",
        help="仅供诊断：允许已知 CPU fallback/来源未知路径继续运行，但不发布 NPU 加速比")
    bn.add_argument(
        "--capability-overlay",
        default=None,
        help="显式启用的 qualification overlay JSON（不自动发现）",
    )
    bn.add_argument(
        "--qualification-evidence-root",
        default=None,
        help="overlay 证据根目录；必须与 --capability-overlay 成对提供",
    )
    bn.add_argument(
        "--analysis-profiles",
        default=None,
        help="与源 SHA/callsite 精确绑定的 runtime profile JSON",
    )

    wiz = sub.add_parser("wizard", help="交互式迁移向导(启动界面→选代码→进度迁移→反馈)")
    wiz.add_argument("file", nargs="?", default=None,
                     help="待迁移程序路径(省略则进入交互提示)")
    wiz.add_argument("--engine", choices=["ast", "llm"], default=None,
                     help="迁移引擎(省略则交互选择)")
    wiz.add_argument("--backend", choices=["shim", "npu"], default="npu")
    wiz.add_argument(
        "--out",
        default=None,
        help="迁移代码目录；默认使用仓库外用户状态目录",
    )
    wiz.add_argument(
        "--reports-dir",
        default=None,
        help="时间戳报告目录；默认使用仓库外用户状态目录",
    )

    doc = sub.add_parser("doctor", help="环境自检:检查 Python/numpy/CANN/NPU/asnumpy")
    doc.add_argument("--backend", choices=["shim", "npu"], default=None,
                     help="以指定后端语义评估硬依赖(npu 更严格)")

    paths = sub.add_parser(
        "paths", help="显示配置、缓存与运行产物的仓库外路径"
    )
    paths.add_argument("--json", action="store_true", help="输出 JSON")
    paths.add_argument(
        "--get",
        choices=("state", "runs", "reports", "benchmark", "wizard", "gaps",
                 "demo", "llm_config", "cannbot_cache"),
        help="只输出一个路径值",
    )

    suite = sub.add_parser(
        "check-suite",
        help="不执行源码，静态检查一组案例是否符合直接迁移/单缺口测试合同",
    )
    suite.add_argument(
        "paths",
        nargs="+",
        help="Python 案例文件或包含案例的目录（目录只扫描当前层 *.py）",
    )
    suite.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "ascend910b"),
        help="单缺口 GapSpec 的目标 SoC（默认 %(default)s）",
    )
    suite.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "9.0.0"),
        help="单缺口 GapSpec 的目标 CANN 版本（默认 %(default)s）",
    )
    suite.add_argument(
        "--expect-direct",
        type=int,
        default=None,
        help="期望的无缺口直接迁移候选数，例如本测试集填 5",
    )
    suite.add_argument(
        "--expect-single-gap",
        type=int,
        default=None,
        help="期望的恰好一个简单缺口候选数，例如本测试集填 5",
    )
    suite.add_argument(
        "--out",
        default=None,
        help="可选：原子写入 JSON 准入报告",
    )

    cfg = sub.add_parser(
        "configure-llm",
        help="在 NumPilot 内填写并安全保存 LLM Base URL、Token 与 Model",
    )
    cfg.add_argument("--base-url", default=None, help="OpenAI 兼容 API 根地址")
    cfg.add_argument("--model", default=None, help="模型名")
    cfg.add_argument("--provider", default=None, help="报告标签(默认 glm)")
    cfg.add_argument(
        "--token-stdin",
        action="store_true",
        help="从标准输入读取 Token；避免把 Token 放入命令行历史/进程参数",
    )
    cfg_mode = cfg.add_mutually_exclusive_group()
    cfg_mode.add_argument(
        "--init",
        action="store_true",
        help="生成私有 JSON 配置模板供手动填写；绝不覆盖已有文件",
    )
    cfg_mode.add_argument(
        "--check",
        action="store_true",
        help="只检查配置是否完整，不联网、不显示 Token",
    )
    cfg_mode.add_argument(
        "--probe",
        action="store_true",
        help="发起一次真实轻量请求；失败返回非零退出码且不显示 Token/响应正文",
    )

    skills = sub.add_parser(
        "setup-skills",
        help="获取并校验固定版本的华为官方 CANNBot 算子开发 Skill",
    )
    skills.add_argument(
        "--root",
        default=None,
        help="校验已有 cannbot-skills checkout；省略时使用版本化用户缓存",
    )
    skills.add_argument(
        "--check",
        action="store_true",
        help="只检查，不在缓存缺失时联网拉取",
    )

    gaps = sub.add_parser(
        "complete-gaps",
        help="检测真正缺失的 NumPy API，并准备或生成未验证的 Ascend C 算子草稿",
    )
    gaps.add_argument("file", help="待分析的 NumPy Python 源文件")
    gaps.add_argument(
        "--out",
        default=str(default_gap_dir()),
        help="规格、草稿和审计报告目录（默认 %(default)s）",
    )
    gaps.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "ascend910b"),
        help="目标 SoC；会进入规格和产物身份",
    )
    gaps.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "unknown"),
        help="目标 CANN 版本；未知时只建议 prepare，不可视为可复用能力",
    )
    gaps.add_argument(
        "--provider",
        choices=["prepare", "llm"],
        default="prepare",
        help="prepare 只写规格；llm 另生成隔离且未验证的源码草稿",
    )
    gaps.add_argument(
        "--operator-goal",
        choices=["accuracy", "performance"],
        default=None,
        help=(
            "算子补全目标；accuracy 是当前已实现流程；performance 为保留选项，"
            "当前会在联网、读取 Skill 和写输出前明确拒绝。交互终端会询问，"
            "非交互默认 accuracy"
        ),
    )
    gaps.add_argument(
        "--max-attempts",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="模型输出未通过本地结构校验时的最大尝试次数(最多 3)",
    )
    gaps.add_argument(
        "--llm-timeout",
        type=int,
        help=(
            "单次模型请求超时秒数（10～3600）；省略时读取 "
            "NUMPILOT_LLM_TIMEOUT，默认 120"
        ),
    )
    gaps.add_argument(
        "--skill-root",
        default=os.getenv("NUMPILOT_CANNBOT_SKILLS", "auto"),
        help="cannbot-skills 根目录；默认自动获取固定官方版本并嵌入模型提示",
    )
    gaps.add_argument(
        "--offline-skills",
        action="store_true",
        help="禁止自动联网获取 Skill；缓存或显式 --skill-root 缺失时失败",
    )
    gaps.add_argument(
        "--asnumpy-reference",
        default=os.getenv("NUMPILOT_ASNUMPY_REFERENCE", ""),
        help="AsNumpy 接入模板目录，仅作为生成契约引用",
    )

    repair = sub.add_parser(
        "repair-draft",
        help="将 CANN 编译/测试诊断回传 LLM，生成新的未验证修复副本",
    )
    repair.add_argument("--spec", required=True, help="签名 draft 对应的 gap-spec.json")
    repair.add_argument("--draft", required=True, help="不可变的原始签名 draft 目录")
    repair.add_argument(
        "--base-candidate",
        help="可选：经 repair report 校验的上一轮候选；只对新增失败文件接力修复",
    )
    repair.add_argument("--out", required=True, help="新修复副本目录，必须不存在")
    repair.add_argument(
        "--generation-request",
        required=True,
        help="原 draft 对应的 GENERATION_REQUEST.md",
    )
    repair.add_argument(
        "--diagnostic-file",
        required=True,
        help="脱敏且有界的 CANN/测试诊断文本",
    )
    repair.add_argument(
        "--path",
        action="append",
        required=True,
        help="要修复的签名项目相对路径；可重复指定",
    )
    repair.add_argument(
        "--max-attempts",
        type=int,
        choices=[1, 2, 3],
        default=2,
    )
    repair.add_argument(
        "--llm-timeout",
        type=int,
        help=(
            "单次模型问答/修复超时秒数（10～3600）；省略时读取 "
            "NUMPILOT_LLM_TIMEOUT，默认 120"
        ),
    )
    repair.add_argument(
        "--repair-profile",
        choices=["compile", "precision", "runtime"],
        default="compile",
        help=(
            "按失败阶段读取对应的外部 CANNBot Skill；默认 compile，"
            "精度/测试失败使用 precision，真机运行失败使用 runtime"
        ),
    )
    repair.add_argument(
        "--skill-root",
        default=os.getenv("NUMPILOT_CANNBOT_SKILLS", "auto"),
    )
    repair.add_argument("--offline-skills", action="store_true")

    seal = sub.add_parser(
        "seal-repair",
        help="复验完整 repair 链并发布新的不可变未验证 Artifact（不调用 LLM）",
    )
    seal.add_argument("--spec", required=True, help="原始 gap-spec.json")
    seal.add_argument("--draft", required=True, help="不可变的原始签名 draft")
    seal.add_argument("--candidate", required=True, help="最终 repair candidate")
    seal.add_argument("--out", required=True, help="新 Artifact 目录，必须不存在")
    seal.add_argument(
        "--skill-root",
        default=os.getenv("NUMPILOT_CANNBOT_SKILLS", "auto"),
        help="repair 时使用的固定 CANNBot checkout；省略时仅查找本地固定缓存",
    )
    seal.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容 repair-draft；seal-repair 始终离线且绝不下载 Skill",
    )

    adapter = sub.add_parser(
        "generate-asnumpy-adapter",
        help="由已配置模型生成隔离的 AsNumpy custom-op 清单与真机测试 overlay",
    )
    adapter.add_argument("--spec", required=True, help="签名 Artifact 对应的 gap-spec.json")
    adapter.add_argument("--artifact", required=True, help="已封存的模型算子 Artifact")
    adapter.add_argument(
        "--asnumpy-reference",
        required=True,
        help="现有 AsNumpy 清单生成器源码目录，仅作为只读接入契约",
    )
    adapter.add_argument("--out", required=True, help="隔离 overlay 与审计报告目录")
    adapter.add_argument(
        "--max-attempts", type=int, choices=[1, 2, 3], default=3
    )
    adapter.add_argument(
        "--llm-timeout",
        type=int,
        help=(
            "单次模型问答/接入文件生成超时秒数（10～3600）；省略时读取 "
            "NUMPILOT_LLM_TIMEOUT，默认 120"
        ),
    )
    adapter.add_argument(
        "--asnumpy-skill-root",
        "--skill-root",
        dest="asnumpy_skill_root",
        default=None,
        help="NumPilot 自带 AsNumpy 接入 Skill；默认读取仓库内固定目录",
    )
    adapter.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容旧命令；内置 AsNumpy Skill 从不联网",
    )

    adapter_repair = sub.add_parser(
        "repair-asnumpy-adapter",
        help="把 AsNumpy 构建/测试原始诊断交给已配置模型，修复一个 overlay 文件",
    )
    adapter_repair.add_argument(
        "--base-adapter",
        required=True,
        help="上一轮带 asnumpy-adapter-report.json 的模型 overlay 根目录",
    )
    adapter_repair.add_argument(
        "--asnumpy-reference",
        required=True,
        help="当前 AsNumpy 隔离工作树，只读提供构建与生成器契约",
    )
    adapter_repair.add_argument(
        "--diagnostic-file",
        required=True,
        help="原始且有界的 AsNumpy 构建/测试诊断",
    )
    adapter_repair.add_argument(
        "--path",
        required=True,
        help="只能修复上一轮模型 overlay 中的一个相对路径",
    )
    adapter_repair.add_argument("--out", required=True, help="新的隔离修复 overlay")
    adapter_repair.add_argument(
        "--repair-profile",
        choices=["compile", "precision", "runtime"],
        default="compile",
        help=(
            "把失败阶段绑定进提示词和审计报告；三个阶段共用同一内置 "
            "AsNumpy 接入 Skill"
        ),
    )
    adapter_repair.add_argument(
        "--max-attempts", type=int, choices=[1, 2, 3], default=3
    )
    adapter_repair.add_argument(
        "--llm-timeout",
        type=int,
        help="单次模型问答/修复超时秒数（10～3600，默认 120）",
    )
    adapter_repair.add_argument(
        "--asnumpy-skill-root",
        "--skill-root",
        dest="asnumpy_skill_root",
        default=None,
        help="NumPilot 自带 AsNumpy 接入 Skill；默认读取仓库内固定目录",
    )
    adapter_repair.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容旧命令；内置 AsNumpy Skill 从不联网",
    )

    args = parser.parse_args(argv)
    try:
        if args.cmd == "configure-llm":
            return _cmd_configure_llm(args)
        if args.cmd == "setup-skills":
            return _cmd_setup_skills(args)
        if args.cmd == "paths":
            return _cmd_paths(args)
        if args.cmd == "check-suite":
            return _cmd_check_suite(args)
        if args.cmd == "complete-gaps":
            return _cmd_complete_gaps(args)
        if args.cmd == "repair-draft":
            return _cmd_repair_draft(args)
        if args.cmd == "seal-repair":
            return _cmd_seal_repair(args)
        if args.cmd == "generate-asnumpy-adapter":
            return _cmd_generate_asnumpy_adapter(args)
        if args.cmd == "repair-asnumpy-adapter":
            return _cmd_repair_asnumpy_adapter(args)
        if args.cmd == "wizard":
            from numpilot.wizard import run_wizard
            return run_wizard(
                file=args.file,
                engine=args.engine,
                backend=args.backend,
                out_dir=args.out,
                reports_dir=args.reports_dir,
            )
        if args.cmd == "doctor":
            return _cmd_doctor(args)
        if args.cmd == "bench-cpu":
            return run_cpu_baseline(args.file, args.out, parse_cpu_threads(args.cpu_threads))
        if args.cmd == "bench-npu":
            if args.engine == "llm" and _load_llm_or_report() is None:
                return 4
            backend, analysis_profiles = _qualified_runtime_inputs(args, "npu")
            _preflight("npu", args.skip_env_check)
            return run_npu_compare(args.file, args.out, engine=_make_engine(args.engine),
                                   threshold=args.threshold,
                                   allow_cpu_fallback=args.allow_cpu_fallback,
                                   backend=backend,
                                   analysis_profiles=analysis_profiles)
        if args.engine == "llm" and _load_llm_or_report() is None:
            return 4
        return _cmd_migrate(args)
    except LlmError as e:
        print(f"\n[LLM] 迁移终止:{e}", file=sys.stderr)
        print("[LLM] 已按 strict 语义直接终止(未回退 AST)。", file=sys.stderr)
        return 4
    except RuntimeActivationConfigError as e:
        print(f"[qualification runtime] {e}", file=sys.stderr)
        return 2
    except CannbotSkillError as e:
        print(f"[CANNBot skill] {e}", file=sys.stderr)
        if (
            getattr(args, "cmd", None) == "complete-gaps"
            and getattr(args, "provider", None) == "llm"
        ):
            print(
                "[生成门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[算子生成] actual_remote_generation_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始算子生成。",
                file=sys.stderr,
            )
        elif getattr(args, "cmd", None) == "repair-draft":
            print(
                "[修复门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[算子修复] actual_remote_repair_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始模型修复。",
                file=sys.stderr,
            )
        elif getattr(args, "cmd", None) == "generate-asnumpy-adapter":
            print(
                "[接入门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入生成] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始 overlay 生成。",
                file=sys.stderr,
            )
        elif getattr(args, "cmd", None) == "repair-asnumpy-adapter":
            print(
                "[接入修复门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入修复] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始修复。",
                file=sys.stderr,
            )
        return 2
    except AsnumpyIntegrationSkillError as e:
        print(f"[AsNumpy integration skill] {e}", file=sys.stderr)
        command = getattr(args, "cmd", None)
        if command == "generate-asnumpy-adapter":
            print(
                "[接入门禁 2/2] fail · NumPilot AsNumpy 案例 Skill 读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入生成] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始 overlay 生成。",
                file=sys.stderr,
            )
        elif command == "repair-asnumpy-adapter":
            print(
                "[接入修复门禁 2/2] fail · NumPilot AsNumpy 案例 Skill 读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入修复] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始修复。",
                file=sys.stderr,
            )
        return 2


if __name__ == "__main__":
    sys.exit(main())
