"""CLI: python -m numpilot <migrate|bench-cpu|bench-npu|doctor> ...

演示三步:
  1) bench-cpu <file>   纯 NumPy 基线，默认请求 BLAS 1 线程并记录实际线程池
  2) bench-npu <file>   审计迁移算子执行来源；仅严格 NPU 结果可与 1) 计算加速比
  migrate <file>        迁移 + 正确性 + "真用了 asnumpy" 叙事(不做性能对比)
"""
import argparse
import os
import sys


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
from numpilot.bench import run_cpu_baseline, run_npu_compare  # noqa: E402
from numpilot.coverage import is_custom                       # noqa: E402
from numpilot.engines.llm import LlmError                     # noqa: E402
from numpilot.env_check import Env                            # noqa: E402
from numpilot.orchestrator import Orchestrator, _make_engine  # noqa: E402
from numpilot.report import save_migration_report            # noqa: E402


def _net_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _render_terminal(res):
    """把迁移结果渲染成分段叙事:分析→迁移→卸载→执行→校验→性能→产物。"""
    analysis = res.get("analysis")
    net = _net_name(res.get("source_path", "<?>"))
    out = []
    out.append(f"── NumPilot 迁移 · {net} " + "─" * max(4, 34 - len(net)))

    # [分析] numpy 调用点 / 去重算子数 / 覆盖情况
    if analysis is not None:
        ops = analysis.op_uses
        distinct = sorted({o.name for o in ops})
        if res["status"] == "unsupported":
            out.append(f"[分析] {len(ops)} 处 numpy 调用 · {len(distinct)} 个算子 · "
                       f"命中 {len(res['unsupported'])} 个覆盖集缺口 ✗")
            out.append("[缺口] " + ", ".join(res["unsupported"]) + "  (无法迁移)")
            return "\n".join(out)
        out.append(f"[分析] {len(ops)} 处 numpy 调用 · {len(distinct)} 个算子 · 全部在 asnumpy 覆盖集 ✓")

    # [迁移] 引擎 + 替换计数
    rewrites = res.get("rewrites") or {}
    total = sum(rewrites.values())
    out.append(f"[迁移] 引擎={res.get('engine')} · 替换 {total} 处 np.* → ap.*")
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
    out.append("[性能] 本命令不发布加速比；请运行独立、fail-closed 的两步评测:")
    out.append(f"       ① python -m numpilot bench-cpu {src} --out out_demo")
    out.append(f"       ② python -m numpilot bench-npu {src} --out out_demo")
    return "\n".join(out)


def _resolve_backend_name(args):
    return args.backend or os.getenv("NUMPILOT_BACKEND", "shim")


def _cmd_doctor(args):
    rep = Env.check(backend=_resolve_backend_name(args))
    print(rep.render())
    return 1 if rep.errors else 0


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
    _preflight(backend_name, args.skip_env_check)

    orch = Orchestrator(
        backend=select_backend(backend_name),
        engine=_make_engine(args.engine),
        threshold=args.threshold,
    )
    res = orch.run(args.file, out_dir=args.out)

    print(_render_terminal(res))
    try:
        rp = save_migration_report(_migrate_report_info(res))
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
    info["n_ops"] = len({o.name for o in a.op_uses}) if a else None
    info["rewrites"] = rewrites
    info["total_rewrites"] = sum(rewrites.values())
    info["extension_apis"] = {op: n for op, n in rewrites.items() if is_custom(op)}
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
    mig.add_argument("--out", default=None, help="报告与迁移代码输出目录")
    mig.add_argument("--threshold", type=int, default=1_000_000)
    mig.add_argument("--skip-env-check", action="store_true",
                     help="跳过 migrate 前置环境自检(缺依赖也强制继续)")

    bc = sub.add_parser("bench-cpu", help="① 纯 numpy 基线, 进程内钉 1 核, 存基线供对比")
    bc.add_argument("file")
    bc.add_argument("--out", default="out_demo", help="基线文件输出目录")
    bc.add_argument("--cpu-threads", default="1",
                    help="请求的 NumPy BLAS 线程数(默认 1；0/all 表示不主动限制)")

    bn = sub.add_parser(
        "bench-npu", help="② 审计并评测 asnumpy；仅严格 NPU 路径发布加速比")
    bn.add_argument("file")
    bn.add_argument("--out", default="out_demo", help="读取 ① 基线 / 写迁移代码的目录")
    bn.add_argument("--engine", choices=["ast", "llm"], default=None)
    bn.add_argument("--threshold", type=int, default=1_000_000)
    bn.add_argument("--skip-env-check", action="store_true")
    bn.add_argument(
        "--allow-cpu-fallback", action="store_true",
        help="仅供诊断：允许已知 CPU fallback/来源未知路径继续运行，但不发布 NPU 加速比")

    wiz = sub.add_parser("wizard", help="交互式迁移向导(启动界面→选代码→进度迁移→反馈)")
    wiz.add_argument("file", nargs="?", default=None,
                     help="待迁移程序路径(省略则进入交互提示)")
    wiz.add_argument("--engine", choices=["ast", "llm"], default=None,
                     help="迁移引擎(省略则交互选择)")
    wiz.add_argument("--backend", choices=["shim", "npu"], default="npu")

    doc = sub.add_parser("doctor", help="环境自检:检查 Python/numpy/CANN/NPU/asnumpy")
    doc.add_argument("--backend", choices=["shim", "npu"], default=None,
                     help="以指定后端语义评估硬依赖(npu 更严格)")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "wizard":
            from numpilot.wizard import run_wizard
            return run_wizard(file=args.file, engine=args.engine, backend=args.backend)
        if args.cmd == "doctor":
            return _cmd_doctor(args)
        if args.cmd == "bench-cpu":
            return run_cpu_baseline(args.file, args.out, parse_cpu_threads(args.cpu_threads))
        if args.cmd == "bench-npu":
            _preflight("npu", args.skip_env_check)
            return run_npu_compare(args.file, args.out, engine=_make_engine(args.engine),
                                   threshold=args.threshold,
                                   allow_cpu_fallback=args.allow_cpu_fallback)
        return _cmd_migrate(args)
    except LlmError as e:
        print(f"\n[LLM] 迁移终止:{e}", file=sys.stderr)
        print("[LLM] 已按 strict 语义直接终止(未回退 AST)。", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
