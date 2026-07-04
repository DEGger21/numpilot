"""CLI: python -m numpilot <migrate|bench-cpu|bench-npu|doctor> ...

演示三步:
  1) bench-cpu <file>   纯 numpy 基线,进程内钉死 1 核(公平对比 NPU 单卡)
  2) bench-npu <file>   放开所有限制,迁移到 asnumpy 在 NPU 上跑,读 1) 的基线算加速比
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

    ⚠️ 必须在 import numpy 之前设置:本机 numpy=scipy-openblas(OpenMP 线程后端),
    默认吃满全部 640 核。只有在加载前设好 *_NUM_THREADS,OpenBLAS 才会真正限核。
    bench-npu / migrate 不走这里(它们要 NPU 满线程),故只对 bench-cpu 生效。
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
from numpilot.engines.llm import LlmError                     # noqa: E402
from numpilot.env_check import Env                            # noqa: E402
from numpilot.orchestrator import Orchestrator, _make_engine  # noqa: E402


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
        out.append(f"[卸载] NPU: {', '.join(npu) or '—'}   CPU: {', '.join(cpu) or '—'}")

    rep = res["report"]

    # [执行] 证明真的用了 asnumpy
    if rep.backend_version:
        loc = rep.backend_module or "?"
        out.append(f"[执行] 后端={rep.backend} · asnumpy {rep.backend_version} ✓ 真机")
        out.append(f"       {loc}")
    else:
        out.append(f"[执行] 后端={rep.backend} · shim 仿真(numpy 承载 asnumpy 语义,数值恒等)")

    # [校验] 数值一致性
    mark = "✓" if rep.numerically_consistent else "✗"
    out.append(f"[校验] 数值一致 {mark}  max_rel_diff={rep.max_rel_diff:.2e}  (tol={rep.tol:.0e})")

    # [产物]
    arts = [res["migrated_path"]] if res.get("migrated_path") else []
    if arts:
        out.append("[产物] " + " · ".join(arts))

    # [性能] 交给两个独立 bench 程序(公平口径:numpy 单核 vs NPU 满线程)
    src = res.get("source_path", "<file>")
    out.append("[性能] 公平对比请分两步跑(见下):")
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
    if res["status"] == "unsupported":
        return 1
    return 0 if res["status"] == "pass" else 2


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
                    help="numpy BLAS 线程数(默认 1=单核公平对比;0/all=吃满全核对照)")

    bn = sub.add_parser("bench-npu", help="② 迁移到 asnumpy 在 NPU 上跑(满线程), 读 ① 基线算加速比")
    bn.add_argument("file")
    bn.add_argument("--out", default="out_demo", help="读取 ① 基线 / 写迁移代码的目录")
    bn.add_argument("--engine", choices=["ast", "llm"], default=None)
    bn.add_argument("--threshold", type=int, default=1_000_000)
    bn.add_argument("--skip-env-check", action="store_true")

    doc = sub.add_parser("doctor", help="环境自检:检查 Python/numpy/CANN/NPU/asnumpy")
    doc.add_argument("--backend", choices=["shim", "npu"], default=None,
                     help="以指定后端语义评估硬依赖(npu 更严格)")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "doctor":
            return _cmd_doctor(args)
        if args.cmd == "bench-cpu":
            return run_cpu_baseline(args.file, args.out, parse_cpu_threads(args.cpu_threads))
        if args.cmd == "bench-npu":
            _preflight("npu", args.skip_env_check)
            return run_npu_compare(args.file, args.out, engine=_make_engine(args.engine),
                                   threshold=args.threshold)
        return _cmd_migrate(args)
    except LlmError as e:
        print(f"\n[LLM] 迁移终止:{e}", file=sys.stderr)
        print("[LLM] 已按 strict 语义直接终止(未回退 AST)。", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
