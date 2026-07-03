"""CLI: python -m numpilot migrate <file> [选项] | python -m numpilot doctor [选项]"""
import argparse
import os
import sys

from numpilot.backends.base import select_backend
from numpilot.env_check import Env
from numpilot.orchestrator import Orchestrator, _make_engine


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

    # [性能] 整网单次前向 numpy@CPU vs asnumpy@后端
    if rep.t_orig_ms is not None and rep.t_mig_ms is not None:
        n = f"{rep.n_elements:,} 元素   " if rep.n_elements is not None else ""
        line = (f"[性能] {n}numpy@CPU {rep.t_orig_ms:.3f}ms   "
                f"asnumpy@{rep.backend} {rep.t_mig_ms:.3f}ms")
        if rep.speedup is not None:
            faster = "asnumpy 更快" if rep.speedup > 1 else "numpy 更快"
            line += f"   {rep.speedup:.2f}x ({faster})"
        out.append(line)

    # [产物]
    arts = [res["migrated_path"]] if res.get("migrated_path") else []
    if arts:
        out.append("[产物] " + " · ".join(arts))
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

    doc = sub.add_parser("doctor", help="环境自检:检查 Python/numpy/CANN/NPU/asnumpy")
    doc.add_argument("--backend", choices=["shim", "npu"], default=None,
                     help="以指定后端语义评估硬依赖(npu 更严格)")

    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        return _cmd_doctor(args)
    return _cmd_migrate(args)


if __name__ == "__main__":
    sys.exit(main())
