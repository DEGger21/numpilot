"""CLI: python -m numpilot migrate <file> [选项] | python -m numpilot doctor [选项]"""
import argparse
import sys

from numpilot.backends.base import select_backend
from numpilot.env_check import Env
from numpilot.orchestrator import Orchestrator, _make_engine


def _resolve_backend_name(args):
    import os
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

    print(f"状态: {res['status']}")
    if res["status"] == "unsupported":
        print("缺口算子:", ", ".join(res["unsupported"]))
        return 1
    rep = res["report"]
    print(f"数值一致: {rep.numerically_consistent}  max_rel_diff={rep.max_rel_diff:.2e}  tol={rep.tol:.0e}")
    if rep.t_orig_ms is not None and rep.t_mig_ms is not None:
        n = f"  ({rep.n_elements:,} 元素)" if rep.n_elements is not None else ""
        print(f"耗时{n}: numpy@CPU {rep.t_orig_ms:.3f} ms  vs  asnumpy@{rep.backend} {rep.t_mig_ms:.3f} ms")
    if rep.speedup is not None:
        faster = "asnumpy 更快" if rep.speedup > 1 else "numpy 更快"
        print(f"加速比(CPU/{rep.backend}): {rep.speedup:.4f}x  ({faster})")
    if res.get("migrated_path"):
        print("迁移代码:", res["migrated_path"])
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
