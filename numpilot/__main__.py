"""CLI: python -m numpilot migrate <file> [选项]"""
import argparse
import sys

from numpilot.backends.base import select_backend
from numpilot.orchestrator import Orchestrator, _make_engine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="numpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    mig = sub.add_parser("migrate", help="迁移一个 numpy 程序到 AsNumpy(NPU)")
    mig.add_argument("file")
    mig.add_argument("--backend", choices=["shim", "npu"], default=None)
    mig.add_argument("--engine", choices=["ast", "llm"], default=None)
    mig.add_argument("--out", default=None, help="报告与迁移代码输出目录")
    mig.add_argument("--threshold", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    orch = Orchestrator(
        backend=select_backend(args.backend),
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
    if rep.speedup is not None:
        print(f"加速比(当前后端): {rep.speedup:.4f}x")
    if res.get("migrated_path"):
        print("迁移代码:", res["migrated_path"])
    return 0 if res["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
