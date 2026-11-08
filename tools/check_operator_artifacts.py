#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from numpilot.operator_artifact_audit import audit_run, audit_store


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit a NumPilot operator-artifact store or complete benchmark run"
    )
    parser.add_argument("root", nargs="?", help="operator-artifacts directory")
    parser.add_argument(
        "--run-root",
        dest="run_root",
        help="audit a complete run root, including cases, refs, store, and summary",
    )
    parser.add_argument("--expected-total", type=int, default=150)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.root) == bool(args.run_root):
        parser.error("provide exactly one positional store root or --run-root")
    report = (
        audit_run(args.run_root, expected_total=args.expected_total)
        if args.run_root
        else audit_store(args.root)
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("operator-artifact-audit: %s" % ("PASS" if report["ok"] else "FAIL"))
        print("root=%s counts=%s" % (report["root"], report["counts"]))
        for error in report["errors"]:
            print("error=%s" % error, file=sys.stderr)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
