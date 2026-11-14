#!/usr/bin/env python3
"""Turn a benchmark case's forward() into an AsNumpy-backed implementation."""
from __future__ import annotations

import ast
import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    source = Path(args.source).read_text(encoding="utf-8")
    marker = "\ndef forward"
    if marker not in source:
        raise SystemExit("source has no top-level forward()")
    header, forward = source.split(marker, 1)
    forward = "def forward" + forward
    tree = ast.parse(source)
    forward_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    names = [arg.arg for arg in forward_node.args.args]
    converted = forward.replace("np.", "ap.")
    lines = converted.splitlines()
    # Keep the signature line, then convert every ndarray parameter to the
    # device-backed AsNumpy type exactly once.
    insert_at = 1
    conversion = [
        "    if isinstance(%s, np.ndarray):" % name
        + "\n        %s = ap.ndarray.from_numpy(%s)" % (name, name)
        for name in names
    ]
    result = "\n".join(lines[:insert_at] + conversion + lines[insert_at:]) + "\n"
    text = "import asnumpy as ap\n" + header + result
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
