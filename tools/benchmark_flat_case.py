#!/usr/bin/env python3
"""Create a straight-line NumPilot surface for one benchmark case.

The benchmark corpus keeps ``build_inputs()`` and ``forward(**inputs)``.  The
ordinary NumPilot script workflow intentionally admits straight-line programs.
This tool lowers documented equivalents, inlines the forward body into a
top-level compute script, and keeps input construction in a host-only helper.

Outputs (inside ``<case>/numpilot-work`` unless --out-dir is given):
  inputs_helper.py  host CPU input builder; imports the original program
  program_flat.py   top-level compute body for interactive-migrate
  flat-meta.json    source hashes, forward parameters, output names, rewrites
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from numpilot.benchmark_lowering import lower_benchmark_program  # noqa: E402


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _private_write(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise SystemExit("refusing to overwrite: %s" % path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def build_flat(case_dir: Path) -> dict:
    case_dir = case_dir.resolve()
    program = case_dir / "program.py"
    if not program.is_file():
        raise SystemExit("missing program.py: %s" % program)
    source = program.read_text(encoding="utf-8")
    lowered = lower_benchmark_program(source)
    tree = ast.parse(lowered.source)
    forward = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "forward"),
        None,
    )
    if forward is None:
        raise SystemExit("lowered source has no forward(): %s" % program)
    args = [arg.arg for arg in forward.args.args]
    body = list(forward.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return_node = body[-1] if body and isinstance(body[-1], ast.Return) else None
    if return_node is None or not isinstance(return_node.value, ast.Dict):
        raise SystemExit("forward must end with a dict return: %s" % program)
    outputs = []
    for key, value in zip(return_node.value.keys, return_node.value.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise SystemExit("forward return keys must be string literals")
        outputs.append((key.value, value))
    compute_body = body[:-1]
    module = ast.Module(body=compute_body, type_ignores=[])
    ast.fix_missing_locations(module)
    compute_source = ast.unparse(module)
    output_lines = []
    for name, value in outputs:
        if not name.isidentifier():
            raise SystemExit("output name is not an identifier: %r" % name)
        output_lines.append("%s = %s" % (name, ast.unparse(value)))
    namespace = runpy.run_path(str(program))
    inputs = namespace["build_inputs"]()
    input_lines = []
    input_facts = {}
    for name in args:
        value = inputs[name]
        shape = tuple(int(item) for item in value.shape)
        dtype = str(value.dtype)
        fill = "True" if dtype == "bool" else "1" if dtype.startswith(("int", "uint")) else "0.5"
        if dtype.startswith("float"):
            import numpy as np
            if np.isnan(value).all():
                fill = "np.nan"
            elif np.isposinf(value).all():
                fill = "np.inf"
            elif np.isneginf(value).all():
                fill = "-np.inf"
        input_lines.append(
            "%s = np.full(%r, %s, dtype=np.%s)"
            % (name, shape, fill, dtype.replace("float32", "float32").replace("float64", "float64"))
        )
        input_facts[name] = {"shape": list(shape), "dtype": dtype, "fill": fill}
    helper = (
        "import runpy\n"
        "from pathlib import Path\n\n"
        "_case = runpy.run_path(\n"
        "    str(Path(__file__).resolve().parent.parent / 'program.py')\n"
        ")\n"
        "_inputs = _case['build_inputs']()\n"
        "globals().update(_inputs)\n"
    )
    flat = (
        "import numpy as np\n\n"
        "%s\n\n"
        "# NUMPILOT_DEVICE_INPUTS_END\n\n"
        "%s\n\n"
        "%s\n"
        % ("\n".join(input_lines), compute_source, "\n".join(output_lines))
    )
    meta = {
        "schema": "numpilot.benchmark-flat-source",
        "version": 1,
        "case": case_dir.name,
        "program": str(program),
        "program_sha256": _sha256(source),
        "lowered_sha256": _sha256(lowered.source),
        "flat_sha256": _sha256(flat),
        "parameters": args,
        "outputs": [name for name, _ in outputs],
        "rewrites": [item.to_dict() for item in lowered.rewrites],
        "notes": list(lowered.notes),
        "input_facts": {
            "values_replaced_with_deterministic_fill": True,
            "inputs": input_facts,
        },
    }
    return {"helper": helper, "flat": flat, "meta": meta}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="benchmark case directory containing program.py")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <case>/numpilot-work)",
    )
    args = parser.parse_args(argv)
    case_dir = Path(args.case)
    out_dir = Path(args.out_dir) if args.out_dir else case_dir / "numpilot-work"
    built = build_flat(case_dir)
    helper = out_dir / "inputs_helper.py"
    flat = out_dir / "program_flat.py"
    meta = out_dir / "flat-meta.json"
    _private_write(helper, built["helper"])
    _private_write(flat, built["flat"])
    _private_write(
        meta,
        json.dumps(built["meta"], ensure_ascii=False, indent=2) + "\n",
    )
    print(flat)
    print(helper)
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
