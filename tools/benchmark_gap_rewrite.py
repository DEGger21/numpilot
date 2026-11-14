#!/usr/bin/env python3
"""Model-backed source rewrite for one benchmark gap case.

This is an experiment tool for the benchmark lane.  It asks the configured
model to rewrite only unsupported calls in ``forward`` using AsNumpy-supported
NuMpy primitives, then validates the candidate locally before writing it:

* the candidate must parse;
* ``build_inputs`` must be byte-for-byte equivalent at the AST level;
* output names must be unchanged;
* CPU outputs of the original and adapted programs must match;
* the benchmark lowering must leave no residual unsupported APIs.

The original program remains the oracle.  Model output and token receipts are
written next to the candidate for audit.
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
from numpilot.interactive_migration import (  # noqa: E402
    _complete_text_with_configured_model,
)

SUPPORTED = None


def _asnumpy_surface() -> list[str]:
    global SUPPORTED
    if SUPPORTED is None:
        import asnumpy

        SUPPORTED = sorted(
            name for name in dir(asnumpy) if not name.startswith("_")
        )
    return SUPPORTED


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_fence(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        first, _, rest = body.partition("\n")
        if rest.endswith("```"):
            rest = rest[:-3]
        body = rest.strip()
    return body


def _function_ast(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        (item for item in tree.body
         if isinstance(item, ast.FunctionDef) and item.name == name),
        None,
    )
    if node is None:
        raise ValueError("missing function %s" % name)
    return ast.unparse(node)


def _residual_missing(source: str) -> list[str]:
    lowered = lower_benchmark_program(source).source
    tree = ast.parse(lowered)
    import asnumpy

    top = {name for name in dir(asnumpy) if not name.startswith("_")}
    sub = {}
    for module_name in ("linalg", "random"):
        module = getattr(asnumpy, module_name, None)
        if module is not None:
            sub[module_name] = {
                name for name in dir(module) if not name.startswith("_")
            }
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "numpy":
                    aliases.add(item.asname or "numpy")
    excluded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_inputs":
            for child in ast.walk(node):
                excluded.add(id(child))
    missing = set()
    gap_keywords = {
        "out", "where", "casting", "initial", "endpoint", "copy",
        "subok", "order", "signature", "extobj",
    }
    for node in ast.walk(tree):
        if id(node) in excluded:
            continue
        if isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name) or current.id not in aliases:
                continue
            path = ".".join(reversed(parts))
            head, _, rest = path.partition(".")
            covered = head in top if not rest else (
                head in sub and rest in sub[head]
            )
            if not covered:
                missing.add(path)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name) or current.id not in aliases:
                continue
            path = ".".join(reversed(parts))
            bad = sorted({
                keyword.arg for keyword in node.keywords
                if keyword.arg in gap_keywords
            })
            if bad:
                missing.add("%s (%s=)" % (path, ",".join(bad)))
    return sorted(missing)


def _run_forward(program: Path, src: str) -> dict:
    namespace = runpy.run_path(str(program))
    inputs = namespace["build_inputs"]()
    outputs = namespace["forward"](**inputs)
    return outputs


def _compare_outputs(original: dict, adapted: dict) -> list[str]:
    import numpy as np

    issues = []
    if set(original) != set(adapted):
        return ["output names differ: %s vs %s" % (
            sorted(original), sorted(adapted)
        )]
    for name in original:
        left, right = original[name], adapted[name]
        if getattr(left, "shape", None) != getattr(right, "shape", None):
            issues.append("%s shape differs" % name)
            continue
        if getattr(left, "dtype", None) != getattr(right, "dtype", None):
            issues.append("%s dtype differs" % name)
            continue
        if not np.allclose(left, right, rtol=1e-4, atol=1e-5, equal_nan=True):
            issues.append("%s values differ" % name)
    return issues


def _rewrite_prompt(source: str, missing: list[str]) -> tuple[str, str]:
    system = (
        "你是 NumPyPilot 的 API Gap Filler。只重写 benchmark 程序的 forward "
        "（以及 forward 可达的 helper），消除给定缺口 API 或不被支持的签名，"
        "保持 build_inputs、输出名称、shape、dtype 和数值语义不变。"
        "只能使用给定 AsNumpy 导出表中的 NumPy API；不要新增文件或依赖，不要"
        "使用切片、`where`、`out`、`casting`、`initial`、`endpoint` 等未授权"
        "签名。二元 API 的标量参数必须显式包装为同 shape 的数组，例如"
        "`np.full(shape, value, dtype=np.float32)` 或 `np.full_like(array, value, "
        "dtype=array.dtype)`；禁止使用 `np.float32(...)` 标量作为 add/maximum/"
        "multiply 等二元 API 的参数。返回完整的 Python 源文件，不要解释，"
        "不要 Markdown 代码围栏。"
    )
    user = (
        "缺口 API/签名：\n- %s\n\nAsNumpy 导出表：\n%s\n\n原始程序：\n%s"
        % (
            "\n- ".join(missing),
            ", ".join(_asnumpy_surface()),
            source,
        )
    )
    return system, user


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="benchmark case directory")
    parser.add_argument("--out", required=True, help="new experiment directory")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--config", default=None, help="llm.json path")
    args = parser.parse_args(argv)

    case_dir = Path(args.case).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit("output already exists: %s" % out)
    out.mkdir(parents=True, mode=0o700)
    program = case_dir / "program.py"
    source = program.read_text(encoding="utf-8")
    missing = _residual_missing(source)
    if not missing:
        raise SystemExit("no residual API gap after benchmark lowering")
    system, user = _rewrite_prompt(source, missing)
    started = __import__("time").time()
    content, identity, validated, metadata, elapsed_ms = (
        _complete_text_with_configured_model(
            system,
            user,
            config_path=args.config,
            max_tokens=args.max_tokens,
        )
    )
    candidate = _strip_fence(content)
    (out / "model-response.md").write_text(candidate + "\n", encoding="utf-8")
    (out / "model-receipt.json").write_text(
        json.dumps(
            {
                "model": getattr(identity, "model", None),
                "provider": getattr(identity, "provider", None),
                "protocol": getattr(identity, "protocol", None),
                "actual_remote_call": True,
                "elapsed_ms": elapsed_ms,
                "wall_seconds": round(__import__("time").time() - started, 3),
                "usage": (metadata or {}).get("usage"),
                "receipt": metadata,
                "missing_before": missing,
                "source_sha256": _sha(source),
                "candidate_sha256": _sha(candidate),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    adapted_path = out / "program_adapted.py"
    adapted_path.write_text(candidate + "\n", encoding="utf-8")

    issues = []
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        issues.append("candidate does not parse: %s" % exc)
    if not issues:
        try:
            if _function_ast(source, "build_inputs") != _function_ast(candidate, "build_inputs"):
                issues.append("build_inputs changed")
        except ValueError as exc:
            issues.append(str(exc))
    if not issues:
        residual = _residual_missing(candidate)
        if residual:
            issues.append("residual gaps: %s" % ", ".join(residual))
    if not issues:
        try:
            original_outputs = _run_forward(program, source)
            adapted_outputs = _run_forward(adapted_path, candidate)
            issues.extend(_compare_outputs(original_outputs, adapted_outputs))
        except Exception as exc:
            issues.append("CPU execution failed: %s: %s" % (
                type(exc).__name__, str(exc)[:300]
            ))
    report = {
        "schema": "numpilot.benchmark-gap-rewrite",
        "version": 1,
        "case": case_dir.name,
        "status": "accepted" if not issues else "rejected",
        "missing_before": missing,
        "residual_missing": _residual_missing(candidate) if not issues else None,
        "issues": issues,
        "model": getattr(identity, "model", None),
        "usage": (metadata or {}).get("usage"),
        "source": str(program),
        "adapted": str(adapted_path),
        "source_sha256": _sha(source),
        "adapted_sha256": _sha(candidate),
    }
    (out / "rewrite-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
