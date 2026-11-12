"""Planner-owned translation taskbook for one NumPy program.

The taskbook is the single hand-off between the Planner and the two downstream
roles: the API Gap Filler and the Code Migrator.  It is deliberately *data
only*: building it never calls a model, never executes user code, and never
claims that a rewrite is correct.

Two different questions share one document:

* ``callsites`` answers "what is every NumPy call, where is it, what did the
  deterministic analyzer prove, and what may the migrated program contain?".
* ``preparation`` answers "which whole-program transformations were already
  proved safe before analysis?" so a migrator never has to rediscover helper
  inlining, static unrolling, destructuring, or documented equivalent
  spellings.

The taskbook is not a capability certificate.  Numerical verification and the
strict-NPU surface audit remain separate, deterministic gates.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from numpilot.python_operators import target_for_operator


SCHEMA = "numpilot.translation-taskbook"
SCHEMA_VERSION = 1

_ALIASES = {
    "abs": "absolute",
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atan2": "arctan2",
    "atanh": "arctanh",
    "fabs": "absolute",
    "pow": "power",
    "round_": "around",
    "round": "around",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256(payload)


def _seal(value: dict) -> dict:
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    value.pop("taskbook_id", None)
    value["taskbook_id"] = "translation-taskbook-" + _canonical_digest(value)
    return value


def _program_shape(source: str) -> dict:
    """Static, source-only structure facts; never executes the program."""

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return {"parsed": False, "error": str(error)}
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    counts = {
        "functions": len(functions),
        "helpers": sum(1 for node in functions if node.name not in {"forward", "build_inputs"}),
        "for": 0,
        "while": 0,
        "if": 0,
        "tuple_unpack": 0,
        "augassign": 0,
        "subscripts": 0,
        "calls": 0,
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            counts["for"] += 1
        elif isinstance(node, ast.While):
            counts["while"] += 1
        elif isinstance(node, ast.If):
            counts["if"] += 1
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, (ast.Tuple, ast.List)) for target in node.targets):
                counts["tuple_unpack"] += 1
        elif isinstance(node, ast.AugAssign):
            counts["augassign"] += 1
        elif isinstance(node, ast.Subscript):
            counts["subscripts"] += 1
        elif isinstance(node, ast.Call):
            counts["calls"] += 1
    forward = next((node for node in functions if node.name == "forward"), None)
    arguments = []
    if forward is not None:
        arguments = [
            argument.arg
            for argument in (
                list(forward.args.posonlyargs)
                + list(forward.args.args)
                + list(forward.args.kwonlyargs)
            )
        ]
    return {
        "parsed": True,
        "entry": "forward" if forward is not None else "module",
        "forward_arguments": arguments,
        "functions": [node.name for node in functions],
        "structure": counts,
    }


def _target_for(use) -> str | None:
    if use.target_api:
        return use.target_api
    name = use.name
    if name.startswith("python_operator."):
        target = target_for_operator(name)
        return "asnumpy." + target if target else None
    if name.startswith(("python_structure.", "operator.")):
        return None
    if name in _ALIASES:
        return "asnumpy." + _ALIASES[name]
    return "asnumpy." + name


def _status_for(use) -> str:
    if use.supported:
        return "ready"
    name = use.name
    reason = use.reason or ""
    if name.startswith(("python_operator.", "python_structure.")):
        return "lowering_required"
    if "未验证 NumPy 关键字" in reason:
        return "consent_required"
    if "绑定" in reason and "数组" in reason:
        return "lowering_required"
    if name in _ALIASES:
        return "alias_required"
    return "gap_required"


def _semantics_for(use) -> list[str]:
    name = use.name
    reason = use.reason or ""
    notes = []
    if use.supported:
        notes.append("AsNumpy call contract accepted by the static analyzer.")
    if name.startswith("python_operator."):
        notes.append(
            "Python dunder/operator must become an explicit ap.* call; "
            "the migrated program may not rely on unverified ndarray operators."
        )
        target = target_for_operator(name)
        if target:
            notes.append(
                "Deterministic target: asnumpy.%s(...). Both operands must be "
                "device arrays; materialize a Python scalar explicitly, e.g. "
                "ap.full(reference.shape, scalar, dtype=reference.dtype), "
                "unless the taskbook declares a verified scalar route."
                % target
            )
        else:
            notes.append(
                "No deterministic AsNumpy target is registered for this "
                "operator; do not guess an API with different semantics."
            )
    elif name.startswith("python_structure."):
        notes.append(
            "Python control-flow/structure must be eliminated by the prepared "
            "source or lowered by the migrator into explicit array calls."
        )
    elif "未验证 NumPy 关键字" in reason:
        notes.append(
            "The keyword contract is not accepted by AsNumpy; silently dropping "
            "the keyword would change observable semantics."
        )
    elif name in _ALIASES:
        notes.append(
            "NumPy documents this as an alias with the same call contract; "
            "the target API is the canonical AsNumpy name."
        )
    if reason:
        notes.append("Analyzer reason: " + reason)
    notes.append(
        "dtype/shape/axis/layout are static-source unknown until a real input "
        "profile or the NumPy oracle supplies evidence."
    )
    return notes


def _callsite(use) -> dict:
    return {
        "id": "L%s:C%s:%s" % (use.lineno, use.col_offset, use.name),
        "lineno": use.lineno,
        "col_offset": getattr(use, "col_offset", 0),
        "name": use.name,
        "qualified_name": use.qualified_name,
        "call_expr": use.call_expr,
        "positional_args": use.positional_args,
        "keyword_args": list(use.keyword_args),
        "kind": use.kind,
        "supported": bool(use.supported),
        "status": _status_for(use),
        "target_api": _target_for(use),
        "semantics": _semantics_for(use),
        "facts": {
            "dtype": {"status": "unknown", "evidence": None},
            "shape": {"status": "unknown", "evidence": None},
            "axis": {"status": "unknown", "evidence": None},
            "layout": {"status": "unknown", "evidence": None},
        },
    }


def _preparation_facts(preparation) -> dict:
    if preparation is None:
        return {
            "applied": False,
            "rewrite_counts": {},
            "rewrites": [],
            "notes": ["no preparation supplied"],
            "boundary": (
                "No whole-program preparation was applied; the analyzed source "
                "is the caller-supplied analysis surface."
            ),
        }
    rewrites = [item.to_dict() for item in getattr(preparation, "rewrites", ())]
    by_kind = {}
    for item in rewrites:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
    return {
        "applied": bool(getattr(preparation, "applied", False)),
        "rewrite_counts": by_kind,
        "rewrites": rewrites,
        "notes": list(getattr(preparation, "notes", ())),
        "boundary": (
            "Preparation is an analysis/migration surface only. The original "
            "source remains the NumPy verification oracle."
        ),
    }


def build_translation_taskbook(
    *,
    analysis,
    route: str,
    source: str,
    prepared_source: str | None = None,
    preparation=None,
    gap_plan=None,
    source_path: str | None = None,
) -> dict:
    """Build the Planner hand-off from deterministic facts only."""

    prepared_source = source if prepared_source is None else prepared_source
    callsites = [_callsite(use) for use in getattr(analysis, "op_uses", ()) or ()]
    gap_tasks = []
    if gap_plan is not None:
        gap_tasks = [task.to_dict() for task in getattr(gap_plan, "tasks", ()) or ()]
    value = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": source_path or getattr(analysis, "source_path", None),
            "sha256": _sha256(source),
            "prepared_sha256": _sha256(prepared_source),
            "prepared": prepared_source != source,
        },
        "route": route,
        "program": _program_shape(prepared_source),
        "original_program": _program_shape(source),
        "preparation": _preparation_facts(preparation),
        "callsites": callsites,
        "gap_tasks": gap_tasks,
        "acceptance": {
            "oracle_source_sha256": _sha256(source),
            "analysis_source_sha256": _sha256(prepared_source),
            "surface_gate": [
                "no residual numpy compute calls in the migrated compute region",
                "no host escape (to_numpy/tolist/item/__array__) except declared output boundaries",
                "every direct ap.* call must be accounted for by a taskbook callsite or authorized expansion",
            ],
            "numeric_gate": [
                "execute the original source and the migrated source on the same inputs",
                "compare every named output: name set, shape, dtype, values, NaN/Inf semantics",
            ],
            "input_gate": "the migrated source must not silently mutate or replace declared inputs",
            "strict_npu": "not_claimed_by_taskbook",
        },
        "unresolved": [
            "runtime dtype/shape/axis/layout profiles",
            "strict-NPU execution evidence",
            "performance evidence",
        ],
    }
    if not callsites:
        value["unresolved"].append("no NumPy callsites were collected")
    return _seal(value)


def validate_translation_taskbook(taskbook) -> dict:
    if not isinstance(taskbook, dict):
        raise ValueError("translation taskbook must be an object")
    value = json.loads(json.dumps(taskbook, ensure_ascii=False, allow_nan=False))
    if (
        value.get("schema") != SCHEMA
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported translation taskbook schema")
    identity = value.pop("taskbook_id", None)
    if identity != "translation-taskbook-" + _canonical_digest(value):
        raise ValueError("translation taskbook content identity mismatch")
    value["taskbook_id"] = identity
    source = value.get("source", {})
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        raise ValueError("translation taskbook source hash is missing")
    if not isinstance(value.get("callsites"), list):
        raise ValueError("translation taskbook callsites must be a list")
    return value


def render_translation_taskbook_markdown(taskbook) -> str:
    value = validate_translation_taskbook(taskbook)

    def one_line(item):
        return str(item).replace("`", "\\`").replace("\n", " ").replace("|", "\\|")

    lines = [
        "# 迁移任务书 / Translation taskbook",
        "",
        "- Taskbook ID: `%s`" % value["taskbook_id"],
        "- Route: `%s`" % value["route"],
        "- Source: `%s`" % one_line(value["source"].get("path") or "<memory>"),
        "- Source SHA-256: `%s`" % value["source"]["sha256"],
        "- Prepared SHA-256: `%s`" % value["source"]["prepared_sha256"],
        "- Prepared: `%s`" % value["source"]["prepared"],
        "",
        "本任务书由 Planner 根据静态分析与确定性准备生成。它不执行源码、不调用模型，"
        "也不是能力或性能证书。数值验证与 strict-NPU 表面审计仍是独立门禁。",
        "",
        "## 1. 程序结构",
        "",
        "- 入口：`%s`；forward 参数：%s。" % (
            value["program"].get("entry"),
            ", ".join("`%s`" % item for item in value["program"].get("forward_arguments", [])) or "无",
        ),
        "- 顶层函数：%s。" % (
            ", ".join("`%s`" % item for item in value["program"].get("functions", [])) or "无"
        ),
        "- 结构计数：`%s`。" % json.dumps(
            value["program"].get("structure", {}), ensure_ascii=False, sort_keys=True
        ),
        "",
        "## 2. 已完成的整程序准备",
        "",
        "- applied：`%s`。" % value["preparation"]["applied"],
        "- rewrite_counts：`%s`。" % json.dumps(
            value["preparation"]["rewrite_counts"], ensure_ascii=False, sort_keys=True
        ),
        "- 边界：%s" % value["preparation"]["boundary"],
        "",
    ]
    for note in value["preparation"]["notes"]:
        lines.append("- note：%s" % one_line(note))
    lines += ["", "## 3. 逐调用点任务", ""]
    for site in value["callsites"]:
        lines += [
            "### `%s` · 行 %s 列 %s" % (
                one_line(site["call_expr"] or site["name"]),
                site["lineno"],
                site["col_offset"],
            ),
            "",
            "- 源 API：`%s`；目标：`%s`。" % (
                one_line(site["qualified_name"] or site["name"]),
                one_line(site["target_api"] or "<需要生成/下降>"),
            ),
            "- 状态：`%s`；静态支持：`%s`。" % (site["status"], site["supported"]),
            "- 位置参数数：%s；关键字：%s。" % (
                site["positional_args"],
                ", ".join("`%s`" % one_line(item) for item in site["keyword_args"]) or "无",
            ),
        ]
        for note in site["semantics"]:
            lines.append("- %s" % one_line(note))
        lines.append("")
    lines += [
        "## 4. 缺口任务（若有）",
        "",
    ]
    if value["gap_tasks"]:
        for task in value["gap_tasks"]:
            lines.append(
                "- `%s`：route=`%s` status=`%s` target=`%s`。%s"
                % (
                    one_line(task["name"]),
                    task["route"],
                    task["status"],
                    one_line(task.get("target") or "-"),
                    one_line(task.get("reason") or ""),
                )
            )
    else:
        lines.append("- 无。")
    lines += [
        "",
        "## 5. 验收边界",
        "",
        "- Oracle：`%s`。" % value["acceptance"]["oracle_source_sha256"],
        "- 表面门禁：",
    ]
    for item in value["acceptance"]["surface_gate"]:
        lines.append("  - %s" % item)
    lines.append("- 数值门禁：")
    for item in value["acceptance"]["numeric_gate"]:
        lines.append("  - %s" % item)
    lines += [
        "- strict-NPU：`%s`。" % value["acceptance"]["strict_npu"],
        "",
        "## 6. 未确认事实",
        "",
        "- %s" % "；".join(value["unresolved"]),
        "",
    ]
    return "\n".join(lines)


def persist_translation_taskbook(out_dir: str, taskbook) -> dict:
    """Write the JSON and Markdown copies; return their paths and identity."""

    value = validate_translation_taskbook(taskbook)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "TRANSLATION_TASKBOOK.json"
    md_path = root / "TRANSLATION_TASKBOOK.md"
    json_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_translation_taskbook_markdown(value), encoding="utf-8")
    return {
        "taskbook_id": value["taskbook_id"],
        "json": {"path": str(json_path), "sha256": _sha256(json_path.read_text(encoding="utf-8"))},
        "markdown": {"path": str(md_path), "sha256": _sha256(md_path.read_text(encoding="utf-8"))},
    }
