#!/usr/bin/env python3
"""Audit cross-file contracts in a model-generated Ascend C draft.

The generator may emit individually plausible files whose contracts disagree
(field names, template arity, shape-pointer types, missing opapi linkage).  This
read-only audit catches those defects before a draft is called build-ready.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STRUCT_RE = re.compile(r"struct\s+(\w+TilingData)\s*\{(.*?)\n\};", re.S)
FIELD_RE = re.compile(r"\b(?:int64_t|int32_t|uint64_t|uint32_t|float|double|bool)\s+(\w+)\s*(?:=|;)")
TILING_USE_RE = re.compile(r"tiling->(\w+)")
GET_SHAPE_RE = re.compile(r"const\s+gert::Shape\s*\*\s*\w+\s*=\s*context->Get(?:Input|Output)Shape\(")
ENTRY_RE = re.compile(r"(template\s*<[^>]*>\s*)?__global__\s+__aicore__\s+void\s+(\w+)\s*\(")
TPL_DECL_RE = re.compile(r"ASCENDC_TPL_ARGS_DECL\s*\(\s*\w+\s*,(.*?)\);", re.S)
TPL_NAME_RE = re.compile(r"ASCENDC_TPL_\w+_DECL\s*\(\s*(\w+)")


def structs(root: Path):
    found = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".h", ".hpp", ".cpp"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in STRUCT_RE.finditer(text):
            name, body = match.group(1), match.group(2)
            fields = []
            for line in body.splitlines():
                if line.strip().startswith("//"):
                    continue
                fields.extend(FIELD_RE.findall(line))
            found.setdefault(name, []).append({
                "path": str(path.relative_to(root)), "fields": tuple(fields),
            })
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.draft).resolve()
    errors = []
    warnings = []
    struct_map = structs(root)
    for name, definitions in struct_map.items():
        variants = {item["fields"] for item in definitions}
        if len(variants) > 1:
            errors.append("TilingData `%s` has conflicting definitions: %s" % (
                name, json.dumps(definitions, ensure_ascii=False)))
    all_fields = {name: set(definitions[0]["fields"]) for name, definitions in struct_map.items()}
    host_files = sorted((root / "op_host").rglob("*tiling*.cpp"))
    for path in host_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for field in sorted(set(TILING_USE_RE.findall(text))):
            if not any(field in fields for fields in all_fields.values()):
                errors.append("host `%s` assigns undeclared tiling field `%s`" % (
                    path.relative_to(root), field))
        if GET_SHAPE_RE.search(text):
            errors.append("host `%s` assigns GetInput/OutputShape() to `gert::Shape*`; "
                          "use `gert::StorageShape*` and GetStorageShape()" % path.relative_to(root))
    entries = []
    for path in sorted((root / "op_kernel").rglob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in ENTRY_RE.finditer(text):
            entries.append((path, match.group(1) or "", match.group(2)))
    if len(entries) > 1:
        warnings.append("multiple kernel entry definitions: %s" % [
            [str(path.relative_to(root)), name] for path, _, name in entries
        ])
    key_files = sorted((root / "op_kernel").rglob("*tiling_key.h"))
    declared = []
    for path in key_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in TPL_DECL_RE.findall(text):
            declared.extend(TPL_NAME_RE.findall(block))
    host_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((root / "op_host").rglob("*.cpp"))
    )
    if entries and declared:
        for path, template, name in entries:
            params = re.findall(r"typename\s+(\w+)|(?:int|int32_t|int64_t|uint32_t|uint64_t)\s+(\w+)", template)
            params = [a or b for a, b in params]
            if params != declared:
                message = "kernel entry `%s` template params %s != ASCENDC_TPL_ARGS_DECL %s" % (
                    path.relative_to(root), params, declared)
                # A non-template entry is acceptable when host tiling uses a
                # constant TilingKey; only template-key dispatch needs arity.
                if not params and "GET_TPL_TILING_KEY" not in host_text:
                    warnings.append(message + " (host uses a constant TilingKey)")
                else:
                    errors.append(message)
    op_api = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((root / "op_api").rglob("*.cpp"))
    )
    host_cmake = root / "op_host/CMakeLists.txt"
    if "l0op::" in op_api and host_cmake.is_file():
        if "target_link_libraries(cust_opapi PRIVATE opapi)" not in host_cmake.read_text(
                encoding="utf-8", errors="replace"):
            errors.append("op_api uses l0op:: symbols but cust_opapi does not link opapi")
    report = {
        "schema": "numpilot.generated-asc-contract-audit.v1",
        "draft": str(root),
        "structs": {name: [list(item["fields"]) for item in defs] for name, defs in struct_map.items()},
        "kernel_entries": [[str(path.relative_to(root)), name] for path, _, name in entries],
        "tiling_key_params": declared,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
