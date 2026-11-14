#!/usr/bin/env python3
"""Collect one generated Ascend C operator into the repository evidence tree."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil


def sanitize(text: str) -> str:
    text = text.replace("/home/ma-user/work/numpilot/", "")
    text = text.replace("/home/ma-user/work/numpilot", ".")
    text = text.replace("/home/ma-user", "<home>")
    text = text.replace("/Users/degger", "<home>")
    text = re.sub(r'(?i)(auth_token\s*:\s*)"[^"]*"', r'\1"<redacted>"', text)
    text = re.sub(r'(?i)(base_url\s*:\s*)"[^"]*"', r'\1"<redacted>"', text)
    text = re.sub(r'(?i)Bearer\s+[A-Za-z0-9._-]+', "Bearer <redacted>", text)
    return text


def copy_text(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        sanitize(src.read_text(encoding="utf-8", errors="replace")),
        encoding="utf-8",
    )


def collect_json_usages(value, out: list[int]) -> None:
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, (int, float)) and not isinstance(total, bool):
                out.append(int(total))
        for item in value.values():
            collect_json_usages(item, out)
    elif isinstance(value, list):
        for item in value:
            collect_json_usages(item, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", help="benchmark case directory")
    parser.add_argument("custom_dir", help="complete-gaps custom-vN directory")
    parser.add_argument("evidence_dir", help="destination <case>/asc_operator evidence dir")
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    case_dir = Path(args.case_dir).resolve()
    custom = Path(args.custom_dir).resolve()
    dest = Path(args.evidence_dir).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    report = json.loads((custom / "gap-report.json").read_text(encoding="utf-8"))
    gaps = sorted((custom / "gaps").glob("*"))
    if not gaps:
        raise SystemExit("no generated gap directory")
    gap = gaps[0]
    copy_text(case_dir / "numpilot-work/asc_operator/numpy_source.py", dest / "numpy_source.py")
    copy_text(gap / "GAP_TASKBOOK.md", dest / "GAP_TASKBOOK.md")
    copy_text(gap / "GAP_TASKBOOK.json", dest / "GAP_TASKBOOK.json")
    copy_text(gap / "gap-spec.json", dest / "gap-spec.json")
    copy_text(custom / "observed-inputs.json", dest / "observed-inputs.json")
    copy_text(custom / "gap-report.json", dest / "gap-report.json")
    copy_text(custom / "generation-preflight.json", dest / "generation-preflight.json")
    draft = gap / "draft"
    if draft.is_dir():
        for source in sorted(draft.rglob("*")):
            if source.is_file():
                copy_text(source, dest / "draft" / source.relative_to(draft))
    usages: list[int] = []
    collect_json_usages(report, usages)
    preflight = custom / "generation-preflight.json"
    if preflight.is_file():
        try:
            collect_json_usages(json.loads(preflight.read_text()), usages)
        except (OSError, ValueError):
            pass
    drafts = report.get("drafts") or []
    model = ""
    attempts = None
    if drafts:
        generation = drafts[0].get("generation") or {}
        model = str(generation.get("configured_model") or generation.get("model") or "")
        attempts = generation.get("attempt_count") or generation.get("attempts")
    lines = [
        "# %s Ascend C 生成证据" % case_dir.name,
        "",
        "- 状态：`%s`" % report.get("status"),
        "- 缺口：`%s`" % (report.get("drafts") or [{}])[0].get("gap_id", "unknown"),
        "- 模型：`%s`" % (model or "deepseek-flash"),
        "- 模型调用轮次：`%s`" % (attempts if attempts is not None else "unknown"),
        "- 生成 tokens（可读 usage 合计）：`%d`" % sum(usages),
    ]
    if args.elapsed_seconds is not None:
        lines.append("- 生成墙钟：`%.1f s`" % args.elapsed_seconds)
    lines.extend((
        "",
        "目录：",
        "",
        "- `numpy_source.py`：放大的 NumPy 源程序；",
        "- `GAP_TASKBOOK.md/json`：缺口任务书；",
        "- `gap-spec.json`：缺口契约；",
        "- `draft/`：模型生成的 Ascend C 算子源码、适配层、构建脚本和单测；",
        "- `gap-report.json` / `generation-preflight.json`：生成回执。",
        "",
        "> 当前仅为 `generated_unverified`：尚未编译、部署、数值校验或激活。",
        "",
    ))
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(dest)
    print("status=%s tokens=%d" % (report.get("status"), sum(usages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
