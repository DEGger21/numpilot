#!/usr/bin/env python3
"""Collect build/deploy/validation/adapter/benchmark evidence for one case."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

TEXT_SUFFIXES = {".json", ".md", ".txt", ".log", ".py", ".cpp", ".hpp", ".h", ".sh", ".jsonl", ".yaml", ".yml"}


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
    if src.suffix.lower() in TEXT_SUFFIXES or src.name in {"status.txt"}:
        dst.write_text(sanitize(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def latest(root: Path, pattern: str) -> Path | None:
    candidates = sorted(path for path in root.glob(pattern) if path.is_dir())
    return candidates[-1] if candidates else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir")
    parser.add_argument("evidence_dir")
    args = parser.parse_args(argv)
    stage = Path(args.stage_dir).resolve()
    dest = Path(args.evidence_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    # Repaired generated draft is the authoritative source for this stage.
    draft = latest(stage / "gaps", "*/draft")
    if draft:
        shutil.copytree(draft, dest / "draft", dirs_exist_ok=True)
        for source in draft.rglob("*"):
            if source.is_file():
                copy_text(source, dest / "draft" / source.relative_to(draft))
    for name in ("gap-report.json", "generation-preflight.json", "observed-inputs.json", "skill-bundle.json", "metrics.json"):
        copy_text(stage / name, dest / name)
    repairs = stage / "repairs"
    if repairs.is_dir():
        for source in repairs.rglob("*"):
            if source.is_file():
                copy_text(source, dest / "repairs" / source.relative_to(repairs))
    build = latest(stage, "build-v*")
    if build:
        for name in ("status.txt", "build.log"):
            copy_text(build / name, dest / "build" / name)
        manifests = sorted((build / "project/build").glob("*_build_manifest.txt"))
        if manifests:
            copy_text(manifests[0], dest / "build" / manifests[0].name)
    deploy = latest(stage, "deploy-v*")
    if deploy:
        for name in ("status.txt", "install.log"):
            copy_text(deploy / name, dest / "deploy" / name)
    verify = latest(stage, "verify-v*")
    if verify:
        for source in verify.rglob("*"):
            if source.is_file():
                copy_text(source, dest / "verify" / source.name)
    copy_text(stage / "verify-v1/contract-audit.json", dest / "verify/contract-audit.json")
    adapter = stage / "adapter-build-v1"
    if adapter.is_dir():
        for name in ("status.txt", "wheel.log", "stage.log"):
            copy_text(adapter / name, dest / "adapter" / name)
        for wheel in (adapter / "wheelhouse").glob("asnumpy-*.whl"):
            copy_text(wheel, dest / "adapter" / wheel.name)
        tree = adapter / "asnumpy"
        for rel in (
            "csrc/custom_ops",
            "include/asnumpy/custom_ops",
            "src/asnumpy/custom_ops.py",
            "src/asnumpy/array.py",
            "bindings/python/bind_custom_ops.cpp",
            "bindings/python/module.cpp",
        ):
            source = tree / rel
            if source.is_dir():
                for item in source.rglob("*"):
                    if item.is_file():
                        copy_text(item, dest / "adapter/source" / item.relative_to(tree))
            else:
                copy_text(source, dest / "adapter/source" / rel)
    bench = stage / "bench-v1"
    if bench.is_dir():
        for name in ("benchmark.json", "route-check.json", "migrated.py", "numpy_source.py", "source-adaptation.json"):
            copy_text(bench / name, dest / "benchmark" / name)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
