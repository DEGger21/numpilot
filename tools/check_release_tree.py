#!/usr/bin/env python3
"""Fail-closed audit for the NumPilot competition release tree.

The audit is read-only and uses only the Python standard library.  It checks
the Git candidate set rather than the raw working directory, so ignored local
venvs and build caches cannot be mistaken for submission contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Iterable
from zipfile import BadZipFile, ZipFile


REQUIRED_FILES = {
    ".codeartsdoer/skills/ProjectSkillStatus.txt",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/COMPETITION-ONE-PAGER.md",
    "docs/README.md",
    "docs/RELEASE-CHECKLIST-DEMO-V1.md",
    "docs/SUBMISSION.md",
    "docs/architecture.md",
    "docs/numpilot-architecture-paper-style-zh-readable.pptx",
    "docs/numpilot-architecture.drawio",
    "evidence/MANIFEST.sha256",
    "evidence/README.md",
    "examples/poisson_source_smooth_l1_map.py",
    "install.sh",
    "pyproject.toml",
    "run_demo.sh",
    "tools/audit_script_benchmark_receipt.py",
    "tools/check_release_tree.py",
    "tools/release_artifact_manifest.py",
    "tools/render_demo_summary.py",
}

FORBIDDEN_EXACT = {
    "NumPilot20260615-v1.pdf",
    "asnumpy_cpp.log",
    "run_poisson_demo.sh",
    "todo.md",
}

FORBIDDEN_PREFIXES = (
    "docs/superpowers/",
    "evidence/demotest-case01-",
    "evidence/heat-diffusion-",
    "evidence/ops-nn-matmul-",
    "evidence/poisson-ritz-p0-",
)

FORBIDDEN_BASENAME_FRAGMENTS = (
    "AI-HANDOFF",
    "NEXT-AI-PROMPT",
    "WORK-IN-PROGRESS",
    "WIP-HANDOFF",
)

FORBIDDEN_SUFFIXES = {
    ".db",
    ".log",
    ".npy",
    ".npz",
    ".orig",
    ".pyc",
    ".tmp",
}

PUBLIC_EVIDENCE_CASES = {
    "competition-ordinary-20260828",
    "competition-poisson-stage-20260828",
}

TEXT_SUFFIXES = {".ipynb", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
PRIVATE_PATH_RE = re.compile(r"/home/(?:ma-user|lixing)(?:/|\\b)")
MAX_FILE_BYTES = 2 * 1024 * 1024


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _candidate_files(root: Path) -> list[str]:
    result = _run_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail}")
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", "surrogateescape")
        target = root / relative
        if target.is_file() or target.is_symlink():
            paths.append(PurePosixPath(relative).as_posix())
    return sorted(set(paths))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_path_reason(relative: str) -> str | None:
    path = PurePosixPath(relative)
    if relative in FORBIDDEN_EXACT:
        return "forbidden release filename"
    if any(relative.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return "historical evidence must stay outside the submission branch"
    if any(fragment in path.name for fragment in FORBIDDEN_BASENAME_FRAGMENTS):
        return "internal handoff/WIP document"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"intermediate artifact suffix {path.suffix.lower()}"
    if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
        return "cache directory"
    if any(part in {"build", "dist"} or part.endswith(".egg-info") for part in path.parts):
        return "build output directory"
    return None


def _validate_evidence_manifest(root: Path, errors: list[str]) -> None:
    evidence = root / "evidence"
    manifest = evidence / "MANIFEST.sha256"
    if not manifest.is_file():
        errors.append("missing evidence/MANIFEST.sha256")
        return

    actual_children = {path.name for path in evidence.iterdir()}
    expected_children = {"README.md", "MANIFEST.sha256", *PUBLIC_EVIDENCE_CASES}
    if actual_children != expected_children:
        errors.append(
            "public evidence root mismatch: "
            f"expected {sorted(expected_children)}, got {sorted(actual_children)}"
        )

    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid evidence manifest line {line_number}")
            continue
        entries.append((digest, relative))

    relatives = [relative for _, relative in entries]
    expected = sorted(
        str(path.relative_to(root))
        for path in evidence.rglob("*")
        if path.is_file() and path != manifest
    )
    if relatives != expected:
        errors.append("evidence manifest is incomplete, duplicated, or not sorted")
        return
    for digest, relative in entries:
        target = root / relative
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid SHA-256 in evidence manifest: {relative}")
        elif _sha256(target) != digest:
            errors.append(f"evidence hash mismatch: {relative}")


def _validate_versions(root: Path, errors: list[str]) -> str | None:
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", pyproject_text)
    version_match = (
        re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project_match.group(1), re.MULTILINE)
        if project_match is not None
        else None
    )
    version = version_match.group(1) if version_match is not None else None
    init_text = (root / "numpilot/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if not isinstance(version, str) or match is None or match.group(1) != version:
        errors.append("pyproject.toml and numpilot.__version__ are inconsistent")
        return version if isinstance(version, str) else None
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {version} " not in changelog:
        errors.append(f"CHANGELOG has no section for {version}")
    return version


def _validate_pptx(root: Path, errors: list[str]) -> None:
    path = root / "docs/numpilot-architecture-paper-style-zh-readable.pptx"
    if not path.is_file():
        return
    try:
        with ZipFile(path) as archive:
            bad = archive.testzip()
            names = set(archive.namelist())
    except BadZipFile:
        errors.append(f"invalid PPTX ZIP container: {path.relative_to(root)}")
        return
    if bad is not None:
        errors.append(f"corrupt PPTX member: {bad}")
    for required in {"[Content_Types].xml", "ppt/presentation.xml", "ppt/slides/slide1.xml"}:
        if required not in names:
            errors.append(f"PPTX missing {required}")


def _scan_public_text(root: Path, files: Iterable[str], errors: list[str]) -> None:
    for relative in files:
        path = PurePosixPath(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not (
            relative == "README.md"
            or relative.startswith("docs/")
            or relative in {"SECURITY.md", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md"}
        ):
            continue
        text = (root / relative).read_text(encoding="utf-8", errors="replace")
        if PRIVATE_PATH_RE.search(text):
            errors.append(f"private absolute workspace path in public text: {relative}")


def audit_release_tree(root: Path, *, require_clean: bool = False) -> dict:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        files = _candidate_files(root)
    except RuntimeError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "file_count": 0}

    file_set = set(files)
    for missing in sorted(REQUIRED_FILES - file_set):
        errors.append(f"missing required release file: {missing}")

    for relative in files:
        target = root / relative
        reason = _forbidden_path_reason(relative)
        if reason:
            errors.append(f"{relative}: {reason}")
        if target.is_symlink():
            errors.append(f"symlink is not allowed in release tree: {relative}")
        elif target.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "<img" not in readme or "development--snapshot" not in readme:
        errors.append("README must retain rendered badges and an honest development-stage label")
    if "release/competition-submission" in readme:
        warnings.append("README is branch-bound; replace with an immutable tag before final RC")

    manifest_in = (root / "MANIFEST.in").read_text(encoding="utf-8")
    for stale in ("demotest-case01", "heat-diffusion-", "ops-nn-matmul-", "poisson-ritz-p0-"):
        if stale in manifest_in:
            errors.append(f"MANIFEST.in still packages historical material: {stale}")

    _scan_public_text(root, files, errors)
    _validate_evidence_manifest(root, errors)
    version = _validate_versions(root, errors)
    _validate_pptx(root, errors)

    status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        errors.append("git status failed")
    elif status.stdout:
        message = "working tree is not clean"
        (errors if require_clean else warnings).append(message)

    return {
        "schema": "numpilot.release_tree_audit.v1",
        "ok": not errors,
        "version": version,
        "file_count": len(files),
        "total_bytes": sum((root / path).stat().st_size for path in files if not (root / path).is_symlink()),
        "public_evidence_cases": sorted(PUBLIC_EVIDENCE_CASES),
        "errors": errors,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_release_tree(args.repo_root, require_clean=args.require_clean)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        label = "PASS" if report["ok"] else "FAIL"
        print(f"release-tree {label}: {report['file_count']} files, {report['total_bytes']} bytes")
        for warning in report["warnings"]:
            print(f"warning: {warning}")
        for error in report["errors"]:
            print(f"error: {error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
