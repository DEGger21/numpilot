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
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile


REQUIRED_FILES = {
    ".codeartsdoer/skills/ProjectSkillStatus.txt",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "README.zh-CN.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/COMPETITION-ONE-PAGER.md",
    "docs/README.md",
    "docs/RELEASE-CHECKLIST-DEMO-V1.md",
    "docs/SUBMISSION.md",
    "docs/architecture.md",
    "docs/assets/architecture/numpilot-architecture-en.svg",
    "docs/assets/architecture/numpilot-architecture-zh.svg",
    "docs/assets/brand/asnumpy-mark.png",
    "docs/assets/brand/ascend-wordmark.png",
    "docs/assets/brand/cann-wordmark.png",
    "docs/assets/brand/codearts-icon.png",
    "docs/assets/brand/numpilot-wordmark.png",
    "docs/assets/brand/README.md",
    "docs/installation.md",
    "docs/installation.zh-CN.md",
    "docs/numpilot-en-v6.svg",
    "docs/numpilot-en.pptx",
    "docs/numpilot-zh-v6.svg",
    "docs/numpilot-zh.pptx",
    "docs/numpilot-architecture.drawio",
    "evidence/MANIFEST.sha256",
    "evidence/README.md",
    "examples/poisson_source_smooth_l1_map.py",
    "install.sh",
    "numpilot/__init__.py",
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
    try:
        manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read evidence/MANIFEST.sha256: {error}")
        return
    for line_number, line in enumerate(manifest_lines, start=1):
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
        elif not target.is_file():
            errors.append(f"evidence manifest target is missing: {relative}")
        elif _sha256(target) != digest:
            errors.append(f"evidence hash mismatch: {relative}")


def _validate_versions(root: Path, errors: list[str]) -> str | None:
    required = (
        root / "pyproject.toml",
        root / "numpilot/__init__.py",
        root / "CHANGELOG.md",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        pyproject_text = required[0].read_text(encoding="utf-8")
        init_text = required[1].read_text(encoding="utf-8")
        changelog = required[2].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read release version metadata: {error}")
        return None
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", pyproject_text)
    version_match = (
        re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project_match.group(1), re.MULTILINE)
        if project_match is not None
        else None
    )
    version = version_match.group(1) if version_match is not None else None
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if not isinstance(version, str) or match is None or match.group(1) != version:
        errors.append("pyproject.toml and numpilot.__version__ are inconsistent")
        return version if isinstance(version, str) else None
    if f"## {version} " not in changelog:
        errors.append(f"CHANGELOG has no section for {version}")
    return version


def _validate_pptx(root: Path, errors: list[str]) -> None:
    for relative, planned_label in (
        ("docs/numpilot-en.pptx", "planned"),
        ("docs/numpilot-zh.pptx", "规划"),
    ):
        path = root / relative
        if not path.is_file():
            continue
        try:
            with ZipFile(path) as archive:
                bad = archive.testzip()
                names = set(archive.namelist())
                payloads = {
                    member: archive.read(member)
                    for member in {
                        "[Content_Types].xml",
                        "ppt/presentation.xml",
                        "ppt/slides/slide1.xml",
                    }
                    if member in names
                }
                notes_payload = (
                    archive.read("ppt/notesSlides/notesSlide1.xml")
                    if "ppt/notesSlides/notesSlide1.xml" in names
                    else None
                )
        except (BadZipFile, KeyError, NotImplementedError, OSError, RuntimeError):
            errors.append(f"invalid PPTX container or slide XML: {relative}")
            continue
        if bad is not None:
            errors.append(f"corrupt PPTX member in {relative}: {bad}")
        for required in {
            "[Content_Types].xml",
            "ppt/presentation.xml",
            "ppt/slides/slide1.xml",
        }:
            if required not in names:
                errors.append(f"{relative} is missing {required}")
        if len(payloads) != 3:
            continue
        try:
            content_types = ET.fromstring(payloads["[Content_Types].xml"])
            presentation = ET.fromstring(payloads["ppt/presentation.xml"])
            slide_element = ET.fromstring(payloads["ppt/slides/slide1.xml"])
        except ET.ParseError:
            errors.append(f"invalid PPTX XML payload: {relative}")
            continue
        expected_roots = (
            (
                content_types,
                "{http://schemas.openxmlformats.org/package/2006/content-types}Types",
            ),
            (
                presentation,
                "{http://schemas.openxmlformats.org/presentationml/2006/main}presentation",
            ),
            (
                slide_element,
                "{http://schemas.openxmlformats.org/presentationml/2006/main}sld",
            ),
        )
        if any(element.tag != expected for element, expected in expected_roots):
            errors.append(f"unexpected PPTX XML root element: {relative}")
            continue
        slide = " ".join(
            node.text or ""
            for node in slide_element.iter(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
            )
        )
        notes = ""
        if notes_payload is not None:
            try:
                notes_element = ET.fromstring(notes_payload)
            except ET.ParseError:
                errors.append(f"invalid PPTX notes XML: {relative}")
                continue
            notes = " ".join(
                node.text or ""
                for node in notes_element.iter(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
                )
            )
        architecture_text = slide + " " + notes
        if "ADAPTER_CANDIDATE" not in slide or planned_label not in slide:
            errors.append(f"{relative} does not expose the current/planned architecture boundary")
        for required_boundary in (
            "CANDIDATE_READY",
            "ops-nn",
            "CANNBot",
        ):
            if required_boundary not in architecture_text:
                errors.append(
                    f"{relative} omits route-specific boundary: {required_boundary}"
                )
        if "Closed-Loop Migration" in slide or "闭环迁移" in slide:
            errors.append(f"{relative} still presents the target loop as the current title")


def _validate_architecture_svg(root: Path, errors: list[str]) -> None:
    for relative in (
        "docs/assets/architecture/numpilot-architecture-en.svg",
        "docs/assets/architecture/numpilot-architecture-zh.svg",
        "docs/numpilot-en-v6.svg",
        "docs/numpilot-zh-v6.svg",
    ):
        path = root / relative
        if not path.is_file():
            continue
        try:
            tree = ET.parse(path)
        except (ET.ParseError, OSError):
            errors.append(f"invalid architecture SVG: {relative}")
            continue
        element = tree.getroot()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            errors.append(f"cannot read architecture SVG: {relative}")
            continue
        if element.tag != "{http://www.w3.org/2000/svg}svg":
            errors.append(f"architecture asset is not SVG: {relative}")
        required_boundaries = (
            "ADAPTER_CANDIDATE",
            "CAPABILITY_AUDIT_REQUIRED",
            "REVIEWED_MATMUL_ONLY",
            "SOURCE_EVIDENCE_ONLY",
            "UNVERIFIED_OPERATOR_DRAFT",
        )
        if any(boundary not in text for boundary in required_boundaries):
            errors.append(f"architecture SVG omits required capability boundaries: {relative}")


def _scan_public_text(root: Path, files: Iterable[str], errors: list[str]) -> None:
    for relative in files:
        path = PurePosixPath(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not (
            relative in {"README.md", "README.zh-CN.md"}
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

    for relative in ("README.md", "README.zh-CN.md"):
        path = root / relative
        if not path.is_file():
            continue
        try:
            readme = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot read {relative}: {error}")
            continue
        if "<img" not in readme or "development--snapshot" not in readme:
            errors.append(
                f"{relative} must retain rendered badges and an honest development-stage label"
            )
        if "docs/assets/architecture/" not in readme or "docs/numpilot-" not in readme:
            errors.append(f"{relative} must link the reviewed SVG and PPT architecture assets")
        if "release/competition-submission" in readme:
            warnings.append(f"{relative} is bound to a retired release branch")

    manifest_path = root / "MANIFEST.in"
    if manifest_path.is_file():
        try:
            manifest_in = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot read MANIFEST.in: {error}")
        else:
            for stale in (
                "demotest-case01",
                "heat-diffusion-",
                "ops-nn-matmul-",
                "poisson-ritz-p0-",
            ):
                if stale in manifest_in:
                    errors.append(f"MANIFEST.in still packages historical material: {stale}")

    _scan_public_text(root, files, errors)
    _validate_evidence_manifest(root, errors)
    version = _validate_versions(root, errors)
    _validate_pptx(root, errors)
    _validate_architecture_svg(root, errors)

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
