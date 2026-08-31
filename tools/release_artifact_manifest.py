#!/usr/bin/env python3
"""Generate and verify a fail-closed NumPilot release artifact manifest.

This tool intentionally depends only on the Python standard library.  It does
not extract or execute archive members.  The generated JSON binds wheel/sdist
bytes, their complete regular-file inventories, core package metadata, and the
Git source identity used for the build.

A dirty source tree is recorded as ``development_only``.  Pass
``--require-clean`` for an RC/release gate.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping
import xml.etree.ElementTree as ET


SCHEMA = "numpilot.release-artifact-manifest"
SCHEMA_VERSION = 1
FORBIDDEN_BASENAMES = {
    ".env",
    "NumPilot20260615-v1.pdf",
    "llm.json",
    "todo.md",
}
FORBIDDEN_PATH_PARTS = {
    ".git",
    ".numpilot-cache",
    ".numpilot-config",
    ".numpilot-state",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".db", ".log", ".npy", ".npz", ".orig", ".pyc", ".pyo", ".tmp"}
REVIEWED_ASSETS = (
    "docs/numpilot-en.pptx",
    "docs/numpilot-zh.pptx",
    "docs/numpilot-en-v6.svg",
    "docs/numpilot-zh-v6.svg",
    "docs/assets/architecture/numpilot-architecture-en.svg",
    "docs/assets/architecture/numpilot-architecture-zh.svg",
    "docs/assets/brand/asnumpy-mark.png",
    "docs/assets/brand/ascend-wordmark.png",
    "docs/assets/brand/cann-wordmark.png",
    "docs/assets/brand/codearts-icon.png",
    "docs/assets/brand/numpilot-wordmark.png",
)


class ReleaseManifestError(RuntimeError):
    """Raised when artifacts cannot support a release manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReleaseManifestError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def _stream_identity(
    stream: BinaryIO, *, retain: bool = False
) -> tuple[int, str, bytes | None]:
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain else None
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
    return size, digest.hexdigest(), None if chunks is None else b"".join(chunks)


def _safe_member_name(raw_name: str) -> str:
    _require(bool(raw_name), "archive contains an empty member name")
    _require("\x00" not in raw_name, f"archive member contains NUL: {raw_name!r}")
    _require("\\" not in raw_name, f"archive member uses backslashes: {raw_name!r}")
    name = raw_name[:-1] if raw_name.endswith("/") else raw_name
    _require(bool(name), f"archive member has an unsafe path: {raw_name!r}")
    _require(not name.startswith("/"), f"archive member is absolute: {raw_name!r}")
    raw_parts = name.split("/")
    _require(
        all(part not in {"", ".", ".."} for part in raw_parts),
        f"archive member has an unsafe path: {raw_name!r}",
    )
    _require(
        not re.match(r"^[A-Za-z]:", raw_parts[0]),
        f"archive member uses a drive-qualified path: {raw_name!r}",
    )
    normalized = PurePosixPath(name).as_posix()
    _require(normalized == name, f"archive member path is not canonical: {raw_name!r}")
    return normalized


def _check_allowed_member(path: str) -> None:
    parts = PurePosixPath(path).parts
    _require(
        not any(part in FORBIDDEN_PATH_PARTS for part in parts),
        f"archive contains a private/cache path: {path}",
    )
    _require(
        parts[-1] not in FORBIDDEN_BASENAMES,
        f"archive contains a forbidden release file: {path}",
    )
    suffix = PurePosixPath(parts[-1]).suffix.lower()
    _require(suffix not in FORBIDDEN_SUFFIXES, f"intermediate artifact in archive: {path}")


def _file_record(path: str, size: int, digest: str) -> dict[str, Any]:
    return {"path": path, "sha256": digest, "size_bytes": size}


def _retain_member_payload(member: str) -> bool:
    return (
        member.endswith("/METADATA")
        or member.endswith("/PKG-INFO")
        or any(member.endswith("/" + suffix) for suffix in REVIEWED_ASSETS)
    )


def _read_zip(path: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    files: list[dict[str, Any]] = []
    metadata: dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = _safe_member_name(info.filename)
                _require(member not in seen, f"duplicate archive member: {member}")
                seen.add(member)
                _check_allowed_member(member)
                unix_mode = info.external_attr >> 16
                _require(
                    not stat.S_ISLNK(unix_mode),
                    f"symbolic link is not allowed in release archive: {member}",
                )
                if info.is_dir():
                    continue
                _require(
                    not (info.flag_bits & 0x1),
                    f"encrypted archive member is not allowed: {member}",
                )
                retain = _retain_member_payload(member)
                with archive.open(info, "r") as stream:
                    size, digest, payload = _stream_identity(stream, retain=retain)
                _require(size == info.file_size, f"archive size mismatch: {member}")
                files.append(_file_record(member, size, digest))
                if payload is not None:
                    metadata[member] = payload
    except (OSError, zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise ReleaseManifestError(f"cannot inspect zip archive {path}: {error}") from error
    return sorted(files, key=lambda item: item["path"]), metadata


def _read_tar(path: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    files: list[dict[str, Any]] = []
    metadata: dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for info in archive:
                member = _safe_member_name(info.name)
                _require(member not in seen, f"duplicate archive member: {member}")
                seen.add(member)
                _check_allowed_member(member)
                if info.isdir():
                    continue
                _require(
                    info.isfile(),
                    f"links and special files are not allowed in release archive: {member}",
                )
                stream = archive.extractfile(info)
                _require(stream is not None, f"cannot read archive member: {member}")
                retain = _retain_member_payload(member)
                with stream:
                    size, digest, payload = _stream_identity(stream, retain=retain)
                _require(size == info.size, f"archive size mismatch: {member}")
                files.append(_file_record(member, size, digest))
                if payload is not None:
                    metadata[member] = payload
    except (OSError, tarfile.TarError) as error:
        raise ReleaseManifestError(f"cannot inspect tar archive {path}: {error}") from error
    return sorted(files, key=lambda item: item["path"]), metadata


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".whl"):
        return "wheel"
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".zip")):
        return "sdist"
    raise ReleaseManifestError(f"unsupported distribution archive: {path.name}")


def _canonical_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _parse_metadata(payload: bytes, source: str) -> dict[str, Any]:
    try:
        message = BytesParser(policy=policy.default).parsebytes(payload)
    except (TypeError, ValueError) as error:
        raise ReleaseManifestError(f"invalid package metadata in {source}: {error}") from error
    name = str(message.get("Name", "")).strip()
    version = str(message.get("Version", "")).strip()
    _require(bool(name), f"package metadata has no Name: {source}")
    _require(bool(version), f"package metadata has no Version: {source}")
    return {
        "license_expression": str(message.get("License-Expression", "")).strip() or None,
        "license_files": sorted(str(value) for value in message.get_all("License-File", [])),
        "name": name,
        "requires_dist": sorted(str(value) for value in message.get_all("Requires-Dist", [])),
        "requires_python": str(message.get("Requires-Python", "")).strip() or None,
        "version": version,
    }


def _select_metadata(
    kind: str, metadata: Mapping[str, bytes]
) -> tuple[str, dict[str, Any]]:
    if kind == "wheel":
        candidates = [name for name in metadata if name.endswith(".dist-info/METADATA")]
        _require(len(candidates) == 1, "wheel must contain exactly one dist-info/METADATA")
    else:
        candidates = [name for name in metadata if name.endswith("/PKG-INFO")]
        _require(bool(candidates), "sdist does not contain PKG-INFO")
        depth = min(len(PurePosixPath(name).parts) for name in candidates)
        candidates = [name for name in candidates if len(PurePosixPath(name).parts) == depth]
        _require(len(candidates) == 1, "sdist has ambiguous root PKG-INFO metadata")
    selected = candidates[0]
    return selected, _parse_metadata(metadata[selected], selected)


def _payload_by_suffix(payloads: Mapping[str, bytes], suffix: str) -> bytes:
    matches = [
        payload
        for path, payload in payloads.items()
        if path == suffix or path.endswith("/" + suffix)
    ]
    _require(len(matches) == 1, f"sdist has ambiguous or missing reviewed asset: {suffix}")
    return matches[0]


def _validate_pptx_payload(payload: bytes, suffix: str) -> None:
    required = {
        "[Content_Types].xml": (
            "{http://schemas.openxmlformats.org/package/2006/content-types}Types"
        ),
        "ppt/presentation.xml": (
            "{http://schemas.openxmlformats.org/presentationml/2006/main}presentation"
        ),
        "ppt/slides/slide1.xml": (
            "{http://schemas.openxmlformats.org/presentationml/2006/main}sld"
        ),
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            bad = archive.testzip()
            _require(bad is None, f"corrupt PPTX member in {suffix}: {bad}")
            names = set(archive.namelist())
            missing = sorted(set(required) - names)
            _require(not missing, f"{suffix} is missing PPTX members: {', '.join(missing)}")
            elements = {
                member: ET.fromstring(archive.read(member)) for member in required
            }
            notes = (
                ET.fromstring(archive.read("ppt/notesSlides/notesSlide1.xml"))
                if "ppt/notesSlides/notesSlide1.xml" in names
                else None
            )
    except (
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as error:
        raise ReleaseManifestError(f"invalid PPTX asset {suffix}: {error}") from error
    for member, expected_root in required.items():
        _require(
            elements[member].tag == expected_root,
            f"unexpected XML root in {suffix}:{member}",
        )
    slide = " ".join(
        node.text or ""
        for node in elements["ppt/slides/slide1.xml"].iter(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        )
    )
    note_text = ""
    if notes is not None:
        note_text = " ".join(
            node.text or ""
            for node in notes.iter(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
            )
        )
    architecture_text = slide + " " + note_text
    planned = "规划" if suffix.endswith("-zh.pptx") else "planned"
    _require(
        "ADAPTER_CANDIDATE" in slide and planned in slide,
        f"{suffix} omits the reviewed current/planned capability boundary",
    )
    for boundary in ("CANDIDATE_READY", "ops-nn", "CANNBot"):
        _require(
            boundary in architecture_text,
            f"{suffix} omits route-specific boundary: {boundary}",
        )


def _validate_svg_payload(payload: bytes, suffix: str) -> None:
    try:
        element = ET.fromstring(payload)
        text = payload.decode("utf-8")
    except (ET.ParseError, UnicodeError) as error:
        raise ReleaseManifestError(f"invalid SVG asset {suffix}: {error}") from error
    _require(
        element.tag == "{http://www.w3.org/2000/svg}svg",
        f"architecture asset is not SVG: {suffix}",
    )
    required_boundaries = (
        "ADAPTER_CANDIDATE",
        "CAPABILITY_AUDIT_REQUIRED",
        "REVIEWED_MATMUL_ONLY",
        "SOURCE_EVIDENCE_ONLY",
        "UNVERIFIED_OPERATOR_DRAFT",
    )
    _require(
        all(boundary in text for boundary in required_boundaries),
        f"{suffix} omits the reviewed capability boundary",
    )


def _validate_png_payload(payload: bytes, suffix: str) -> None:
    _require(payload.startswith(b"\x89PNG\r\n\x1a\n"), f"invalid PNG signature: {suffix}")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(payload):
        _require(offset + 12 <= len(payload), f"truncated PNG chunk: {suffix}")
        size = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + size
        _require(end <= len(payload), f"truncated PNG payload: {suffix}")
        data = payload[offset + 8 : offset + 8 + size]
        expected_crc = int.from_bytes(payload[offset + 8 + size : end], "big")
        actual_crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        _require(actual_crc == expected_crc, f"PNG CRC mismatch: {suffix}")
        chunks.append((chunk_type, data))
        offset = end
        if chunk_type == b"IEND":
            break
    _require(offset == len(payload), f"trailing bytes after PNG IEND: {suffix}")
    _require(
        bool(chunks) and chunks[0][0] == b"IHDR" and len(chunks[0][1]) == 13,
        f"invalid PNG header: {suffix}",
    )
    _require(any(kind == b"IDAT" for kind, _ in chunks), f"PNG has no IDAT: {suffix}")
    _require(chunks[-1] == (b"IEND", b""), f"PNG has no terminal IEND: {suffix}")
    width = int.from_bytes(chunks[0][1][0:4], "big")
    height = int.from_bytes(chunks[0][1][4:8], "big")
    _require(width > 0 and height > 0, f"invalid PNG dimensions: {suffix}")


def _validate_reviewed_assets(payloads: Mapping[str, bytes]) -> None:
    for suffix in REVIEWED_ASSETS:
        payload = _payload_by_suffix(payloads, suffix)
        if suffix.endswith(".pptx"):
            _validate_pptx_payload(payload, suffix)
        elif suffix.endswith(".svg"):
            _validate_svg_payload(payload, suffix)
        else:
            _validate_png_payload(payload, suffix)


def _check_required_members(
    kind: str,
    files: list[dict[str, Any]],
    name: str,
    version: str,
    metadata_path: str,
) -> None:
    paths = {str(item["path"]) for item in files}
    package = _canonical_project_name(name).replace("-", "_")
    common = [
        f"{package}/__init__.py",
        f"{package}/__main__.py",
        f"{package}/_script_benchmark_worker.py",
        f"{package}/_script_verification_worker.py",
        f"{package}/script_benchmark.py",
        f"{package}/script_migration.py",
        f"{package}/script_verification.py",
        f"{package}/agents/analysis.py",
        f"{package}/agents/decision.py",
        f"{package}/agents/migration.py",
        f"{package}/agents/verification.py",
    ]
    installed_skill_data = [
        "share/numpilot/codearts/ProjectSkillStatus.txt",
        "share/numpilot/codearts/agents/numpilot-analyzer.md",
        "share/numpilot/codearts/agents/numpilot-evidence-auditor.md",
        "share/numpilot/codearts/agents/numpilot-operator-engineer.md",
        "share/numpilot/codearts/agents/numpilot-orchestrator.md",
        "share/numpilot/skills/numpilot-migration-gate/SKILL.md",
        "share/numpilot/skills/asnumpy-custom-op-integration/SKILL.md",
        "share/numpilot/skills/asnumpy-custom-op-integration/agents/openai.yaml",
        "share/numpilot/skills/asnumpy-custom-op-integration/references/legacy-case-lessons.md",
        "share/numpilot/skills/asnumpy-custom-op-integration/references/manifest-workflow.md",
        "share/numpilot/skills/asnumpy-custom-op-integration/references/official-aclnn-existing-module.md",
        "share/numpilot/skills/asnumpy-custom-op-integration/references/qualification-and-debug.md",
    ]
    if kind == "wheel":
        metadata_parts = PurePosixPath(metadata_path).parts
        _require(
            len(metadata_parts) == 2 and metadata_parts[1] == "METADATA",
            "wheel metadata must be in a top-level dist-info directory",
        )
        dist_info = metadata_parts[0]
        expected_dist_info = f"{package}-{version}.dist-info"
        _require(
            dist_info == expected_dist_info,
            f"wheel dist-info root mismatch: expected {expected_dist_info}, got {dist_info}",
        )
        data_root = f"{package}-{version}.data/data"
        required = common + [
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/RECORD",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md",
            *(f"{data_root}/{member}" for member in installed_skill_data),
        ]
        allowed_roots = (f"{package}/", f"{dist_info}/", f"{data_root}/")
        misplaced = sorted(
            path for path in paths if not any(path.startswith(root) for root in allowed_roots)
        )
        _require(
            not misplaced,
            f"wheel contains members outside canonical roots: {', '.join(misplaced[:5])}",
        )
    else:
        source_root = f"{_canonical_project_name(name)}-{version}"
        _require(
            metadata_path == f"{source_root}/PKG-INFO",
            f"sdist root mismatch: expected {source_root}",
        )
        misplaced = sorted(
            path for path in paths if not path.startswith(source_root + "/")
        )
        _require(
            not misplaced,
            f"sdist contains members outside canonical root: {', '.join(misplaced[:5])}",
        )
        source_required = common + [
            "LICENSE",
            "MANIFEST.in",
            "README.md",
            "README.zh-CN.md",
            "THIRD_PARTY_NOTICES.md",
            "pyproject.toml",
            "run_demo.sh",
            "docs/installation.md",
            "docs/installation.zh-CN.md",
            "docs/numpilot-en.pptx",
            "docs/numpilot-zh.pptx",
            "docs/numpilot-en-v6.svg",
            "docs/numpilot-zh-v6.svg",
            "docs/assets/architecture/numpilot-architecture-en.svg",
            "docs/assets/architecture/numpilot-architecture-zh.svg",
            "docs/assets/brand/asnumpy-mark.png",
            "docs/assets/brand/ascend-wordmark.png",
            "docs/assets/brand/cann-wordmark.png",
            "docs/assets/brand/codearts-icon.png",
            "docs/assets/brand/numpilot-wordmark.png",
            "docs/assets/brand/README.md",
            "docs/FRESH-NOTEBOOK-QUICKSTART.md",
            "docs/NumPilot-Demo-v1-Quickstart.ipynb",
            "examples/competition_power_log_exp.py",
            "evidence/competition-ordinary-20260828/qualified-analysis.json",
            "evidence/competition-ordinary-20260828/script-benchmark-report.json",
            "evidence/competition-ordinary-20260828/script-verification-report.json",
            "evidence/competition-poisson-stage-20260828/qualified-analysis.json",
            "evidence/competition-poisson-stage-20260828/script-benchmark-report.json",
            "evidence/competition-poisson-stage-20260828/script-verification-report.json",
            "evidence/MANIFEST.sha256",
            ".codeartsdoer/agents/numpilot-analyzer.md",
            ".codeartsdoer/agents/numpilot-evidence-auditor.md",
            ".codeartsdoer/agents/numpilot-operator-engineer.md",
            ".codeartsdoer/agents/numpilot-orchestrator.md",
            ".codeartsdoer/skills/ProjectSkillStatus.txt",
            ".codeartsdoer/skills/numpilot-migration-gate/SKILL.md",
            ".codeartsdoer/skills/asnumpy-custom-op-integration/SKILL.md",
            ".codeartsdoer/skills/asnumpy-custom-op-integration/agents/openai.yaml",
            ".codeartsdoer/skills/asnumpy-custom-op-integration/references/legacy-case-lessons.md",
            ".codeartsdoer/skills/asnumpy-custom-op-integration/references/manifest-workflow.md",
            ".codeartsdoer/skills/asnumpy-custom-op-integration/references/official-aclnn-existing-module.md",
            ".codeartsdoer/skills/asnumpy-custom-op-integration/references/qualification-and-debug.md",
            "tools/check_release_tree.py",
            "tools/release_artifact_manifest.py",
        ]
        required = [f"{source_root}/{member}" for member in source_required]
    missing = [path for path in required if path not in paths]
    _require(not missing, f"{kind} is missing required members: {', '.join(missing)}")


def inspect_artifact(
    artifact: str | Path, *, expected_name: str, expected_version: str
) -> dict[str, Any]:
    path = Path(artifact).resolve()
    _require(path.is_file(), f"release artifact does not exist: {path}")
    kind = _artifact_kind(path)
    if kind == "wheel" or path.name.lower().endswith(".zip"):
        files, retained_payloads = _read_zip(path)
    else:
        files, retained_payloads = _read_tar(path)
    _require(bool(files), f"release artifact is empty: {path.name}")
    metadata_path, metadata = _select_metadata(kind, retained_payloads)
    _require(
        _canonical_project_name(metadata["name"])
        == _canonical_project_name(expected_name),
        f"artifact project mismatch: expected {expected_name}, got {metadata['name']}",
    )
    _require(
        metadata["version"] == expected_version,
        f"artifact version mismatch: expected {expected_version}, got {metadata['version']}",
    )
    _check_required_members(
        kind,
        files,
        expected_name,
        expected_version,
        metadata_path,
    )
    if kind == "sdist":
        _validate_reviewed_assets(retained_payloads)
    return {
        "filename": path.name,
        "kind": kind,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "regular_file_count": len(files),
        "regular_files": files,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_project(repo_root: Path) -> dict[str, str]:
    pyproject = repo_root / "pyproject.toml"
    try:
        lines = pyproject.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ReleaseManifestError(f"cannot read project metadata: {error}") from error
    in_project = False
    values: dict[str, str] = {}
    assignment = re.compile(
        r"^(name|version)\s*=\s*(?P<value>(?:\"(?:[^\"\\]|\\.)*\"|'[^']*'))"
        r"\s*(?:#.*)?$"
    )
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if not in_project or not line or line.startswith("#"):
            continue
        match = assignment.fullmatch(line)
        if match is None:
            continue
        try:
            parsed = ast.literal_eval(match.group("value"))
        except (SyntaxError, ValueError) as error:
            raise ReleaseManifestError(
                f"invalid static [project] assignment: {line}"
            ) from error
        _require(isinstance(parsed, str) and bool(parsed), f"empty project {match[1]}")
        values[match[1]] = parsed
    _require(bool(values.get("name")), "pyproject project.name is missing or dynamic")
    _require(bool(values.get("version")), "pyproject project.version is missing or dynamic")
    return {"name": values["name"], "version": values["version"]}


def _git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise ReleaseManifestError(f"cannot inspect Git provenance: {detail.strip()}") from error
    return completed.stdout.strip()


def _source_provenance(repo_root: Path) -> dict[str, Any]:
    _require(repo_root.is_dir(), f"repository root does not exist: {repo_root}")
    _require(
        _git(repo_root, "rev-parse", "--is-inside-work-tree") == "true",
        f"not a Git work tree: {repo_root}",
    )
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    changes = [] if not status else status.splitlines()
    tags = _git(repo_root, "tag", "--points-at", "HEAD").splitlines()
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "commit_time": _git(repo_root, "show", "-s", "--format=%cI", "HEAD"),
        "dirty": bool(changes),
        "dirty_entry_count": len(changes),
        "exact_tags": sorted(tag for tag in tags if tag),
        "pyproject_sha256": _sha256(repo_root / "pyproject.toml"),
    }


def _consistent_metadata(artifacts: list[dict[str, Any]]) -> None:
    _require(bool(artifacts), "at least one release artifact is required")
    keys = ("name", "version", "requires_python", "requires_dist", "license_expression")
    reference = artifacts[0]["metadata"]
    for artifact in artifacts[1:]:
        metadata = artifact["metadata"]
        mismatches = [key for key in keys if metadata.get(key) != reference.get(key)]
        _require(
            not mismatches,
            f"wheel/sdist metadata disagree for: {', '.join(mismatches)}",
        )


def _bind_reviewed_source_assets(
    artifacts: list[dict[str, Any]], repo_root: Path
) -> dict[str, str]:
    sdists = [artifact for artifact in artifacts if artifact["kind"] == "sdist"]
    if not sdists:
        return {}
    _require(len(sdists) == 1, "release manifest must contain at most one sdist")
    records = sdists[0]["regular_files"]
    bindings: dict[str, str] = {}
    for relative in REVIEWED_ASSETS:
        source = repo_root / relative
        _require(source.is_file(), f"reviewed source asset is missing: {relative}")
        source_digest = _sha256(source)
        matches = [
            item
            for item in records
            if item["path"] == relative or item["path"].endswith("/" + relative)
        ]
        _require(len(matches) == 1, f"sdist asset identity is ambiguous: {relative}")
        _require(
            matches[0]["sha256"] == source_digest,
            f"sdist asset differs from reviewed source: {relative}",
        )
        bindings[relative] = source_digest
    return bindings


def generate_manifest(
    artifacts: Iterable[str | Path],
    *,
    repo_root: str | Path,
    require_clean: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    project = _load_project(root)
    artifact_paths = [Path(path).resolve() for path in artifacts]
    _require(bool(artifact_paths), "at least one release artifact is required")
    filenames = [path.name for path in artifact_paths]
    _require(len(filenames) == len(set(filenames)), "artifact filenames must be unique")
    inspected = [
        inspect_artifact(
            path,
            expected_name=project["name"],
            expected_version=project["version"],
        )
        for path in artifact_paths
    ]
    inspected.sort(key=lambda item: item["filename"])
    _consistent_metadata(inspected)
    provenance = _source_provenance(root)
    provenance["reviewed_assets_sha256"] = _bind_reviewed_source_assets(inspected, root)
    kinds = [artifact["kind"] for artifact in inspected]
    pair_complete = (
        kinds.count("wheel") == 1 and kinds.count("sdist") == 1 and len(kinds) == 2
    )
    if require_clean:
        _require(not provenance["dirty"], "release source tree is dirty")
        _require(
            pair_complete,
            "release candidate requires exactly one wheel and one sdist",
        )
    eligible = not provenance["dirty"] and pair_complete
    return {
        "artifacts": inspected,
        "build_environment": {
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project": project,
        "release_candidate_eligible": eligible,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": provenance,
        "status": "release_candidate_ready" if eligible else "development_only",
        "tool": {
            "path": "tools/release_artifact_manifest.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }


def verify_manifest(
    manifest_path: str | Path, *, artifact_directory: str | Path | None = None
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseManifestError(f"cannot read release manifest: {error}") from error
    _require(isinstance(manifest, dict), "release manifest root is not an object")
    _require(manifest.get("schema") == SCHEMA, "unexpected release manifest schema")
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported release manifest schema version",
    )
    project = manifest.get("project")
    _require(isinstance(project, dict), "release manifest project block is missing")
    name = project.get("name")
    version = project.get("version")
    _require(isinstance(name, str) and bool(name), "manifest project name is missing")
    _require(isinstance(version, str) and bool(version), "manifest version is missing")
    recorded = manifest.get("artifacts")
    _require(isinstance(recorded, list) and bool(recorded), "manifest has no artifacts")
    directory = path.parent if artifact_directory is None else Path(artifact_directory).resolve()
    checked: list[dict[str, Any]] = []
    filenames: set[str] = set()
    for entry in recorded:
        _require(isinstance(entry, dict), "invalid artifact manifest entry")
        filename = entry.get("filename")
        _require(isinstance(filename, str), "artifact filename is missing")
        _require(Path(filename).name == filename, f"unsafe artifact filename: {filename!r}")
        _require(filename not in filenames, f"duplicate artifact filename: {filename}")
        filenames.add(filename)
        actual = inspect_artifact(
            directory / filename, expected_name=name, expected_version=version
        )
        _require(actual == entry, f"artifact no longer matches manifest: {filename}")
        checked.append(actual)
    _consistent_metadata(checked)
    return {
        "artifacts_verified": len(checked),
        "manifest_sha256": _sha256(path),
        "status": "passed",
    }


def _write_json(path: Path, value: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise ReleaseManifestError(f"output already exists (use --force): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", help="wheel/sdist archives to inventory")
    parser.add_argument("--output", type=Path, help="new JSON manifest path")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    parser.add_argument("--verify", type=Path, help="verify an existing JSON manifest")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="archive directory for --verify (defaults to manifest directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if args.artifacts or args.output is not None or args.require_clean or args.force:
                parser.error(
                    "--verify cannot be combined with artifacts, --output, "
                    "--require-clean, or --force"
                )
            result = verify_manifest(args.verify, artifact_directory=args.artifact_dir)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.artifact_dir is not None:
            parser.error("--artifact-dir requires --verify")
        if not args.artifacts or args.output is None:
            parser.error("generation requires artifacts and --output")
        manifest = generate_manifest(
            args.artifacts,
            repo_root=args.repo_root,
            require_clean=args.require_clean,
        )
        _write_json(args.output.resolve(), manifest, force=args.force)
        print(
            json.dumps(
                {
                    "artifacts": len(manifest["artifacts"]),
                    "output": os.fspath(args.output.resolve()),
                    "status": manifest["status"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except ReleaseManifestError as error:
        print(f"release manifest failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
