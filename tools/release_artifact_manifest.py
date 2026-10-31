#!/usr/bin/env python3
# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate and verify a fail-closed NumPilot release artifact manifest.

This tool intentionally depends only on the Python standard library.  It does
not extract or execute archive members.  The generated JSON binds wheel/sdist
bytes, their complete regular-file inventories, core package metadata, and the
Git source identity used for the build.  When requested, it also binds the
frozen AsNumpy component manifest, license, SBOM, and exact wheel bytes carried
by both archives.

A dirty source tree is recorded as ``development_only``.  Pass
``--require-clean`` for an RC/release gate.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
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
    "TEMP-RECORDING-COMMANDS.zh-CN.md",
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
    "docs/assets/architecture/numpilot-architecture-en.png",
    "docs/assets/architecture/numpilot-architecture-zh.png",
    "docs/assets/architecture/numpilot-two-routes.zh-CN.png",
    "docs/assets/brand/asnumpy-mark.png",
    "docs/assets/brand/ascend-wordmark.png",
    "docs/assets/brand/cann-wordmark.png",
    "docs/assets/brand/codearts-icon.png",
    "docs/assets/brand/numpilot-wordmark.png",
)


class ReleaseManifestError(RuntimeError):
    """Raised when artifacts cannot support a release manifest."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
        if suffix.endswith(".svg"):
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
        f"{package}/cli_ui.py",
        f"{package}/add_one_gap.py",
        f"{package}/demo_llm_operator_generation.py",
        f"{package}/interactive_migration.py",
        f"{package}/interactive_report.py",
        f"{package}/llm_check.py",
        f"{package}/migration_skill.py",
        f"{package}/repl.py",
        f"{package}/runtime_extensions.py",
        f"{package}/_script_benchmark_worker.py",
        f"{package}/_script_verification_worker.py",
        f"{package}/script_benchmark.py",
        f"{package}/script_migration.py",
        f"{package}/script_verification.py",
        f"{package}/workflow.py",
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
        "share/numpilot/skills/numpilot-migration-gate/agents/openai.yaml",
        "share/numpilot/skills/numpilot-migration-gate/references/codearts-sidebar.md",
        "share/numpilot/skills/numpilot-migration-gate/references/terminal-contract.md",
        "share/numpilot/skill/README.md",
        "share/numpilot/skill/SKILL.md",
        "share/numpilot/skill/agents/openai.yaml",
        "share/numpilot/skill/references/codearts-sidebar.md",
        "share/numpilot/skill/references/terminal-contract.md",
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
            "docs/assets/architecture/numpilot-architecture-en.png",
            "docs/assets/architecture/numpilot-architecture-zh.png",
            "docs/assets/brand/asnumpy-mark.png",
            "docs/assets/brand/ascend-wordmark.png",
            "docs/assets/brand/cann-wordmark.png",
            "docs/assets/brand/codearts-icon.png",
            "docs/assets/brand/numpilot-wordmark.png",
            "docs/assets/brand/README.md",
            "examples/custom_operator_add_one.py",
            "examples/add_one_migration_demo.py",
            "examples/black_scholes_portfolio.py",
            ".codeartsdoer/agents/numpilot-analyzer.md",
            ".codeartsdoer/agents/numpilot-evidence-auditor.md",
            ".codeartsdoer/agents/numpilot-operator-engineer.md",
            ".codeartsdoer/agents/numpilot-orchestrator.md",
            ".codeartsdoer/skills/ProjectSkillStatus.txt",
            ".codeartsdoer/skills/numpilot-migration-gate/SKILL.md",
            ".codeartsdoer/skills/numpilot-migration-gate/agents/openai.yaml",
            ".codeartsdoer/skills/numpilot-migration-gate/references/codearts-sidebar.md",
            ".codeartsdoer/skills/numpilot-migration-gate/references/terminal-contract.md",
            "skill/README.md",
            "skill/SKILL.md",
            "skill/agents/openai.yaml",
            "skill/references/codearts-sidebar.md",
            "skill/references/terminal-contract.md",
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


_ASNUMPY_COMPONENT_FILES = (
    "manifest.json",
    "README.md",
    "LICENSE",
    "SBOM.spdx.json",
)


def _component_record_matches(
    artifact: Mapping[str, Any], name: str
) -> list[dict[str, Any]]:
    """Return archive records for one release-adjacent component filename."""

    records = artifact.get("regular_files")
    if not isinstance(records, list):
        return []
    suffixes = (
        "/share/numpilot/components/asnumpy/" + name,
        "/components/asnumpy/" + name,
    )
    return [
        item
        for item in records
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and any(item["path"].endswith(suffix) for suffix in suffixes)
    ]


def _read_component_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Read the public component lock and return it with source identities."""

    manifest_path = root / "components" / "asnumpy" / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    try:
        payload = manifest_path.read_bytes()
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _strict_object(pairs),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-standard JSON constant: %s" % value)
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ReleaseManifestError(f"invalid AsNumpy component manifest: {error}") from error
    _require(isinstance(raw, dict), "AsNumpy component manifest root is not an object")
    _require(raw.get("schema") == "numpilot.asnumpy-component", "unexpected AsNumpy component schema")
    _require(raw.get("schema_version") == 1, "unsupported AsNumpy component schema version")
    wheel = raw.get("wheel")
    _require(isinstance(wheel, dict), "AsNumpy component manifest wheel is missing")
    wheel_name = wheel.get("filename")
    wheel_digest = wheel.get("sha256")
    _require(isinstance(wheel_name, str) and Path(wheel_name).name == wheel_name, "unsafe AsNumpy component wheel filename")
    _require(isinstance(wheel_digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", wheel_digest) is not None, "invalid AsNumpy component wheel SHA-256")
    component_id = raw.get("component_id")
    version = raw.get("version")
    _require(isinstance(component_id, str) and component_id, "AsNumpy component id is missing")
    _require(isinstance(version, str) and version, "AsNumpy component version is missing")
    source_root = manifest_path.parent
    source_files: dict[str, dict[str, Any]] = {}
    for name in (*_ASNUMPY_COMPONENT_FILES, wheel_name):
        target = source_root / name
        _require(target.is_file() and not target.is_symlink(), f"AsNumpy component attachment is missing: {name}")
        digest = _sha256(target)
        source_files[name] = {
            "filename": name,
            "sha256": digest,
            "size_bytes": target.stat().st_size,
        }
    for field, filename_key in (("license_file", "LICENSE"), ("sbom", "SBOM.spdx.json")):
        block = raw.get(field)
        if isinstance(block, dict):
            declared_name = block.get("filename")
            declared_hash = block.get("sha256")
            _require(declared_name == filename_key, f"AsNumpy component {field} filename is not bound")
            _require(declared_hash == source_files[filename_key]["sha256"], f"AsNumpy component {field} hash does not match source")
    _require(wheel_digest.lower() == source_files[wheel_name]["sha256"], "AsNumpy component wheel hash does not match source")
    return raw, {
        "manifest_sha256": _sha256(manifest_path),
        "component_id": component_id,
        "version": version,
        "wheel_filename": wheel_name,
        "wheel_sha256": wheel_digest.lower(),
        "source_files": source_files,
    }


def _bind_asnumpy_component(
    artifacts: list[dict[str, Any]],
    repo_root: Path,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Bind the frozen AsNumpy bundle into the top-level release manifest.

    This is intentionally optional for synthetic/development projects that do
    not ship AsNumpy.  A real NumPilot RC passes ``required=True`` so both the
    sdist and wheel carry the same manifest, legal attachments and exact wheel
    bytes.
    """

    loaded = _read_component_lock(repo_root)
    if loaded is None:
        _require(not required, "release candidate requires components/asnumpy/manifest.json")
        return {}
    raw, identity = loaded
    names = (*_ASNUMPY_COMPONENT_FILES, identity["wheel_filename"])
    by_artifact: dict[str, Any] = {}
    for artifact in artifacts:
        kind = str(artifact.get("kind"))
        bindings: dict[str, Any] = {}
        for name in names:
            matches = _component_record_matches(artifact, name)
            _require(len(matches) == 1, f"{kind} does not carry exactly one AsNumpy component file: {name}")
            match = matches[0]
            expected = identity["source_files"][name]
            _require(match.get("sha256") == expected["sha256"], f"{kind} AsNumpy component differs from source: {name}")
            _require(match.get("size_bytes") == expected["size_bytes"], f"{kind} AsNumpy component size differs from source: {name}")
            bindings[name] = {
                "path": match["path"],
                "sha256": match["sha256"],
                "size_bytes": match["size_bytes"],
            }
        by_artifact[kind] = bindings
    return {
        "schema": raw["schema"],
        "schema_version": raw["schema_version"],
        "component_id": identity["component_id"],
        "version": identity["version"],
        "manifest_sha256": identity["manifest_sha256"],
        "wheel_filename": identity["wheel_filename"],
        "wheel_sha256": identity["wheel_sha256"],
        "artifacts": by_artifact,
    }


def _verify_asnumpy_component_binding(
    binding: Mapping[str, Any], artifacts: list[dict[str, Any]]
) -> None:
    """Verify a previously generated component binding against archives."""

    _require(binding.get("schema") == "numpilot.asnumpy-component", "invalid AsNumpy component binding schema")
    _require(binding.get("schema_version") == 1, "invalid AsNumpy component binding version")
    wheel_name = binding.get("wheel_filename")
    wheel_digest = binding.get("wheel_sha256")
    _require(isinstance(wheel_name, str) and isinstance(wheel_digest, str), "incomplete AsNumpy component binding")
    artifact_bindings = binding.get("artifacts")
    _require(isinstance(artifact_bindings, dict), "AsNumpy component artifact bindings are missing")
    expected_names = {"manifest.json", "README.md", "LICENSE", "SBOM.spdx.json", wheel_name}
    expected_kinds = {str(artifact.get("kind")) for artifact in artifacts}
    _require(set(artifact_bindings) == expected_kinds, "AsNumpy component artifact bindings do not match release artifacts")
    for artifact in artifacts:
        kind = str(artifact.get("kind"))
        declared = artifact_bindings.get(kind)
        _require(isinstance(declared, dict), f"AsNumpy component binding has no {kind} entry")
        _require(set(declared) == expected_names, f"{kind} AsNumpy component binding is incomplete")
        for name, record in declared.items():
            _require(isinstance(record, dict), f"invalid AsNumpy component binding record: {name}")
            record_path = record.get("path")
            record_hash = record.get("sha256")
            record_size = record.get("size_bytes")
            record_posix = PurePosixPath(record_path) if isinstance(record_path, str) else None
            _require(
                isinstance(record_path, str)
                and bool(record_path)
                and record_posix is not None
                and not record_posix.is_absolute()
                and record_posix.as_posix() == record_path
                and all(part not in {"", ".", ".."} for part in record_posix.parts)
                and "\\" not in record_path,
                f"invalid {kind} AsNumpy component member path: {name}",
            )
            _require(
                isinstance(record_hash, str)
                and re.fullmatch(r"[0-9a-f]{64}", record_hash) is not None
                and isinstance(record_size, int)
                and record_size >= 0,
                f"invalid {kind} AsNumpy component member identity: {name}",
            )
            matches = [
                item
                for item in artifact.get("regular_files", [])
                if isinstance(item, Mapping) and item.get("path") == record_path
            ]
            _require(len(matches) == 1, f"{kind} AsNumpy component member is missing: {name}")
            _require(matches[0].get("sha256") == record_hash, f"{kind} AsNumpy component hash changed: {name}")
            _require(matches[0].get("size_bytes") == record_size, f"{kind} AsNumpy component size changed: {name}")
        wheel_records = [record for name, record in declared.items() if name == wheel_name]
        _require(len(wheel_records) == 1 and wheel_records[0].get("sha256") == wheel_digest, f"{kind} AsNumpy wheel binding changed")


def generate_manifest(
    artifacts: Iterable[str | Path],
    *,
    repo_root: str | Path,
    require_clean: bool = False,
    require_asnumpy_component: bool = False,
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
    component_binding = _bind_asnumpy_component(
        inspected, root, required=require_asnumpy_component
    )
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
        "components": (
            {"asnumpy": component_binding} if component_binding else {}
        ),
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
    manifest_path: str | Path,
    *,
    artifact_directory: str | Path | None = None,
    require_asnumpy_component: bool = False,
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
    components = manifest.get("components", {})
    _require(isinstance(components, dict), "release manifest components block is invalid")
    component_binding = components.get("asnumpy")
    if component_binding is None:
        _require(not require_asnumpy_component, "release manifest has no AsNumpy component binding")
    else:
        _require(isinstance(component_binding, dict), "release manifest AsNumpy component binding is invalid")
        _verify_asnumpy_component_binding(component_binding, checked)
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
    parser.add_argument(
        "--require-asnumpy-component",
        action="store_true",
        help="require the frozen components/asnumpy bundle and bind it into the manifest",
    )
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
            if (
                args.artifacts
                or args.output is not None
                or args.require_clean
                or args.force
            ):
                parser.error(
                    "--verify cannot be combined with artifacts, --output, "
                    "--require-clean, or --force"
                )
            result = verify_manifest(
                args.verify,
                artifact_directory=args.artifact_dir,
                require_asnumpy_component=args.require_asnumpy_component,
            )
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
            require_asnumpy_component=args.require_asnumpy_component,
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
