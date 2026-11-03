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

"""Fail-closed audit for the NumPilot public release tree.

The audit is read-only and uses only the Python standard library.  It checks
the Git candidate set rather than the raw working directory, so ignored local
venvs and build caches cannot be mistaken for release contents.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import sys
from typing import Iterable
import xml.etree.ElementTree as ET
from email.parser import Parser
from zipfile import BadZipFile, ZipFile


REQUIRED_FILES = {
    ".codeartsdoer/skills/ProjectSkillStatus.txt",
    "CHANGELOG.md",
    "components/asnumpy/manifest.json",
    "components/asnumpy/source-manifest.json",
    "components/asnumpy/asnumpy-source-1e7177f12039b2d87e3cdff6693459d895744f68.tar.gz",
    "components/cannbot-skills/39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c/manifest.json",
    "components/cannbot-skills/39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c/bundle.json",
    "docs/asnumpy-api-support.zh-CN.md",
    "docs/asnumpy-source-extension.zh-CN.md",
    "docs/asnumpy-contract-audit.zh-CN.md",
    "components/asnumpy/README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "llm.example.json",
    "README.md",
    "README.zh-CN.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/README.md",
    "docs/SUBMISSION.md",
    "docs/numpilot-asnumpy-custom-operator-flow.drawio",
    "docs/numpilot-asnumpy-custom-operator-flow.png",
    "docs/architecture.md",
    "docs/assets/architecture/numpilot-architecture-en.png",
    "docs/assets/architecture/numpilot-architecture-zh.png",
    "docs/assets/architecture/numpilot-two-routes.zh-CN.png",
    "docs/assets/brand/asnumpy-mark.png",
    "docs/assets/brand/ascend-wordmark.png",
    "docs/assets/brand/cann-wordmark.png",
    "docs/assets/brand/codearts-icon.png",
    "docs/assets/brand/numpilot-wordmark.png",
    "docs/assets/brand/README.md",
    "docs/installation.md",
    "docs/installation.zh-CN.md",
    ".codeartsdoer/skills/numpilot-migration-gate/SKILL.md",
    ".codeartsdoer/skills/numpilot-migration-gate/agents/openai.yaml",
    ".codeartsdoer/skills/numpilot-migration-gate/references/codearts-sidebar.md",
    ".codeartsdoer/skills/numpilot-migration-gate/references/terminal-contract.md",
    "skill/README.md",
    "skill/SKILL.md",
    "skill/agents/openai.yaml",
    "skill/references/codearts-sidebar.md",
    "skill/references/terminal-contract.md",
    "examples/add_one_migration_demo.py",
    "examples/custom_operator_add_one.py",
    "examples/poisson_source_smooth_l1_map.py",
    "examples/black_scholes_portfolio.py",
    "install.sh",
    "numpilot/__init__.py",
    "numpilot/add_one_gap.py",
    "numpilot/cli_ui.py",
    "numpilot/demo_llm_operator_generation.py",
    "numpilot/interactive_migration.py",
    "numpilot/interactive_report.py",
    "numpilot/llm_check.py",
    "numpilot/migration_skill.py",
    "numpilot/repl.py",
    "numpilot/runtime_extensions.py",
    "numpilot/workflow.py",
    "pyproject.toml",
    "run_demo.sh",
    "tools/audit_script_benchmark_receipt.py",
    "tools/check_release_tree.py",
    "tools/extract_asnumpy_source.py",
    "tools/asnumpy_contract_probe.py",
    "tools/release_artifact_manifest.py",
    "tools/render_demo_summary.py",
    "tools/render_migration_report.py",
}

FORBIDDEN_EXACT = {
    "NumPilot20260615-v1.pdf",
    "TEMP-RECORDING-COMMANDS.zh-CN.md",
    "asnumpy_cpp.log",
    "llm.json",
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

TEXT_SUFFIXES = {".ipynb", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
# Match both a path below the known workspace homes and the home directory
# itself.  Keep ``\b`` as a regex word boundary (rather than the literal
# two-character sequence ``\\b``) so a redacted path ending at the username is
# still rejected.
PRIVATE_PATH_RE = re.compile(r"/home/(?:ma-user|lixing)(?:/|\b)")
MAX_FILE_BYTES = 2 * 1024 * 1024

# Keep the component audit independent from the package importer.  This tool
# is also run from a source checkout before a wheel exists, so importing
# ``numpilot.bootstrap`` here would make the release gate depend on optional
# runtime packages (and could execute environment discovery as a side effect).
ASNUMPY_COMPONENT_MANIFEST = "components/asnumpy/manifest.json"
ASNUMPY_COMPONENT_README = "components/asnumpy/README.md"
ASNUMPY_COMPONENT_PREFIX = "components/asnumpy/"
ASNUMPY_COMPONENT_SCHEMA = "numpilot.asnumpy-component"
ASNUMPY_COMPONENT_SCHEMA_VERSION = 1
ASNUMPY_WHEEL_MAX_FILES = 4096
ASNUMPY_WHEEL_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
ASNUMPY_SOURCE_MAX_BYTES = 256 * 1024 * 1024
ASNUMPY_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASNUMPY_WHEEL_RE = re.compile(
    r"^asnumpy-[A-Za-z0-9][A-Za-z0-9.+!-]*-[A-Za-z0-9_]+-[A-Za-z0-9_]+-[A-Za-z0-9_]+\.whl$"
)


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
        return "historical evidence must stay outside the release branch"
    if any(fragment in path.name for fragment in FORBIDDEN_BASENAME_FRAGMENTS):
        return "internal handoff/WIP document"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"intermediate artifact suffix {path.suffix.lower()}"
    if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
        return "cache directory"
    if any(part in {"build", "dist"} or part.endswith(".egg-info") for part in path.parts):
        return "build output directory"
    return None


def _strict_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys in a component lock.

    A release lock is an integrity boundary.  The default ``json.loads``
    behaviour (last duplicate key wins) would allow two parsers to observe
    different wheel identities, so duplicate keys are treated as malformed.
    """

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _component_safe_relative(value: object, label: str) -> str:
    """Validate a component-declared relative file path."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or "\\" in value
        or ".." in candidate.parts
        or "." in candidate.parts
        or candidate.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} contains control characters")
    return value


def _component_declared_file(
    raw: dict[str, object],
    *,
    keys: tuple[str, ...],
    label: str,
) -> tuple[str, str | None] | None:
    """Read an optional ``path``/``sha256`` declaration from a lock.

    For compatibility with hand-authored locks, both ``{"path": ...,
    "sha256": ...}`` and a bare filename are accepted.  A separate
    ``<label>_sha256`` field is also understood.
    """

    present = [key for key in keys if key in raw]
    digest_key = f"{label}_sha256"
    if digest_key in raw:
        present.append(digest_key)
    if not present:
        return None

    value = next((raw[key] for key in keys if key in raw), None)
    digest: object = raw.get(digest_key)
    if isinstance(value, dict):
        path_value = value.get("path", value.get("filename"))
        if digest is None:
            digest = value.get("sha256")
    else:
        path_value = value
    if path_value is None:
        raise ValueError(f"{label} declaration has no path")
    path = _component_safe_relative(path_value, f"{label}.path")
    if digest is not None:
        if not isinstance(digest, str) or ASNUMPY_SHA256_RE.fullmatch(digest.lower()) is None:
            raise ValueError(f"{label}.sha256 is not a 64-character lowercase SHA-256")
        digest = digest.lower()
    return path, digest  # type: ignore[return-value]


def _component_scan_private(value: object, where: str, errors: list[str]) -> None:
    """Recursively reject private workspace paths in public component data."""

    if isinstance(value, str):
        if PRIVATE_PATH_RE.search(value):
            errors.append(f"private absolute workspace path in AsNumpy component: {where}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _component_scan_private(item, f"{where}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _component_scan_private(item, f"{where}[{index}]", errors)


def _component_file_hash(
    root: Path,
    relative: str,
    *,
    label: str,
    expected: str | None,
    errors: list[str],
    available_files: set[str] | None = None,
) -> Path | None:
    """Check an optional component attachment and return its real path."""

    candidate_relative = f"{ASNUMPY_COMPONENT_PREFIX}{relative}"
    if available_files is not None and candidate_relative not in available_files:
        errors.append(
            f"AsNumpy component {label} is not part of the Git release candidate: {candidate_relative}"
        )
        return None
    target = root / relative
    try:
        base = root.resolve()
        resolved = target.resolve(strict=False)
        if base != resolved and base not in resolved.parents:
            errors.append(
                f"AsNumpy component {label} escapes its component directory: {relative}"
            )
            return None
        if target.is_symlink():
            errors.append(f"AsNumpy component {label} must not be a symlink: {relative}")
            return None
        if not target.is_file():
            errors.append(f"AsNumpy component {label} is missing: {relative}")
            return None
        if target.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"AsNumpy component {label} exceeds {MAX_FILE_BYTES} bytes: {relative}")
            return None
        payload = target.read_bytes()
    except OSError as exc:
        errors.append(f"cannot read AsNumpy component {label} {relative}: {exc}")
        return None
    if expected is not None:
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            errors.append(
                f"AsNumpy component {label} SHA-256 mismatch: {relative} "
                f"(expected {expected}, got {observed})"
            )
    try:
        text = payload.decode("utf-8", "strict")
    except UnicodeDecodeError:
        text = ""
    if label == "license" and not payload.strip():
        errors.append(f"AsNumpy component license is empty: {relative}")
    if label == "sbom":
        try:
            sbom = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"AsNumpy component SBOM is not valid JSON: {relative}: {exc}")
        else:
            if not isinstance(sbom, dict) or not (
                str(sbom.get("spdxVersion", "")).startswith("SPDX-")
                or str(sbom.get("bomFormat", "")).lower() == "cyclonedx"
            ):
                errors.append(f"AsNumpy component SBOM is not a recognized SPDX/CycloneDX document: {relative}")
    if PRIVATE_PATH_RE.search(text):
        errors.append(f"private absolute workspace path in AsNumpy component {relative}")
    return target


def _component_wheel_member_name(raw_name: str) -> str:
    """Normalize and validate one wheel member name."""

    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ValueError(f"unsafe wheel member path: {raw_name!r}")
    name = raw_name[:-1] if raw_name.endswith("/") else raw_name
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "" in path.parts
        or ":" in path.parts[0]
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe wheel member path: {raw_name!r}")
    return name


def _component_wheel_identity(
    wheel: Path,
    raw_manifest: dict[str, object],
    errors: list[str],
) -> dict[str, object] | None:
    """Verify wheel bytes and return the identity used by the component lock."""

    wheel_block = raw_manifest.get("wheel")
    if not isinstance(wheel_block, dict):
        errors.append("AsNumpy component manifest wheel must be an object")
        return None
    expected_digest = wheel_block.get("sha256")
    if not isinstance(expected_digest, str) or ASNUMPY_SHA256_RE.fullmatch(expected_digest.lower()) is None:
        errors.append("AsNumpy component manifest wheel.sha256 is invalid")
        return None
    expected_digest = expected_digest.lower()
    try:
        observed_digest = _sha256(wheel)
    except OSError as exc:
        errors.append(f"cannot hash AsNumpy component wheel: {exc}")
        return None
    if observed_digest != expected_digest:
        errors.append(
            "AsNumpy component wheel SHA-256 mismatch: "
            f"expected {expected_digest}, got {observed_digest}"
        )
        return None

    try:
        with ZipFile(wheel) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > ASNUMPY_WHEEL_MAX_FILES:
                raise ValueError("invalid file count")
            if sum(info.file_size for info in infos) > ASNUMPY_WHEEL_MAX_UNCOMPRESSED_BYTES:
                raise ValueError("uncompressed size limit exceeded")
            names: list[str] = []
            payloads: dict[str, bytes] = {}
            for info in infos:
                name = _component_wheel_member_name(info.filename)
                if name in names:
                    raise ValueError(f"duplicate member path: {name}")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"symbolic link member: {name}")
                names.append(name)
                with archive.open(info, "r") as stream:
                    payloads[name] = stream.read()
    except (BadZipFile, EOFError, KeyError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"invalid AsNumpy component wheel: {exc}")
        return None

    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    if not (len(metadata_names) == len(wheel_names) == len(record_names) == 1):
        errors.append("AsNumpy component wheel must contain one METADATA, WHEEL and RECORD")
        return None
    metadata_name, wheel_name, record_name = metadata_names[0], wheel_names[0], record_names[0]
    dist_info_roots = {
        name.rsplit("/", 1)[0]
        for name in (metadata_name, wheel_name, record_name)
    }
    if len(dist_info_roots) != 1:
        errors.append("AsNumpy component wheel metadata files use different dist-info roots")
    metadata_bytes = payloads[metadata_name]
    wheel_bytes = payloads[wheel_name]
    record_bytes = payloads[record_name]
    metadata_expected = wheel_block.get("metadata_sha256")
    record_expected = wheel_block.get("record_sha256")
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    record_digest = hashlib.sha256(record_bytes).hexdigest()
    for label, expected, observed in (
        ("METADATA", metadata_expected, metadata_digest),
        ("RECORD", record_expected, record_digest),
    ):
        if expected is None:
            errors.append(f"AsNumpy component manifest does not bind wheel {label} SHA-256")
        elif not isinstance(expected, str) or ASNUMPY_SHA256_RE.fullmatch(expected.lower()) is None:
            errors.append(f"AsNumpy component manifest wheel {label.lower()}_sha256 is invalid")
        elif expected.lower() != observed:
            errors.append(
                f"AsNumpy component wheel {label} SHA-256 mismatch: "
                f"expected {expected.lower()}, got {observed}"
            )

    try:
        metadata = Parser().parsestr(metadata_bytes.decode("utf-8"))
        wheel_metadata = Parser().parsestr(wheel_bytes.decode("utf-8"))
        record_text = record_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"AsNumpy component wheel metadata is not UTF-8: {exc}")
        return None
    distribution = str(metadata.get("Name", ""))
    version = str(metadata.get("Version", "")).strip()
    canonical_distribution = re.sub(r"[-_.]+", "-", distribution).lower()
    if canonical_distribution != "asnumpy":
        errors.append(f"AsNumpy component wheel distribution is not AsNumpy: {distribution!r}")
    expected_version = raw_manifest.get("version")
    if isinstance(expected_version, str) and version != expected_version:
        errors.append(
            f"AsNumpy component wheel version mismatch: expected {expected_version}, got {version}"
        )
    tags = [str(value) for value in wheel_metadata.get_all("Tag", [])]
    expected_tags = wheel_block.get("tags")
    if not isinstance(expected_tags, list) or not all(isinstance(value, str) for value in expected_tags):
        errors.append("AsNumpy component manifest wheel.tags is invalid")
    elif len(tags) != len(set(tags)) or len(expected_tags) != len(set(expected_tags)):
        errors.append("AsNumpy component wheel tags must not contain duplicates")
    elif set(tags) != set(expected_tags):
        errors.append(
            "AsNumpy component wheel tags mismatch: "
            f"expected {sorted(expected_tags)}, got {sorted(tags)}"
        )
    runtime = raw_manifest.get("runtime")
    if isinstance(runtime, dict):
        python_mm = runtime.get("python_major_minor")
        machine = runtime.get("machine")
        implementation = runtime.get("python_implementation")
        if isinstance(python_mm, str) and re.fullmatch(r"\d+\.\d+", python_mm):
            python_token = "cp" + python_mm.replace(".", "")
            compatible = [tag for tag in tags if tag.startswith(python_token + "-")]
            if implementation == "CPython" and not compatible:
                errors.append(
                    "AsNumpy component wheel tags do not match runtime Python "
                    f"{python_mm} ({python_token})"
                )
            if isinstance(machine, str) and machine:
                platform_token = machine.lower().replace("-", "_")
                if compatible and not any(platform_token in tag.lower() for tag in compatible):
                    errors.append(
                        "AsNumpy component wheel tags do not match runtime machine "
                        f"{machine}"
                    )

    try:
        rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except csv.Error as exc:
        errors.append(f"AsNumpy component wheel RECORD is invalid CSV: {exc}")
        rows = []
    record_map: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            errors.append("AsNumpy component wheel RECORD rows must have three fields")
            continue
        name, digest_field, size_field = row
        try:
            _component_wheel_member_name(name)
        except ValueError as exc:
            errors.append(f"AsNumpy component wheel RECORD has unsafe path: {exc}")
        if name in record_map:
            errors.append(f"AsNumpy component wheel RECORD has duplicate path: {name}")
            continue
        record_map[name] = (digest_field, size_field)
    if set(record_map) != set(names):
        errors.append("AsNumpy component wheel RECORD does not cover the archive exactly")
    member_hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        member_hashes[name] = hashlib.sha256(payload).hexdigest()
        digest_field, size_field = record_map.get(name, (None, None))
        if name == record_name:
            if digest_field not in ("", None) or size_field not in ("", None):
                errors.append("AsNumpy component wheel RECORD self-entry must be unhashed")
            continue
        if not isinstance(digest_field, str) or not digest_field.startswith("sha256="):
            errors.append(f"AsNumpy component wheel RECORD entry is not SHA-256 bound: {name}")
            continue
        if not isinstance(size_field, str) or not size_field.isdigit():
            errors.append(f"AsNumpy component wheel RECORD size is invalid: {name}")
            continue
        encoded = digest_field.split("=", 1)[1]
        try:
            expected_member = base64.b64decode(
                encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
            )
        except (binascii.Error, ValueError):
            errors.append(f"AsNumpy component wheel RECORD hash is invalid: {name}")
            continue
        if expected_member.hex() != member_hashes[name] or int(size_field) != len(payload):
            errors.append(f"AsNumpy component wheel RECORD hash or size mismatch: {name}")
    for name, payload in payloads.items():
        suffix = PurePosixPath(name).suffix.lower()
        if name.endswith(("/METADATA", "/WHEEL", "/RECORD")) or suffix in TEXT_SUFFIXES:
            if PRIVATE_PATH_RE.search(payload.decode("utf-8", "replace")):
                errors.append(f"private absolute workspace path in AsNumpy wheel: {name}")

    module_name = "asnumpy/__init__.py"
    native_names = sorted(
        name
        for name in names
        if PurePosixPath(name).parent == PurePosixPath("asnumpy") and name.endswith(".so")
    )
    if module_name not in payloads or not native_names:
        errors.append("AsNumpy component wheel must contain asnumpy/__init__.py and a native .so")
    observed_installed: dict[str, object] = {
        "module": {
            "path": module_name,
            "sha256": member_hashes.get(module_name, ""),
        },
        "native_modules": [
            {"path": name, "sha256": member_hashes[name]} for name in native_names
        ],
    }
    expected_installed = wheel_block.get("installed_files")
    if expected_installed is None:
        errors.append("AsNumpy component manifest does not bind installed file identity")
    elif not isinstance(expected_installed, dict):
        errors.append("AsNumpy component manifest wheel.installed_files is invalid")
    elif expected_installed != observed_installed:
        errors.append(
            "AsNumpy component installed-file identity mismatch: "
            f"expected {expected_installed}, got {observed_installed}"
        )

    return {
        "filename": wheel.name,
        "sha256": observed_digest,
        "metadata_sha256": metadata_digest,
        "record_sha256": record_digest,
        "version": version,
        "tags": tags,
        "installed_files": observed_installed,
    }


def _component_source_bundle_identity(
    root: Path,
    raw_manifest: dict[str, object],
    file_set: set[str],
    errors: list[str],
) -> dict[str, object] | None:
    """Verify the optional complete AsNumpy source snapshot."""

    block = raw_manifest.get("source_bundle")
    if block is None:
        if raw_manifest.get("artifact_required") is True:
            errors.append("AsNumpy frozen component must bind source_bundle")
        return None
    if not isinstance(block, dict):
        errors.append("AsNumpy component source_bundle must be an object")
        return None
    filename = block.get("filename")
    manifest_filename = block.get("manifest_filename")
    try:
        filename = _component_safe_relative(filename, "source_bundle.filename")
        manifest_filename = _component_safe_relative(
            manifest_filename, "source_bundle.manifest_filename"
        )
    except ValueError as exc:
        errors.append(f"invalid AsNumpy source_bundle path: {exc}")
        return None
    if filename == manifest_filename or not filename.endswith(".tar.gz"):
        errors.append("AsNumpy source_bundle filenames are invalid")
        return None
    if PurePosixPath(filename).name != filename or PurePosixPath(manifest_filename).name != manifest_filename:
        errors.append("AsNumpy source_bundle filenames must be basenames")
        return None
    archive_relative = f"{ASNUMPY_COMPONENT_PREFIX}{filename}"
    source_manifest_relative = f"{ASNUMPY_COMPONENT_PREFIX}{manifest_filename}"
    for relative in (archive_relative, source_manifest_relative):
        if relative not in file_set:
            errors.append(f"AsNumpy source_bundle attachment is not in release candidate: {relative}")
    archive_path = root / ASNUMPY_COMPONENT_PREFIX / filename
    source_manifest_path = root / ASNUMPY_COMPONENT_PREFIX / manifest_filename
    if not archive_path.is_file() or archive_path.is_symlink():
        errors.append(f"AsNumpy source_bundle archive is missing: {filename}")
        return None
    if not source_manifest_path.is_file() or source_manifest_path.is_symlink():
        errors.append(f"AsNumpy source_bundle manifest is missing: {manifest_filename}")
        return None
    try:
        archive_bytes = archive_path.read_bytes()
        source_manifest_bytes = source_manifest_path.read_bytes()
    except OSError as exc:
        errors.append(f"cannot read AsNumpy source_bundle: {exc}")
        return None
    if len(archive_bytes) > ASNUMPY_SOURCE_MAX_BYTES:
        errors.append("AsNumpy source_bundle archive exceeds size limit")
        return None

    def digest(value: object, label: str) -> str | None:
        if not isinstance(value, str) or ASNUMPY_SHA256_RE.fullmatch(value.lower()) is None:
            errors.append(f"AsNumpy source_bundle {label} SHA-256 is invalid")
            return None
        return value.lower()

    expected_archive_sha = digest(block.get("sha256"), "archive")
    expected_manifest_sha = digest(block.get("manifest_sha256"), "manifest")
    if expected_archive_sha is not None and hashlib.sha256(archive_bytes).hexdigest() != expected_archive_sha:
        errors.append("AsNumpy source_bundle archive SHA-256 mismatch")
    if expected_manifest_sha is not None and hashlib.sha256(source_manifest_bytes).hexdigest() != expected_manifest_sha:
        errors.append("AsNumpy source_bundle manifest SHA-256 mismatch")
    try:
        source_manifest = json.loads(
            source_manifest_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid AsNumpy source-manifest: {exc}")
        return None
    if not isinstance(source_manifest, dict):
        errors.append("AsNumpy source-manifest root must be an object")
        return None
    for key, expected in (
        ("schema", "numpilot.asnumpy-source"),
        ("schema_version", 1),
        ("repository", raw_manifest.get("repository")),
        ("commit", raw_manifest.get("commit")),
        ("tree", raw_manifest.get("tree")),
        ("license", raw_manifest.get("license")),
    ):
        if source_manifest.get(key) != expected:
            errors.append(f"AsNumpy source-manifest {key} does not match component manifest")
    archive_meta = source_manifest.get("archive")
    files_meta = source_manifest.get("files")
    if not isinstance(archive_meta, dict) or not isinstance(files_meta, dict):
        errors.append("AsNumpy source-manifest must contain archive and files objects")
        return None
    for key in ("filename", "format", "prefix", "sha256", "bytes", "file_count", "total_file_bytes"):
        if archive_meta.get(key) != block.get(key):
            errors.append(f"AsNumpy source-manifest archive.{key} does not match component manifest")
    prefix = block.get("prefix")
    if not isinstance(prefix, str) or not prefix.endswith("/") or "\\" in prefix:
        errors.append("AsNumpy source_bundle prefix is invalid")
        return None
    file_count = block.get("file_count")
    total_file_bytes = block.get("total_file_bytes")
    archive_size = block.get("bytes")
    if not all(isinstance(value, int) and value > 0 for value in (file_count, total_file_bytes, archive_size)):
        errors.append("AsNumpy source_bundle counts are invalid")
        return None
    if len(files_meta) != file_count:
        errors.append("AsNumpy source-manifest file count mismatch")
    observed: dict[str, str] = {}
    observed_total = 0
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk() or not member.name.startswith(prefix):
                    errors.append("AsNumpy source archive contains an unsafe member")
                    continue
                relative = member.name[len(prefix):]
                path = PurePosixPath(relative)
                if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or relative in observed:
                    errors.append("AsNumpy source archive contains an unsafe path")
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    errors.append("AsNumpy source archive member is unreadable")
                    continue
                payload = stream.read()
                observed[relative] = hashlib.sha256(payload).hexdigest()
                observed_total += len(payload)
    except (OSError, tarfile.TarError, EOFError, ValueError) as exc:
        errors.append(f"invalid AsNumpy source archive: {exc}")
        return None
    if observed != files_meta:
        errors.append("AsNumpy source archive file inventory mismatch")
    if observed_total != total_file_bytes:
        errors.append("AsNumpy source archive total byte count mismatch")
    if len(archive_bytes) != archive_size:
        errors.append("AsNumpy source archive byte count mismatch")
    return {
        "filename": filename,
        "sha256": expected_archive_sha,
        "manifest_filename": manifest_filename,
        "manifest_sha256": expected_manifest_sha,
        "file_count": len(observed),
        "total_file_bytes": observed_total,
    }


def _validate_asnumpy_component(
    root: Path,
    files: Iterable[str],
    errors: list[str],
    warnings: list[str],
    *,
    require_artifact: bool = False,
) -> dict[str, object]:
    """Audit the release-adjacent AsNumpy lock and optional binary bundle.

    The source tree intentionally permits a development checkout without the
    platform-specific wheel.  Such a tree receives a warning in the normal
    audit and fails only when ``--require-asnumpy-component`` is requested.
    This keeps local documentation/tests useful while making the final release
    gate explicit and fail-closed.
    """

    file_set = set(files)
    result: dict[str, object] = {
        "schema": ASNUMPY_COMPONENT_SCHEMA,
        "manifest": ASNUMPY_COMPONENT_MANIFEST in file_set,
        "artifact_required": None,
        "wheel_present": False,
        "wheel_verified": False,
    }
    manifest_path = root / ASNUMPY_COMPONENT_MANIFEST
    readme_path = root / ASNUMPY_COMPONENT_README
    component_dir = root / ASNUMPY_COMPONENT_PREFIX
    if component_dir.is_symlink():
        errors.append("AsNumpy component directory must not be a symlink")
        return result
    if not manifest_path.is_file():
        # REQUIRED_FILES emits the canonical missing-file diagnostic.  Keep a
        # second component-specific error only for strict mode, where it is
        # useful to explain why the binary gate cannot run.
        if require_artifact:
            errors.append("AsNumpy component manifest is required for strict release validation")
        return result
    if manifest_path.is_symlink():
        errors.append("AsNumpy component manifest must not be a symlink")
        return result
    try:
        payload = manifest_path.read_bytes()
        if len(payload) > 512 * 1024:
            raise ValueError("manifest exceeds size limit")
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_json_pairs)
        if not isinstance(raw, dict):
            raise ValueError("manifest root must be an object")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid AsNumpy component manifest: {exc}")
        return result
    _component_scan_private(raw, ASNUMPY_COMPONENT_MANIFEST, errors)
    if raw.get("schema") != ASNUMPY_COMPONENT_SCHEMA:
        errors.append("unexpected AsNumpy component manifest schema")
    if raw.get("schema_version") != ASNUMPY_COMPONENT_SCHEMA_VERSION:
        errors.append("unsupported AsNumpy component manifest version")
    # Validate the identity fields even when the binary attachment is absent;
    # otherwise a malformed lock could be hidden behind the normal
    # development warning and become the source of truth at release time.
    for field in ("component_id", "distribution", "version", "repository", "commit", "tree", "license"):
        value = raw.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            errors.append(f"AsNumpy component manifest {field} is invalid")
    distribution = raw.get("distribution")
    if isinstance(distribution, str) and re.sub(r"[-_.]+", "-", distribution).lower() != "asnumpy":
        errors.append("AsNumpy component manifest distribution must be AsNumpy")
    repository = raw.get("repository")
    if isinstance(repository, str) and "://" not in repository:
        errors.append("AsNumpy component manifest repository must be an explicit URL")
    for field in ("commit", "tree"):
        value = raw.get(field)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is None:
            errors.append(f"AsNumpy component manifest {field} is not a 40-character Git SHA")
    build = raw.get("build")
    if isinstance(build, dict):
        for build_field, identity_field in (("source_commit", "commit"), ("source_tree", "tree")):
            declared = build.get(build_field)
            identity = raw.get(identity_field)
            if declared is not None and declared != identity:
                errors.append(
                    f"AsNumpy component build.{build_field} does not match manifest {identity_field}"
                )
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("AsNumpy component manifest runtime must be an object")
    elif any(
        not isinstance(value, (str, int, float, bool)) and value is not None
        for value in runtime.values()
    ):
        errors.append("AsNumpy component manifest runtime values must be scalar")
    if raw.get("artifact_required") is True and isinstance(runtime, dict):
        required_runtime = (
            "cann_version",
            "machine",
            "loguru_version",
            "numpy_version",
            "python_implementation",
            "python_major_minor",
        )
        missing_runtime = [key for key in required_runtime if key not in runtime]
        if missing_runtime:
            errors.append(
                "AsNumpy component manifest runtime constraints are missing: "
                + ", ".join(missing_runtime)
            )
        else:
            for key in required_runtime:
                value = runtime[key]
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"AsNumpy component manifest runtime constraint is invalid: {key}"
                    )
    source_files = raw.get("source_files_sha256")
    if not isinstance(source_files, dict):
        errors.append("AsNumpy component manifest source_files_sha256 must be an object")
    else:
        for relative, digest in source_files.items():
            try:
                _component_safe_relative(relative, "source file path")
            except ValueError as exc:
                errors.append(f"invalid AsNumpy component source file path: {exc}")
            if not isinstance(digest, str) or ASNUMPY_SHA256_RE.fullmatch(digest.lower()) is None:
                errors.append(f"invalid AsNumpy component source file SHA-256: {relative}")
    source_bundle_identity = _component_source_bundle_identity(root, raw, file_set, errors)
    if source_bundle_identity is not None:
        result["source_bundle"] = source_bundle_identity
    if not isinstance(raw.get("wheel"), dict):
        errors.append("AsNumpy component manifest wheel must be an object")
    artifact_required = raw.get("artifact_required")
    result["artifact_required"] = artifact_required
    if not isinstance(artifact_required, bool):
        errors.append("AsNumpy component artifact_required must be boolean")
        artifact_required = True
    if not readme_path.is_file():
        errors.append(f"missing AsNumpy component README: {ASNUMPY_COMPONENT_README}")
    elif readme_path.is_symlink():
        errors.append("AsNumpy component README must not be a symlink")
    else:
        try:
            readme_text = readme_path.read_text(encoding="utf-8")
            if PRIVATE_PATH_RE.search(readme_text):
                errors.append(f"private absolute workspace path in public text: {ASNUMPY_COMPONENT_README}")
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read {ASNUMPY_COMPONENT_README}: {exc}")

    # Optional legal/provenance attachments are checked only when declared by
    # the lock.  This lets the current development lock remain honest while
    # preventing a release lock from silently pointing at a missing SBOM.
    declared_attachments: dict[str, Path] = {}
    for label, keys in (
        ("license", ("license_file", "license_path")),
        ("sbom", ("sbom", "sbom_file", "sbom_path")),
        ("release_manifest", ("release_manifest", "release_manifest_file")),
    ):
        try:
            declaration = _component_declared_file(raw, keys=keys, label=label)
        except ValueError as exc:
            errors.append(f"invalid AsNumpy component {label} declaration: {exc}")
            continue
        if declaration is None:
            continue
        relative, expected = declaration
        if expected is None:
            message = f"AsNumpy component {label} declaration has no SHA-256"
            (errors if require_artifact else warnings).append(message)
        attachment = _component_file_hash(
            root / ASNUMPY_COMPONENT_PREFIX,
            relative,
            label=label,
            expected=expected,
            errors=errors,
            available_files=file_set,
        )
        if attachment is not None:
            declared_attachments[label] = attachment
        if attachment is not None and label == "sbom":
            declaration_value = next(
                (raw[key] for key in keys if key in raw), None
            )
            declared_format = (
                declaration_value.get("format")
                if isinstance(declaration_value, dict)
                else None
            )
            if declared_format:
                try:
                    sbom_payload = json.loads(attachment.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    sbom_payload = None
                if isinstance(sbom_payload, dict):
                    observed_format = (
                        sbom_payload.get("spdxVersion")
                        if str(declared_format).upper().startswith("SPDX-")
                        else sbom_payload.get("bomFormat")
                    )
                    if observed_format != declared_format:
                        errors.append(
                            "AsNumpy component SBOM format mismatch: "
                            f"expected {declared_format}, got {observed_format}"
                        )

    if artifact_required is True:
        # A frozen component must carry its legal notice and SPDX inventory;
        # allowing a binary-only lock would make the public release
        # provenance incomplete.  Development locks (artifact_required=false)
        # remain allowed to omit these optional attachments.
        for required_label in ("license", "sbom"):
            if required_label not in declared_attachments:
                errors.append(
                    f"AsNumpy component artifact_required=true must bind {required_label} attachment"
                )

    wheel_block = raw.get("wheel")
    expected_filename = wheel_block.get("filename") if isinstance(wheel_block, dict) else None
    if isinstance(wheel_block, dict):
        wheel_digest = wheel_block.get("sha256")
        if not isinstance(wheel_digest, str) or ASNUMPY_SHA256_RE.fullmatch(wheel_digest.lower()) is None:
            errors.append("AsNumpy component manifest wheel.sha256 is invalid")
        tags = wheel_block.get("tags")
        if not isinstance(tags, list) or not tags or not all(
            isinstance(tag, str) and bool(tag.strip()) and tag == tag.strip() for tag in tags
        ):
            errors.append("AsNumpy component manifest wheel.tags is invalid")
        elif len(tags) != len(set(tags)):
            errors.append("AsNumpy component manifest wheel.tags must not contain duplicates")
        for digest_key in ("metadata_sha256", "record_sha256"):
            digest = wheel_block.get(digest_key)
            if digest is not None and (
                not isinstance(digest, str)
                or ASNUMPY_SHA256_RE.fullmatch(digest.lower()) is None
            ):
                errors.append(f"AsNumpy component manifest wheel.{digest_key} is invalid")
    wheel_candidates = sorted(
        path
        for path in (root / ASNUMPY_COMPONENT_PREFIX).glob("*.whl")
        if (path.is_file() or path.is_symlink())
        and f"{ASNUMPY_COMPONENT_PREFIX}{path.name}" in file_set
    )
    result["wheel_files"] = [path.name for path in wheel_candidates]
    if expected_filename is not None and (
        not isinstance(expected_filename, str)
        or ASNUMPY_WHEEL_RE.fullmatch(expected_filename) is None
        or Path(expected_filename).name != expected_filename
    ):
        errors.append("AsNumpy component manifest wheel.filename is unsafe")
        expected_filename = None
    unexpected = [path.name for path in wheel_candidates if path.name != expected_filename]
    if unexpected:
        errors.append(
            "unexpected AsNumpy component wheel attachment(s): " + ", ".join(unexpected)
        )
    wheel_path = (
        root / ASNUMPY_COMPONENT_PREFIX / expected_filename
        if isinstance(expected_filename, str)
        else None
    )
    wheel_relative = (
        f"{ASNUMPY_COMPONENT_PREFIX}{expected_filename}"
        if isinstance(expected_filename, str)
        else None
    )
    wheel_in_candidate = wheel_relative is not None and wheel_relative in file_set
    if wheel_path is None or not wheel_path.is_file() or wheel_path.is_symlink():
        result["wheel_present"] = False
        message = "AsNumpy component wheel attachment is missing"
        (errors if require_artifact else warnings).append(message)
    elif not wheel_in_candidate:
        result["wheel_present"] = False
        message = (
            "AsNumpy component wheel attachment is not part of the Git release candidate"
        )
        (errors if require_artifact else warnings).append(message)
    else:
        result["wheel_present"] = True
        wheel_errors_before = len(errors)
        identity = _component_wheel_identity(wheel_path, raw, errors)
        if identity is not None and len(errors) == wheel_errors_before:
            result["wheel_verified"] = True
            result["wheel_identity"] = identity
    artifact_status = raw.get("artifact_status")
    if artifact_status is not None and not isinstance(artifact_status, str):
        errors.append("AsNumpy component artifact_status is invalid")
    elif isinstance(artifact_status, str) and artifact_status not in {
        "bundled-and-verified",
        "external",
        "missing",
        "pending",
        "unverified",
    }:
        errors.append(f"AsNumpy component artifact_status is unknown: {artifact_status}")
    elif artifact_status == "bundled-and-verified" and not result["wheel_verified"]:
        errors.append(
            "AsNumpy component artifact_status claims bundled-and-verified, but wheel verification failed"
        )
    elif artifact_status in {"missing", "pending", "unverified"} and result["wheel_verified"]:
        errors.append(
            "AsNumpy component artifact_status does not match the verified wheel attachment"
        )
    sbom_path = declared_attachments.get("sbom")
    if sbom_path is not None and isinstance(wheel_block, dict):
        try:
            sbom_text = sbom_path.read_text(encoding="utf-8")
            sbom_payload = json.loads(sbom_text)
        except (OSError, UnicodeError, json.JSONDecodeError):
            sbom_payload = None
            sbom_text = ""
        expected_wheel_sha = wheel_block.get("sha256")
        expected_version = raw.get("version")
        if isinstance(sbom_payload, dict):
            # The SBOM is an attachment to this exact component, not a generic
            # package inventory.  Require it to mention the locked wheel hash
            # and version so a stale SBOM cannot silently travel with a new
            # binary.
            if isinstance(expected_wheel_sha, str) and expected_wheel_sha.lower() not in sbom_text.lower():
                errors.append("AsNumpy component SBOM does not bind the locked wheel SHA-256")
            if isinstance(expected_version, str) and expected_version not in sbom_text:
                errors.append("AsNumpy component SBOM does not bind the locked version")
    if artifact_required is True and not result["wheel_present"] and require_artifact:
        # Keep one deterministic diagnostic even if filename validation also
        # failed; callers can key on this message in CI.
        if "AsNumpy component wheel attachment is missing" not in errors:
            errors.append("AsNumpy component artifact_required=true but wheel is not attached")
    return result


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


def _validate_architecture_png(root: Path, errors: list[str]) -> None:
    for relative in (
        "docs/assets/architecture/numpilot-architecture-en.png",
        "docs/assets/architecture/numpilot-architecture-zh.png",
    ):
        path = root / relative
        if not path.is_file():
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            errors.append(f"cannot read architecture PNG: {relative}")
            continue
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            errors.append(f"architecture asset is not PNG: {relative}")

def _scan_public_text(root: Path, files: Iterable[str], errors: list[str]) -> None:
    for relative in files:
        path = PurePosixPath(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not (
            relative in {"README.md", "README.zh-CN.md"}
            or relative.startswith("docs/")
            or relative.startswith(ASNUMPY_COMPONENT_PREFIX)
            or relative.startswith("evidence/")
            or relative in {"SECURITY.md", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md"}
        ):
            continue
        text = (root / relative).read_text(encoding="utf-8", errors="replace")
        if PRIVATE_PATH_RE.search(text):
            errors.append(f"private absolute workspace path in public text: {relative}")


def audit_release_tree(
    root: Path,
    *,
    require_clean: bool = False,
    require_asnumpy_component: bool = False,
) -> dict:
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
        if "<img" not in readme or re.search(
            r"status-(?:alpha|beta|stable)", readme
        ) is None:
            errors.append(
                f"{relative} must retain rendered badges and an honest release-stage label"
            )
        if "docs/assets/architecture/" not in readme:
            errors.append(f"{relative} must link the product architecture assets")
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
    asnumpy_component = _validate_asnumpy_component(
        root,
        files,
        errors,
        warnings,
        require_artifact=require_asnumpy_component,
    )
    version = _validate_versions(root, errors)
    _validate_architecture_png(root, errors)

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
        "asnumpy_component": asnumpy_component,
        "errors": errors,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--require-asnumpy-component",
        action="store_true",
        help="require and fully verify the hash-bound AsNumpy wheel attachment",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_release_tree(
        args.repo_root,
        require_clean=args.require_clean,
        require_asnumpy_component=args.require_asnumpy_component,
    )
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
