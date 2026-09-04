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

"""Read-only competition submission readiness audit.

The contest asks for considerably more than a runnable migration script:
CodeArts development records, a Huawei Cloud demo, reproducible source and
evidence, and a real business/data context are all part of the judging
criteria.  This module turns those requirements into an intentionally
conservative local checklist.

The auditor never imports or executes a project module, never calls a model or
the network, and never treats declarative ``.codeartsdoer`` files as proof that
CodeArts actually ran.  It only reads bounded metadata, hashes files named by
an existing manifest, and (optionally) reads local git status.  A report may be
written to an explicitly supplied path; that is the sole write side effect.

The machine-readable evidence contracts accepted by this checker are kept
small and redacted:

``evidence/codearts-usage.json``
    ``actual_execution: true``, ``platform`` containing CodeArts, and a
    ``features`` array.  The five feature names used by the contest are
    ``code_generation``, ``knowledge_qa``, ``unit_test_generation``,
    ``codebase_index`` and ``spec_driven_development``.  An optional
    ``records`` mapping/list can bind each feature to a platform export via
    ``record_id``, repository-relative ``record_path``, an explicit UTC
    timestamp, ``sha256`` and ``status: passed``.  The compact contract stays
    backwards compatible; when ``records`` is supplied, only complete,
    hash-matching entries are considered traceable.

``evidence/cloud-deployment.json``
    ``provider`` containing Huawei/华为云, ``status`` equal to ``verified`` or
    ``deployed``, and ``verification`` true.  Credentials are never read or
    printed; an endpoint is reduced to its hostname if displayed.

``evidence/workload-manifest.json``
    A real workload declaration with non-empty ``data_source``, ``license``,
    ``beneficiary``, ``source_sha256`` and ``data_sha256``.  The declaration
    is evidence of a submission package, not a claim that NumPilot independently
    verified the business relationship.

These files are optional today, so a current development checkout is expected
to report blockers.  That is preferable to a green score based on prose or
repository configuration alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
CONTEST_DEADLINE = date(2026, 9, 5)

PASS = "pass"
PARTIAL = "partial"
WARN = "warn"
BLOCKED = "blocked"
INFO = "info"

_MAX_TEXT_BYTES = 2 * 1024 * 1024
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_ENTRIES = 20_000
_MAX_HASH_BYTES = 128 * 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 2 * 1024 * 1024


def _contains_control_text(value: str) -> bool:
    """Return whether text contains terminal-control or Unicode format chars.

    Path and report metadata eventually appear in human-readable/JSON output.
    Rejecting all Unicode ``C*`` categories (in addition to non-printable
    characters) keeps bidi/invisible marks from creating misleading output.
    This mirrors the admission rule used by :mod:`numpilot.llm_config` and
    :mod:`numpilot.workflow`.
    """

    return any(
        not character.isprintable()
        or unicodedata.category(character).startswith("C")
        for character in value
    )

_FEATURE_ALIASES = {
    "code_generation": (
        "code_generation",
        "code generation",
        "代码生成",
        "生成代码",
    ),
    "knowledge_qa": (
        "knowledge_qa",
        "knowledge qa",
        "knowledge_q&a",
        "研发知识问答",
        "知识问答",
    ),
    "unit_test_generation": (
        "unit_test_generation",
        "unit test generation",
        "unit_test",
        "单元测试用例生成",
        "单元测试生成",
    ),
    "codebase_index": (
        "codebase_index",
        "codebase index",
        "codebase索引",
        "代码库索引",
        "codebase",
    ),
    "spec_driven_development": (
        "spec_driven_development",
        "spec-driven development",
        "spec driven development",
        "规范驱动开发",
        "规范驱动",
    ),
}
_REQUIRED_FEATURES = tuple(_FEATURE_ALIASES)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_RECORD_ENTRIES = 64
_MAX_RECORD_ID_LENGTH = 512
_RECORD_FIELDS = ("record_id", "record_path", "timestamp_utc", "sha256", "status")
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
_VIDEO_URL_RE = re.compile(
    r"https?://[^\s<>\"']+\.(?:mp4|mov|webm|mkv)(?:\?[^\s<>\"']*)?",
    re.IGNORECASE,
)
_VIDEO_MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]]*(?:video|demo|recording|视频|演示|录屏)[^\]]*)\]\(\s*(https?://[^\s)<>\"]+)\s*\)",
    re.IGNORECASE,
)
_VIDEO_HTML_LINK_RE = re.compile(
    r"<a\b[^>]*href=[\"'](https?://[^\"'<>\s]+)[\"'][^>]*>"
    r"[^<]*(?:video|demo|recording|视频|演示|录屏)[^<]*</a>",
    re.IGNORECASE,
)
_VIDEO_HOST_HINTS = (
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "b23.tv",
    "v.qq.com",
    "douyin.com",
    "kuaishou.com",
)

# Feature exports from CodeArts are normally exact identifiers, but some
# account-level reports use a short human label around the identifier (for
# example ``"enabled: code_generation"``).  Keep accepting those labels while
# refusing an easy false positive such as ``"not code_generation"``.  The
# readiness audit is intentionally conservative: a negative statement must not
# be promoted to evidence merely because it contains the required feature
# name.
_FEATURE_NEGATION_MARKERS = (
    "not",
    "no",
    "without",
    "missing",
    "unsupported",
    "unavailable",
    "disabled",
    "not_enabled",
    "false",
    "未",
    "无",
    "缺",
    "不支持",
    "不具备",
    "尚未",
    "没有",
    "未启用",
    "禁用",
    "关闭",
)
_FEATURE_ENGLISH_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "without",
        "missing",
        "unsupported",
        "unavailable",
        "disabled",
        "not_enabled",
        "false",
    }
)


def _feature_alias_matches(item: str, alias: str) -> bool:
    """Match a feature alias without treating an explicit negation as proof."""

    if item == alias:
        return True
    marker = item.find(alias)
    if marker < 0:
        return False
    prefix = item[:marker].strip(" :：-_()[]{}")
    suffix = item[marker + len(alias) :].strip(" :：-_()[]{}")

    def _is_negated(fragment: str) -> bool:
        if not fragment:
            return False
        # English markers are separated by whitespace in exported labels;
        # Chinese markers are commonly attached directly to the feature name.
        lowered = fragment.lower()
        for negative in _FEATURE_NEGATION_MARKERS:
            candidate = negative.lower()
            if lowered == candidate or lowered.endswith(" " + candidate):
                return True
            if negative in _FEATURE_ENGLISH_NEGATIONS:
                if lowered.startswith(candidate + " ") or lowered.startswith(
                    candidate + ":"
                ):
                    return True
            elif candidate and (
                lowered.startswith(candidate) or lowered.endswith(candidate)
            ):
                return True
        return False

    return not _is_negated(prefix) and not _is_negated(suffix)


@dataclass(frozen=True)
class ReadinessCheck:
    """One auditable requirement and its observed local evidence."""

    check_id: str
    dimension: str
    title: str
    status: str
    critical: bool
    requirement: str
    evidence: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "dimension": self.dimension,
            "title": self.title,
            "status": self.status,
            "critical": self.critical,
            "requirement": self.requirement,
            "evidence": list(self.evidence),
            "gaps": list(self.gaps),
            "details": _json_safe(dict(self.details)),
        }


@dataclass(frozen=True)
class CompetitionReadinessReport:
    """Serializable result of :func:`assess_competition_readiness`."""

    generated_at_utc: str
    repo_root: str
    read_only: bool
    source_execution_performed: bool
    network_accessed: bool
    deadline: Mapping[str, Any]
    git: Mapping[str, Any]
    llm_config: Mapping[str, Any]
    dimensions: Mapping[str, Mapping[str, Any]]
    checks: tuple[ReadinessCheck, ...]
    blockers: tuple[str, ...]
    next_actions: tuple[str, ...]
    overall_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": "numpilot-competition-readiness-v1",
            "generated_at_utc": self.generated_at_utc,
            "repo_root": self.repo_root,
            "read_only": self.read_only,
            "source_execution_performed": self.source_execution_performed,
            "network_accessed": self.network_accessed,
            "deadline": _json_safe(dict(self.deadline)),
            "git": _json_safe(dict(self.git)),
            "llm_config": _json_safe(dict(self.llm_config)),
            "dimensions": _json_safe(dict(self.dimensions)),
            "checks": [item.to_dict() for item in self.checks],
            "blockers": list(self.blockers),
            "next_actions": list(self.next_actions),
            "overall_status": self.overall_status,
        }


def _json_safe(value: Any) -> Any:
    """Convert paths/tuples recursively without exposing secrets."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _has_symlink_component(path: Path) -> bool:
    """Return whether *path* or any existing parent is a symlink.

    Readiness is often run against a checkout containing untrusted artifacts.
    Checking only the final directory entry is insufficient: a symlinked
    ``evidence/`` directory could otherwise make the audit read credentials or
    claim files from outside the checkout.  This lexical walk deliberately
    avoids ``resolve()`` until after the check.
    """

    try:
        rendered = os.fspath(path)
        if not isinstance(rendered, str):
            return True
        # Preserve the caller's ``..`` components while checking.  Calling
        # ``abspath`` first would normalize ``link/..`` and hide a symlink
        # that the kernel resolves before applying ``..``.
        expanded = os.path.expanduser(rendered)
        current = (
            Path(os.path.abspath(os.sep))
            if os.path.isabs(expanded)
            else Path(os.getcwd())
        )
        for component in expanded.split(os.sep):
            if not component or component == ".":
                continue
            if component == "..":
                current = current.parent
                continue
            candidate = current / component
            if os.path.lexists(candidate) and candidate.is_symlink():
                return True
            current = candidate
        return False
    except (OSError, RuntimeError, TypeError, ValueError):
        # If a component cannot be inspected, fail closed rather than
        # following it through a potentially hostile mount/link.
        return True


def _safe_regular_file(path: Path) -> bool:
    """Check a regular file without following a symlinked path component."""

    try:
        return (
            not _has_symlink_component(path)
            and path.is_file()
            and not path.is_symlink()
        )
    except OSError:
        return False


def _validated_repo_root(value: str | os.PathLike[str]) -> Path:
    """Return a lexical checkout root without following symlink components.

    ``Path.resolve()`` is convenient for ordinary callers, but resolving first
    would erase the very symlink boundary this read-only auditor promises to
    reject (for example, ``checkout/evidence`` redirected elsewhere).  Keep
    the path lexical, validate its existing components, and only then let the
    individual bounded readers inspect files below it.
    """

    try:
        rendered = os.fspath(value)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("仓库路径无效") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or _contains_control_text(rendered)
        or len(rendered) > 4096
    ):
        raise ValueError("仓库路径无效")
    try:
        expanded = os.path.expanduser(rendered)
        # Validate the unnormalized spelling before deriving the canonical
        # root; otherwise ``symlink/..`` could evade the boundary check.
        if _has_symlink_component(Path(expanded)):
            raise ValueError("仓库路径不得包含符号链接")
        root = Path(os.path.abspath(expanded))
    except ValueError as exc:
        if "符号链接" in str(exc):
            raise
        raise ValueError("仓库路径无效") from exc
    except (OSError, RuntimeError) as exc:
        raise ValueError("仓库路径无效") from exc
    return root


def _bounded_bytes(path: Path, limit: int) -> Optional[bytes]:
    """Read a regular, non-symlink file up to ``limit`` bytes.

    A size check followed by ``Path.read_bytes`` is racy: a concurrent writer
    can grow the file after the check and force an unbounded allocation.  Use
    one descriptor, bounded chunks, and a final metadata comparison instead.
    """

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        return None
    descriptor = -1
    try:
        # Keep the inexpensive lexical check for clear symlink semantics, then
        # let O_NOFOLLOW/fstat establish the descriptor's actual file type.
        if not _safe_regular_file(path):
            return None
        flags = os.O_RDONLY
        for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(os.fspath(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            return None
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                return None
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            return None
        return b"".join(chunks)
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _bounded_text(path: Path, limit: int = _MAX_TEXT_BYTES) -> Optional[str]:
    payload = _bounded_bytes(path, limit)
    if payload is None:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _bounded_json(path: Path) -> Optional[dict[str, Any]]:
    text = _bounded_text(path, _MAX_JSON_BYTES)
    if text is None:
        return None
    try:
        value = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(path)


def _existing(root: Path, candidates: Iterable[str]) -> list[str]:
    found = []
    for name in candidates:
        path = root / name
        try:
            if _safe_regular_file(path):
                found.append(name)
        except OSError:
            continue
    return found


def _find_named(root: Path, names: Iterable[str]) -> list[Path]:
    """Find exact relative names in a bounded set of likely evidence folders."""

    candidates: list[Path] = []
    names = tuple(names)
    for rel in names:
        candidates.append(root / rel)
    evidence = root / "evidence"
    if (
        not _has_symlink_component(evidence)
        and evidence.is_dir()
        and not evidence.is_symlink()
    ):
        try:
            for child in evidence.iterdir():
                if (
                    not _has_symlink_component(child)
                    and child.is_dir()
                    and not child.is_symlink()
                ):
                    for rel in names:
                        # Only the basename is appended below evidence/<case>/.
                        if "/" not in rel:
                            candidates.append(child / rel)
        except OSError:
            pass
    result = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if _safe_regular_file(path):
                result.append(path)
        except OSError:
            continue
    return result


def _status_for_dimension(checks: Sequence[ReadinessCheck]) -> str:
    if any(item.status == BLOCKED and item.critical for item in checks):
        return BLOCKED
    if any(item.status in {PARTIAL, WARN} for item in checks):
        return PARTIAL
    if any(item.status == BLOCKED for item in checks):
        return PARTIAL
    return PASS


def _git_snapshot(root: Path) -> dict[str, Any]:
    """Collect local git facts; no remote operation is ever attempted."""

    result: dict[str, Any] = {
        "is_repository": False,
        "git_root": None,
        "root_matches": False,
        "commit": None,
        "branch": None,
        "tag": None,
        "dirty": None,
        "status_entries": None,
        "remote_checked": False,
    }

    def run_git(*args: str) -> Optional[str]:
        try:
            # A hostile checkout may configure a fsmonitor command or hooks.
            # Disable all optional/configured callbacks: this audit is read
            # only and must not execute repository-controlled programs.
            environment = os.environ.copy()
            environment.update(
                {
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_OPTIONAL_LOCKS": "0",
                }
            )
            completed = subprocess.run(
                [
                    "git",
                    "-c",
                    "core.fsmonitor=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-C",
                    str(root),
                    *args,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, UnicodeError, subprocess.SubprocessError):
            return None
        try:
            returncode = completed.returncode
            output = completed.stdout
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        if returncode != 0:
            return None
        # ``git status --untracked-files=all`` is attacker-influenced and can
        # enumerate an arbitrarily large tree.  Keep the report bounded and
        # fail closed when a command returns an unexpected output object.
        if not isinstance(output, str) or len(output) > _MAX_GIT_OUTPUT_BYTES:
            return None
        return output.strip()

    git_root = run_git("rev-parse", "--show-toplevel")
    if git_root is None:
        return result
    # ``git -C subdir`` succeeds against an enclosing checkout.  That is not
    # sufficient for a release audit: a caller asking us to inspect a nested
    # directory must not inherit the parent repository's clean commit/tag and
    # accidentally bind evidence outside the declared root.  Compare lexical
    # real paths only after the repository root has already passed the
    # symlink-rejection gate in ``_validated_repo_root``.
    result["git_root"] = git_root
    try:
        result["root_matches"] = (
            os.path.normcase(os.path.realpath(git_root))
            == os.path.normcase(os.path.realpath(str(root)))
        )
    except (OSError, TypeError, ValueError):
        result["root_matches"] = False
    if not result["root_matches"]:
        return result
    result["is_repository"] = True
    result["commit"] = run_git("rev-parse", "HEAD")
    result["branch"] = run_git("symbolic-ref", "--short", "HEAD")
    result["tag"] = run_git("describe", "--tags", "--exact-match", "HEAD")
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    if status is None:
        # An unavailable/oversized status must not be mistaken for a clean
        # checkout.  Preserve ``None`` so C10 remains blocked until a bounded
        # status can be observed.
        result["status_entries"] = None
        result["dirty"] = None
    else:
        entries = status.splitlines()
        result["status_entries"] = len(entries)
        result["dirty"] = bool(entries)
    return result


def _llm_snapshot(
    root: Path, config_path: Optional[str | os.PathLike[str]]
) -> dict[str, Any]:
    """Inspect only existence/permissions; never open the config contents.

    Even parsing a JSON object would load the token into this process.  The
    readiness audit therefore reports ``configured`` as ``None`` and leaves
    semantic validation to the explicit ``configure-llm --check`` command.
    """

    # Reuse the framework's resolver so explicit paths, NUMPILOT_* overrides,
    # XDG_CONFIG_HOME, HOME, and the persistent runtime aliases all describe
    # exactly the same file as ``configure-llm``/``load_llm_config``.  Import
    # lazily: this audit still never reads the config contents or validates a
    # model endpoint.
    path_valid = True
    try:
        from numpilot.llm_config import llm_config_path

        path = llm_config_path(config_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        path = Path("<invalid-llm-config-path>")
        path_valid = False
    info: dict[str, Any] = {
        "path": str(path),
        "path_valid": path_valid,
        "exists": False,
        "regular_file": False,
        "permissions_safe": None,
        "configured": None,
        "content_read": False,
    }
    if not path_valid:
        return info
    try:
        if _has_symlink_component(path):
            # Keep the metadata useful while refusing to bless a redirected
            # private configuration file.
            info["exists"] = path.exists()
            return info
        info["exists"] = path.exists()
        info["regular_file"] = _safe_regular_file(path)
        if info["regular_file"] and os.name == "posix":
            info["permissions_safe"] = (path.stat().st_mode & 0o077) == 0
    except OSError:
        return info
    return info


def _manifest_hash_snapshot(root: Path) -> dict[str, Any]:
    """Verify an existing evidence manifest without invoking ``sha256sum``."""

    manifest = root / "evidence" / "MANIFEST.sha256"
    info: dict[str, Any] = {
        "path": "evidence/MANIFEST.sha256",
        "exists": False,
        "entries": 0,
        "checked": 0,
        "missing": [],
        "mismatched": [],
        "duplicates": [],
        "unlisted": [],
        "invalid": 0,
        "complete": False,
    }
    payload = _bounded_text(manifest, _MAX_TEXT_BYTES)
    if payload is None:
        return info
    info["exists"] = True
    lines = payload.splitlines()
    if len(lines) > _MAX_MANIFEST_ENTRIES:
        info["invalid"] = len(lines) - _MAX_MANIFEST_ENTRIES
        lines = lines[:_MAX_MANIFEST_ENTRIES]
    listed: set[str] = set()
    for line in lines:
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or _SHA256_RE.fullmatch(fields[0]) is None:
            if line.strip():
                info["invalid"] += 1
            continue
        digest, name = fields
        # sha256sum emits two spaces for ordinary names; tolerate one as well.
        name = name.lstrip(" *")
        # The repository-level manifest is scoped to evidence/; reject
        # aliases such as ``evidence/../README.md`` even when their resolved
        # target happens to remain inside the checkout.
        try:
            canonical_name = PurePosixPath(name).as_posix()
        except (TypeError, ValueError):
            canonical_name = ""
        if (
            not name
            or "\x00" in name
            or canonical_name != name
            or not name.startswith("evidence/")
            or len(name) > 4096
            or _contains_control_text(name)
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            info["invalid"] += 1
            continue
        if name in listed:
            info["duplicates"].append(name)
            continue
        listed.add(name)
        original_path = root / name
        try:
            # A manifest must not silently bless a symlink, even when its
            # target remains inside the checkout.
            if _has_symlink_component(original_path) or original_path.is_symlink():
                info["invalid"] += 1
                continue
        except OSError:
            info["missing"].append(name)
            continue
        try:
            # Resolve only for the lexical containment check; the actual
            # bytes are read from ``original_path`` through one bounded,
            # no-follow descriptor below.  This avoids a symlink replacement
            # race between ``resolve`` and ``open``.
            path = original_path.resolve()
            path.relative_to(root.resolve())
        except (OSError, RuntimeError, TypeError, ValueError):
            info["invalid"] += 1
            continue
        info["entries"] += 1
        try:
            if not path.is_file():
                info["missing"].append(name)
                continue
            if path.stat().st_size > _MAX_HASH_BYTES:
                info["invalid"] += 1
                continue
            payload = _bounded_bytes(original_path, _MAX_HASH_BYTES)
            if payload is None:
                info["missing"].append(name)
                continue
            hasher = hashlib.sha256(payload)
            info["checked"] += 1
            if hasher.hexdigest() != digest.lower():
                info["mismatched"].append(name)
        except (OSError, RuntimeError, TypeError, ValueError):
            info["missing"].append(name)
    # A complete checksum list must cover every regular evidence file, not
    # merely verify the entries it happens to contain.  This catches an
    # unlisted screenshot/log accidentally added after the manifest was
    # generated.  Symlinks are invalid evidence regardless of their target.
    actual: set[str] = set()
    try:
        for index, candidate in enumerate((root / "evidence").rglob("*")):
            if index >= _MAX_MANIFEST_ENTRIES:
                info["invalid"] += 1
                break
            if candidate == manifest:
                continue
            if candidate.is_symlink():
                info["invalid"] += 1
                continue
            if candidate.is_file():
                actual.add(candidate.relative_to(root).as_posix())
    except (OSError, ValueError, RuntimeError):
        info["invalid"] += 1
    info["unlisted"] = sorted(actual - listed)
    info["complete"] = bool(
        info["exists"]
        and info["entries"] > 0
        and info["checked"] == info["entries"]
        and not info["missing"]
        and not info["mismatched"]
        and not info["duplicates"]
        and not info["unlisted"]
        and not info["invalid"]
    )
    return info


def _canonical_feature(value: Any) -> Optional[str]:
    """Return a required CodeArts feature name for a safe human label.

    CodeArts exports normally use the identifiers in ``_REQUIRED_FEATURES``,
    while account-level exports sometimes add a short label around one of the
    identifiers.  Reuse the same alias/negation rules as the legacy
    ``features`` list, but keep this helper strict about untrusted values so a
    malformed ``records`` object cannot leak control characters into the
    report.
    """

    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_RECORD_ID_LENGTH
        or _contains_control_text(value)
    ):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    for feature in _REQUIRED_FEATURES:
        if any(
            _feature_alias_matches(normalized, alias.lower())
            for alias in _FEATURE_ALIASES[feature]
        ):
            return feature
    return None


def _record_timestamp_is_utc(value: Any) -> bool:
    """Validate an ISO-8601 timestamp that explicitly names UTC.

    A naive timestamp is ambiguous in a cross-account export.  Requiring a
    trailing ``Z`` or ``+00:00`` makes the contract deterministic while still
    accepting the two common serializers used by CodeArts integrations.
    """

    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or _contains_control_text(value)
        or not (value.endswith("Z") or value.endswith("+00:00"))
    ):
        return False
    try:
        rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(rendered)
    except (TypeError, ValueError, OverflowError):
        return False
    try:
        return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    except (OverflowError, ValueError):
        return False


def _safe_codearts_record_path(root: Path, value: Any) -> tuple[Path | None, str | None]:
    """Validate a repository-relative record path without following links.

    The record path is intentionally lexical: absolute paths, ``..`` aliases,
    Windows separators and symlinked components are rejected before any bytes
    are read.  Returning the canonical relative spelling separately keeps
    report output free of host-specific absolute paths.
    """

    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or _contains_control_text(value)
        or "\\" in value
    ):
        return None, None
    try:
        canonical = PurePosixPath(value).as_posix()
    except (TypeError, ValueError):
        return None, None
    parts = value.split("/")
    if (
        canonical != value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        return None, None
    try:
        candidate = root.joinpath(*parts)
        candidate.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, None
    if not _safe_regular_file(candidate):
        return None, None
    return candidate, canonical


def _codearts_records(root: Path, obj: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate optional per-feature CodeArts export records.

    The original compact contract (``actual_execution`` + ``features``) is
    deliberately retained for existing evidence.  When a ``records`` field
    is present, however, it is treated as an explicit traceability claim and
    every required feature must carry all five fields plus a matching local
    file hash before it is counted as verified.  Both a mapping keyed by
    feature and a list of objects with a ``feature`` field are accepted so
    exports from different CodeArts views remain easy to ingest.
    """

    base: dict[str, Any] = {
        "present": False,
        "shape_valid": None,
        "status": "legacy",
        "complete": False,
        "required_features": list(_REQUIRED_FEATURES),
        "verified_features": [],
        "missing_features": list(_REQUIRED_FEATURES),
        "invalid_features": [],
        "duplicate_features": [],
        "file_missing_features": [],
        "hash_mismatched_features": [],
        "entries": {},
    }
    if not isinstance(obj, Mapping) or "records" not in obj:
        return base

    base["present"] = True
    raw = obj.get("records")
    pairs: list[tuple[Any, Any]] = []
    shape_valid = True
    if isinstance(raw, Mapping):
        # Also accept a single record object with an explicit feature/name;
        # this gives a useful partial result instead of treating its fields as
        # feature keys.
        if (
            any(field in raw for field in _RECORD_FIELDS)
            and ("feature" in raw or "name" in raw)
        ):
            pairs.append((raw.get("feature", raw.get("name")), raw))
        else:
            if len(raw) > _MAX_RECORD_ENTRIES:
                shape_valid = False
            pairs.extend(list(raw.items())[:_MAX_RECORD_ENTRIES])
    elif isinstance(raw, list):
        if len(raw) > _MAX_RECORD_ENTRIES:
            shape_valid = False
        for item in raw[:_MAX_RECORD_ENTRIES]:
            if isinstance(item, Mapping):
                pairs.append((item.get("feature", item.get("name")), item))
            else:
                pairs.append((None, item))
    else:
        shape_valid = False
    base["shape_valid"] = bool(shape_valid and isinstance(raw, (Mapping, list)))

    seen: set[str] = set()
    for raw_feature, raw_entry in pairs:
        feature = _canonical_feature(raw_feature)
        if feature is None:
            # Do not echo an untrusted key or value in a public report.
            base["invalid_features"].append("<invalid>")
            continue
        if feature in seen:
            base["duplicate_features"].append(feature)
            continue
        seen.add(feature)
        details: dict[str, Any] = {"valid": False, "issues": []}
        if not isinstance(raw_entry, Mapping):
            details["issues"] = ["record_not_object"]
            base["invalid_features"].append(feature)
            base["entries"][feature] = details
            continue

        record_id = raw_entry.get("record_id")
        if (
            isinstance(record_id, str)
            and record_id
            and len(record_id) <= _MAX_RECORD_ID_LENGTH
            and not _contains_control_text(record_id)
        ):
            details["record_id"] = record_id
        else:
            details["issues"].append("record_id_invalid")

        record_path, relative_path = _safe_codearts_record_path(
            root, raw_entry.get("record_path")
        )
        if record_path is None or relative_path is None:
            details["issues"].append("record_path_invalid")
        else:
            details["record_path"] = relative_path

        timestamp = raw_entry.get("timestamp_utc")
        if _record_timestamp_is_utc(timestamp):
            details["timestamp_utc"] = timestamp
        else:
            details["issues"].append("timestamp_utc_invalid")

        digest = raw_entry.get("sha256")
        digest_ok = isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None
        if digest_ok:
            details["sha256"] = digest.lower()
        else:
            details["issues"].append("sha256_invalid")

        if raw_entry.get("status") == "passed":
            details["status"] = "passed"
        else:
            details["issues"].append("status_not_passed")

        if record_path is not None and digest_ok:
            payload = _bounded_bytes(record_path, _MAX_HASH_BYTES)
            if payload is None:
                details["issues"].append("record_file_unreadable")
                base["file_missing_features"].append(feature)
            elif hashlib.sha256(payload).hexdigest() != str(digest).lower():
                details["issues"].append("record_hash_mismatch")
                base["hash_mismatched_features"].append(feature)

        if not details["issues"]:
            details["valid"] = True
            base["verified_features"].append(feature)
        else:
            base["invalid_features"].append(feature)
        base["entries"][feature] = details

    base["verified_features"] = [
        feature for feature in _REQUIRED_FEATURES if feature in base["verified_features"]
    ]
    base["missing_features"] = [
        feature
        for feature in _REQUIRED_FEATURES
        if feature not in base["verified_features"]
    ]
    # Preserve order while removing duplicate diagnostics.
    for key in (
        "invalid_features",
        "duplicate_features",
        "file_missing_features",
        "hash_mismatched_features",
    ):
        base[key] = list(dict.fromkeys(base[key]))
    base["complete"] = bool(
        base["shape_valid"]
        and len(base["verified_features"]) == len(_REQUIRED_FEATURES)
        and not base["invalid_features"]
        and not base["duplicate_features"]
    )
    if base["complete"]:
        base["status"] = "verified"
    elif base["verified_features"]:
        base["status"] = "partial"
    else:
        base["status"] = "blocked"
    return base


def _codearts_evidence(root: Path) -> dict[str, Any]:
    candidates = _find_named(
        root,
        (
            "evidence/codearts-usage.json",
            "evidence/codearts.json",
            "codearts-usage.json",
        ),
    )
    repo_config = _existing(
        root,
        (
            ".codeartsdoer/agents/numpilot-orchestrator.md",
            "skill/SKILL.md",
            ".codeartsdoer/skills/numpilot-migration-gate/SKILL.md",
        ),
    )
    best: Optional[Path] = None
    obj: Optional[dict[str, Any]] = None
    for path in candidates:
        parsed = _bounded_json(path)
        if parsed is not None:
            best, obj = path, parsed
            break
    features_raw: list[Any] = []
    actual = False
    platform = ""
    if obj is not None:
        actual = obj.get("actual_execution") is True
        platform = str(obj.get("platform") or obj.get("provider") or "")
        raw = obj.get("features", [])
        if isinstance(raw, list):
            features_raw = raw
    normalized = {str(value).strip().lower() for value in features_raw}
    found_features = []
    missing_features = []
    for feature in _REQUIRED_FEATURES:
        aliases = _FEATURE_ALIASES[feature]
        if any(
            _feature_alias_matches(item, alias.lower())
            for item in normalized
            for alias in aliases
        ):
            found_features.append(feature)
        else:
            missing_features.append(feature)
    platform_ok = "codearts" in platform.lower() or "码道" in platform
    verified = bool(best and actual and platform_ok)
    records = _codearts_records(root, obj)
    return {
        "repo_config_present": bool(repo_config),
        "repo_config": repo_config,
        "evidence_path": _relative(root, best) if best else None,
        "actual_execution": actual,
        "platform_ok": platform_ok,
        "verified": verified,
        "features": found_features,
        "missing_features": missing_features,
        "proof_level": "local_self_attestation" if best else "none",
        "platform_api_verified": False,
        # ``records`` is an optional stronger contract.  Keep the compact
        # legacy fields above untouched so existing evidence remains readable,
        # while exposing enough detail for a final-submission audit to reject
        # incomplete or hash-drifted platform exports.
        "records_present": records["present"],
        "records_complete": records["complete"],
        "records_status": records["status"],
        "records_verified_features": records["verified_features"],
        "records_missing_features": records["missing_features"],
        "records": records,
    }


def _cloud_evidence(root: Path) -> dict[str, Any]:
    candidates = _find_named(
        root,
        (
            "evidence/cloud-deployment.json",
            "evidence/huawei-cloud-deployment.json",
            "cloud-deployment.json",
        ),
    )
    best: Optional[Path] = None
    obj: Optional[dict[str, Any]] = None
    for path in candidates:
        parsed = _bounded_json(path)
        if parsed is not None:
            best, obj = path, parsed
            break
    provider = str(obj.get("provider") or "") if obj else ""
    status = str(obj.get("status") or "").lower() if obj else ""
    verification = bool(obj.get("verification") is True) if obj else False
    provider_ok = "huawei" in provider.lower() or "华为" in provider
    endpoint = (obj.get("endpoint") or obj.get("url")) if obj else None
    verified = bool(
        best
        and provider_ok
        # Keep this contract aligned with the documented evidence format.
        # Generic workflow statuses such as ``passed`` do not establish that
        # the deployment was actually verified, and accepting them would turn
        # an unrelated receipt into a Huawei Cloud claim.
        and status in {"verified", "deployed"}
        and verification
    )
    endpoint_host = None
    endpoint_scheme = None
    if isinstance(endpoint, str):
        try:
            parsed = urlsplit(endpoint)
            endpoint_host = parsed.hostname
            endpoint_scheme = parsed.scheme.lower()
        except ValueError:
            endpoint_host = None
            endpoint_scheme = None
    endpoint_valid = bool(endpoint_host and endpoint_scheme in {"http", "https"})
    verified = bool(verified and endpoint_valid)
    return {
        "evidence_path": _relative(root, best) if best else None,
        "provider_ok": provider_ok,
        "status": status or None,
        "verification": verification,
        "verified": verified,
        "endpoint_valid": endpoint_valid,
        "endpoint_host": endpoint_host,
        "credentials_read": False,
        "proof_level": "local_self_attestation" if best else "none",
        "endpoint_reachable_checked": False,
    }


def _workload_evidence(root: Path) -> dict[str, Any]:
    candidates = _find_named(
        root,
        (
            "evidence/workload-manifest.json",
            "evidence/real-workload.json",
            "workload-manifest.json",
        ),
    )
    best: Optional[Path] = None
    obj: Optional[dict[str, Any]] = None
    for path in candidates:
        parsed = _bounded_json(path)
        if parsed is not None:
            best, obj = path, parsed
            break
    required = ("data_source", "license", "beneficiary", "source_sha256", "data_sha256")
    present = {
        key: bool(obj and isinstance(obj.get(key), str) and obj.get(key).strip())
        for key in required
    }
    hashes_ok = bool(
        present["source_sha256"]
        and present["data_sha256"]
        and _SHA256_RE.fullmatch(str(obj.get("source_sha256")))
        and _SHA256_RE.fullmatch(str(obj.get("data_sha256")))
    ) if obj else False
    return {
        "evidence_path": _relative(root, best) if best else None,
        "fields": present,
        "hashes_format_valid": hashes_ok,
        "verified": bool(best and all(present.values()) and hashes_ok),
        "proof_level": "local_manifest_declaration" if best else "none",
        "source_or_beneficiary_independently_verified": False,
    }


def _video_evidence(root: Path) -> dict[str, Any]:
    found: list[str] = []
    for base in (root, root / "docs", root / "evidence", root / "submission"):
        try:
            if _has_symlink_component(base) or not base.is_dir() or base.is_symlink():
                continue
            for path in base.iterdir():
                if (
                    _safe_regular_file(path)
                    and path.suffix.lower() in _VIDEO_SUFFIXES
                    and path.stat().st_size > 0
                ):
                    found.append(_relative(root, path))
        except OSError:
            continue
    links: list[str] = []
    for rel in ("README.md", "README.zh-CN.md", "docs/SUBMISSION.md", "docs/COMPETITION-ONE-PAGER.md"):
        text = _bounded_text(root / rel)
        if text:
            links.extend(_VIDEO_URL_RE.findall(text))
            # Most hosted demos use a page URL rather than a media suffix
            # (for example a Bilibili/YouTube video page or a Huawei Cloud
            # preview route).  Only accept links whose surrounding Markdown /
            # HTML label explicitly says video/demo/recording; this avoids
            # turning every ordinary documentation hyperlink into evidence.
            links.extend(match[1] for match in _VIDEO_MARKDOWN_LINK_RE.findall(text))
            links.extend(_VIDEO_HTML_LINK_RE.findall(text))
    metadata_links: list[str] = []
    metadata_paths = _find_named(
        root,
        ("evidence/video.json", "evidence/demo-environment.json", "video.json"),
    )
    for path in metadata_paths:
        value = _bounded_json(path)
        if not value or value.get("accessible") is not True:
            continue
        url = value.get("url") or value.get("video_url") or value.get("endpoint")
        if isinstance(url, str):
            try:
                scheme = urlsplit(url).scheme
            except ValueError:
                scheme = ""
            if scheme in {"http", "https"}:
                metadata_links.append(url)
    links.extend(metadata_links)
    # A plain hosted URL without a media suffix or an explicit label is not
    # enough.  Known video hosts are a useful fallback for terse README lines
    # such as ``视频：https://www.bilibili.com/video/...``; they remain
    # unverified until a local video or ``accessible=true`` metadata receipt
    # is supplied.
    for rel in ("README.md", "README.zh-CN.md", "docs/SUBMISSION.md", "docs/COMPETITION-ONE-PAGER.md"):
        text = _bounded_text(root / rel)
        if not text:
            continue
        for candidate in re.findall(r"https?://[^\s<>\"'()]+", text):
            try:
                host = urlsplit(candidate).hostname or ""
            except ValueError:
                continue
            if any(host.lower() == hint or host.lower().endswith("." + hint) for hint in _VIDEO_HOST_HINTS):
                links.append(candidate)
    # Keep only hosts in the report; a README may contain a private signed
    # URL.  De-duplicate before counting so a URL matched by both its media
    # suffix and its known host is not reported as multiple pieces of evidence.
    links = sorted(set(links))
    metadata_links = sorted(set(metadata_links))
    link_hosts: list[str] = []
    for link in links:
        try:
            host = urlsplit(link).hostname
        except ValueError:
            host = None
        if host:
            link_hosts.append(host)
    verified = bool(found or metadata_links)
    return {
        "files": sorted(set(found)),
        "links": sorted(set(link_hosts)),
        "link_count": len(links),
        "metadata": [_relative(root, path) for path in metadata_paths],
        "present": bool(found or links),
        "verified": verified,
        "unverified_link_count": max(0, len(set(links) - set(metadata_links))),
    }


def _release_artifact_snapshot(root: Path) -> dict[str, Any]:
    artifacts: list[str] = []
    for base in (root / "submission", root / "dist", root / "release"):
        try:
            if _has_symlink_component(base) or not base.is_dir() or base.is_symlink():
                continue
            for path in base.iterdir():
                if _safe_regular_file(path) and path.suffix.lower() in {".zip", ".whl", ".gz", ".tar"}:
                    artifacts.append(_relative(root, path))
        except OSError:
            continue
    sbom = _existing(root, ("SBOM.json", "sbom.json", "docs/SBOM.json", "evidence/SBOM.json"))
    notices = _existing(root, ("THIRD_PARTY_NOTICES.md", "LICENSE"))
    return {
        "artifacts": sorted(set(artifacts)),
        "package_present": bool(artifacts),
        "sbom_present": bool(sbom),
        "license_notice_present": len(notices) >= 2,
    }


def _registration_evidence(root: Path) -> dict[str, Any]:
    """Find a non-empty registration form without guessing its contents."""

    candidates: list[str] = []
    # Registration forms are commonly returned as a stamped PDF/image, but
    # some institutions keep the approved office form as a spreadsheet.  The
    # auditor still checks only a non-empty filename/file; it never opens or
    # interprets the form and therefore cannot claim that the submission
    # system's fields match.
    allowed_suffixes = {
        ".pdf",
        ".doc",
        ".docx",
        ".odt",
        ".rtf",
        ".xls",
        ".xlsx",
        ".ods",
        ".jpg",
        ".jpeg",
        ".png",
    }
    needles = ("registration", "报名", "申报", "entry-form")
    for base in (root, root / "submission", root / "docs"):
        try:
            if _has_symlink_component(base) or not base.is_dir() or base.is_symlink():
                continue
            for path in base.iterdir():
                name = path.name.lower()
                if (
                    _safe_regular_file(path)
                    and path.suffix.lower() in allowed_suffixes
                    and any(needle in name for needle in needles)
                    and path.stat().st_size > 0
                ):
                    candidates.append(_relative(root, path))
        except OSError:
            continue
    return {"files": sorted(set(candidates)), "present": bool(candidates)}


def _runtime_receipt_snapshot(root: Path) -> dict[str, Any]:
    """Recognize only receipts that carry a meaningful runtime proof.

    Merely naming a file ``script-benchmark-report.json`` is not evidence of a
    real Ascend run.  Keep this check offline, but require the producer's
    passed/performance gates and a matched runtime identity before reporting a
    positive authenticity signal.
    """

    candidates = _find_named(
        root, ("script-benchmark-report.json", "bootstrap-receipt.json")
    )
    verified: list[str] = []
    rejected: list[str] = []
    for path in candidates:
        value = _bounded_json(path)
        valid = False
        if value is not None and path.name == "script-benchmark-report.json":
            performance = value.get("performance")
            gates = value.get("gates")
            workers = value.get("workers")
            npu_worker = workers.get("npu") if isinstance(workers, Mapping) else None
            identity = (
                npu_worker.get("runtime_identity_after")
                if isinstance(npu_worker, Mapping)
                else None
            )
            identity_ok = bool(
                isinstance(identity, Mapping)
                and identity.get("status") == "matched"
                and _SHA256_RE.fullmatch(str(identity.get("identity_sha256", "")))
            )
            gates_ok = bool(
                isinstance(gates, Mapping)
                and gates
                and all(value is True for value in gates.values())
            )
            valid = bool(
                value.get("status") == "passed"
                and isinstance(performance, Mapping)
                and performance.get("performance_valid") is True
                and identity_ok
                and gates_ok
            )
        elif value is not None and path.name == "bootstrap-receipt.json":
            verification = value.get("verification")
            asnumpy = verification.get("asnumpy") if isinstance(verification, Mapping) else None
            valid = bool(
                value.get("status") == "ready"
                and value.get("ready") is True
                and isinstance(asnumpy, Mapping)
                and asnumpy.get("status") == "passed"
                and asnumpy.get("wheel_file_identity_verified") is True
            )
        relative = _relative(root, path)
        (verified if valid else rejected).append(relative)
    return {
        "files": [_relative(root, path) for path in candidates],
        "verified_files": verified,
        "rejected_files": rejected,
        "verified": bool(verified),
    }


def _docs_snapshot(root: Path) -> dict[str, Any]:
    required = {
        "readme": ("README.md", "README.zh-CN.md"),
        "architecture": ("docs/architecture.md", "docs/numpilot-architecture.drawio"),
        "installation": ("docs/installation.md", "docs/installation.zh-CN.md"),
        "run_demo": ("run_demo.sh",),
        "dependency_manifest": ("pyproject.toml",),
        "tests": ("tests",),
        "license": ("LICENSE", "THIRD_PARTY_NOTICES.md"),
    }
    result: dict[str, Any] = {}
    for key, paths in required.items():
        result[key] = (
            _existing(root, paths)
            if key != "tests"
            else ["tests"]
            if (root / "tests").is_dir() and not (root / "tests").is_symlink()
            else []
        )
    return result


def assess_competition_readiness(
    repo_root: str | os.PathLike[str] = ".",
    *,
    llm_config: Optional[str | os.PathLike[str]] = None,
    now: Optional[datetime] = None,
) -> CompetitionReadinessReport:
    """Audit a checkout against the contest materials, without side effects.

    ``now`` is injectable for deterministic tests.  The report deliberately
    reports gaps instead of assigning an official score; only contest judges
    can award points.
    """

    root = _validated_repo_root(repo_root)
    # Only ``None`` requests the clock.  Using ``now or ...`` would silently
    # accept a falsy non-datetime (for example ``0``) and produce a report for
    # the current date instead of rejecting the invalid injection.
    current = datetime.now(timezone.utc) if now is None else now
    if not isinstance(current, datetime):
        raise ValueError("检查时间必须是 datetime")
    # The deadline and ``generated_at_utc`` are UTC-bound contracts.  A caller
    # may inject an aware datetime in another zone; derive the calendar date
    # only after conversion so a near-midnight offset cannot shift the
    # remaining-days status by one day.
    if current.tzinfo is None or current.utcoffset() is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    days_remaining = (CONTEST_DEADLINE - current.date()).days
    deadline_status = PASS if days_remaining > 7 else WARN if days_remaining >= 0 else BLOCKED
    deadline = {
        "date": CONTEST_DEADLINE.isoformat(),
        "days_remaining": days_remaining,
        "status": deadline_status,
        "source": "XH-202618 PDF",
    }

    git = _git_snapshot(root)
    # Preserve the PathLike boundary for the snapshot validator.  Converting
    # here with ``str(...)`` could invoke an arbitrary ``__str__`` method and
    # would make path handling differ from ``llm_config_path``/the CLI.
    llm = _llm_snapshot(root, llm_config)
    docs = _docs_snapshot(root)
    manifest = _manifest_hash_snapshot(root)
    codearts = _codearts_evidence(root)
    cloud = _cloud_evidence(root)
    workload = _workload_evidence(root)
    video = _video_evidence(root)
    release = _release_artifact_snapshot(root)
    registration = _registration_evidence(root)

    checks: list[ReadinessCheck] = []

    def add(
        check_id: str,
        dimension: str,
        title: str,
        status: str,
        critical: bool,
        requirement: str,
        evidence: Iterable[str] = (),
        gaps: Iterable[str] = (),
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        checks.append(
            ReadinessCheck(
                check_id,
                dimension,
                title,
                status,
                critical,
                requirement,
                tuple(evidence),
                tuple(gaps),
                dict(details or {}),
            )
        )

    # Completeness (official maximum 50 points).
    source_ok = bool(
        (root / "numpilot").is_dir()
        and not (root / "numpilot").is_symlink()
        and (root / "pyproject.toml").is_file()
        and not (root / "pyproject.toml").is_symlink()
    )
    add(
        "C01",
        "completeness",
        "源码与依赖入口",
        PASS if source_ok else BLOCKED,
        True,
        "提交完整源代码和环境依赖说明",
        [x for x in ("numpilot/", "pyproject.toml") if (root / x).exists()],
        () if source_ok else ("缺少 numpilot/ 或 pyproject.toml",),
    )
    ppt = _existing(root, ("docs/numpilot-zh.pptx", "docs/numpilot-en.pptx"))
    add(
        "C02",
        "completeness",
        "整体介绍 PPT",
        PASS if ppt else BLOCKED,
        True,
        "提交场景、需求、架构、CodeArts过程、难点、测试、效果和扩展性 PPT",
        ppt,
        () if ppt else ("缺少 docs/numpilot-zh.pptx 或等价最终 PPT",),
    )
    # The PDF accepts either a demonstrable video *or* an accessible online
    # environment.  A verified Huawei Cloud deployment receipt therefore
    # closes this material requirement even when the team chooses to demo
    # live instead of attaching a media file.  The cloud contract remains a
    # local self-attestation (no network probe is performed), so the separate
    # C07/A04 checks and the manual checklist still apply.
    demo_environment_present = bool(
        cloud["evidence_path"] and cloud["endpoint_valid"]
    )
    video_status = (
        PASS
        if video["verified"] or cloud["verified"]
        else PARTIAL
        if video["present"] or demo_environment_present
        else BLOCKED
    )
    demo_evidence = list(video["files"] + video["links"])
    if cloud["verified"]:
        demo_evidence.extend(
            item for item in (cloud["evidence_path"], cloud["endpoint_host"])
            if item
        )
    add(
        "C03",
        "completeness",
        "演示视频或环境",
        video_status,
        True,
        "提交可访问视频或演示环境；在线环境须部署在华为云",
        tuple(sorted(set(demo_evidence))),
        (
            ()
            if video["verified"] or cloud["verified"]
            else ("已有链接尚未提供可核验访问元数据；补充视频文件或 accessible=true 的演示回执",)
            if video["present"] or demo_environment_present
            else ("未发现视频文件或明确视频链接；需补交并记录访问方式",)
        ),
        {"video": video, "cloud_environment": cloud},
    )
    docs_ok = all(bool(value) for value in docs.values())
    add(
        "C04",
        "completeness",
        "部署与复现说明",
        PASS if docs_ok else PARTIAL,
        True,
        "依赖、部署步骤、运行指南和代码注释可复现",
        tuple(path for values in docs.values() for path in values),
        () if docs_ok else ("补齐缺失的安装/架构/运行/依赖/测试说明",),
        {"files": docs},
    )
    codearts_records_incomplete = bool(
        codearts["records_present"] and not codearts["records_complete"]
    )
    if not codearts["verified"]:
        codearts_status = BLOCKED
    elif not codearts_records_incomplete:
        # No ``records`` field keeps the original compact contract
        # backwards-compatible.  A complete optional records block is also a
        # full pass.
        codearts_status = PASS
    elif codearts["records_verified_features"]:
        codearts_status = PARTIAL
    else:
        codearts_status = BLOCKED
    codearts_gaps: list[str] = []
    if not codearts["verified"]:
        codearts_gaps.append(
            "仓库 .codeartsdoer 配置不是平台执行证据；在目标 CodeArts 账号完成一次真实任务并导出记录"
        )
    elif codearts_records_incomplete:
        codearts_gaps.append(
            "CodeArts records 不完整或无法校验；补齐每个 feature 的 record_id、record_path、UTC 时间、SHA-256 和 passed 回执"
        )
    add(
        "C05",
        "completeness",
        "CodeArts 实际开发记录",
        codearts_status,
        True,
        "全程使用 CodeArts 代码智能体并保留平台运行记录",
        tuple(codearts["repo_config"] + ([codearts["evidence_path"]] if codearts["evidence_path"] else [])),
        tuple(codearts_gaps),
        {"codearts": codearts},
    )
    feature_status = (
        BLOCKED
        if not codearts["features"]
        else PARTIAL
        if codearts["missing_features"] or codearts_records_incomplete
        else PASS
    )
    if not codearts["verified"] and not codearts["features"]:
        feature_status = BLOCKED
    feature_gaps = ["缺少 %s 记录" % item for item in codearts["missing_features"]]
    if codearts_records_incomplete:
        feature_gaps.append(
            "records 可追溯性为 %s（仅完整且哈希匹配的记录计入）"
            % codearts["records_status"]
        )
    add(
        "C06",
        "completeness",
        "CodeArts 五类核心功能记录",
        feature_status,
        True,
        "代码生成、研发知识问答、单元测试生成、Codebase 索引、规范驱动开发均有记录",
        tuple(codearts["features"]),
        tuple(feature_gaps),
        {
            "required_features": list(_REQUIRED_FEATURES),
            "records": codearts["records"],
        },
    )
    add(
        "C07",
        "completeness",
        "华为云在线部署",
        PASS if cloud["verified"] else BLOCKED,
        True,
        "演示环境应部署在华为云并可访问验证",
        tuple(x for x in (cloud["evidence_path"], cloud["endpoint_host"]) if x),
        () if cloud["verified"] else ("补充华为云部署回执、访问验证时间和脱敏 URL/账号说明",),
        {"cloud": cloud},
    )
    add(
        "C08",
        "completeness",
        "最终交付包与 SBOM",
        (
            PASS
            if release["package_present"]
            and release["sbom_present"]
            and release["license_notice_present"]
            else BLOCKED
            if not release["package_present"] or not release["license_notice_present"]
            else PARTIAL
        ),
        True,
        "代码 ZIP、完整依赖、许可证清单和 SBOM 可核查",
        tuple(release["artifacts"] + (["SBOM"] if release["sbom_present"] else []) + (["LICENSE/THIRD_PARTY_NOTICES"] if release["license_notice_present"] else [])),
        tuple(
            item
            for item, missing in (
                ("缺少最终 ZIP/wheel/sdist", not release["package_present"]),
                ("缺少 SBOM", not release["sbom_present"]),
                ("缺少 LICENSE 或 THIRD_PARTY_NOTICES", not release["license_notice_present"]),
            )
            if missing
        ),
        {"release": release},
    )
    manifest_status = PASS if manifest["complete"] else BLOCKED if not manifest["exists"] else PARTIAL
    add(
        "C09",
        "completeness",
        "证据清单完整性",
        manifest_status,
        True,
        "证据带 SHA-256 清单并能从冻结版本回读",
        [manifest["path"]] if manifest["exists"] else [],
        () if manifest["complete"] else ("重新生成并校验 evidence/MANIFEST.sha256",),
        {"manifest": manifest},
    )
    clean_status = PASS if git.get("is_repository") and git.get("dirty") is False and git.get("commit") else BLOCKED
    add(
        "C10",
        "completeness",
        "冻结提交/tag",
        clean_status,
        True,
        "PPT、视频、源码和证据绑定同一干净不可变版本",
        tuple(x for x in (git.get("commit"), git.get("tag")) if x),
        () if clean_status == PASS else ("在干净 commit/tag 上重跑证据并生成最终提交包",),
        {"git": git},
    )
    # The command surface is a conservative proxy for the documented gap
    # state; no module is imported and no command is executed.
    cli_text = _bounded_text(root / "numpilot" / "__main__.py") or ""
    gap_closed = all(token in cli_text for token in ("integrate", "qualify", "activate", "rescan"))
    add(
        "C11",
        "completeness",
        "缺口恢复闭环",
        PASS if gap_closed else BLOCKED,
        False,
        "候选应完成 integrate→qualify→activate→rescan→最终验证编排",
        ["numpilot/__main__.py"] if cli_text else [],
        () if gap_closed else ("当前命令面未证明完整 integrate/qualify/activate/rescan 自动闭环",),
        {"heuristic_only": True},
    )
    add(
        "C12",
        "completeness",
        "报名表与提交清单",
        PASS if registration["present"] else BLOCKED,
        True,
        "提交报名表并与系统填报信息保持一致",
        tuple(registration["files"]),
        () if registration["present"] else ("将盖章/审核通过的报名表放入最终提交 ZIP",),
        {"registration": registration},
    )

    # Innovation (official maximum 25 points).
    architecture_text = "\n".join(
        filter(None, (_bounded_text(root / "README.zh-CN.md"), _bounded_text(root / "docs/architecture.md")))
    )
    novelty_terms = ("失败关闭", "official-first", "证据", "五态", "CodeArts")
    novelty_count = sum(term.lower() in architecture_text.lower() for term in novelty_terms)
    add(
        "I01",
        "innovation",
        "技术创新叙事",
        PASS if novelty_count >= 3 else PARTIAL,
        False,
        "场景新颖、痛点明确并有可审计 AI 工程方法",
        ("README.zh-CN.md", "docs/architecture.md") if architecture_text else (),
        () if novelty_count >= 3 else ("补充可量化创新点与对比基线",),
        {"recognized_terms": novelty_count},
    )
    ux_terms = ("wizard", "rich", "交互", "verify-script", "benchmark-script")
    ux_count = sum(term.lower() in architecture_text.lower() for term in ux_terms)
    add(
        "I02",
        "innovation",
        "用户体验与交互",
        PASS if ux_count >= 2 and (root / "run_demo.sh").is_file() else PARTIAL,
        False,
        "操作流程清晰、交互流畅并能展示失败原因",
        tuple(x for x in ("run_demo.sh", "numpilot/cli_ui.py") if (root / x).exists()),
        () if ux_count >= 2 and (root / "run_demo.sh").is_file() else ("补充可视化流程、进度、错误恢复和现场操作说明",),
    )
    add(
        "I03",
        "innovation",
        "规模化/商业价值",
        PASS if workload["verified"] and cloud["verified"] else PARTIAL,
        False,
        "明确受益方、落地价值和场景扩展性",
        [workload["evidence_path"]] if workload["evidence_path"] else [],
        () if workload["verified"] else ("选择真实行业负载，量化受益方、节省时间/成本和可扩展边界",),
    )
    add(
        "I04",
        "innovation",
        "华为开放能力融合",
        PASS if codearts["verified"] and cloud["verified"] else WARN,
        False,
        "有效集成 CodeArts/华为云等开放能力可酌情加分",
        tuple(x for x in (codearts["evidence_path"], cloud["evidence_path"]) if x),
        () if codearts["verified"] and cloud["verified"] else ("保存真实 CodeArts/华为云调用与部署记录",),
    )

    # Authenticity (official maximum 25 points).
    add(
        "A01",
        "authenticity",
        "真实数据与可追溯来源",
        PASS if workload["verified"] else BLOCKED,
        True,
        "数据来源真实可追溯并绑定哈希/许可证",
        [workload["evidence_path"]] if workload["evidence_path"] else [],
        () if workload["verified"] else ("提交 data_source、license、data_sha256 和可核查来源",),
        {"workload": workload},
    )
    add(
        "A02",
        "authenticity",
        "真实业务与受益方",
        PASS if workload["verified"] else BLOCKED,
        True,
        "场景来自实际工程需求且有明确使用方/受益方",
        [workload["evidence_path"]] if workload["evidence_path"] else [],
        () if workload["verified"] else ("在脱敏材料中明确实际用户/受益方和验证范围",),
    )
    runtime_receipts = _runtime_receipt_snapshot(root)
    add(
        "A03",
        "authenticity",
        "真实 Ascend 环境验证",
        PASS if runtime_receipts["verified"] else PARTIAL,
        False,
        "方案应在真实环境部署或验证，并记录设备/runtime 身份",
        tuple(runtime_receipts["files"][:8]),
        ()
        if runtime_receipts["verified"]
        else ("保留真实 Ascend/设备/runtime 身份和原始回执（需 passed + identity matched）",),
        {"runtime": runtime_receipts},
    )
    add(
        "A04",
        "authenticity",
        "华为云环境真实性",
        PASS if cloud["verified"] else BLOCKED,
        True,
        "演示/验证环境部署在华为云并可由评委访问",
        [cloud["evidence_path"]] if cloud["evidence_path"] else [],
        () if cloud["verified"] else ("部署到华为云，做外部访问验收并保存脱敏回执",),
    )
    add(
        "A05",
        "authenticity",
        "截止日期风险",
        deadline_status,
        False,
        "在 2026-09-05 前提交完整材料",
        ("XH-202618 PDF",),
        () if deadline_status == PASS else ("立即冻结材料并安排云部署/CodeArts/真实案例验收",),
        {"days_remaining": days_remaining},
    )

    dimensions: dict[str, dict[str, Any]] = {}
    max_points = {"completeness": 50, "innovation": 25, "authenticity": 25}
    labels = {"completeness": "作品完整性", "innovation": "应用创新性", "authenticity": "应用真实性"}
    for key, points in max_points.items():
        subset = [item for item in checks if item.dimension == key]
        passed = sum(item.status == PASS for item in subset)
        dimensions[key] = {
            "label": labels[key],
            "official_max_points": points,
            "status": _status_for_dimension(subset),
            "checks": [item.check_id for item in subset],
            "passed_checks": passed,
            "total_checks": len(subset),
            "coverage_ratio": round(passed / len(subset), 3) if subset else 0.0,
            "official_score": None,
        }

    # ``--strict`` is the final-submission gate: a critical requirement that
    # is only partial (for example, a package without its SBOM or an
    # incomplete CodeArts feature set) must not be mistaken for a releasable
    # pass.  Keep the individual status so the normal report can still
    # distinguish ``blocked`` from ``partial`` and explain the lower-tier
    # submission option.
    blockers = tuple(
        "%s %s: %s" % (
            item.check_id,
            item.title,
            item.gaps[0] if item.gaps else "critical evidence is incomplete",
        )
        for item in checks
        if item.critical and item.status != PASS
    )
    next_actions: list[str] = []
    # Surface every mandatory category before advisory refinements.  One
    # headline action per check prevents CodeArts' five individual feature
    # names (or a packaging sub-list) from crowding real-data, registration,
    # and cloud blockers out of the concise summary; the full gaps remain in
    # each check's JSON block.
    ordered_checks = sorted(
        enumerate(checks),
        key=lambda pair: (not pair[1].critical, pair[0]),
    )
    for _, item in ordered_checks:
        if not item.gaps:
            continue
        gap = item.gaps[0]
        if gap not in next_actions:
            next_actions.append(gap)
        if len(next_actions) >= 12:
            break
    overall = BLOCKED if blockers else PARTIAL if any(item.status in {PARTIAL, WARN} for item in checks) else PASS
    return CompetitionReadinessReport(
        generated_at_utc=current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        repo_root=str(root),
        read_only=True,
        source_execution_performed=False,
        network_accessed=False,
        deadline=deadline,
        git=git,
        llm_config=llm,
        dimensions=dimensions,
        checks=tuple(checks),
        blockers=blockers,
        next_actions=tuple(next_actions),
        overall_status=overall,
    )


def render_readiness(report: CompetitionReadinessReport) -> str:
    """Return a stable human-readable summary (no color or terminal effects)."""

    lines = [
        "NumPilot 比赛就绪度检查（只读；未执行源码、未联网）",
        "=" * 56,
        "总体状态: %s" % report.overall_status.upper(),
        "截止日期: %s（剩余 %s 天）" % (report.deadline["date"], report.deadline["days_remaining"]),
    ]
    for key, dimension in report.dimensions.items():
        lines.append(
            "[%s] %s · 官方满分=%s · 本地证据覆盖=%s/%s"
            % (
                dimension["status"].upper(),
                dimension["label"],
                dimension["official_max_points"],
                dimension["passed_checks"],
                dimension["total_checks"],
            )
        )
    lines.append("")
    lines.append("关键检查:")
    for item in report.checks:
        marker = {PASS: "✓", PARTIAL: "~", WARN: "!", BLOCKED: "✗", INFO: "i"}.get(item.status, "?")
        lines.append("  %s %-4s %-24s %s" % (marker, item.check_id, item.title, item.status))
    if report.blockers:
        lines.append("")
        lines.append("发布阻断:")
        lines.extend("  ✗ " + item for item in report.blockers[:12])
    if report.next_actions:
        lines.append("")
        lines.append("下一步:")
        lines.extend("  → " + item for item in report.next_actions[:8])
    return "\n".join(lines)


def write_readiness_report(report: CompetitionReadinessReport, path: str | os.PathLike[str]) -> Path:
    """Write one redacted JSON report without overwriting an existing file."""

    # The report is an evidence artifact, so a caller must not be able to
    # redirect it through an existing symlink (including a symlinked parent).
    # ``Path.resolve`` alone is insufficient here: it would follow the link
    # and make an outside file look like an ordinary target.  Reject every
    # existing symlink component before creating the parent directory.
    try:
        rendered = os.fspath(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("报告路径无效") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or _contains_control_text(rendered)
        or len(rendered) > 4096
    ):
        raise ValueError("报告路径无效")
    try:
        expanded = os.path.expanduser(rendered)
        # Check the original lexical components before ``abspath`` collapses
        # ``link/..``.  The kernel follows ``link`` first, so normalizing the
        # spelling would otherwise hide a redirect outside the reviewed tree.
        if _has_symlink_component(Path(expanded)):
            raise ValueError("报告路径包含符号链接")
        target = Path(os.path.abspath(expanded))
    except ValueError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ValueError("报告路径无效") from exc
    current = target
    while True:
        try:
            if os.path.lexists(current) and current.is_symlink():
                raise ValueError("报告路径包含符号链接")
        except OSError as exc:
            raise OSError("无法检查报告路径") from exc
        if current == current.parent:
            break
        current = current.parent

    if target.exists() or target.is_symlink():
        raise FileExistsError(str(target))
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # The report itself is private, but its containing directory also
        # controls who can enumerate/replace evidence names.  Do not silently
        # publish into an existing shared directory (for example ``/tmp``).
        # Only the direct parent is checked: a private directory beneath a
        # shared system temp root is the documented and safe invocation.
        if (
            _has_symlink_component(parent)
            or parent.is_symlink()
            or not parent.is_dir()
        ):
            raise ValueError("报告父目录必须是真实目录且不得包含符号链接")
        if os.name == "posix" and (parent.stat().st_mode & 0o077):
            raise ValueError("报告父目录必须具有私有权限（0700）")
    except ValueError:
        raise
    except OSError as exc:
        raise OSError("无法创建或检查报告父目录") from exc
    payload = (json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Publish with a hard link rather than os.replace.  ``replace`` would
    # silently clobber a target created by another process after the initial
    # exists() check; link(2) is atomic and fails if the destination appears.
    fd = -1
    temporary = ""
    try:
        fd, temporary = tempfile.mkstemp(
            prefix="." + target.name + ".", suffix=".tmp", dir=str(parent)
        )
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise
        except OSError as exc:
            # Some filesystems/platforms do not support hard links.  Do not
            # fall back to replace (which violates write-once semantics).
            raise OSError("无法以不可覆盖方式发布报告") from exc
        # Remove the staging name only after the destination link exists.  A
        # previous implementation cleared the variable without unlinking it,
        # leaving one hidden ``.readiness.*.tmp`` file per audit run.
        try:
            os.unlink(temporary)
        except OSError as exc:
            raise OSError("报告已发布但无法清理临时文件") from exc
        temporary = ""
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    except FileExistsError as exc:
        raise FileExistsError(str(target)) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return target


__all__ = [
    "BLOCKED",
    "CONTEST_DEADLINE",
    "CompetitionReadinessReport",
    "INFO",
    "PASS",
    "PARTIAL",
    "ReadinessCheck",
    "SCHEMA_VERSION",
    "WARN",
    "assess_competition_readiness",
    "render_readiness",
    "write_readiness_report",
]
