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

"""Truthful aggregate reports for the interactive migration session.

The stage-specific receipts remain authoritative.  This module builds a small
navigation report for humans: it points at those receipts, inventories the
files NumPilot created, and only repeats numeric/performance claims that are
present in validated workflow receipts.  Model prose and private model
configuration are intentionally never copied into this report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any


MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_INVENTORY_FILES = 512
MAX_HASHED_FILE_BYTES = 128 * 1024 * 1024


class InteractiveReportError(RuntimeError):
    """The aggregate report could not be built or published safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _safe_text(value: object, *, limit: int = 512) -> str:
    try:
        rendered = str(value)
    except Exception:
        rendered = "unknown"
    return "".join(
        character if character.isprintable() and character not in "\r\n" else "�"
        for character in rendered[:limit]
    )


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        payload = path.read_bytes()
        if len(payload) > MAX_RECEIPT_BYTES:
            return None
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, Mapping) else None


def _hash_regular_file(path: Path) -> tuple[int, str] | None:
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_HASHED_FILE_BYTES:
            return None
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            return None
        return before.st_size, digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def inventory_generated_files(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Return a bounded inventory of regular, non-symlink generated files."""

    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_root in roots:
        try:
            root = Path(raw_root).absolute()
        except (OSError, TypeError, ValueError):
            continue
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            candidates = sorted(root.rglob("*"), key=lambda item: str(item))
        except OSError:
            continue
        for candidate in candidates:
            if len(files) >= MAX_INVENTORY_FILES:
                return files
            try:
                absolute = str(candidate.absolute())
                if absolute in seen or candidate.is_symlink() or not candidate.is_file():
                    continue
            except OSError:
                continue
            hashed = _hash_regular_file(candidate)
            if hashed is None:
                continue
            size, digest = hashed
            seen.add(absolute)
            files.append({"path": absolute, "bytes": size, "sha256": digest})
    return files



def _ast_api_calls(path: Path | None, package: str) -> list[dict[str, Any]]:
    """Extract framework-owned ``package`` calls without executing user code."""

    if path is None or path.is_symlink() or not path.is_file():
        return []
    try:
        source = path.read_text(encoding="utf-8")
        if len(source.encode("utf-8")) > MAX_RECEIPT_BYTES:
            return []
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError):
        return []
    aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == package:
                    aliases.add(item.asname or package.split(".")[-1])
    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Name) or owner.id not in aliases:
            continue
        calls.append({
            "api": "%s.%s" % (package, node.func.attr),
            "line": int(getattr(node, "lineno", 0) or 0),
            "column": int(getattr(node, "col_offset", 0) or 0),
            "positional_args": len(node.args),
            "keyword_args": sorted(
                item.arg for item in node.keywords if item.arg is not None
            ),
        })
    return sorted(calls, key=lambda item: (item["line"], item["column"], item["api"]))


def _api_rewrite_facts(source_path: Path | None, migrated_path: Path | None) -> list[dict[str, Any]]:
    """Compare source/migrated AST call surfaces and report every API rewrite."""

    source_calls = _ast_api_calls(source_path, "numpy")
    migrated_calls = _ast_api_calls(migrated_path, "asnumpy")
    source_by_name: dict[str, list[dict[str, Any]]] = {}
    migrated_by_name: dict[str, list[dict[str, Any]]] = {}
    for item in source_calls:
        source_by_name.setdefault(item["api"].split(".", 1)[1], []).append(item)
    for item in migrated_calls:
        migrated_by_name.setdefault(item["api"].split(".", 1)[1], []).append(item)
    names = sorted(set(source_by_name) | set(migrated_by_name))
    result: list[dict[str, Any]] = []
    for name in names:
        source_items = source_by_name.get(name, [])
        migrated_items = migrated_by_name.get(name, [])
        result.append({
            "source_api": "numpy.%s" % name,
            "target_api": "asnumpy.%s" % name,
            "source_call_count": len(source_items),
            "target_call_count": len(migrated_items),
            "source_lines": [item["line"] for item in source_items],
            "target_lines": [item["line"] for item in migrated_items],
            "status": (
                "rewritten" if source_items and len(migrated_items) >= len(source_items)
                else "introduced" if migrated_items and not source_items
                else "not_rewritten"
            ),
        })
    return result


def _gap_completion_facts(raw_gap: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Expose explicit gap completions; a no-gap route is always an empty list."""

    if not isinstance(raw_gap, Mapping):
        return []
    status = str(raw_gap.get("status", ""))
    if status in {"", "not_requested", "no_gap"}:
        return []
    candidates = raw_gap.get("completions") or raw_gap.get("gap_specs") or raw_gap.get("gaps")
    if not isinstance(candidates, list):
        candidates = []
    result: list[dict[str, Any]] = []
    for item in candidates[:128]:
        if not isinstance(item, Mapping):
            continue
        result.append({
            key: item.get(key)
            for key in (
                "qualified_api", "asnumpy_api", "status", "route", "interface",
                "provenance", "generated_file", "operator_spec", "report",
                "actual_remote_call", "claim_boundary",
            )
            if key in item
        })
    if result:
        return result
    return [{
        "status": _safe_text(status, limit=64),
        "route": _safe_text(raw_gap.get("route", "unknown"), limit=128),
        "actual_remote_call": raw_gap.get("actual_remote_call") is True,
    }]


def _session_elapsed_seconds(session_root: Path) -> float | None:
    """Return explicit interactive session wall time when start/finish receipts exist."""

    started = _read_json(session_root / "session-start.json")
    finished = _read_json(session_root / "session-finish.json")
    if not isinstance(started, Mapping) or not isinstance(finished, Mapping):
        return None
    try:
        value = float(finished.get("epoch_seconds")) - float(started.get("epoch_seconds"))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value < 7 * 24 * 3600 else None


def _workflow_timing_facts(workflow_report: Mapping[str, Any] | None, session_root: Path) -> dict[str, Any]:
    """Normalize migration, benchmark and end-to-end timings with explicit nulls."""

    stage_elapsed = workflow_report.get("timing", {}).get("stage_elapsed_seconds", {}) if isinstance(workflow_report, Mapping) else {}
    performance = workflow_report.get("performance") if isinstance(workflow_report, Mapping) else None
    if not isinstance(performance, Mapping):
        performance = {}
    valid = performance.get("performance_valid") is True
    return {
        "migration_elapsed_seconds": stage_elapsed.get("02-migrate") if isinstance(stage_elapsed, Mapping) else None,
        "numpy_ms": performance.get("cpu_median_ms") if valid else None,
        "asnumpy_ms": performance.get("npu_median_ms") if valid else None,
        "e2e_seconds": _session_elapsed_seconds(session_root),
        "speedup": performance.get("median_speedup") if valid else None,
        "performance_valid": valid,
        "status": "passed" if valid else "not_run",
        "scope": "steady_device_resident_marker" if valid else None,
    }

def _workflow_facts(workflow: Mapping[str, Any] | None, source_path: Path | None = None) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "status": "not_run",
        "report": None,
        "migrated_code": None,
        "stages": [],
        "numeric_verification": {"status": "not_run", "comparisons": {}},
        "performance": {
            "status": "not_run",
            "performance_valid": False,
            "speedup": None,
            "measured_scale": None,
            "advantage_threshold": None,
            "threshold_status": "not_measured",
        },
        "timing": {
            "migration_elapsed_seconds": None,
            "numpy_ms": None,
            "asnumpy_ms": None,
            "e2e_seconds": None,
            "speedup": None,
            "performance_valid": False,
            "status": "not_run",
            "scope": None,
        },
        "api_rewrites": [],
    }
    if not isinstance(workflow, Mapping):
        return facts
    report_value = workflow.get("workflow_report")
    report = Path(report_value) if isinstance(report_value, str) and report_value else None
    run_root = report.parent if report is not None else None
    facts["status"] = _safe_text(workflow.get("status", "unknown"), limit=64)
    facts["report"] = str(report) if report is not None else None
    facts["stages"] = [
        {
            "name": _safe_text(item.get("name", "unknown"), limit=128),
            "status": _safe_text(item.get("status", "unknown"), limit=64),
            "elapsed_seconds": item.get("elapsed_seconds"),
        }
        for item in workflow.get("stages", ())
        if isinstance(item, Mapping)
    ]
    if run_root is None:
        return facts
    migrated = run_root / "migration" / "migrated.py"
    facts["api_rewrites"] = _api_rewrite_facts(source_path, migrated)
    workflow_report_value = _read_json(report) if report is not None else None
    facts["timing"] = _workflow_timing_facts(workflow_report_value, run_root.parent)
    if migrated.is_file() and not migrated.is_symlink():
        facts["migrated_code"] = str(migrated)

    verification = _read_json(run_root / "script-verification-report.json")
    if verification is not None:
        comparisons = verification.get("comparisons")
        facts["numeric_verification"] = {
            "status": _safe_text(verification.get("status", "unknown"), limit=64),
            "comparisons": dict(comparisons) if isinstance(comparisons, Mapping) else {},
        }

    benchmark = _read_json(run_root / "script-benchmark-report.json")
    performance = benchmark.get("performance") if benchmark is not None else None
    if isinstance(performance, Mapping):
        valid = performance.get("performance_valid") is True
        measured_scale = None
        workers = benchmark.get("workers") if benchmark is not None else None
        npu_worker = workers.get("npu") if isinstance(workers, Mapping) else None
        outputs = npu_worker.get("outputs") if isinstance(npu_worker, Mapping) else None
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], Mapping):
            measured_scale = {
                key: outputs[0].get(key)
                for key in ("shape", "dtype", "elements", "data_bytes")
            }
        facts["performance"] = {
            "status": "passed" if valid else "not_valid",
            "performance_valid": valid,
            "cpu_median_ms": performance.get("cpu_median_ms") if valid else None,
            "npu_median_ms": performance.get("npu_median_ms") if valid else None,
            "speedup": performance.get("median_speedup") if valid else None,
            "conservative_speedup": (
                performance.get("conservative_cpu_p10_over_npu_p90") if valid else None
            ),
            "measured_scale": measured_scale,
            # One benchmark point cannot establish a crossover threshold.
            "advantage_threshold": None,
            "threshold_status": "not_measured",
            "scope": "steady_device_resident_marker" if valid else None,
        }
    return facts


def build_session_report(
    *,
    session_root: Path,
    source: Mapping[str, Any],
    status: str,
    api_analysis: Mapping[str, Any] | None = None,
    interpretation: Mapping[str, Any] | None = None,
    gap_result: Mapping[str, Any] | None = None,
    workflow: Mapping[str, Any] | None = None,
    extra_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build a machine-readable summary without upgrading any stage claim."""

    source_path = Path(source.get("path")) if isinstance(source, Mapping) and isinstance(source.get("path"), str) else None
    workflow_facts = _workflow_facts(workflow, source_path)
    roots = [Path(session_root)]
    workflow_report = workflow_facts.get("report")
    if isinstance(workflow_report, str):
        roots.append(Path(workflow_report).parent)
    roots.extend(Path(item) for item in extra_roots)
    raw_gap = gap_result if isinstance(gap_result, Mapping) else {}
    safe_gap = {
        key: raw_gap.get(key)
        for key in (
            "status",
            "route",
            "interface",
            "provenance",
            "scope",
            "actual_remote_call",
            "report",
            "generated_file",
            "operator_spec",
            "relation_report",
            "extension_manifest",
            "qualification_report",
            "lowered_source",
            "lowering_report",
            "lowering_verification_report",
            "upstream_modified",
            "cannbot_draft_activated",
        )
        if key in raw_gap
    }
    # Only a positive, separately evidenced integration boundary may expose an
    # interface claim.  A model-generation process returning rc=0 is not one.
    boundary = raw_gap.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        boundary = {}
    runtime_matched = (
        isinstance(api_analysis, Mapping)
        and api_analysis.get("runtime_identity_status") == "matched"
    )
    interface_available = boundary.get("interface_available") is True
    # A no-gap direct migration does not produce a gap claim boundary. When
    # the analyzer has matched the AsNumpy runtime, expose direct API
    # availability without claiming any generated operator.
    if not interface_available and raw_gap.get("status") == "no_gap" and runtime_matched:
        interface_available = True
    safe_gap["claim_boundary"] = {
        key: boundary.get(key) is True
        for key in (
            "generated",
            "compiled",
            "unit_tested",
            "numerically_verified",
            "integration_verified",
            "qualified",
            "activated",
            "rescanned",
        )
    }
    safe_gap["claim_boundary"]["interface_available"] = interface_available
    safe_gap["interface_available"] = interface_available
    return {
        "schema_version": 1,
        "record_type": "numpilot_interactive_migration_report",
        "created_at": _utc_now(),
        "status": _safe_text(status, limit=64),
        "source": dict(source),
        "api_analysis": dict(api_analysis or {}),
        "interpretation": dict(interpretation or {}),
        "gap_recovery": safe_gap,
        "gap_completions": _gap_completion_facts(raw_gap),
        "workflow": workflow_facts,
        "migration": {
            "source_path": source.get("path") if isinstance(source, Mapping) else None,
            "migrated_path": workflow_facts.get("migrated_code"),
            "api_rewrites": workflow_facts.get("api_rewrites", []),
            "gap_completions": _gap_completion_facts(raw_gap),
            "timing": workflow_facts.get("timing", {}),
        },
        "generated_files": inventory_generated_files(roots),
        "claims_boundary": {
            "model_interpretation_is_advisory": True,
            "operator_draft_is_not_activated": raw_gap.get("cannbot_draft_activated") is not True,
            "numeric_claim_requires_passed_verification": True,
            "speedup_requires_performance_valid": True,
            "scale_threshold_requires_multi_scale_measurement": True,
        },
    }


def render_session_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact human report used at the end of every session."""

    source = report.get("source") if isinstance(report.get("source"), Mapping) else {}
    gap = report.get("gap_recovery") if isinstance(report.get("gap_recovery"), Mapping) else {}
    workflow = report.get("workflow") if isinstance(report.get("workflow"), Mapping) else {}
    numeric = workflow.get("numeric_verification") if isinstance(workflow.get("numeric_verification"), Mapping) else {}
    performance = workflow.get("performance") if isinstance(workflow.get("performance"), Mapping) else {}
    lines = [
        "# NumPilot 交互迁移报告",
        "",
        "- 会话状态：`%s`" % _safe_text(report.get("status", "unknown"), limit=64),
        "- 原始代码：`%s`" % _safe_text(source.get("path", "unknown")),
        "- 原始代码 SHA-256：`%s`" % _safe_text(source.get("sha256", "unknown"), limit=80),
        "- 算子补全状态：`%s`" % _safe_text(gap.get("status", "not_run"), limit=64),
        "- AsNumpy 接口可用：`%s`" % ("true" if gap.get("interface_available") is True else "false"),
        "- 整体迁移状态：`%s`" % _safe_text(workflow.get("status", "not_run"), limit=64),
        "- 迁移后代码：`%s`" % _safe_text(workflow.get("migrated_code") or "未生成"),
        "- 数值精度验证：`%s`" % _safe_text(numeric.get("status", "not_run"), limit=64),
    ]
    migration = report.get("migration") if isinstance(report.get("migration"), Mapping) else {}
    timing = migration.get("timing") if isinstance(migration.get("timing"), Mapping) else {}
    lines.extend(("", "## 迁移摘要", "",
        "- 迁移阶段耗时：`%s s`" % _safe_text(timing.get("migration_elapsed_seconds"), limit=64),
        "- 全流程端到端耗时：`%s s`" % _safe_text(timing.get("e2e_seconds"), limit=64),
        "- NumPy CPU：`%s ms`" % _safe_text(timing.get("numpy_ms"), limit=64),
        "- AsNumpy/NPU：`%s ms`" % _safe_text(timing.get("asnumpy_ms"), limit=64),
        "- 加速比：`%s x`" % _safe_text(timing.get("speedup"), limit=64),
        "- 性能状态：`%s`" % _safe_text(timing.get("status", "not_run"), limit=64),
        "", "### API 改写", ""))
    rewrites = migration.get("api_rewrites")
    if isinstance(rewrites, list) and rewrites:
        for item in rewrites:
            if isinstance(item, Mapping):
                lines.append("- `%s → %s`：source=%s target=%s status=%s" % (
                    _safe_text(item.get("source_api")), _safe_text(item.get("target_api")),
                    item.get("source_call_count", 0), item.get("target_call_count", 0),
                    _safe_text(item.get("status", "unknown"), limit=64)))
    else:
        lines.append("- 无 API 调用改写记录")
    lines.extend(("", "### 缺口补全", "",))
    completions = report.get("gap_completions")
    if isinstance(completions, list) and completions:
        for item in completions:
            lines.append("- `%s`" % _safe_text(item))
    else:
        lines.append("- `[]`（本次没有缺口补全）")
    if performance.get("performance_valid") is True:
        lines.extend((
            "- 当前实测加速比：`%s x`" % _safe_text(performance.get("speedup"), limit=64),
            "- 当前实测规模：`%s`" % _safe_text(performance.get("measured_scale")),
            "- 显著优势起始规模：`未测量（需要多尺度 sweep，不能由单点外推）`",
        ))
    else:
        lines.extend((
            "- 性能验证：`not_run / performance_valid=false`",
            "- 显著优势起始规模：`未测量`",
        ))
    lines.extend(("", "## 本次生成的文件", ""))
    files = report.get("generated_files")
    if isinstance(files, list) and files:
        for item in files:
            if isinstance(item, Mapping):
                lines.append(
                    "- `%s`（%s bytes，sha256 `%s`）"
                    % (
                        _safe_text(item.get("path", "unknown"), limit=4096),
                        item.get("bytes", "?"),
                        _safe_text(item.get("sha256", "unknown"), limit=80),
                    )
                )
    else:
        lines.append("- 无")
    lines.extend((
        "",
        "> 性能结论只适用于回执中的当前数据规模与 steady device-resident 计时口径；",
        "> 未经过构建、数值、激活和重扫的 CANNBot 草稿不等于已接入接口。",
        "",
    ))
    return "\n".join(lines)


def _atomic_private_write(path: Path, payload: bytes) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise InteractiveReportError("report target must be a fresh absolute path")
    parent = path.parent
    try:
        if parent.is_symlink() or not parent.is_dir():
            raise InteractiveReportError("report parent is unavailable")
        if os.name == "posix" and stat.S_IMODE(parent.stat().st_mode) & 0o077:
            raise InteractiveReportError("report parent must be private")
    except OSError as exc:
        raise InteractiveReportError("report parent cannot be inspected") from exc
    if not payload or len(payload) > MAX_REPORT_BYTES:
        raise InteractiveReportError("report payload is empty or oversized")
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + path.name + "-", suffix=".tmp", dir=str(parent)
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = ""
    except (FileExistsError, OSError) as exc:
        raise InteractiveReportError("report publication failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return path


def write_session_reports(session_root: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    """Publish JSON and Markdown final reports as write-once private files."""

    root = Path(session_root).absolute()
    json_payload = (
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    json_path = _atomic_private_write(root / "migration-report.json", json_payload)
    markdown_path = _atomic_private_write(
        root / "migration-report.md", render_session_markdown(report).encode("utf-8")
    )
    return json_path, markdown_path


__all__ = [
    "InteractiveReportError",
    "build_session_report",
    "inventory_generated_files",
    "render_session_markdown",
    "write_session_reports",
]
