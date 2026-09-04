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


def _workflow_facts(workflow: Mapping[str, Any] | None) -> dict[str, Any]:
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

    workflow_facts = _workflow_facts(workflow)
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
            "interface_available",
        )
    }
    safe_gap["interface_available"] = boundary.get("interface_available") is True
    return {
        "schema_version": 1,
        "record_type": "numpilot_interactive_migration_report",
        "created_at": _utc_now(),
        "status": _safe_text(status, limit=64),
        "source": dict(source),
        "api_analysis": dict(api_analysis or {}),
        "interpretation": dict(interpretation or {}),
        "gap_recovery": safe_gap,
        "workflow": workflow_facts,
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
