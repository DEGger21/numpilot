"""Receipt-only ingestion for the official-first recovery prefix.

This module joins three receipts already produced by NumPilot:

* a schema-2 ordinary-script migration report that stopped at a gap;
* its canonical ``gap-request.json``;
* an ``official-resolution.json`` produced by the local resolver.

The adapter validates and cross-binds those receipts before publishing the
``CREATED -> ANALYZED -> GAP_PREPARED -> RESOLVED`` prefix in a
:class:`~numpilot.gaps.recovery.RecoveryStore`.  It does not refresh the
installed CANN inventory, invoke a model, build or import AsNumPy, execute user
code, or activate a capability.  The resulting state therefore records only
historical, content-bound producer receipts and keeps every side-effect truth
value false.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from numpilot.gaps.official_resolution import (
    OfficialOperatorResolutionReport,
    OfficialResolutionError,
    gap_spec_sha256,
    load_official_resolution_report,
)
from numpilot.gaps.pipeline import (
    GapCompletionError,
    GapPreparation,
    ScriptGapRequest,
    load_script_gap_request,
    prepare_gap_completion,
)
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    RecoveryBinding,
    RecoveryError,
    RecoveryEvent,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryPolicy,
    RecoveryPolicyProfile,
    RecoveryRun,
    RecoveryRuntime,
    RecoverySource,
    RecoveryStatus,
    RecoveryStore,
    RecoveryTarget,
    SideEffectTruth,
    StageReceiptKind,
    StageReceiptScope,
    StageReceiptVerdict,
    create_stage_receipt,
    load_recovery_status,
    recovery_binding_sha256,
    recovery_policy_for_profile,
)


_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_MIGRATION_REPORT_BYTES = 8 * 1024 * 1024
_MAX_GAP_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_OFFICIAL_REPORT_BYTES = 32 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_MIGRATION_REPORT_FIELDS = frozenset(
    {
        "actual_remote_call",
        "analysis",
        "ast_audit",
        "execution_performed",
        "fallback_accepted",
        "generated_by",
        "mode",
        "numerical_verification",
        "performance_valid",
        "published_files",
        "qualified_analysis",
        "schema_version",
        "source_sha256",
        "status",
        "strict_npu",
    }
)
_ANALYSIS_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "backend",
        "backend_version",
        "audit_id",
        "counts",
        "decisions",
        "scan",
    }
)
_SCAN_FIELDS = frozenset(
    {
        "schema_version",
        "ok",
        "source_path",
        "source_sha256",
        "issues",
        "uses",
        "rewrite_count",
        "numpy_pi_rewrite_count",
        "synthetic_divide_count",
        "terminal_item_count",
        "output_boundary_count",
    }
)
_QUALIFIED_ANALYSIS_FIELDS = frozenset(
    {
        "schema_version",
        "analysis_mode",
        "actual_remote_call",
        "execution_performed",
        "performance_valid",
        "source_path",
        "source_sha256",
        "runtime_identity_sha256",
        "runtime_identity_status",
        "decision",
        "next_action",
        "speedup",
        "imports",
        "issues",
        "calls",
        "required_rewrites",
    }
)
_GAP_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_type",
        "status",
        "source_path",
        "source_sha256",
        "backend",
        "backend_version",
        "audit_id",
        "target_soc",
        "cann_version",
        "gaps",
        "gap_specs",
        "generated_code",
        "qualification_performed",
        "activation_performed",
    }
)
_RESOLVED_STATUSES = frozenset(
    {
        "official_candidate_found",
        "new_kernel_generation_allowed",
        "mixed_routes",
    }
)
_NON_BLOCKED_ROUTES = frozenset(
    {
        "installed_cann_adapter_candidate",
        "official_source_candidate",
        "new_kernel_generation_allowed",
    }
)
_PHASE_INDEX = {
    phase: index
    for index, phase in enumerate(
        (
            RecoveryPhase.CREATED,
            RecoveryPhase.ANALYZED,
            RecoveryPhase.GAP_PREPARED,
            RecoveryPhase.RESOLVED,
            RecoveryPhase.CANDIDATE_READY,
            RecoveryPhase.ADAPTER_CANDIDATE,
            RecoveryPhase.INTEGRATED_CANDIDATE,
            RecoveryPhase.QUALIFIED,
            RecoveryPhase.ACTIVATED,
            RecoveryPhase.RESCANNED,
            RecoveryPhase.FINAL_MIGRATED,
            RecoveryPhase.VERIFIED,
        )
    )
}


class RecoveryIngestError(RecoveryError):
    """A producer receipt or existing recovery prefix failed closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OfficialRecoveryIngestResult:
    """Structured result returned to a future CLI adapter."""

    recovery: RecoveryStatus
    created: bool
    appended_revisions: int
    input_receipts: Tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.recovery, RecoveryStatus):
            raise RecoveryIngestError(
                "invalid_result", "ingest result requires a recovery status"
            )
        if not isinstance(self.created, bool):
            raise RecoveryIngestError(
                "invalid_result", "ingest result created flag is invalid"
            )
        if (
            not isinstance(self.appended_revisions, int)
            or isinstance(self.appended_revisions, bool)
            or self.appended_revisions < 0
        ):
            raise RecoveryIngestError(
                "invalid_result", "ingest revision count is invalid"
            )
        refs = tuple(self.input_receipts)
        if len(refs) != 3 or any(not isinstance(item, EvidenceRef) for item in refs):
            raise RecoveryIngestError(
                "invalid_result", "ingest result requires three input receipts"
            )
        if len({item.name for item in refs}) != len(refs):
            raise RecoveryIngestError(
                "invalid_result", "ingest input receipt names are duplicated"
            )
        object.__setattr__(
            self, "input_receipts", tuple(sorted(refs, key=lambda item: item.name))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic, JSON-compatible summary."""

        return {
            "schema_version": 1,
            "record_type": "numpilot_official_recovery_ingest",
            "receipt_ingestion_only": True,
            "current_environment_recheck": False,
            "created": self.created,
            "appended_revisions": self.appended_revisions,
            "input_receipts": [item.to_dict() for item in self.input_receipts],
            "side_effects_performed": SideEffectTruth().to_dict(),
            "recovery": self.recovery.to_dict(),
        }


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _ValidatedInputs:
    source: _Snapshot
    migration: _Snapshot
    gap_request: _Snapshot
    official_resolution: _Snapshot
    migration_value: Mapping[str, Any]
    gap_request_value: Mapping[str, Any]
    selection: ScriptGapRequest
    preparation: GapPreparation
    official_report: OfficialOperatorResolutionReport


def _bounded_path(value: os.PathLike[str] | str, label: str) -> Path:
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise RecoveryIngestError("invalid_path", "invalid %s path" % label) from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4096
        or "\x00" in rendered
    ):
        raise RecoveryIngestError("invalid_path", "invalid %s path" % label)
    return Path(os.path.abspath(os.path.expanduser(rendered)))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path
    while True:
        try:
            if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
                raise RecoveryIngestError(
                    "unsafe_input", "%s path contains a symlink" % label
                )
        except OSError as exc:
            raise RecoveryIngestError(
                "input_read_failed", "cannot inspect %s path" % label
            ) from exc
        if current == current.parent:
            return
        current = current.parent


def _read_snapshot(path: Path, maximum: int, label: str) -> _Snapshot:
    _reject_symlink_components(path, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryIngestError(
                "unsafe_input", "%s must be a regular file" % label
            )
        if before.st_size > maximum:
            raise RecoveryIngestError(
                "input_too_large", "%s exceeds the size limit" % label
            )
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum:
            raise RecoveryIngestError(
                "input_too_large", "%s exceeds the size limit" % label
            )
        after = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RecoveryIngestError(
                "input_drift", "%s changed while it was read" % label
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        verification = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            verification.append(chunk)
            remaining -= len(chunk)
        if b"".join(verification) != payload:
            raise RecoveryIngestError(
                "input_drift", "%s changed while it was read" % label
            )
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise RecoveryIngestError(
                "input_drift", "%s changed while it was read" % label
            )
        _reject_symlink_components(path, label)
        return _Snapshot(
            path=path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
        )
    except RecoveryIngestError:
        raise
    except OSError as exc:
        raise RecoveryIngestError(
            "input_read_failed", "cannot read %s" % label
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _bounded_integer(raw: str) -> int:
    if len(raw) > 32:
        raise ValueError("JSON integer exceeds the digit limit")
    return int(raw)


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("JSON number is not finite")
    return value


def _reject_constant(raw: str) -> None:
    raise ValueError("non-standard JSON constant: %s" % raw)


def _parse_object(snapshot: _Snapshot, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_int=_bounded_integer,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise RecoveryIngestError(
            "invalid_json", "%s is not bounded strict UTF-8 JSON" % label
        ) from exc
    if not isinstance(value, dict):
        raise RecoveryIngestError("invalid_json", "%s must be an object" % label)
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(_SHA256.fullmatch(value))
        and value != "0" * 64
    )


def _bounded_text(value: object, label: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or "\x00" in value
        or value != value.strip()
    ):
        raise RecoveryIngestError("invalid_receipt", "invalid %s" % label)
    return value


def _same_path(left: str, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(str(right))


def _validate_migration_report(
    value: Mapping[str, Any], source: _Snapshot
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if frozenset(value) != _MIGRATION_REPORT_FIELDS:
        raise RecoveryIngestError(
            "invalid_migration_report",
            "gap migration report fields are incomplete or unknown",
        )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 2
        or value.get("mode") != "script"
        or value.get("status") != "gap_generation_required"
        or value.get("actual_remote_call") is not False
        or value.get("generated_by") is not None
        or value.get("fallback_accepted") is not False
        or value.get("ast_audit") != "not_run"
        or value.get("execution_performed") is not False
        or value.get("numerical_verification") != "not_run"
        or value.get("performance_valid") is not False
        or value.get("strict_npu") is not False
        or value.get("published_files") != ["gap-request.json"]
        or value.get("source_sha256") != source.sha256
    ):
        raise RecoveryIngestError(
            "invalid_migration_report",
            "migration report is not an inert schema-2 gap hand-off",
        )
    analysis = value.get("analysis")
    qualified = value.get("qualified_analysis")
    if not isinstance(analysis, Mapping) or not isinstance(qualified, Mapping):
        raise RecoveryIngestError(
            "invalid_migration_report", "migration analysis sections are invalid"
        )
    scan = analysis.get("scan")
    if (
        frozenset(analysis) != _ANALYSIS_FIELDS
        or frozenset(qualified) != _QUALIFIED_ANALYSIS_FIELDS
        or not isinstance(scan, Mapping)
        or frozenset(scan) != _SCAN_FIELDS
    ):
        raise RecoveryIngestError(
            "invalid_migration_report",
            "migration analysis fields are incomplete or unknown",
        )
    if (
        type(analysis.get("schema_version")) is not int
        or analysis.get("schema_version") != 1
        or analysis.get("mode") != "script"
        or analysis.get("backend") != "npu"
        or not isinstance(analysis.get("backend_version"), (str, type(None)))
        or not _bounded_text(analysis.get("audit_id"), "analysis audit id", 256)
        or type(scan.get("schema_version")) is not int
        or scan.get("schema_version") != 1
        or scan.get("ok") is not True
        or scan.get("source_sha256") != source.sha256
        or scan.get("issues") != []
        or not isinstance(scan.get("uses"), list)
        or not isinstance(analysis.get("counts"), Mapping)
        or not isinstance(analysis.get("decisions"), list)
        or not _same_path(
            _bounded_text(scan.get("source_path"), "scan source path"), source.path
        )
    ):
        raise RecoveryIngestError(
            "invalid_migration_report", "migration analysis binding is invalid"
        )
    backend_version = analysis.get("backend_version")
    if backend_version is not None:
        _bounded_text(backend_version, "analysis backend version", 512)
    if (
        type(qualified.get("schema_version")) is not int
        or qualified.get("schema_version") != 1
        or qualified.get("analysis_mode") != "STATIC_AST_ONLY"
        or qualified.get("actual_remote_call") is not False
        or qualified.get("execution_performed") is not False
        or qualified.get("performance_valid") is not False
        or qualified.get("runtime_identity_status") != "matched"
        or not _is_sha256(qualified.get("runtime_identity_sha256"))
        or qualified.get("speedup") is not None
        or not isinstance(qualified.get("imports"), list)
        or not isinstance(qualified.get("issues"), list)
        or not isinstance(qualified.get("calls"), list)
        or not isinstance(qualified.get("required_rewrites"), list)
        or qualified.get("source_sha256") != source.sha256
        or not _same_path(
            _bounded_text(
                qualified.get("source_path"), "qualified analysis source path"
            ),
            source.path,
        )
    ):
        raise RecoveryIngestError(
            "invalid_migration_report",
            "qualified analysis runtime/source binding is invalid",
        )
    return analysis, scan, qualified


def _validate_gap_request_value(
    value: Mapping[str, Any], source: _Snapshot
) -> None:
    if frozenset(value) != _GAP_REQUEST_FIELDS:
        raise RecoveryIngestError(
            "invalid_gap_request", "gap request fields are incomplete or unknown"
        )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("request_type") != "script_api_gap"
        or value.get("status") != "gap_generation_required"
        or value.get("backend") != "npu"
        or value.get("source_sha256") != source.sha256
        or value.get("generated_code") is not False
        or value.get("qualification_performed") is not False
        or value.get("activation_performed") is not False
        or not _same_path(
            _bounded_text(value.get("source_path"), "gap request source path"),
            source.path,
        )
        or not isinstance(value.get("gaps"), list)
        or not isinstance(value.get("gap_specs"), list)
        or not value.get("gap_specs")
    ):
        raise RecoveryIngestError(
            "invalid_gap_request", "gap request is not an inert script gap hand-off"
        )
    _bounded_text(value.get("audit_id"), "gap request audit id", 256)
    _bounded_text(value.get("target_soc"), "gap request target SoC", 128)
    _bounded_text(value.get("cann_version"), "gap request CANN version", 128)
    backend_version = value.get("backend_version")
    if backend_version is not None:
        _bounded_text(backend_version, "gap request backend version", 512)


def _validate_cross_bindings(
    migration: Mapping[str, Any],
    gap_request: Mapping[str, Any],
    source: _Snapshot,
) -> None:
    analysis, _scan, qualified = _validate_migration_report(migration, source)
    _validate_gap_request_value(gap_request, source)
    if (
        analysis.get("backend") != gap_request.get("backend")
        or analysis.get("backend_version") != gap_request.get("backend_version")
        or analysis.get("audit_id") != gap_request.get("audit_id")
        or qualified.get("source_sha256") != gap_request.get("source_sha256")
    ):
        raise RecoveryIngestError(
            "receipt_binding_mismatch",
            "migration report and gap request runtime bindings differ",
        )
    decisions = analysis.get("decisions")
    raw_gaps = gap_request.get("gaps")
    if not isinstance(decisions, list) or not isinstance(raw_gaps, list):
        raise RecoveryIngestError(
            "receipt_binding_mismatch",
            "migration report and gap request decisions are invalid",
        )
    by_api = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise RecoveryIngestError(
                "receipt_binding_mismatch", "migration decision is invalid"
            )
        qualified_api = decision.get("qualified_api")
        if not isinstance(qualified_api, str) or qualified_api in by_api:
            raise RecoveryIngestError(
                "receipt_binding_mismatch",
                "migration decisions have invalid identities",
            )
        by_api[qualified_api] = decision
    for gap in raw_gaps:
        if not isinstance(gap, Mapping):
            raise RecoveryIngestError(
                "receipt_binding_mismatch", "gap request decision is invalid"
            )
        qualified_api = gap.get("qualified_api")
        decision = by_api.get(qualified_api)
        if (
            decision is None
            or gap.get("current_available") is not decision.get("available")
            or gap.get("current_status") != decision.get("status")
            or gap.get("current_provenance") != decision.get("provenance")
            or gap.get("generatable") is not decision.get("generatable")
            or gap.get("reason") != decision.get("evidence")
            or gap.get("callsites") != decision.get("callsites")
        ):
            raise RecoveryIngestError(
                "receipt_binding_mismatch",
                "gap request decision differs from migration analysis",
            )


def _load_validated_inputs(
    source_path: os.PathLike[str] | str,
    migration_report_path: os.PathLike[str] | str,
    gap_request_path: os.PathLike[str] | str,
    official_resolution_path: os.PathLike[str] | str,
) -> _ValidatedInputs:
    source_file = _bounded_path(source_path, "source")
    migration_file = _bounded_path(migration_report_path, "migration report")
    gap_file = _bounded_path(gap_request_path, "gap request")
    official_file = _bounded_path(
        official_resolution_path, "official resolution report"
    )
    source = _read_snapshot(source_file, _MAX_SOURCE_BYTES, "source")
    migration = _read_snapshot(
        migration_file, _MAX_MIGRATION_REPORT_BYTES, "migration report"
    )
    gap_request = _read_snapshot(gap_file, _MAX_GAP_REQUEST_BYTES, "gap request")
    official_resolution = _read_snapshot(
        official_file, _MAX_OFFICIAL_REPORT_BYTES, "official resolution report"
    )
    identities = {
        (item.device, item.inode)
        for item in (source, migration, gap_request, official_resolution)
    }
    if len(identities) != 4:
        raise RecoveryIngestError(
            "input_alias", "recovery producer inputs must be distinct files"
        )
    if migration.path.parent != gap_request.path.parent or gap_request.path.name != (
        "gap-request.json"
    ):
        raise RecoveryIngestError(
            "receipt_binding_mismatch",
            "gap request is not the migration report's sibling publication",
        )
    migration_value = _parse_object(migration, "migration report")
    gap_value = _parse_object(gap_request, "gap request")
    official_value = _parse_object(official_resolution, "official resolution report")
    _validate_cross_bindings(migration_value, gap_value, source)
    target_soc = _bounded_text(gap_value.get("target_soc"), "target SoC", 128)
    cann_version = _bounded_text(
        gap_value.get("cann_version"), "CANN version", 128
    )
    try:
        selection = load_script_gap_request(
            str(gap_request.path),
            str(source.path),
            target_soc=target_soc,
            cann_version=cann_version,
        )
        if selection.blocked:
            raise RecoveryIngestError(
                "unsupported_gap_bundle",
                "receipt-only recovery requires every requested gap to be generatable",
            )
        preparation = prepare_gap_completion(
            str(source.path),
            target_soc=target_soc,
            cann_version=cann_version,
            requested_specs=selection.specs,
            requested_blocked=selection.blocked,
        )
    except RecoveryIngestError:
        raise
    except GapCompletionError as exc:
        raise RecoveryIngestError(
            "invalid_gap_request", "gap request failed canonical reconstruction"
        ) from exc
    try:
        official_report = load_official_resolution_report(str(official_resolution.path))
        official_report.validate_preparation(preparation)
    except OfficialResolutionError as exc:
        raise RecoveryIngestError(
            "invalid_official_resolution",
            "official resolution failed receipt/preparation validation",
        ) from exc
    if official_value != official_report.to_dict():
        raise RecoveryIngestError(
            "invalid_official_resolution",
            "official resolution JSON differs from its canonical receipt",
        )
    if (
        official_report.target_soc != target_soc
        or official_report.cann_version != cann_version
        or official_report.source_sha256 != source.sha256
        or not _same_path(official_report.source_path, source.path)
        or official_report.status not in _RESOLVED_STATUSES
        or not official_report.decisions
        or any(
            decision.route not in _NON_BLOCKED_ROUTES
            or decision.activation_allowed is not False
            for decision in official_report.decisions
        )
        or official_value.get("activation_allowed") is not False
        or official_value.get("network_access_performed") is not False
        or official_value.get("system_write_performed") is not False
    ):
        raise RecoveryIngestError(
            "unresolved_official_receipt",
            "official receipt is blocked, side-effectful, or target-mismatched",
        )
    expected_specs = {
        (spec.gap_id, spec.qualified_api): gap_spec_sha256(spec)
        for spec in selection.specs
    }
    actual_decisions = {
        (item.gap_id, item.qualified_api): item.gap_spec_sha256
        for item in official_report.decisions
    }
    if expected_specs != actual_decisions:
        raise RecoveryIngestError(
            "receipt_binding_mismatch",
            "official decisions do not exactly cover the requested GapSpecs",
        )
    return _ValidatedInputs(
        source=source,
        migration=migration,
        gap_request=gap_request,
        official_resolution=official_resolution,
        migration_value=migration_value,
        gap_request_value=gap_value,
        selection=selection,
        preparation=preparation,
        official_report=official_report,
    )


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RecoveryIngestError(
            "invalid_receipt", "producer receipt is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _default_run_id(inputs: _ValidatedInputs) -> str:
    identity = {
        "source_sha256": inputs.source.sha256,
        "migration_report_sha256": inputs.migration.sha256,
        "gap_request_sha256": inputs.gap_request.sha256,
        "runtime_identity_sha256": inputs.migration_value[
            "qualified_analysis"
        ]["runtime_identity_sha256"],
    }
    return "recovery-%s" % _canonical_sha256(identity)[:20]


def _input_refs(inputs: _ValidatedInputs) -> Tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            "gap_request", str(inputs.gap_request.path), inputs.gap_request.sha256
        ),
        EvidenceRef(
            "official_resolution",
            str(inputs.official_resolution.path),
            inputs.official_resolution.sha256,
        ),
        EvidenceRef(
            "script_migration_report",
            str(inputs.migration.path),
            inputs.migration.sha256,
        ),
    )


def _per_gap_refs(
    inputs: _ValidatedInputs,
) -> Tuple[Mapping[str, EvidenceRef], Mapping[str, EvidenceRef]]:
    spec_refs = {
        spec.gap_id: EvidenceRef(
            "gap_spec",
            "%s#gap_specs/%s" % (inputs.gap_request.path, spec.gap_id),
            gap_spec_sha256(spec),
        )
        for spec in inputs.selection.specs
    }
    decision_refs = {
        decision.gap_id: EvidenceRef(
            "resolution_decision",
            "%s#decisions/%s"
            % (inputs.official_resolution.path, decision.gap_id),
            _canonical_sha256(decision.to_dict()),
        )
        for decision in inputs.official_report.decisions
    }
    if set(spec_refs) != set(decision_refs):
        raise RecoveryIngestError(
            "receipt_binding_mismatch",
            "per-gap producer receipt coverage is inconsistent",
        )
    return spec_refs, decision_refs


def _binding(
    inputs: _ValidatedInputs,
    run_id: Optional[str],
    policy: Optional[RecoveryPolicy],
    profile: Optional[RecoveryPolicyProfile | str],
) -> RecoveryBinding:
    selected_run_id = _default_run_id(inputs) if run_id is None else run_id
    if not isinstance(selected_run_id, str) or not _RUN_ID.fullmatch(
        selected_run_id
    ):
        raise RecoveryIngestError("invalid_run_id", "invalid recovery run id")
    if policy is not None and profile is not None:
        raise RecoveryIngestError(
            "policy_conflict",
            "recovery policy and named profile are mutually exclusive",
        )
    if policy is None:
        selected_profile = (
            RecoveryPolicyProfile.RECEIPT_ONLY if profile is None else profile
        )
        try:
            selected_policy = recovery_policy_for_profile(selected_profile)
        except RecoveryError as exc:
            raise RecoveryIngestError(
                "invalid_policy_profile", "invalid recovery policy profile"
            ) from exc
    else:
        selected_policy = policy
    if not isinstance(selected_policy, RecoveryPolicy):
        raise RecoveryIngestError(
            "invalid_policy", "recovery policy must be a RecoveryPolicy"
        )
    qualified = inputs.migration_value["qualified_analysis"]
    return RecoveryBinding(
        run=RecoveryRun(selected_run_id),
        source=RecoverySource(str(inputs.source.path), inputs.source.sha256),
        target=RecoveryTarget(
            inputs.gap_request_value["target_soc"],
            inputs.gap_request_value["cann_version"],
        ),
        runtime=RecoveryRuntime(
            backend=inputs.gap_request_value["backend"],
            audit_id=inputs.gap_request_value["audit_id"],
            identity_sha256=qualified["runtime_identity_sha256"],
        ),
        policy=selected_policy,
    )


def _find_ref(refs: Sequence[EvidenceRef], name: str) -> Optional[EvidenceRef]:
    return next((item for item in refs if item.name == name), None)


def _require_existing_prefix(
    status: RecoveryStatus,
    binding: RecoveryBinding,
    global_refs: Mapping[RecoveryPhase, EvidenceRef],
    gap_refs: Mapping[RecoveryPhase, Mapping[str, EvidenceRef]],
) -> None:
    state = status.state
    if state.binding != binding:
        raise RecoveryIngestError(
            "state_binding_mismatch", "existing recovery binding differs"
        )
    expected_ids = tuple(sorted(gap_refs[RecoveryPhase.GAP_PREPARED]))
    if tuple(item.gap_id for item in state.gaps) != expected_ids:
        raise RecoveryIngestError(
            "state_binding_mismatch", "existing recovery gap set differs"
        )
    binding_digest = recovery_binding_sha256(binding)
    stage_kinds = {
        RecoveryPhase.ANALYZED: StageReceiptKind.ANALYSIS_REPORT,
        RecoveryPhase.GAP_PREPARED: StageReceiptKind.GAP_SPEC,
        RecoveryPhase.RESOLVED: StageReceiptKind.OFFICIAL_RESOLUTION_DECISION,
    }
    covered = frozenset(expected_ids)
    for phase in (
        RecoveryPhase.ANALYZED,
        RecoveryPhase.GAP_PREPARED,
        RecoveryPhase.RESOLVED,
    ):
        if _PHASE_INDEX[state.phase] < _PHASE_INDEX[phase]:
            continue
        expected_ref = global_refs[phase]
        if _find_ref(state.refs, expected_ref.name) != expected_ref:
            raise RecoveryIngestError(
                "state_receipt_mismatch",
                "existing recovery lacks the expected %s ref" % phase.value,
            )
        matches = [
            item
            for item in state.stage_receipts
            if item.phase is phase
            and item.kind is stage_kinds[phase]
            and item.scope is StageReceiptScope.RUN
            and frozenset(item.covered_gap_ids) == covered
            and item.binding_sha256 == binding_digest
            and item.verdict is StageReceiptVerdict.SATISFIED
            and item.ref == expected_ref
        ]
        if len(matches) != 1:
            raise RecoveryIngestError(
                "state_receipt_mismatch",
                "existing recovery stage receipt is absent or ambiguous",
            )
        expected_per_gap = gap_refs.get(phase, {})
        for gap in state.gaps:
            expected_gap_ref = expected_per_gap.get(gap.gap_id)
            if expected_gap_ref is not None and _find_ref(
                gap.refs, expected_gap_ref.name
            ) != expected_gap_ref:
                raise RecoveryIngestError(
                    "state_receipt_mismatch",
                    "existing recovery per-gap receipt differs",
                )


def _progress_gaps(
    status: RecoveryStatus,
    phase: RecoveryPhase,
    additions: Mapping[str, EvidenceRef],
) -> Tuple[GapRecoveryState, ...]:
    progressed = []
    for gap in status.state.gaps:
        if _PHASE_INDEX[gap.phase] != _PHASE_INDEX[phase] - 1:
            raise RecoveryIngestError(
                "state_not_synchronous",
                "RUN-scoped ingestion requires every gap to advance together",
            )
        refs = tuple(gap.refs)
        addition = additions.get(gap.gap_id)
        if addition is not None:
            refs += (addition,)
        progressed.append(
            GapRecoveryState(
                gap_id=gap.gap_id,
                phase=phase,
                outcome=RecoveryOutcome.READY,
                attempt_count=gap.attempt_count,
                refs=refs,
            )
        )
    return tuple(progressed)


def _append_stage(
    store: RecoveryStore,
    status: RecoveryStatus,
    *,
    phase: RecoveryPhase,
    kind: StageReceiptKind,
    ref: EvidenceRef,
    gap_ref_additions: Mapping[str, EvidenceRef],
) -> RecoveryStatus:
    covered = tuple(item.gap_id for item in status.state.gaps)
    stage_receipt = create_stage_receipt(
        phase=phase,
        kind=kind,
        scope=StageReceiptScope.RUN,
        covered_gap_ids=covered,
        binding=status.state.binding,
        parent_state_sha256=status.state_sha256,
        verdict=StageReceiptVerdict.SATISFIED,
        ref=ref,
    )
    return store.append(
        RecoveryEvent.bound_to(
            status.state,
            phase=phase,
            refs=(ref,),
            stage_receipts=(stage_receipt,),
            gaps=_progress_gaps(status, phase, gap_ref_additions),
            side_effects=SideEffectTruth(),
        )
    )


def _verify_unchanged(snapshot: _Snapshot, maximum: int, label: str) -> None:
    current = _read_snapshot(snapshot.path, maximum, label)
    if current != snapshot:
        raise RecoveryIngestError(
            "input_drift", "%s changed during recovery ingestion" % label
        )


def ingest_official_recovery(
    source_path: os.PathLike[str] | str,
    migration_report_path: os.PathLike[str] | str,
    gap_request_path: os.PathLike[str] | str,
    official_resolution_path: os.PathLike[str] | str,
    state_dir: os.PathLike[str] | str,
    *,
    run_id: Optional[str] = None,
    policy: Optional[RecoveryPolicy] = None,
    profile: Optional[RecoveryPolicyProfile | str] = None,
) -> OfficialRecoveryIngestResult:
    """Validate three producer receipts and idempotently record their prefix.

    ``profile`` accepts one public named authorization set and defaults to
    ``receipt-only``.  The lower-level ``policy`` argument remains available
    to trusted Python callers, but it is mutually exclusive with ``profile``.
    Selecting a broader profile does not perform or claim any side effects;
    the appended events always carry an all-false :class:`SideEffectTruth`.
    """

    inputs = _load_validated_inputs(
        source_path,
        migration_report_path,
        gap_request_path,
        official_resolution_path,
    )
    recovery_root = _bounded_path(state_dir, "recovery state")
    binding = _binding(inputs, run_id, policy, profile)
    gap_ids = tuple(sorted(spec.gap_id for spec in inputs.selection.specs))
    if not gap_ids:
        raise RecoveryIngestError(
            "invalid_gap_request", "gap request contains no recoverable gaps"
        )
    refs = {item.name: item for item in _input_refs(inputs)}
    spec_refs, decision_refs = _per_gap_refs(inputs)
    global_refs = {
        RecoveryPhase.ANALYZED: refs["script_migration_report"],
        RecoveryPhase.GAP_PREPARED: refs["gap_request"],
        RecoveryPhase.RESOLVED: refs["official_resolution"],
    }
    gap_refs = {
        RecoveryPhase.ANALYZED: {},
        RecoveryPhase.GAP_PREPARED: spec_refs,
        RecoveryPhase.RESOLVED: decision_refs,
    }
    store = RecoveryStore(recovery_root)
    created = False
    try:
        status = store.create(binding, gap_ids=gap_ids)
        created = True
    except RecoveryError:
        try:
            status = store.status()
        except RecoveryError as exc:
            raise RecoveryIngestError(
                "state_unavailable", "recovery store cannot be created or resumed"
            ) from exc
    _require_existing_prefix(status, binding, global_refs, gap_refs)
    appended = 0
    stages = (
        (RecoveryPhase.ANALYZED, StageReceiptKind.ANALYSIS_REPORT),
        (RecoveryPhase.GAP_PREPARED, StageReceiptKind.GAP_SPEC),
        (
            RecoveryPhase.RESOLVED,
            StageReceiptKind.OFFICIAL_RESOLUTION_DECISION,
        ),
    )
    for phase, kind in stages:
        if _PHASE_INDEX[status.state.phase] >= _PHASE_INDEX[phase]:
            continue
        if status.state.outcome is not RecoveryOutcome.READY:
            raise RecoveryIngestError(
                "state_not_ready", "recovery state is not ready to advance"
            )
        try:
            status = _append_stage(
                store,
                status,
                phase=phase,
                kind=kind,
                ref=global_refs[phase],
                gap_ref_additions=gap_refs[phase],
            )
        except RecoveryError as exc:
            # A racing identical writer may have won publication.  Reload and
            # accept only an exact receipt-bound prefix.
            try:
                candidate = store.status()
                _require_existing_prefix(candidate, binding, global_refs, gap_refs)
            except RecoveryError as reload_exc:
                raise RecoveryIngestError(
                    "state_append_failed", "cannot append recovery stage"
                ) from reload_exc
            if _PHASE_INDEX[candidate.state.phase] < _PHASE_INDEX[phase]:
                raise RecoveryIngestError(
                    "state_append_failed", "recovery stage was not published"
                ) from exc
            status = candidate
        else:
            appended += 1
        _require_existing_prefix(status, binding, global_refs, gap_refs)
    if _PHASE_INDEX[status.state.phase] < _PHASE_INDEX[RecoveryPhase.RESOLVED]:
        raise RecoveryIngestError(
            "state_incomplete", "official recovery prefix did not reach RESOLVED"
        )
    for snapshot, maximum, label in (
        (inputs.source, _MAX_SOURCE_BYTES, "source"),
        (
            inputs.migration,
            _MAX_MIGRATION_REPORT_BYTES,
            "migration report",
        ),
        (inputs.gap_request, _MAX_GAP_REQUEST_BYTES, "gap request"),
        (
            inputs.official_resolution,
            _MAX_OFFICIAL_REPORT_BYTES,
            "official resolution report",
        ),
    ):
        _verify_unchanged(snapshot, maximum, label)
    return OfficialRecoveryIngestResult(
        recovery=status,
        created=created,
        appended_revisions=appended,
        input_receipts=tuple(refs.values()),
    )


def recovery_status(state_dir: os.PathLike[str] | str) -> RecoveryStatus:
    """Load and verify the complete recovery chain without side effects."""

    return load_recovery_status(_bounded_path(state_dir, "recovery state"))


__all__ = [
    "OfficialRecoveryIngestResult",
    "RecoveryIngestError",
    "ingest_official_recovery",
    "recovery_status",
]
