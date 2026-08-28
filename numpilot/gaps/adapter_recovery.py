"""Attach a generated official-ACLNN AsNumPy adapter to recovery state.

This module implements the narrow ``CANDIDATE_READY -> ADAPTER_CANDIDATE``
receipt boundary.  It never invokes a model or accesses the network.  Instead,
it revalidates an already-published schema-4 provider report, its complete
candidate tree, the reviewed MatMul existing-module context plan, and the
official-binary operator candidate to which the report claims to belong.

The resulting recovery event truthfully records the model and network effects
proved by that producer report.  Build, integration, import, execution,
verification, system-write, subprocess, and activation effects remain false.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Dict, Mapping, Sequence, Tuple

from numpilot.asnumpy_adapter_context import (
    AsnumpyAdapterContextError,
    EXISTING_MODULE_MODE,
    OFFICIAL_ACLNN_ARCHITECTURE,
    plan_official_aclnn_adapter_context,
)
from numpilot.gaps.official_adapter import (
    OfficialAdapterInputError,
    bind_official_adapter_context,
    load_operator_candidate_artifact,
    prepare_official_candidate_adapter_input,
)
from numpilot.gaps.official_resolution import (
    OfficialResolutionError,
    load_official_resolution_report,
)
from numpilot.gaps.pipeline import (
    GapCompletionError,
    load_script_gap_request,
    prepare_gap_completion,
)
from numpilot.gaps.provider import (
    DraftGenerationError,
    _load_asnumpy_adapter_report,
)
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    OperatorCandidateArtifact,
    OperatorCandidateOrigin,
    RecoveryError,
    RecoveryEvent,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryStatus,
    RecoveryStore,
    SideEffectTruth,
    StageReceiptKind,
    StageReceiptScope,
    StageReceiptVerdict,
    create_stage_receipt,
)


ADAPTER_RECOVERY_SCHEMA_VERSION = 1
_MAX_INPUT_BYTES = 2 * 1024 * 1024
_MAX_OFFICIAL_REPORT_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_APPEND_ATTEMPTS = 256
_ADAPTER_REF_NAME = "asnumpy_adapter_candidate"
_REVIEWED_QUALIFIED_API = "numpy.matmul"
_REVIEWED_OVERLAY_PATHS = (
    "csrc/linalg/product.cpp",
    "src/asnumpy/linalg/direct.py",
    "tests/asnumpy_tests/linalg_tests/test_product.py",
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


class AdapterRecoveryError(RecoveryError):
    """The adapter receipt failed closed with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OfficialAdapterRecoveryResult:
    """Deterministic result of attaching one generated adapter receipt."""

    recovery: RecoveryStatus
    gap_id: str
    candidate_id: str
    adapter_artifact_id: str
    adapter_ref: EvidenceRef
    adapter_output_root: str
    appended_revisions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ADAPTER_RECOVERY_SCHEMA_VERSION,
            "record_type": "numpilot_official_adapter_recovery",
            "route": OperatorCandidateOrigin.OFFICIAL_BINARY.value,
            "gap_id": self.gap_id,
            "candidate_id": self.candidate_id,
            "adapter_artifact_id": self.adapter_artifact_id,
            "adapter_ref": self.adapter_ref.to_dict(),
            "adapter_output_root": self.adapter_output_root,
            "producer_receipt_revalidated": True,
            "model_invoked_by_ingest": False,
            "appended_revisions": self.appended_revisions,
            "side_effects_performed": SideEffectTruth(
                network_access_performed=True,
                model_invocation_performed=True,
            ).to_dict(),
            "recovery": self.recovery.to_dict(),
        }


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _ValidatedAdapter:
    root: Path
    report: Mapping[str, Any]
    report_ref: EvidenceRef


def _absolute_path(value: os.PathLike[str] | str, label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise AdapterRecoveryError("invalid_path", "invalid %s path" % label) from exc
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw) > 4096
        or "\x00" in raw
    ):
        raise AdapterRecoveryError("invalid_path", "invalid %s path" % label)
    try:
        return Path(os.path.abspath(os.path.expanduser(raw)))
    except (OSError, ValueError) as exc:
        raise AdapterRecoveryError("invalid_path", "invalid %s path" % label) from exc


def _directory_identity(path: Path, label: str) -> Tuple[int, int]:
    """Open every path component with ``O_NOFOLLOW`` and return its inode."""

    parts = path.parts
    if len(parts) < 1 or not path.is_absolute():
        raise AdapterRecoveryError("invalid_path", "invalid %s path" % label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path.anchor, flags)
        for part in parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise AdapterRecoveryError(
                "unsafe_input", "%s must be a real directory" % label
            )
        return metadata.st_dev, metadata.st_ino
    except AdapterRecoveryError:
        raise
    except OSError as exc:
        raise AdapterRecoveryError(
            "unsafe_input", "cannot safely open %s" % label
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _snapshot_regular_file(path: Path, maximum: int, label: str) -> _Snapshot:
    """Read one stable, bounded regular file through no-follow directory FDs."""

    if not path.is_absolute() or len(path.parts) < 2 or not path.name:
        raise AdapterRecoveryError("invalid_path", "invalid %s path" % label)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_descriptor = -1
    descriptor = -1
    try:
        directory_descriptor = os.open(path.anchor, directory_flags)
        for part in path.parts[1:-1]:
            next_descriptor = os.open(
                part, directory_flags, dir_fd=directory_descriptor
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(path.name, file_flags, dir_fd=directory_descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterRecoveryError(
                "unsafe_input", "%s must be a regular file" % label
            )
        if before.st_size <= 0 or before.st_size > maximum:
            raise AdapterRecoveryError(
                "input_size_invalid", "%s exceeds its size bounds" % label
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
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            len(payload) != before.st_size
            or len(payload) > maximum
            or before_identity != after_identity
        ):
            raise AdapterRecoveryError(
                "input_drift", "%s changed while being read" % label
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
            raise AdapterRecoveryError(
                "input_drift", "%s changed while being read" % label
            )
    except AdapterRecoveryError:
        raise
    except OSError as exc:
        raise AdapterRecoveryError(
            "unsafe_input", "cannot safely read %s" % label
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    return _Snapshot(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=before_identity,
    )


def _strict_json_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AdapterRecoveryError(
                "invalid_adapter_report", "adapter report contains duplicate keys"
            )
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise AdapterRecoveryError(
        "invalid_adapter_report", "adapter report contains a non-finite number"
    )


def _strict_report(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except AdapterRecoveryError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AdapterRecoveryError(
            "invalid_adapter_report", "adapter report is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise AdapterRecoveryError(
            "invalid_adapter_report", "adapter report must be an object"
        )
    return value


def _find_ref(refs: Sequence[EvidenceRef], name: str, label: str) -> EvidenceRef:
    matches = tuple(item for item in refs if item.name == name)
    if len(matches) != 1:
        raise AdapterRecoveryError(
            "missing_binding", "%s requires exactly one %s reference" % (label, name)
        )
    return matches[0]


def _verify_bound_file(ref: EvidenceRef, maximum: int, label: str) -> Path:
    path = _absolute_path(ref.locator, label)
    if ref.locator != str(path):
        raise AdapterRecoveryError(
            "binding_path_mismatch", "%s locator is not canonical" % label
        )
    snapshot = _snapshot_regular_file(path, maximum, label)
    if snapshot.sha256 != ref.sha256:
        raise AdapterRecoveryError(
            "binding_hash_mismatch", "%s differs from recovery state" % label
        )
    return path


def _candidate_gap(
    status: RecoveryStatus,
    candidate_path: Path,
) -> Tuple[OperatorCandidateArtifact, GapRecoveryState, _Snapshot]:
    snapshot = _snapshot_regular_file(
        candidate_path, _MAX_INPUT_BYTES, "operator candidate"
    )
    try:
        candidate = load_operator_candidate_artifact(candidate_path)
    except (OfficialAdapterInputError, RecoveryError, TypeError, ValueError) as exc:
        raise AdapterRecoveryError(
            "invalid_candidate", "operator candidate is invalid"
        ) from exc
    if candidate.origin_kind is not OperatorCandidateOrigin.OFFICIAL_BINARY:
        raise AdapterRecoveryError(
            "wrong_candidate_origin",
            "adapter recovery requires an official_binary operator candidate",
        )
    matches = tuple(
        item for item in status.state.gaps if item.gap_id == candidate.gap_id
    )
    if len(matches) != 1:
        raise AdapterRecoveryError(
            "candidate_gap_mismatch", "candidate is not part of this recovery run"
        )
    gap = matches[0]
    candidate_ref = _find_ref(gap.refs, "operator_candidate", "gap state")
    if (
        candidate_ref.locator != str(candidate_path)
        or candidate_ref.sha256 != snapshot.sha256
    ):
        raise AdapterRecoveryError(
            "candidate_binding_mismatch",
            "candidate path or content differs from the gap-state reference",
        )
    candidate_receipts = tuple(
        item
        for item in status.state.stage_receipts
        if item.phase is RecoveryPhase.CANDIDATE_READY
        and item.kind is StageReceiptKind.OPERATOR_CANDIDATE
        and item.scope is StageReceiptScope.GAPS
        and item.covered_gap_ids == (gap.gap_id,)
        and item.verdict is StageReceiptVerdict.SATISFIED
        and item.ref == candidate_ref
    )
    if len(candidate_receipts) != 1:
        raise AdapterRecoveryError(
            "candidate_receipt_missing",
            "candidate lacks its exact CANDIDATE_READY stage receipt",
        )
    if (
        candidate.target_soc != status.state.target.target_soc
        or candidate.cann_version != status.state.target.cann_version
    ):
        raise AdapterRecoveryError(
            "candidate_target_mismatch", "candidate target differs from recovery state"
        )
    gap_spec_ref = _find_ref(gap.refs, "gap_spec", "gap state")
    decision_ref = _find_ref(gap.refs, "resolution_decision", "gap state")
    if (
        candidate.gap_spec_ref != gap_spec_ref
        or candidate.resolution_decision_ref != decision_ref
    ):
        raise AdapterRecoveryError(
            "candidate_binding_mismatch",
            "candidate GapSpec or resolution binding differs from gap state",
        )
    return candidate, gap, snapshot


def _official_adapter_input(
    status: RecoveryStatus,
    candidate: OperatorCandidateArtifact,
):
    state = status.state
    source_path = _absolute_path(state.source.path, "recovery source")
    source_snapshot = _snapshot_regular_file(
        source_path, _MAX_SOURCE_BYTES, "recovery source"
    )
    if source_snapshot.sha256 != state.source.sha256:
        raise AdapterRecoveryError(
            "binding_hash_mismatch", "recovery source content drifted"
        )
    gap_request_ref = _find_ref(state.refs, "gap_request", "recovery state")
    official_ref = _find_ref(
        state.refs, "official_resolution", "recovery state"
    )
    gap_request_path = _verify_bound_file(
        gap_request_ref, _MAX_INPUT_BYTES, "gap request"
    )
    official_path = _verify_bound_file(
        official_ref, _MAX_OFFICIAL_REPORT_BYTES, "official resolution"
    )
    try:
        selection = load_script_gap_request(
            str(gap_request_path),
            str(source_path),
            target_soc=state.target.target_soc,
            cann_version=state.target.cann_version,
        )
        if selection.blocked:
            raise AdapterRecoveryError(
                "unsupported_gap_bundle", "adapter recovery requires generatable gaps"
            )
        preparation = prepare_gap_completion(
            str(source_path),
            target_soc=state.target.target_soc,
            cann_version=state.target.cann_version,
            requested_specs=selection.specs,
            requested_blocked=selection.blocked,
        )
        official_report = load_official_resolution_report(str(official_path))
        official_report.validate_preparation(preparation)
    except AdapterRecoveryError:
        raise
    except (GapCompletionError, OfficialResolutionError, OSError, ValueError) as exc:
        raise AdapterRecoveryError(
            "invalid_bound_receipt",
            "cannot reconstruct the bound GapSpec and official resolution",
        ) from exc
    specs = tuple(item for item in selection.specs if item.gap_id == candidate.gap_id)
    decisions = tuple(
        item for item in official_report.decisions if item.gap_id == candidate.gap_id
    )
    if len(specs) != 1 or len(decisions) != 1:
        raise AdapterRecoveryError(
            "candidate_gap_mismatch",
            "bound receipts do not contain exactly one candidate gap",
        )
    try:
        return prepare_official_candidate_adapter_input(
            candidate, specs[0], decisions[0]
        )
    except (OfficialAdapterInputError, RecoveryError, TypeError, ValueError) as exc:
        raise AdapterRecoveryError(
            "official_adapter_binding_failed",
            "candidate cannot produce an official ACLNN adapter contract",
        ) from exc


def _load_adapter_output(
    root: Path,
    candidate: OperatorCandidateArtifact,
    prepared,
) -> _ValidatedAdapter:
    root_identity = _directory_identity(root, "adapter output root")
    report_path = root / "asnumpy-adapter-report.json"
    before = _snapshot_regular_file(
        report_path, _MAX_INPUT_BYTES, "AsNumPy adapter report"
    )
    raw_report = _strict_report(before.payload)
    if (
        raw_report.get("schema_version") != 4
        or raw_report.get("status") != "generated_unverified"
        or raw_report.get("actual_remote_call") is not True
    ):
        raise AdapterRecoveryError(
            "wrong_adapter_profile",
            "official adapter must be a generated_unverified schema-4 remote receipt",
        )
    try:
        report, payload, file_hashes = _load_asnumpy_adapter_report(root)
    except (DraftGenerationError, OSError, TypeError, ValueError) as exc:
        raise AdapterRecoveryError(
            "invalid_adapter_output",
            "AsNumPy adapter report or candidate tree is invalid",
        ) from exc
    if payload != before.payload:
        raise AdapterRecoveryError(
            "adapter_output_drift", "adapter report changed during validation"
        )
    if report.get("source_artifact_id") != candidate.candidate_id:
        raise AdapterRecoveryError(
            "adapter_source_mismatch",
            "adapter source_artifact_id differs from the operator candidate",
        )
    if candidate.qualified_api != _REVIEWED_QUALIFIED_API:
        raise AdapterRecoveryError(
            "unreviewed_official_adapter",
            "only the reviewed numpy.matmul official ACLNN plan is accepted",
        )
    context_plan = report.get("asnumpy_adapter_context_plan")
    if (
        not isinstance(context_plan, Mapping)
        or context_plan.get("architecture") != OFFICIAL_ACLNN_ARCHITECTURE
        or context_plan.get("mode") != EXISTING_MODULE_MODE
        or tuple(context_plan.get("expected_overlay_paths", ()))
        != _REVIEWED_OVERLAY_PATHS
        or tuple(sorted(file_hashes)) != tuple(sorted(_REVIEWED_OVERLAY_PATHS))
    ):
        raise AdapterRecoveryError(
            "wrong_adapter_architecture",
            "adapter does not use the reviewed MatMul existing-module overlay",
        )
    try:
        selected_library = prepared.selected_library
        fresh_plan = plan_official_aclnn_adapter_context(
            context_plan.get("reference_root"),
            "matmul",
            selected_library_relative_path=selected_library[
                "relative_path"
            ],
            selected_library_sha256=selected_library["sha256"],
        )
    except (AsnumpyAdapterContextError, OSError, TypeError, ValueError) as exc:
        raise AdapterRecoveryError(
            "adapter_context_invalid",
            "cannot revalidate the reviewed AsNumPy context plan",
        ) from exc
    if fresh_plan.manifest() != dict(context_plan):
        raise AdapterRecoveryError(
            "adapter_context_drift",
            "current AsNumPy context differs from the generated adapter plan",
        )
    try:
        ready = bind_official_adapter_context(
            prepared, fresh_plan, api_name="matmul"
        )
    except (OfficialAdapterInputError, TypeError, ValueError) as exc:
        raise AdapterRecoveryError(
            "adapter_context_invalid",
            "reviewed context cannot open the official adapter model gate",
        ) from exc
    if report.get("adapter_contract_sha256") != ready.adapter_contract_sha256:
        raise AdapterRecoveryError(
            "adapter_contract_mismatch",
            "adapter contract differs from the official candidate binding",
        )

    # A second complete provider reload catches ordinary concurrent drift and
    # guarantees that the exact report bytes used for the EvidenceRef still
    # describe the exact tree accepted above.
    after = _snapshot_regular_file(
        report_path, _MAX_INPUT_BYTES, "AsNumPy adapter report"
    )
    if (
        after != before
        or _directory_identity(root, "adapter output root") != root_identity
    ):
        raise AdapterRecoveryError(
            "adapter_output_drift", "adapter output changed during validation"
        )
    try:
        report_again, payload_again, hashes_again = _load_asnumpy_adapter_report(root)
    except (DraftGenerationError, OSError, TypeError, ValueError) as exc:
        raise AdapterRecoveryError(
            "adapter_output_drift", "adapter output changed during validation"
        ) from exc
    if (
        report_again != report
        or payload_again != payload
        or hashes_again != file_hashes
        or _directory_identity(root, "adapter output root") != root_identity
    ):
        raise AdapterRecoveryError(
            "adapter_output_drift", "adapter output changed during validation"
        )
    return _ValidatedAdapter(
        root=root,
        report=report,
        report_ref=EvidenceRef(
            name=_ADAPTER_REF_NAME,
            locator=str(report_path),
            sha256=before.sha256,
        ),
    )


def _gap_for_candidate(status: RecoveryStatus, gap_id: str) -> GapRecoveryState:
    matches = tuple(item for item in status.state.gaps if item.gap_id == gap_id)
    if len(matches) != 1:
        raise AdapterRecoveryError(
            "candidate_gap_mismatch", "candidate gap disappeared from recovery state"
        )
    gap = matches[0]
    _find_ref(gap.refs, "operator_candidate", "gap state")
    return gap


def _existing_adapter_matches(
    status: RecoveryStatus,
    *,
    gap_id: str,
    adapter_ref: EvidenceRef,
) -> bool:
    gap = _gap_for_candidate(status, gap_id)
    refs = tuple(item for item in gap.refs if item.name == _ADAPTER_REF_NAME)
    phase_reached = _PHASE_INDEX[gap.phase] >= _PHASE_INDEX[
        RecoveryPhase.ADAPTER_CANDIDATE
    ]
    if not refs:
        if phase_reached:
            raise AdapterRecoveryError(
                "adapter_state_mismatch",
                "advanced gap is missing its adapter candidate reference",
            )
        return False
    if len(refs) != 1 or refs[0] != adapter_ref:
        raise AdapterRecoveryError(
            "adapter_state_drift",
            "immutable adapter candidate reference differs from this output",
        )
    receipts = tuple(
        item
        for item in status.state.stage_receipts
        if item.phase is RecoveryPhase.ADAPTER_CANDIDATE
        and item.kind is StageReceiptKind.ASNUMPY_ADAPTER_CANDIDATE
        and item.scope is StageReceiptScope.GAPS
        and item.covered_gap_ids == (gap_id,)
        and item.verdict is StageReceiptVerdict.SATISFIED
        and item.ref == adapter_ref
    )
    if len(receipts) != 1 or not phase_reached:
        raise AdapterRecoveryError(
            "adapter_state_mismatch",
            "adapter reference lacks its exact ADAPTER_CANDIDATE receipt",
        )
    if not (
        status.state.side_effects.network_access_performed
        and status.state.side_effects.model_invocation_performed
    ):
        raise AdapterRecoveryError(
            "adapter_state_mismatch",
            "adapter state does not record its model and network effects",
        )
    return True


def _result(
    status: RecoveryStatus,
    candidate: OperatorCandidateArtifact,
    adapter: _ValidatedAdapter,
    appended: int,
) -> OfficialAdapterRecoveryResult:
    return OfficialAdapterRecoveryResult(
        recovery=status,
        gap_id=candidate.gap_id,
        candidate_id=candidate.candidate_id,
        adapter_artifact_id=str(adapter.report["adapter_artifact_id"]),
        adapter_ref=adapter.report_ref,
        adapter_output_root=str(adapter.root),
        appended_revisions=appended,
    )


def ingest_official_adapter_candidate(
    state_dir: os.PathLike[str] | str,
    candidate_path: os.PathLike[str] | str,
    adapter_output_root: os.PathLike[str] | str,
) -> OfficialAdapterRecoveryResult:
    """Revalidate one generated official adapter and append its stage receipt.

    The function is intentionally an ingest boundary: it performs no model or
    network operation.  Its immutable recovery policy must nevertheless allow
    both effects because the accepted provider receipt proves that the adapter
    producer already performed them.
    """

    store = RecoveryStore(state_dir)
    try:
        status = store.status()
    except RecoveryError as exc:
        raise AdapterRecoveryError(
            "state_unavailable", "cannot verify the complete recovery chain"
        ) from exc
    if not status.state.policy.network_allowed:
        raise AdapterRecoveryError(
            "network_not_authorized",
            "frozen recovery policy does not allow adapter network access",
        )
    if not status.state.policy.model_allowed:
        raise AdapterRecoveryError(
            "model_not_authorized",
            "frozen recovery policy does not allow adapter model invocation",
        )
    candidate_file = _absolute_path(candidate_path, "operator candidate")
    adapter_root = _absolute_path(adapter_output_root, "adapter output root")
    candidate, gap, candidate_before = _candidate_gap(status, candidate_file)
    if _PHASE_INDEX[gap.phase] < _PHASE_INDEX[RecoveryPhase.CANDIDATE_READY]:
        raise AdapterRecoveryError(
            "state_phase_mismatch",
            "adapter recovery requires a CANDIDATE_READY gap",
        )
    prepared = _official_adapter_input(status, candidate)
    adapter = _load_adapter_output(adapter_root, candidate, prepared)
    candidate_after = _snapshot_regular_file(
        candidate_file, _MAX_INPUT_BYTES, "operator candidate"
    )
    if candidate_after != candidate_before:
        raise AdapterRecoveryError(
            "candidate_drift", "operator candidate changed during adapter validation"
        )

    for _attempt in range(_MAX_APPEND_ATTEMPTS):
        try:
            current = store.status()
        except RecoveryError as exc:
            raise AdapterRecoveryError(
                "state_unavailable", "cannot reload the complete recovery chain"
            ) from exc
        if _existing_adapter_matches(
            current, gap_id=candidate.gap_id, adapter_ref=adapter.report_ref
        ):
            return _result(current, candidate, adapter, 0)
        current_gap = _gap_for_candidate(current, candidate.gap_id)
        current_candidate_ref = _find_ref(
            current_gap.refs, "operator_candidate", "gap state"
        )
        if (
            current_candidate_ref.locator != str(candidate_file)
            or current_candidate_ref.sha256 != candidate_before.sha256
        ):
            raise AdapterRecoveryError(
                "candidate_binding_mismatch",
                "candidate reference changed while appending adapter state",
            )
        if (
            current_gap.phase is not RecoveryPhase.CANDIDATE_READY
            or current.state.outcome is not RecoveryOutcome.READY
        ):
            raise AdapterRecoveryError(
                "state_phase_mismatch",
                "gap is not ready for its adapter candidate transition",
            )
        next_gaps = tuple(
            GapRecoveryState(
                gap_id=item.gap_id,
                phase=(
                    RecoveryPhase.ADAPTER_CANDIDATE
                    if item.gap_id == candidate.gap_id
                    else item.phase
                ),
                outcome=item.outcome,
                attempt_count=item.attempt_count,
                refs=(
                    tuple(item.refs) + (adapter.report_ref,)
                    if item.gap_id == candidate.gap_id
                    else item.refs
                ),
                reason=item.reason,
            )
            for item in current.state.gaps
        )
        global_phase = min(next_gaps, key=lambda item: _PHASE_INDEX[item.phase]).phase
        receipt = create_stage_receipt(
            phase=RecoveryPhase.ADAPTER_CANDIDATE,
            kind=StageReceiptKind.ASNUMPY_ADAPTER_CANDIDATE,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(candidate.gap_id,),
            binding=current.state.binding,
            parent_state_sha256=current.state_sha256,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=adapter.report_ref,
        )
        event = RecoveryEvent.bound_to(
            current.state,
            phase=global_phase,
            outcome=RecoveryOutcome.READY,
            gaps=next_gaps,
            stage_receipts=(receipt,),
            side_effects=SideEffectTruth(
                network_access_performed=True,
                model_invocation_performed=True,
            ),
            reason=(
                "schema-4 official ACLNN AsNumPy adapter candidate was "
                "content-bound; integration, build, execution, and activation "
                "remain pending"
            ),
        )
        try:
            appended = store.append(event)
        except RecoveryError:
            # An identical producer, or another gap producer, may have won the
            # write-once revision.  Reload and either accept the exact winner
            # on the next iteration or retry against the new parent digest.
            continue
        return _result(appended, candidate, adapter, 1)
    raise AdapterRecoveryError(
        "state_append_failed", "recovery state changed too often to append adapter"
    )


__all__ = [
    "ADAPTER_RECOVERY_SCHEMA_VERSION",
    "AdapterRecoveryError",
    "OfficialAdapterRecoveryResult",
    "ingest_official_adapter_candidate",
]
