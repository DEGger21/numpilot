"""Native recovery bridge for ``official_candidate_gap`` hand-offs.

This module is deliberately separate from the ordinary ``script_api_gap``
ingest path.  It consumes the already validated official candidate chain,
requires an explicit runtime binding, and advances the existing RecoveryStore
through ``CANDIDATE_READY`` without invoking a model, building code, importing
AsNumpy, executing on NPU, or activating a capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from numpilot.gaps.official_candidate_handoff import (
    OfficialCandidateHandoffError,
    load_official_candidate_handoff,
)
from numpilot.gaps.official_candidate_preflight import (
    OfficialCandidatePreflightError,
    prepare_official_adapter_input_from_preflight,
)
from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    OfficialResolutionError,
    gap_spec_sha256,
    load_official_resolution_report,
)
from numpilot.gaps.pipeline import GapCompletionError, prepare_gap_completion
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    LicenseBinding,
    OperatorCandidateArtifact,
    OperatorCandidateOrigin,
    RecoveryBinding,
    RecoveryError,
    RecoveryEvent,
    RecoveryOutcome,
    RecoveryPhase,
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
    create_operator_candidate,
    create_stage_receipt,
    recovery_policy_for_profile,
)

SCHEMA = "numpilot.official-candidate-recovery.v1"
STATUS = "candidate_ready"
_EXTERNAL_CANN_TERMS = "LicenseRef-External-Installed-CANN-Terms"
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_BYTES = 4 * 1024 * 1024


class OfficialCandidateRecoveryError(RecoveryError):
    """The official candidate recovery input is stale or unsafe."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OfficialCandidateRecoveryResult:
    recovery: RecoveryStatus
    output_dir: str
    candidates: Tuple[OperatorCandidateArtifact, ...]
    appended_revisions: int
    side_effects: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": STATUS,
            "output_dir": self.output_dir,
            "appended_revisions": self.appended_revisions,
            "candidates": [item.to_dict() for item in self.candidates],
            "side_effects": dict(self.side_effects),
            "recovery": self.recovery.to_dict(),
        }


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise OfficialCandidateRecoveryError("invalid canonical evidence", "evidence is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path, label: str, maximum: int) -> str:
    try:
        details = path.stat()
        if not path.is_file() or path.is_symlink() or details.st_size > maximum:
            raise OfficialCandidateRecoveryError("invalid_input", "%s must be a bounded regular file" % label)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OfficialCandidateRecoveryError:
        raise
    except OSError as exc:
        raise OfficialCandidateRecoveryError("invalid_input", "cannot read %s" % label) from exc


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > _MAX_RECEIPT_BYTES:
            raise OfficialCandidateRecoveryError("invalid_input", "%s must be a bounded regular file" % label)
        value = json.loads(path.read_text(encoding="utf-8"))
    except OfficialCandidateRecoveryError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise OfficialCandidateRecoveryError("invalid_input", "cannot read %s" % label) from exc
    if not isinstance(value, Mapping):
        raise OfficialCandidateRecoveryError("invalid_input", "%s must be an object" % label)
    return value


def _candidate_refs(
    resolution_path: Path,
    decision: Any,
    report: Any,
    handoff_path: Path,
    preflight_path: Path,
    spec: Any,
) -> OperatorCandidateArtifact:
    installed = decision.installed_cann
    candidate_mapping = installed.get("candidate")
    header = installed.get("selected_header")
    library = installed.get("selected_library")
    symbols = installed.get("required_symbols")
    if not isinstance(candidate_mapping, Mapping) or not isinstance(header, Mapping) or not isinstance(library, Mapping):
        raise OfficialCandidateRecoveryError("incomplete_candidate", "candidate/header/library evidence is incomplete")
    if not isinstance(symbols, list) or len(symbols) != 2 or any(not isinstance(item, str) or not item for item in symbols):
        raise OfficialCandidateRecoveryError("incomplete_candidate", "official candidate must bind both ACLNN stages")
    mapping_sha = installed.get("candidate_mapping_sha256")
    if not isinstance(mapping_sha, str) or len(mapping_sha) != 64:
        raise OfficialCandidateRecoveryError("incomplete_candidate", "candidate mapping hash is missing")
    try:
        int(mapping_sha, 16)
    except ValueError as exc:
        raise OfficialCandidateRecoveryError("incomplete_candidate", "candidate mapping hash is invalid") from exc
    resolution_abs = str(resolution_path.absolute())
    gid = decision.gap_id
    candidate_refs = (
        EvidenceRef("official_candidate_mapping", resolution_abs + "#decisions/%s/installed_cann/candidate" % gid, mapping_sha),
        EvidenceRef("official_selected_header", resolution_abs + "#decisions/%s/installed_cann/selected_header" % gid, str(header["sha256"])),
        EvidenceRef("official_selected_library", resolution_abs + "#decisions/%s/installed_cann/selected_library" % gid, str(library["sha256"])),
        EvidenceRef("official_required_symbols", resolution_abs + "#decisions/%s/installed_cann/required_symbols" % gid, _sha256(_canonical({"required_symbols": list(symbols)}))),
        EvidenceRef("official_semantic_review", resolution_abs + "#decisions/%s/installed_cann/candidate/semantic_review" % gid, _sha256(_canonical({"semantic_review": list(candidate_mapping.get("semantic_review", []))}))),
    )
    license_ref = EvidenceRef(
        "external_installed_cann_terms",
        resolution_abs + "#external-installed-cann-terms",
        _sha256(_canonical({
            "identifier": _EXTERNAL_CANN_TERMS,
            "accepted": False,
            "cann_root": report.cann_root,
            "cann_version": report.cann_version,
            "target_soc": report.target_soc,
            "gap_id": gid,
            "qualified_api": decision.qualified_api,
        })),
    )
    return create_operator_candidate(
        origin_kind=OperatorCandidateOrigin.OFFICIAL_BINARY,
        gap_id=spec.gap_id,
        qualified_api=spec.qualified_api,
        gap_spec_ref=EvidenceRef("gap_spec", str(handoff_path.absolute()) + "#gap_specs/%s" % spec.gap_id, gap_spec_sha256(spec)),
        resolution_decision_ref=EvidenceRef("resolution_decision", resolution_abs + "#decisions/%s" % gid, _sha256(_canonical(decision.to_dict()))),
        target_soc=spec.target_soc,
        cann_version=spec.cann_version,
        license=LicenseBinding(_EXTERNAL_CANN_TERMS, False, license_ref),
        provenance_refs=(
            EvidenceRef("official_resolution", resolution_abs, _sha256(_canonical(report.to_dict()))),
            EvidenceRef("official_candidate_preflight", str(preflight_path.absolute()), _file_sha256(preflight_path, "preflight", _MAX_RECEIPT_BYTES)),
        ),
        candidate_refs=candidate_refs,
    )


def _write_candidate(path: Path, candidate: OperatorCandidateArtifact) -> EvidenceRef:
    payload = (_canonical(candidate.to_dict()) + b"\n")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_details = path.parent.stat()
    if not path.parent.is_dir() or parent_details.st_mode & 0o777 != 0o700:
        raise OfficialCandidateRecoveryError("candidate_artifact_permissions", "candidate artifact directory must be private 0700")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600 or path.read_bytes() != payload:
            raise OfficialCandidateRecoveryError("candidate_artifact_drift", "candidate artifact already exists with different content")
    else:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    return EvidenceRef("official_candidate_%s" % candidate.gap_id, str(path.absolute()), _sha256(payload))


def _advance(
    store: RecoveryStore,
    status: RecoveryStatus,
    phase: RecoveryPhase,
    kind: StageReceiptKind,
    ref_by_gap: Mapping[str, EvidenceRef],
    reason: str,
) -> RecoveryStatus:
    if status.state.phase is phase:
        return status
    if status.state.phase.value == "candidate_ready":
        return status
    if status.state.phase.value != {
        RecoveryPhase.ANALYZED: RecoveryPhase.CREATED,
        RecoveryPhase.GAP_PREPARED: RecoveryPhase.ANALYZED,
        RecoveryPhase.RESOLVED: RecoveryPhase.GAP_PREPARED,
        RecoveryPhase.CANDIDATE_READY: RecoveryPhase.RESOLVED,
    }[phase].value:
        raise OfficialCandidateRecoveryError("state_phase_mismatch", "official recovery state is not at the expected predecessor phase")
    gaps = tuple(
        GapRecoveryState(
            gap_id=gap.gap_id,
            phase=phase,
            outcome=RecoveryOutcome.READY,
            attempt_count=gap.attempt_count,
            refs=tuple(gap.refs) + (ref_by_gap[gap.gap_id],),
        )
        for gap in status.state.gaps
    )
    receipts = tuple(
        create_stage_receipt(
            phase=phase,
            kind=kind,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(gap.gap_id,),
            binding=status.state.binding,
            parent_state_sha256=status.state_sha256,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=ref_by_gap[gap.gap_id],
        )
        for gap in status.state.gaps
    )
    try:
        return store.append(RecoveryEvent.bound_to(
            status.state,
            phase=phase,
            outcome=RecoveryOutcome.READY,
            refs=tuple(ref_by_gap.values()),
            gaps=gaps,
            stage_receipts=receipts,
            side_effects=SideEffectTruth(),
            reason=reason,
        ))
    except RecoveryError as exc:
        raise OfficialCandidateRecoveryError("state_append_failed", "cannot append official recovery stage") from exc


def materialize_official_candidate_recovery(
    handoff_path: str,
    preflight_path: str,
    source_path: str,
    official_resolution_path: str,
    context_gate_path: str,
    state_dir: str,
    *,
    runtime_backend: str,
    runtime_audit_id: str,
    runtime_identity_sha256: str,
    run_id: Optional[str] = None,
    profile: RecoveryPolicyProfile | str = RecoveryPolicyProfile.RECEIPT_ONLY,
    output_dir: Optional[str] = None,
) -> OfficialCandidateRecoveryResult:
    """Consume the official candidate chain and reach ``CANDIDATE_READY``."""
    try:
        selected_profile = RecoveryPolicyProfile(profile)
    except ValueError as exc:
        raise OfficialCandidateRecoveryError("invalid_profile", "unsupported recovery policy profile") from exc
    if selected_profile is not RecoveryPolicyProfile.RECEIPT_ONLY:
        raise OfficialCandidateRecoveryError("unsafe_profile", "official candidate recovery requires receipt-only policy")
    source_file = Path(source_path).absolute()
    handoff_file = Path(handoff_path).absolute()
    preflight_file = Path(preflight_path).absolute()
    resolution_file = Path(official_resolution_path).absolute()
    gate_file = Path(context_gate_path).absolute()
    try:
        gate = _read_json(gate_file, "context gate")
        required_gate_fields = {
            "schema", "status", "source", "reference_root", "qualified_api",
            "candidate_id", "context_plan_sha256", "adapter_contract_sha256",
            "model_invocation_performed", "network_access_performed",
            "system_write_performed", "asnumpy_import_performed", "build_performed",
            "execution_performed", "activation_performed", "next_action",
        }
        if set(gate) != required_gate_fields or gate.get("schema") != "numpilot.official-adapter-context-gate.v1" or gate.get("status") != "ready_for_official_adapter_model" or gate.get("next_action") != "invoke_configured_model_for_official_adapter":
            raise OfficialCandidateRecoveryError("unsafe_gate", "official adapter model gate receipt is not ready")
        if any(gate.get(name) is not False for name in (
            "model_invocation_performed", "network_access_performed",
            "system_write_performed", "asnumpy_import_performed", "build_performed",
            "execution_performed", "activation_performed",
        )):
            raise OfficialCandidateRecoveryError("unsafe_gate", "official adapter model gate receipt claims a side effect")
        if gate.get("source") != str(source_file):
            raise OfficialCandidateRecoveryError("unsafe_gate", "official adapter model gate source identity drift")
        handoff = load_official_candidate_handoff(str(handoff_file), str(source_file), str(resolution_file))
        report = load_official_resolution_report(str(resolution_file))
        preparation = prepare_gap_completion(str(source_file), target_soc=report.target_soc, cann_version=report.cann_version)
        report.validate_preparation(preparation)
        if len(preparation.specs) != len(report.decisions) or not preparation.specs:
            raise OfficialCandidateRecoveryError("coverage_mismatch", "official hand-off coverage is empty or inconsistent")
        if _file_sha256(source_file, "source", _MAX_SOURCE_BYTES) != handoff.payload["source_sha256"]:
            raise OfficialCandidateRecoveryError("source_drift", "source differs from official hand-off")
        # Re-run the typed adapter input bridge before creating RecoveryStore state.
        inert = prepare_official_adapter_input_from_preflight(str(handoff_file), str(preflight_file), str(source_file), str(resolution_file))
        if inert.operator_manifest.get("candidate_id") != gate.get("candidate_id") or inert.operator_manifest.get("qualified_api") != gate.get("qualified_api"):
            raise OfficialCandidateRecoveryError("candidate_drift", "context gate candidate differs from hand-off")
    except OfficialCandidateRecoveryError:
        raise
    except (OfficialAdapterContextGateError, OfficialCandidateHandoffError, OfficialCandidatePreflightError, OfficialResolutionError, GapCompletionError, OSError, ValueError, TypeError) as exc:
        raise OfficialCandidateRecoveryError("invalid_input", "official recovery inputs failed validation") from exc
    if any(decision.route != INSTALLED_ADAPTER_ROUTE for decision in report.decisions):
        raise OfficialCandidateRecoveryError("unsupported_route", "official recovery accepts only installed ACLNN adapter candidates")
    specs = {(item.gap_id, item.qualified_api): item for item in preparation.specs}
    candidates = []
    for decision in report.decisions:
        spec = specs.get((decision.gap_id, decision.qualified_api))
        if spec is None:
            raise OfficialCandidateRecoveryError("coverage_mismatch", "official decision is not bound to current GapSpec")
        candidates.append(_candidate_refs(resolution_file, decision, report, handoff_file, preflight_file, spec))
    candidates = tuple(candidates)
    target = RecoveryTarget(report.target_soc, report.cann_version)
    source = RecoverySource(str(source_file), handoff.payload["source_sha256"])
    runtime = RecoveryRuntime(runtime_backend, runtime_audit_id, runtime_identity_sha256)
    binding = RecoveryBinding(
        run=RecoveryRun(run_id or "official-%s" % handoff.handoff_id[:24]),
        source=source,
        target=target,
        runtime=runtime,
        policy=recovery_policy_for_profile(RecoveryPolicyProfile.RECEIPT_ONLY),
    )
    store = RecoveryStore(state_dir)
    gap_ids = tuple(sorted(item.gap_id for item in candidates))
    try:
        status = store.create(binding, gap_ids=gap_ids)
    except RecoveryError:
        try:
            status = store.status()
        except RecoveryError as exc:
            raise OfficialCandidateRecoveryError("state_unavailable", "cannot create or load official recovery state") from exc
        if status.state.binding != binding or tuple(item.gap_id for item in status.state.gaps) != gap_ids:
            raise OfficialCandidateRecoveryError("state_binding_mismatch", "existing recovery state is bound to another run")
    output = Path(output_dir).absolute() if output_dir else Path(state_dir).absolute() / "official-candidates"
    artifact_refs = {}
    for candidate in candidates:
        artifact_refs[candidate.gap_id] = _write_candidate(output / (candidate.gap_id + ".json"), candidate)
    handoff_ref = {item.gap_id: EvidenceRef("official_handoff_%s" % item.gap_id, str(handoff_file), _file_sha256(handoff_file, "handoff", _MAX_RECEIPT_BYTES)) for item in candidates}
    preflight_ref = {item.gap_id: EvidenceRef("official_preflight_%s" % item.gap_id, str(preflight_file), _file_sha256(preflight_file, "preflight", _MAX_RECEIPT_BYTES)) for item in candidates}
    resolution_ref = {item.gap_id: EvidenceRef("official_resolution_%s" % item.gap_id, str(resolution_file), _file_sha256(resolution_file, "resolution", _MAX_RECEIPT_BYTES)) for item in candidates}
    stage_plan = (
        (RecoveryPhase.ANALYZED, StageReceiptKind.ANALYSIS_REPORT, handoff_ref, "official hand-off was validated"),
        (RecoveryPhase.GAP_PREPARED, StageReceiptKind.GAP_SPEC, preflight_ref, "official preflight was validated"),
        (RecoveryPhase.RESOLVED, StageReceiptKind.OFFICIAL_RESOLUTION_DECISION, resolution_ref, "official resolution was revalidated"),
        (RecoveryPhase.CANDIDATE_READY, StageReceiptKind.OPERATOR_CANDIDATE, artifact_refs, "official binary candidates are content-bound; adapter/model gates remain pending"),
    )
    appended = 0
    for phase, kind, refs, reason in stage_plan:
        if status.state.phase is phase or status.state.phase.value == "candidate_ready":
            continue
        status = _advance(store, status, phase, kind, refs, reason)
        appended += 1
    if status.state.phase is not RecoveryPhase.CANDIDATE_READY:
        raise OfficialCandidateRecoveryError("state_incomplete", "official recovery did not reach CANDIDATE_READY")
    return OfficialCandidateRecoveryResult(
        recovery=status,
        output_dir=str(output),
        candidates=candidates,
        appended_revisions=appended,
        side_effects={
            "model_invocation_performed": False,
            "network_access_performed": False,
            "system_write_performed": False,
            "asnumpy_import_performed": False,
            "verification_performed": False,
            "build_performed": False,
            "execution_performed": False,
            "activation_performed": False,
        },
    )


__all__ = [
    "SCHEMA",
    "STATUS",
    "OfficialCandidateRecoveryError",
    "OfficialCandidateRecoveryResult",
    "materialize_official_candidate_recovery",
]
