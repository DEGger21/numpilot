"""Read-only preflight for an installed official ACLNN candidate hand-off.

This stage re-resolves the pinned source against the current CANN inventory and
re-hashes the selected header/library. It publishes an inert receipt for the
future recovery/materializer stage; it never creates a RecoveryStore state,
invokes a model, imports AsNumpy, builds, writes system OPP, or activates.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from numpilot.gaps.official_candidate_handoff import (
    OfficialCandidateHandoffError,
    load_official_candidate_handoff,
)
from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    OfficialResolutionError,
    load_official_resolution_report,
    resolve_official_operators,
)
from numpilot.gaps.pipeline import GapCompletionError, prepare_gap_completion

SCHEMA = "numpilot.official-candidate-preflight.v1"
STATUS = "ready_for_candidate_materialization"
_MAX_HEADER_BYTES = 2 * 1024 * 1024
_MAX_LIBRARY_BYTES = 512 * 1024 * 1024
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024


class OfficialCandidatePreflightError(RuntimeError):
    """The official candidate hand-off failed current-environment recheck."""


def _sha256_file(path: Path, maximum: int, label: str) -> tuple[str, int]:
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise OfficialCandidatePreflightError("cannot open %s" % label) from exc
    if not resolved.is_file() or stat.st_size <= 0 or stat.st_size > maximum:
        raise OfficialCandidatePreflightError("invalid %s file" % label)
    digest = hashlib.sha256()
    total = 0
    try:
        with resolved.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise OfficialCandidatePreflightError("%s exceeds size limit" % label)
                digest.update(chunk)
    except OSError as exc:
        raise OfficialCandidatePreflightError("cannot read %s" % label) from exc
    if total != stat.st_size:
        raise OfficialCandidatePreflightError("%s changed while being read" % label)
    return digest.hexdigest(), total


def _safe_under(root: str, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise OfficialCandidatePreflightError("invalid %s relative path" % label)
    try:
        root_path = Path(root).resolve(strict=True)
        candidate = (root_path / relative).resolve(strict=True)
    except OSError as exc:
        raise OfficialCandidatePreflightError("invalid %s path" % label) from exc
    if candidate != root_path and root_path not in candidate.parents:
        raise OfficialCandidatePreflightError("%s escapes pinned CANN root" % label)
    return candidate


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OfficialCandidatePreflightError("refusing to overwrite preflight receipt")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise OfficialCandidatePreflightError("preflight receipt is too large")
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise OfficialCandidatePreflightError("cannot publish preflight receipt") from exc


def preflight_official_candidate(
    handoff_path: str,
    source_path: str,
    official_resolution_path: str,
    out_path: str,
) -> Mapping[str, Any]:
    """Recheck one hand-off and publish an inert materialization preflight."""
    try:
        handoff = load_official_candidate_handoff(
            handoff_path, source_path, official_resolution_path
        )
        report = load_official_resolution_report(official_resolution_path)
        preparation = prepare_gap_completion(
            source_path,
            target_soc=report.target_soc,
            cann_version=report.cann_version,
        )
        report.validate_preparation(preparation)
        fresh = resolve_official_operators(
            preparation,
            cann_root=report.cann_root,
            ops_nn_root=report.ops_nn_root,
            accept_cann_osl=report.cann_osl_accepted,
        )
        fresh.validate_preparation(preparation)
    except (OfficialCandidateHandoffError, OfficialResolutionError, GapCompletionError, OSError, ValueError, TypeError) as exc:
        raise OfficialCandidatePreflightError(str(exc)) from exc
    if fresh.resolution_id != report.resolution_id:
        raise OfficialCandidatePreflightError("current official resolution differs from hand-off")
    if fresh.status != "official_candidate_found" or not fresh.decisions:
        raise OfficialCandidatePreflightError("current route is not official_candidate_found")
    by_gap = {spec.gap_id: spec for spec in preparation.specs}
    decisions = []
    for decision in fresh.decisions:
        if decision.route != INSTALLED_ADAPTER_ROUTE:
            raise OfficialCandidatePreflightError("preflight accepts installed ACLNN candidates only")
        installed = decision.installed_cann
        header_evidence = installed.get("selected_header")
        library_evidence = installed.get("selected_library")
        if not isinstance(header_evidence, Mapping) or not isinstance(library_evidence, Mapping):
            raise OfficialCandidatePreflightError("selected header/library evidence is missing")
        header_path = _safe_under(report.cann_root, header_evidence.get("relative_path"), "header")
        library_path = _safe_under(report.cann_root, library_evidence.get("relative_path"), "library")
        header_sha, header_size = _sha256_file(header_path, _MAX_HEADER_BYTES, "header")
        library_sha, library_size = _sha256_file(library_path, _MAX_LIBRARY_BYTES, "library")
        if header_sha != header_evidence.get("sha256") or library_sha != library_evidence.get("sha256"):
            raise OfficialCandidatePreflightError("selected header/library hash drift")
        if decision.gap_id not in by_gap:
            raise OfficialCandidatePreflightError("official decision is not bound to a current GapSpec")
        decisions.append({
            "gap_id": decision.gap_id,
            "qualified_api": decision.qualified_api,
            "candidate_id": installed.get("candidate", {}).get("candidate_id"),
            "gap_spec_sha256": decision.gap_spec_sha256,
            "required_symbols": list(installed.get("required_symbols", [])),
            "header_relative_path": header_evidence.get("relative_path"),
            "header_sha256": header_sha,
            "header_size": header_size,
            "library_relative_path": library_evidence.get("relative_path"),
            "library_sha256": library_sha,
            "library_size": library_size,
        })
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": STATUS,
        "handoff_id": handoff.handoff_id,
        "source_path": str(Path(source_path).absolute()),
        "source_sha256": handoff.payload["source_sha256"],
        "official_resolution_path": str(Path(official_resolution_path).absolute()),
        "resolution_id": fresh.resolution_id,
        "target_soc": fresh.target_soc,
        "cann_version": fresh.cann_version,
        "decisions": decisions,
        "required_next_gate": "materialize_candidate_recovery",
        "subprocess_performed": True,
        "model_invocation_performed": False,
        "network_access_performed": False,
        "system_write_performed": False,
        "asnumpy_import_performed": False,
        "build_performed": False,
        "execution_performed": False,
        "activation_performed": False,
    }
    out = Path(out_path).absolute()
    _write_once(out, payload)
    return payload


def prepare_official_adapter_input_from_preflight(
    handoff_path: str,
    preflight_path: str,
    source_path: str,
    official_resolution_path: str,
):
    """Create an inert official adapter input from a validated preflight.

    This is an in-memory bridge only. It does not create a RecoveryStore state,
    publish a candidate artifact, open the model gate, import AsNumpy, or write
    an overlay. A future recovery stage must persist the typed candidate and
    advance the state machine before any adapter generation is allowed.
    """

    from numpilot.gaps.official_adapter import (
        OFFICIAL_CANDIDATE_MAPPING_REF,
        OFFICIAL_REQUIRED_SYMBOLS_REF,
        OFFICIAL_SELECTED_HEADER_REF,
        OFFICIAL_SELECTED_LIBRARY_REF,
        OFFICIAL_SEMANTIC_REVIEW_REF,
        OfficialCandidateAdapterInput,
        prepare_official_candidate_adapter_input,
    )
    from numpilot.gaps.official_resolution import gap_spec_sha256
    from numpilot.gaps.recovery import (
        EvidenceRef,
        LicenseBinding,
        OperatorCandidateOrigin,
        create_operator_candidate,
    )

    def canonical_sha(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    try:
        handoff = load_official_candidate_handoff(
            handoff_path, source_path, official_resolution_path
        )
        report = load_official_resolution_report(official_resolution_path)
        preparation = prepare_gap_completion(
            source_path,
            target_soc=report.target_soc,
            cann_version=report.cann_version,
        )
        report.validate_preparation(preparation)
        preflight_file = Path(preflight_path).absolute()
        preflight = json.loads(preflight_file.read_text(encoding="utf-8"))
    except (OfficialCandidateHandoffError, OfficialResolutionError, GapCompletionError,
            OSError, UnicodeError, ValueError, TypeError) as exc:
        raise OfficialCandidatePreflightError("cannot load adapter preflight inputs") from exc

    if not isinstance(preflight, Mapping):
        raise OfficialCandidatePreflightError("adapter preflight must be a JSON object")
    if (
        preflight.get("schema") != SCHEMA
        or preflight.get("schema_version") != 1
        or preflight.get("status") != STATUS
        or preflight.get("handoff_id") != handoff.handoff_id
        or preflight.get("resolution_id") != report.resolution_id
        or preflight.get("source_path") != str(Path(source_path).absolute())
        or preflight.get("source_sha256") != handoff.payload["source_sha256"]
        or preflight.get("target_soc") != report.target_soc
        or preflight.get("cann_version") != report.cann_version
        or preflight.get("required_next_gate") != "materialize_candidate_recovery"
    ):
        raise OfficialCandidatePreflightError("adapter preflight binding is stale")
    for name in (
        "model_invocation_performed", "network_access_performed",
        "system_write_performed", "asnumpy_import_performed", "build_performed",
        "execution_performed", "activation_performed",
    ):
        if preflight.get(name) is not False:
            raise OfficialCandidatePreflightError("unsafe preflight flag: %s" % name)

    if len(preparation.specs) != 1 or len(report.decisions) != 1:
        raise OfficialCandidatePreflightError(
            "adapter input bridge currently requires exactly one official candidate"
        )
    spec = preparation.specs[0]
    decision = report.decisions[0]
    if decision.route != INSTALLED_ADAPTER_ROUTE or decision.gap_spec_sha256 != gap_spec_sha256(spec):
        raise OfficialCandidatePreflightError("official candidate is not bound to the current GapSpec")
    installed = decision.installed_cann
    candidate_mapping = installed.get("candidate")
    if not isinstance(candidate_mapping, Mapping):
        raise OfficialCandidatePreflightError("official candidate mapping is missing")
    required_symbols = installed.get("required_symbols")
    header = installed.get("selected_header")
    library = installed.get("selected_library")
    if not isinstance(required_symbols, list) or not isinstance(header, Mapping) or not isinstance(library, Mapping):
        raise OfficialCandidatePreflightError("official candidate ABI evidence is incomplete")

    candidate_refs = (
        EvidenceRef(
            OFFICIAL_CANDIDATE_MAPPING_REF,
            str(Path(official_resolution_path).absolute()) + "#decisions/%s/installed_cann/candidate" % decision.gap_id,
            str(installed["candidate_mapping_sha256"]),
        ),
        EvidenceRef(
            OFFICIAL_SELECTED_HEADER_REF,
            str(Path(official_resolution_path).absolute()) + "#decisions/%s/installed_cann/selected_header" % decision.gap_id,
            str(header["sha256"]),
        ),
        EvidenceRef(
            OFFICIAL_SELECTED_LIBRARY_REF,
            str(Path(official_resolution_path).absolute()) + "#decisions/%s/installed_cann/selected_library" % decision.gap_id,
            str(library["sha256"]),
        ),
        EvidenceRef(
            OFFICIAL_REQUIRED_SYMBOLS_REF,
            str(Path(official_resolution_path).absolute()) + "#decisions/%s/installed_cann/required_symbols" % decision.gap_id,
            canonical_sha({"required_symbols": list(required_symbols)}),
        ),
        EvidenceRef(
            OFFICIAL_SEMANTIC_REVIEW_REF,
            str(Path(official_resolution_path).absolute()) + "#decisions/%s/installed_cann/candidate/semantic_review" % decision.gap_id,
            canonical_sha({"semantic_review": list(candidate_mapping.get("semantic_review", []))}),
        ),
    )
    provenance_refs = (
        EvidenceRef(
            "official_resolution",
            str(Path(official_resolution_path).absolute()),
            canonical_sha(report.to_dict()),
        ),
        EvidenceRef(
            "official_candidate_preflight",
            str(Path(preflight_file).absolute()),
            _sha256_file(preflight_file, _MAX_RECEIPT_BYTES, "preflight")[0],
        ),
    )
    license_ref = EvidenceRef(
        "external_installed_cann_terms",
        str(Path(official_resolution_path).absolute()) + "#external-installed-cann-terms",
        canonical_sha({
            "identifier": "LicenseRef-External-Installed-CANN-Terms",
            "accepted": False,
            "cann_root": report.cann_root,
            "cann_version": report.cann_version,
            "target_soc": report.target_soc,
            "gap_id": decision.gap_id,
            "qualified_api": decision.qualified_api,
        }),
    )
    candidate = create_operator_candidate(
        origin_kind=OperatorCandidateOrigin.OFFICIAL_BINARY,
        gap_id=spec.gap_id,
        qualified_api=spec.qualified_api,
        gap_spec_ref=EvidenceRef(
            "gap_spec",
            str(Path(handoff_path).absolute()) + "#gap_specs/%s" % spec.gap_id,
            gap_spec_sha256(spec),
        ),
        resolution_decision_ref=EvidenceRef(
            "resolution_decision",
            str(Path(official_resolution_path).absolute()) + "#decisions/%s" % decision.gap_id,
            canonical_sha(decision.to_dict()),
        ),
        target_soc=spec.target_soc,
        cann_version=spec.cann_version,
        license=LicenseBinding(
            "LicenseRef-External-Installed-CANN-Terms", False, license_ref
        ),
        provenance_refs=provenance_refs,
        candidate_refs=candidate_refs,
    )
    try:
        prepared = prepare_official_candidate_adapter_input(candidate, spec, decision)
    except Exception as exc:
        raise OfficialCandidatePreflightError("official adapter input contract failed") from exc
    if not isinstance(prepared, OfficialCandidateAdapterInput) or prepared.model_invocation_allowed:
        raise OfficialCandidatePreflightError("adapter input bridge opened the model gate")
    return prepared


__all__ = [
    "SCHEMA",
    "STATUS",
    "OfficialCandidatePreflightError",
    "preflight_official_candidate",
    "prepare_official_adapter_input_from_preflight",
]
