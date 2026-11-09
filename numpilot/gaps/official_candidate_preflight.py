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


__all__ = ["SCHEMA", "STATUS", "OfficialCandidatePreflightError", "preflight_official_candidate"]
