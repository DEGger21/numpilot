"""Explicit hand-off contract for installed ACLNN adapter candidates.

The ordinary ``script_api_gap`` path intentionally accepts only generatable
missing APIs.  An installed ACLNN candidate is a different route: this module
creates a small, hash-bound receipt that can be consumed by a future adapter
recovery stage without weakening the generic model gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    OfficialResolutionError,
    gap_spec_sha256,
    load_official_resolution_report,
)
from numpilot.gaps.pipeline import GapCompletionError, GapPreparation, prepare_gap_completion

SCHEMA = "numpilot.official-candidate-gap.v1"
SCHEMA_VERSION = 1
REQUEST_TYPE = "official_candidate_gap"
STATUS = "candidate_adapter_required"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _same_path(left: str, right: str) -> bool:
    return os.path.abspath(left) == os.path.abspath(right)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise OfficialCandidateHandoffError("invalid %s hash" % label)
    try:
        int(value, 16)
    except ValueError as exc:
        raise OfficialCandidateHandoffError("invalid %s hash" % label) from exc
    return value


class OfficialCandidateHandoffError(RuntimeError):
    """The candidate hand-off is stale, ambiguous, or unsafe."""


@dataclass(frozen=True)
class OfficialCandidateHandoff:
    """Validated, inert candidate-adapter hand-off receipt."""

    payload: Mapping[str, Any]

    @property
    def handoff_id(self) -> str:
        return str(self.payload["handoff_id"])

    @property
    def source_path(self) -> str:
        return str(self.payload["source_path"])

    @property
    def target_soc(self) -> str:
        return str(self.payload["target_soc"])

    @property
    def cann_version(self) -> str:
        return str(self.payload["cann_version"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _candidate_evidence(decision: Any) -> dict[str, Any]:
    if getattr(decision, "route", None) != INSTALLED_ADAPTER_ROUTE:
        raise OfficialCandidateHandoffError(
            "only installed_cann_adapter_candidate can enter this hand-off"
        )
    if getattr(decision, "activation_allowed", True) is not False:
        raise OfficialCandidateHandoffError("candidate activation must remain false")
    installed = getattr(decision, "installed_cann", None)
    if not isinstance(installed, Mapping):
        raise OfficialCandidateHandoffError("installed candidate evidence is missing")
    if installed.get("adapter_candidate") is not True:
        raise OfficialCandidateHandoffError("installed evidence is not an adapter candidate")
    if installed.get("missing_symbols") not in ([], None):
        raise OfficialCandidateHandoffError("candidate has unresolved ACLNN symbols")
    candidate = installed.get("candidate")
    header = installed.get("selected_header")
    library = installed.get("selected_library")
    symbols = installed.get("required_symbols")
    if not isinstance(candidate, Mapping) or not isinstance(header, Mapping) or not isinstance(library, Mapping):
        raise OfficialCandidateHandoffError("candidate/header/library evidence is incomplete")
    if not isinstance(symbols, list) or len(symbols) != 2 or any(
        not isinstance(item, str) or not item for item in symbols
    ):
        raise OfficialCandidateHandoffError("candidate must bind both ACLNN stages")
    header_sha = _hash(header.get("sha256"), "header")
    library_sha = _hash(library.get("sha256"), "library")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise OfficialCandidateHandoffError("candidate id is missing")
    return {
        "gap_id": decision.gap_id,
        "qualified_api": decision.qualified_api,
        "gap_spec_sha256": decision.gap_spec_sha256,
        "route": decision.route,
        "candidate_id": candidate_id,
        "required_symbols": list(symbols),
        "header_path": header.get("relative_path"),
        "header_sha256": header_sha,
        "library_path": library.get("relative_path"),
        "library_sha256": library_sha,
        "semantic_equivalence": installed.get("semantic_equivalence"),
    }


def _build_payload(
    source_path: Path,
    source: str,
    resolution_path: Path,
    report: Any,
    preparation: GapPreparation,
) -> dict[str, Any]:
    if report.status != "official_candidate_found":
        raise OfficialCandidateHandoffError(
            "official resolution status is not official_candidate_found"
        )
    if not report.decisions or any(
        item.route != INSTALLED_ADAPTER_ROUTE for item in report.decisions
    ):
        raise OfficialCandidateHandoffError(
            "hand-off requires only installed ACLNN adapter candidates"
        )
    if preparation.blocked:
        raise OfficialCandidateHandoffError(
            "source contains blocked or capability-audit-required calls"
        )
    spec_by_key = {
        (spec.gap_id, spec.qualified_api): spec for spec in preparation.specs
    }
    decisions = []
    for decision in report.decisions:
        spec = spec_by_key.get((decision.gap_id, decision.qualified_api))
        if spec is None or gap_spec_sha256(spec) != decision.gap_spec_sha256:
            raise OfficialCandidateHandoffError(
                "official decision is not bound to the current GapSpec"
            )
        decisions.append(_candidate_evidence(decision))
    source_sha = _sha256_text(source)
    if report.source_sha256 != source_sha or not _same_path(report.source_path, str(source_path)):
        raise OfficialCandidateHandoffError("official resolution is not bound to source")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_type": REQUEST_TYPE,
        "status": STATUS,
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "target_soc": report.target_soc,
        "cann_version": report.cann_version,
        "official_resolution_path": str(resolution_path),
        "official_resolution_sha256": _sha256_file(resolution_path),
        "resolution_id": report.resolution_id,
        "gap_specs": [spec.to_dict() for spec in preparation.specs],
        "decisions": decisions,
        "generated_code": False,
        "qualification_performed": False,
        "activation_performed": False,
        "network_access_performed": False,
        "system_write_performed": False,
    }
    payload["handoff_id"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def prepare_official_candidate_handoff(
    source_path: str,
    official_resolution_path: str,
    out_path: str,
) -> OfficialCandidateHandoff:
    """Create a write-once candidate hand-off after local revalidation."""

    source_file = Path(source_path).absolute()
    resolution_file = Path(official_resolution_path).absolute()
    output_file = Path(out_path).absolute()
    if not source_file.is_file() or not resolution_file.is_file():
        raise OfficialCandidateHandoffError("source and resolution must be regular files")
    if output_file.exists() or output_file.is_symlink():
        raise OfficialCandidateHandoffError("refusing to overwrite existing hand-off")
    try:
        source = source_file.read_text(encoding="utf-8")
        report = load_official_resolution_report(str(resolution_file))
        preparation = prepare_gap_completion(
            str(source_file),
            target_soc=report.target_soc,
            cann_version=report.cann_version,
        )
        report.validate_preparation(preparation)
        payload = _build_payload(source_file, source, resolution_file, report, preparation)
    except (OSError, UnicodeError, GapCompletionError, OfficialResolutionError) as exc:
        raise OfficialCandidateHandoffError(str(exc)) from exc
    output_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(str(output_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    return OfficialCandidateHandoff(payload)


def load_official_candidate_handoff(
    handoff_path: str,
    source_path: str,
    official_resolution_path: str,
) -> OfficialCandidateHandoff:
    """Reload and revalidate a candidate hand-off before adapter work."""

    path = Path(handoff_path).absolute()
    source_file = Path(source_path).absolute()
    resolution_file = Path(official_resolution_path).absolute()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise OfficialCandidateHandoffError("hand-off must be a JSON object")
        if data.get("schema") != SCHEMA or data.get("schema_version") != SCHEMA_VERSION:
            raise OfficialCandidateHandoffError("unsupported hand-off schema")
        if data.get("request_type") != REQUEST_TYPE or data.get("status") != STATUS:
            raise OfficialCandidateHandoffError("invalid hand-off status")
        for name in (
            "generated_code", "qualification_performed", "activation_performed",
            "network_access_performed", "system_write_performed",
        ):
            if data.get(name) is not False:
                raise OfficialCandidateHandoffError("unsafe hand-off flag: %s" % name)
        source = source_file.read_text(encoding="utf-8")
        if data.get("source_path") != str(source_file) or data.get("source_sha256") != _sha256_text(source):
            raise OfficialCandidateHandoffError("hand-off source binding is stale")
        if data.get("official_resolution_path") != str(resolution_file) or data.get("official_resolution_sha256") != _sha256_file(resolution_file):
            raise OfficialCandidateHandoffError("hand-off resolution binding is stale")
        report = load_official_resolution_report(str(resolution_file))
        preparation = prepare_gap_completion(
            str(source_file), target_soc=report.target_soc, cann_version=report.cann_version
        )
        report.validate_preparation(preparation)
        expected = _build_payload(source_file, source, resolution_file, report, preparation)
        if data != expected:
            raise OfficialCandidateHandoffError("hand-off content does not match current receipts")
        return OfficialCandidateHandoff(data)
    except OfficialCandidateHandoffError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, GapCompletionError, OfficialResolutionError) as exc:
        raise OfficialCandidateHandoffError(str(exc)) from exc


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "REQUEST_TYPE",
    "STATUS",
    "OfficialCandidateHandoff",
    "OfficialCandidateHandoffError",
    "prepare_official_candidate_handoff",
    "load_official_candidate_handoff",
]
