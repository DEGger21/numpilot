"""Hash-bound case-level performance evidence ingestion."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

_SCHEMA = "numpilot.performance-evidence-manifest"
_RECEIPT_TYPE = "numpilot_ordinary_script_benchmark"
_SCOPE = "steady_device_resident_marker_suffix"
_SHA256_CHARS = frozenset("0123456789abcdef")


class PerformanceEvidenceError(ValueError):
    """The performance evidence manifest or receipt is invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise PerformanceEvidenceError(f"{field} must be a lowercase SHA-256")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PerformanceEvidenceError(f"evidence path is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PerformanceEvidenceError(f"cannot read performance evidence: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise PerformanceEvidenceError("performance evidence must be a JSON object")
    return value


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\x00" in value:
        raise PerformanceEvidenceError("performance evidence path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PerformanceEvidenceError("performance evidence path escapes root")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise PerformanceEvidenceError("performance evidence path escapes root") from None
    return candidate


def _runtime_identity(receipt: Mapping[str, Any]) -> str:
    workers = receipt.get("workers")
    npu = workers.get("npu") if isinstance(workers, Mapping) else None
    if not isinstance(npu, Mapping):
        raise PerformanceEvidenceError("performance receipt has no NPU worker")
    before = npu.get("runtime_identity_before")
    after = npu.get("runtime_identity_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise PerformanceEvidenceError("performance receipt has no runtime identity pair")
    before_status = before.get("status")
    after_status = after.get("status")
    before_sha = _require_sha(before.get("identity_sha256"), "runtime identity")
    after_sha = _require_sha(after.get("identity_sha256"), "runtime identity")
    if before_status != "matched" or after_status != "matched" or before_sha != after_sha:
        raise PerformanceEvidenceError("performance runtime identity is not stable and matched")
    return before_sha


def _validated_ref(root: Path, sample_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    path = _relative_file(root, entry.get("path"))
    declared_sha = _require_sha(entry.get("sha256"), "performance receipt")
    actual_sha = _sha256(path)
    if actual_sha != declared_sha:
        raise PerformanceEvidenceError("performance receipt hash mismatch")
    receipt = _read_object(path)
    if receipt.get("receipt_type") == "numpilot_scoped_sample_benchmark":
        from numpilot.scoped_performance import SAMPLE_ID, validate_receipt
        if sample_id != SAMPLE_ID:
            raise PerformanceEvidenceError("scoped performance sample identity mismatch")
        try:
            validated = validate_receipt(root, receipt)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise PerformanceEvidenceError(str(exc)) from None
        selector = entry.get("selector", {})
        if not isinstance(selector, Mapping):
            raise PerformanceEvidenceError("performance selector is invalid")
        return {
            "schema": "numpilot.performance-evidence-ref", "schema_version": 1,
            "sample_id": sample_id,
            "path": path.relative_to(root.resolve()).as_posix(),
            "receipt_sha256": declared_sha,
            "receipt_type": "numpilot_scoped_sample_benchmark",
            "status": "passed", "performance_valid": True,
            "selector": dict(selector), "evidence_root": str(root.absolute()),
            **validated,
        }
    if receipt.get("schema_version") != 1 or receipt.get("receipt_type") != _RECEIPT_TYPE:
        raise PerformanceEvidenceError("performance receipt schema is invalid")
    if receipt.get("status") != "passed":
        raise PerformanceEvidenceError("performance receipt is not passed")
    performance = receipt.get("performance")
    if (
        not isinstance(performance, Mapping)
        or performance.get("performance_valid") is not True
        or performance.get("issues") not in (None, [])
    ):
        raise PerformanceEvidenceError("performance receipt is not performance-valid")
    gates = receipt.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(value is not True for value in gates.values()):
        raise PerformanceEvidenceError("performance receipt gates are not all passed")
    comparisons = receipt.get("comparisons")
    if not isinstance(comparisons, Mapping) or not comparisons or any(
        not isinstance(value, Mapping) or value.get("consistent") is not True
        for value in comparisons.values()
    ):
        raise PerformanceEvidenceError("performance receipt comparisons are not all consistent")
    boundary = receipt.get("evidence_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("steady_state") is not True or boundary.get("device_resident") is not True or boundary.get("end_to_end") is not False:
        raise PerformanceEvidenceError("performance receipt evidence boundary is invalid")
    protocol = receipt.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("scope") != _SCOPE:
        raise PerformanceEvidenceError("performance receipt scope is invalid")
    binding = receipt.get("binding")
    source = binding.get("source") if isinstance(binding, Mapping) else None
    source_sha = _require_sha(source.get("sha256") if isinstance(source, Mapping) else None, "performance source")
    runtime_sha = _runtime_identity(receipt)
    metrics = {
        name: performance.get(name)
        for name in (
            "median_speedup",
            "conservative_cpu_p10_over_npu_p90",
            "cpu_median_ms",
            "npu_median_ms",
        )
        if performance.get(name) is not None
    }
    selector = entry.get("selector", {})
    if not isinstance(selector, Mapping):
        raise PerformanceEvidenceError("performance selector is invalid")
    return {
        "schema": "numpilot.performance-evidence-ref",
        "schema_version": 1,
        "sample_id": sample_id,
        "path": path.relative_to(root.resolve()).as_posix(),
        "receipt_sha256": declared_sha,
        "receipt_type": _RECEIPT_TYPE,
        "status": "passed",
        "performance_valid": True,
        "source_sha256": source_sha,
        "migrated_sha256": (
            binding.get("migrated", {}).get("sha256")
            if isinstance(binding.get("migrated"), Mapping)
            else None
        ),
        "migration_report_sha256": (
            binding.get("migration_report", {}).get("sha256")
            if isinstance(binding.get("migration_report"), Mapping)
            else None
        ),
        "runtime_identity_sha256": runtime_sha,
        "scope": _SCOPE,
        "selector": dict(selector),
        "metrics": metrics,
    }


def _compatible_scopes(produced_ref, current_ref):
    from numpilot.scoped_performance import load_scope
    if not isinstance(current_ref, Mapping):
        raise PerformanceEvidenceError("current scoped qualification is required")
    try:
        produced = load_scope(produced_ref)
        current = load_scope(current_ref)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PerformanceEvidenceError(str(exc)) from None
    for field in ("sample_id", "source_sha256", "analysis_source_sha256",
                  "generated_source_sha256", "profile_sha256", "runtime_identity_sha256",
                  "overlay_sha256", "qualification_id", "required_ops"):
        if field not in produced or produced.get(field) != current.get(field):
            raise PerformanceEvidenceError("performance current scope mismatch: " + field)
    for field in ("baseline_qualification_id", "baseline_certificate_sha256"):
        if produced.get(field) != current.get(field):
            raise PerformanceEvidenceError("performance current baseline scope mismatch: " + field)


def verify_reference(ref, *, expected_sample_id=None, expected_scoped_qualification=None):
    """Revalidate a scoped performance reference, including its current case binding."""
    if not isinstance(ref, Mapping) or ref.get("receipt_type") != "numpilot_scoped_sample_benchmark":
        raise PerformanceEvidenceError("a scoped performance reference is required")
    if expected_sample_id is not None and ref.get("sample_id") != expected_sample_id:
        raise PerformanceEvidenceError("performance sample identity mismatch")
    raw_root = ref.get("evidence_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise PerformanceEvidenceError("performance evidence root is missing")
    root = Path(raw_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise PerformanceEvidenceError("performance evidence root is invalid")
    validated = _validated_ref(root, ref.get("sample_id"), {
        "path": ref.get("path"), "sha256": ref.get("receipt_sha256"),
        "selector": ref.get("selector", {}),
    })
    current = ref.get("current_scoped_qualification_ref")
    if current is not None:
        _compatible_scopes(validated.get("scoped_qualification_ref"), current)
        validated["current_scoped_qualification_ref"] = dict(current)
        validated["current_scoped_qualification_receipt_sha256"] = current.get("receipt_sha256")
    if expected_scoped_qualification is not None:
        if current != expected_scoped_qualification:
            raise PerformanceEvidenceError("performance reference is not bound to the current case scope")
        _compatible_scopes(validated.get("scoped_qualification_ref"), expected_scoped_qualification)
    if dict(ref) != validated:
        raise PerformanceEvidenceError("performance evidence reference does not match validated files")
    return validated


@dataclass(frozen=True)
class PerformanceEvidenceResolution:
    available: bool
    qualified: bool
    ref: dict[str, Any] | None
    reason: str


class PerformanceEvidenceCatalog:
    """Load a manifest of independently produced, hash-bound receipts."""

    def __init__(self, root: str | Path):
        candidate = Path(root).expanduser().absolute()
        if candidate.is_symlink() or not candidate.is_dir():
            raise PerformanceEvidenceError("performance evidence root is invalid")
        self.root = candidate
        manifest_path = candidate / "manifest.json"
        manifest = _read_object(manifest_path)
        if manifest.get("schema") != _SCHEMA or manifest.get("schema_version") != 1:
            raise PerformanceEvidenceError("performance evidence manifest schema is invalid")
        entries = manifest.get("entries")
        if not isinstance(entries, Mapping):
            raise PerformanceEvidenceError("performance evidence manifest entries are invalid")
        self._entries: dict[str, PerformanceEvidenceResolution] = {}
        for raw_sample_id, raw_entry in entries.items():
            sample_id = str(raw_sample_id)
            if not sample_id or not isinstance(raw_entry, Mapping):
                raise PerformanceEvidenceError("performance evidence manifest entry is invalid")
            try:
                ref = _validated_ref(candidate, sample_id, raw_entry)
            except PerformanceEvidenceError as exc:
                self._entries[sample_id] = PerformanceEvidenceResolution(True, False, None, str(exc))
            else:
                self._entries[sample_id] = PerformanceEvidenceResolution(True, False, ref, "")

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def resolve(
        self,
        sample_id: str,
        *,
        source_sha256: str,
        backend: str,
        strict_npu_qualified: bool | None,
        numerical_verified: bool,
        runtime_identity_sha256: str | None,
        scoped_qualification: Mapping[str, Any] | None = None,
    ) -> PerformanceEvidenceResolution:
        entry = self._entries.get(sample_id)
        if entry is None:
            return PerformanceEvidenceResolution(False, None, None, "no performance evidence entry")
        if entry.ref is None:
            return entry
        ref = entry.ref
        if ref.get("source_sha256") != source_sha256:
            return PerformanceEvidenceResolution(True, False, ref, "performance source hash does not match case")
        if backend != "npu" or strict_npu_qualified is not True or numerical_verified is not True:
            return PerformanceEvidenceResolution(True, False, ref, "case has not passed strict NPU and numerical gates")
        if ref.get("runtime_identity_sha256") != runtime_identity_sha256:
            return PerformanceEvidenceResolution(True, False, ref, "performance runtime identity does not match case")
        if ref.get("receipt_type") == "numpilot_scoped_sample_benchmark":
            try:
                _compatible_scopes(ref.get("scoped_qualification_ref"), scoped_qualification)
            except PerformanceEvidenceError as exc:
                return PerformanceEvidenceResolution(True, False, ref, str(exc))
            ref = {**ref, "current_scoped_qualification_ref": dict(scoped_qualification),
                   "current_scoped_qualification_receipt_sha256": scoped_qualification.get("receipt_sha256")}
        return PerformanceEvidenceResolution(True, True, ref, "performance evidence accepted")
