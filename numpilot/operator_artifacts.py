"""Content-addressed storage for model-generated operator artifacts.

The benchmark runner may visit 150 samples, but an operator candidate is stored
once per semantic/operator/environment key and benchmark cases keep small
bindings to that record.  Candidates, qualification receipts, and activation
receipts are immutable JSON/source artifacts; the index is atomically rebuilt
so a run can be replayed without contacting a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

SCHEMA_VERSION = 1
STORE_SCHEMA = "numpilot.operator-artifact-store"
CANDIDATE_SCHEMA = "numpilot.operator-candidate"
QUALIFICATION_SCHEMA = "numpilot.operator-qualification"
ACTIVATION_SCHEMA = "numpilot.operator-activation"
BINDING_SCHEMA = "numpilot.operator-sample-binding"
RUNTIME_QUALIFICATION_SCHEMA = "numpilot.runtime-qualification"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SECRET_KEYS = {
    "token", "api_key", "apikey", "authorization", "secret", "password",
    "endpoint", "base_url", "raw_response", "response_text", "prompt",
}


class OperatorArtifactStoreError(RuntimeError):
    """A store write would be unsafe or violates an artifact dependency."""


@dataclass(frozen=True)
class OperatorKey:
    operator_name: str
    semantic_signature: str
    gap_kind: str = ""
    target_soc: str = ""
    cann_version: str = ""
    runtime_identity_sha256: str = ""
    asnumpy_baseline_sha256: str = ""
    api_catalog_sha256: str = ""
    cannbot_bundle_sha256: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "operator_name": self.operator_name,
            "semantic_signature": self.semantic_signature,
            "gap_kind": self.gap_kind,
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "asnumpy_baseline_sha256": self.asnumpy_baseline_sha256,
            "api_catalog_sha256": self.api_catalog_sha256,
            "cannbot_bundle_sha256": self.cannbot_bundle_sha256,
        }

    @property
    def operator_id(self) -> str:
        return _sha256(_canonical(self.to_dict()))


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    operator_id: str
    artifact_id: str
    receipt_path: str
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    source_artifact_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "kind": self.kind,
            "operator_id": self.operator_id,
            "artifact_id": self.artifact_id,
            "receipt_path": self.receipt_path,
        }
        if self.source_path is not None:
            value["source_path"] = self.source_path
        if self.source_sha256 is not None:
            value["source_sha256"] = self.source_sha256
        if self.source_artifact_id is not None:
            value["source_artifact_id"] = self.source_artifact_id
        return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise OperatorArtifactStoreError("artifact metadata is not canonical JSON") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise OperatorArtifactStoreError(f"{label} is invalid")
    if _SAFE.fullmatch(value):
        return value
    # Sample names may contain '/' (for example group/case).  Normalize them
    # deterministically while keeping the original in the binding receipt.
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not normalized:
        raise OperatorArtifactStoreError(f"{label} is invalid")
    return normalized[:96] + "-" + _sha256(value.encode("utf-8"))[:16]


def _validate_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OperatorArtifactStoreError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_source_id(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not _SOURCE_ID.fullmatch(value):
        raise OperatorArtifactStoreError(f"{label} is invalid")
    return value


def _reject_sensitive(value: Any, path: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_lower = str(key).lower()
            if key_lower in _SECRET_KEYS or any(token in key_lower for token in ("api_token", "access_token", "private_key")):
                raise OperatorArtifactStoreError(f"sensitive field is not allowed in {path}.{key}")
            _reject_sensitive(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive(child, f"{path}[{index}]")


def _stable_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(receipt)
    for key in ("created_at", "stored_at", "generated_at", "updated_at"):
        result.pop(key, None)
    _reject_sensitive(result)
    _canonical(result)
    return result


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o600, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise OperatorArtifactStoreError(f"refusing symlink artifact path: {path}")
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise OperatorArtifactStoreError(f"immutable artifact already exists with different content: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate_runtime_qualification_scope(sample_id: str, receipt: Mapping[str, Any]):
    """Revalidate an explicit sample scope; never upgrade a global identity.

    This helper is shared by store insertion and offline store auditing. A
    scope label alone cannot replace the existing matched-runtime gate.
    """
    ref = receipt.get("runtime_qualification_scope")
    if ref is None:
        return None
    if (receipt.get("backend") != "npu"
            or receipt.get("strict_npu_qualified") is not True
            or receipt.get("numerical_verified") is not True
            or receipt.get("runtime_identity_status") not in {"matched", "audit_required"}):
        raise OperatorArtifactStoreError("scoped qualification requires strict numerical NPU evidence")
    _validate_sha(receipt.get("source_sha256"), "scoped source_sha256")
    _validate_sha(receipt.get("runtime_identity_sha256"), "runtime_identity_sha256")
    from numpilot.scoped_qualification import validate_scoped_qualification
    try:
        scope = validate_scoped_qualification(
            ref, expected_sample_id=sample_id,
            expected_source_sha256=receipt["source_sha256"],
            expected_runtime_identity_sha256=receipt["runtime_identity_sha256"],
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise OperatorArtifactStoreError("scoped qualification rejected: " + str(exc)) from exc
    if scope["runtime_identity_status"] != receipt["runtime_identity_status"]:
        raise OperatorArtifactStoreError("scoped qualification cannot promote global runtime status")
    return scope


def validate_scoped_performance_reference(sample_id: str, receipt: Mapping[str, Any]) -> None:
    """A scoped runtime fact does not by itself establish performance evidence."""
    if receipt.get("performance_qualified") is not True:
        return
    if receipt.get("runtime_qualification_scope") is None:
        return
    evidence = receipt.get("performance_evidence")
    if (not isinstance(evidence, Mapping)
            or evidence.get("identity_scope") != "official_overlay_callsite"
            or evidence.get("receipt_type") != "numpilot_scoped_sample_benchmark"):
        raise OperatorArtifactStoreError("scoped performance requires separate exact-sample evidence")
    from numpilot.performance_evidence import verify_reference
    try:
        verified = verify_reference(evidence, expected_sample_id=sample_id,
            expected_scoped_qualification=receipt["runtime_qualification_scope"])
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise OperatorArtifactStoreError("scoped performance evidence rejected: " + str(exc)) from exc
    if (verified.get("runtime_identity_sha256") != receipt.get("runtime_identity_sha256")
            or verified.get("source_sha256") != receipt.get("source_sha256")):
        raise OperatorArtifactStoreError("scoped performance sample/runtime binding mismatch")


class OperatorArtifactStore:
    """Persistent content-addressed operator artifacts for one benchmark run.

    Layout::

        <root>/store.json
        <root>/operators/<operator_id>/operator.json
        <root>/operators/<operator_id>/candidates/<candidate_id>/
        <root>/operators/<operator_id>/qualifications/<qualification_id>/
        <root>/operators/<operator_id>/activations/<activation_id>/
        <root>/samples/<sample_id>-<operator_id>.json

    The root should be the benchmark run output directory (for example
    ``runs/phase15/full``); copying it is sufficient for offline replay.
    """

    def __init__(self, root: os.PathLike[str] | str):
        self.root = Path(root).expanduser().absolute()
        if self.root.exists() and self.root.is_symlink():
            raise OperatorArtifactStoreError("artifact store root must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.operators = self.root / "operators"
        self.samples = self.root / "samples"
        self.runtime_qualifications = self.root / "runtime-qualifications"
        self.operators.mkdir(mode=0o700, exist_ok=True)
        self.samples.mkdir(mode=0o700, exist_ok=True)
        self.runtime_qualifications.mkdir(mode=0o700, exist_ok=True)
        self.store_path = self.root / "store.json"
        self.lock_path = self.root / ".store.lock"
        initial = {
            "schema": STORE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "artifact_policy": {
                "candidate_status": "generated_unverified",
                "activation_scope": "run",
                "upstream_modified": False,
                "model_response_persisted": False,
            },
        }
        if not self.store_path.exists():
            _atomic_bytes(self.store_path, _json(initial))
        else:
            loaded = self._read_json(self.store_path)
            if loaded.get("schema") != STORE_SCHEMA or loaded.get("schema_version") != SCHEMA_VERSION:
                raise OperatorArtifactStoreError("unsupported artifact store schema")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise OperatorArtifactStoreError(f"artifact receipt is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OperatorArtifactStoreError(f"invalid artifact receipt: {path}") from exc
        if not isinstance(payload, dict):
            raise OperatorArtifactStoreError(f"artifact receipt must be an object: {path}")
        return payload

    def _operator_dir(self, operator_id: str) -> Path:
        _validate_sha(operator_id, "operator_id")
        return self.operators / operator_id

    def ensure_operator(self, key: OperatorKey) -> str:
        if not key.operator_name or not key.semantic_signature:
            raise OperatorArtifactStoreError("operator name and semantic signature are required")
        for field_name in ("runtime_identity_sha256", "asnumpy_baseline_sha256", "api_catalog_sha256", "cannbot_bundle_sha256"):
            value = getattr(key, field_name)
            if value:
                _validate_sha(value, field_name)
        operator_id = key.operator_id
        directory = self._operator_dir(operator_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema": "numpilot.operator-record",
            "schema_version": SCHEMA_VERSION,
            "operator_id": operator_id,
            "key": key.to_dict(),
            "candidates": [],
            "qualifications": [],
            "activations": [],
            "latest": {"candidate": None, "qualification": None, "activation": None},
        }
        path = directory / "operator.json"
        if path.exists():
            existing = self._read_json(path)
            if existing.get("operator_id") != operator_id or existing.get("key") != key.to_dict():
                raise OperatorArtifactStoreError("operator id collision or key mismatch")
        else:
            _atomic_bytes(path, _json(payload))
        return operator_id

    def put_candidate(
        self,
        key: OperatorKey,
        source: bytes | str,
        receipt: Mapping[str, Any],
        *,
        source_name: str = "operator-draft.cpp",
    ) -> ArtifactRef:
        operator_id = self.ensure_operator(key)
        source_bytes = source.encode("utf-8") if isinstance(source, str) else bytes(source)
        if not source_bytes or len(source_bytes) > 2 * 1024 * 1024:
            raise OperatorArtifactStoreError("candidate source is empty or too large")
        source_sha = _sha256(source_bytes)
        stable = _stable_receipt(receipt)
        source_artifact_id = stable.get("artifact_id") or stable.get("source_artifact_id")
        if source_artifact_id is not None:
            if not isinstance(source_artifact_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", source_artifact_id):
                raise OperatorArtifactStoreError("source artifact_id is invalid")
        candidate_id = _sha256(_canonical({
            "operator_id": operator_id,
            "source_sha256": source_sha,
            "source_name": source_name,
            "receipt": stable,
        }))
        directory = self._operator_dir(operator_id) / "candidates" / candidate_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        source_name = _safe_component(Path(source_name).name, "source_name")
        source_path = directory / source_name
        _atomic_bytes(source_path, source_bytes)
        stored = {
            "schema": CANDIDATE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "operator_id": operator_id,
            "candidate_id": candidate_id,
            "status": "generated_unverified",
            "compiled": False,
            "activated": False,
            "source_name": source_name,
            "source_sha256": source_sha,
            "source_bytes": len(source_bytes),
            "source_artifact_id": source_artifact_id,
            "stored_at": _now(),
            "receipt": stable,
        }
        receipt_path = directory / "candidate.json"
        if receipt_path.exists():
            existing = self._read_json(receipt_path)
            comparable_existing = dict(existing)
            comparable_stored = dict(stored)
            comparable_existing.pop("stored_at", None)
            comparable_stored.pop("stored_at", None)
            if comparable_existing != comparable_stored:
                raise OperatorArtifactStoreError(
                    f"immutable artifact already exists with different content: {receipt_path}"
                )
            stored = existing
        else:
            _atomic_bytes(receipt_path, _json(stored))
        self._append_operator(operator_id, "candidates", candidate_id, "candidate", candidate_id)
        return ArtifactRef("candidate", operator_id, candidate_id, str(receipt_path), str(source_path), source_sha, source_artifact_id)

    def put_candidate_tree(
        self, key: OperatorKey, root: os.PathLike[str] | str, receipt: Mapping[str, Any]
    ) -> ArtifactRef:
        """Import an immutable generated-draft workspace into the store.

        The signed provider manifest remains outside the imported payload, while
        its stable artifact id is preserved as ``source_artifact_id``.  This
        keeps the store content-addressed without rewriting the provider's
        immutable draft identity.
        """
        workspace = Path(root).expanduser().absolute()
        if workspace.is_symlink() or not workspace.is_dir():
            raise OperatorArtifactStoreError("candidate workspace must be a regular directory")
        files: dict[str, bytes] = {}
        total = 0
        for path in sorted(workspace.rglob("*")):
            if path.is_symlink():
                raise OperatorArtifactStoreError("candidate workspace contains a symlink")
            if not path.is_file():
                continue
            relative = path.relative_to(workspace)
            if relative.is_absolute() or ".." in relative.parts:
                raise OperatorArtifactStoreError("candidate workspace path escapes root")
            rel = relative.as_posix()
            if rel == "operator-artifact.json" or any(part.startswith(".") for part in relative.parts):
                continue
            payload = path.read_bytes()
            if len(files) >= 256 or total + len(payload) > 2 * 1024 * 1024:
                raise OperatorArtifactStoreError("candidate workspace exceeds storage limits")
            total += len(payload)
            files[rel] = payload
        if not files:
            raise OperatorArtifactStoreError("candidate workspace has no importable files")

        operator_id = self.ensure_operator(key)
        stable = _stable_receipt(receipt)
        file_hashes = {name: _sha256(payload) for name, payload in files.items()}
        source_artifact_id = stable.get("artifact_id") or stable.get("source_artifact_id")
        if source_artifact_id is not None and (
            not isinstance(source_artifact_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", source_artifact_id)
        ):
            raise OperatorArtifactStoreError("source artifact_id is invalid")
        candidate_id = _sha256(_canonical({
            "operator_id": operator_id,
            "files": file_hashes,
            "receipt": stable,
        }))
        directory = self._operator_dir(operator_id) / "candidates" / candidate_id
        files_dir = directory / "files"
        for rel, payload in files.items():
            target = files_dir / rel
            if target.is_symlink():
                raise OperatorArtifactStoreError("refusing symlink candidate file")
            _atomic_bytes(target, payload)

        manifest_payload = _json(file_hashes)
        manifest_path = directory / "files-manifest.json"
        _atomic_bytes(manifest_path, manifest_payload)
        stored = {
            "schema": CANDIDATE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "operator_id": operator_id,
            "candidate_id": candidate_id,
            "status": "generated_unverified",
            "compiled": False,
            "activated": False,
            "source_kind": "workspace",
            "source_name": "files-manifest.json",
            "source_sha256": _sha256(manifest_payload),
            "source_bytes": total,
            "source_artifact_id": source_artifact_id,
            "files_sha256": file_hashes,
            "stored_at": _now(),
            "receipt": stable,
        }
        receipt_path = directory / "candidate.json"
        if receipt_path.exists():
            existing = self._read_json(receipt_path)
            comparable_existing = dict(existing)
            comparable_stored = dict(stored)
            comparable_existing.pop("stored_at", None)
            comparable_stored.pop("stored_at", None)
            if comparable_existing != comparable_stored:
                raise OperatorArtifactStoreError(
                    f"immutable artifact already exists with different content: {receipt_path}"
                )
            stored = existing
        else:
            _atomic_bytes(receipt_path, _json(stored))
        self._append_operator(operator_id, "candidates", candidate_id, "candidate", candidate_id)
        return ArtifactRef(
            "candidate",
            operator_id,
            candidate_id,
            str(receipt_path),
            str(files_dir),
            _sha256(manifest_payload),
            source_artifact_id,
        )

    def put_qualification(
        self,
        operator_id: str,
        candidate_id: str,
        receipt: Mapping[str, Any],
    ) -> ArtifactRef:
        candidate = self._find_artifact(operator_id, "candidates", candidate_id, "candidate.json")
        stable = _stable_receipt(receipt)
        if stable.get("candidate_id") not in {None, candidate_id}:
            raise OperatorArtifactStoreError("qualification candidate_id does not match")
        status = stable.get("status", "failed")
        if status not in {"qualified", "failed", "blocked", "rejected"}:
            raise OperatorArtifactStoreError("qualification status is invalid")
        if stable.get("interface_available") is True and status != "qualified":
            raise OperatorArtifactStoreError("interface_available requires qualified status")
        if status == "qualified" and stable.get("interface_available") is not True:
            raise OperatorArtifactStoreError("qualified status requires interface_available=true")
        source_artifact_id = _validate_source_id(
            stable.get("qualification_id") or stable.get("source_artifact_id"),
            "source qualification id",
        )
        qualification_id = _sha256(_canonical({
            "operator_id": operator_id,
            "candidate_id": candidate_id,
            "receipt": stable,
        }))
        directory = self._operator_dir(operator_id) / "qualifications" / qualification_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        stored = {
            "schema": QUALIFICATION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "operator_id": operator_id,
            "candidate_id": candidate_id,
            "qualification_id": qualification_id,
            "source_artifact_id": source_artifact_id,
            "status": status,
            "stored_at": _now(),
            "receipt": stable,
        }
        receipt_path = directory / "qualification.json"
        _atomic_bytes(receipt_path, _json(stored))
        self._append_operator(operator_id, "qualifications", qualification_id, "qualification", qualification_id)
        return ArtifactRef(
            "qualification", operator_id, qualification_id, str(receipt_path),
            source_artifact_id=source_artifact_id,
        )

    def put_activation(
        self,
        operator_id: str,
        qualification_id: str,
        receipt: Mapping[str, Any],
    ) -> ArtifactRef:
        qualification = self._find_artifact(operator_id, "qualifications", qualification_id, "qualification.json")
        if qualification.get("status") != "qualified":
            raise OperatorArtifactStoreError("activation requires a qualified artifact")
        stable = _stable_receipt(receipt)
        if stable.get("qualification_id") not in {None, qualification_id}:
            raise OperatorArtifactStoreError("activation qualification_id does not match")
        status = stable.get("status", "failed")
        if status not in {"activated", "failed", "blocked"}:
            raise OperatorArtifactStoreError("activation status is invalid")
        if status == "activated":
            if stable.get("activated") is not True or stable.get("upstream_modified") is not False:
                raise OperatorArtifactStoreError("activation must be explicit and upstream_modified=false")
        source_artifact_id = _validate_source_id(
            stable.get("activation_id") or stable.get("source_artifact_id"),
            "source activation id",
        )
        activation_id = _sha256(_canonical({
            "operator_id": operator_id,
            "qualification_id": qualification_id,
            "receipt": stable,
        }))
        directory = self._operator_dir(operator_id) / "activations" / activation_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        stored = {
            "schema": ACTIVATION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "operator_id": operator_id,
            "qualification_id": qualification_id,
            "activation_id": activation_id,
            "source_artifact_id": source_artifact_id,
            "scope": "run",
            "status": status,
            "stored_at": _now(),
            "receipt": stable,
        }
        receipt_path = directory / "activation.json"
        _atomic_bytes(receipt_path, _json(stored))
        self._append_operator(operator_id, "activations", activation_id, "activation", activation_id)
        return ArtifactRef(
            "activation", operator_id, activation_id, str(receipt_path),
            source_artifact_id=source_artifact_id,
        )

    def put_runtime_qualification(
        self, sample_id: str, receipt: Mapping[str, Any]
    ) -> ArtifactRef:
        """Store one case-scoped runtime qualification gate.

        Direct AsNumpy calls do not have a generated operator candidate, so
        their strict-NPU evidence is stored separately from candidate/
        qualification/activation artifacts.  The receipt remains content
        addressed and is safe to replay offline.
        """
        if not isinstance(sample_id, str) or not sample_id:
            raise OperatorArtifactStoreError("sample_id is invalid")
        stable = _stable_receipt(receipt)
        backend = stable.get("backend")
        strict = stable.get("strict_npu_qualified")
        numerical = stable.get("numerical_verified")
        runtime_status = stable.get("runtime_identity_status")
        runtime_sha = stable.get("runtime_identity_sha256")
        scoped = validate_runtime_qualification_scope(sample_id, stable)
        if strict is True:
            if backend != "npu" or (runtime_status != "matched" and scoped is None):
                raise OperatorArtifactStoreError(
                    "strict runtime qualification requires matched NPU identity"
                )
            _validate_sha(runtime_sha, "runtime_identity_sha256")
            if numerical is not True:
                raise OperatorArtifactStoreError(
                    "strict runtime qualification requires numerical verification"
                )
        if runtime_sha is not None:
            _validate_sha(runtime_sha, "runtime_identity_sha256")
        performance = stable.get("performance_qualified")
        if performance is True:
            if (
                strict is not True
                or numerical is not True
                or backend != "npu"
                or (runtime_status != "matched" and scoped is None)
            ):
                raise OperatorArtifactStoreError(
                    "performance qualification requires strict NPU and numerical verification"
                )
            evidence = stable.get("performance_evidence")
            if not isinstance(evidence, Mapping):
                raise OperatorArtifactStoreError(
                    "performance qualification requires evidence reference"
                )
            if evidence.get("schema") != "numpilot.performance-evidence-ref":
                raise OperatorArtifactStoreError(
                    "performance evidence schema is invalid"
                )
            if evidence.get("schema_version") != 1:
                raise OperatorArtifactStoreError(
                    "performance evidence schema version is invalid"
                )
            if evidence.get("status") != "passed" or evidence.get("performance_valid") is not True:
                raise OperatorArtifactStoreError(
                    "performance evidence is not a passed receipt"
                )
            _validate_sha(evidence.get("receipt_sha256"), "performance receipt_sha256")
            if evidence.get("runtime_identity_sha256") != runtime_sha:
                raise OperatorArtifactStoreError(
                    "performance evidence runtime identity does not match"
                )
            if stable.get("source_sha256") is not None and evidence.get("source_sha256") != stable.get("source_sha256"):
                raise OperatorArtifactStoreError(
                    "performance evidence source does not match sample"
                )
        if scoped is not None:
            validate_scoped_performance_reference(sample_id, stable)
        gate_status = (
            "qualified"
            if strict is True and numerical is True
            else "failed"
            if backend == "npu"
            else "not_applicable"
        )
        qualification_id = _sha256(_canonical({
            "sample_id": sample_id,
            "receipt": stable,
        }))
        stored = {
            "schema": RUNTIME_QUALIFICATION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "runtime_qualification_id": qualification_id,
            "sample_id": sample_id,
            "status": gate_status,
            "strict_npu_qualified": strict,
            "numerical_verified": numerical,
            "performance_qualified": stable.get("performance_qualified"),
            "runtime_identity_status": runtime_status,
            "runtime_identity_sha256": runtime_sha,
            **({"runtime_qualification_scope": stable["runtime_qualification_scope"],
                "identity_scope": scoped["identity_scope"],
                "global_runtime_audit_promoted": False} if scoped is not None else {}),
            "stored_at": _now(),
            "receipt": stable,
        }
        path = self.runtime_qualifications / f"{qualification_id}.json"
        _atomic_bytes(path, _json(stored))
        return ArtifactRef(
            "runtime_qualification",
            "",
            qualification_id,
            str(path),
        )

    def bind_sample(
        self,
        sample_id: str,
        operator_id: str,
        *,
        candidate_id: str,
        qualification_id: Optional[str] = None,
        activation_id: Optional[str] = None,
        status: str = "candidate",
        callsite: str = "",
        runtime_qualification_id: Optional[str] = None,
        strict_npu_qualified: Optional[bool] = None,
        performance_qualified: Optional[bool] = None,
    ) -> Path:
        if status not in {"candidate", "qualified", "activated", "blocked", "rejected"}:
            raise OperatorArtifactStoreError("sample binding status is invalid")
        runtime_payload = None
        if runtime_qualification_id is not None:
            _validate_sha(runtime_qualification_id, "runtime_qualification_id")
            runtime_path = self.runtime_qualifications / f"{runtime_qualification_id}.json"
            runtime_payload = self._read_json(runtime_path)
        if performance_qualified is True and (
            runtime_payload is None or runtime_payload.get("performance_qualified") is not True
        ):
            raise OperatorArtifactStoreError(
                "performance sample binding requires a qualified runtime receipt"
            )
        candidate = self._find_artifact(operator_id, "candidates", candidate_id, "candidate.json")
        qualification = None
        if qualification_id is not None:
            qualification = self._find_artifact(operator_id, "qualifications", qualification_id, "qualification.json")
            if qualification.get("candidate_id") != candidate_id:
                raise OperatorArtifactStoreError("sample qualification does not match candidate")
        if status == "qualified":
            if qualification is None or qualification.get("status") != "qualified":
                raise OperatorArtifactStoreError("qualified sample requires a qualified qualification")
        if status == "candidate" and (qualification_id is not None or activation_id is not None):
            raise OperatorArtifactStoreError("candidate sample cannot reference qualification or activation")
        activation = None
        if activation_id is not None:
            activation = self._find_artifact(operator_id, "activations", activation_id, "activation.json")
            if activation.get("qualification_id") != qualification_id:
                raise OperatorArtifactStoreError("sample activation does not match qualification")
        if status == "activated":
            if activation is None or activation.get("status") != "activated":
                raise OperatorArtifactStoreError("activated sample requires an activated activation")
            if qualification is None or qualification.get("status") != "qualified":
                raise OperatorArtifactStoreError("activated sample requires a qualified qualification")
        sample_safe = _safe_component(sample_id, "sample_id")
        path = self.samples / f"{sample_safe}-{operator_id}.json"
        payload = {
            "schema": BINDING_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "operator_id": operator_id,
            "candidate_id": candidate_id,
            "qualification_id": qualification_id,
            "activation_id": activation_id,
            "runtime_qualification_id": runtime_qualification_id,
            "strict_npu_qualified": strict_npu_qualified,
            "performance_qualified": performance_qualified,
            "status": status,
            "callsite": callsite,
            "candidate_source_sha256": candidate.get("source_sha256"),
            "stored_at": _now(),
        }
        _atomic_bytes(path, _json(payload), replace=True)
        return path

    def record_sample_without_operator(
        self,
        sample_id: str,
        *,
        status: str,
        reason: str = "",
        route: str = "",
        runtime_qualification_id: Optional[str] = None,
        strict_npu_qualified: Optional[bool] = None,
        performance_qualified: Optional[bool] = None,
    ) -> Path:
        """Record a benchmark sample that did not need a generated operator.

        Direct, rejected, and unsupported cases still belong to the 150-case
        run, but they must not be represented as fake operator candidates.
        This receipt keeps coverage accounting complete while preserving the
        candidate/qualification dependency rules.
        """
        allowed = {"not_applicable", "blocked", "rejected", "unsupported", "failed"}
        if status not in allowed:
            raise OperatorArtifactStoreError("sample status is invalid")
        runtime_payload = None
        if runtime_qualification_id is not None:
            _validate_sha(runtime_qualification_id, "runtime_qualification_id")
            runtime_path = self.runtime_qualifications / f"{runtime_qualification_id}.json"
            runtime_payload = self._read_json(runtime_path)
        if performance_qualified is True and (
            runtime_payload is None or runtime_payload.get("performance_qualified") is not True
        ):
            raise OperatorArtifactStoreError(
                "performance sample binding requires a qualified runtime receipt"
            )
        sample_safe = _safe_component(sample_id, "sample_id")
        path = self.samples / f"{sample_safe}-none.json"
        payload = {
            "schema": BINDING_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "operator_id": None,
            "candidate_id": None,
            "qualification_id": None,
            "activation_id": None,
            "runtime_qualification_id": runtime_qualification_id,
            "strict_npu_qualified": strict_npu_qualified,
            "performance_qualified": performance_qualified,
            "status": status,
            "route": route,
            "reason": reason[:512],
            "stored_at": _now(),
        }
        _atomic_bytes(path, _json(payload), replace=True)
        return path

    def list_bindings(self) -> tuple[dict[str, Any], ...]:
        """Return sample references in deterministic order for benchmark replay."""
        bindings = []
        for path in sorted(self.samples.glob("*.json")):
            if path.is_symlink():
                raise OperatorArtifactStoreError(f"refusing symlink sample binding: {path}")
            bindings.append(self._read_json(path))
        return tuple(bindings)

    def summary(self) -> dict[str, Any]:
        """Return a sanitized aggregate suitable for a 150-case summary receipt."""
        bindings = self.list_bindings()
        statuses: dict[str, int] = {}
        operators = set()
        candidates = set()
        qualifications = set()
        activations = set()
        for binding in bindings:
            status = str(binding.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
            if binding.get("operator_id"):
                operators.add(binding["operator_id"])
            if binding.get("candidate_id"):
                candidates.add(binding["candidate_id"])
            if binding.get("qualification_id"):
                qualifications.add(binding["qualification_id"])
            if binding.get("activation_id"):
                activations.add(binding["activation_id"])
        runtime_files = sorted(self.runtime_qualifications.glob("*.json"))
        runtime_qualified = 0
        performance_valid = 0
        for path in runtime_files:
            try:
                runtime = self._read_json(path)
            except OperatorArtifactStoreError:
                continue
            runtime_qualified += runtime.get("status") == "qualified"
            performance_valid += runtime.get("performance_qualified") is True
        return {
            "schema": "numpilot.operator-artifact-summary",
            "schema_version": SCHEMA_VERSION,
            "sample_bindings": len(bindings),
            "unique_operators": len(operators),
            "unique_candidates": len(candidates),
            "unique_qualifications": len(qualifications),
            "unique_activations": len(activations),
            "runtime_qualifications": len(runtime_files),
            "strict_npu_qualified": runtime_qualified,
            "performance_valid": performance_valid,
            "statuses": dict(sorted(statuses.items())),
        }

    def verify(self) -> dict[str, Any]:
        """Validate hashes, references, and lifecycle dependencies offline."""
        errors: list[str] = []
        operators = candidates = qualifications = activations = 0
        try:
            root_record = self._read_json(self.store_path)
            if root_record.get("schema") != STORE_SCHEMA:
                errors.append("store_schema_mismatch")
        except OperatorArtifactStoreError as exc:
            errors.append(str(exc))
        for operator_dir in sorted(self.operators.iterdir() if self.operators.exists() else ()):
            if operator_dir.is_symlink() or not operator_dir.is_dir():
                errors.append("invalid_operator_directory:%s" % operator_dir.name)
                continue
            operator_path = operator_dir / "operator.json"
            try:
                record = self._read_json(operator_path)
                key = OperatorKey(**record["key"])
                if record.get("operator_id") != key.operator_id or operator_dir.name != key.operator_id:
                    errors.append("operator_id_mismatch:%s" % operator_dir.name)
                operators += 1
            except Exception as exc:
                errors.append("invalid_operator:%s:%s" % (operator_dir.name, type(exc).__name__))
                continue
            for group, filename, counter in (("candidates", "candidate.json", "candidate"), ("qualifications", "qualification.json", "qualification"), ("activations", "activation.json", "activation")):
                directory = operator_dir / group
                if not directory.exists():
                    continue
                for artifact_dir in sorted(directory.iterdir()):
                    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
                        errors.append("invalid_artifact_directory:%s" % artifact_dir)
                        continue
                    try:
                        payload = self._read_json(artifact_dir / filename)
                        artifact_id = payload.get({"candidate": "candidate_id", "qualification": "qualification_id", "activation": "activation_id"}[counter])
                        if artifact_id != artifact_dir.name:
                            errors.append("artifact_id_mismatch:%s" % artifact_dir)
                        if counter == "candidate":
                            source = artifact_dir / str(payload.get("source_name", ""))
                            if source.is_symlink() or not source.is_file() or _sha256(source.read_bytes()) != payload.get("source_sha256"):
                                errors.append("candidate_source_hash_mismatch:%s" % artifact_dir)
                        elif counter == "qualification":
                            candidate_id = payload.get("candidate_id")
                            if not (operator_dir / "candidates" / str(candidate_id) / "candidate.json").is_file():
                                errors.append("qualification_candidate_missing:%s" % artifact_dir)
                        else:
                            qualification_id = payload.get("qualification_id")
                            qualification = operator_dir / "qualifications" / str(qualification_id) / "qualification.json"
                            if not qualification.is_file() or self._read_json(qualification).get("status") != "qualified":
                                errors.append("activation_qualification_invalid:%s" % artifact_dir)
                        if counter == "candidate": candidates += 1
                        elif counter == "qualification": qualifications += 1
                        else: activations += 1
                    except Exception as exc:
                        errors.append("invalid_%s:%s:%s" % (counter, artifact_dir.name, type(exc).__name__))
        try:
            bindings = self.list_bindings()
            for binding in bindings:
                operator_id = binding.get("operator_id")
                if operator_id is None:
                    if any(binding.get(key) is not None for key in ("candidate_id", "qualification_id", "activation_id")):
                        errors.append("empty_binding_has_artifact:%s" % binding.get("sample_id"))
                    continue
                candidate_id = binding.get("candidate_id")
                self._find_artifact(operator_id, "candidates", candidate_id, "candidate.json")
                if binding.get("qualification_id") is not None:
                    self._find_artifact(operator_id, "qualifications", binding["qualification_id"], "qualification.json")
                if binding.get("activation_id") is not None:
                    self._find_artifact(operator_id, "activations", binding["activation_id"], "activation.json")
        except Exception as exc:
            errors.append("binding_validation:%s" % type(exc).__name__)
            bindings = ()
        return {
            "schema": "numpilot.operator-artifact-verification",
            "schema_version": SCHEMA_VERSION,
            "ok": not errors,
            "operators": operators,
            "candidates": candidates,
            "qualifications": qualifications,
            "activations": activations,
            "sample_bindings": len(bindings),
            "errors": sorted(set(errors)),
        }

    def gc_dry_run(self) -> dict[str, Any]:
        """Mark unreferenced objects without deleting anything."""
        referenced = {"candidates": set(), "qualifications": set(), "activations": set()}
        for binding in self.list_bindings():
            if binding.get("operator_id") is None:
                continue
            for group, key in (("candidates", "candidate_id"), ("qualifications", "qualification_id"), ("activations", "activation_id")):
                if binding.get(key):
                    referenced[group].add((str(binding["operator_id"]), str(binding[key])))
        orphaned: list[dict[str, str]] = []
        for operator_dir in sorted(self.operators.iterdir() if self.operators.exists() else ()):
            if not operator_dir.is_dir() or operator_dir.is_symlink():
                continue
            for group in referenced:
                directory = operator_dir / group
                if not directory.exists():
                    continue
                for artifact_dir in sorted(directory.iterdir()):
                    if artifact_dir.is_dir() and not artifact_dir.is_symlink() and (operator_dir.name, artifact_dir.name) not in referenced[group]:
                        orphaned.append({"group": group, "operator_id": operator_dir.name, "artifact_id": artifact_dir.name})
        return {
            "schema": "numpilot.operator-artifact-gc",
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "delete_performed": False,
            "referenced": {key: len(value) for key, value in referenced.items()},
            "orphaned": orphaned,
        }

    def _find_artifact(self, operator_id: str, group: str, artifact_id: str, filename: str) -> dict[str, Any]:
        _validate_sha(operator_id, "operator_id")
        _validate_sha(artifact_id, "artifact_id")
        path = self._operator_dir(operator_id) / group / artifact_id / filename
        return self._read_json(path)

    def _append_operator(self, operator_id: str, field: str, artifact_id: str, latest_field: str, latest_id: str) -> None:
        path = self._operator_dir(operator_id) / "operator.json"
        self.lock_path.touch(mode=0o600, exist_ok=True)
        handle = self.lock_path.open("r+")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            payload = self._read_json(path)
            values = list(payload.get(field, []))
            if artifact_id not in values:
                values.append(artifact_id)
            payload[field] = values
            latest = dict(payload.get("latest") or {})
            latest[latest_field] = latest_id
            payload["latest"] = latest
            _atomic_bytes(path, _json(payload), replace=True)
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
