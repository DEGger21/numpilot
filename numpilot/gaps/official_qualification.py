"""Fail-closed admission of official ACLNN capability qualifications.

The custom-operator qualification schema describes generated artifacts and a
model-assisted repair pipeline.  Official ACLNN qualification is a different
thing: an already integrated AsNumpy wheel was built from a hash-bound source
revision, executed on the device, and compared against an independent NumPy
oracle.  This module therefore defines a separate, explicit schema and adapter
instead of reinterpreting artifact-manifest evidence as official evidence.

The overlay is deliberately private input.  NumPilot only checks bounded JSON
reports, file hashes, the live runtime identity and the base capability
manifest; it never builds, imports or executes anything named by the overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    OperatorSignature,
    _PROFILE_DTYPES_V1,
    parse_profile_constraints,
)
from numpilot.capability_catalog import (
    CapabilityCatalogError,
    validate_canonical_key,
)
from numpilot.gaps.qualification import (
    QualificationError,
    capability_manifest_sha256,
)


OFFICIAL_OVERLAY_SCHEMA = "numpilot.official-aclnn-capability-overlay"
OFFICIAL_GATE_SCHEMA = "numpilot.official-aclnn-gate-report"
OFFICIAL_SCHEMA_VERSION = 1
OFFICIAL_PRODUCER = "numpilot-official-aclnn-qualification-v1"
OFFICIAL_QUALIFICATION_PREFIX = "official-aclnn-"
OFFICIAL_PROVENANCE = "native_npu"
OFFICIAL_GATES = (
    "artifact_integrity",
    "build",
    "unit_test",
    "numerical_validation",
    "public_asnumpy_api",
    "npu_execution",
)
OFFICIAL_GATE_STATUSES = frozenset({"passed", "failed"})
OFFICIAL_OVERLAY_AUDIT_TAG = "official-qualification-"

ZERO_SHA256 = "0" * 64
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
_MAX_TEXT_CHARS = 512
_MAX_PATH_CHARS = 512
_MAX_PATH_DEPTH = 16
_MAX_ENTRIES = 64
_MAX_TARGETS = 64
_MAX_CLAIM_DEPTH = 6
_MAX_CLAIM_ITEMS = 512
_MAX_CLAIM_BYTES = 256 * 1024
_MAX_SOURCE_FILES = 1_000_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
_QUALIFICATION_ID = re.compile(
    r"^" + re.escape(OFFICIAL_QUALIFICATION_PREFIX) + r"[0-9a-f]{20}$"
)
_REGISTRY_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_WRAPPER_PATH = re.compile(
    r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)
_GAP_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,191}$")
_DTYPE_FLOAT32 = "dtype=float32"
_OFFICIAL_API_WRAPPER = {
    "concatenate": "array.py",
    "ones": "array.py",
    "zeros": "array.py",
    "invert": "logic.py",
    "argsort": "sorting.py",
    "searchsorted": "sorting.py",
    "median": "statistics.py",
}


def _canonical_json(data: Any) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: frozenset,
    label: str,
) -> None:
    if not isinstance(data, Mapping):
        raise QualificationError("%s must be an object" % label)
    actual = frozenset(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append("missing=%s" % ",".join(missing))
        if extra:
            detail.append("unexpected=%s" % ",".join(extra))
        raise QualificationError(
            "%s fields are invalid (%s)" % (label, "; ".join(detail))
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise QualificationError("%s must be a lowercase SHA-256" % label)
    return value


def _require_int(value: Any, label: str, minimum: int = 0, maximum: int = 2 ** 63 - 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise QualificationError("%s is invalid" % label)
    return value


def _require_text(value: Any, label: str, *, limit: int = _MAX_TEXT_CHARS) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > limit
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise QualificationError("%s is invalid" % label)
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    text = _require_text(value, label, limit=_MAX_PATH_CHARS)
    if "\\" in text or "\x00" in text:
        raise QualificationError("%s is not a safe relative path" % label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or len(path.parts) > _MAX_PATH_DEPTH
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise QualificationError("%s is not a safe relative path" % label)
    return str(path)


def _map_sha256(value: Any, label: str, *, allow_empty: bool = False) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise QualificationError("%s must be an object" % label)
    if not value and not allow_empty:
        raise QualificationError("%s must not be empty" % label)
    result = {}
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 256
            or not _WRAPPER_PATH.fullmatch(name)
        ):
            raise QualificationError("%s contains an invalid file name" % label)
        result[name] = _require_sha256(digest, "%s for %s" % (label, name))
    return MappingProxyType(dict(sorted(result.items())))


def _require_claim_value(value: Any, depth: int = 0) -> None:
    if depth > _MAX_CLAIM_DEPTH:
        raise QualificationError("gate claim nesting is too deep")
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value) > _MAX_TEXT_CHARS:
            raise QualificationError("gate claim text is too long")
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if not -(2 ** 63) <= value <= 2 ** 63:
            raise QualificationError("gate claim integer is out of bounds")
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise QualificationError("gate claim float is not finite")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_CLAIM_ITEMS:
            raise QualificationError("gate claim list is too long")
        for item in value:
            _require_claim_value(item, depth + 1)
        return
    if isinstance(value, Mapping):
        if len(value) > _MAX_CLAIM_ITEMS:
            raise QualificationError("gate claim object is too large")
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise QualificationError("gate claim key is invalid")
            _require_claim_value(item, depth + 1)
        return
    raise QualificationError("gate claim value is not JSON-safe")


def _require_claims(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("%s must be an object" % label)
    _require_claim_value(value)
    if len(_canonical_json(value)) > _MAX_CLAIM_BYTES:
        raise QualificationError("%s exceeds the claim size limit" % label)
    return MappingProxyType(dict(value))


def _require_evidence(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("%s must be an object" % label)
    _require_claim_value(value)
    if len(_canonical_json(value)) > _MAX_CLAIM_BYTES:
        raise QualificationError("%s exceeds the evidence size limit" % label)
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class OfficialFileRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "official evidence path")
        _require_sha256(self.sha256, "official evidence hash")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialFileRef":
        _require_exact_fields(
            data, frozenset({"path", "sha256"}), "official file reference"
        )
        return cls(path=data["path"], sha256=data["sha256"])

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class OfficialRuntimeBinding:
    asnumpy_version: str
    audit_id: str
    identity_sha256: str
    status: str
    module_sha256: str
    native_module_sha256: str
    wrapper_sha256: Mapping[str, str]
    cann_version: str
    soc_model: str
    python_version: str

    def __post_init__(self) -> None:
        for name, value in (
            ("AsNumpy version", self.asnumpy_version),
            ("runtime audit ID", self.audit_id),
            ("runtime status", self.status),
            ("CANN version", self.cann_version),
            ("SoC model", self.soc_model),
            ("Python version", self.python_version),
        ):
            _require_text(value, name)
        _require_sha256(self.identity_sha256, "runtime identity")
        _require_sha256(self.module_sha256, "runtime module")
        _require_sha256(self.native_module_sha256, "runtime native module")
        object.__setattr__(
            self,
            "wrapper_sha256",
            _map_sha256(self.wrapper_sha256, "runtime wrapper hashes"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialRuntimeBinding":
        _require_exact_fields(
            data,
            frozenset({
                "asnumpy_version",
                "audit_id",
                "identity_sha256",
                "status",
                "module_sha256",
                "native_module_sha256",
                "wrapper_sha256",
                "cann_version",
                "soc_model",
                "python_version",
            }),
            "official runtime binding",
        )
        return cls(
            asnumpy_version=data["asnumpy_version"],
            audit_id=data["audit_id"],
            identity_sha256=data["identity_sha256"],
            status=data["status"],
            module_sha256=data["module_sha256"],
            native_module_sha256=data["native_module_sha256"],
            wrapper_sha256=data["wrapper_sha256"],
            cann_version=data["cann_version"],
            soc_model=data["soc_model"],
            python_version=data["python_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asnumpy_version": self.asnumpy_version,
            "audit_id": self.audit_id,
            "identity_sha256": self.identity_sha256,
            "status": self.status,
            "module_sha256": self.module_sha256,
            "native_module_sha256": self.native_module_sha256,
            "wrapper_sha256": dict(self.wrapper_sha256),
            "cann_version": self.cann_version,
            "soc_model": self.soc_model,
            "python_version": self.python_version,
        }


@dataclass(frozen=True)
class OfficialIntegrationBinding:
    attempt: str
    transaction_id: str
    transaction_sha256: str
    transaction: OfficialFileRef
    wheel: OfficialFileRef
    gate_result: OfficialFileRef
    gate_log: OfficialFileRef
    build_log: OfficialFileRef
    candidate_test_log: OfficialFileRef
    source_tree_sha256: str
    source_file_count: int
    source_revision: str
    source_overlay_sha256: Mapping[str, str]
    runtime_core_sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.attempt, "integration attempt")
        _require_text(self.transaction_id, "integration transaction ID")
        _require_sha256(self.transaction_sha256, "integration transaction")
        for name, ref in (
            ("integration transaction", self.transaction),
            ("candidate wheel", self.wheel),
            ("gate result", self.gate_result),
            ("gate log", self.gate_log),
            ("build log", self.build_log),
            ("candidate test log", self.candidate_test_log),
        ):
            if not isinstance(ref, OfficialFileRef):
                raise QualificationError("%s must be a file reference" % name)
            if not ref.path.startswith(self.attempt + "/"):
                raise QualificationError(
                    "%s is outside the bound integration attempt" % name
                )
        _require_sha256(self.source_tree_sha256, "integration source tree")
        _require_int(
            self.source_file_count,
            "integration source file count",
            maximum=_MAX_SOURCE_FILES,
        )
        if not _GIT_OBJECT_ID.fullmatch(self.source_revision):
            raise QualificationError("integration source revision is invalid")
        object.__setattr__(
            self,
            "source_overlay_sha256",
            _map_sha256(
                self.source_overlay_sha256,
                "integration source overlay hashes",
            ),
        )
        _require_sha256(self.runtime_core_sha256, "integration runtime core")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialIntegrationBinding":
        _require_exact_fields(
            data,
            frozenset({
                "attempt",
                "transaction_id",
                "transaction_sha256",
                "transaction",
                "wheel",
                "gate_result",
                "gate_log",
                "build_log",
                "candidate_test_log",
                "source_tree_sha256",
                "source_file_count",
                "source_revision",
                "source_overlay_sha256",
                "runtime_core_sha256",
            }),
            "official integration binding",
        )
        return cls(
            attempt=data["attempt"],
            transaction_id=data["transaction_id"],
            transaction_sha256=data["transaction_sha256"],
            transaction=OfficialFileRef.from_dict(data["transaction"]),
            wheel=OfficialFileRef.from_dict(data["wheel"]),
            gate_result=OfficialFileRef.from_dict(data["gate_result"]),
            gate_log=OfficialFileRef.from_dict(data["gate_log"]),
            build_log=OfficialFileRef.from_dict(data["build_log"]),
            candidate_test_log=OfficialFileRef.from_dict(
                data["candidate_test_log"]
            ),
            source_tree_sha256=data["source_tree_sha256"],
            source_file_count=data["source_file_count"],
            source_revision=data["source_revision"],
            source_overlay_sha256=data["source_overlay_sha256"],
            runtime_core_sha256=data["runtime_core_sha256"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "transaction_id": self.transaction_id,
            "transaction_sha256": self.transaction_sha256,
            "transaction": self.transaction.to_dict(),
            "wheel": self.wheel.to_dict(),
            "gate_result": self.gate_result.to_dict(),
            "gate_log": self.gate_log.to_dict(),
            "build_log": self.build_log.to_dict(),
            "candidate_test_log": self.candidate_test_log.to_dict(),
            "source_tree_sha256": self.source_tree_sha256,
            "source_file_count": self.source_file_count,
            "source_revision": self.source_revision,
            "source_overlay_sha256": dict(self.source_overlay_sha256),
            "runtime_core_sha256": self.runtime_core_sha256,
        }


@dataclass(frozen=True)
class OfficialGapBinding:
    gap_id: str
    request: OfficialFileRef
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.gap_id, str) or not _GAP_ID.fullmatch(self.gap_id):
            raise QualificationError("official gap ID is invalid")
        if not isinstance(self.request, OfficialFileRef):
            raise QualificationError("official gap request must be a file reference")
        _require_sha256(self.source_sha256, "official gap source")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialGapBinding":
        _require_exact_fields(
            data,
            frozenset({"gap_id", "request", "source_sha256"}),
            "official gap binding",
        )
        return cls(
            gap_id=data["gap_id"],
            request=OfficialFileRef.from_dict(data["request"]),
            source_sha256=data["source_sha256"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "request": self.request.to_dict(),
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class OfficialSourceBinding:
    logical_path: str
    bundle_path: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.logical_path, "official source path")
        _safe_relative_path(self.bundle_path, "official source bundle path")
        if not self.bundle_path.startswith("source/"):
            raise QualificationError(
                "official source evidence must live under source/"
            )
        _require_sha256(self.sha256, "official source")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialSourceBinding":
        _require_exact_fields(
            data,
            frozenset({"logical_path", "bundle_path", "sha256"}),
            "official source binding",
        )
        return cls(
            logical_path=data["logical_path"],
            bundle_path=data["bundle_path"],
            sha256=data["sha256"],
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "logical_path": self.logical_path,
            "bundle_path": self.bundle_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OfficialOverlayEntry:
    registry_key: str
    qualified_api: str
    asnumpy_api: str
    provenance: str
    constraints: Tuple[str, ...]
    signature_variants: Tuple[OperatorSignature, ...]

    def __post_init__(self) -> None:
        try:
            key = validate_canonical_key(self.registry_key)
        except CapabilityCatalogError as exc:
            raise QualificationError("official entry registry key is invalid") from exc
        if key != self.registry_key or "." in key:
            raise QualificationError(
                "official ACLNN overlay only admits top-level registry keys"
            )
        if self.qualified_api != "numpy." + key:
            raise QualificationError("official entry qualified API is inconsistent")
        if self.asnumpy_api != "asnumpy." + key:
            raise QualificationError("official entry AsNumpy API is inconsistent")
        if self.provenance != OFFICIAL_PROVENANCE:
            raise QualificationError(
                "official ACLNN overlay only admits native NPU provenance"
            )
        if (
            not isinstance(self.constraints, tuple)
            or not self.constraints
            or len(self.constraints) > 64
            or any(
                not isinstance(item, str) or not item
                for item in self.constraints
            )
            or len(set(self.constraints)) != len(self.constraints)
        ):
            raise QualificationError("official entry constraints are invalid")
        parsed = parse_profile_constraints(self.constraints)
        if parsed["dtype"] not in _PROFILE_DTYPES_V1:
            raise QualificationError(
                "official ACLNN overlay only admits one audited dtype profile"
            )
        if (
            not isinstance(self.signature_variants, tuple)
            or not self.signature_variants
            or len(self.signature_variants) > 32
            or any(
                not isinstance(item, OperatorSignature)
                for item in self.signature_variants
            )
            or len(set(self.signature_variants)) != len(self.signature_variants)
        ):
            raise QualificationError("official entry signatures are invalid")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialOverlayEntry":
        _require_exact_fields(
            data,
            frozenset({
                "registry_key",
                "qualified_api",
                "asnumpy_api",
                "provenance",
                "constraints",
                "signature_variants",
            }),
            "official overlay entry",
        )
        constraints = data["constraints"]
        if not isinstance(constraints, (list, tuple)):
            raise QualificationError("official entry constraints must be a list")
        signatures = data["signature_variants"]
        if not isinstance(signatures, (list, tuple)):
            raise QualificationError("official entry signatures must be a list")
        try:
            parsed_signatures = tuple(
                OperatorSignature.from_dict(item) for item in signatures
            )
        except (TypeError, ValueError) as exc:
            raise QualificationError(
                "official entry signature variant is invalid"
            ) from exc
        return cls(
            registry_key=data["registry_key"],
            qualified_api=data["qualified_api"],
            asnumpy_api=data["asnumpy_api"],
            provenance=data["provenance"],
            constraints=tuple(constraints),
            signature_variants=parsed_signatures,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_key": self.registry_key,
            "qualified_api": self.qualified_api,
            "asnumpy_api": self.asnumpy_api,
            "provenance": self.provenance,
            "constraints": list(self.constraints),
            "signature_variants": [
                item.to_dict() for item in self.signature_variants
            ],
        }


def compute_official_qualification_id(data: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in data.items()
        if key != "qualification_id"
    }
    digest = _sha256_bytes(_canonical_json(body))
    return OFFICIAL_QUALIFICATION_PREFIX + digest[:20]


@dataclass(frozen=True)
class OfficialCapabilityOverlay:
    producer: str
    qualification_id: str
    revision: int
    parent_sha256: str
    base_manifest_sha256: str
    runtime: OfficialRuntimeBinding
    integration: OfficialIntegrationBinding
    gap_spec: OfficialGapBinding
    source: OfficialSourceBinding
    gates: Mapping[str, OfficialFileRef]
    entries: Tuple[OfficialOverlayEntry, ...]
    schema: str = OFFICIAL_OVERLAY_SCHEMA
    schema_version: int = OFFICIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != OFFICIAL_OVERLAY_SCHEMA:
            raise QualificationError("official overlay schema is unsupported")
        if self.schema_version != OFFICIAL_SCHEMA_VERSION:
            raise QualificationError("official overlay schema version is unsupported")
        _require_text(self.producer, "official overlay producer", limit=128)
        if not isinstance(self.qualification_id, str) or not _QUALIFICATION_ID.fullmatch(
            self.qualification_id
        ):
            raise QualificationError("official qualification ID is invalid")
        _require_int(self.revision, "official overlay revision", minimum=1)
        _require_sha256(self.parent_sha256, "official overlay parent")
        _require_sha256(self.base_manifest_sha256, "official overlay base manifest")
        for name, value, expected_type in (
            ("runtime binding", self.runtime, OfficialRuntimeBinding),
            ("integration binding", self.integration, OfficialIntegrationBinding),
            ("gap binding", self.gap_spec, OfficialGapBinding),
            ("source binding", self.source, OfficialSourceBinding),
        ):
            if not isinstance(value, expected_type):
                raise QualificationError("official %s type is invalid" % name)
        if not isinstance(self.gates, Mapping):
            raise QualificationError("official overlay gates must be an object")
        gate_names = frozenset(self.gates)
        if gate_names != frozenset(OFFICIAL_GATES):
            raise QualificationError(
                "official overlay must bind exactly the six qualification gates"
            )
        normalized_gates = {}
        for name in OFFICIAL_GATES:
            ref = self.gates[name]
            if not isinstance(ref, OfficialFileRef):
                raise QualificationError("official gate reference is invalid")
            if not ref.path.startswith("gates/") or not ref.path.endswith(".json"):
                raise QualificationError(
                    "official gate evidence must live under gates/*.json"
                )
            normalized_gates[name] = ref
        object.__setattr__(
            self, "gates", MappingProxyType(normalized_gates)
        )
        if (
            not isinstance(self.entries, tuple)
            or not self.entries
            or len(self.entries) > _MAX_ENTRIES
            or any(
                not isinstance(entry, OfficialOverlayEntry)
                for entry in self.entries
            )
        ):
            raise QualificationError("official overlay entries are invalid")
        keys = [entry.registry_key for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise QualificationError("official overlay contains duplicate entries")

    @property
    def entry_keys(self) -> Tuple[str, ...]:
        return tuple(entry.registry_key for entry in self.entries)

    @property
    def content_sha256(self) -> str:
        return _sha256_bytes(self.to_bytes())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "producer": self.producer,
            "qualification_id": self.qualification_id,
            "revision": self.revision,
            "parent_sha256": self.parent_sha256,
            "base_manifest_sha256": self.base_manifest_sha256,
            "runtime": self.runtime.to_dict(),
            "integration": self.integration.to_dict(),
            "gap_spec": self.gap_spec.to_dict(),
            "source": self.source.to_dict(),
            "gates": {
                name: self.gates[name].to_dict() for name in OFFICIAL_GATES
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialCapabilityOverlay":
        _require_exact_fields(
            data,
            frozenset({
                "schema",
                "schema_version",
                "producer",
                "qualification_id",
                "revision",
                "parent_sha256",
                "base_manifest_sha256",
                "runtime",
                "integration",
                "gap_spec",
                "source",
                "gates",
                "entries",
            }),
            "official capability overlay",
        )
        if data["schema"] != OFFICIAL_OVERLAY_SCHEMA:
            raise QualificationError("official overlay schema is unsupported")
        _require_int(
            data["schema_version"],
            "official overlay schema version",
            minimum=1,
        )
        expected_id = compute_official_qualification_id(data)
        if data["qualification_id"] != expected_id:
            raise QualificationError(
                "official overlay qualification ID does not derive from its content"
            )
        raw_entries = data["entries"]
        if not isinstance(raw_entries, (list, tuple)):
            raise QualificationError("official overlay entries must be a list")
        raw_gates = data["gates"]
        if not isinstance(raw_gates, Mapping):
            raise QualificationError("official overlay gates must be an object")
        return cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            producer=data["producer"],
            qualification_id=data["qualification_id"],
            revision=data["revision"],
            parent_sha256=data["parent_sha256"],
            base_manifest_sha256=data["base_manifest_sha256"],
            runtime=OfficialRuntimeBinding.from_dict(data["runtime"]),
            integration=OfficialIntegrationBinding.from_dict(data["integration"]),
            gap_spec=OfficialGapBinding.from_dict(data["gap_spec"]),
            source=OfficialSourceBinding.from_dict(data["source"]),
            gates={
                name: OfficialFileRef.from_dict(raw_gates[name])
                for name in raw_gates
            },
            entries=tuple(
                OfficialOverlayEntry.from_dict(item) for item in raw_entries
            ),
        )


@dataclass(frozen=True)
class OfficialGateTarget:
    status: str
    claims: Mapping[str, Any]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in OFFICIAL_GATE_STATUSES:
            raise QualificationError("official gate target status is invalid")
        object.__setattr__(
            self,
            "claims",
            _require_claims(self.claims, "official gate target claims"),
        )
        object.__setattr__(
            self,
            "evidence",
            _require_evidence(self.evidence, "official gate target evidence"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialGateTarget":
        _require_exact_fields(
            data,
            frozenset({"status", "claims", "evidence"}),
            "official gate target",
        )
        return cls(
            status=data["status"],
            claims=data["claims"],
            evidence=data["evidence"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "claims": dict(self.claims),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class OfficialGateReport:
    gate: str
    status: str
    producer: str
    targets: Mapping[str, OfficialGateTarget]
    schema: str = OFFICIAL_GATE_SCHEMA
    schema_version: int = OFFICIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != OFFICIAL_GATE_SCHEMA:
            raise QualificationError("official gate schema is unsupported")
        if self.schema_version != OFFICIAL_SCHEMA_VERSION:
            raise QualificationError("official gate schema version is unsupported")
        if self.gate not in OFFICIAL_GATES:
            raise QualificationError("official gate name is invalid")
        if self.status not in OFFICIAL_GATE_STATUSES:
            raise QualificationError("official gate status is invalid")
        _require_text(self.producer, "official gate producer", limit=128)
        if not isinstance(self.targets, Mapping) or not self.targets:
            raise QualificationError("official gate targets must be non-empty")
        if len(self.targets) > _MAX_TARGETS:
            raise QualificationError("official gate has too many targets")
        normalized = {}
        for name, target in self.targets.items():
            if (
                not isinstance(name, str)
                or not name
                or not _REGISTRY_KEY.fullmatch(name)
            ):
                raise QualificationError("official gate target name is invalid")
            if not isinstance(target, OfficialGateTarget):
                raise QualificationError("official gate target type is invalid")
            normalized[name] = target
        expected_status = (
            "passed"
            if all(item.status == "passed" for item in normalized.values())
            else "failed"
        )
        if self.status != expected_status:
            raise QualificationError("official gate status is inconsistent")
        object.__setattr__(self, "targets", MappingProxyType(
            dict(sorted(normalized.items()))
        ))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialGateReport":
        _require_exact_fields(
            data,
            frozenset({"schema", "schema_version", "gate", "status", "producer", "targets"}),
            "official gate report",
        )
        targets = data["targets"]
        if not isinstance(targets, Mapping):
            raise QualificationError("official gate targets must be an object")
        return cls(
            schema=data["schema"],
            schema_version=data["schema_version"],
            gate=data["gate"],
            status=data["status"],
            producer=data["producer"],
            targets={
                name: OfficialGateTarget.from_dict(target)
                for name, target in targets.items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "gate": self.gate,
            "status": self.status,
            "producer": self.producer,
            "targets": {
                name: target.to_dict()
                for name, target in sorted(self.targets.items())
            },
        }

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")


def build_official_gate_report(
    gate: str,
    targets: Mapping[str, Tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    producer: str = OFFICIAL_PRODUCER,
) -> OfficialGateReport:
    """Build one typed gate report from ``(status, claims, evidence)`` triples."""

    return OfficialGateReport(
        gate=gate,
        status=(
            "passed"
            if targets and all(item[0] == "passed" for item in targets.values())
            else "failed"
        ),
        producer=producer,
        targets={
            name: OfficialGateTarget(
                status=status,
                claims=claims,
                evidence=evidence,
            )
            for name, (status, claims, evidence) in targets.items()
        },
    )


def is_official_overlay_payload(data: Any) -> bool:
    return isinstance(data, Mapping) and data.get("schema") == OFFICIAL_OVERLAY_SCHEMA


def official_overlay_from_payload(data: Mapping[str, Any]) -> OfficialCapabilityOverlay:
    return OfficialCapabilityOverlay.from_dict(data)


def is_official_overlay(value: Any) -> bool:
    return isinstance(value, OfficialCapabilityOverlay)


def official_overlay_file_sha256(overlay: OfficialCapabilityOverlay) -> str:
    """Return the canonical on-disk identity used by runtime admission."""

    return _sha256_bytes(overlay.to_bytes())


def _validate_private_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationError("%s is unavailable" % label) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o022
    ):
        raise QualificationError("%s must be a private directory" % label)


def _read_private_file(path: Path, label: str, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise QualificationError("%s is unavailable" % label) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o022
            or metadata.st_size > max_bytes
        ):
            raise QualificationError("%s is not a private regular file" % label)
        chunks = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise QualificationError("%s exceeds the size limit" % label)
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_json(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
        data = json.loads(text)
    except (UnicodeError, ValueError) as exc:
        raise QualificationError("%s is not valid UTF-8 JSON" % label) from exc
    if not isinstance(data, dict):
        raise QualificationError("%s must be a JSON object" % label)
    return data


def load_official_overlay(path: str) -> OfficialCapabilityOverlay:
    target = Path(path)
    _validate_private_directory(target.parent, "official overlay parent")
    payload = _read_private_file(
        target, "official overlay", max_bytes=_MAX_JSON_BYTES
    )
    return OfficialCapabilityOverlay.from_dict(
        _parse_json(payload, "official overlay")
    )


def load_official_gate_report(path: str) -> OfficialGateReport:
    target = Path(path)
    _validate_private_directory(target.parent, "official gate parent")
    payload = _read_private_file(
        target, "official gate report", max_bytes=_MAX_JSON_BYTES
    )
    return OfficialGateReport.from_dict(
        _parse_json(payload, "official gate report")
    )


def _evidence_path(evidence_root: str, ref: OfficialFileRef) -> Path:
    root = Path(evidence_root)
    _validate_private_directory(root, "qualification evidence root")
    current = root
    parts = PurePosixPath(ref.path).parts
    for part in parts[:-1]:
        current = current / part
        _validate_private_directory(current, "qualification evidence directory")
    return root.joinpath(*parts)


def _load_bound_file(
    evidence_root: str,
    ref: OfficialFileRef,
    label: str,
) -> bytes:
    path = _evidence_path(evidence_root, ref)
    payload = _read_private_file(
        path, label, max_bytes=_MAX_EVIDENCE_FILE_BYTES
    )
    if _sha256_bytes(payload) != ref.sha256:
        raise QualificationError("%s hash does not match the overlay" % label)
    return payload


def _load_bound_json(
    evidence_root: str,
    ref: OfficialFileRef,
    label: str,
) -> Mapping[str, Any]:
    return _parse_json(
        _load_bound_file(evidence_root, ref, label), label
    )


def _transaction_sha256(transaction: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in transaction.items()
        if key != "transaction_sha256"
    }
    return _sha256_bytes(_canonical_json(body))


def _runtime_value(identity: Any, name: str) -> Any:
    if isinstance(identity, Mapping):
        return identity.get(name)
    return getattr(identity, name, None)


def _verify_runtime_binding(
    binding: OfficialRuntimeBinding,
    current_identity: Any,
) -> None:
    if current_identity is None:
        raise QualificationError(
            "official ACLNN overlay requires a live runtime identity"
        )
    identity_sha = _runtime_value(current_identity, "identity_sha256")
    if identity_sha != binding.identity_sha256:
        raise QualificationError("official overlay runtime identity drift")
    comparisons = (
        ("asnumpy_version", binding.asnumpy_version),
        ("audit_id", binding.audit_id),
        ("status", binding.status),
        ("module_sha256", binding.module_sha256),
        ("native_module_sha256", binding.native_module_sha256),
        ("cann_version", binding.cann_version),
        ("soc_model", binding.soc_model),
        ("python_version", binding.python_version),
    )
    for name, expected in comparisons:
        if _runtime_value(current_identity, name) != expected:
            raise QualificationError(
                "official overlay runtime %s drift" % name
            )
    wrappers = _runtime_value(current_identity, "wrapper_sha256")
    if not isinstance(wrappers, Mapping):
        raise QualificationError("official overlay runtime wrappers are unavailable")
    if dict(wrappers) != dict(binding.wrapper_sha256):
        raise QualificationError("official overlay runtime wrapper drift")


def _verify_base_manifest(
    base: CapabilityManifest,
    overlay: OfficialCapabilityOverlay,
) -> None:
    if capability_manifest_sha256(base) != overlay.base_manifest_sha256:
        raise QualificationError("official overlay base manifest drift")


def _require_transaction_field(
    transaction: Mapping[str, Any],
    path: Tuple[str, ...],
    expected: Any,
    label: str,
) -> None:
    current: Any = transaction
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise QualificationError("integration transaction is missing %s" % label)
        current = current[part]
    if current != expected:
        raise QualificationError("integration transaction %s mismatch" % label)


def _verify_integration(
    overlay: OfficialCapabilityOverlay,
    evidence_root: str,
) -> Mapping[str, Any]:
    integration = overlay.integration
    transaction = _load_bound_json(
        evidence_root,
        integration.transaction,
        "integration transaction",
    )
    _require_transaction_field(
        transaction, ("record_type",), "numpilot_integration_transaction",
        "record type",
    )
    _require_transaction_field(
        transaction, ("schema_version",), 1, "schema version"
    )
    _require_transaction_field(
        transaction, ("transaction_id",), integration.transaction_id,
        "transaction ID",
    )
    _require_transaction_field(
        transaction, ("transaction_sha256",), integration.transaction_sha256,
        "canonical transaction SHA-256",
    )
    if _transaction_sha256(transaction) != integration.transaction_sha256:
        raise QualificationError(
            "integration transaction content does not reproduce its canonical hash"
        )
    _require_transaction_field(
        transaction, ("gap_id",), overlay.gap_spec.gap_id, "gap ID"
    )
    _require_transaction_field(
        transaction, ("build", "status"), "passed", "build status"
    )
    _require_transaction_field(
        transaction, ("build", "returncode"), 0, "build return code"
    )
    _require_transaction_field(
        transaction, ("build", "wheel_sha256"), integration.wheel.sha256,
        "wheel hash",
    )
    _require_transaction_field(
        transaction, ("gate", "status"), "passed", "gate status"
    )
    _require_transaction_field(
        transaction, ("gate", "returncode"), 0, "gate return code"
    )
    _require_transaction_field(
        transaction, ("gate", "result_sha256"), integration.gate_result.sha256,
        "gate result hash",
    )
    _require_transaction_field(
        transaction, ("gate", "log_sha256"), integration.gate_log.sha256,
        "gate log hash",
    )
    _require_transaction_field(
        transaction, ("candidate_tests", "status"), "passed",
        "candidate test status",
    )
    _require_transaction_field(
        transaction, ("candidate_tests", "returncode"), 0,
        "candidate test return code",
    )
    _require_transaction_field(
        transaction, ("candidate_tests", "log_sha256"),
        integration.candidate_test_log.sha256,
        "candidate test log hash",
    )
    _require_transaction_field(
        transaction, ("source", "tree_sha256"), integration.source_tree_sha256,
        "source tree hash",
    )
    _require_transaction_field(
        transaction, ("source", "file_count"), integration.source_file_count,
        "source file count",
    )
    _require_transaction_field(
        transaction, ("source", "revision"), integration.source_revision,
        "source revision",
    )
    _require_transaction_field(
        transaction, ("runtime", "core_sha256"), integration.runtime_core_sha256,
        "runtime core hash",
    )

    transaction_overlay = transaction.get("source", {}).get("overlay_sha256")
    if not isinstance(transaction_overlay, Mapping) or dict(
        transaction_overlay
    ) != dict(integration.source_overlay_sha256):
        raise QualificationError(
            "integration source overlay hashes do not match the transaction"
        )
    for relative, digest in integration.source_overlay_sha256.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise QualificationError(
                "integration source overlay contains an invalid hash"
            )

    expected_paths = {
        "wheel": integration.wheel,
        "gate_result": integration.gate_result,
        "gate_log": integration.gate_log,
        "build_log": integration.build_log,
        "candidate_test_log": integration.candidate_test_log,
    }
    transaction_paths = {
        "wheel": transaction["build"].get("wheel_relative"),
        "gate_result": transaction.get("gate", {}).get("result_relative"),
        "gate_log": transaction.get("gate", {}).get("log_relative"),
        "build_log": transaction.get("build", {}).get("log_relative"),
        "candidate_test_log": transaction.get("candidate_tests", {}).get(
            "log_relative"
        ),
    }
    for name, ref in expected_paths.items():
        relative = transaction_paths[name]
        if not isinstance(relative, str) or not relative:
            raise QualificationError(
                "integration transaction is missing the %s path" % name
            )
        expected = str(
            PurePosixPath(integration.attempt) / PurePosixPath(relative)
        )
        if ref.path != expected:
            raise QualificationError(
                "official %s path does not match the transaction" % name
            )
        _load_bound_file(evidence_root, ref, "official %s" % name)
    wheel_filename = transaction["build"].get("wheel_filename")
    if wheel_filename != PurePosixPath(integration.wheel.path).name:
        raise QualificationError(
            "official wheel path does not match the transaction filename"
        )

    gap_source_sha256 = overlay.gap_spec.source_sha256
    if overlay.source.sha256 != gap_source_sha256:
        raise QualificationError(
            "official source binding and bound gap source disagree"
        )
    source_payload = _load_bound_file(
        evidence_root,
        OfficialFileRef(
            path=overlay.source.bundle_path,
            sha256=overlay.source.sha256,
        ),
        "official source binding",
    )
    if hashlib.sha256(source_payload).hexdigest() != overlay.source.sha256:
        raise QualificationError("official source binding hash mismatch")

    # The wheel is the integrated artifact, the runtime site is an extraction
    # of that same wheel, and the source overlay is the revision that produced
    # both.  Cross-check the three views instead of trusting one report.
    source_overlay = dict(integration.source_overlay_sha256)
    runtime = overlay.runtime
    bound_wrappers = 0
    for source_relative, source_hash in sorted(source_overlay.items()):
        prefix = "src/asnumpy/"
        if not source_relative.startswith(prefix) or not source_relative.endswith(".py"):
            continue
        wrapper_hash = runtime.wrapper_sha256.get(source_relative[len(prefix):])
        if wrapper_hash is None:
            continue
        if source_hash != wrapper_hash:
            raise QualificationError(
                "integrated source and live runtime disagree on %s"
                % source_relative
            )
        bound_wrappers += 1
    init_hash = source_overlay.get("src/asnumpy/__init__.py")
    if init_hash is None or init_hash != runtime.wrapper_sha256.get("__init__.py"):
        raise QualificationError(
            "integration transaction does not bind the package module"
        )
    if not bound_wrappers:
        raise QualificationError(
            "integration transaction binds no live runtime wrapper"
        )
    if runtime.native_module_sha256 != integration.runtime_core_sha256:
        raise QualificationError(
            "integrated runtime core and live native module disagree"
        )
    return transaction


def validate_official_handoff_evidence(
    request: Mapping[str, Any],
    source_payload: bytes,
    resolution_payload: bytes,
) -> None:
    """Reconstruct a native official handoff from portable, hash-bound bytes.

    Original absolute paths remain identity labels, never locations to read.
    This permits an archived qualification bundle to be checked without its
    creator's /tmp files, while retaining the complete GapSpec and official
    decision bindings. No model, import, build or device execution occurs.
    """
    from numpilot.agents.analysis import AnalysisAgent
    from numpilot.gaps.official_candidate_handoff import (
        SCHEMA, SCHEMA_VERSION, REQUEST_TYPE, STATUS,
        OfficialCandidateHandoffError, _candidate_evidence,
    )
    from numpilot.gaps.official_resolution import (
        OfficialOperatorResolutionReport, OfficialResolutionError,
    )
    from numpilot.gaps.pipeline import GapPreparation
    from numpilot.gaps.spec import blocked_unsupported, build_gap_specs

    expected_fields = frozenset({
        "schema", "schema_version", "request_type", "status", "source_path",
        "source_sha256", "target_soc", "cann_version", "official_resolution_path",
        "official_resolution_sha256", "resolution_id", "gap_specs", "decisions",
        "generated_code", "qualification_performed", "activation_performed",
        "network_access_performed", "system_write_performed", "handoff_id",
    })
    _require_exact_fields(request, expected_fields, "official candidate handoff")
    if (
        request.get("schema") != SCHEMA
        or type(request.get("schema_version")) is not int
        or request.get("schema_version") != SCHEMA_VERSION
        or request.get("request_type") != REQUEST_TYPE
        or request.get("status") != STATUS
    ):
        raise QualificationError("official handoff schema or status is invalid")
    for key in ("generated_code", "qualification_performed", "activation_performed",
                "network_access_performed", "system_write_performed"):
        if request.get(key) is not False:
            raise QualificationError("official handoff has unsafe flag: " + key)
    source_label = request.get("source_path")
    resolution_label = request.get("official_resolution_path")
    for value in (source_label, resolution_label):
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise QualificationError("official handoff identity path must be absolute")
    if _sha256_bytes(source_payload) != request.get("source_sha256"):
        raise QualificationError("official handoff source hash mismatch")
    if _sha256_bytes(resolution_payload) != request.get("official_resolution_sha256"):
        raise QualificationError("official handoff resolution hash mismatch")
    body = {key: value for key, value in request.items() if key != "handoff_id"}
    if _sha256_bytes(_canonical_json(body)) != request.get("handoff_id"):
        raise QualificationError("official handoff canonical identity mismatch")
    try:
        report = OfficialOperatorResolutionReport.from_dict(
            _parse_json(resolution_payload, "official resolution")
        )
        if (
            report.status != "official_candidate_found"
            or report.resolution_id != request.get("resolution_id")
            or report.target_soc != request.get("target_soc")
            or report.cann_version != request.get("cann_version")
        ):
            raise QualificationError("official handoff resolution identity mismatch")
        source = source_payload.decode("utf-8")
        analysis = AnalysisAgent().analyze(source, source_label)
        specs = build_gap_specs(
            source, analysis, target_soc=report.target_soc, cann_version=report.cann_version,
        )
        blocked = tuple(blocked_unsupported(analysis))
        if blocked:
            raise QualificationError("official handoff source has blocked calls")
        preparation = GapPreparation(
            source_path=source_label, source=source, specs=tuple(specs),
            unsupported=tuple(analysis.unsupported), blocked=blocked,
        )
        report.validate_preparation(preparation)
        if request.get("gap_specs") != [spec.to_dict() for spec in specs]:
            raise QualificationError("official handoff GapSpec content mismatch")
        expected_decisions = [_candidate_evidence(item) for item in report.decisions]
        if not expected_decisions or request.get("decisions") != expected_decisions:
            raise QualificationError("official handoff candidate decisions mismatch")
    except QualificationError:
        raise
    except (UnicodeError, SyntaxError, TypeError, ValueError,
            OfficialCandidateHandoffError, OfficialResolutionError) as exc:
        raise QualificationError("official handoff evidence is invalid: %s" % exc) from exc


def _verify_gap(
    overlay: OfficialCapabilityOverlay,
    evidence_root: str,
) -> None:
    request = _load_bound_json(
        evidence_root,
        overlay.gap_spec.request,
        "gap request",
    )
    request_type = request.get("request_type")
    if request_type == "official_candidate_gap":
        resolution_ref = OfficialFileRef(
            path="official-resolution.json",
            sha256=_require_sha256(request.get("official_resolution_sha256"), "official resolution"),
        )
        resolution_payload = _load_bound_file(evidence_root, resolution_ref, "official resolution")
        source_payload = _load_bound_file(
            evidence_root,
            OfficialFileRef(path=overlay.source.bundle_path, sha256=overlay.source.sha256),
            "official source binding",
        )
        validate_official_handoff_evidence(request, source_payload, resolution_payload)
        normalize_soc = lambda value: "".join(ch for ch in str(value).casefold() if ch.isalnum())
        if (
            request.get("cann_version") != overlay.runtime.cann_version
            or normalize_soc(request.get("target_soc")) != normalize_soc(overlay.runtime.soc_model)
        ):
            raise QualificationError("official handoff target and live runtime disagree")
    elif request_type != "script_api_gap":
        raise QualificationError("bound gap request has an unexpected type")
    if request.get("source_sha256") != overlay.gap_spec.source_sha256:
        raise QualificationError("bound gap request source hash mismatch")
    specs = request.get("gap_specs")
    if not isinstance(specs, list):
        raise QualificationError("bound gap request has no gap specifications")
    matches = [
        item
        for item in specs
        if isinstance(item, Mapping)
        and item.get("gap_id") == overlay.gap_spec.gap_id
    ]
    if len(matches) != 1:
        raise QualificationError("bound gap request does not contain the gap ID")
    spec = matches[0]
    if spec.get("source_sha256") != overlay.gap_spec.source_sha256:
        raise QualificationError("bound gap specification source hash mismatch")
    spec_source_path = spec.get("source_path")
    request_source_path = request.get("source_path")
    if (
        not isinstance(spec_source_path, str)
        or not spec_source_path
        or spec_source_path != request_source_path
    ):
        raise QualificationError("bound gap specification source path is invalid")
    variants = spec.get("signature_variants")
    if not isinstance(variants, list) or not variants:
        raise QualificationError("bound gap specification has no signature variants")
    source_api = spec.get("qualified_api")
    entries = {entry.qualified_api: entry for entry in overlay.entries}
    if source_api not in entries:
        raise QualificationError(
            "bound gap API is not admitted by this official overlay"
        )
    if spec.get("asnumpy_api") != entries[source_api].asnumpy_api:
        raise QualificationError("bound gap AsNumpy API is inconsistent")
    admitted = {
        (item.positional_args, item.keyword_args)
        for item in entries[source_api].signature_variants
    }
    for item in variants:
        if not isinstance(item, Mapping):
            raise QualificationError("bound gap signature variant is invalid")
        try:
            signature = OperatorSignature.from_dict(item)
        except ValueError as exc:
            raise QualificationError(
                "bound gap signature variant is invalid"
            ) from exc
        if (signature.positional_args, signature.keyword_args) not in admitted:
            raise QualificationError(
                "official overlay does not cover the bound gap signature"
            )


_GATE_CLAIM_TYPES = {
    "artifact_integrity": {
        "transaction_file_sha256": "sha",
        "transaction_sha256": "sha",
        "wheel_sha256": "sha",
        "runtime_identity_sha256": "sha",
        "source_sha256": "sha",
        "gap_request_sha256": "sha",
        "base_manifest_sha256": "sha",
        "candidate_test_log_sha256": "sha",
        "gate_result_sha256": "sha",
        "gate_log_sha256": "sha",
        "source_overlay_sha256": "sha_map",
        "runtime_core_sha256": "sha",
    },
    "build": {
        "build_status": "text",
        "wheel_sha256": "sha",
        "wheel_member_count": "int",
        "wheel_member_sha256": "sha_map",
    },
    "unit_test": {
        "returncode": "int",
        "passed": "int",
        "failed": "int",
        "errors": "int",
        "junit_sha256": "sha",
        "test_names": "text_list",
    },
    "numerical_validation": {
        "cases": "int",
        "oracle_comparisons": "int",
        "mismatches": "int",
        "elements_compared": "int",
        "oracle": "text",
    },
    "public_asnumpy_api": {
        "api": "text",
        "callable": "bool",
        "signature_variants_verified": "int",
        "rejection_cases": "int",
        "host_fallback_markers": "int",
        "wrapper_sha256": "sha",
    },
    "npu_execution": {
        "device_count": "int",
        "soc_model": "text",
        "native_calls": "int",
        "kernel_launches": "int",
        "repeated_calls": "int",
        "repeated_stable": "bool",
        "host_fallback_calls": "int",
    },
}


def _claim_matches_type(value: Any, kind: str) -> bool:
    if kind == "sha":
        return isinstance(value, str) and bool(_SHA256.fullmatch(value))
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if kind == "bool":
        return type(value) is bool
    if kind == "text":
        return isinstance(value, str) and bool(value) and len(value) <= _MAX_TEXT_CHARS
    if kind == "text_list":
        return (
            isinstance(value, list)
            and bool(value)
            and len(value) <= _MAX_CLAIM_ITEMS
            and all(
                isinstance(item, str) and item and len(item) <= _MAX_TEXT_CHARS
                for item in value
            )
        )
    if kind == "sha_map":
        return (
            isinstance(value, Mapping)
            and bool(value)
            and all(
                isinstance(name, str)
                and name
                and isinstance(digest, str)
                and bool(_SHA256.fullmatch(digest))
                for name, digest in value.items()
            )
        )
    raise AssertionError("unknown claim type")


def _verify_gate_claims(
    gate: str,
    target: str,
    claims: Mapping[str, Any],
    overlay: OfficialCapabilityOverlay,
) -> None:
    required = _GATE_CLAIM_TYPES[gate]
    missing = sorted(set(required) - set(claims))
    if missing:
        raise QualificationError(
            "%s gate target %s is missing claims: %s"
            % (gate, target, ",".join(missing))
        )
    for name, kind in required.items():
        if not _claim_matches_type(claims[name], kind):
            raise QualificationError(
                "%s gate target %s claim %s is invalid"
                % (gate, target, name)
            )
    integration = overlay.integration
    if gate == "artifact_integrity":
        expected = {
            "transaction_file_sha256": integration.transaction.sha256,
            "transaction_sha256": integration.transaction_sha256,
            "wheel_sha256": integration.wheel.sha256,
            "runtime_identity_sha256": overlay.runtime.identity_sha256,
            "source_sha256": overlay.source.sha256,
            "gap_request_sha256": overlay.gap_spec.request.sha256,
            "base_manifest_sha256": overlay.base_manifest_sha256,
            "candidate_test_log_sha256": integration.candidate_test_log.sha256,
            "gate_result_sha256": integration.gate_result.sha256,
            "gate_log_sha256": integration.gate_log.sha256,
            "runtime_core_sha256": integration.runtime_core_sha256,
        }
        for name, value in expected.items():
            if claims[name] != value:
                raise QualificationError(
                    "artifact_integrity claim %s is not bound to the overlay"
                    % name
                )
        if dict(claims["source_overlay_sha256"]) != dict(
            integration.source_overlay_sha256
        ):
            raise QualificationError(
                "artifact_integrity source overlay claims are not bound"
            )
    elif gate == "build":
        if claims["build_status"] != "passed":
            raise QualificationError("build gate did not pass")
        if claims["wheel_sha256"] != integration.wheel.sha256:
            raise QualificationError("build gate wheel hash mismatch")
        members = dict(claims["wheel_member_sha256"])
        wrapper = _OFFICIAL_API_WRAPPER.get(target)
        if wrapper is None:
            raise QualificationError("build gate target has no audited wrapper")
        for member in ("asnumpy/" + wrapper, "asnumpy/__init__.py"):
            if member not in members:
                raise QualificationError(
                    "build gate does not bind wheel member %s" % member
                )
        core_members = [
            name
            for name in members
            if name.startswith("asnumpy/_core") and name.endswith(".so")
        ]
        if not core_members:
            raise QualificationError(
                "build gate does not bind the native wheel member"
            )
        if members["asnumpy/" + wrapper] != overlay.runtime.wrapper_sha256[wrapper]:
            raise QualificationError(
                "build gate wheel wrapper hash disagrees with the runtime"
            )
        if members["asnumpy/__init__.py"] != overlay.runtime.wrapper_sha256[
            "__init__.py"
        ]:
            raise QualificationError(
                "build gate module hash disagrees with the runtime"
            )
        if any(
            members[name] != integration.runtime_core_sha256
            for name in core_members
        ):
            raise QualificationError(
                "build gate native module hash disagrees with the runtime"
            )
    elif gate == "unit_test":
        if claims["failed"] != 0 or claims["errors"] != 0:
            raise QualificationError("unit_test gate contains failures")
        if claims["passed"] < 1 or claims["returncode"] != 0:
            raise QualificationError("unit_test gate did not pass")
        # The gate replays the candidate module's regression suite against the
        # integrated wheel.  Behaviour of the newly admitted API itself is
        # proven by the numerical, public-API and NPU-execution gates; a
        # target-named test only exists when the adapter authored one.
        if target not in _OFFICIAL_API_WRAPPER:
            raise QualificationError("unit_test gate target is not qualified")
        if not claims["test_names"]:
            raise QualificationError(
                "unit_test gate recorded no candidate tests for %s" % target
            )
    elif gate == "numerical_validation":
        if claims["mismatches"] != 0:
            raise QualificationError(
                "numerical_validation gate contains mismatches"
            )
        if claims["cases"] < 1 or claims["oracle_comparisons"] < 1:
            raise QualificationError(
                "numerical_validation gate contains no oracle evidence"
            )
        if claims["oracle"] != "numpy":
            raise QualificationError(
                "numerical_validation oracle is not the NumPy oracle"
            )
    elif gate == "public_asnumpy_api":
        if not claims["callable"]:
            raise QualificationError("public API gate is not callable")
        if claims["api"] != "asnumpy." + target:
            raise QualificationError("public API gate target mismatch")
        if claims["signature_variants_verified"] < 1:
            raise QualificationError("public API gate verified no signature")
        if claims["rejection_cases"] < 2:
            raise QualificationError(
                "public API gate has insufficient rejection evidence"
            )
        if claims["host_fallback_markers"] != 0:
            raise QualificationError(
                "public API wrapper contains a host fallback marker"
            )
        wrapper = _OFFICIAL_API_WRAPPER.get(target)
        if wrapper is None:
            raise QualificationError("public API gate target has no audited wrapper")
        if claims["wrapper_sha256"] != overlay.runtime.wrapper_sha256.get(wrapper):
            raise QualificationError(
                "public API gate wrapper hash disagrees with the runtime"
            )
    elif gate == "npu_execution":
        if claims["device_count"] < 1:
            raise QualificationError("npu_execution gate found no device")
        if claims["soc_model"] != overlay.runtime.soc_model:
            raise QualificationError("npu_execution SoC model mismatch")
        if claims["native_calls"] < 1 or claims["kernel_launches"] < 1:
            raise QualificationError(
                "npu_execution gate proved no native kernel launch"
            )
        if claims["repeated_calls"] < 2 or not claims["repeated_stable"]:
            raise QualificationError(
                "npu_execution gate did not prove repeated-call stability"
            )
        if claims["host_fallback_calls"] != 0:
            raise QualificationError(
                "npu_execution gate observed a host fallback call"
            )


_GATES_REQUIRING_EVIDENCE_FILES = frozenset({
    "unit_test",
    "numerical_validation",
    "public_asnumpy_api",
    "npu_execution",
})


def _verify_gate_evidence_files(
    evidence_root: str,
    gate: str,
    target: str,
    target_report: OfficialGateTarget,
) -> None:
    raw_files = target_report.evidence.get("files")
    if raw_files is None:
        raw_files = []
    if not isinstance(raw_files, (list, tuple)):
        raise QualificationError(
            "%s gate target %s evidence files are invalid" % (gate, target)
        )
    if gate in _GATES_REQUIRING_EVIDENCE_FILES and not raw_files:
        raise QualificationError(
            "%s gate target %s has no evidence file" % (gate, target)
        )
    seen = set()
    refs = []
    for item in raw_files:
        ref = OfficialFileRef.from_dict(item)
        if not ref.path.startswith("gates/"):
            raise QualificationError(
                "%s gate evidence must live under gates/" % gate
            )
        if ref.path in seen:
            raise QualificationError(
                "%s gate contains duplicate evidence files" % gate
            )
        seen.add(ref.path)
        _load_bound_file(evidence_root, ref, "%s gate evidence" % gate)
        refs.append(ref)
    if gate == "unit_test":
        junit_sha256 = target_report.claims["junit_sha256"]
        if not any(ref.sha256 == junit_sha256 for ref in refs):
            raise QualificationError(
                "unit_test gate does not bind the recorded JUnit report"
            )


def _verify_extended_concat_ranks(
    overlay: OfficialCapabilityOverlay,
    target_report: OfficialGateTarget,
    evidence_root: str,
) -> None:
    """Admit ranks beyond the historic rank-2 profile only from raw oracles."""
    entry = next(item for item in overlay.entries if item.registry_key == "concatenate")
    profile = parse_profile_constraints(entry.constraints)
    lower, upper = profile["rank"], profile.get("rank_max")
    if lower == upper == 2:
        return  # Preserve the already-published rank-2 evidence contract.
    if upper is None or lower < 1 or upper > 3:
        raise QualificationError("concatenate rank range exceeds the reviewed oracle profile")
    claimed = target_report.claims.get("tested_ranks")
    if (
        not isinstance(claimed, (tuple, list))
        or any(type(rank) is not int for rank in claimed)
        or sorted(set(claimed)) != list(claimed)
        or not set(range(lower, upper + 1)).issubset(claimed)
    ):
        raise QualificationError("concatenate rank admission lacks tested-rank claims")
    expected_name = "numerical_validation.concatenate.json"
    refs = [
        OfficialFileRef.from_dict(item)
        for item in target_report.evidence.get("files", ())
        if isinstance(item, Mapping) and PurePosixPath(str(item.get("path", ""))).name == expected_name
    ]
    if len(refs) != 1:
        raise QualificationError("concatenate rank admission requires one raw oracle receipt")
    raw = _load_bound_json(evidence_root, refs[0], "concatenate raw oracle evidence")
    if (
        raw.get("schema") != "numpilot.official-aclnn-gate-evidence"
        or type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1
        or raw.get("gate") != "numerical_validation"
        or raw.get("target") != "concatenate" or raw.get("oracle") != "numpy"
    ):
        raise QualificationError("concatenate raw oracle schema is invalid")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise QualificationError("concatenate rank admission has no raw cases")
    observed = set()
    comparisons = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise QualificationError("concatenate raw oracle case is invalid")
        shape = case.get("shape")
        count = case.get("comparisons")
        if (
            case.get("ok") is not True or case.get("dtype") != "float32"
            or not isinstance(shape, list) or not shape
            or any(type(extent) is not int or extent <= 0 for extent in shape)
            or type(count) is not int or count < 3
        ):
            raise QualificationError("concatenate raw oracle case did not pass")
        observed.add(len(shape))
        comparisons += count
    if sorted(observed) != list(claimed):
        raise QualificationError("concatenate tested ranks disagree with raw oracle cases")
    totals = raw.get("totals")
    if not isinstance(totals, Mapping):
        raise QualificationError("concatenate raw oracle totals are missing")
    for key, value in (("cases", len(cases)), ("oracle_comparisons", comparisons), ("mismatches", 0)):
        if totals.get(key) != value or target_report.claims.get(key) != value:
            raise QualificationError("concatenate raw oracle totals disagree: " + key)
    if totals.get("tested_ranks") != list(claimed):
        raise QualificationError("concatenate raw oracle rank totals disagree")


def _verify_integrated_cat_kernel_profile(
    overlay: OfficialCapabilityOverlay,
    target_report: OfficialGateTarget,
    evidence_root: str,
) -> None:
    """Recount observed Cat kernels; Python dispatch counts are not kernels."""
    import csv
    import io
    import math

    entry = next(item for item in overlay.entries if item.registry_key == "concatenate")
    parsed = parse_profile_constraints(entry.constraints)
    if parsed["rank"] == parsed.get("rank_max") == 2 and not target_report.claims.get("execution_profile"):
        return  # Historic producer contract; do not reinterpret archived reports.
    claims = target_report.claims
    if (
        claims.get("execution_profile") != "official_concat_rank1_3_aclnn_v1"
        or claims.get("kernel_evidence_scope") != "independent_profile"
        or type(claims.get("device_count")) is not int or claims.get("device_count") != 1
    ):
        raise QualificationError("integrated concatenate requires an independent single-NPU profile")
    refs = [OfficialFileRef.from_dict(item) for item in target_report.evidence.get("files", ())]
    manifests = [ref for ref in refs if PurePosixPath(ref.path).name == "kernel-profile.json"]
    if len(manifests) != 1:
        raise QualificationError("integrated concatenate requires one kernel profile manifest")
    manifest_ref = manifests[0]
    profile = _load_bound_json(evidence_root, manifest_ref, "Cat kernel profile")
    if (
        profile.get("schema") != "numpilot.concatenate-kernel-profile.v1"
        or profile.get("api") != "numpy.concatenate" or profile.get("status") != "passed"
        or profile.get("performance_claim") is not False
    ):
        raise QualificationError("Cat kernel profile schema or status is invalid")
    expected = {
        "transaction_sha256": overlay.integration.transaction_sha256,
        "transaction_file_sha256": overlay.integration.transaction.sha256,
        "runtime_core_sha256": overlay.integration.runtime_core_sha256,
        "runtime_module_sha256": overlay.runtime.module_sha256,
        "runtime_array_sha256": overlay.runtime.wrapper_sha256["array.py"],
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise QualificationError("Cat kernel profile binding mismatch: " + key)
    by_path = {ref.path: ref.sha256 for ref in refs}
    profile_root = PurePosixPath(manifest_ref.path).parent

    def load_member(raw):
        if not isinstance(raw, Mapping):
            raise QualificationError("Cat profile member reference is invalid")
        relative = _safe_relative_path(raw.get("path"), "Cat profile member")
        ref = OfficialFileRef(str(profile_root / relative), raw.get("sha256"))
        if by_path.get(ref.path) != ref.sha256:
            raise QualificationError("Cat profile member is not bound by the gate report")
        return _load_bound_file(evidence_root, ref, "Cat profile member")

    worker_ref = profile.get("worker")
    load_member(worker_ref)
    execution = _parse_json(load_member(profile.get("execution")), "Cat execution receipt")
    if (
        execution.get("schema") != "numpilot.cat-profile-worker.v1"
        or execution.get("status") != "passed"
        or execution.get("worker_sha256") != worker_ref.get("sha256")
        or execution.get("transaction_sha256") != overlay.integration.transaction.sha256
    ):
        raise QualificationError("Cat worker identity mismatch")
    runtime = execution.get("runtime")
    if not isinstance(runtime, Mapping):
        raise QualificationError("Cat worker runtime is missing")
    for key, digest in (
        ("core", overlay.integration.runtime_core_sha256),
        ("module", overlay.runtime.module_sha256),
        ("array", overlay.runtime.wrapper_sha256["array.py"]),
    ):
        item = runtime.get(key)
        if not isinstance(item, Mapping) or item.get("sha256") != digest:
            raise QualificationError("Cat worker runtime binding mismatch")
    cases = execution.get("cases")
    if not isinstance(cases, list) or not cases:
        raise QualificationError("Cat worker has no numerical cases")
    observed_ranks = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise QualificationError("Cat worker case is invalid")
        rank, shape, axis = case.get("rank"), case.get("shape"), case.get("axis")
        if (
            type(rank) is not int or rank not in (2, 3)
            or not isinstance(shape, list) or len(shape) != rank
            or any(type(n) is not int or n <= 0 for n in shape)
            or type(axis) is not int or not -rank <= axis < rank
            or case.get("dtype") != "float32" or case.get("exact_equal") is not True
        ):
            raise QualificationError("Cat worker numerical profile did not pass")
        observed_ranks.append(rank)
    if any(observed_ranks.count(rank) < 2 for rank in (2, 3)) or profile.get("tested_ranks") != [2, 3]:
        raise QualificationError("Cat profile must repeat rank-2 and rank-3 cases")
    dispatch = execution.get("dispatch")
    if not isinstance(dispatch, Mapping):
        raise QualificationError("Cat worker dispatch is missing")
    if (
        type(dispatch.get("legacy_calls")) is not int or dispatch.get("legacy_calls") != 0
        or type(dispatch.get("native_calls")) is not int
        or dispatch.get("native_calls") != len(cases)
        or profile.get("native_calls") != len(cases)
    ):
        raise QualificationError("Cat worker native dispatch count is invalid")
    traces = profile.get("traces")
    if not isinstance(traces, list) or not traces:
        raise QualificationError("Cat profile raw traces are missing")
    seen, count, type_counts = set(), 0, {}
    for trace in traces:
        if not isinstance(trace, Mapping) or trace.get("path") in seen:
            raise QualificationError("Cat profile repeats or omits a trace path")
        seen.add(trace.get("path"))
        try:
            rows = list(csv.DictReader(io.StringIO(load_member(trace).decode("utf-8-sig"))))
        except (UnicodeError, csv.Error) as exc:
            raise QualificationError("Cat profile raw trace is unreadable") from exc
        if type(trace.get("rows")) is not int or trace.get("rows") != len(rows):
            raise QualificationError("Cat profile raw trace row count mismatch")
        for row in rows:
            op_type = row.get("OP Type", "")
            type_counts[op_type] = type_counts.get(op_type, 0) + 1
            if op_type == "ConcatD" and row.get("Op Name", "").startswith("aclnnCat_ConcatD_"):
                if row.get("Task Type") != "AI_VECTOR_CORE" or row.get("Device_id") != "0":
                    raise QualificationError("Cat kernel profile task or device is invalid")
                try:
                    duration = float(row.get("Task Duration(us)", ""))
                except (TypeError, ValueError) as exc:
                    raise QualificationError("Cat kernel profile duration is invalid") from exc
                if not math.isfinite(duration) or duration <= 0:
                    raise QualificationError("Cat kernel profile duration must be finite and positive")
                count += 1
    if (
        count < 1 or type(profile.get("cat_kernel_rows")) is not int
        or count != profile.get("cat_kernel_rows") or count != claims.get("kernel_launches")
        or profile.get("op_type_counts") != type_counts
    ):
        raise QualificationError("Cat kernel launch claims disagree with raw trace rows")


def _load_and_verify_gates(
    overlay: OfficialCapabilityOverlay,
    evidence_root: str,
) -> Dict[str, OfficialGateReport]:
    reports = {}
    entry_keys = frozenset(overlay.entry_keys)
    producers = {overlay.producer}
    for gate in OFFICIAL_GATES:
        ref = overlay.gates[gate]
        stored = _load_bound_file(
            evidence_root, ref, "official %s gate" % gate
        )
        report = OfficialGateReport.from_dict(
            _parse_json(stored, "official %s gate" % gate)
        )
        if report.gate != gate:
            raise QualificationError("official gate report name mismatch")
        if report.producer not in producers:
            raise QualificationError(
                "official gate report producer is not the overlay producer"
            )
        if report.status != "passed":
            raise QualificationError("official gate report did not pass")
        if frozenset(report.targets) != entry_keys:
            raise QualificationError(
                "official gate report targets do not match the overlay entries"
            )
        for target, target_report in report.targets.items():
            if target_report.status != "passed":
                raise QualificationError(
                    "official %s gate target %s did not pass" % (gate, target)
                )
            _verify_gate_claims(gate, target, target_report.claims, overlay)
            _verify_gate_evidence_files(
                evidence_root, gate, target, target_report
            )
            if gate == "numerical_validation" and target == "concatenate":
                _verify_extended_concat_ranks(overlay, target_report, evidence_root)
            if gate == "npu_execution" and target == "concatenate":
                _verify_integrated_cat_kernel_profile(overlay, target_report, evidence_root)
        reports[gate] = report
    return reports


def apply_official_overlay(
    base: CapabilityManifest,
    overlay: OfficialCapabilityOverlay,
    *,
    evidence_root: str,
    trusted_producers: Sequence[str],
    current_identity: Any,
) -> CapabilityManifest:
    """Validate every official binding, then return one atomic manifest."""

    overlay = OfficialCapabilityOverlay.from_dict(overlay.to_dict())
    trusted = frozenset(str(item) for item in trusted_producers)
    if overlay.producer not in trusted:
        raise QualificationError("official overlay producer is not trusted")
    _verify_runtime_binding(overlay.runtime, current_identity)
    _verify_base_manifest(base, overlay)
    _verify_integration(overlay, evidence_root)
    _verify_gap(overlay, evidence_root)
    _load_and_verify_gates(overlay, evidence_root)

    evidence = "official ACLNN qualification %s" % overlay.qualification_id
    additions = {}
    for entry in overlay.entries:
        existing = base.operators.get(entry.registry_key)
        if existing is None or not existing.available:
            raise QualificationError(
                "official overlay API is absent from the runtime"
            )
        if existing.provenance is ExecutionProvenance.NATIVE_NPU:
            if existing.evidence != evidence:
                raise QualificationError(
                    "official overlay conflicts with an existing qualification"
                )
        elif existing.provenance is not ExecutionProvenance.UNKNOWN:
            raise QualificationError(
                "official overlay cannot replace an audited capability"
            )
        additions[entry.registry_key] = OperatorCapability(
            name=entry.registry_key,
            available=True,
            provenance=ExecutionProvenance.NATIVE_NPU,
            evidence=evidence,
            constraints=entry.constraints,
            signature_variants=entry.signature_variants,
        )
    operators = dict(base.operators)
    operators.update(additions)
    suffix = overlay.qualification_id[len(OFFICIAL_QUALIFICATION_PREFIX):]
    return CapabilityManifest(
        backend=base.backend,
        backend_version=base.backend_version,
        operators=operators,
        environment=dict(base.environment),
        schema_version=base.schema_version,
        audit_id="%s+%s%s" % (base.audit_id, OFFICIAL_OVERLAY_AUDIT_TAG, suffix),
    )


def _write_private_file(path: Path, payload: bytes, *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_official_overlay_atomic(
    path: str,
    overlay: OfficialCapabilityOverlay,
    *,
    expected_previous_sha256: str = ZERO_SHA256,
) -> str:
    """Publish one official overlay with a locked compare-and-swap."""

    import fcntl

    expected = _require_sha256(
        expected_previous_sha256, "official expected previous overlay"
    )
    validated = OfficialCapabilityOverlay.from_dict(overlay.to_dict())
    payload = validated.to_bytes()
    target = Path(path)
    parent = target.parent
    if parent.exists():
        _validate_private_directory(parent, "official overlay parent")
    else:
        try:
            parent.mkdir(parents=True, mode=0o700)
        except OSError as exc:
            raise QualificationError(
                "cannot create official overlay parent"
            ) from exc
        _validate_private_directory(parent, "official overlay parent")

    lock_path = parent / ("." + target.name + ".lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        lock_fd = os.open(str(lock_path), flags, 0o600)
    except OSError as exc:
        raise QualificationError("cannot open official overlay lock") from exc
    temp_name = ""
    try:
        lock_stat = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or lock_stat.st_mode & 0o022
        ):
            raise QualificationError(
                "official overlay lock must be a private, single-link regular file"
            )
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        previous_payload = None
        if os.path.lexists(target):
            previous_payload = _read_private_file(
                target, "existing official overlay", max_bytes=_MAX_JSON_BYTES
            )
            previous_sha = _sha256_bytes(previous_payload)
            if expected == ZERO_SHA256:
                raise QualificationError(
                    "official overlay already exists; expected no previous file"
                )
            if previous_sha != expected:
                raise QualificationError(
                    "official overlay compare-and-swap failed"
                )
            previous = OfficialCapabilityOverlay.from_dict(
                _parse_json(previous_payload, "existing official overlay")
            )
            if validated.parent_sha256 != previous_sha:
                raise QualificationError(
                    "official overlay parent SHA-256 does not match"
                )
            if validated.revision != previous.revision + 1:
                raise QualificationError(
                    "official overlay revision is not monotonic"
                )
            if previous_payload == payload:
                return previous_sha
        else:
            if expected != ZERO_SHA256:
                raise QualificationError(
                    "official overlay compare-and-swap expected a previous file"
                )
            if (
                validated.revision != 1
                or validated.parent_sha256 != ZERO_SHA256
            ):
                raise QualificationError(
                    "first official overlay must start at revision 1"
                )

        for counter in range(100):
            candidate = parent / (
                ".%s.%d.%d.tmp" % (target.name, os.getpid(), counter)
            )
            if not os.path.lexists(candidate):
                temp_name = candidate.name
                break
        else:
            raise QualificationError("cannot allocate official overlay temporary file")
        _write_private_file(parent / temp_name, payload, exclusive=True)
        os.replace(str(parent / temp_name), str(target))
        temp_name = ""
        parent_descriptor = os.open(
            str(parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        published = _read_private_file(
            target, "published official overlay", max_bytes=_MAX_JSON_BYTES
        )
        published_sha = _sha256_bytes(published)
        if published != payload:
            raise QualificationError(
                "official overlay post-publish verification failed"
            )
        return published_sha
    finally:
        if temp_name:
            try:
                os.unlink(str(parent / temp_name))
            except OSError:
                pass
        os.close(lock_fd)
