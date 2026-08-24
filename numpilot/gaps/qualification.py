"""Fail-closed admission of externally produced operator qualifications.

This module deliberately does not build, import, or execute generated code.  A
trusted, out-of-process harness owns build and NPU validation; NumPilot only
checks its bounded JSON reports, content hashes, runtime binding, and publishes
references through an atomic capability overlay.
"""

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    OperatorSignature,
    parse_profile_constraints,
)
from numpilot.gaps.models import (
    ARTIFACT_SCHEMA_VERSION,
    GAP_SCHEMA_VERSION,
    GapSpec,
)


QUALIFICATION_SCHEMA_VERSION = 1
GATE_REPORT_SCHEMA_VERSION = 2
OVERLAY_SCHEMA_VERSION = 1
ZERO_SHA256 = "0" * 64
REQUIRED_QUALIFICATION_GATES = frozenset({
    "artifact_integrity",
    "build",
    "unit_test",
    "numerical_validation",
    "public_asnumpy_api",
    "npu_execution",
})
DEFAULT_TRUSTED_PRODUCERS = frozenset({
    "numpilot-qualification-harness-v1",
})

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUALIFICATION_ID = re.compile(r"^qual-[0-9a-f]{20}$")
_REGISTRY_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_BOUND_FILE_BYTES = 64 * 1024 * 1024
_MAX_RUNTIME_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_PATH_CHARS = 512
_MAX_PATH_DEPTH = 16
_MAX_FILES = 256
_MAX_CONSTRAINTS = 128
_MAX_ARTIFACT_TOTAL_BYTES = 2 * 1024 * 1024
_MAX_ARTIFACT_TREE_ENTRIES = _MAX_FILES * (_MAX_PATH_DEPTH + 1)
_QUALIFICATION_FIELDS = frozenset({
    "schema_version",
    "qualification_id",
    "gap_id",
    "registry_key",
    "qualified_api",
    "asnumpy_api",
    "gap_spec_path",
    "gap_spec_sha256",
    "source_path",
    "source_sha256",
    "artifact_manifest_path",
    "artifact_manifest_sha256",
    "artifact_id",
    "artifact_files",
    "runtime_artifacts",
    "runtime",
    "constraints",
    "gates",
})
_GATE_REPORT_FIELDS = frozenset({
    "schema_version",
    "gate",
    "status",
    "producer",
    "artifact_id",
    "gap_spec_sha256",
    "artifact_manifest_sha256",
    "constraints_sha256",
    "target",
    "claims",
})
_GAP_SPEC_FIELDS = frozenset({
    "schema_version",
    "gap_id",
    "api_name",
    "qualified_api",
    "asnumpy_api",
    "source_path",
    "source_sha256",
    "callsites",
    "signature_variants",
    "target_soc",
    "cann_version",
    "route",
    "route_decisions",
    "shape_rules",
    "dtype_rules",
    "semantic_source",
    "reference_cases",
})
_GAP_CALLSITE_FIELDS = frozenset({
    "lineno",
    "col_offset",
    "qualified_api",
    "call_expr",
    "reason",
})
_GAP_SIGNATURE_FIELDS = frozenset({
    "positional_args",
    "keyword_args",
})
_ARTIFACT_FIELDS = frozenset({
    "schema_version",
    "artifact_id",
    "gap_id",
    "workspace",
    "summary",
    "status",
    "files",
    "generation",
    "gates",
})
_UNVERIFIED_ARTIFACT_GATES = {
    "spec": "prepared",
    "generation": "passed",
    "build": "pending",
    "unit_test": "pending",
    "numerical_validation": "pending",
    "activation": "pending",
}
_GENERATION_REQUIRED_FIELDS = frozenset({
    "mode",
    "provider",
    "model",
    "attempt_count",
    "response_sha256",
    "local_validation",
    "fallback",
})
_GATE_CLAIM_FIELDS = {
    "artifact_integrity": frozenset({"verified_files", "mismatches"}),
    "build": frozenset({"exit_code", "output_artifacts"}),
    "unit_test": frozenset({
        "passed",
        "failed",
        "covered_signature_variants",
        "signature_variants_sha256",
    }),
    "numerical_validation": frozenset({
        "oracle",
        "cases",
        "failed",
        "cpu_fallback_detected",
        "covered_signature_variants",
        "signature_variants_sha256",
    }),
    "public_asnumpy_api": frozenset({
        "api",
        "callable",
        "cpu_fallback_detected",
        "covered_signature_variants",
        "signature_variants_sha256",
    }),
    "npu_execution": frozenset({
        "backend",
        "device_count",
        "kernel_launches",
        "cpu_fallback_detected",
        "covered_signature_variants",
        "signature_variants_sha256",
    }),
}
_SIGNATURE_COVERAGE_GATES = frozenset({
    "unit_test",
    "numerical_validation",
    "public_asnumpy_api",
    "npu_execution",
})


class QualificationError(RuntimeError):
    """Qualification or overlay evidence failed closed."""


class OverlayDurabilityError(QualificationError):
    """The new overlay is visible, but crash durability could not be confirmed."""

    def __init__(self, published_sha256: str):
        self.published_sha256 = published_sha256
        super().__init__(
            "capability overlay was replaced, but directory fsync failed; "
            "the new overlay may already be visible"
        )


class OverlayPostPublishError(OverlayDurabilityError):
    """A post-replace cleanup failed after the new overlay became visible."""

    def __init__(self, published_sha256: str, phase: str):
        self.phase = phase
        self.published_sha256 = published_sha256
        QualificationError.__init__(
            self,
            "capability overlay was replaced, but %s failed; "
            "the new overlay may already be visible" % phase,
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def capability_manifest_sha256(manifest: CapabilityManifest) -> str:
    """Return the stable identity used to bind an overlay to its base."""

    return _sha256_bytes(_canonical_json(manifest.to_dict()))


def qualification_constraints_sha256(constraints: Sequence[str]) -> str:
    """Return the identity gate reports use to bind the admitted input scope."""

    return _sha256_bytes(
        _canonical_json({"constraints": [str(item) for item in constraints]})
    )


def qualification_signatures_sha256(
    signatures: Sequence[OperatorSignature],
) -> str:
    """Return the canonical identity of the exact admitted signature set.

    Signature variants are an unordered admission set.  Sorting their already
    validated typed representation keeps the evidence binding stable if a
    producer emits the same variants in a different order.
    """

    if isinstance(signatures, (str, bytes)) or not isinstance(
        signatures, Sequence
    ):
        raise QualificationError("signature variants must be a sequence")
    normalized = tuple(signatures)
    if not normalized or any(
        not isinstance(signature, OperatorSignature)
        for signature in normalized
    ):
        raise QualificationError(
            "signature variants must contain typed OperatorSignature values"
        )
    if len(set(normalized)) != len(normalized):
        raise QualificationError("signature variants contain duplicates")
    canonical = sorted(
        normalized,
        key=lambda signature: (
            signature.positional_args,
            signature.keyword_args,
        ),
    )
    return _sha256_bytes(_canonical_json({
        "signature_variants": [
            signature.to_dict() for signature in canonical
        ],
    }))


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError("JSON contains duplicate key: %s" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise QualificationError("JSON contains non-standard constant: %s" % value)


def _parse_json(payload: bytes, label: str) -> Mapping[str, Any]:
    if len(payload) > _MAX_JSON_BYTES:
        raise QualificationError("%s exceeds the JSON size limit" % label)
    try:
        data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except QualificationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QualificationError("%s is not valid JSON" % label) from exc
    if not isinstance(data, dict):
        raise QualificationError("%s must contain a JSON object" % label)
    return data


def _require_exact_fields(
    data: Mapping[str, Any], expected: frozenset, label: str
) -> None:
    actual = frozenset(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise QualificationError(
            "%s fields are invalid (%s)" % (label, "; ".join(details))
        )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise QualificationError("%s must be a lowercase SHA-256" % label)
    return value


def _require_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError("%s must be an integer" % label)
    return value


def _require_known_text(value: object, label: str, *, limit: int = 512) -> str:
    if not isinstance(value, str):
        raise QualificationError("%s must be text" % label)
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > limit
        or normalized.lower() in {"unknown", "unset", "none", "n/a"}
    ):
        raise QualificationError("%s must be concrete and bounded" % label)
    return normalized


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise QualificationError("%s must be a non-empty relative path" % label)
    if (
        len(value) > _MAX_PATH_CHARS
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise QualificationError("%s is unsafe" % label)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or len(path.parts) > _MAX_PATH_DEPTH
    ):
        raise QualificationError("%s is unsafe" % label)
    return path


def _validate_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise QualificationError("evidence root must be a real directory")
    try:
        mode = root.stat().st_mode
    except OSError as exc:
        raise QualificationError("cannot inspect evidence root") from exc
    if mode & 0o022:
        raise QualificationError(
            "evidence root must not be group/world writable"
        )


def _validate_private_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise QualificationError("%s must be a real directory" % label)
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise QualificationError("cannot inspect %s" % label) from exc
    if mode & 0o022:
        raise QualificationError("%s must not be group/world writable" % label)


def _bound_path(root: Path, relative: PurePosixPath, label: str) -> Path:
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise QualificationError("%s must not traverse a symlink" % label)
        if index < len(relative.parts) - 1:
            _validate_private_directory(current, label + " parent")
    try:
        mode = current.stat().st_mode
    except OSError as exc:
        raise QualificationError("%s is missing" % label) from exc
    if not stat.S_ISREG(mode):
        raise QualificationError("%s must be a regular file" % label)
    if mode & 0o022:
        raise QualificationError("%s must not be group/world writable" % label)
    return current


def _read_private_regular_file(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one bounded file from the descriptor that was actually checked."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        # Avoid blocking if a path is raced to a FIFO before ``open``.  This
        # flag has no effect on regular-file reads.
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise QualificationError("cannot open %s" % label) from exc

    primary_error = None
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise QualificationError("%s must be a regular file" % label)
        if file_stat.st_mode & 0o022:
            raise QualificationError(
                "%s must not be group/world writable" % label
            )
        if file_stat.st_size > max_bytes:
            raise QualificationError("%s exceeds the size limit" % label)
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise QualificationError("%s exceeds the size limit" % label)
        return b"".join(chunks)
    except QualificationError as exc:
        primary_error = exc
        raise
    except OSError as exc:
        primary_error = exc
        raise QualificationError("cannot read %s" % label) from exc
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            if primary_error is None:
                raise QualificationError("cannot close %s" % label) from exc


def _read_bound_file(
    root: Path,
    value: object,
    expected_sha256: object,
    label: str,
    *,
    max_bytes: int = _MAX_BOUND_FILE_BYTES,
) -> Tuple[Path, bytes]:
    relative = _safe_relative_path(value, label + " path")
    path = _bound_path(root, relative, label)
    payload = _read_private_regular_file(path, label, max_bytes=max_bytes)
    expected = _require_sha256(expected_sha256, label + " SHA-256")
    if _sha256_bytes(payload) != expected:
        raise QualificationError("%s hash mismatch" % label)
    return path, payload


def _normalize_soc(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _constraints_from_gap_spec(spec: GapSpec) -> Tuple[str, ...]:
    """Map the concrete v1 GapSpec scope to canonical runtime constraints."""

    if tuple(spec.dtype_rules) != ("float32",):
        raise QualificationError(
            "GapSpec dtype rules are not a concrete v1 float32 scope"
        )
    rank_rules = [
        rule for rule in spec.shape_rules
        if isinstance(rule, str) and re.fullmatch(r"rank>=(0|[1-9][0-9]*)", rule)
    ]
    contiguous_rules = [
        rule for rule in spec.shape_rules
        if rule in {"contiguous", "contiguous=true"}
    ]
    numel_rules = [
        rule for rule in spec.shape_rules
        if isinstance(rule, str)
        and re.fullmatch(r"numel>=(0|[1-9][0-9]*)", rule)
    ]
    if (
        len(rank_rules) != 1
        or len(contiguous_rules) != 1
        or len(numel_rules) > 1
        or len(spec.shape_rules) != 2 + len(numel_rules)
    ):
        raise QualificationError(
            "GapSpec shape rules are not a concrete v1 "
            "rank/contiguous/optional-numel scope"
        )
    constraints = (
        "dtype=float32",
        rank_rules[0],
        "contiguous=true",
    ) + tuple(numel_rules)
    try:
        parse_profile_constraints(constraints)
    except ValueError as exc:
        raise QualificationError("GapSpec profile scope is invalid") from exc
    return constraints


@dataclass(frozen=True)
class GateEvidence:
    name: str
    status: str
    report_path: str
    report_sha256: str
    producer: str
    format_version: int = GATE_REPORT_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "GateEvidence":
        if not isinstance(data, dict):
            raise QualificationError("gate %s must be an object" % name)
        _require_exact_fields(
            data,
            frozenset({
                "status",
                "report_path",
                "report_sha256",
                "producer",
                "format_version",
            }),
            "gate %s" % name,
        )
        if name not in REQUIRED_QUALIFICATION_GATES:
            raise QualificationError("unknown qualification gate: %s" % name)
        if data["status"] != "passed":
            raise QualificationError("gate %s did not pass" % name)
        version = _require_exact_int(
            data["format_version"], "gate %s format version" % name
        )
        if version != GATE_REPORT_SCHEMA_VERSION:
            raise QualificationError("gate %s format version is unsupported" % name)
        report_path = str(_safe_relative_path(
            data["report_path"], "gate %s report" % name
        ))
        return cls(
            name=name,
            status="passed",
            report_path=report_path,
            report_sha256=_require_sha256(
                data["report_sha256"], "gate %s report" % name
            ),
            producer=_require_known_text(
                data["producer"], "gate %s producer" % name, limit=128
            ),
            format_version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "report_path": self.report_path,
            "report_sha256": self.report_sha256,
            "producer": self.producer,
            "format_version": self.format_version,
        }


@dataclass(frozen=True)
class RuntimeBinding:
    backend: str
    backend_version: str
    backend_audit_id: str
    backend_module_sha256: str
    cann_version: str
    soc_version: str
    python_version: str
    opp_manifest_path: str
    opp_manifest_sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeBinding":
        if not isinstance(data, dict):
            raise QualificationError("runtime binding must be an object")
        expected = frozenset({
            "backend",
            "backend_version",
            "backend_audit_id",
            "backend_module_sha256",
            "cann_version",
            "soc_version",
            "python_version",
            "opp_manifest_path",
            "opp_manifest_sha256",
        })
        _require_exact_fields(data, expected, "runtime binding")
        if data["backend"] != "npu":
            raise QualificationError("qualified runtime backend must be npu")
        return cls(
            backend="npu",
            backend_version=_require_known_text(
                data["backend_version"], "runtime backend_version", limit=256
            ),
            backend_audit_id=_require_known_text(
                data["backend_audit_id"], "runtime backend_audit_id", limit=256
            ),
            backend_module_sha256=_require_sha256(
                data["backend_module_sha256"], "runtime backend module"
            ),
            cann_version=_require_known_text(
                data["cann_version"], "runtime CANN version", limit=128
            ),
            soc_version=_require_known_text(
                data["soc_version"], "runtime SoC version", limit=128
            ),
            python_version=_require_known_text(
                data["python_version"], "runtime Python version", limit=128
            ),
            opp_manifest_path=str(_safe_relative_path(
                data["opp_manifest_path"], "runtime OPP manifest"
            )),
            opp_manifest_sha256=_require_sha256(
                data["opp_manifest_sha256"], "runtime OPP manifest"
            ),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "backend_audit_id": self.backend_audit_id,
            "backend_module_sha256": self.backend_module_sha256,
            "cann_version": self.cann_version,
            "soc_version": self.soc_version,
            "python_version": self.python_version,
            "opp_manifest_path": self.opp_manifest_path,
            "opp_manifest_sha256": self.opp_manifest_sha256,
        }


def _parse_hash_map(value: object, label: str) -> Dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise QualificationError("%s must be a non-empty object" % label)
    if len(value) > _MAX_FILES:
        raise QualificationError("%s contains too many files" % label)
    result = {}
    for raw_path, raw_digest in value.items():
        path = str(_safe_relative_path(raw_path, label + " file"))
        if path in result:
            raise QualificationError("%s contains duplicate paths" % label)
        result[path] = _require_sha256(raw_digest, "%s %s" % (label, path))
    return result


@dataclass(frozen=True)
class QualificationEvidence:
    qualification_id: str
    gap_id: str
    registry_key: str
    qualified_api: str
    asnumpy_api: str
    gap_spec_path: str
    gap_spec_sha256: str
    source_path: str
    source_sha256: str
    artifact_manifest_path: str
    artifact_manifest_sha256: str
    artifact_id: str
    artifact_files: Mapping[str, str]
    runtime_artifacts: Mapping[str, str]
    runtime: RuntimeBinding
    constraints: Tuple[str, ...]
    gates: Mapping[str, GateEvidence]
    schema_version: int = QUALIFICATION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualificationEvidence":
        if not isinstance(data, dict):
            raise QualificationError("qualification evidence must be an object")
        _require_exact_fields(data, _QUALIFICATION_FIELDS, "qualification")
        version = _require_exact_int(
            data["schema_version"], "qualification schema"
        )
        if version != QUALIFICATION_SCHEMA_VERSION:
            raise QualificationError("qualification schema is unsupported")
        qualification_id = str(data["qualification_id"])
        if not _QUALIFICATION_ID.fullmatch(qualification_id):
            raise QualificationError("qualification_id is invalid")
        if not isinstance(data["registry_key"], str):
            raise QualificationError("registry_key must be text")
        registry_key = data["registry_key"]
        if not _REGISTRY_KEY.fullmatch(registry_key):
            raise QualificationError("registry_key is invalid")
        if data["qualified_api"] not in {
            "numpy." + registry_key,
            "operator." + registry_key,
        }:
            raise QualificationError("qualified API does not match registry_key")
        if data["asnumpy_api"] != "asnumpy." + registry_key:
            raise QualificationError("AsNumpy API does not match registry_key")
        raw_constraints = data["constraints"]
        if not isinstance(raw_constraints, list) or not raw_constraints:
            raise QualificationError("qualification constraints must be non-empty")
        if len(raw_constraints) > _MAX_CONSTRAINTS:
            raise QualificationError("qualification contains too many constraints")
        constraints = tuple(
            _require_known_text(item, "qualification constraint", limit=512)
            for item in raw_constraints
        )
        if len(set(constraints)) != len(constraints):
            raise QualificationError("qualification constraints contain duplicates")
        if any(
            marker in item.lower()
            for item in constraints
            for marker in ("runtime_profile_required", "unrestricted", "all inputs")
        ):
            raise QualificationError("qualification constraints are not concrete")
        raw_gates = data["gates"]
        if not isinstance(raw_gates, dict):
            raise QualificationError("qualification gates must be an object")
        if frozenset(raw_gates) != REQUIRED_QUALIFICATION_GATES:
            raise QualificationError("qualification gate set is incomplete or unknown")
        gates = {
            str(name): GateEvidence.from_dict(str(name), gate)
            for name, gate in raw_gates.items()
        }
        evidence = cls(
            schema_version=version,
            qualification_id=qualification_id,
            gap_id=_require_known_text(data["gap_id"], "gap_id", limit=128),
            registry_key=registry_key,
            qualified_api=str(data["qualified_api"]),
            asnumpy_api=str(data["asnumpy_api"]),
            gap_spec_path=str(_safe_relative_path(
                data["gap_spec_path"], "gap spec"
            )),
            gap_spec_sha256=_require_sha256(
                data["gap_spec_sha256"], "gap spec"
            ),
            source_path=str(_safe_relative_path(
                data["source_path"], "source file"
            )),
            source_sha256=_require_sha256(
                data["source_sha256"], "source file"
            ),
            artifact_manifest_path=str(_safe_relative_path(
                data["artifact_manifest_path"], "artifact manifest"
            )),
            artifact_manifest_sha256=_require_sha256(
                data["artifact_manifest_sha256"], "artifact manifest"
            ),
            artifact_id=_require_known_text(
                data["artifact_id"], "artifact_id", limit=128
            ),
            artifact_files=_parse_hash_map(
                data["artifact_files"], "artifact files"
            ),
            runtime_artifacts=_parse_hash_map(
                data["runtime_artifacts"], "runtime artifacts"
            ),
            runtime=RuntimeBinding.from_dict(data["runtime"]),
            constraints=constraints,
            gates=gates,
        )
        if compute_qualification_id(evidence.to_dict()) != qualification_id:
            raise QualificationError("qualification_id does not match its content")
        return evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "qualification_id": self.qualification_id,
            "gap_id": self.gap_id,
            "registry_key": self.registry_key,
            "qualified_api": self.qualified_api,
            "asnumpy_api": self.asnumpy_api,
            "gap_spec_path": self.gap_spec_path,
            "gap_spec_sha256": self.gap_spec_sha256,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "artifact_manifest_path": self.artifact_manifest_path,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_id": self.artifact_id,
            "artifact_files": dict(sorted(self.artifact_files.items())),
            "runtime_artifacts": dict(sorted(self.runtime_artifacts.items())),
            "runtime": self.runtime.to_dict(),
            "constraints": list(self.constraints),
            "gates": {
                name: gate.to_dict()
                for name, gate in sorted(self.gates.items())
            },
        }


def compute_qualification_id(data: Mapping[str, Any]) -> str:
    """Compute the content-derived ID, excluding the ID field itself."""

    identity = dict(data)
    identity.pop("qualification_id", None)
    return "qual-" + _sha256_bytes(_canonical_json(identity))[:20]


def load_qualification(
    path: str,
    *,
    evidence_root: str,
    expected_sha256: Optional[str] = None,
) -> QualificationEvidence:
    root = Path(evidence_root)
    _validate_root(root)
    relative = _safe_relative_path(path, "qualification evidence")
    target = _bound_path(root, relative, "qualification evidence")
    payload = _read_private_regular_file(
        target,
        "qualification evidence",
        max_bytes=_MAX_JSON_BYTES,
    )
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, "qualification evidence")
        if _sha256_bytes(payload) != expected:
            raise QualificationError("qualification evidence hash mismatch")
    return QualificationEvidence.from_dict(
        _parse_json(payload, "qualification evidence")
    )

def _draft_artifact_id(
    gap_id: str,
    files: Mapping[str, str],
    generation: Mapping[str, Any],
) -> str:
    identity = json.dumps(
        {"gap_id": gap_id, "files": files, "generation": generation},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "draft-" + _sha256_bytes(identity)[:20]


def _validate_artifact_tree(
    root: Path,
    evidence: QualificationEvidence,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    _require_exact_fields(manifest, _ARTIFACT_FIELDS, "artifact manifest")
    if (
        isinstance(manifest.get("schema_version"), bool)
        or not isinstance(manifest.get("schema_version"), int)
        or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise QualificationError("artifact manifest schema is unsupported")
    if manifest.get("gates") != _UNVERIFIED_ARTIFACT_GATES:
        raise QualificationError("artifact unverified gates were modified")
    if not isinstance(manifest.get("summary"), str):
        raise QualificationError("artifact summary is invalid")
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise QualificationError("artifact generation evidence is missing")
    if generation.get("mode") not in {"remote_llm", "injected_test_double"}:
        raise QualificationError("artifact generation mode is invalid")
    if generation.get("fallback") != "disabled":
        raise QualificationError("artifact generation fallback was not disabled")
    missing_generation = _GENERATION_REQUIRED_FIELDS - frozenset(generation)
    if missing_generation:
        raise QualificationError(
            "artifact generation evidence is incomplete: %s"
            % ",".join(sorted(missing_generation))
        )
    if generation.get("local_validation") != "passed":
        raise QualificationError("artifact local generation validation did not pass")
    _require_known_text(generation.get("provider"), "artifact provider", limit=128)
    _require_known_text(generation.get("model"), "artifact model", limit=256)
    attempt_count = generation.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or attempt_count > 48
    ):
        raise QualificationError("artifact generation attempt count is invalid")
    response_hashes = generation.get("response_sha256")
    if (
        not isinstance(response_hashes, list)
        or not response_hashes
        or len(response_hashes) > attempt_count
    ):
        raise QualificationError("artifact generation response hashes are invalid")
    for index, digest in enumerate(response_hashes):
        _require_sha256(digest, "artifact response %d" % index)
    if manifest.get("artifact_id") != evidence.artifact_id:
        raise QualificationError("artifact_id does not match its manifest")
    if manifest.get("gap_id") != evidence.gap_id:
        raise QualificationError("artifact belongs to another gap")
    if manifest.get("status") != "generated_unverified":
        raise QualificationError("artifact lifecycle state is not admissible")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise QualificationError("artifact manifest has no file map")
    manifest_files = _parse_hash_map(raw_files, "artifact manifest files")
    if manifest_files != dict(evidence.artifact_files):
        raise QualificationError("artifact file maps differ")
    expected_artifact_id = _draft_artifact_id(
        evidence.gap_id,
        manifest_files,
        generation,
    )
    if evidence.artifact_id != expected_artifact_id:
        raise QualificationError("artifact_id is not derived from artifact content")

    artifact_dir = manifest_path.parent
    if manifest.get("workspace") != str(artifact_dir):
        raise QualificationError("artifact workspace identity mismatch")
    try:
        artifact_dir.relative_to(root)
    except ValueError as exc:
        raise QualificationError("artifact workspace escapes evidence root") from exc
    actual = set()
    tree_entries = 0
    try:
        for candidate in artifact_dir.rglob("*"):
            tree_entries += 1
            if tree_entries > _MAX_ARTIFACT_TREE_ENTRIES:
                raise QualificationError("artifact tree contains too many entries")
            if candidate.is_symlink():
                raise QualificationError("artifact tree contains a symlink")
            mode = candidate.stat().st_mode
            if stat.S_ISDIR(mode):
                if mode & 0o022:
                    raise QualificationError(
                        "artifact directory is writable by group/world"
                    )
                continue
            if not stat.S_ISREG(mode):
                raise QualificationError("artifact tree contains a special file")
            relative = candidate.relative_to(artifact_dir).as_posix()
            if candidate != manifest_path:
                actual.add(relative)
    except QualificationError:
        raise
    except OSError as exc:
        raise QualificationError("cannot enumerate artifact tree") from exc
    if actual != set(manifest_files):
        raise QualificationError("artifact file set differs from its manifest")
    total_bytes = 0
    for relative, digest in manifest_files.items():
        path = artifact_dir.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise QualificationError("artifact file is missing: %s" % relative)
        payload = _read_private_regular_file(
            path,
            "artifact file: %s" % relative,
            max_bytes=_MAX_ARTIFACT_TOTAL_BYTES - total_bytes,
        )
        total_bytes += len(payload)
        if _sha256_bytes(payload) != digest:
            raise QualificationError("artifact file hash mismatch: %s" % relative)


def _require_nonnegative_int(claims: Mapping[str, Any], key: str) -> int:
    value = claims.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualificationError("gate claim %s must be a non-negative integer" % key)
    return value


def _validate_gate_claims(
    name: str,
    claims: Mapping[str, Any],
    evidence: QualificationEvidence,
    expected_signatures: Tuple[OperatorSignature, ...],
) -> None:
    if not isinstance(claims, dict):
        raise QualificationError("gate %s claims must be an object" % name)
    _require_exact_fields(
        claims,
        _GATE_CLAIM_FIELDS[name],
        "gate %s claims" % name,
    )
    if name in _SIGNATURE_COVERAGE_GATES:
        covered = _require_nonnegative_int(
            claims, "covered_signature_variants"
        )
        if covered != len(expected_signatures):
            raise QualificationError(
                "gate %s signature coverage count mismatch" % name
            )
        signature_digest = _require_sha256(
            claims.get("signature_variants_sha256"),
            "gate %s signature variants" % name,
        )
        if signature_digest != qualification_signatures_sha256(
            expected_signatures
        ):
            raise QualificationError(
                "gate %s signature coverage digest mismatch" % name
            )
    if name == "artifact_integrity":
        if _require_nonnegative_int(claims, "verified_files") != len(
            evidence.artifact_files
        ):
            raise QualificationError("artifact integrity file count mismatch")
        if _require_nonnegative_int(claims, "mismatches") != 0:
            raise QualificationError("artifact integrity reported mismatches")
    elif name == "build":
        if _require_nonnegative_int(claims, "exit_code") != 0:
            raise QualificationError("build gate exit code is non-zero")
        if _require_nonnegative_int(claims, "output_artifacts") < 1:
            raise QualificationError("build gate produced no artifact")
    elif name == "unit_test":
        if _require_nonnegative_int(claims, "passed") < 1:
            raise QualificationError("unit test gate ran no passing test")
        if _require_nonnegative_int(claims, "failed") != 0:
            raise QualificationError("unit test gate reported failures")
    elif name == "numerical_validation":
        if claims.get("oracle") != "numpy":
            raise QualificationError("numerical gate must use the NumPy oracle")
        if _require_nonnegative_int(claims, "cases") < 1:
            raise QualificationError("numerical gate ran no cases")
        if _require_nonnegative_int(claims, "failed") != 0:
            raise QualificationError("numerical gate reported failures")
        if claims.get("cpu_fallback_detected") is not False:
            raise QualificationError("numerical gate did not exclude CPU fallback")
    elif name == "public_asnumpy_api":
        if claims.get("api") != evidence.asnumpy_api:
            raise QualificationError("public API gate tested another API")
        if claims.get("callable") is not True:
            raise QualificationError("public AsNumpy API is not callable")
        if claims.get("cpu_fallback_detected") is not False:
            raise QualificationError("public API gate did not exclude CPU fallback")
    elif name == "npu_execution":
        if claims.get("backend") != "npu":
            raise QualificationError("execution gate did not use the NPU backend")
        if _require_nonnegative_int(claims, "device_count") < 1:
            raise QualificationError("execution gate observed no NPU device")
        if _require_nonnegative_int(claims, "kernel_launches") < 1:
            raise QualificationError("execution gate observed no kernel launch")
        if claims.get("cpu_fallback_detected") is not False:
            raise QualificationError("execution gate detected or omitted CPU fallback")


def _validate_gate_report(
    root: Path,
    gate: GateEvidence,
    evidence: QualificationEvidence,
    trusted_producers: frozenset,
    expected_signatures: Tuple[OperatorSignature, ...],
) -> None:
    if gate.producer not in trusted_producers:
        raise QualificationError("gate producer is not trusted: %s" % gate.producer)
    _, payload = _read_bound_file(
        root,
        gate.report_path,
        gate.report_sha256,
        "gate %s report" % gate.name,
        max_bytes=_MAX_JSON_BYTES,
    )
    report = _parse_json(payload, "gate %s report" % gate.name)
    _require_exact_fields(report, _GATE_REPORT_FIELDS, "gate %s report" % gate.name)
    if (
        isinstance(report["schema_version"], bool)
        or not isinstance(report["schema_version"], int)
        or report["schema_version"] != gate.format_version
    ):
        raise QualificationError("gate report schema mismatch")
    if report["gate"] != gate.name or report["status"] != "passed":
        raise QualificationError("gate report identity/status mismatch")
    if report["producer"] != gate.producer:
        raise QualificationError("gate report producer mismatch")
    if report["artifact_id"] != evidence.artifact_id:
        raise QualificationError("gate report artifact mismatch")
    if report["gap_spec_sha256"] != evidence.gap_spec_sha256:
        raise QualificationError("gate report gap spec mismatch")
    if report["artifact_manifest_sha256"] != evidence.artifact_manifest_sha256:
        raise QualificationError("gate report artifact manifest mismatch")
    if report["constraints_sha256"] != qualification_constraints_sha256(
        evidence.constraints
    ):
        raise QualificationError("gate report qualification constraints mismatch")
    if report["target"] != evidence.runtime.to_dict():
        raise QualificationError("gate report runtime target mismatch")
    _validate_gate_claims(
        gate.name,
        report["claims"],
        evidence,
        expected_signatures,
    )


def _validate_runtime_binding(
    root: Path,
    evidence: QualificationEvidence,
    current_manifest: CapabilityManifest,
) -> None:
    runtime = evidence.runtime
    if current_manifest.backend != runtime.backend:
        raise QualificationError("runtime backend drift")
    if current_manifest.backend_version != runtime.backend_version:
        raise QualificationError("runtime backend version drift")
    if current_manifest.audit_id != runtime.backend_audit_id:
        raise QualificationError("runtime backend audit drift")
    environment = current_manifest.environment
    for key in ("module_path", "cann_version", "soc_version", "python_version"):
        _require_known_text(environment.get(key), "current runtime %s" % key)
    if environment["cann_version"] != runtime.cann_version:
        raise QualificationError("runtime CANN version drift")
    if _normalize_soc(environment["soc_version"]) != _normalize_soc(runtime.soc_version):
        raise QualificationError("runtime SoC version drift")
    if environment["python_version"] != runtime.python_version:
        raise QualificationError("runtime Python version drift")
    module_path = Path(environment["module_path"])
    if (
        not module_path.is_absolute()
        or module_path.is_symlink()
        or not module_path.is_file()
    ):
        raise QualificationError("runtime backend module is not a regular file")
    _validate_private_directory(module_path.parent, "runtime backend module parent")
    module_payload = _read_private_regular_file(
        module_path,
        "runtime backend module",
        max_bytes=_MAX_BOUND_FILE_BYTES,
    )
    if _sha256_bytes(module_payload) != runtime.backend_module_sha256:
        raise QualificationError("runtime backend module hash drift")
    if runtime.backend_module_sha256 not in set(evidence.runtime_artifacts.values()):
        raise QualificationError("runtime artifacts do not bind the backend module")
    _, opp_payload = _read_bound_file(
        root,
        runtime.opp_manifest_path,
        runtime.opp_manifest_sha256,
        "runtime OPP manifest",
    )
    runtime_total_bytes = len(opp_payload)
    for relative, digest in evidence.runtime_artifacts.items():
        _, payload = _read_bound_file(
            root,
            relative,
            digest,
            "runtime artifact %s" % relative,
            max_bytes=min(
                _MAX_BOUND_FILE_BYTES,
                _MAX_RUNTIME_TOTAL_BYTES - runtime_total_bytes,
            ),
        )
        runtime_total_bytes += len(payload)


def validate_qualification(
    evidence: QualificationEvidence,
    *,
    evidence_root: str,
    current_manifest: CapabilityManifest,
    trusted_producers: Sequence[str] = tuple(DEFAULT_TRUSTED_PRODUCERS),
) -> OperatorCapability:
    """Validate a qualification without executing any supplied artifact."""

    evidence = QualificationEvidence.from_dict(evidence.to_dict())
    root = Path(evidence_root)
    _validate_root(root)
    _, gap_payload = _read_bound_file(
        root,
        evidence.gap_spec_path,
        evidence.gap_spec_sha256,
        "gap spec",
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        raw_spec = _parse_json(gap_payload, "gap spec")
        _require_exact_fields(raw_spec, _GAP_SPEC_FIELDS, "gap spec")
        if (
            isinstance(raw_spec.get("schema_version"), bool)
            or not isinstance(raw_spec.get("schema_version"), int)
            or raw_spec.get("schema_version") != GAP_SCHEMA_VERSION
        ):
            raise QualificationError("gap spec schema is unsupported")
        for field in (
            "gap_id",
            "api_name",
            "qualified_api",
            "asnumpy_api",
            "source_path",
            "source_sha256",
            "target_soc",
            "cann_version",
            "route",
            "semantic_source",
        ):
            if not isinstance(raw_spec[field], str):
                raise QualificationError("gap spec %s must be text" % field)
        raw_callsites = raw_spec.get("callsites")
        if not isinstance(raw_callsites, list) or not raw_callsites:
            raise QualificationError("gap spec callsites must be a non-empty list")
        if len(raw_callsites) > _MAX_FILES:
            raise QualificationError("gap spec contains too many callsites")
        for index, callsite in enumerate(raw_callsites):
            if not isinstance(callsite, dict):
                raise QualificationError("gap callsite must be an object")
            _require_exact_fields(
                callsite,
                _GAP_CALLSITE_FIELDS,
                "gap callsite %d" % index,
            )
            _require_exact_int(
                callsite["lineno"], "gap callsite %d line" % index
            )
            _require_exact_int(
                callsite["col_offset"], "gap callsite %d column" % index
            )
            for field in ("qualified_api", "call_expr", "reason"):
                if not isinstance(callsite[field], str):
                    raise QualificationError(
                        "gap callsite %d %s must be text" % (index, field)
                    )
        for field in (
            "signature_variants",
            "route_decisions",
            "shape_rules",
            "dtype_rules",
            "reference_cases",
        ):
            value = raw_spec[field]
            if not isinstance(value, list):
                raise QualificationError("gap spec %s must be a list" % field)
            if len(value) > _MAX_FILES:
                raise QualificationError("gap spec %s is too large" % field)
        for field in ("signature_variants", "reference_cases"):
            if any(not isinstance(item, dict) for item in raw_spec[field]):
                raise QualificationError(
                    "gap spec %s entries must be objects" % field
                )
        if not raw_spec["signature_variants"]:
            raise QualificationError(
                "gap spec signature_variants must be non-empty"
            )
        for index, signature in enumerate(raw_spec["signature_variants"]):
            _require_exact_fields(
                signature,
                _GAP_SIGNATURE_FIELDS,
                "gap signature %d" % index,
            )
            positional = _require_exact_int(
                signature["positional_args"],
                "gap signature %d positional_args" % index,
            )
            if positional < 0 or positional > 64:
                raise QualificationError(
                    "gap signature positional_args is out of bounds"
                )
            keywords = signature["keyword_args"]
            if not isinstance(keywords, list) or len(keywords) > 64:
                raise QualificationError(
                    "gap signature keyword_args must be a bounded list"
                )
            normalized_keywords = tuple(
                _require_known_text(
                    item,
                    "gap signature keyword argument",
                    limit=128,
                )
                for item in keywords
            )
            if len(set(normalized_keywords)) != len(normalized_keywords):
                raise QualificationError(
                    "gap signature contains duplicate keyword arguments"
                )
            try:
                OperatorSignature(
                    positional_args=positional,
                    keyword_args=normalized_keywords,
                )
            except ValueError as exc:
                raise QualificationError("gap signature is invalid") from exc
        try:
            typed_signatures = tuple(
                OperatorSignature.from_dict(item)
                for item in raw_spec["signature_variants"]
            )
        except ValueError as exc:
            raise QualificationError("gap signature is invalid") from exc
        if len(set(typed_signatures)) != len(typed_signatures):
            raise QualificationError("gap signatures contain duplicates")
        if any(
            not isinstance(item, str)
            for field in ("shape_rules", "dtype_rules")
            for item in raw_spec[field]
        ):
            raise QualificationError("gap spec rules must be text")
        for decision in raw_spec["route_decisions"]:
            if (
                not isinstance(decision, dict)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in decision.items()
                )
            ):
                raise QualificationError(
                    "gap spec route decisions must contain text objects"
                )
        spec = GapSpec.from_dict(raw_spec)
    except QualificationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("gap spec contract is invalid") from exc
    if (
        spec.gap_id != evidence.gap_id
        or spec.api_name != evidence.registry_key
        or spec.qualified_api != evidence.qualified_api
        or spec.asnumpy_api != evidence.asnumpy_api
    ):
        raise QualificationError("qualification and GapSpec identities differ")
    spec_source_path = str(_safe_relative_path(
        spec.source_path, "GapSpec source file"
    ))
    if spec_source_path != evidence.source_path:
        raise QualificationError("qualification and GapSpec source paths differ")
    if any(
        callsite.qualified_api != evidence.qualified_api
        for callsite in spec.callsites
    ):
        raise QualificationError("GapSpec callsite API differs from qualification")
    if spec.cann_version != evidence.runtime.cann_version:
        raise QualificationError("GapSpec CANN target differs from runtime")
    if _normalize_soc(spec.target_soc) != _normalize_soc(evidence.runtime.soc_version):
        raise QualificationError("GapSpec SoC target differs from runtime")
    if spec.source_sha256 != evidence.source_sha256:
        raise QualificationError("qualification and GapSpec source hashes differ")
    expected_constraints = _constraints_from_gap_spec(spec)
    if evidence.constraints != expected_constraints:
        raise QualificationError(
            "qualification constraints mismatch the concrete GapSpec scope"
        )
    try:
        parse_profile_constraints(evidence.constraints)
    except ValueError as exc:
        raise QualificationError("qualification constraints are invalid") from exc
    _read_bound_file(
        root,
        evidence.source_path,
        evidence.source_sha256,
        "source file",
    )

    manifest_path, manifest_payload = _read_bound_file(
        root,
        evidence.artifact_manifest_path,
        evidence.artifact_manifest_sha256,
        "artifact manifest",
        max_bytes=_MAX_JSON_BYTES,
    )
    artifact_manifest = _parse_json(manifest_payload, "artifact manifest")
    _validate_artifact_tree(root, evidence, manifest_path, artifact_manifest)
    _validate_runtime_binding(root, evidence, current_manifest)
    capability = current_manifest.operators.get(evidence.registry_key)
    if capability is None or not capability.available:
        raise QualificationError("qualified public API is not callable at runtime")
    if capability.provenance not in {
        ExecutionProvenance.UNKNOWN,
        ExecutionProvenance.CUSTOM_NPU,
    }:
        raise QualificationError("overlay may not replace audited provenance")

    producers = frozenset(str(item) for item in trusted_producers)
    if not producers:
        raise QualificationError("trusted producer set must be non-empty")
    for name in sorted(REQUIRED_QUALIFICATION_GATES):
        _validate_gate_report(
            root,
            evidence.gates[name],
            evidence,
            producers,
            typed_signatures,
        )
    return OperatorCapability(
        name=evidence.registry_key,
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="qualification:%s" % evidence.qualification_id,
        constraints=evidence.constraints,
        signature_variants=tuple(
            typed_signatures
        ),
    )


@dataclass(frozen=True)
class QualificationRef:
    registry_key: str
    evidence_path: str
    evidence_sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualificationRef":
        if not isinstance(data, dict):
            raise QualificationError("overlay entry must be an object")
        _require_exact_fields(
            data,
            frozenset({"registry_key", "evidence_path", "evidence_sha256"}),
            "overlay entry",
        )
        if not isinstance(data["registry_key"], str):
            raise QualificationError("overlay registry_key must be text")
        registry_key = data["registry_key"]
        if not _REGISTRY_KEY.fullmatch(registry_key):
            raise QualificationError("overlay registry_key is invalid")
        return cls(
            registry_key=registry_key,
            evidence_path=str(_safe_relative_path(
                data["evidence_path"], "overlay evidence"
            )),
            evidence_sha256=_require_sha256(
                data["evidence_sha256"], "overlay evidence"
            ),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "registry_key": self.registry_key,
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class CapabilityOverlay:
    revision: int
    parent_sha256: str
    base_manifest_sha256: str
    entries: Tuple[QualificationRef, ...]
    schema_version: int = OVERLAY_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityOverlay":
        if not isinstance(data, dict):
            raise QualificationError("capability overlay must be an object")
        _require_exact_fields(
            data,
            frozenset({
                "schema_version",
                "revision",
                "parent_sha256",
                "base_manifest_sha256",
                "entries",
            }),
            "capability overlay",
        )
        schema = _require_exact_int(data["schema_version"], "overlay schema")
        revision = _require_exact_int(data["revision"], "overlay revision")
        if schema != OVERLAY_SCHEMA_VERSION:
            raise QualificationError("overlay schema is unsupported")
        if revision < 1:
            raise QualificationError("overlay revision must be positive")
        raw_entries = data["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise QualificationError("overlay entries must be non-empty")
        if len(raw_entries) > _MAX_FILES:
            raise QualificationError("overlay contains too many entries")
        entries = tuple(QualificationRef.from_dict(item) for item in raw_entries)
        keys = [entry.registry_key for entry in entries]
        if len(set(keys)) != len(keys):
            raise QualificationError("overlay contains duplicate registry keys")
        return cls(
            schema_version=schema,
            revision=revision,
            parent_sha256=_require_sha256(data["parent_sha256"], "overlay parent"),
            base_manifest_sha256=_require_sha256(
                data["base_manifest_sha256"], "overlay base manifest"
            ),
            entries=entries,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "parent_sha256": self.parent_sha256,
            "base_manifest_sha256": self.base_manifest_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8")


def load_overlay(path: str) -> CapabilityOverlay:
    target = Path(path)
    _validate_private_directory(target.parent, "overlay parent")
    payload = _read_private_regular_file(
        target,
        "overlay",
        max_bytes=_MAX_JSON_BYTES,
    )
    return CapabilityOverlay.from_dict(_parse_json(payload, "capability overlay"))


def apply_overlay(
    base: CapabilityManifest,
    overlay: CapabilityOverlay,
    *,
    evidence_root: str,
    trusted_producers: Sequence[str] = tuple(DEFAULT_TRUSTED_PRODUCERS),
) -> CapabilityManifest:
    """Validate every entry, then return one all-or-nothing manifest snapshot."""

    overlay = CapabilityOverlay.from_dict(overlay.to_dict())
    if capability_manifest_sha256(base) != overlay.base_manifest_sha256:
        raise QualificationError("capability overlay base manifest drift")
    additions = {}
    for ref in overlay.entries:
        evidence = load_qualification(
            ref.evidence_path,
            evidence_root=evidence_root,
            expected_sha256=ref.evidence_sha256,
        )
        if evidence.registry_key != ref.registry_key:
            raise QualificationError("overlay entry registry key mismatch")
        candidate = validate_qualification(
            evidence,
            evidence_root=evidence_root,
            current_manifest=base,
            trusted_producers=trusted_producers,
        )
        existing = base.operators.get(ref.registry_key)
        if existing is None or not existing.available:
            raise QualificationError("overlay API is absent from the runtime")
        if existing.provenance is ExecutionProvenance.CUSTOM_NPU:
            if existing.evidence != candidate.evidence:
                raise QualificationError("overlay conflicts with an existing qualification")
        elif existing.provenance is not ExecutionProvenance.UNKNOWN:
            raise QualificationError("overlay cannot replace an audited capability")
        additions[ref.registry_key] = candidate
    operators = dict(base.operators)
    operators.update(additions)
    overlay_digest = _sha256_bytes(overlay.to_bytes())[:16]
    return CapabilityManifest(
        backend=base.backend,
        backend_version=base.backend_version,
        operators=operators,
        environment=dict(base.environment),
        schema_version=base.schema_version,
        audit_id="%s+qualification-%s" % (base.audit_id, overlay_digest),
    )


def publish_overlay_atomic(
    path: str,
    overlay: CapabilityOverlay,
    *,
    expected_previous_sha256: str = ZERO_SHA256,
) -> str:
    """Publish an overlay with a locked compare-and-swap transaction."""

    expected = _require_sha256(
        expected_previous_sha256, "expected previous overlay"
    )
    validated_overlay = CapabilityOverlay.from_dict(overlay.to_dict())
    target = Path(path)
    parent = target.parent
    if parent.exists():
        _validate_private_directory(parent, "overlay parent")
    else:
        try:
            parent.mkdir(parents=True, mode=0o700)
        except OSError as exc:
            raise QualificationError("cannot create overlay parent") from exc
        _validate_private_directory(parent, "overlay parent")
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
        raise QualificationError("cannot open overlay lock") from exc
    temp_name = ""
    published_sha256 = ""
    primary_error = None
    try:
        lock_stat = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or lock_stat.st_mode & 0o022
        ):
            raise QualificationError(
                "overlay lock must be a private, single-link regular file"
            )
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        previous_payload = b""
        previous_revision = 0
        if os.path.lexists(target):
            previous_payload = _read_private_regular_file(
                target,
                "existing overlay",
                max_bytes=_MAX_JSON_BYTES,
            )
            previous = CapabilityOverlay.from_dict(
                _parse_json(previous_payload, "existing capability overlay")
            )
            previous_revision = previous.revision
        actual_previous = (
            _sha256_bytes(previous_payload) if previous_payload else ZERO_SHA256
        )
        if actual_previous != expected:
            raise QualificationError("capability overlay compare-and-swap failed")
        if validated_overlay.parent_sha256 != actual_previous:
            raise QualificationError("overlay parent SHA-256 does not match")
        if validated_overlay.revision != previous_revision + 1:
            raise QualificationError("overlay revision is not monotonic")
        payload = validated_overlay.to_bytes()
        fd, temp_name = tempfile.mkstemp(
            prefix="." + target.name + "-", suffix=".tmp", dir=str(parent)
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            if hasattr(os, "O_CLOEXEC"):
                directory_flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                directory_flags |= os.O_NOFOLLOW
            directory_fd = os.open(str(parent), directory_flags)
            directory_error = None
            try:
                os.replace(temp_name, target)
                temp_name = ""
                published_sha256 = _sha256_bytes(payload)
                try:
                    os.fsync(directory_fd)
                except OSError as exc:
                    raise OverlayDurabilityError(published_sha256) from exc
            except Exception as exc:
                directory_error = exc
                raise
            finally:
                try:
                    os.close(directory_fd)
                except OSError as exc:
                    if directory_error is not None:
                        pass
                    elif published_sha256:
                        raise OverlayPostPublishError(
                            published_sha256, "directory close"
                        ) from exc
                    else:
                        raise
        finally:
            if fd >= 0:
                os.close(fd)
        return published_sha256
    except QualificationError as exc:
        primary_error = exc
        raise
    except OSError as exc:
        primary_error = exc
        if published_sha256:
            raise OverlayPostPublishError(
                published_sha256, "post-publish operation"
            ) from exc
        raise QualificationError("cannot atomically publish capability overlay") from exc
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        unlock_error = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError as exc:
            unlock_error = exc
        try:
            os.close(lock_fd)
        except OSError as exc:
            if unlock_error is None:
                unlock_error = exc
        if unlock_error is not None and primary_error is None:
            if published_sha256:
                raise OverlayPostPublishError(
                    published_sha256, "lock release"
                ) from unlock_error
            raise QualificationError("cannot release overlay lock") from unlock_error
