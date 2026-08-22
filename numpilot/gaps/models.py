"""Versioned, JSON-compatible contracts for operator-gap completion."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Mapping, Tuple


GAP_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_GAP_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_QUALIFIED_SOURCE_API = re.compile(
    r"^(?:numpy|operator)\.[A-Za-z][A-Za-z0-9_]{0,127}$"
)


@dataclass(frozen=True)
class GapCallsite:
    lineno: int
    col_offset: int
    qualified_api: str
    call_expr: str
    reason: str

    def __post_init__(self) -> None:
        if self.lineno < 1 or self.col_offset < 0:
            raise ValueError("gap callsite location is invalid")
        if not _QUALIFIED_SOURCE_API.fullmatch(self.qualified_api):
            raise ValueError("gap callsite must name a qualified source API")
        if len(self.call_expr) > 512:
            raise ValueError("gap callsite expression is too large")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "qualified_api": self.qualified_api,
            "call_expr": self.call_expr,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GapCallsite":
        return cls(
            lineno=int(data["lineno"]),
            col_offset=int(data.get("col_offset", 0)),
            qualified_api=str(data["qualified_api"]),
            call_expr=str(data["call_expr"]),
            reason=str(data["reason"]),
        )


@dataclass(frozen=True)
class GapSpec:
    gap_id: str
    api_name: str
    qualified_api: str
    asnumpy_api: str
    source_path: str
    source_sha256: str
    callsites: Tuple[GapCallsite, ...]
    signature_variants: Tuple[Mapping[str, Any], ...]
    target_soc: str
    cann_version: str
    route: str = "ascendc_registry"
    route_decisions: Tuple[Mapping[str, str], ...] = ()
    shape_rules: Tuple[str, ...] = ("runtime_profile_required",)
    dtype_rules: Tuple[str, ...] = ("runtime_profile_required",)
    semantic_source: str = ""
    reference_cases: Tuple[Mapping[str, Any], ...] = ()
    schema_version: int = GAP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GAP_SCHEMA_VERSION:
            raise ValueError("unsupported gap schema version")
        if not _SAFE_GAP_ID.fullmatch(self.gap_id):
            raise ValueError("invalid gap_id")
        if not _QUALIFIED_SOURCE_API.fullmatch(self.qualified_api):
            raise ValueError("qualified_api must be top-level numpy.* or operator.*")
        if not self.asnumpy_api.startswith("asnumpy."):
            raise ValueError("asnumpy_api must start with asnumpy.")
        if not _HEX_SHA256.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        if not self.callsites:
            raise ValueError("GapSpec requires at least one callsite")
        if not self.target_soc.strip() or len(self.target_soc) > 128:
            raise ValueError("target_soc must be non-empty and bounded")
        if not self.cann_version.strip() or len(self.cann_version) > 128:
            raise ValueError("cann_version must be non-empty and bounded")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gap_id": self.gap_id,
            "api_name": self.api_name,
            "qualified_api": self.qualified_api,
            "asnumpy_api": self.asnumpy_api,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "callsites": [item.to_dict() for item in self.callsites],
            "signature_variants": [
                dict(item) for item in self.signature_variants
            ],
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
            "route": self.route,
            "route_decisions": [
                dict(item) for item in self.route_decisions
            ],
            "shape_rules": list(self.shape_rules),
            "dtype_rules": list(self.dtype_rules),
            "semantic_source": self.semantic_source,
            "reference_cases": [
                dict(item) for item in self.reference_cases
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GapSpec":
        version = int(data.get("schema_version", 0))
        if version != GAP_SCHEMA_VERSION:
            raise ValueError(
                "unsupported gap schema version: %r (expected %r)"
                % (version, GAP_SCHEMA_VERSION)
            )
        return cls(
            schema_version=version,
            gap_id=str(data["gap_id"]),
            api_name=str(data["api_name"]),
            qualified_api=str(data["qualified_api"]),
            asnumpy_api=str(data["asnumpy_api"]),
            source_path=str(data["source_path"]),
            source_sha256=str(data["source_sha256"]),
            callsites=tuple(
                GapCallsite.from_dict(item)
                for item in data.get("callsites", ())
            ),
            signature_variants=tuple(
                dict(item) for item in data.get("signature_variants", ())
            ),
            target_soc=str(data["target_soc"]),
            cann_version=str(data["cann_version"]),
            route=str(data.get("route", "ascendc_registry")),
            route_decisions=tuple(
                dict(item) for item in data.get("route_decisions", ())
            ),
            shape_rules=tuple(
                str(item)
                for item in data.get(
                    "shape_rules", ("runtime_profile_required",)
                )
            ),
            dtype_rules=tuple(
                str(item)
                for item in data.get(
                    "dtype_rules", ("runtime_profile_required",)
                )
            ),
            semantic_source=str(data.get("semantic_source", "")),
            reference_cases=tuple(
                dict(item) for item in data.get("reference_cases", ())
            ),
        )


@dataclass(frozen=True)
class OperatorDraft:
    artifact_id: str
    gap_id: str
    workspace: str
    summary: str
    files: Mapping[str, str]
    generation: Mapping[str, Any] = field(default_factory=dict)
    status: str = "generated_unverified"
    gates: Mapping[str, str] = field(default_factory=lambda: {
        "spec": "prepared",
        "generation": "passed",
        "build": "pending",
        "unit_test": "pending",
        "numerical_validation": "pending",
        "activation": "pending",
    })
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "gap_id": self.gap_id,
            "workspace": self.workspace,
            "summary": self.summary,
            "status": self.status,
            "files": {
                str(name): str(digest)
                for name, digest in sorted(self.files.items())
            },
            "generation": {
                str(name): value
                for name, value in sorted(self.generation.items())
            },
            "gates": {
                str(name): str(status)
                for name, status in sorted(self.gates.items())
            },
        }
