"""Versioned backend capability and execution-provenance contracts.

``coverage.py`` answers the static question "does NumPilot know this API name?".
This module answers the stricter runtime questions:

* does the selected backend actually expose the API?
* is the implementation proven to execute on NPU, or is it a CPU fallback?

The models intentionally contain only JSON-compatible values at their boundary so
they can be persisted in reports and consumed by benchmark/verification gates.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
import platform
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


CAPABILITY_SCHEMA_VERSION = "1.0"
UNKNOWN_ENVIRONMENT_VALUE = "unknown"


def _module_text_attribute(module: Any, *names: str) -> Optional[str]:
    for name in names:
        try:
            value = getattr(module, name, None)
        except Exception:
            continue
        if value is not None and not callable(value) and str(value).strip():
            return str(value).strip()
    return None


def _read_cann_version(cann_home: Optional[str]) -> Optional[str]:
    """Read a small, conventional CANN version file without invoking runtime APIs."""

    if not cann_home:
        return None
    candidates = (
        os.path.join(cann_home, "version.info"),
        os.path.join(cann_home, "opp", "version.info"),
    )
    opp_home = os.getenv("ASCEND_OPP_PATH")
    if opp_home:
        candidates += (os.path.join(opp_home, "version.info"),)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as version_file:
                for index, line in enumerate(version_file):
                    if index >= 64:
                        break
                    key, separator, value = line.partition("=")
                    if separator and key.strip().lower() == "version":
                        parsed = value.strip().strip('"\'')
                        if parsed:
                            return parsed
        except (OSError, UnicodeError):
            continue
    return None


def detect_runtime_environment(module: Any) -> Dict[str, str]:
    """Capture reproducibility metadata using only side-effect-free probes.

    In particular, this helper does not import ``acl`` or initialise an NPU
    context.  Missing module attributes/environment variables become explicit
    ``"unknown"`` values rather than optimistic guesses.
    """

    unknown = UNKNOWN_ENVIRONMENT_VALUE
    module_path = _module_text_attribute(module, "__file__") or unknown
    cann_home = (
        os.getenv("ASCEND_HOME_PATH")
        or os.getenv("ASCEND_TOOLKIT_HOME")
        or os.getenv("CANN_HOME")
        or unknown
    )
    cann_version = (
        _module_text_attribute(module, "__cann_version__", "cann_version")
        or os.getenv("CANN_VERSION")
        or _read_cann_version(None if cann_home == unknown else cann_home)
        or unknown
    )
    soc_version = (
        _module_text_attribute(module, "__soc_version__", "soc_version")
        or os.getenv("ASCEND_SOC_VERSION")
        or os.getenv("NPU_SOC_VERSION")
        or os.getenv("SOC_VERSION")
        or unknown
    )
    return {
        "module_path": module_path,
        "cann_home": cann_home,
        "cann_version": cann_version,
        "soc_version": soc_version,
        "python_version": platform.python_version(),
    }


class ExecutionProvenance(str, Enum):
    """Where an operator implementation actually performs its computation."""

    NATIVE_NPU = "native_npu"
    CUSTOM_NPU = "custom_npu"
    COMPOSITE_NPU = "composite_npu"
    HOST_ASSISTED = "host_assisted"
    CPU_FALLBACK = "cpu_fallback"
    CONDITIONAL_CPU_FALLBACK = "conditional_cpu_fallback"
    SHIM_CPU = "shim_cpu"
    UNKNOWN = "unknown"

    @property
    def is_strict_npu(self) -> bool:
        return self in {
            ExecutionProvenance.NATIVE_NPU,
            ExecutionProvenance.CUSTOM_NPU,
            ExecutionProvenance.COMPOSITE_NPU,
        }


@dataclass(frozen=True)
class OperatorCapability:
    """Runtime availability and audited provenance for one API."""

    name: str
    available: bool
    provenance: ExecutionProvenance = ExecutionProvenance.UNKNOWN
    evidence: str = ""
    constraints: Tuple[str, ...] = ()

    @property
    def strict_npu(self) -> bool:
        return self.available and self.provenance.is_strict_npu

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "provenance": self.provenance.value,
            "strict_npu": self.strict_npu,
            "evidence": self.evidence,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorCapability":
        return cls(
            name=str(data["name"]),
            available=bool(data["available"]),
            provenance=ExecutionProvenance(data.get("provenance", "unknown")),
            evidence=str(data.get("evidence", "")),
            constraints=tuple(str(item) for item in data.get("constraints", ())),
        )


@dataclass(frozen=True)
class CapabilityManifest:
    """Serializable capability snapshot for one concrete backend module."""

    backend: str
    backend_version: Optional[str]
    operators: Mapping[str, OperatorCapability] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    audit_id: str = "runtime"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "environment": {
                str(name): str(value)
                for name, value in sorted(self.environment.items())
            },
            "operators": {
                name: capability.to_dict()
                for name, capability in sorted(self.operators.items())
            },
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityManifest":
        schema_version = str(data.get("schema_version", ""))
        if schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported capability schema version: %r (expected %r)"
                % (schema_version, CAPABILITY_SCHEMA_VERSION)
            )
        operators = {
            str(name): OperatorCapability.from_dict(value)
            for name, value in data.get("operators", {}).items()
        }
        return cls(
            backend=str(data["backend"]),
            backend_version=(
                None if data.get("backend_version") is None else str(data["backend_version"])
            ),
            operators=operators,
            environment={
                str(name): str(value)
                for name, value in data.get("environment", {}).items()
            },
            schema_version=schema_version,
            audit_id=str(data.get("audit_id", "runtime")),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CapabilityManifest":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("capability manifest JSON must contain an object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class CapabilityValidation:
    """Result of validating the APIs requested by an offload plan."""

    manifest: CapabilityManifest
    required_ops: Tuple[str, ...]
    strict_npu: bool
    missing: Tuple[str, ...] = ()
    non_strict: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing and (not self.strict_npu or not self.non_strict)

    @property
    def capabilities(self) -> Mapping[str, OperatorCapability]:
        return self.manifest.operators

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.manifest.schema_version,
            "backend": self.manifest.backend,
            "backend_version": self.manifest.backend_version,
            "audit_id": self.manifest.audit_id,
            "environment": {
                str(name): str(value)
                for name, value in sorted(self.manifest.environment.items())
            },
            "strict_npu": self.strict_npu,
            "ok": self.ok,
            "required_ops": list(self.required_ops),
            "missing": list(self.missing),
            "non_strict": list(self.non_strict),
            "capabilities": {
                name: capability.to_dict()
                for name, capability in sorted(self.capabilities.items())
            },
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


class BackendCapabilityError(RuntimeError):
    """Raised when a backend cannot honestly satisfy an offload plan."""

    def __init__(self, validation: CapabilityValidation):
        self.validation = validation
        details = []
        if validation.missing:
            details.append("missing=" + ",".join(validation.missing))
        if validation.strict_npu and validation.non_strict:
            details.append("non_strict_npu=" + ",".join(validation.non_strict))
        version = validation.manifest.backend_version or "unknown"
        message = "backend %s@%s cannot satisfy required APIs (%s)" % (
            validation.manifest.backend,
            version,
            "; ".join(details) or "unknown capability error",
        )
        super().__init__(message)


def validate_manifest(
    manifest: CapabilityManifest,
    required_ops: Iterable[str],
    *,
    strict_npu: bool,
) -> CapabilityValidation:
    """Validate availability and, optionally, strict-NPU provenance."""

    required = tuple(sorted({str(name) for name in required_ops}))
    missing = []
    non_strict = []
    for name in required:
        capability = manifest.operators.get(name)
        if capability is None or not capability.available:
            missing.append(name)
        elif not capability.strict_npu:
            non_strict.append(name)
    return CapabilityValidation(
        manifest=manifest,
        required_ops=required,
        strict_npu=strict_npu,
        missing=tuple(missing),
        non_strict=tuple(non_strict),
    )
