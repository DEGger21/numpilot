"""Versioned backend capability and execution-provenance contracts.

``coverage.py`` answers the static question "does NumPilot know this API name?".
This module answers the stricter runtime questions:

* does the selected backend actually expose the API?
* is the implementation proven to execute on NPU, or is it a CPU fallback?

The models intentionally contain only JSON-compatible values at their boundary so
they can be persisted in reports and consumed by benchmark/verification gates.
"""

from dataclasses import dataclass, field
import copy
from enum import Enum
import hashlib
import json
import os
import platform
import re
import stat
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CAPABILITY_SCHEMA_VERSION = "1.1"
ANALYSIS_PROFILE_SCHEMA_VERSION = 1
RUNTIME_IDENTITY_SCHEMA_VERSION = 1
UNKNOWN_ENVIRONMENT_VALUE = "unknown"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUALIFIED_SOURCE_API = re.compile(
    r"^(?:numpy|operator)\.[A-Za-z][A-Za-z0-9_]{0,127}$"
)
_PROFILE_DTYPE = re.compile(r"^dtype=([a-z0-9_]+)$")
_PROFILE_RANK = re.compile(r"^rank>=(0|[1-9][0-9]*)$")
_PROFILE_RANK_MAX = re.compile(r"^rank<=(0|[1-9][0-9]*)$")
_PROFILE_NUMEL = re.compile(r"^numel>=(0|[1-9][0-9]*)$")
_PROFILE_CONTIGUOUS = re.compile(r"^contiguous=(true|false)$")
_PROFILE_DTYPES_V1 = frozenset({"float32"})


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate key: %s" % key)
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError("JSON contains non-standard constant: %s" % value)


def _strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


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


_RUNTIME_IDENTITY_STATUSES = frozenset({"matched", "audit_required"})
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,255}$")
_WRAPPER_PATH = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$"
)


def _runtime_identity_text(value: Any, field_name: str, *, limit: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("runtime identity %s is invalid" % field_name)
    return value


def _runtime_identity_hash(value: Any, field_name: str) -> str:
    value = _runtime_identity_text(value, field_name, limit=64)
    if value != UNKNOWN_ENVIRONMENT_VALUE and not _SHA256.fullmatch(value):
        raise ValueError("runtime identity %s is invalid" % field_name)
    return value


def _runtime_identity_path(value: Any, field_name: str) -> str:
    value = _runtime_identity_text(value, field_name)
    if value != UNKNOWN_ENVIRONMENT_VALUE and not os.path.isabs(value):
        raise ValueError("runtime identity %s must be absolute" % field_name)
    return value


@dataclass(frozen=True)
class RuntimeIdentity:
    """Canonical, fail-closed identity of one concrete AsNumpy runtime."""

    asnumpy_repository: str
    asnumpy_commit: str
    asnumpy_tree: str
    asnumpy_version: str
    wrapper_sha256: Mapping[str, str]
    module_path: str
    module_sha256: str
    native_module_path: str
    native_module_sha256: str
    cann_home: str
    cann_version: str
    soc_model: str
    python_executable: str
    python_version: str
    bootstrap_receipt_path: str
    bootstrap_receipt_sha256: str
    audit_id: str
    status: str
    issues: Tuple[str, ...]
    schema_version: int = RUNTIME_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != RUNTIME_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError("unsupported runtime identity schema version")
        if (
            not isinstance(self.status, str)
            or self.status not in _RUNTIME_IDENTITY_STATUSES
        ):
            raise ValueError("runtime identity status is invalid")

        repository = _runtime_identity_text(
            self.asnumpy_repository, "AsNumpy repository"
        )
        if repository != UNKNOWN_ENVIRONMENT_VALUE and "://" not in repository:
            raise ValueError("runtime identity AsNumpy repository is invalid")
        for field_name, value in (
            ("AsNumpy commit", self.asnumpy_commit),
            ("AsNumpy tree", self.asnumpy_tree),
        ):
            text = _runtime_identity_text(value, field_name, limit=40)
            if text != UNKNOWN_ENVIRONMENT_VALUE and not _GIT_OBJECT_ID.fullmatch(text):
                raise ValueError("runtime identity %s is invalid" % field_name)
        for field_name, value in (
            ("AsNumpy version", self.asnumpy_version),
            ("CANN version", self.cann_version),
            ("SoC model", self.soc_model),
            ("Python version", self.python_version),
            ("audit ID", self.audit_id),
        ):
            text = _runtime_identity_text(value, field_name, limit=256)
            if text != UNKNOWN_ENVIRONMENT_VALUE and not _IDENTITY_TOKEN.fullmatch(text):
                raise ValueError("runtime identity %s is invalid" % field_name)

        for field_name, value in (
            ("module path", self.module_path),
            ("native module path", self.native_module_path),
            ("CANN home", self.cann_home),
            ("Python executable", self.python_executable),
            ("bootstrap receipt path", self.bootstrap_receipt_path),
        ):
            _runtime_identity_path(value, field_name)
        for field_name, value in (
            ("module SHA-256", self.module_sha256),
            ("native module SHA-256", self.native_module_sha256),
            ("bootstrap receipt SHA-256", self.bootstrap_receipt_sha256),
        ):
            _runtime_identity_hash(value, field_name)

        if not isinstance(self.wrapper_sha256, Mapping) or not self.wrapper_sha256:
            raise ValueError("runtime identity wrapper hashes are invalid")
        wrappers = {}
        for name, digest in self.wrapper_sha256.items():
            if (
                not isinstance(name, str)
                or len(name) > 256
                or not _WRAPPER_PATH.fullmatch(name)
            ):
                raise ValueError("runtime identity wrapper path is invalid")
            wrappers[name] = _runtime_identity_hash(
                digest, "wrapper SHA-256 for %s" % name
            )
        object.__setattr__(
            self, "wrapper_sha256", MappingProxyType(dict(sorted(wrappers.items())))
        )

        if (
            not isinstance(self.issues, tuple)
            or any(
                not isinstance(issue, str)
                or not issue
                or len(issue) > 512
                or issue != issue.strip()
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in issue
                )
                for issue in self.issues
            )
            or tuple(sorted(set(self.issues))) != self.issues
        ):
            raise ValueError("runtime identity issues are invalid")
        if self.status == "matched" and self.issues:
            raise ValueError("matched runtime identity cannot contain issues")
        if self.status == "audit_required" and not self.issues:
            raise ValueError("audit-required runtime identity must contain issues")

        critical = (
            repository,
            self.asnumpy_commit,
            self.asnumpy_tree,
            self.asnumpy_version,
            self.module_path,
            self.module_sha256,
            self.native_module_path,
            self.native_module_sha256,
            self.cann_home,
            self.cann_version,
            self.soc_model,
            self.python_executable,
            self.python_version,
            self.bootstrap_receipt_path,
            self.bootstrap_receipt_sha256,
            self.audit_id,
            *self.wrapper_sha256.values(),
        )
        if self.status == "matched" and UNKNOWN_ENVIRONMENT_VALUE in critical:
            raise ValueError("matched runtime identity contains unknown critical value")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asnumpy_repository": self.asnumpy_repository,
            "asnumpy_commit": self.asnumpy_commit,
            "asnumpy_tree": self.asnumpy_tree,
            "asnumpy_version": self.asnumpy_version,
            "wrapper_sha256": dict(self.wrapper_sha256),
            "module_path": self.module_path,
            "module_sha256": self.module_sha256,
            "native_module_path": self.native_module_path,
            "native_module_sha256": self.native_module_sha256,
            "cann_home": self.cann_home,
            "cann_version": self.cann_version,
            "soc_model": self.soc_model,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "bootstrap_receipt_path": self.bootstrap_receipt_path,
            "bootstrap_receipt_sha256": self.bootstrap_receipt_sha256,
            "audit_id": self.audit_id,
            "status": self.status,
            "issues": list(self.issues),
        }

    @property
    def identity_sha256(self) -> str:
        canonical = json.dumps(
            self._identity_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        result = self._identity_dict()
        result["identity_sha256"] = self.identity_sha256
        return result

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeIdentity":
        fields = {
            "schema_version",
            "asnumpy_repository",
            "asnumpy_commit",
            "asnumpy_tree",
            "asnumpy_version",
            "wrapper_sha256",
            "module_path",
            "module_sha256",
            "native_module_path",
            "native_module_sha256",
            "cann_home",
            "cann_version",
            "soc_model",
            "python_executable",
            "python_version",
            "bootstrap_receipt_path",
            "bootstrap_receipt_sha256",
            "audit_id",
            "identity_sha256",
            "status",
            "issues",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("runtime identity fields are invalid")
        if not isinstance(data["wrapper_sha256"], Mapping):
            raise ValueError("runtime identity wrapper hashes must be an object")
        if not isinstance(data["issues"], (list, tuple)):
            raise ValueError("runtime identity issues must be a list")
        identity = cls(
            schema_version=data["schema_version"],
            asnumpy_repository=data["asnumpy_repository"],
            asnumpy_commit=data["asnumpy_commit"],
            asnumpy_tree=data["asnumpy_tree"],
            asnumpy_version=data["asnumpy_version"],
            wrapper_sha256=dict(data["wrapper_sha256"]),
            module_path=data["module_path"],
            module_sha256=data["module_sha256"],
            native_module_path=data["native_module_path"],
            native_module_sha256=data["native_module_sha256"],
            cann_home=data["cann_home"],
            cann_version=data["cann_version"],
            soc_model=data["soc_model"],
            python_executable=data["python_executable"],
            python_version=data["python_version"],
            bootstrap_receipt_path=data["bootstrap_receipt_path"],
            bootstrap_receipt_sha256=data["bootstrap_receipt_sha256"],
            audit_id=data["audit_id"],
            status=data["status"],
            issues=tuple(data["issues"]),
        )
        supplied_hash = data["identity_sha256"]
        if not isinstance(supplied_hash, str) or supplied_hash != identity.identity_sha256:
            raise ValueError("runtime identity identity_sha256 mismatch")
        return identity

    @classmethod
    def from_json(cls, payload: str) -> "RuntimeIdentity":
        data = _strict_json_loads(payload)
        if not isinstance(data, dict):
            raise ValueError("runtime identity JSON must contain an object")
        return cls.from_dict(data)


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
class OperatorSignature:
    """One exact public-call signature admitted by qualification evidence."""

    positional_args: int
    keyword_args: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.positional_args, bool)
            or not isinstance(self.positional_args, int)
            or self.positional_args < 0
            or self.positional_args > 64
        ):
            raise ValueError("operator signature positional_args is invalid")
        if (
            not isinstance(self.keyword_args, tuple)
            or len(self.keyword_args) > 64
            or any(
                not isinstance(name, str) or not name or len(name) > 128
                for name in self.keyword_args
            )
            or tuple(sorted(set(self.keyword_args))) != self.keyword_args
        ):
            raise ValueError("operator signature keyword_args is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positional_args": self.positional_args,
            "keyword_args": list(self.keyword_args),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorSignature":
        if not isinstance(data, Mapping):
            raise ValueError("operator signature must be an object")
        if frozenset(data) != {"positional_args", "keyword_args"}:
            raise ValueError("operator signature fields are invalid")
        positional = data["positional_args"]
        keywords = data["keyword_args"]
        if not isinstance(keywords, (list, tuple)):
            raise ValueError("operator signature keyword_args must be a list")
        return cls(
            positional_args=positional,
            keyword_args=tuple(keywords),
        )


@dataclass(frozen=True)
class TensorProfile:
    """Runtime facts extracted from one concrete tensor operand."""

    dtype: str
    shape: Tuple[int, ...]
    c_contiguous: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dtype, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.dtype)
        ):
            raise ValueError("tensor profile dtype is invalid")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) > 64
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                or dimension > (2 ** 63 - 1)
                for dimension in self.shape
            )
        ):
            raise ValueError("tensor profile shape is invalid")
        if type(self.c_contiguous) is not bool:
            raise ValueError("tensor profile contiguity is invalid")

    @property
    def rank(self) -> int:
        return len(self.shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "rank": self.rank,
            "c_contiguous": self.c_contiguous,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorProfile":
        if not isinstance(data, Mapping):
            raise ValueError("tensor profile must be an object")
        if frozenset(data) != {"dtype", "shape", "rank", "c_contiguous"}:
            raise ValueError("tensor profile fields are invalid")
        shape = data["shape"]
        if not isinstance(shape, (list, tuple)):
            raise ValueError("tensor profile shape must be a list")
        profile = cls(
            dtype=data["dtype"],
            shape=tuple(shape),
            c_contiguous=data["c_contiguous"],
        )
        rank = data["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != profile.rank:
            raise ValueError("tensor profile rank is inconsistent")
        return profile


@dataclass(frozen=True)
class OperatorCallProfile:
    """Exact signature plus all tensor operands observed at a public API call."""

    inputs: Tuple[TensorProfile, ...]
    positional_args: int
    keyword_args: Tuple[str, ...]
    unprofiled_operands: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            signature = OperatorSignature(
                positional_args=self.positional_args,
                keyword_args=self.keyword_args,
            )
        except ValueError as exc:
            raise ValueError("operator call profile signature is invalid") from exc
        if (
            not isinstance(self.inputs, tuple)
            or any(not isinstance(profile, TensorProfile) for profile in self.inputs)
            or len(self.inputs) > 128
        ):
            raise ValueError("operator call profile inputs are invalid")
        if (
            not isinstance(self.unprofiled_operands, tuple)
            or any(
                not isinstance(label, str) or not label or len(label) > 256
                for label in self.unprofiled_operands
            )
            or len(set(self.unprofiled_operands)) != len(self.unprofiled_operands)
        ):
            raise ValueError("operator call profile unprofiled operands are invalid")
        operand_count = signature.positional_args + len(signature.keyword_args)
        if len(self.inputs) + len(self.unprofiled_operands) != operand_count:
            raise ValueError("operator call profile operand count is inconsistent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": [profile.to_dict() for profile in self.inputs],
            "positional_args": self.positional_args,
            "keyword_args": list(self.keyword_args),
            "unprofiled_operands": list(self.unprofiled_operands),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorCallProfile":
        if not isinstance(data, Mapping):
            raise ValueError("operator call profile must be an object")
        if frozenset(data) != {
            "inputs",
            "positional_args",
            "keyword_args",
            "unprofiled_operands",
        }:
            raise ValueError("operator call profile fields are invalid")
        inputs = data["inputs"]
        keywords = data["keyword_args"]
        unprofiled = data["unprofiled_operands"]
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("operator call profile inputs must be a list")
        if not isinstance(keywords, (list, tuple)):
            raise ValueError("operator call profile keyword_args must be a list")
        if not isinstance(unprofiled, (list, tuple)):
            raise ValueError(
                "operator call profile unprofiled_operands must be a list"
            )
        return cls(
            inputs=tuple(TensorProfile.from_dict(item) for item in inputs),
            positional_args=data["positional_args"],
            keyword_args=tuple(keywords),
            unprofiled_operands=tuple(unprofiled),
        )


@dataclass(frozen=True)
class AnalysisCallsiteProfile:
    """One externally collected profile bound to an exact source callsite."""

    qualified_name: str
    lineno: int
    col_offset: int
    profile: OperatorCallProfile

    def __post_init__(self) -> None:
        if (
            not isinstance(self.qualified_name, str)
            or not _QUALIFIED_SOURCE_API.fullmatch(self.qualified_name)
        ):
            raise ValueError("analysis callsite API is invalid")
        if (
            isinstance(self.lineno, bool)
            or not isinstance(self.lineno, int)
            or self.lineno < 1
            or self.lineno > (2 ** 31 - 1)
        ):
            raise ValueError("analysis callsite line is invalid")
        if (
            isinstance(self.col_offset, bool)
            or not isinstance(self.col_offset, int)
            or self.col_offset < 0
            or self.col_offset > (2 ** 31 - 1)
        ):
            raise ValueError("analysis callsite column is invalid")
        if not isinstance(self.profile, OperatorCallProfile):
            raise ValueError("analysis callsite profile is invalid")

    @property
    def identity(self) -> Tuple[Any, ...]:
        return (
            self.qualified_name,
            self.lineno,
            self.col_offset,
            self.profile.positional_args,
            self.profile.keyword_args,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "profile": self.profile.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisCallsiteProfile":
        if not isinstance(data, Mapping):
            raise ValueError("analysis callsite profile must be an object")
        if frozenset(data) != {
            "qualified_name",
            "lineno",
            "col_offset",
            "profile",
        }:
            raise ValueError("analysis callsite profile fields are invalid")
        return cls(
            qualified_name=data["qualified_name"],
            lineno=data["lineno"],
            col_offset=data["col_offset"],
            profile=OperatorCallProfile.from_dict(data["profile"]),
        )


@dataclass(frozen=True)
class AnalysisProfileSet:
    """Versioned callsite profiles tied to the exact analyzed source bytes."""

    source_sha256: str
    callsites: Tuple[AnalysisCallsiteProfile, ...]
    schema_version: int = ANALYSIS_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ANALYSIS_PROFILE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported analysis profile schema")
        if not isinstance(self.source_sha256, str) or not _SHA256.fullmatch(
            self.source_sha256
        ):
            raise ValueError("analysis profile source_sha256 is invalid")
        if (
            not isinstance(self.callsites, tuple)
            or not self.callsites
            or len(self.callsites) > 4096
            or any(
                not isinstance(callsite, AnalysisCallsiteProfile)
                for callsite in self.callsites
            )
        ):
            raise ValueError("analysis profile callsites are invalid")
        identities = tuple(callsite.identity for callsite in self.callsites)
        if len(set(identities)) != len(identities):
            raise ValueError("analysis profile contains duplicate callsites")

    def find(
        self,
        qualified_name: str,
        lineno: int,
        col_offset: int,
        positional_args: int,
        keyword_args: Tuple[str, ...],
    ) -> Optional[OperatorCallProfile]:
        identity = (
            qualified_name,
            lineno,
            col_offset,
            positional_args,
            keyword_args,
        )
        for callsite in self.callsites:
            if callsite.identity == identity:
                return callsite.profile
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "callsites": [callsite.to_dict() for callsite in self.callsites],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisProfileSet":
        if not isinstance(data, Mapping):
            raise ValueError("analysis profile set must be an object")
        if frozenset(data) != {"schema_version", "source_sha256", "callsites"}:
            raise ValueError("analysis profile set fields are invalid")
        callsites = data["callsites"]
        if not isinstance(callsites, (list, tuple)):
            raise ValueError("analysis profile callsites must be a list")
        return cls(
            schema_version=data["schema_version"],
            source_sha256=data["source_sha256"],
            callsites=tuple(
                AnalysisCallsiteProfile.from_dict(item) for item in callsites
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "AnalysisProfileSet":
        data = _strict_json_loads(payload)
        if not isinstance(data, dict):
            raise ValueError("analysis profile JSON must contain an object")
        return cls.from_dict(data)


def load_analysis_profile_set(
    path: str,
    *,
    max_bytes: int = 8 * 1024 * 1024,
) -> AnalysisProfileSet:
    """Load one bounded, non-symlink profile receipt from an explicit path."""

    if not isinstance(path, str) or not path:
        raise ValueError("analysis profile path must be non-empty text")
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise ValueError("analysis profile size limit is invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("cannot open analysis profile file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("analysis profile must be a regular file")
        if metadata.st_nlink != 1:
            raise ValueError("analysis profile must have one hard link")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(
                "analysis profile must not be group/world writable"
            )
        if metadata.st_size > max_bytes:
            raise ValueError("analysis profile exceeds the size limit")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("analysis profile exceeds the size limit")
        try:
            payload = b"".join(chunks).decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("analysis profile must be UTF-8 JSON") from exc
    finally:
        os.close(descriptor)
    return AnalysisProfileSet.from_json(payload)


@dataclass(frozen=True)
class CapabilityMatch:
    """Fail-closed result of matching a qualified scope to one real call."""

    matched: bool
    reasons: Tuple[str, ...]
    profile: OperatorCallProfile

    def __post_init__(self) -> None:
        if type(self.matched) is not bool:
            raise ValueError("capability match status is invalid")
        if (
            not isinstance(self.reasons, tuple)
            or any(not isinstance(reason, str) or not reason for reason in self.reasons)
        ):
            raise ValueError("capability match reasons are invalid")
        if self.matched == bool(self.reasons):
            raise ValueError("capability match status and reasons are inconsistent")
        if not isinstance(self.profile, OperatorCallProfile):
            raise ValueError("capability match profile is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "reasons": list(self.reasons),
            "profile": self.profile.to_dict(),
        }


def _c_contiguous_from_shape_strides(
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    itemsize: int,
) -> Optional[bool]:
    if len(shape) != len(strides) or itemsize <= 0:
        return None
    if not shape or any(dimension == 0 for dimension in shape):
        return True
    expected = itemsize
    for dimension, stride in zip(reversed(shape), reversed(strides)):
        if dimension > 1 and stride != expected:
            return False
        expected *= max(dimension, 1)
    return True


def tensor_profile_from_value(value: Any) -> Optional[TensorProfile]:
    """Extract a bounded profile without copying a device tensor to the host."""

    try:
        import numpy as np
    except ImportError:
        return None
    # The generic extractor is deliberately exact.  Backend-specific device
    # tensor classes require an explicitly injected, source-audited extractor;
    # unrestricted duck typing would let arbitrary objects forge metadata.
    if type(value) is not np.ndarray:
        return None
    try:
        raw_shape = getattr(value, "shape")
        raw_dtype = getattr(value, "dtype")
        shape = tuple(raw_shape)
    except (AttributeError, TypeError, ValueError):
        return None
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in shape
    ):
        return None
    try:
        dtype = np.dtype(raw_dtype)
        if not dtype.isnative:
            return None
        dtype_name = dtype.name
        itemsize = int(dtype.itemsize)
    except (TypeError, ValueError, AttributeError):
        return None

    contiguous = None
    try:
        flags = getattr(value, "flags")
        flag = getattr(flags, "c_contiguous", None)
        if flag is None:
            flag = flags["C_CONTIGUOUS"]
        if type(flag) is bool:
            contiguous = flag
        elif isinstance(flag, np.bool_):
            contiguous = bool(flag)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if contiguous is None:
        try:
            raw_strides = getattr(value, "strides")
            strides = tuple(raw_strides)
            if any(
                isinstance(stride, bool) or not isinstance(stride, int)
                for stride in strides
            ):
                return None
        except (AttributeError, TypeError, ValueError):
            return None
        contiguous = _c_contiguous_from_shape_strides(shape, strides, itemsize)
    if contiguous is None:
        return None
    return TensorProfile(
        dtype=dtype_name,
        shape=shape,
        c_contiguous=contiguous,
    )


def profile_operator_call(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    tensor_profile_extractor=tensor_profile_from_value,
) -> OperatorCallProfile:
    """Profile every operand; values that cannot be profiled stay explicit."""

    if not callable(tensor_profile_extractor):
        raise TypeError("tensor_profile_extractor must be callable")
    profiles = []
    unprofiled = []
    for index, value in enumerate(args):
        try:
            profile = tensor_profile_extractor(value)
        except Exception:
            profile = None
        if profile is not None and not isinstance(profile, TensorProfile):
            profile = None
        if profile is None:
            unprofiled.append("arg[%d]" % index)
        else:
            profiles.append(profile)
    for name in sorted(kwargs):
        try:
            profile = tensor_profile_extractor(kwargs[name])
        except Exception:
            profile = None
        if profile is not None and not isinstance(profile, TensorProfile):
            profile = None
        if profile is None:
            unprofiled.append("kwarg[%s]" % name)
        else:
            profiles.append(profile)
    return OperatorCallProfile(
        inputs=tuple(profiles),
        positional_args=len(args),
        keyword_args=tuple(sorted(kwargs)),
        unprofiled_operands=tuple(unprofiled),
    )


@dataclass(frozen=True)
class OperatorCapability:
    """Runtime availability and audited provenance for one API."""

    name: str
    available: bool
    provenance: ExecutionProvenance = ExecutionProvenance.UNKNOWN
    evidence: str = ""
    constraints: Tuple[str, ...] = ()
    signature_variants: Tuple[OperatorSignature, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("operator capability name is invalid")
        if type(self.available) is not bool:
            raise ValueError("operator capability available must be boolean")
        if not isinstance(self.provenance, ExecutionProvenance):
            raise ValueError("operator capability provenance is invalid")
        if not isinstance(self.evidence, str):
            raise ValueError("operator capability evidence must be text")
        if (
            not isinstance(self.constraints, tuple)
            or any(
                not isinstance(constraint, str) or not constraint
                for constraint in self.constraints
            )
            or len(set(self.constraints)) != len(self.constraints)
        ):
            raise ValueError("operator capability constraints are invalid")
        if (
            not isinstance(self.signature_variants, tuple)
            or any(
                not isinstance(signature, OperatorSignature)
                for signature in self.signature_variants
            )
            or len(set(self.signature_variants)) != len(self.signature_variants)
        ):
            raise ValueError("operator capability signatures are invalid")

    @property
    def strict_npu(self) -> bool:
        if not self.available or not self.provenance.is_strict_npu:
            return False
        # Constraints are descriptive evidence until a caller supplies a
        # profile-aware matcher.  A name-only capability check must never
        # broaden any scoped result to every dtype/shape/layout.
        if self.constraints or self.signature_variants:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "provenance": self.provenance.value,
            "strict_npu": self.strict_npu,
            "evidence": self.evidence,
            "constraints": list(self.constraints),
            "signature_variants": [
                signature.to_dict() for signature in self.signature_variants
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorCapability":
        if not isinstance(data, Mapping):
            raise ValueError("operator capability must be an object")
        if frozenset(data) != {
            "name",
            "available",
            "provenance",
            "strict_npu",
            "evidence",
            "constraints",
            "signature_variants",
        }:
            raise ValueError("operator capability fields are invalid")
        if not isinstance(data.get("name"), str):
            raise ValueError("operator capability name must be text")
        if type(data.get("available")) is not bool:
            raise ValueError("operator capability available must be boolean")
        constraints = data.get("constraints", ())
        signatures = data.get("signature_variants", ())
        if not isinstance(constraints, (list, tuple)):
            raise ValueError("operator capability constraints must be a list")
        if not isinstance(signatures, (list, tuple)):
            raise ValueError("operator capability signatures must be a list")
        capability = cls(
            name=data["name"],
            available=data["available"],
            provenance=ExecutionProvenance(data.get("provenance", "unknown")),
            evidence=data["evidence"],
            constraints=tuple(constraints),
            signature_variants=tuple(
                OperatorSignature.from_dict(item)
                for item in signatures
            ),
        )
        if type(data["strict_npu"]) is not bool:
            raise ValueError("operator capability strict_npu must be boolean")
        if data["strict_npu"] != capability.strict_npu:
            raise ValueError("operator capability strict_npu is inconsistent")
        return capability


def parse_profile_constraints(constraints: Sequence[str]) -> Dict[str, Any]:
    """Parse the complete, canonical v1 scoped-qualification grammar."""

    if not isinstance(constraints, (tuple, list)) or not constraints:
        raise ValueError("profile constraints must be a non-empty sequence")
    parsed = {}
    for constraint in constraints:
        if not isinstance(constraint, str):
            raise ValueError("profile constraint must be text")
        dtype_match = _PROFILE_DTYPE.fullmatch(constraint)
        rank_match = _PROFILE_RANK.fullmatch(constraint)
        rank_max_match = _PROFILE_RANK_MAX.fullmatch(constraint)
        numel_match = _PROFILE_NUMEL.fullmatch(constraint)
        contiguous_match = _PROFILE_CONTIGUOUS.fullmatch(constraint)
        if dtype_match:
            key = "dtype"
            value = dtype_match.group(1)
            if value not in _PROFILE_DTYPES_V1:
                raise ValueError("unsupported dtype constraint: %s" % constraint)
        elif rank_match:
            key = "rank"
            value = int(rank_match.group(1))
            if value > 64:
                raise ValueError("rank constraint is out of bounds")
        elif rank_max_match:
            key = "rank_max"
            value = int(rank_max_match.group(1))
            if value > 64:
                raise ValueError("maximum rank constraint is out of bounds")
        elif numel_match:
            key = "numel"
            value = int(numel_match.group(1))
            if value > (2 ** 63 - 1):
                raise ValueError("numel constraint is out of bounds")
        elif contiguous_match:
            key = "contiguous"
            value = contiguous_match.group(1) == "true"
        else:
            raise ValueError("unsupported profile constraint: %s" % constraint)
        if key in parsed:
            raise ValueError("duplicate profile constraint kind: %s" % key)
        parsed[key] = value
    missing = {"dtype", "rank", "contiguous"} - set(parsed)
    if missing:
        raise ValueError(
            "profile constraints are incomplete: %s" % ",".join(sorted(missing))
        )
    if "rank_max" in parsed and parsed["rank_max"] < parsed["rank"]:
        raise ValueError("maximum rank constraint is below minimum rank")
    return parsed


def match_capability_profile(
    capability: OperatorCapability,
    profile: OperatorCallProfile,
) -> CapabilityMatch:
    """Match the deliberately small v1 qualification-constraint language.

    Unindexed rules apply to every tensor operand.  Anything outside the v1
    grammar is rejected instead of being treated as descriptive metadata.
    """

    reasons = []
    if not capability.available:
        reasons.append("operator is unavailable")
    if not capability.provenance.is_strict_npu:
        reasons.append("operator provenance is not strict NPU")

    if capability.signature_variants and not capability.constraints:
        reasons.append("scoped capability has no profile constraints")
    if capability.signature_variants:
        signature = OperatorSignature(
            positional_args=profile.positional_args,
            keyword_args=profile.keyword_args,
        )
        if signature not in capability.signature_variants:
            reasons.append("call signature is outside the qualified variants")
    elif capability.constraints:
        reasons.append("scoped capability has no qualified call signatures")

    if capability.constraints:
        if not profile.inputs:
            reasons.append("call exposes no profileable tensor operands")
        if profile.unprofiled_operands:
            reasons.append(
                "call contains unprofiled operands: "
                + ",".join(profile.unprofiled_operands)
            )
        try:
            parsed = parse_profile_constraints(capability.constraints)
        except ValueError as exc:
            parsed = {}
            reasons.append(str(exc))

        for index, tensor in enumerate(profile.inputs):
            if "dtype" in parsed and tensor.dtype != parsed["dtype"]:
                reasons.append(
                    "input[%d] dtype=%s does not satisfy dtype=%s"
                    % (index, tensor.dtype, parsed["dtype"])
                )
            if "rank" in parsed and tensor.rank < parsed["rank"]:
                reasons.append(
                    "input[%d] rank=%d does not satisfy rank>=%d"
                    % (index, tensor.rank, parsed["rank"])
                )
            if "rank_max" in parsed and tensor.rank > parsed["rank_max"]:
                reasons.append(
                    "input[%d] rank=%d does not satisfy rank<=%d"
                    % (index, tensor.rank, parsed["rank_max"])
                )
            if "numel" in parsed:
                numel = 1
                for dimension in tensor.shape:
                    numel *= dimension
                if numel < parsed["numel"]:
                    reasons.append(
                        "input[%d] numel=%d does not satisfy numel>=%d"
                        % (index, numel, parsed["numel"])
                    )
            if (
                "contiguous" in parsed
                and tensor.c_contiguous is not parsed["contiguous"]
            ):
                reasons.append(
                    "input[%d] contiguous=%s does not satisfy contiguous=%s"
                    % (
                        index,
                        str(tensor.c_contiguous).lower(),
                        str(parsed["contiguous"]).lower(),
                    )
                )

    return CapabilityMatch(
        matched=not reasons,
        reasons=tuple(reasons),
        profile=profile,
    )


class RuntimeProfileError(RuntimeError):
    """Raised before dispatch when a real call exceeds its admitted scope."""

    def __init__(self, name: str, match: CapabilityMatch):
        self.name = name
        self.match = match
        super().__init__(
            "runtime profile rejected for %s: %s"
            % (name, "; ".join(match.reasons) or "unknown mismatch")
        )


class RuntimeProfileGuard:
    """Per-execution guard for deferred, profile-scoped strict-NPU APIs."""

    def __init__(
        self,
        manifest: "CapabilityManifest",
        required_ops: Iterable[str],
        *,
        tensor_profile_extractor=tensor_profile_from_value,
        runtime_module=None,
        runtime_targets: Optional[Mapping[str, Any]] = None,
    ):
        self.manifest = manifest
        if not callable(tensor_profile_extractor):
            raise TypeError("tensor_profile_extractor must be callable")
        self.tensor_profile_extractor = tensor_profile_extractor
        self.required_ops = tuple(sorted({str(name) for name in required_ops}))
        self.profile_scoped = []
        capabilities = {}
        missing = []
        non_strict = []
        for name in self.required_ops:
            capability = manifest.operators.get(name)
            if capability is not None:
                capabilities[name] = capability
            if capability is None or not capability.available:
                missing.append(name)
            elif capability.strict_npu:
                continue
            elif (
                capability.provenance.is_strict_npu
                and capability.constraints
                and capability.signature_variants
            ):
                try:
                    parse_profile_constraints(capability.constraints)
                except ValueError:
                    non_strict.append(name)
                else:
                    self.profile_scoped.append(name)
            else:
                non_strict.append(name)
        if missing or non_strict:
            raise BackendCapabilityError(CapabilityValidation(
                manifest=manifest,
                required_ops=self.required_ops,
                strict_npu=True,
                missing=tuple(missing),
                non_strict=tuple(non_strict),
            ))
        if (runtime_module is None) != (runtime_targets is None):
            raise ValueError(
                "runtime_module and runtime_targets must be provided together"
            )
        if runtime_targets is not None:
            if (
                not isinstance(runtime_targets, Mapping)
                or set(runtime_targets) != set(self.required_ops)
                or any(not callable(target) for target in runtime_targets.values())
            ):
                raise ValueError("runtime target snapshot is invalid")
            self._runtime_targets = MappingProxyType(dict(runtime_targets))
        else:
            self._runtime_targets = None
        self._runtime_module = runtime_module
        self._target_integrity_errors = []
        self.profile_scoped = tuple(self.profile_scoped)
        self._capabilities = MappingProxyType(capabilities)
        self._checks = {name: [] for name in self.profile_scoped}

    @property
    def runtime_targets(self) -> Optional[Mapping[str, Any]]:
        """Return the exact callables bound to this audited runtime snapshot."""

        if self._runtime_targets is None:
            return None
        return MappingProxyType(dict(self._runtime_targets))

    def _validate_target_identity(self, name: str) -> None:
        if self._runtime_targets is None:
            return
        expected = self._runtime_targets[name]
        try:
            current = getattr(self._runtime_module, name)
        except Exception as exc:
            detail = "%s lookup failed: %s: %s" % (
                name, type(exc).__name__, exc
            )
        else:
            if current is expected:
                return
            detail = "%s callable identity drifted after capability audit" % name
        self._target_integrity_errors.append(detail)
        raise RuntimeError("runtime target integrity rejected: " + detail)

    @property
    def checks(self) -> Dict[str, Any]:
        """Return a detached receipt snapshot, never mutable guard state."""

        return copy.deepcopy(self._checks)

    def validate_call(
        self,
        name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        if name not in self.required_ops:
            raise RuntimeError("runtime call was not admitted: %s" % name)
        self._validate_target_identity(name)
        if name not in self.profile_scoped:
            return
        capability = self._capabilities[name]
        match = match_capability_profile(
            capability,
            profile_operator_call(
                args,
                kwargs,
                tensor_profile_extractor=self.tensor_profile_extractor,
            ),
        )
        self._checks[name].append(match.to_dict())
        if not match.matched:
            raise RuntimeProfileError(name, match)

    @property
    def runtime_profile_ok(self) -> bool:
        return not self._target_integrity_errors and all(
            self._checks[name]
            and all(check["matched"] for check in self._checks[name])
            for name in self.profile_scoped
        )

    @property
    def profile_status(self) -> str:
        if self._target_integrity_errors:
            return "rejected"
        if not self.profile_scoped:
            return "not_required"
        if any(
            not check["matched"]
            for name in self.profile_scoped
            for check in self._checks[name]
        ):
            return "rejected"
        if self.runtime_profile_ok:
            return "matched"
        return "pending"

    @property
    def ok(self) -> bool:
        return self.profile_status in {"not_required", "matched"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.manifest.schema_version,
            "backend": self.manifest.backend,
            "backend_version": self.manifest.backend_version,
            "audit_id": self.manifest.audit_id,
            "environment": dict(sorted(self.manifest.environment.items())),
            "strict_npu": self.ok,
            "ok": self.ok,
            "preflight_ok": True,
            "preflight_strict_npu": True,
            "profile_status": self.profile_status,
            "required_ops": list(self.required_ops),
            "missing": [],
            "non_strict": [],
            "profile_scoped": list(self.profile_scoped),
            "runtime_profile_ok": self.runtime_profile_ok,
            "runtime_profile_checks": self.checks,
            "runtime_target_integrity_ok": not self._target_integrity_errors,
            "runtime_target_integrity_errors": list(
                self._target_integrity_errors
            ),
            "capabilities": {
                name: self._capabilities[name].to_dict()
                for name in self.required_ops
            },
        }


@dataclass(frozen=True)
class CapabilityManifest:
    """Serializable capability snapshot for one concrete backend module."""

    backend: str
    backend_version: Optional[str]
    operators: Mapping[str, OperatorCapability] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    audit_id: str = "runtime"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("capability manifest backend is invalid")
        if self.backend_version is not None and not isinstance(
            self.backend_version, str
        ):
            raise ValueError("capability manifest backend_version is invalid")
        if not isinstance(self.audit_id, str) or not self.audit_id:
            raise ValueError("capability manifest audit_id is invalid")
        if self.schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported capability schema version")
        if (
            not isinstance(self.operators, Mapping)
            or any(
                not isinstance(name, str)
                or not isinstance(capability, OperatorCapability)
                or capability.name != name
                for name, capability in self.operators.items()
            )
        ):
            raise ValueError("capability manifest operators are invalid")
        if (
            not isinstance(self.environment, Mapping)
            or any(
                not isinstance(name, str)
                or not isinstance(value, str)
                for name, value in self.environment.items()
            )
        ):
            raise ValueError("capability manifest environment is invalid")
        # ``frozen=True`` does not recursively freeze caller-owned mappings.
        # Copy them so a guard/receipt cannot be promoted by mutating the dict
        # that was used to construct this manifest.
        object.__setattr__(
            self,
            "operators",
            MappingProxyType(dict(self.operators)),
        )
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(self.environment)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "environment": {
                name: value
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
        if not isinstance(data, Mapping):
            raise ValueError("capability manifest must be an object")
        if frozenset(data) != {
            "schema_version",
            "audit_id",
            "backend",
            "backend_version",
            "environment",
            "operators",
        }:
            raise ValueError("capability manifest fields are invalid")
        schema_version = data.get("schema_version")
        if schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported capability schema version: %r (expected %r)"
                % (schema_version, CAPABILITY_SCHEMA_VERSION)
            )
        for field_name in ("backend", "audit_id"):
            if not isinstance(data[field_name], str):
                raise ValueError(
                    "capability manifest %s must be text" % field_name
                )
        if data["backend_version"] is not None and not isinstance(
            data["backend_version"], str
        ):
            raise ValueError("capability manifest backend_version must be text or null")
        raw_operators = data["operators"]
        raw_environment = data["environment"]
        if not isinstance(raw_operators, Mapping):
            raise ValueError("capability manifest operators must be an object")
        if not isinstance(raw_environment, Mapping):
            raise ValueError("capability manifest environment must be an object")
        if any(not isinstance(name, str) for name in raw_operators):
            raise ValueError("capability manifest operator names must be text")
        if any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in raw_environment.items()
        ):
            raise ValueError("capability manifest environment must contain text")
        operators = {
            name: OperatorCapability.from_dict(value)
            for name, value in raw_operators.items()
        }
        return cls(
            backend=data["backend"],
            backend_version=data["backend_version"],
            operators=operators,
            environment=dict(raw_environment),
            schema_version=schema_version,
            audit_id=data["audit_id"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "CapabilityManifest":
        data = _strict_json_loads(payload)
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


def strict_receipt_issues(
    receipt: Any,
    required_ops: Iterable[str],
    *,
    allow_profile_pending: bool = False,
    allow_scoped: bool = False,
) -> Tuple[str, ...]:
    """Validate an injected backend receipt before trusting its NPU claim.

    This deliberately re-checks the JSON boundary instead of assuming a third-
    party ``validate_required_ops(..., raise_on_error=True)`` implementation
    honored that argument.  Profile-scoped capabilities may be pending only
    when the receipt explicitly carries a successful preflight and a concrete
    strict-NPU capability for every required operator.
    """

    if not isinstance(receipt, Mapping):
        return ("capability receipt is not an object",)
    required = tuple(sorted({str(name) for name in required_ops}))
    issues = []
    raw_required = receipt.get("required_ops")
    if not isinstance(raw_required, (list, tuple)) or tuple(sorted({
        str(name) for name in raw_required
    })) != required:
        issues.append("capability receipt required_ops mismatch")

    pending = bool(
        allow_profile_pending
        and receipt.get("preflight_ok") is True
        and receipt.get("profile_status") == "pending"
    )
    if pending:
        if receipt.get("ok") is not False or receipt.get("strict_npu") is not False:
            issues.append("pending profile receipt status is inconsistent")
    else:
        if receipt.get("ok") is not True:
            issues.append("capability receipt did not pass")
        if receipt.get("strict_npu") is not True:
            issues.append("capability receipt is not strict NPU")

    for field_name in ("missing", "non_strict"):
        value = receipt.get(field_name)
        if not isinstance(value, (list, tuple)) or value:
            issues.append("capability receipt %s is not empty" % field_name)

    capabilities = receipt.get("capabilities")
    if not isinstance(capabilities, Mapping):
        issues.append("capability receipt has no operator map")
        return tuple(issues)
    if set(str(name) for name in capabilities) != set(required):
        issues.append("capability receipt operator map mismatch")

    for name in required:
        capability = capabilities.get(name)
        to_dict = getattr(capability, "to_dict", None)
        if callable(to_dict):
            capability = to_dict()
        if not isinstance(capability, Mapping):
            issues.append("capability %s is not an object" % name)
            continue
        try:
            parsed_capability = OperatorCapability.from_dict(capability)
        except (TypeError, ValueError) as exc:
            issues.append(
                "capability %s schema rejected: %s" % (name, exc)
            )
            continue
        if parsed_capability.name != name:
            issues.append("capability %s identity mismatch" % name)
        if not parsed_capability.available:
            issues.append("capability %s is unavailable" % name)
        if not parsed_capability.provenance.is_strict_npu:
            issues.append("capability %s provenance is not strict NPU" % name)
        has_constraints = bool(parsed_capability.constraints)
        has_signatures = bool(parsed_capability.signature_variants)
        scoped = has_constraints or has_signatures
        if scoped:
            if not (has_constraints and has_signatures):
                issues.append("capability %s scoped contract is incomplete" % name)
                continue
            try:
                parse_profile_constraints(parsed_capability.constraints)
            except ValueError as exc:
                issues.append(
                    "capability %s profile constraints rejected: %s"
                    % (name, exc)
                )
            if not allow_scoped:
                issues.append(
                    "capability %s requires a concrete runtime profile guard"
                    % name
                )
            if receipt.get("profile_status") not in {"pending", "matched"}:
                issues.append("capability %s has no profile receipt" % name)
            if parsed_capability.strict_npu:
                issues.append("capability %s scoped status is inconsistent" % name)
        elif not parsed_capability.strict_npu:
            issues.append("capability %s is not strict NPU" % name)
    return tuple(issues)
