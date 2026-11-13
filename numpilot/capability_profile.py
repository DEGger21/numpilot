"""Qualified AsNumpy route and call-contract profiles.

The fixed catalog answers whether an API exists.  This module deliberately
answers two different questions without conflating them:

* where the public AsNumpy route performs its computation; and
* whether one exact NumPy call is inside the audited dtype/axis/rank/signature
  contract.

Being one of the 114 known NPU route names is therefore necessary but never,
by itself, sufficient for a ``DIRECT`` decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from numpilot.capabilities import (
    OperatorSignature,
    parse_profile_constraints,
)
from numpilot.capability_catalog import (
    ASNUMPY_CAPABILITY_CATALOG,
    CapabilityCatalogError,
    validate_canonical_key,
)


QUALIFIED_ROUTE_SCHEMA_VERSION = 1
OVERLAY_ROUTE_ADMISSION_SCHEMA_VERSION = 1
OPS_NN_MATMUL_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev118-g48de2d208-opsnn-matmul-2d-f32-v1"
)
# The current release-adjacent component carries the same independently
# verified FP32 ACLNN MatMul dispatch. Keep this identity separate from the
# historical ops-nn wheel so a route is admitted only for the exact runtime
# that supplied the evidence.
ASNUMPY_INTERNAL_COMPONENT_MATMUL_AUDIT_ID = (
    "asnumpy-0.3.1-dev70-g60d139c54-internal-component-"
    "manifestff21a984-wheel-d3ea03b2-core44c5b349-v3"
)
_NATIVE_MATMUL_AUDIT_IDS = frozenset({
    OPS_NN_MATMUL_ASNUMPY_AUDIT_ID,
    ASNUMPY_INTERNAL_COMPONENT_MATMUL_AUDIT_ID,
})
_OFFICIAL_QUALIFICATION_ID = re.compile(r"^official-aclnn-[0-9a-f]{20}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ComputationRoute(str, Enum):
    """Source-audited public-wrapper computation route."""

    NATIVE_NPU = "NATIVE_NPU"
    COMPOSITE_NPU = "COMPOSITE_NPU"
    HOST_SCALAR_BOUNDARY = "HOST_SCALAR_BOUNDARY"
    HOST_ASSISTED = "HOST_ASSISTED"
    CONDITIONAL_CPU_FALLBACK = "CONDITIONAL_CPU_FALLBACK"
    CPU_FALLBACK = "CPU_FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"

    @property
    def is_npu_computation(self) -> bool:
        return self in {
            ComputationRoute.NATIVE_NPU,
            ComputationRoute.COMPOSITE_NPU,
            ComputationRoute.HOST_SCALAR_BOUNDARY,
        }

    @property
    def is_performance_eligible(self) -> bool:
        return self in {
            ComputationRoute.NATIVE_NPU,
            ComputationRoute.COMPOSITE_NPU,
        }


class QualificationStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CapabilityDecision(str, Enum):
    DIRECT = "DIRECT"
    FALLBACK_CONSENT = "FALLBACK_CONSENT"
    GAP_REQUIRED = "GAP_REQUIRED"
    CAPABILITY_AUDIT_REQUIRED = "CAPABILITY_AUDIT_REQUIRED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class RouteEntry:
    canonical_key: str
    source_api: str
    asnumpy_api: str
    route: ComputationRoute
    evidence: str
    constraints: Tuple[str, ...] = ()
    signature_variants: Tuple[OperatorSignature, ...] = ()

    def __post_init__(self) -> None:
        validate_canonical_key(self.canonical_key)
        if self.source_api != "numpy." + self.canonical_key:
            raise ValueError("route entry source API is inconsistent")
        if self.asnumpy_api != "asnumpy." + self.canonical_key:
            raise ValueError("route entry AsNumpy API is inconsistent")
        if not isinstance(self.route, ComputationRoute):
            raise ValueError("route entry computation route is invalid")
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError("route entry evidence is invalid")
        if (
            not isinstance(self.constraints, tuple)
            or len(set(self.constraints)) != len(self.constraints)
            or any(
                not isinstance(item, str) or not item
                for item in self.constraints
            )
        ):
            raise ValueError("route entry constraints are invalid")
        if (
            not isinstance(self.signature_variants, tuple)
            or any(
                not isinstance(item, OperatorSignature)
                for item in self.signature_variants
            )
            or len(set(self.signature_variants)) != len(self.signature_variants)
        ):
            raise ValueError("route entry signature variants are invalid")
        if self.signature_variants and not self.constraints:
            raise ValueError("scoped route entry requires profile constraints")
        if self.constraints:
            try:
                parse_profile_constraints(self.constraints)
            except ValueError as exc:
                raise ValueError(
                    "route entry profile constraints are invalid"
                ) from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "source_api": self.source_api,
            "asnumpy_api": self.asnumpy_api,
            "route": self.route.value,
            "evidence": self.evidence,
            "constraints": list(self.constraints),
            "signature_variants": [
                item.to_dict() for item in self.signature_variants
            ],
        }


@dataclass(frozen=True)
class QualifiedRouteManifest:
    repository: str
    commit: str
    tree: str
    entries: Mapping[str, RouteEntry]
    schema_version: int = QUALIFIED_ROUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFIED_ROUTE_SCHEMA_VERSION:
            raise ValueError("unsupported qualified route schema")
        if not isinstance(self.repository, str) or not self.repository:
            raise ValueError("route manifest repository is invalid")
        for name, value in (("commit", self.commit), ("tree", self.tree)):
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("route manifest %s is invalid" % name)
        if (
            not isinstance(self.entries, Mapping)
            or any(
                not isinstance(key, str)
                or not isinstance(entry, RouteEntry)
                or entry.canonical_key != key
                for key, entry in self.entries.items()
            )
        ):
            raise ValueError("route manifest entries are invalid")
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "commit": self.commit,
            "tree": self.tree,
            "counts": {
                "all": len(self.entries),
                "top_level": sum("." not in key for key in self.entries),
                "known_top_level_npu_routes": sum(
                    "." not in key and entry.route.is_npu_computation
                    for key, entry in self.entries.items()
                ),
            },
            "entries": {
                key: entry.to_dict()
                for key, entry in sorted(self.entries.items())
            },
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )


_TOP_LEVEL_CPU = frozenset({"inner", "matmul", "outer"})
_TOP_LEVEL_CONDITIONAL = frozenset({"dot", "vdot", "floor"})
_TOP_LEVEL_HOST_ASSISTED = frozenset({"signbit"})
_LINALG_ROUTES = {
    "linalg.det": ComputationRoute.COMPOSITE_NPU,
    "linalg.inv": ComputationRoute.NATIVE_NPU,
    "linalg.qr": ComputationRoute.NATIVE_NPU,
    "linalg.matrix_power": ComputationRoute.CPU_FALLBACK,
    "linalg.norm": ComputationRoute.CPU_FALLBACK,
    "linalg.slogdet": ComputationRoute.CONDITIONAL_CPU_FALLBACK,
}
_RANDOM_NATIVE = frozenset({
    "random.normal",
    "random.standard_normal",
    "random.uniform",
})


def _route_for_catalog_key(key: str) -> ComputationRoute:
    if key in _TOP_LEVEL_CPU:
        return ComputationRoute.CPU_FALLBACK
    if key in _TOP_LEVEL_CONDITIONAL:
        return ComputationRoute.CONDITIONAL_CPU_FALLBACK
    if key in _TOP_LEVEL_HOST_ASSISTED:
        return ComputationRoute.HOST_ASSISTED
    if key in _LINALG_ROUTES:
        return _LINALG_ROUTES[key]
    if key.startswith("random."):
        return (
            ComputationRoute.NATIVE_NPU
            if key in _RANDOM_NATIVE
            else ComputationRoute.COMPOSITE_NPU
        )
    return ComputationRoute.NATIVE_NPU


def _route_evidence(key: str, route: ComputationRoute) -> str:
    if route is ComputationRoute.CPU_FALLBACK:
        return "fixed AsNumpy public wrapper computes through NumPy on the host"
    if route is ComputationRoute.CONDITIONAL_CPU_FALLBACK:
        return "fixed AsNumpy public wrapper performs host inspection or a conditional NumPy fallback"
    if route is ComputationRoute.HOST_ASSISTED:
        return "fixed AsNumpy public wrapper always performs host-side result assistance"
    if route is ComputationRoute.COMPOSITE_NPU:
        return "fixed AsNumpy public wrapper composes audited NPU bindings"
    return "fixed AsNumpy public wrapper delegates to an NPU binding"


def build_pinned_route_manifest() -> QualifiedRouteManifest:
    catalog = ASNUMPY_CAPABILITY_CATALOG
    entries = {}
    for item in catalog.entries:
        route = _route_for_catalog_key(item.canonical_key)
        entries[item.canonical_key] = RouteEntry(
            canonical_key=item.canonical_key,
            source_api=item.source_api,
            asnumpy_api=item.asnumpy_api,
            route=route,
            evidence=_route_evidence(item.canonical_key, route),
        )
    return QualifiedRouteManifest(
        repository=catalog.repository,
        commit=catalog.commit,
        tree=catalog.tree,
        entries=entries,
    )


def build_runtime_route_manifest(
    audit_id: Optional[str],
) -> QualifiedRouteManifest:
    """Return route overrides admitted by one exact runtime audit.

    The pinned 1929a3 runtime keeps ``matmul`` as a host fallback.  Only the
    hash-bound ops-nn integration runtime promotes its narrowly qualified
    public path to native NPU computation.
    """

    pinned = build_pinned_route_manifest()
    if audit_id not in _NATIVE_MATMUL_AUDIT_IDS:
        return pinned
    entries = dict(pinned.entries)
    current = entries["matmul"]
    entries["matmul"] = RouteEntry(
        canonical_key=current.canonical_key,
        source_api=current.source_api,
        asnumpy_api=current.asnumpy_api,
        route=ComputationRoute.NATIVE_NPU,
        evidence=(
            "hash-bound AsNumpy runtime delegates the qualified 2-D float32 "
            "path to aclnnMatmul with cubeMathType=KEEP_DTYPE"
        ),
    )
    return QualifiedRouteManifest(
        repository=pinned.repository,
        commit=pinned.commit,
        tree=pinned.tree,
        entries=entries,
    )


@dataclass(frozen=True)
class OverlayRouteAdmission:
    """One exact official-overlay admission bound to a live runtime identity.

    Unlike the pinned route manifest, this object is evidence for a *scoped*
    route: every entry carries the profile constraints and call signatures
    proven by the official ACLNN qualification.  Possessing a route entry is
    therefore not enough; :func:`_scoped_admission_issue` must also prove that
    one call lies inside the admitted profile.
    """

    qualification_id: str
    overlay_sha256: str
    base_manifest_sha256: str
    audit_id: str
    runtime_identity_sha256: str
    entries: Mapping[str, RouteEntry]
    schema_version: int = OVERLAY_ROUTE_ADMISSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OVERLAY_ROUTE_ADMISSION_SCHEMA_VERSION:
            raise ValueError("unsupported overlay route admission schema")
        if (
            not isinstance(self.qualification_id, str)
            or not _OFFICIAL_QUALIFICATION_ID.fullmatch(self.qualification_id)
        ):
            raise ValueError("overlay qualification ID is invalid")
        for name, value in (
            ("overlay", self.overlay_sha256),
            ("base manifest", self.base_manifest_sha256),
            ("runtime identity", self.runtime_identity_sha256),
        ):
            if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
                raise ValueError("overlay %s hash is invalid" % name)
        if not isinstance(self.audit_id, str) or not self.audit_id:
            raise ValueError("overlay admitted audit ID is invalid")
        if (
            not isinstance(self.entries, Mapping)
            or not self.entries
            or any(
                not isinstance(key, str)
                or not isinstance(entry, RouteEntry)
                or entry.canonical_key != key
                or entry.route is not ComputationRoute.NATIVE_NPU
                or not entry.constraints
                for key, entry in self.entries.items()
            )
        ):
            raise ValueError("overlay route entries are invalid")
        object.__setattr__(
            self, "entries", MappingProxyType(dict(sorted(self.entries.items())))
        )

    @property
    def entry_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self.entries))

    def fingerprint(self) -> Dict[str, Any]:
        """Return the deterministic subset carried by an analysis receipt."""

        return {
            "schema_version": self.schema_version,
            "qualification_id": self.qualification_id,
            "overlay_sha256": self.overlay_sha256,
            "base_manifest_sha256": self.base_manifest_sha256,
            "admitted_audit_id": self.audit_id,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "entry_keys": list(self.entry_keys),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.fingerprint()
        payload["entries"] = {
            key: entry.to_dict() for key, entry in sorted(self.entries.items())
        }
        return payload


def build_overlay_route_manifest(
    base: QualifiedRouteManifest,
    admission: OverlayRouteAdmission,
) -> QualifiedRouteManifest:
    """Overlay one exact official admission on the pinned route manifest."""

    if not isinstance(base, QualifiedRouteManifest):
        raise TypeError("base must be QualifiedRouteManifest")
    if not isinstance(admission, OverlayRouteAdmission):
        raise TypeError("admission must be OverlayRouteAdmission")
    entries = dict(base.entries)
    entries.update(admission.entries)
    return QualifiedRouteManifest(
        repository=base.repository,
        commit=base.commit,
        tree=base.tree,
        entries=entries,
        schema_version=base.schema_version,
    )


def _scoped_admission_issue(
    entry: RouteEntry,
    facts: StaticCallFacts,
) -> Optional[str]:
    """Return why one call is outside an official scoped admission."""

    issues = []
    try:
        parsed = parse_profile_constraints(entry.constraints)
    except ValueError as exc:
        return "scoped profile constraints are invalid: %s" % exc
    dtype = facts.dtype
    if dtype in {None, "unknown"}:
        issues.append("dtype is not statically proven")
    elif dtype != parsed["dtype"]:
        issues.append(
            "dtype=%s does not satisfy dtype=%s" % (dtype, parsed["dtype"])
        )
    if facts.rank is None:
        issues.append("rank is not statically proven")
    else:
        if facts.rank < parsed["rank"]:
            issues.append(
                "rank=%d does not satisfy rank>=%d"
                % (facts.rank, parsed["rank"])
            )
        if "rank_max" in parsed and facts.rank > parsed["rank_max"]:
            issues.append(
                "rank=%d does not satisfy rank<=%d"
                % (facts.rank, parsed["rank_max"])
            )
    if facts.contiguous is not True:
        if facts.contiguous is False:
            issues.append("operand is not C-contiguous")
        else:
            issues.append("contiguity is not statically proven")
    if entry.canonical_key == "concatenate":
        axis = facts.axis
        if not isinstance(axis, int) or isinstance(axis, bool):
            issues.append("concatenate axis is not statically proven")
        elif facts.rank is not None and not _static_axis_in_rank(axis, facts.rank):
            issues.append(
                "axis=%d is outside rank=%d" % (axis, facts.rank)
            )
    return "; ".join(issues) if issues else None


@dataclass(frozen=True)
class StaticCallFacts:
    """JSON-safe facts inferred without executing the source."""

    positional_args: int
    keyword_args: Tuple[str, ...]
    source_contract_ok: Optional[bool]
    dtype: Optional[str] = "unknown"
    rank: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    contiguous: Optional[bool] = None
    axis: Any = "not_applicable"
    equation: Optional[str] = "not_applicable"
    output_use: str = "unknown"

    def __post_init__(self) -> None:
        if (
            isinstance(self.positional_args, bool)
            or not isinstance(self.positional_args, int)
            or not 0 <= self.positional_args <= 64
        ):
            raise ValueError("static call positional_args is invalid")
        if (
            not isinstance(self.keyword_args, tuple)
            or tuple(sorted(set(self.keyword_args))) != self.keyword_args
            or any(
                not isinstance(name, str) or not name or len(name) > 128
                for name in self.keyword_args
            )
        ):
            raise ValueError("static call keyword_args are invalid")
        if self.source_contract_ok not in {True, False, None}:
            raise ValueError("static call source contract state is invalid")
        if self.dtype is not None and (
            not isinstance(self.dtype, str) or not self.dtype
        ):
            raise ValueError("static call dtype is invalid")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank <= 64
        ):
            raise ValueError("static call rank is invalid")
        if self.shape is not None and (
            not isinstance(self.shape, tuple)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.shape
            )
            or (self.rank is not None and len(self.shape) != self.rank)
        ):
            raise ValueError("static call shape is invalid")
        if self.contiguous not in {True, False, None}:
            raise ValueError("static call contiguity is invalid")
        valid_axis = (
            self.axis in {None, "unknown", "omitted", "not_applicable"}
            if not isinstance(self.axis, tuple)
            else all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in self.axis
            )
        )
        if not valid_axis and not (
            isinstance(self.axis, int) and not isinstance(self.axis, bool)
        ):
            raise ValueError("static call axis is invalid")
        if self.equation is not None and (
            not isinstance(self.equation, str) or not self.equation
        ):
            raise ValueError("static call equation is invalid")
        if self.output_use not in {"device", "terminal", "unknown"}:
            raise ValueError("static call output_use is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positional_args": self.positional_args,
            "keyword_args": list(self.keyword_args),
            "source_contract_ok": self.source_contract_ok,
            "dtype": self.dtype,
            "rank": self.rank,
            "shape": None if self.shape is None else list(self.shape),
            "contiguous": self.contiguous,
            "axis": list(self.axis) if isinstance(self.axis, tuple) else self.axis,
            "equation": self.equation,
            "output_use": self.output_use,
        }


@dataclass(frozen=True)
class CallContractQualification:
    signature: QualificationStatus
    dtype: QualificationStatus
    axis: QualificationStatus
    rank: QualificationStatus
    equation: QualificationStatus
    output: QualificationStatus
    randomness: QualificationStatus = QualificationStatus.NOT_APPLICABLE

    @property
    def qualified(self) -> bool:
        return all(
            value in {
                QualificationStatus.QUALIFIED,
                QualificationStatus.NOT_APPLICABLE,
            }
            for value in (
                self.signature,
                self.dtype,
                self.axis,
                self.rank,
                self.equation,
                self.output,
                self.randomness,
            )
        )

    @property
    def rejected(self) -> bool:
        return QualificationStatus.REJECTED in {
            self.signature,
            self.dtype,
            self.axis,
            self.rank,
            self.equation,
            self.output,
            self.randomness,
        }

    def to_dict(self) -> Dict[str, str]:
        return {
            "signature": self.signature.value,
            "dtype": self.dtype.value,
            "axis": self.axis.value,
            "rank": self.rank.value,
            "equation": self.equation.value,
            "output": self.output.value,
            "randomness": self.randomness.value,
        }


@dataclass(frozen=True)
class QualifiedCallDecision:
    canonical_key: str
    source_api: str
    route: ComputationRoute
    effective_route: ComputationRoute
    contract: CallContractQualification
    decision: CapabilityDecision
    reason_code: str
    reasons: Tuple[str, ...]
    next_action: str
    facts: StaticCallFacts

    @property
    def performance_eligible(self) -> bool:
        return (
            self.decision is CapabilityDecision.DIRECT
            and self.effective_route.is_performance_eligible
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "source_api": self.source_api,
            "route": self.route.value,
            "effective_route": self.effective_route.value,
            "contract": self.contract.to_dict(),
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "reasons": list(self.reasons),
            "next_action": self.next_action,
            "performance_eligible": self.performance_eligible,
            "facts": self.facts.to_dict(),
        }


_CREATION_REQUIRES_DTYPE = frozenset({
    "empty", "empty_like", "eye", "full", "full_like", "identity",
    "ones", "ones_like", "zeros", "zeros_like",
})
_INTEGER_AXIS_REQUIRED = frozenset({
    "cross", "cumprod", "cumsum", "nancumprod", "nancumsum", "sort",
})
_GLOBAL_REDUCTIONS = frozenset({
    "all", "amax", "amin", "any", "max", "mean", "min", "nanmax",
    "nanprod", "nansum", "prod", "sum",
})
_HOST_SCALAR_REDUCTIONS = _GLOBAL_REDUCTIONS - {"all", "any"}
_AUDITED_EINSUM_EQUATIONS = frozenset({
    "a,b->ab",
    "abcd,abced->abce",
})

# Runtime contract probe 2026-11-02 found semantic differences that are
# independent of API existence and route discovery.  Keep the route visible,
# but never admit these calls as direct until a new qualification receipt
# binds the corrected implementation and boundary behavior.  Each reason
# names the deterministic probe case and the audit document so the decision
# stays traceable without trusting a free-form model summary.
_SEMANTIC_CONTRACT_GAPS = {
    "fmax": (
        "AsNumpy propagates NaN where NumPy fmax ignores it",
        "tools/asnumpy_contract_probe.py case fmax_nan_ignores_nan "
        "(docs/asnumpy-contract-audit.zh-CN.md)",
    ),
    "copysign": (
        "AsNumpy loses the sign of negative zero",
        "tools/asnumpy_contract_probe.py case copysign_preserves_negative_zero "
        "(docs/asnumpy-contract-audit.zh-CN.md)",
    ),
}


def _base_status(value: Any) -> QualificationStatus:
    if value is True:
        return QualificationStatus.QUALIFIED
    if value is False:
        return QualificationStatus.REJECTED
    return QualificationStatus.PENDING


def _known_dtype_status(
    key: str,
    dtype: Optional[str],
    route: Optional[ComputationRoute] = None,
) -> QualificationStatus:
    if dtype == "unknown":
        return QualificationStatus.PENDING
    if dtype is None:
        return (
            QualificationStatus.REJECTED
            if key in _CREATION_REQUIRES_DTYPE
            else QualificationStatus.PENDING
        )
    if key == "linalg.det":
        return (
            QualificationStatus.QUALIFIED
            if dtype == "float32"
            else QualificationStatus.REJECTED
        )
    if key == "matmul" and route is ComputationRoute.NATIVE_NPU:
        return (
            QualificationStatus.QUALIFIED
            if dtype == "float32"
            else QualificationStatus.REJECTED
        )
    if key.startswith("random."):
        return QualificationStatus.QUALIFIED
    return (
        QualificationStatus.QUALIFIED
        if dtype in {"float32", "bool"}
        else QualificationStatus.PENDING
    )


def _rank_status(
    key: str,
    facts: StaticCallFacts,
    route: Optional[ComputationRoute] = None,
) -> QualificationStatus:
    if facts.rank is None or facts.shape is None:
        return QualificationStatus.PENDING
    if key == "linalg.det":
        if facts.rank != 2:
            return QualificationStatus.REJECTED
        if len(facts.shape) != 2 or facts.shape[0] != facts.shape[1]:
            return QualificationStatus.REJECTED
    if (
        key == "matmul"
        and route is ComputationRoute.NATIVE_NPU
        and facts.rank != 2
    ):
        return QualificationStatus.REJECTED
    return QualificationStatus.QUALIFIED


def _static_axis_in_rank(axis: int, rank: Optional[int]) -> Optional[bool]:
    if rank is None:
        return None
    if rank == 0:
        return False
    return -rank <= axis < rank


def _axis_status(key: str, facts: StaticCallFacts) -> QualificationStatus:
    axis = facts.axis
    if key in _INTEGER_AXIS_REQUIRED:
        if axis in {None, "omitted"}:
            return QualificationStatus.REJECTED
        if axis == "unknown":
            return QualificationStatus.PENDING
        if not isinstance(axis, int) or isinstance(axis, bool):
            return QualificationStatus.REJECTED
        in_rank = _static_axis_in_rank(axis, facts.rank)
        return (
            QualificationStatus.PENDING
            if in_rank is None
            else QualificationStatus.QUALIFIED
            if in_rank
            else QualificationStatus.REJECTED
        )
    if key in _GLOBAL_REDUCTIONS:
        if axis == "unknown":
            return QualificationStatus.PENDING
        if axis in {None, "omitted"}:
            return QualificationStatus.QUALIFIED
        if isinstance(axis, int) and not isinstance(axis, bool):
            in_rank = _static_axis_in_rank(axis, facts.rank)
            return (
                QualificationStatus.PENDING
                if in_rank is None
                else QualificationStatus.QUALIFIED
                if in_rank
                else QualificationStatus.REJECTED
            )
        if key in {"all", "any"} and (
            isinstance(axis, tuple)
            and len(axis) == 1
            and isinstance(axis[0], int)
        ):
            in_rank = _static_axis_in_rank(axis[0], facts.rank)
            return (
                QualificationStatus.PENDING
                if in_rank is None
                else QualificationStatus.QUALIFIED
                if in_rank
                else QualificationStatus.REJECTED
            )
        return QualificationStatus.REJECTED
    return QualificationStatus.NOT_APPLICABLE


def _signature_status(
    key: str,
    facts: StaticCallFacts,
    route: Optional[ComputationRoute] = None,
) -> QualificationStatus:
    base = _base_status(facts.source_contract_ok)
    if base is not QualificationStatus.QUALIFIED:
        return base
    if key == "linalg.det" and not (
        (facts.positional_args == 1 and not facts.keyword_args)
        or (facts.positional_args == 0 and facts.keyword_args == ("a",))
    ):
        return QualificationStatus.REJECTED
    if key == "einsum" and not (
        facts.positional_args == 3 and not facts.keyword_args
    ):
        return QualificationStatus.REJECTED
    if key == "matmul" and route is ComputationRoute.NATIVE_NPU and not (
        facts.positional_args == 2 and not facts.keyword_args
    ):
        return QualificationStatus.REJECTED
    return QualificationStatus.QUALIFIED


def _equation_status(key: str, equation: Optional[str]) -> QualificationStatus:
    if key != "einsum":
        return QualificationStatus.NOT_APPLICABLE
    if equation in {None, "unknown"}:
        return QualificationStatus.PENDING
    return (
        QualificationStatus.QUALIFIED
        if equation in _AUDITED_EINSUM_EQUATIONS
        else QualificationStatus.REJECTED
    )


def _output_status(
    key: str,
    facts: StaticCallFacts,
) -> Tuple[QualificationStatus, Optional[ComputationRoute]]:
    if key in _HOST_SCALAR_REDUCTIONS and facts.axis in {None, "omitted"}:
        if facts.output_use == "terminal":
            return QualificationStatus.QUALIFIED, ComputationRoute.HOST_SCALAR_BOUNDARY
        if facts.output_use == "device":
            return QualificationStatus.REJECTED, ComputationRoute.HOST_SCALAR_BOUNDARY
        return QualificationStatus.PENDING, ComputationRoute.HOST_SCALAR_BOUNDARY
    if key.startswith("random.") and facts.dtype == "float64":
        if facts.output_use == "device":
            return QualificationStatus.PENDING, None
    if facts.output_use == "unknown":
        return QualificationStatus.PENDING, None
    return QualificationStatus.QUALIFIED, None


def _contract_reasons(
    key: str,
    contract: CallContractQualification,
    facts: StaticCallFacts,
) -> Tuple[str, ...]:
    reasons = []
    for name in ("signature", "dtype", "axis", "rank", "equation", "output", "randomness"):
        status = getattr(contract, name)
        if status is QualificationStatus.REJECTED:
            reasons.append("%s call contract is rejected" % name)
        elif status is QualificationStatus.PENDING:
            reasons.append("%s call contract requires qualification" % name)
    if key == "linalg.det" and contract.rank is QualificationStatus.REJECTED:
        reasons.append("linalg.det requires a rank-2 square input")
    if key == "linalg.det" and contract.dtype is QualificationStatus.REJECTED:
        reasons.append("linalg.det dtype must be float32 in the golden profile")
    if key == "matmul" and contract.rank is QualificationStatus.REJECTED:
        reasons.append("qualified native matmul requires two rank-2 inputs")
    if key == "matmul" and contract.dtype is QualificationStatus.REJECTED:
        reasons.append("qualified native matmul requires float32 inputs")
    if key == "matmul" and contract.signature is QualificationStatus.REJECTED:
        reasons.append("qualified native matmul accepts exactly two positional inputs")
    if key.startswith("random.") and facts.dtype == "float64":
        reasons.append(
            "random output is float64 and the downstream AsNumpy chain has no audited astype contract"
        )
    return tuple(dict.fromkeys(reasons))


def qualify_call(
    canonical_key: str,
    facts: StaticCallFacts,
    *,
    manifest: Optional[QualifiedRouteManifest] = None,
) -> QualifiedCallDecision:
    """Classify one inert, statically-profiled call against the pinned route."""

    try:
        key = validate_canonical_key(canonical_key)
    except CapabilityCatalogError:
        # An unknown but syntactically safe name must still become a real gap;
        # malformed paths are a structural BLOCKED result instead.
        if (
            isinstance(canonical_key, str)
            and canonical_key
            and all(part.isidentifier() and not part.startswith("_") for part in canonical_key.split("."))
        ):
            key = canonical_key
        else:
            raise
    if not isinstance(facts, StaticCallFacts):
        raise TypeError("facts must be StaticCallFacts")
    active = manifest or build_pinned_route_manifest()
    entry = active.entries.get(key)
    if entry is None:
        contract = CallContractQualification(
            signature=QualificationStatus.NOT_APPLICABLE,
            dtype=QualificationStatus.NOT_APPLICABLE,
            axis=QualificationStatus.NOT_APPLICABLE,
            rank=QualificationStatus.NOT_APPLICABLE,
            equation=QualificationStatus.NOT_APPLICABLE,
            output=QualificationStatus.NOT_APPLICABLE,
        )
        return QualifiedCallDecision(
            canonical_key=key,
            source_api="numpy." + key,
            route=ComputationRoute.UNAVAILABLE,
            effective_route=ComputationRoute.UNAVAILABLE,
            contract=contract,
            decision=CapabilityDecision.GAP_REQUIRED,
            reason_code="API_UNAVAILABLE",
            reasons=("API is absent from the fixed AsNumpy capability catalog",),
            next_action="create a source-bound gap request; do not call the migration model",
            facts=facts,
        )

    route = entry.route
    signature = _signature_status(key, facts, route)
    dtype = _known_dtype_status(key, facts.dtype, route)
    axis = _axis_status(key, facts)
    rank = _rank_status(key, facts, route)
    equation = _equation_status(key, facts.equation)
    output, effective_override = _output_status(key, facts)
    randomness = (
        QualificationStatus.PENDING
        if key.startswith("random.")
        else QualificationStatus.NOT_APPLICABLE
    )
    contract = CallContractQualification(
        signature=signature,
        dtype=dtype,
        axis=axis,
        rank=rank,
        equation=equation,
        output=output,
        randomness=randomness,
    )
    effective_route = effective_override or route
    scoped = bool(entry.constraints or entry.signature_variants)
    scoped_issue = _scoped_admission_issue(entry, facts) if scoped else None

    if contract.rejected:
        decision = CapabilityDecision.BLOCKED
        reason_code = "CALL_CONTRACT_REWRITE_REQUIRED"
        reasons = _contract_reasons(key, contract, facts)
        next_action = "rewrite or explicitly qualify the call contract before migration"
    elif key in _SEMANTIC_CONTRACT_GAPS:
        decision = CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
        reason_code = "SEMANTIC_CONTRACT_GAP"
        reasons = (entry.evidence,) + _SEMANTIC_CONTRACT_GAPS[key]
        next_action = (
            "collect a new source-bound semantic qualification receipt; "
            "do not direct-migrate this call"
        )
    elif route in {
        ComputationRoute.CPU_FALLBACK,
        ComputationRoute.CONDITIONAL_CPU_FALLBACK,
        ComputationRoute.HOST_ASSISTED,
    }:
        decision = CapabilityDecision.FALLBACK_CONSENT
        reason_code = "AUDITED_NON_STRICT_ROUTE"
        reasons = (entry.evidence,)
        next_action = "require explicit fallback consent; never publish strict-NPU performance"
    elif route is ComputationRoute.UNKNOWN:
        decision = CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
        reason_code = "ROUTE_AUDIT_REQUIRED"
        reasons = (entry.evidence,)
        next_action = "audit the existing runtime route; do not generate a replacement operator"
    elif not contract.qualified:
        decision = CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
        reason_code = "CALL_PROFILE_AUDIT_REQUIRED"
        reasons = _contract_reasons(key, contract, facts)
        if scoped_issue is not None:
            reasons = reasons + (scoped_issue,)
        next_action = "collect an exact source-bound call profile; do not call the migration model"
    elif scoped_issue is not None:
        decision = CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
        reason_code = "CALL_PROFILE_AUDIT_REQUIRED"
        reasons = (
            entry.evidence,
            "call is outside the official scoped admission: " + scoped_issue,
        )
        next_action = (
            "collect a new exact source-bound qualification for this call "
            "profile; do not call the migration model"
        )
    else:
        decision = CapabilityDecision.DIRECT
        reason_code = "QUALIFIED_DIRECT"
        reasons = (entry.evidence,)
        next_action = "eligible for hash-bound migration under a matching runtime identity"

    return QualifiedCallDecision(
        canonical_key=key,
        source_api=entry.source_api,
        route=route,
        effective_route=effective_route,
        contract=contract,
        decision=decision,
        reason_code=reason_code,
        reasons=reasons,
        next_action=next_action,
        facts=facts,
    )


_DECISION_PRIORITY = {
    CapabilityDecision.DIRECT: 0,
    CapabilityDecision.FALLBACK_CONSENT: 1,
    CapabilityDecision.GAP_REQUIRED: 2,
    CapabilityDecision.CAPABILITY_AUDIT_REQUIRED: 3,
    CapabilityDecision.BLOCKED: 4,
}


def aggregate_decisions(decisions: Iterable[CapabilityDecision]) -> CapabilityDecision:
    values = tuple(decisions)
    if not values or any(not isinstance(value, CapabilityDecision) for value in values):
        raise ValueError("at least one valid capability decision is required")
    return max(values, key=_DECISION_PRIORITY.__getitem__)
