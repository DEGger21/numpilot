"""Deterministic, fail-closed state for recovering operator gaps.

The recovery control plane is deliberately boring.  It validates immutable
inputs, reduces explicit events, and publishes private write-once JSON records
in a SHA-256 hash chain.  It does *not* discover operators, invoke a model,
build code, import AsNumPy, execute a candidate, access the network, or activate
a capability.  Those operations belong to separately trusted stages and may
only be represented here by content-bound references and truthful side-effect
claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


RECOVERY_SCHEMA_VERSION = 2
STAGE_RECEIPT_SCHEMA_VERSION = 1
OPERATOR_CANDIDATE_SCHEMA_VERSION = 1
ZERO_SHA256 = "0" * 64
_MAX_STATE_BYTES = 2 * 1024 * 1024
_MAX_HISTORY_RECORDS = 100_000
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_REF_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_QUALIFIED_API = re.compile(
    r"^(?:numpy|operator)(?:\.[A-Za-z][A-Za-z0-9_]{0,127})+$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LICENSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+_.:-]{0,127}$")


class RecoveryError(RuntimeError):
    """Recovery state or candidate evidence failed closed."""


class RecoveryPhase(str, Enum):
    CREATED = "created"
    ANALYZED = "analyzed"
    GAP_PREPARED = "gap_prepared"
    RESOLVED = "resolved"
    CANDIDATE_READY = "candidate_ready"
    ADAPTER_CANDIDATE = "adapter_candidate"
    INTEGRATED_CANDIDATE = "integrated_candidate"
    QUALIFIED = "qualified"
    ACTIVATED = "activated"
    RESCANNED = "rescanned"
    FINAL_MIGRATED = "final_migrated"
    VERIFIED = "verified"


class RecoveryOutcome(str, Enum):
    READY = "ready"
    AWAITING_INPUT = "awaiting_input"
    BLOCKED = "blocked"
    FAILED_ATTEMPT = "failed_attempt"
    COMPLETE = "complete"


class RecoveryPolicyProfile(str, Enum):
    """Named, deterministic authorization sets for a recovery run.

    Profiles grant permission only.  Every producer event must still report
    the side effects it actually performed, and every phase transition still
    requires its independently bound stage receipt.
    """

    RECEIPT_ONLY = "receipt-only"
    CANDIDATE_GENERATION = "candidate-generation"
    FULL_LOCAL_RECOVERY = "full-local-recovery"


class StageReceiptKind(str, Enum):
    """Known producer receipt types that may satisfy one recovery phase."""

    ANALYSIS_REPORT = "analysis_report"
    GAP_SPEC = "gap_spec"
    OFFICIAL_RESOLUTION_DECISION = "official_resolution_decision"
    OPERATOR_CANDIDATE = "operator_candidate"
    ASNUMPY_ADAPTER_CANDIDATE = "asnumpy_adapter_candidate"
    INTEGRATION_TRANSACTION = "integration_transaction"
    QUALIFICATION_REPORT = "qualification_report"
    ACTIVATION_RECEIPT = "activation_receipt"
    CAPABILITY_RESCAN_REPORT = "capability_rescan_report"
    SCRIPT_MIGRATION_REPORT = "script_migration_report"
    SCRIPT_VERIFICATION_RECEIPT = "script_verification_receipt"


class StageReceiptScope(str, Enum):
    """Whether one receipt covers the complete run or a gap subset."""

    RUN = "run"
    GAPS = "gaps"


class StageReceiptVerdict(str, Enum):
    """Normalized result of a stage-specific producer receipt adapter."""

    SATISFIED = "satisfied"
    FAILED = "failed"
    BLOCKED = "blocked"


class OperatorCandidateOrigin(str, Enum):
    OFFICIAL_BINARY = "official_binary"
    OFFICIAL_SOURCE = "official_source"
    REMOTE_LLM = "remote_llm"


_ORIGIN_RESOLUTION_ROUTE = {
    OperatorCandidateOrigin.OFFICIAL_BINARY: "installed_cann_adapter_candidate",
    OperatorCandidateOrigin.OFFICIAL_SOURCE: "official_source_candidate",
    OperatorCandidateOrigin.REMOTE_LLM: "new_kernel_generation_allowed",
}
_PHASES = tuple(RecoveryPhase)
_PHASE_INDEX = {phase: index for index, phase in enumerate(_PHASES)}
_STAGE_KIND_BY_PHASE = {
    RecoveryPhase.ANALYZED: StageReceiptKind.ANALYSIS_REPORT,
    RecoveryPhase.GAP_PREPARED: StageReceiptKind.GAP_SPEC,
    RecoveryPhase.RESOLVED: StageReceiptKind.OFFICIAL_RESOLUTION_DECISION,
    RecoveryPhase.CANDIDATE_READY: StageReceiptKind.OPERATOR_CANDIDATE,
    RecoveryPhase.ADAPTER_CANDIDATE: StageReceiptKind.ASNUMPY_ADAPTER_CANDIDATE,
    RecoveryPhase.INTEGRATED_CANDIDATE: StageReceiptKind.INTEGRATION_TRANSACTION,
    RecoveryPhase.QUALIFIED: StageReceiptKind.QUALIFICATION_REPORT,
    RecoveryPhase.ACTIVATED: StageReceiptKind.ACTIVATION_RECEIPT,
    RecoveryPhase.RESCANNED: StageReceiptKind.CAPABILITY_RESCAN_REPORT,
    RecoveryPhase.FINAL_MIGRATED: StageReceiptKind.SCRIPT_MIGRATION_REPORT,
    RecoveryPhase.VERIFIED: StageReceiptKind.SCRIPT_VERIFICATION_RECEIPT,
}
_NEXT_ACTION = {
    RecoveryPhase.CREATED: "analyze_source",
    RecoveryPhase.ANALYZED: "prepare_gap_specs",
    RecoveryPhase.GAP_PREPARED: "resolve_official_candidates",
    RecoveryPhase.RESOLVED: "materialize_operator_candidate",
    RecoveryPhase.CANDIDATE_READY: "prepare_adapter_candidate",
    RecoveryPhase.ADAPTER_CANDIDATE: "integrate_candidate",
    RecoveryPhase.INTEGRATED_CANDIDATE: "qualify_candidate",
    RecoveryPhase.QUALIFIED: "activate_qualified_capability",
    RecoveryPhase.ACTIVATED: "rescan_capabilities",
    RecoveryPhase.RESCANNED: "migrate_final_script",
    RecoveryPhase.FINAL_MIGRATED: "verify_final_migration",
    RecoveryPhase.VERIFIED: "complete",
}


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RecoveryError("recovery value is not canonical JSON") from exc


def _canonical_file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_fields(
    value: object,
    expected: Iterable[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RecoveryError("%s must be an object with string keys" % label)
    expected_set = frozenset(expected)
    actual = frozenset(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unknown = sorted(actual - expected_set)
        raise RecoveryError(
            "%s fields are incomplete or unknown (missing=%r, unknown=%r)"
            % (label, missing, unknown)
        )
    return value


def _bounded_text(
    value: object,
    label: str,
    maximum: int = 4096,
    *,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or len(value) > maximum
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise RecoveryError("invalid %s" % label)
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RecoveryError("%s must be boolean" % label)
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RecoveryError("%s must be an integer >= %d" % (label, minimum))
    return value


def _digest(value: object, label: str, *, zero_allowed: bool = False) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RecoveryError("%s must be a lowercase SHA-256" % label)
    if not zero_allowed and value == ZERO_SHA256:
        raise RecoveryError("%s may not be the zero SHA-256" % label)
    return value


def _enum(enum_type: type[Enum], value: object, label: str) -> Any:
    if not isinstance(value, str):
        raise RecoveryError("invalid %s" % label)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise RecoveryError("invalid %s: %s" % (label, value)) from exc


def _normalize_refs(
    refs: Sequence["EvidenceRef"],
    label: str,
    *,
    require_nonempty: bool = False,
) -> Tuple["EvidenceRef", ...]:
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence):
        raise RecoveryError("%s must be a sequence" % label)
    normalized = tuple(refs)
    if require_nonempty and not normalized:
        raise RecoveryError("%s must not be empty" % label)
    if len(normalized) > 256 or any(
        not isinstance(item, EvidenceRef) for item in normalized
    ):
        raise RecoveryError("%s is invalid or too large" % label)
    names = [item.name for item in normalized]
    if len(set(names)) != len(names):
        raise RecoveryError("%s contains duplicate names" % label)
    return tuple(sorted(normalized, key=lambda item: item.name))


@dataclass(frozen=True)
class EvidenceRef:
    """A route-neutral, content-bound reference; it never dereferences itself."""

    name: str
    locator: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_REF_NAME.fullmatch(self.name):
            raise RecoveryError("invalid evidence reference name")
        _bounded_text(self.locator, "evidence reference locator")
        _digest(self.sha256, "evidence reference digest")

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "locator": self.locator,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceRef":
        data = _strict_fields(value, {"name", "locator", "sha256"}, "evidence ref")
        return cls(
            name=_bounded_text(data["name"], "evidence reference name", 128),
            locator=_bounded_text(data["locator"], "evidence reference locator"),
            sha256=_digest(data["sha256"], "evidence reference digest"),
        )


@dataclass(frozen=True)
class RecoveryRun:
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _SAFE_ID.fullmatch(self.run_id):
            raise RecoveryError("invalid recovery run id")

    def to_dict(self) -> Dict[str, str]:
        return {"run_id": self.run_id}

    @classmethod
    def from_dict(cls, value: object) -> "RecoveryRun":
        data = _strict_fields(value, {"run_id"}, "recovery run")
        return cls(run_id=_bounded_text(data["run_id"], "recovery run id", 128))


@dataclass(frozen=True)
class RecoverySource:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _bounded_text(self.path, "recovery source path")
        if not Path(self.path).is_absolute():
            raise RecoveryError("recovery source path must be absolute")
        _digest(self.sha256, "recovery source digest")

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> "RecoverySource":
        data = _strict_fields(value, {"path", "sha256"}, "recovery source")
        return cls(
            path=_bounded_text(data["path"], "recovery source path"),
            sha256=_digest(data["sha256"], "recovery source digest"),
        )


@dataclass(frozen=True)
class RecoveryTarget:
    target_soc: str
    cann_version: str

    def __post_init__(self) -> None:
        _bounded_text(self.target_soc, "target SoC", 128)
        _bounded_text(self.cann_version, "CANN version", 128)

    def to_dict(self) -> Dict[str, str]:
        return {
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecoveryTarget":
        data = _strict_fields(value, {"target_soc", "cann_version"}, "recovery target")
        return cls(
            target_soc=_bounded_text(data["target_soc"], "target SoC", 128),
            cann_version=_bounded_text(data["cann_version"], "CANN version", 128),
        )


@dataclass(frozen=True)
class RecoveryRuntime:
    backend: str
    audit_id: str
    identity_sha256: str

    def __post_init__(self) -> None:
        _bounded_text(self.backend, "runtime backend", 128)
        _bounded_text(self.audit_id, "runtime audit id", 256)
        _digest(self.identity_sha256, "runtime identity digest")

    def to_dict(self) -> Dict[str, str]:
        return {
            "backend": self.backend,
            "audit_id": self.audit_id,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecoveryRuntime":
        data = _strict_fields(
            value,
            {"backend", "audit_id", "identity_sha256"},
            "recovery runtime",
        )
        return cls(
            backend=_bounded_text(data["backend"], "runtime backend", 128),
            audit_id=_bounded_text(data["audit_id"], "runtime audit id", 256),
            identity_sha256=_digest(
                data["identity_sha256"], "runtime identity digest"
            ),
        )


@dataclass(frozen=True)
class RecoveryPolicy:
    fallback_policy: str
    official_first: bool = True
    subprocess_allowed: bool = False
    network_allowed: bool = False
    system_write_allowed: bool = False
    model_allowed: bool = False
    asnumpy_import_allowed: bool = False
    build_allowed: bool = False
    execution_allowed: bool = False
    activation_allowed: bool = False

    def __post_init__(self) -> None:
        if self.fallback_policy not in {"ask", "accept", "reject"}:
            raise RecoveryError("invalid recovery fallback policy")
        for name in (
            "official_first",
            "subprocess_allowed",
            "network_allowed",
            "system_write_allowed",
            "model_allowed",
            "asnumpy_import_allowed",
            "build_allowed",
            "execution_allowed",
            "activation_allowed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise RecoveryError("recovery policy %s must be boolean" % name)
        if not self.official_first:
            raise RecoveryError("gap recovery policy must remain official-first")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fallback_policy": self.fallback_policy,
            "official_first": self.official_first,
            "subprocess_allowed": self.subprocess_allowed,
            "network_allowed": self.network_allowed,
            "system_write_allowed": self.system_write_allowed,
            "model_allowed": self.model_allowed,
            "asnumpy_import_allowed": self.asnumpy_import_allowed,
            "build_allowed": self.build_allowed,
            "execution_allowed": self.execution_allowed,
            "activation_allowed": self.activation_allowed,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecoveryPolicy":
        fields = {
            "fallback_policy",
            "official_first",
            "subprocess_allowed",
            "network_allowed",
            "system_write_allowed",
            "model_allowed",
            "asnumpy_import_allowed",
            "build_allowed",
            "execution_allowed",
            "activation_allowed",
        }
        data = _strict_fields(value, fields, "recovery policy")
        return cls(
            fallback_policy=_bounded_text(
                data["fallback_policy"], "fallback policy", 16
            ),
            official_first=_bool(data["official_first"], "official_first"),
            subprocess_allowed=_bool(
                data["subprocess_allowed"], "subprocess_allowed"
            ),
            network_allowed=_bool(data["network_allowed"], "network_allowed"),
            system_write_allowed=_bool(
                data["system_write_allowed"], "system_write_allowed"
            ),
            model_allowed=_bool(data["model_allowed"], "model_allowed"),
            asnumpy_import_allowed=_bool(
                data["asnumpy_import_allowed"], "asnumpy_import_allowed"
            ),
            build_allowed=_bool(data["build_allowed"], "build_allowed"),
            execution_allowed=_bool(
                data["execution_allowed"], "execution_allowed"
            ),
            activation_allowed=_bool(
                data["activation_allowed"], "activation_allowed"
            ),
        )


_RECOVERY_POLICY_PROFILE_FLAGS = {
    RecoveryPolicyProfile.RECEIPT_ONLY: {
        "subprocess_allowed": False,
        "network_allowed": False,
        "system_write_allowed": False,
        "model_allowed": False,
        "asnumpy_import_allowed": False,
        "build_allowed": False,
        "execution_allowed": False,
        "activation_allowed": False,
    },
    RecoveryPolicyProfile.CANDIDATE_GENERATION: {
        "subprocess_allowed": True,
        "network_allowed": True,
        "system_write_allowed": False,
        "model_allowed": True,
        "asnumpy_import_allowed": False,
        "build_allowed": False,
        "execution_allowed": False,
        "activation_allowed": False,
    },
    RecoveryPolicyProfile.FULL_LOCAL_RECOVERY: {
        "subprocess_allowed": True,
        "network_allowed": True,
        "system_write_allowed": True,
        "model_allowed": True,
        "asnumpy_import_allowed": True,
        "build_allowed": True,
        "execution_allowed": True,
        "activation_allowed": True,
    },
}


def recovery_policy_for_profile(
    profile: RecoveryPolicyProfile | str = RecoveryPolicyProfile.RECEIPT_ONLY,
) -> RecoveryPolicy:
    """Return the immutable policy mapped to one public profile name."""

    selected = _enum(
        RecoveryPolicyProfile, profile, "recovery policy profile"
    )
    return RecoveryPolicy(
        fallback_policy="reject",
        official_first=True,
        **_RECOVERY_POLICY_PROFILE_FLAGS[selected],
    )


@dataclass(frozen=True)
class RecoveryBinding:
    run: RecoveryRun
    source: RecoverySource
    target: RecoveryTarget
    runtime: RecoveryRuntime
    policy: RecoveryPolicy

    def __post_init__(self) -> None:
        expected = (
            ("run", self.run, RecoveryRun),
            ("source", self.source, RecoverySource),
            ("target", self.target, RecoveryTarget),
            ("runtime", self.runtime, RecoveryRuntime),
            ("policy", self.policy, RecoveryPolicy),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise RecoveryError("invalid recovery binding %s" % name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "runtime": self.runtime.to_dict(),
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecoveryBinding":
        data = _strict_fields(
            value,
            {"run", "source", "target", "runtime", "policy"},
            "recovery binding",
        )
        return cls(
            run=RecoveryRun.from_dict(data["run"]),
            source=RecoverySource.from_dict(data["source"]),
            target=RecoveryTarget.from_dict(data["target"]),
            runtime=RecoveryRuntime.from_dict(data["runtime"]),
            policy=RecoveryPolicy.from_dict(data["policy"]),
        )


def recovery_binding_sha256(binding: RecoveryBinding) -> str:
    """Return the canonical digest used to bind every stage receipt."""

    if not isinstance(binding, RecoveryBinding):
        raise RecoveryError("binding digest requires a RecoveryBinding")
    return _sha256(_canonical_bytes(binding.to_dict()))


def _normalize_gap_ids(
    gap_ids: Sequence[str], label: str, *, require_nonempty: bool = False
) -> Tuple[str, ...]:
    if isinstance(gap_ids, (str, bytes)) or not isinstance(gap_ids, Sequence):
        raise RecoveryError("%s must be a sequence" % label)
    normalized = tuple(gap_ids)
    if require_nonempty and not normalized:
        raise RecoveryError("%s must not be empty" % label)
    if len(normalized) > 128 or any(
        not isinstance(gap_id, str) or not _SAFE_ID.fullmatch(gap_id)
        for gap_id in normalized
    ):
        raise RecoveryError("%s is invalid or too large" % label)
    if len(set(normalized)) != len(normalized):
        raise RecoveryError("%s contains duplicates" % label)
    return tuple(sorted(normalized))


def _stage_receipt_identity_body(
    *,
    phase: RecoveryPhase,
    kind: StageReceiptKind,
    scope: StageReceiptScope,
    covered_gap_ids: Sequence[str],
    binding_sha256: str,
    parent_state_sha256: str,
    verdict: StageReceiptVerdict,
    ref: EvidenceRef,
) -> Dict[str, Any]:
    return {
        "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
        "record_type": "numpilot_recovery_stage_receipt_ref",
        "phase": phase.value,
        "kind": kind.value,
        "scope": scope.value,
        "covered_gap_ids": list(covered_gap_ids),
        "binding_sha256": binding_sha256,
        "parent_state_sha256": parent_state_sha256,
        "verdict": verdict.value,
        "ref": ref.to_dict(),
    }


@dataclass(frozen=True)
class StageReceiptRef:
    """A typed, phase-bound reference emitted by a trusted receipt adapter."""

    receipt_id: str
    phase: RecoveryPhase
    kind: StageReceiptKind
    scope: StageReceiptScope
    covered_gap_ids: Tuple[str, ...]
    binding_sha256: str
    parent_state_sha256: str
    verdict: StageReceiptVerdict
    ref: EvidenceRef
    schema_version: int = STAGE_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != STAGE_RECEIPT_SCHEMA_VERSION
        ):
            raise RecoveryError("unsupported stage receipt schema")
        if not isinstance(self.receipt_id, str) or not _SAFE_ID.fullmatch(
            self.receipt_id
        ):
            raise RecoveryError("invalid stage receipt id")
        phase = _enum(RecoveryPhase, self.phase, "stage receipt phase")
        if phase is RecoveryPhase.CREATED:
            raise RecoveryError("CREATED cannot be satisfied by a stage receipt")
        kind = _enum(StageReceiptKind, self.kind, "stage receipt kind")
        if _STAGE_KIND_BY_PHASE[phase] is not kind:
            raise RecoveryError("stage receipt kind does not match its phase")
        scope = _enum(StageReceiptScope, self.scope, "stage receipt scope")
        covered = _normalize_gap_ids(
            self.covered_gap_ids,
            "stage receipt covered gap ids",
            require_nonempty=scope is StageReceiptScope.GAPS,
        )
        binding_digest = _digest(
            self.binding_sha256, "stage receipt binding digest"
        )
        parent_digest = _digest(
            self.parent_state_sha256, "stage receipt parent state digest"
        )
        verdict = _enum(
            StageReceiptVerdict, self.verdict, "stage receipt verdict"
        )
        if not isinstance(self.ref, EvidenceRef):
            raise RecoveryError("stage receipt requires a content-bound ref")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "covered_gap_ids", covered)
        object.__setattr__(self, "binding_sha256", binding_digest)
        object.__setattr__(self, "parent_state_sha256", parent_digest)
        object.__setattr__(self, "verdict", verdict)
        body = _stage_receipt_identity_body(
            phase=phase,
            kind=kind,
            scope=scope,
            covered_gap_ids=covered,
            binding_sha256=binding_digest,
            parent_state_sha256=parent_digest,
            verdict=verdict,
            ref=self.ref,
        )
        expected_id = "stage-%s" % _sha256(_canonical_bytes(body))[:24]
        if self.receipt_id != expected_id:
            raise RecoveryError("stage receipt id does not match its bindings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            **_stage_receipt_identity_body(
                phase=self.phase,
                kind=self.kind,
                scope=self.scope,
                covered_gap_ids=self.covered_gap_ids,
                binding_sha256=self.binding_sha256,
                parent_state_sha256=self.parent_state_sha256,
                verdict=self.verdict,
                ref=self.ref,
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "StageReceiptRef":
        fields = {
            "schema_version",
            "record_type",
            "receipt_id",
            "phase",
            "kind",
            "scope",
            "covered_gap_ids",
            "binding_sha256",
            "parent_state_sha256",
            "verdict",
            "ref",
        }
        data = _strict_fields(value, fields, "stage receipt ref")
        if data["schema_version"] != STAGE_RECEIPT_SCHEMA_VERSION:
            raise RecoveryError("unsupported stage receipt schema")
        if data["record_type"] != "numpilot_recovery_stage_receipt_ref":
            raise RecoveryError("invalid stage receipt record type")
        if not isinstance(data["covered_gap_ids"], list):
            raise RecoveryError("stage receipt covered gap ids must be an array")
        return cls(
            schema_version=_integer(
                data["schema_version"], "stage receipt schema", 1
            ),
            receipt_id=_bounded_text(
                data["receipt_id"], "stage receipt id", 128
            ),
            phase=_enum(RecoveryPhase, data["phase"], "stage receipt phase"),
            kind=_enum(StageReceiptKind, data["kind"], "stage receipt kind"),
            scope=_enum(StageReceiptScope, data["scope"], "stage receipt scope"),
            covered_gap_ids=tuple(data["covered_gap_ids"]),
            binding_sha256=_digest(
                data["binding_sha256"], "stage receipt binding digest"
            ),
            parent_state_sha256=_digest(
                data["parent_state_sha256"], "stage receipt parent state digest"
            ),
            verdict=_enum(
                StageReceiptVerdict, data["verdict"], "stage receipt verdict"
            ),
            ref=EvidenceRef.from_dict(data["ref"]),
        )


def create_stage_receipt(
    *,
    phase: RecoveryPhase | str,
    kind: StageReceiptKind | str,
    scope: StageReceiptScope | str,
    covered_gap_ids: Sequence[str],
    binding: RecoveryBinding,
    parent_state_sha256: str,
    verdict: StageReceiptVerdict | str,
    ref: EvidenceRef,
) -> StageReceiptRef:
    """Create a deterministic, fully bound stage receipt reference."""

    if not isinstance(binding, RecoveryBinding):
        raise RecoveryError("stage receipt creation requires a RecoveryBinding")
    receipt_phase = _enum(RecoveryPhase, phase, "stage receipt phase")
    receipt_kind = _enum(StageReceiptKind, kind, "stage receipt kind")
    receipt_scope = _enum(StageReceiptScope, scope, "stage receipt scope")
    covered = _normalize_gap_ids(
        covered_gap_ids,
        "stage receipt covered gap ids",
        require_nonempty=receipt_scope is StageReceiptScope.GAPS,
    )
    binding_digest = recovery_binding_sha256(binding)
    parent_digest = _digest(
        parent_state_sha256, "stage receipt parent state digest"
    )
    receipt_verdict = _enum(
        StageReceiptVerdict, verdict, "stage receipt verdict"
    )
    if not isinstance(ref, EvidenceRef):
        raise RecoveryError("stage receipt creation requires an EvidenceRef")
    body = _stage_receipt_identity_body(
        phase=receipt_phase,
        kind=receipt_kind,
        scope=receipt_scope,
        covered_gap_ids=covered,
        binding_sha256=binding_digest,
        parent_state_sha256=parent_digest,
        verdict=receipt_verdict,
        ref=ref,
    )
    return StageReceiptRef(
        receipt_id="stage-%s" % _sha256(_canonical_bytes(body))[:24],
        phase=receipt_phase,
        kind=receipt_kind,
        scope=receipt_scope,
        covered_gap_ids=covered,
        binding_sha256=binding_digest,
        parent_state_sha256=parent_digest,
        verdict=receipt_verdict,
        ref=ref,
    )


def stage_receipt_sha256(receipt: StageReceiptRef) -> str:
    if not isinstance(receipt, StageReceiptRef):
        raise RecoveryError("stage receipt digest requires a StageReceiptRef")
    return _sha256(_canonical_bytes(receipt.to_dict()))


def _normalize_stage_receipts(
    receipts: Sequence[StageReceiptRef], label: str
) -> Tuple[StageReceiptRef, ...]:
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise RecoveryError("%s must be a sequence" % label)
    normalized = tuple(receipts)
    if len(normalized) > 4096 or any(
        not isinstance(receipt, StageReceiptRef) for receipt in normalized
    ):
        raise RecoveryError("%s is invalid or too large" % label)
    receipt_ids = [receipt.receipt_id for receipt in normalized]
    if len(set(receipt_ids)) != len(receipt_ids):
        raise RecoveryError("%s contains duplicate receipt ids" % label)
    evidence_digests = [receipt.ref.sha256 for receipt in normalized]
    if len(set(evidence_digests)) != len(evidence_digests):
        raise RecoveryError("%s reuses producer receipt content" % label)
    return tuple(sorted(normalized, key=lambda receipt: receipt.receipt_id))


@dataclass(frozen=True)
class SideEffectTruth:
    subprocess_performed: bool = False
    network_access_performed: bool = False
    system_write_performed: bool = False
    model_invocation_performed: bool = False
    asnumpy_import_performed: bool = False
    build_performed: bool = False
    execution_performed: bool = False
    verification_performed: bool = False
    activation_performed: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), bool):
                raise RecoveryError("side-effect truth %s must be boolean" % name)

    @property
    def any_performed(self) -> bool:
        return any(getattr(self, name) for name in self.__dataclass_fields__)

    def cumulative_with(self, event: "SideEffectTruth") -> "SideEffectTruth":
        """Return the historical OR of this state and one event's claims."""

        if not isinstance(event, SideEffectTruth):
            raise RecoveryError("cumulative side-effect truth is invalid")
        return SideEffectTruth(
            **{
                name: getattr(self, name) or getattr(event, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "SideEffectTruth":
        fields = {
            "subprocess_performed",
            "network_access_performed",
            "system_write_performed",
            "model_invocation_performed",
            "asnumpy_import_performed",
            "build_performed",
            "execution_performed",
            "verification_performed",
            "activation_performed",
        }
        data = _strict_fields(value, fields, "side-effect truth")
        return cls(**{name: _bool(data[name], name) for name in fields})

    def validate_policy(self, policy: RecoveryPolicy) -> None:
        pairs = (
            (self.subprocess_performed, policy.subprocess_allowed, "subprocess"),
            (self.network_access_performed, policy.network_allowed, "network access"),
            (self.system_write_performed, policy.system_write_allowed, "system write"),
            (self.model_invocation_performed, policy.model_allowed, "model invocation"),
            (
                self.asnumpy_import_performed,
                policy.asnumpy_import_allowed,
                "AsNumPy import",
            ),
            (self.build_performed, policy.build_allowed, "build"),
            (self.execution_performed, policy.execution_allowed, "execution"),
            (
                self.verification_performed,
                policy.execution_allowed,
                "verification",
            ),
            (self.activation_performed, policy.activation_allowed, "activation"),
        )
        for performed, allowed, label in pairs:
            if performed and not allowed:
                raise RecoveryError(
                    "%s side effect was declared without frozen policy authorization"
                    % label
                )


@dataclass(frozen=True)
class GapRecoveryState:
    gap_id: str
    phase: RecoveryPhase
    outcome: RecoveryOutcome
    attempt_count: int = 0
    refs: Tuple[EvidenceRef, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.gap_id, str) or not _SAFE_ID.fullmatch(self.gap_id):
            raise RecoveryError("invalid recovery gap id")
        phase = _enum(RecoveryPhase, self.phase, "gap recovery phase")
        outcome = _enum(RecoveryOutcome, self.outcome, "gap recovery outcome")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "outcome", outcome)
        _integer(self.attempt_count, "gap attempt_count")
        object.__setattr__(self, "refs", _normalize_refs(self.refs, "gap refs"))
        _bounded_text(self.reason, "gap reason", 2048, allow_empty=True)
        if outcome in {
            RecoveryOutcome.AWAITING_INPUT,
            RecoveryOutcome.BLOCKED,
            RecoveryOutcome.FAILED_ATTEMPT,
        } and not self.reason:
            raise RecoveryError("non-ready gap outcome requires a reason")
        if outcome is RecoveryOutcome.COMPLETE and phase is not RecoveryPhase.VERIFIED:
            raise RecoveryError("only a verified gap can be complete")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "phase": self.phase.value,
            "outcome": self.outcome.value,
            "attempt_count": self.attempt_count,
            "refs": [item.to_dict() for item in self.refs],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> "GapRecoveryState":
        fields = {"gap_id", "phase", "outcome", "attempt_count", "refs", "reason"}
        data = _strict_fields(value, fields, "gap recovery state")
        raw_refs = data["refs"]
        if not isinstance(raw_refs, list):
            raise RecoveryError("gap refs must be an array")
        return cls(
            gap_id=_bounded_text(data["gap_id"], "gap id", 128),
            phase=_enum(RecoveryPhase, data["phase"], "gap recovery phase"),
            outcome=_enum(RecoveryOutcome, data["outcome"], "gap recovery outcome"),
            attempt_count=_integer(data["attempt_count"], "gap attempt_count"),
            refs=tuple(EvidenceRef.from_dict(item) for item in raw_refs),
            reason=_bounded_text(data["reason"], "gap reason", 2048, allow_empty=True),
        )


def _aggregate_gap_outcome(gaps: Sequence[GapRecoveryState]) -> RecoveryOutcome:
    if not gaps:
        return RecoveryOutcome.READY
    for outcome in (
        RecoveryOutcome.FAILED_ATTEMPT,
        RecoveryOutcome.BLOCKED,
        RecoveryOutcome.AWAITING_INPUT,
    ):
        if any(item.outcome is outcome for item in gaps):
            return outcome
    if all(item.outcome is RecoveryOutcome.COMPLETE for item in gaps):
        return RecoveryOutcome.COMPLETE
    return RecoveryOutcome.READY


@dataclass(frozen=True)
class RecoveryState:
    revision: int
    parent_state_sha256: str
    run: RecoveryRun
    source: RecoverySource
    target: RecoveryTarget
    runtime: RecoveryRuntime
    policy: RecoveryPolicy
    phase: RecoveryPhase
    outcome: RecoveryOutcome
    refs: Tuple[EvidenceRef, ...] = ()
    gaps: Tuple[GapRecoveryState, ...] = ()
    stage_receipts: Tuple[StageReceiptRef, ...] = ()
    side_effects: SideEffectTruth = field(default_factory=SideEffectTruth)
    event_side_effects: SideEffectTruth = field(default_factory=SideEffectTruth)
    reason: str = ""
    schema_version: int = RECOVERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != RECOVERY_SCHEMA_VERSION
        ):
            raise RecoveryError("unsupported recovery state schema")
        _integer(self.revision, "recovery revision", 1)
        _digest(
            self.parent_state_sha256,
            "parent state digest",
            zero_allowed=self.revision == 1,
        )
        if self.revision == 1 and self.parent_state_sha256 != ZERO_SHA256:
            raise RecoveryError("initial recovery state must use the zero parent digest")
        if self.revision > 1 and self.parent_state_sha256 == ZERO_SHA256:
            raise RecoveryError("non-initial recovery state requires a parent digest")
        binding = RecoveryBinding(
            self.run, self.source, self.target, self.runtime, self.policy
        )
        phase = _enum(RecoveryPhase, self.phase, "recovery phase")
        outcome = _enum(RecoveryOutcome, self.outcome, "recovery outcome")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "refs", _normalize_refs(self.refs, "recovery refs"))
        if isinstance(self.gaps, (str, bytes)) or not isinstance(self.gaps, Sequence):
            raise RecoveryError("recovery gaps must be a sequence")
        gaps = tuple(self.gaps)
        if len(gaps) > 128 or any(not isinstance(item, GapRecoveryState) for item in gaps):
            raise RecoveryError("recovery gaps are invalid or too numerous")
        if len({item.gap_id for item in gaps}) != len(gaps):
            raise RecoveryError("recovery gap ids are duplicated")
        gaps = tuple(sorted(gaps, key=lambda item: item.gap_id))
        object.__setattr__(self, "gaps", gaps)
        receipts = _normalize_stage_receipts(
            self.stage_receipts, "recovery stage receipts"
        )
        object.__setattr__(self, "stage_receipts", receipts)
        binding_digest = recovery_binding_sha256(binding)
        gap_ids = tuple(item.gap_id for item in gaps)
        gap_id_set = set(gap_ids)
        for receipt in receipts:
            if receipt.binding_sha256 != binding_digest:
                raise RecoveryError("stage receipt is bound to another recovery run")
            covered = set(receipt.covered_gap_ids)
            if receipt.scope is StageReceiptScope.RUN:
                if tuple(receipt.covered_gap_ids) != gap_ids:
                    raise RecoveryError(
                        "run-scoped stage receipt must cover every recovery gap"
                    )
            elif not covered.issubset(gap_id_set):
                raise RecoveryError("stage receipt covers an unknown recovery gap")
        if not isinstance(self.side_effects, SideEffectTruth):
            raise RecoveryError("invalid side-effect truth")
        self.side_effects.validate_policy(self.policy)
        if not isinstance(self.event_side_effects, SideEffectTruth):
            raise RecoveryError("invalid event side-effect truth")
        self.event_side_effects.validate_policy(self.policy)
        if self.revision == 1:
            if phase is not RecoveryPhase.CREATED or outcome is not RecoveryOutcome.READY:
                raise RecoveryError("initial recovery state must be CREATED and READY")
            if receipts or self.side_effects.any_performed or self.event_side_effects.any_performed:
                raise RecoveryError(
                    "initial recovery state may not contain stage receipts or side effects"
                )
            if any(
                item.phase is not RecoveryPhase.CREATED
                or item.outcome is not RecoveryOutcome.READY
                or item.attempt_count != 0
                or item.refs
                or item.reason
                for item in gaps
            ):
                raise RecoveryError("initial recovery gaps must be pristine CREATED states")
        activation_reached = _PHASE_INDEX[phase] >= _PHASE_INDEX[
            RecoveryPhase.ACTIVATED
        ]
        if self.side_effects.activation_performed is not activation_reached:
            raise RecoveryError(
                "cumulative activation truth must become true exactly at "
                "the ACTIVATED phase"
            )
        _bounded_text(self.reason, "recovery reason", 2048, allow_empty=True)
        if outcome in {
            RecoveryOutcome.AWAITING_INPUT,
            RecoveryOutcome.BLOCKED,
            RecoveryOutcome.FAILED_ATTEMPT,
        } and not self.reason:
            raise RecoveryError("non-ready recovery outcome requires a reason")
        if gaps:
            minimum_phase = min(_PHASE_INDEX[item.phase] for item in gaps)
            if minimum_phase != _PHASE_INDEX[phase]:
                raise RecoveryError(
                    "global recovery phase must equal the least advanced gap phase"
                )
            if _aggregate_gap_outcome(gaps) is not outcome:
                raise RecoveryError("global recovery outcome contradicts per-gap state")
        if outcome is RecoveryOutcome.COMPLETE:
            if phase is not RecoveryPhase.VERIFIED:
                raise RecoveryError("only verified recovery can be complete")
            if gaps and any(item.outcome is not RecoveryOutcome.COMPLETE for item in gaps):
                raise RecoveryError("all gaps must be complete")
        if phase is RecoveryPhase.VERIFIED and outcome is not RecoveryOutcome.COMPLETE:
            raise RecoveryError("verified recovery must be complete")
        if phase is RecoveryPhase.VERIFIED and not (
            self.side_effects.execution_performed
            and self.side_effects.verification_performed
        ):
            raise RecoveryError(
                "verified recovery requires execution and verification truth"
            )

    @property
    def binding(self) -> RecoveryBinding:
        return RecoveryBinding(
            self.run, self.source, self.target, self.runtime, self.policy
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_type": "numpilot_gap_recovery_state",
            "revision": self.revision,
            "parent_state_sha256": self.parent_state_sha256,
            "run": self.run.to_dict(),
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "runtime": self.runtime.to_dict(),
            "policy": self.policy.to_dict(),
            "phase": self.phase.value,
            "outcome": self.outcome.value,
            "refs": [item.to_dict() for item in self.refs],
            "gaps": [item.to_dict() for item in self.gaps],
            "stage_receipts": [item.to_dict() for item in self.stage_receipts],
            "side_effects": self.side_effects.to_dict(),
            "event_side_effects": self.event_side_effects.to_dict(),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecoveryState":
        fields = {
            "schema_version",
            "record_type",
            "revision",
            "parent_state_sha256",
            "run",
            "source",
            "target",
            "runtime",
            "policy",
            "phase",
            "outcome",
            "refs",
            "gaps",
            "stage_receipts",
            "side_effects",
            "event_side_effects",
            "reason",
        }
        data = _strict_fields(value, fields, "recovery state")
        if data["record_type"] != "numpilot_gap_recovery_state":
            raise RecoveryError("invalid recovery state record type")
        if data["schema_version"] != RECOVERY_SCHEMA_VERSION:
            raise RecoveryError("unsupported recovery state schema")
        if (
            not isinstance(data["refs"], list)
            or not isinstance(data["gaps"], list)
            or not isinstance(data["stage_receipts"], list)
        ):
            raise RecoveryError("recovery refs, gaps, and stage receipts must be arrays")
        return cls(
            schema_version=_integer(data["schema_version"], "schema version", 1),
            revision=_integer(data["revision"], "recovery revision", 1),
            parent_state_sha256=_digest(
                data["parent_state_sha256"],
                "parent state digest",
                zero_allowed=True,
            ),
            run=RecoveryRun.from_dict(data["run"]),
            source=RecoverySource.from_dict(data["source"]),
            target=RecoveryTarget.from_dict(data["target"]),
            runtime=RecoveryRuntime.from_dict(data["runtime"]),
            policy=RecoveryPolicy.from_dict(data["policy"]),
            phase=_enum(RecoveryPhase, data["phase"], "recovery phase"),
            outcome=_enum(RecoveryOutcome, data["outcome"], "recovery outcome"),
            refs=tuple(EvidenceRef.from_dict(item) for item in data["refs"]),
            gaps=tuple(GapRecoveryState.from_dict(item) for item in data["gaps"]),
            stage_receipts=tuple(
                StageReceiptRef.from_dict(item) for item in data["stage_receipts"]
            ),
            side_effects=SideEffectTruth.from_dict(data["side_effects"]),
            event_side_effects=SideEffectTruth.from_dict(
                data["event_side_effects"]
            ),
            reason=_bounded_text(data["reason"], "recovery reason", 2048, allow_empty=True),
        )


@dataclass(frozen=True)
class RecoveryEvent:
    """One pure reducer input, explicitly bound to immutable run inputs."""

    binding: RecoveryBinding
    phase: RecoveryPhase
    outcome: RecoveryOutcome
    refs: Tuple[EvidenceRef, ...] = ()
    gaps: Tuple[GapRecoveryState, ...] = ()
    stage_receipts: Tuple[StageReceiptRef, ...] = ()
    side_effects: SideEffectTruth = field(default_factory=SideEffectTruth)
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.binding, RecoveryBinding):
            raise RecoveryError("recovery event requires an immutable binding")
        object.__setattr__(
            self, "phase", _enum(RecoveryPhase, self.phase, "event phase")
        )
        object.__setattr__(
            self, "outcome", _enum(RecoveryOutcome, self.outcome, "event outcome")
        )
        object.__setattr__(self, "refs", _normalize_refs(self.refs, "event refs"))
        if isinstance(self.gaps, (str, bytes)) or not isinstance(self.gaps, Sequence):
            raise RecoveryError("event gaps must be a sequence")
        if any(not isinstance(item, GapRecoveryState) for item in self.gaps):
            raise RecoveryError("event gaps are invalid")
        gaps = tuple(self.gaps)
        object.__setattr__(self, "gaps", gaps)
        receipts = _normalize_stage_receipts(
            self.stage_receipts, "event stage receipts"
        )
        object.__setattr__(self, "stage_receipts", receipts)
        binding_digest = recovery_binding_sha256(self.binding)
        gap_ids = tuple(sorted(item.gap_id for item in gaps))
        gap_id_set = set(gap_ids)
        for receipt in receipts:
            if receipt.binding_sha256 != binding_digest:
                raise RecoveryError("event stage receipt binding does not match")
            covered = set(receipt.covered_gap_ids)
            if receipt.scope is StageReceiptScope.RUN:
                if receipt.covered_gap_ids != gap_ids:
                    raise RecoveryError(
                        "run-scoped event receipt must cover every recovery gap"
                    )
            elif not covered.issubset(gap_id_set):
                raise RecoveryError("event stage receipt covers an unknown gap")
        if not isinstance(self.side_effects, SideEffectTruth):
            raise RecoveryError("event side-effect truth is invalid")
        self.side_effects.validate_policy(self.binding.policy)
        _bounded_text(self.reason, "event reason", 2048, allow_empty=True)

    @classmethod
    def bound_to(
        cls,
        state: RecoveryState,
        *,
        phase: RecoveryPhase,
        outcome: RecoveryOutcome = RecoveryOutcome.READY,
        refs: Sequence[EvidenceRef] = (),
        gaps: Optional[Sequence[GapRecoveryState]] = None,
        stage_receipts: Sequence[StageReceiptRef] = (),
        side_effects: Optional[SideEffectTruth] = None,
        reason: str = "",
    ) -> "RecoveryEvent":
        return cls(
            binding=state.binding,
            phase=phase,
            outcome=outcome,
            refs=tuple(refs),
            gaps=state.gaps if gaps is None else tuple(gaps),
            stage_receipts=tuple(stage_receipts),
            side_effects=(
                side_effects if side_effects is not None else SideEffectTruth()
            ),
            reason=reason,
        )


def _merge_refs(
    previous: Sequence[EvidenceRef], additions: Sequence[EvidenceRef]
) -> Tuple[EvidenceRef, ...]:
    merged = {item.name: item for item in previous}
    for item in additions:
        existing = merged.get(item.name)
        if existing is not None and existing != item:
            raise RecoveryError(
                "evidence reference drift for immutable name: %s" % item.name
            )
        merged[item.name] = item
    return tuple(sorted(merged.values(), key=lambda item: item.name))


def _merge_stage_receipts(
    previous: Sequence[StageReceiptRef], additions: Sequence[StageReceiptRef]
) -> Tuple[StageReceiptRef, ...]:
    merged = {item.receipt_id: item for item in previous}
    used_content = {item.ref.sha256 for item in previous}
    for item in additions:
        if item.receipt_id in merged:
            raise RecoveryError("event stage receipt must be newly added")
        if item.ref.sha256 in used_content:
            raise RecoveryError("producer receipt content may not be reused")
        merged[item.receipt_id] = item
        used_content.add(item.ref.sha256)
    return _normalize_stage_receipts(
        tuple(merged.values()), "recovery stage receipts"
    )


def _validate_gap_transition(
    previous: Sequence[GapRecoveryState],
    current: Sequence[GapRecoveryState],
) -> None:
    before = {item.gap_id: item for item in previous}
    after = {item.gap_id: item for item in current}
    if set(before) != set(after):
        raise RecoveryError("recovery gap set drifted after creation")
    if len(after) != len(current):
        raise RecoveryError("event contains duplicate gap ids")
    for gap_id, item in after.items():
        old = before.get(gap_id)
        if old is None:
            continue
        delta = _PHASE_INDEX[item.phase] - _PHASE_INDEX[old.phase]
        if delta < 0 or delta > 1:
            raise RecoveryError("gap phase may advance by exactly one stage at most")
        old_refs = {ref.name: ref for ref in old.refs}
        for ref in item.refs:
            if ref.name in old_refs and old_refs[ref.name] != ref:
                raise RecoveryError("per-gap evidence reference drift")
        if not set(old_refs).issubset({ref.name for ref in item.refs}):
            raise RecoveryError("per-gap evidence references may not be removed")
        if item.outcome is RecoveryOutcome.FAILED_ATTEMPT:
            if delta != 0 or item.attempt_count != old.attempt_count + 1:
                raise RecoveryError(
                    "failed gap attempt must increment attempt_count without advancing"
                )
        elif item.attempt_count != old.attempt_count:
            raise RecoveryError("gap attempt_count changes only on a failed attempt")


def _advanced_gaps_by_phase(
    previous: RecoveryState, current: RecoveryState
) -> Dict[RecoveryPhase, set[str]]:
    before = {item.gap_id: item for item in previous.gaps}
    advanced: Dict[RecoveryPhase, set[str]] = {}
    for item in current.gaps:
        old = before[item.gap_id]
        if _PHASE_INDEX[item.phase] == _PHASE_INDEX[old.phase] + 1:
            advanced.setdefault(item.phase, set()).add(item.gap_id)
    if not current.gaps:
        delta = _PHASE_INDEX[current.phase] - _PHASE_INDEX[previous.phase]
        if delta == 1:
            advanced[current.phase] = set()
    return advanced


def _validate_stage_receipt_transition(
    previous: RecoveryState,
    current: RecoveryState,
    previous_sha256: str,
) -> None:
    before = {item.receipt_id: item for item in previous.stage_receipts}
    after = {item.receipt_id: item for item in current.stage_receipts}
    if not set(before).issubset(after):
        raise RecoveryError("stage receipts may not be removed")
    for receipt_id, item in before.items():
        if after[receipt_id] != item:
            raise RecoveryError("stage receipt drift: %s" % receipt_id)
    additions = tuple(
        item for receipt_id, item in after.items() if receipt_id not in before
    )
    binding_digest = recovery_binding_sha256(current.binding)
    known_gap_ids = {item.gap_id for item in current.gaps}
    for receipt in additions:
        if receipt.binding_sha256 != binding_digest:
            raise RecoveryError("new stage receipt binding does not match")
        if receipt.parent_state_sha256 != previous_sha256:
            raise RecoveryError("new stage receipt is not bound to the current head")
        if receipt.scope is StageReceiptScope.RUN:
            if set(receipt.covered_gap_ids) != known_gap_ids:
                raise RecoveryError("run-scoped receipt coverage does not match the run")
        elif not set(receipt.covered_gap_ids).issubset(known_gap_ids):
            raise RecoveryError("gap-scoped receipt covers an unknown gap")

    advanced = _advanced_gaps_by_phase(previous, current)
    satisfied = tuple(
        receipt
        for receipt in additions
        if receipt.verdict is StageReceiptVerdict.SATISFIED
    )
    for receipt in satisfied:
        if receipt.phase not in advanced:
            raise RecoveryError(
                "satisfied stage receipt does not match a phase advance"
            )

    for phase, expected_gap_ids in advanced.items():
        matching = tuple(receipt for receipt in satisfied if receipt.phase is phase)
        if not matching:
            raise RecoveryError(
                "phase advance requires a new satisfied %s receipt"
                % _STAGE_KIND_BY_PHASE[phase].value
            )
        covered: set[str] = set()
        for receipt in matching:
            receipt_gaps = set(receipt.covered_gap_ids)
            if covered.intersection(receipt_gaps):
                raise RecoveryError("stage receipts overlap gap coverage")
            covered.update(receipt_gaps)
        if covered != expected_gap_ids:
            raise RecoveryError("stage receipt coverage does not match advanced gaps")

    before_gaps = {item.gap_id: item for item in previous.gaps}
    for receipt in additions:
        if receipt.verdict is StageReceiptVerdict.SATISFIED:
            continue
        if receipt.scope is StageReceiptScope.RUN and not previous.gaps:
            expected_phase = _PHASES[_PHASE_INDEX[previous.phase] + 1]
            if receipt.phase is not expected_phase:
                raise RecoveryError("failed stage receipt targets the wrong phase")
            continue
        for gap_id in receipt.covered_gap_ids:
            old = before_gaps[gap_id]
            if _PHASE_INDEX[old.phase] + 1 >= len(_PHASES):
                raise RecoveryError("completed gap cannot receive another stage receipt")
            if receipt.phase is not _PHASES[_PHASE_INDEX[old.phase] + 1]:
                raise RecoveryError("failed stage receipt targets the wrong gap phase")


def _validate_successor(
    previous: RecoveryState,
    current: RecoveryState,
    previous_sha256: str,
) -> None:
    if current.revision != previous.revision + 1:
        raise RecoveryError("recovery revision is not contiguous")
    if current.parent_state_sha256 != previous_sha256:
        raise RecoveryError("recovery parent state digest does not match")
    if current.binding != previous.binding:
        raise RecoveryError("recovery inputs drifted after run creation")
    delta = _PHASE_INDEX[current.phase] - _PHASE_INDEX[previous.phase]
    if delta < 0 or delta > 1:
        raise RecoveryError("recovery phase may advance by exactly one stage at most")
    if current.outcome in {
        RecoveryOutcome.AWAITING_INPUT,
        RecoveryOutcome.BLOCKED,
        RecoveryOutcome.FAILED_ATTEMPT,
    } and delta != 0:
        raise RecoveryError("non-ready or failed recovery may not advance phase")
    if current.outcome is RecoveryOutcome.FAILED_ATTEMPT and not any(
        item.outcome is RecoveryOutcome.FAILED_ATTEMPT for item in current.gaps
    ) and current.gaps:
        raise RecoveryError("failed recovery attempt requires a failed gap attempt")
    if previous.outcome is RecoveryOutcome.COMPLETE:
        raise RecoveryError("completed recovery is terminal")
    old_refs = {item.name: item for item in previous.refs}
    new_refs = {item.name: item for item in current.refs}
    if not set(old_refs).issubset(new_refs):
        raise RecoveryError("recovery evidence references may not be removed")
    for name, item in old_refs.items():
        if new_refs[name] != item:
            raise RecoveryError("recovery evidence reference drift: %s" % name)
    _validate_gap_transition(previous.gaps, current.gaps)
    expected_side_effects = previous.side_effects.cumulative_with(
        current.event_side_effects
    )
    if current.side_effects != expected_side_effects:
        raise RecoveryError("cumulative side-effect truth is inconsistent")
    if (
        current.event_side_effects.activation_performed
        and current.phase is not RecoveryPhase.ACTIVATED
    ):
        raise RecoveryError(
            "activation may be declared only when the resulting phase is ACTIVATED"
        )
    if current.phase is RecoveryPhase.VERIFIED:
        if not (
            current.event_side_effects.execution_performed
            and current.event_side_effects.verification_performed
        ):
            raise RecoveryError(
                "VERIFIED requires execution and verification in the current event"
            )
    _validate_stage_receipt_transition(previous, current, previous_sha256)


def reduce_recovery(
    previous: RecoveryState,
    event: RecoveryEvent,
    *,
    parent_state_sha256: str,
) -> RecoveryState:
    """Purely reduce one bound event into the next hash-chain state."""

    if not isinstance(previous, RecoveryState) or not isinstance(event, RecoveryEvent):
        raise RecoveryError("recovery reducer received an invalid value")
    parent_digest = _digest(parent_state_sha256, "parent state digest")
    expected_parent_digest = _sha256(
        _canonical_file_bytes(previous.to_dict())
    )
    if parent_digest != expected_parent_digest:
        raise RecoveryError("parent state digest does not match the supplied state")
    if event.binding != previous.binding:
        raise RecoveryError("recovery input drift rejected")
    if (
        event.side_effects.activation_performed
        and event.phase is not RecoveryPhase.ACTIVATED
    ):
        raise RecoveryError(
            "activation may be declared only when the resulting phase is ACTIVATED"
        )
    if event.phase is RecoveryPhase.VERIFIED and not (
        event.side_effects.execution_performed
        and event.side_effects.verification_performed
    ):
        raise RecoveryError(
            "VERIFIED requires execution and verification in the current event"
        )
    gaps = event.gaps
    outcome = event.outcome
    if gaps and _aggregate_gap_outcome(gaps) is not outcome:
        raise RecoveryError("event outcome contradicts per-gap state")
    state = RecoveryState(
        revision=previous.revision + 1,
        parent_state_sha256=parent_digest,
        run=previous.run,
        source=previous.source,
        target=previous.target,
        runtime=previous.runtime,
        policy=previous.policy,
        phase=event.phase,
        outcome=outcome,
        refs=_merge_refs(previous.refs, event.refs),
        gaps=gaps,
        stage_receipts=_merge_stage_receipts(
            previous.stage_receipts, event.stage_receipts
        ),
        side_effects=previous.side_effects.cumulative_with(event.side_effects),
        event_side_effects=event.side_effects,
        reason=event.reason,
    )
    _validate_successor(previous, state, parent_digest)
    return state


def next_action(state: RecoveryState) -> str:
    """Return one deterministic action code; it never performs the action."""

    if not isinstance(state, RecoveryState):
        raise RecoveryError("next_action requires a RecoveryState")
    if state.outcome is RecoveryOutcome.COMPLETE:
        return "none"
    if state.outcome is RecoveryOutcome.AWAITING_INPUT:
        return "collect_required_input"
    if state.outcome is RecoveryOutcome.BLOCKED:
        return "inspect_blocker"
    if state.outcome is RecoveryOutcome.FAILED_ATTEMPT:
        return "retry_%s" % _NEXT_ACTION[state.phase]
    return _NEXT_ACTION[state.phase]


@dataclass(frozen=True)
class RecoveryStatus:
    state: RecoveryState
    state_sha256: str
    history_length: int
    next_action: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, RecoveryState):
            raise RecoveryError("recovery status requires a RecoveryState")
        _digest(self.state_sha256, "recovery status state digest")
        expected_digest = _sha256(_canonical_file_bytes(self.state.to_dict()))
        if self.state_sha256 != expected_digest:
            raise RecoveryError("recovery status state digest is inconsistent")
        _integer(self.history_length, "recovery status history length", 1)
        if self.history_length != self.state.revision:
            raise RecoveryError("recovery status history length is inconsistent")
        if self.next_action != next_action(self.state):
            raise RecoveryError("recovery status next action is inconsistent")

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-compatible status snapshot."""

        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "record_type": "numpilot_gap_recovery_status",
            "history_length": self.history_length,
            "state_sha256": self.state_sha256,
            "next_action": self.next_action,
            "state": self.state.to_dict(),
        }


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if os.path.lexists(current) and current.is_symlink():
            raise RecoveryError("recovery path contains a symlink: %s" % current)
        if current == current.parent:
            return
        current = current.parent


def _private_directory(path: Path, *, create: bool) -> int:
    _reject_symlink_components(path)
    created = False
    if not os.path.lexists(path):
        if not create:
            raise RecoveryError("recovery directory does not exist: %s" % path)
        parent = path.parent
        _reject_symlink_components(parent)
        if not parent.is_dir():
            raise RecoveryError("recovery directory parent does not exist: %s" % parent)
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise RecoveryError("cannot create recovery directory: %s" % path) from exc
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RecoveryError("cannot open recovery directory: %s" % path) from exc
    if created:
        try:
            os.fchmod(descriptor, 0o700)
        except OSError as exc:
            os.close(descriptor)
            raise RecoveryError(
                "cannot make recovery directory private: %s" % path
            ) from exc
    details = os.fstat(descriptor)
    if not stat.S_ISDIR(details.st_mode):
        os.close(descriptor)
        raise RecoveryError("recovery path is not a directory: %s" % path)
    if stat.S_IMODE(details.st_mode) != 0o700:
        os.close(descriptor)
        raise RecoveryError("recovery directory mode must be exactly 0700: %s" % path)
    return descriptor


def _parse_json(payload: bytes, label: str) -> object:
    def pairs(items: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise RecoveryError("duplicate JSON key in %s: %s" % (label, key))
            value[key] = item
        return value

    def integer(raw: str) -> int:
        if len(raw.lstrip("-")) > 64:
            raise RecoveryError("oversized integer in %s" % label)
        return int(raw)

    def constant(raw: str) -> None:
        raise RecoveryError("non-finite number in %s: %s" % (label, raw))

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_int=integer,
            parse_constant=constant,
        )
    except RecoveryError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        raise RecoveryError("invalid JSON in %s" % label) from exc


def _read_private_state(directory_fd: int, name: str) -> Tuple[RecoveryState, bytes]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RecoveryError("cannot open recovery state: %s" % name) from exc
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
            or details.st_size <= 0
            or details.st_size > _MAX_STATE_BYTES
        ):
            raise RecoveryError(
                "recovery state must be a bounded private 0600 regular file: %s"
                % name
            )
        chunks = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 128 * 1024))
            if not chunk:
                raise RecoveryError("recovery state was truncated: %s" % name)
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    state = RecoveryState.from_dict(_parse_json(payload, name))
    if payload != _canonical_file_bytes(state.to_dict()):
        raise RecoveryError("recovery state is not canonical JSON: %s" % name)
    return state, payload


def _publish_exclusive(directory_fd: int, name: str, payload: bytes) -> None:
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAX_STATE_BYTES
    ):
        raise RecoveryError("recovery state payload exceeds the size limit")
    temporary = ".state-%s.tmp" % secrets.token_hex(16)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RecoveryError("short write while publishing recovery state")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise RecoveryError("recovery revision already exists: %s" % name) from exc
        linked = True
        os.unlink(temporary, dir_fd=directory_fd)
        temporary = ""
        os.fsync(directory_fd)
    except RecoveryError:
        raise
    except OSError as exc:
        visible = " after exclusive publication" if linked else ""
        raise RecoveryError("cannot publish recovery state%s" % visible) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass


class RecoveryStore:
    """A private ``states/000001.json`` write-once SHA-256 chain."""

    def __init__(self, root: os.PathLike[str] | str):
        raw = os.fspath(root)
        if not isinstance(raw, str) or not raw or "\x00" in raw or len(raw) > 4096:
            raise RecoveryError("invalid recovery store path")
        self.root = Path(os.path.abspath(os.path.expanduser(raw)))
        self.states = self.root / "states"

    def _open_states(
        self,
        *,
        create: bool,
        exclusive: bool,
    ) -> Tuple[int, int]:
        """Open the store while holding its directory advisory lock.

        Readers share the root-directory lock; create/append hold it
        exclusively from head inspection through publication.  A live
        ``.state-*.tmp`` is therefore never mistaken for an interrupted
        publication by a cooperating reader.  If a writer exits abnormally,
        the kernel releases its lock and the still-present temporary remains
        visible to :meth:`_state_names`, which continues to fail closed.
        """

        root_fd = _private_directory(self.root, create=create)
        try:
            try:
                fcntl.flock(
                    root_fd,
                    fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
                )
            except OSError as exc:
                raise RecoveryError("cannot lock recovery store") from exc
            states_fd = _private_directory(self.states, create=create)
        except Exception:
            os.close(root_fd)
            raise
        return root_fd, states_fd

    @staticmethod
    def _state_names(directory_fd: int) -> Tuple[str, ...]:
        try:
            names = os.listdir(directory_fd)
        except OSError as exc:
            raise RecoveryError("cannot list recovery states") from exc
        state_names = []
        for name in names:
            if re.fullmatch(r"[0-9]{6}\.json", name):
                state_names.append(name)
            elif name.startswith(".state-") and name.endswith(".tmp"):
                raise RecoveryError("incomplete recovery publication is present")
            else:
                raise RecoveryError("unknown entry in recovery states directory: %s" % name)
        return tuple(sorted(state_names))

    def create(
        self,
        binding: RecoveryBinding,
        *,
        gap_ids: Sequence[str] = (),
        refs: Sequence[EvidenceRef] = (),
    ) -> RecoveryStatus:
        if not isinstance(binding, RecoveryBinding):
            raise RecoveryError("recovery creation requires an immutable binding")
        if isinstance(gap_ids, (str, bytes)) or not isinstance(gap_ids, Sequence):
            raise RecoveryError("gap_ids must be a sequence")
        if len(set(gap_ids)) != len(gap_ids):
            raise RecoveryError("gap_ids are duplicated")
        gaps = tuple(
            GapRecoveryState(
                gap_id=str(gap_id),
                phase=RecoveryPhase.CREATED,
                outcome=RecoveryOutcome.READY,
            )
            for gap_id in gap_ids
        )
        initial = RecoveryState(
            revision=1,
            parent_state_sha256=ZERO_SHA256,
            run=binding.run,
            source=binding.source,
            target=binding.target,
            runtime=binding.runtime,
            policy=binding.policy,
            phase=RecoveryPhase.CREATED,
            outcome=RecoveryOutcome.READY,
            refs=tuple(refs),
            gaps=gaps,
            side_effects=SideEffectTruth(),
        )
        root_fd, states_fd = self._open_states(create=True, exclusive=True)
        try:
            if self._state_names(states_fd):
                raise RecoveryError("recovery store is already initialized")
            payload = _canonical_file_bytes(initial.to_dict())
            _publish_exclusive(states_fd, "000001.json", payload)
        finally:
            os.close(states_fd)
            os.close(root_fd)
        return RecoveryStatus(initial, _sha256(payload), 1, next_action(initial))

    def _load_history_locked(
        self,
        states_fd: int,
    ) -> Tuple[Tuple[RecoveryState, str], ...]:
        """Validate a complete chain while the caller retains the root lock."""

        names = self._state_names(states_fd)
        if not names:
            raise RecoveryError("recovery state history is empty")
        if len(names) > _MAX_HISTORY_RECORDS:
            raise RecoveryError("recovery state history is too large")
        expected_names = tuple(
            "%06d.json" % revision
            for revision in range(1, len(names) + 1)
        )
        if names != expected_names:
            raise RecoveryError("recovery revisions are not contiguous")
        history = []
        previous: Optional[RecoveryState] = None
        previous_sha = ZERO_SHA256
        for name in names:
            state, payload = _read_private_state(states_fd, name)
            digest = _sha256(payload)
            if state.revision != len(history) + 1:
                raise RecoveryError("state revision does not match its filename")
            if previous is None:
                if state.parent_state_sha256 != ZERO_SHA256:
                    raise RecoveryError("initial state parent digest is invalid")
            else:
                _validate_successor(previous, state, previous_sha)
            history.append((state, digest))
            previous = state
            previous_sha = digest
        return tuple(history)

    @staticmethod
    def _status_from_history(
        history: Tuple[Tuple[RecoveryState, str], ...],
    ) -> RecoveryStatus:
        state, digest = history[-1]
        return RecoveryStatus(state, digest, len(history), next_action(state))

    def load_history(self) -> Tuple[Tuple[RecoveryState, str], ...]:
        root_fd, states_fd = self._open_states(
            create=False,
            exclusive=False,
        )
        try:
            return self._load_history_locked(states_fd)
        finally:
            os.close(states_fd)
            os.close(root_fd)

    def status(self) -> RecoveryStatus:
        root_fd, states_fd = self._open_states(
            create=False,
            exclusive=False,
        )
        try:
            return self._status_from_history(
                self._load_history_locked(states_fd)
            )
        finally:
            os.close(states_fd)
            os.close(root_fd)

    def append(self, event: RecoveryEvent) -> RecoveryStatus:
        root_fd, states_fd = self._open_states(
            create=False,
            exclusive=True,
        )
        try:
            current_status = self._status_from_history(
                self._load_history_locked(states_fd)
            )
            next_state = reduce_recovery(
                current_status.state,
                event,
                parent_state_sha256=current_status.state_sha256,
            )
            payload = _canonical_file_bytes(next_state.to_dict())
            name = "%06d.json" % next_state.revision

            # Re-read the head immediately before publication.  The advisory
            # lock serializes cooperating writers; this extra comparison also
            # fails closed if a non-cooperating process touched the directory.
            names = self._state_names(states_fd)
            if not names or names[-1] != "%06d.json" % current_status.state.revision:
                raise RecoveryError("recovery head changed before append")
            head, head_payload = _read_private_state(states_fd, names[-1])
            if head != current_status.state or _sha256(head_payload) != current_status.state_sha256:
                raise RecoveryError("recovery head changed before append")
            _publish_exclusive(states_fd, name, payload)
            return RecoveryStatus(
                next_state,
                _sha256(payload),
                current_status.history_length + 1,
                next_action(next_state),
            )
        finally:
            os.close(states_fd)
            os.close(root_fd)


def load_recovery_status(root: os.PathLike[str] | str) -> RecoveryStatus:
    """Load and verify the complete chain, returning its deterministic head."""

    return RecoveryStore(root).status()


@dataclass(frozen=True)
class LicenseBinding:
    identifier: str
    accepted: bool
    ref: EvidenceRef

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not _LICENSE_ID.fullmatch(
            self.identifier
        ):
            raise RecoveryError("invalid candidate license identifier")
        if not isinstance(self.accepted, bool):
            raise RecoveryError("candidate license acceptance must be boolean")
        if not isinstance(self.ref, EvidenceRef):
            raise RecoveryError("candidate license requires a content-bound ref")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "accepted": self.accepted,
            "ref": self.ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "LicenseBinding":
        data = _strict_fields(value, {"identifier", "accepted", "ref"}, "license binding")
        return cls(
            identifier=_bounded_text(data["identifier"], "license identifier", 128),
            accepted=_bool(data["accepted"], "license acceptance"),
            ref=EvidenceRef.from_dict(data["ref"]),
        )


def _candidate_identity_body(
    *,
    origin_kind: OperatorCandidateOrigin,
    gap_id: str,
    qualified_api: str,
    gap_spec_ref: EvidenceRef,
    resolution_decision_ref: EvidenceRef,
    target_soc: str,
    cann_version: str,
    license: LicenseBinding,
    provenance_refs: Sequence[EvidenceRef],
    candidate_refs: Sequence[EvidenceRef],
) -> Dict[str, Any]:
    return {
        "origin_kind": origin_kind.value,
        "gap_id": gap_id,
        "qualified_api": qualified_api,
        "gap_spec_ref": gap_spec_ref.to_dict(),
        "resolution_decision_ref": resolution_decision_ref.to_dict(),
        "target_soc": target_soc,
        "cann_version": cann_version,
        "license": license.to_dict(),
        "provenance_refs": [item.to_dict() for item in provenance_refs],
        "candidate_refs": [item.to_dict() for item in candidate_refs],
        "activation_allowed": False,
    }


@dataclass(frozen=True)
class OperatorCandidateArtifact:
    """A route-neutral candidate receipt; never an activation capability."""

    candidate_id: str
    origin_kind: OperatorCandidateOrigin
    gap_id: str
    qualified_api: str
    gap_spec_ref: EvidenceRef
    resolution_decision_ref: EvidenceRef
    target_soc: str
    cann_version: str
    license: LicenseBinding
    provenance_refs: Tuple[EvidenceRef, ...]
    candidate_refs: Tuple[EvidenceRef, ...]
    activation_allowed: bool = False
    schema_version: int = OPERATOR_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != OPERATOR_CANDIDATE_SCHEMA_VERSION
        ):
            raise RecoveryError("unsupported operator candidate schema")
        origin = _enum(
            OperatorCandidateOrigin, self.origin_kind, "operator candidate origin"
        )
        object.__setattr__(self, "origin_kind", origin)
        if not isinstance(self.gap_id, str) or not _SAFE_ID.fullmatch(self.gap_id):
            raise RecoveryError("invalid candidate gap id")
        _bounded_text(self.qualified_api, "candidate qualified API", 256)
        if not _QUALIFIED_API.fullmatch(self.qualified_api):
            raise RecoveryError("candidate qualified API is not supported")
        if not isinstance(self.gap_spec_ref, EvidenceRef) or not isinstance(
            self.resolution_decision_ref, EvidenceRef
        ):
            raise RecoveryError("candidate requires gap and resolution references")
        _bounded_text(self.target_soc, "candidate target SoC", 128)
        _bounded_text(self.cann_version, "candidate CANN version", 128)
        if not isinstance(self.license, LicenseBinding):
            raise RecoveryError("candidate requires an explicit license binding")
        provenance = _normalize_refs(
            self.provenance_refs,
            "candidate provenance refs",
            require_nonempty=True,
        )
        candidates = _normalize_refs(
            self.candidate_refs,
            "candidate content refs",
            require_nonempty=True,
        )
        object.__setattr__(self, "provenance_refs", provenance)
        object.__setattr__(self, "candidate_refs", candidates)
        if self.activation_allowed is not False:
            raise RecoveryError("operator candidate can never allow activation")
        body = _candidate_identity_body(
            origin_kind=origin,
            gap_id=self.gap_id,
            qualified_api=self.qualified_api,
            gap_spec_ref=self.gap_spec_ref,
            resolution_decision_ref=self.resolution_decision_ref,
            target_soc=self.target_soc,
            cann_version=self.cann_version,
            license=self.license,
            provenance_refs=provenance,
            candidate_refs=candidates,
        )
        expected_id = "candidate-%s" % _sha256(_canonical_bytes(body))[:20]
        if self.candidate_id != expected_id:
            raise RecoveryError("operator candidate id does not match its bindings")

    def to_dict(self) -> Dict[str, Any]:
        body = _candidate_identity_body(
            origin_kind=self.origin_kind,
            gap_id=self.gap_id,
            qualified_api=self.qualified_api,
            gap_spec_ref=self.gap_spec_ref,
            resolution_decision_ref=self.resolution_decision_ref,
            target_soc=self.target_soc,
            cann_version=self.cann_version,
            license=self.license,
            provenance_refs=self.provenance_refs,
            candidate_refs=self.candidate_refs,
        )
        return {
            "schema_version": self.schema_version,
            "artifact_type": "operator_candidate",
            "candidate_id": self.candidate_id,
            **body,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OperatorCandidateArtifact":
        fields = {
            "schema_version",
            "artifact_type",
            "candidate_id",
            "origin_kind",
            "gap_id",
            "qualified_api",
            "gap_spec_ref",
            "resolution_decision_ref",
            "target_soc",
            "cann_version",
            "license",
            "provenance_refs",
            "candidate_refs",
            "activation_allowed",
        }
        data = _strict_fields(value, fields, "operator candidate artifact")
        if data["schema_version"] != OPERATOR_CANDIDATE_SCHEMA_VERSION:
            raise RecoveryError("unsupported operator candidate schema")
        if data["artifact_type"] != "operator_candidate":
            raise RecoveryError("invalid operator candidate artifact type")
        if not isinstance(data["provenance_refs"], list) or not isinstance(
            data["candidate_refs"], list
        ):
            raise RecoveryError("candidate refs must be arrays")
        return cls(
            schema_version=_integer(data["schema_version"], "candidate schema", 1),
            candidate_id=_bounded_text(data["candidate_id"], "candidate id", 128),
            origin_kind=_enum(
                OperatorCandidateOrigin, data["origin_kind"], "candidate origin"
            ),
            gap_id=_bounded_text(data["gap_id"], "candidate gap id", 128),
            qualified_api=_bounded_text(
                data["qualified_api"], "candidate qualified API", 256
            ),
            gap_spec_ref=EvidenceRef.from_dict(data["gap_spec_ref"]),
            resolution_decision_ref=EvidenceRef.from_dict(
                data["resolution_decision_ref"]
            ),
            target_soc=_bounded_text(data["target_soc"], "candidate target SoC", 128),
            cann_version=_bounded_text(
                data["cann_version"], "candidate CANN version", 128
            ),
            license=LicenseBinding.from_dict(data["license"]),
            provenance_refs=tuple(
                EvidenceRef.from_dict(item) for item in data["provenance_refs"]
            ),
            candidate_refs=tuple(
                EvidenceRef.from_dict(item) for item in data["candidate_refs"]
            ),
            activation_allowed=_bool(
                data["activation_allowed"], "candidate activation permission"
            ),
        )


def create_operator_candidate(
    *,
    origin_kind: OperatorCandidateOrigin | str,
    gap_id: str,
    qualified_api: str,
    gap_spec_ref: EvidenceRef,
    resolution_decision_ref: EvidenceRef,
    target_soc: str,
    cann_version: str,
    license: LicenseBinding,
    provenance_refs: Sequence[EvidenceRef],
    candidate_refs: Sequence[EvidenceRef],
) -> OperatorCandidateArtifact:
    """Create a deterministic candidate id from all immutable bindings."""

    origin = _enum(OperatorCandidateOrigin, origin_kind, "candidate origin")
    provenance = _normalize_refs(
        provenance_refs, "candidate provenance refs", require_nonempty=True
    )
    candidates = _normalize_refs(
        candidate_refs, "candidate content refs", require_nonempty=True
    )
    body = _candidate_identity_body(
        origin_kind=origin,
        gap_id=gap_id,
        qualified_api=qualified_api,
        gap_spec_ref=gap_spec_ref,
        resolution_decision_ref=resolution_decision_ref,
        target_soc=target_soc,
        cann_version=cann_version,
        license=license,
        provenance_refs=provenance,
        candidate_refs=candidates,
    )
    candidate_id = "candidate-%s" % _sha256(_canonical_bytes(body))[:20]
    return OperatorCandidateArtifact(
        candidate_id=candidate_id,
        origin_kind=origin,
        gap_id=gap_id,
        qualified_api=qualified_api,
        gap_spec_ref=gap_spec_ref,
        resolution_decision_ref=resolution_decision_ref,
        target_soc=target_soc,
        cann_version=cann_version,
        license=license,
        provenance_refs=provenance,
        candidate_refs=candidates,
    )


def operator_candidate_sha256(candidate: OperatorCandidateArtifact) -> str:
    if not isinstance(candidate, OperatorCandidateArtifact):
        raise RecoveryError("candidate digest requires an OperatorCandidateArtifact")
    return _sha256(_canonical_bytes(candidate.to_dict()))


def validate_operator_candidate_binding(
    candidate: OperatorCandidateArtifact,
    *,
    gap_spec: Mapping[str, Any],
    resolution_decision: Mapping[str, Any],
) -> None:
    """Validate supplied JSON objects against the candidate's content bindings."""

    if not isinstance(candidate, OperatorCandidateArtifact):
        raise RecoveryError("invalid operator candidate")
    if not isinstance(gap_spec, Mapping) or not isinstance(
        resolution_decision, Mapping
    ):
        raise RecoveryError("candidate binding inputs must be objects")
    gap_payload = _canonical_bytes(dict(gap_spec))
    decision_payload = _canonical_bytes(dict(resolution_decision))
    if _sha256(gap_payload) != candidate.gap_spec_ref.sha256:
        raise RecoveryError("candidate GapSpec binding does not match")
    if _sha256(decision_payload) != candidate.resolution_decision_ref.sha256:
        raise RecoveryError("candidate resolution binding does not match")
    if gap_spec.get("gap_id") != candidate.gap_id:
        raise RecoveryError("candidate is bound to another gap id")
    if gap_spec.get("qualified_api") != candidate.qualified_api:
        raise RecoveryError("candidate is bound to another qualified API")
    if gap_spec.get("target_soc") != candidate.target_soc:
        raise RecoveryError("candidate is bound to another target SoC")
    if gap_spec.get("cann_version") != candidate.cann_version:
        raise RecoveryError("candidate is bound to another CANN version")
    if resolution_decision.get("gap_id") != candidate.gap_id:
        raise RecoveryError("resolution decision is bound to another gap")
    if resolution_decision.get("qualified_api") != candidate.qualified_api:
        raise RecoveryError("resolution decision is bound to another API")
    expected_route = _ORIGIN_RESOLUTION_ROUTE[candidate.origin_kind]
    if resolution_decision.get("route") != expected_route:
        raise RecoveryError(
            "candidate origin %s requires resolution route %s"
            % (candidate.origin_kind.value, expected_route)
        )
