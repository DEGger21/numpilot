"""Fail-closed routing for installed and source-backed official operators.

The resolver deliberately performs discovery only.  It never loads a CANN
shared library, builds an operator, writes an OPP tree, fetches a repository,
calls a model, or activates a capability.  Its output is a content-bound
receipt that tells the caller which *next* workflow is permitted:

1. an installed ACLNN adapter candidate;
2. a version-pinned official source candidate requiring semantic audit;
3. new-kernel generation, but only after both official layers returned a
   clean ``no_match``; or
4. a fail-closed block.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from numpilot.gaps.installed_cann import (
    ACLNN_ADAPTER_CANDIDATES,
    AMBIGUOUS,
    ERROR,
    EXACT_CANDIDATE,
    INCOMPATIBLE,
    NO_MATCH,
    CATALOG_REVISION,
    CannInventoryError,
    InstalledCannInventory,
    canonical_target_soc,
    inventory_installed_cann,
    resolve_installed_aclnn_from_inventory,
)
from numpilot.gaps.models import GapSpec
from numpilot.gaps.official_source import (
    SOURCE_SEARCH_CONTRACT_REVISION,
    OpsNnSourceResolver,
)
from numpilot.gaps.pipeline import GapPreparation
from numpilot.gaps.spec import is_custom_generation_operator_gap


OFFICIAL_RESOLUTION_SCHEMA_VERSION = 1
INSTALLED_ADAPTER_ROUTE = "installed_cann_adapter_candidate"
OFFICIAL_SOURCE_ROUTE = "official_source_candidate"
NEW_KERNEL_ROUTE = "new_kernel_generation_allowed"
# Explicit escape hatch for a declared domain OperatorSpec whose API is not in
# the reviewed official catalog.  This is deliberately distinct from
# ``NEW_KERNEL_ROUTE``: the latter means two audited official providers proved
# a clean miss, while this route records that official equivalence is *not*
# proven and the user explicitly opted into an unverified LLM draft.
CUSTOM_OPERATOR_ROUTE = "custom_operator_generation_allowed"
# Alias kept for callers that describe the route as a generation candidate.
CUSTOM_GENERATION_ROUTE = CUSTOM_OPERATOR_ROUTE
BLOCKED_ROUTE = "blocked"

_ROUTES = frozenset({
    INSTALLED_ADAPTER_ROUTE,
    OFFICIAL_SOURCE_ROUTE,
    NEW_KERNEL_ROUTE,
    CUSTOM_OPERATOR_ROUTE,
    BLOCKED_ROUTE,
})
_PROVIDER_STATUSES = frozenset({
    EXACT_CANDIDATE,
    NO_MATCH,
    INCOMPATIBLE,
    AMBIGUOUS,
    ERROR,
})
_HEX_SHA256 = frozenset("0123456789abcdef")
_HEX_SHA1 = frozenset("0123456789abcdef")
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MAX_JSON_INTEGER_DIGITS = 64


class OfficialResolutionError(RuntimeError):
    """Official-operator discovery or receipt validation failed safely."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_SHA256 for character in value)
    )


def _is_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in _HEX_SHA1 for character in value)
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def gap_spec_sha256(spec: GapSpec) -> str:
    """Bind a route decision to the complete canonical GapSpec."""

    if not isinstance(spec, GapSpec):
        raise TypeError("official resolution requires a GapSpec")
    return hashlib.sha256(_canonical_bytes(spec.to_dict())).hexdigest()


def _bounded_text(value: object, label: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 and character not in "\t\n\r" for character in value)
    ):
        raise OfficialResolutionError("invalid official-resolution %s" % label)
    return value


def _provider_payload(value: object, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialResolutionError("%s result must be an object" % label)
    if any(not isinstance(key, str) for key in value):
        raise OfficialResolutionError("%s result has a non-string key" % label)
    payload = dict(value)
    status = payload.get("status")
    if status not in _PROVIDER_STATUSES:
        raise OfficialResolutionError("%s result has an invalid status" % label)
    # Keep provider evidence JSON-compatible and bounded before it enters a
    # receipt.  Round-tripping also rejects NaN and non-string object keys.
    try:
        encoded = _canonical_bytes(payload)
        decoded = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise OfficialResolutionError(
            "%s result is not canonical JSON" % label
        ) from exc
    if len(encoded) > _MAX_REPORT_BYTES // 2:
        raise OfficialResolutionError("%s result is too large" % label)
    return decoded


def _source_has_candidates(payload: Mapping[str, Any]) -> bool:
    candidates = payload.get("candidates")
    return isinstance(candidates, list) and bool(candidates) and all(
        isinstance(item, Mapping) for item in candidates
    )


def _custom_generation_proof(payload: Mapping[str, Any]) -> bool:
    """Validate the non-equivalence evidence attached to a custom route.

    The proof intentionally says that no official equivalence was established;
    it is not a synthetic ``no_match`` result.  Keeping this as a strict,
    bounded object lets receipt loading reject a hand-edited route even before
    the source-bound recheck runs.
    """

    proof = payload.get("custom_generation_proof")
    return (
        isinstance(proof, Mapping)
        and set(proof) == {
            "operator_spec_provenance",
            "operator_spec_route",
            "official_equivalence",
        }
        and proof.get("operator_spec_provenance")
        == "custom_generation_candidate"
        and proof.get("operator_spec_route")
        == "ascendc_registry_candidate"
        and proof.get("official_equivalence") == "not_proven"
    )


def _custom_route_evidence(
    installed: Mapping[str, Any],
    source: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether provider evidence is eligible for custom opt-in.

    ``catalog_unreviewed_api`` is an ambiguity, not a negative result.  A
    source candidate, an inventory error, or any other provider status must
    remain blocked/official-first and cannot be converted to this route.
    """

    if not (
        installed.get("status") == AMBIGUOUS
        and installed.get("reason_code") == "catalog_unreviewed_api"
        and installed.get("adapter_candidate") is False
        and installed.get("semantic_equivalence") == "unverified"
    ):
        return False
    if source is None:
        return True
    if source.get("status") in {EXACT_CANDIDATE, AMBIGUOUS}:
        # A positive-looking source result is allowed to win only when it has
        # an actual candidate list; an empty/malformed positive result is not
        # evidence for either route and must fail closed.
        return False
    if _source_clean_no_match(source):
        return True
    # ``_blocked_provider_payload`` is the only accepted non-negative source
    # fallback: it records why a local source checkout was not consulted while
    # keeping activation disabled and the candidate set empty.  Arbitrary
    # malformed payloads do not unlock custom generation.
    return (
        source.get("provider") == "ops_nn_source"
        and source.get("status") in {INCOMPATIBLE, ERROR}
        and source.get("candidates") == []
        and source.get("activation_allowed") is False
    )


def _soc_identity(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.casefold() if character.isalnum())


def _installed_clean_no_match(payload: Mapping[str, Any]) -> bool:
    """Accept only a catalog-covered, target-bound negative ACLNN result."""

    candidate = payload.get("candidate")
    required = payload.get("required_symbols")
    missing = payload.get("missing_symbols")
    audited_versions = (
        candidate.get("audited_cann_versions")
        if isinstance(candidate, Mapping) else None
    )
    audited_socs = (
        candidate.get("audited_target_socs")
        if isinstance(candidate, Mapping) else None
    )
    target = _soc_identity(payload.get("expected_target_soc"))
    reviewed_candidate = next(
        (
            item for item in ACLNN_ADAPTER_CANDIDATES
            if isinstance(candidate, Mapping)
            and item.candidate_id == candidate.get("candidate_id")
        ),
        None,
    )
    return (
        payload.get("schema_version") == 1
        and payload.get("status") == NO_MATCH
        and payload.get("reason_code") == "candidate_artifacts_absent"
        and payload.get("catalog_revision") == CATALOG_REVISION
        and payload.get("adapter_candidate") is False
        and payload.get("semantic_equivalence") == "unverified"
        and payload.get("provenance")
        == "installed_cann_toolkit_read_only_inventory"
        and isinstance(candidate, Mapping)
        and reviewed_candidate is not None
        and candidate == reviewed_candidate.to_dict()
        and candidate.get("qualified_api") == payload.get("qualified_api")
        and payload.get("candidate_mapping_sha256")
        == reviewed_candidate.mapping_sha256
        and isinstance(audited_versions, list)
        and payload.get("expected_cann_version") in audited_versions
        and isinstance(audited_socs, list)
        and bool(target)
        and target in {_soc_identity(item) for item in audited_socs}
        and isinstance(required, list)
        and bool(required)
        and all(isinstance(item, str) and item for item in required)
        and isinstance(missing, list)
        and sorted(missing) == sorted(required)
        and payload.get("selected_header") is None
        and payload.get("selected_library") is None
    )


def _source_clean_no_match(payload: Mapping[str, Any]) -> bool:
    """Accept only a pinned tree-wide reviewed-alias negative result."""

    aliases = payload.get("searched_aliases")
    requested = payload.get("requested")
    return (
        payload.get("schema_version") == 1
        and payload.get("provider") == "ops_nn_source"
        and payload.get("status") == NO_MATCH
        and payload.get("negative_search_complete") is True
        and payload.get("search_contract_revision")
        == SOURCE_SEARCH_CONTRACT_REVISION
        and isinstance(aliases, list)
        and bool(aliases)
        and all(isinstance(item, str) and item for item in aliases)
        and isinstance(requested, Mapping)
        and payload.get("candidates") == []
        and payload.get("selected_candidate_id") == ""
        and payload.get("requires_audit") is False
        and payload.get("activation_allowed") is False
        and isinstance(payload.get("source_ref"), str)
        and _is_sha1(payload.get("source_commit"))
        and _is_sha1(payload.get("source_tree"))
    )


@dataclass(frozen=True)
class OfficialGapResolution:
    """One GapSpec's official-first routing decision."""

    gap_id: str
    qualified_api: str
    gap_spec_sha256: str
    route: str
    reason: str
    installed_cann: Mapping[str, Any]
    ops_nn_source: Optional[Mapping[str, Any]]
    new_kernel_generation_allowed: bool
    adapter_generation_required: bool
    activation_allowed: bool = False

    def __post_init__(self) -> None:
        _bounded_text(self.gap_id, "gap_id", 128)
        _bounded_text(self.qualified_api, "qualified_api", 256)
        _bounded_text(self.reason, "decision reason", 1024)
        if not _is_sha256(self.gap_spec_sha256):
            raise OfficialResolutionError("invalid GapSpec digest")
        if self.route not in _ROUTES:
            raise OfficialResolutionError("invalid official-resolution route")
        if self.activation_allowed:
            raise OfficialResolutionError(
                "official discovery cannot activate a capability"
            )
        installed = _provider_payload(self.installed_cann, "installed CANN")
        source = (
            None
            if self.ops_nn_source is None
            else _provider_payload(self.ops_nn_source, "ops-nn source")
        )
        installed_status = installed["status"]
        source_status = source["status"] if source is not None else None
        source_candidate = (
            source is not None
            and source_status in {EXACT_CANDIDATE, AMBIGUOUS}
            and _source_has_candidates(source)
        )
        expected_route = BLOCKED_ROUTE
        if installed_status == EXACT_CANDIDATE and source is None:
            expected_route = INSTALLED_ADAPTER_ROUTE
        elif _installed_clean_no_match(installed) and source_candidate:
            expected_route = OFFICIAL_SOURCE_ROUTE
        elif _installed_clean_no_match(installed) and (
            source is not None and _source_clean_no_match(source)
        ):
            expected_route = NEW_KERNEL_ROUTE
        elif _custom_route_evidence(installed, source) and _custom_generation_proof(
            installed
        ):
            # This branch is intentionally after all official candidates and
            # clean misses.  ``catalog_unreviewed_api`` remains ambiguous; the
            # custom route merely records an explicit user opt-in and never
            # relabels that ambiguity as ``no_match``.
            expected_route = CUSTOM_OPERATOR_ROUTE
        if self.route == CUSTOM_OPERATOR_ROUTE and not self.qualified_api.startswith(
            "operator."
        ):
            raise OfficialResolutionError(
                "custom route requires an operator.* qualified API"
            )
        if self.route != expected_route:
            raise OfficialResolutionError(
                "official-resolution route contradicts provider evidence"
            )
        if self.new_kernel_generation_allowed != (
            self.route in {NEW_KERNEL_ROUTE, CUSTOM_OPERATOR_ROUTE}
        ):
            raise OfficialResolutionError(
                "new-kernel permission contradicts the selected route"
            )
        expected_adapter = self.route == INSTALLED_ADAPTER_ROUTE
        if self.adapter_generation_required != expected_adapter:
            raise OfficialResolutionError(
                "adapter requirement contradicts the selected route"
            )
        object.__setattr__(self, "installed_cann", installed)
        object.__setattr__(self, "ops_nn_source", source)

    @property
    def semantic_audit_required(self) -> bool:
        return self.route in {
            INSTALLED_ADAPTER_ROUTE,
            OFFICIAL_SOURCE_ROUTE,
            CUSTOM_OPERATOR_ROUTE,
        }

    @property
    def source_build_required(self) -> bool:
        return self.route == OFFICIAL_SOURCE_ROUTE

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "gap_id": self.gap_id,
            "qualified_api": self.qualified_api,
            "gap_spec_sha256": self.gap_spec_sha256,
            "route": self.route,
            "reason": self.reason,
            "installed_cann": dict(self.installed_cann),
            "ops_nn_source": (
                dict(self.ops_nn_source)
                if self.ops_nn_source is not None else None
            ),
            "new_kernel_generation_allowed": self.new_kernel_generation_allowed,
            "adapter_generation_required": self.adapter_generation_required,
            "adapter_generation_allowed": False,
            "semantic_audit_required": self.semantic_audit_required,
            "source_build_required": self.source_build_required,
            "activation_allowed": False,
        }
        if self.route == CUSTOM_OPERATOR_ROUTE:
            # Explicitly state that this is an unreviewed custom route.  This
            # is evidence of *non-equivalence*, never a fabricated official
            # negative result.
            result["custom_generation_proof"] = {
                "operator_spec_provenance": "custom_generation_candidate",
                "operator_spec_route": "ascendc_registry_candidate",
                "official_equivalence": "not_proven",
            }
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialGapResolution":
        expected = {
            "gap_id",
            "qualified_api",
            "gap_spec_sha256",
            "route",
            "reason",
            "installed_cann",
            "ops_nn_source",
            "new_kernel_generation_allowed",
            "adapter_generation_required",
            "adapter_generation_allowed",
            "semantic_audit_required",
            "source_build_required",
            "activation_allowed",
        }
        if not isinstance(data, Mapping):
            raise OfficialResolutionError(
                "official gap decision fields are incomplete or unknown"
            )
        actual_fields = set(data)
        if actual_fields - expected - {"custom_generation_proof"}:
            raise OfficialResolutionError(
                "official gap decision fields are incomplete or unknown"
            )
        if "custom_generation_proof" in data:
            if data.get("route") != CUSTOM_OPERATOR_ROUTE or not _custom_generation_proof(
                data
            ):
                raise OfficialResolutionError(
                    "custom generation proof is invalid or misplaced"
                )
        elif data.get("route") == CUSTOM_OPERATOR_ROUTE:
            raise OfficialResolutionError(
                "custom route requires an explicit non-equivalence proof"
            )
        if actual_fields - {"custom_generation_proof"} != expected:
            raise OfficialResolutionError(
                "official gap decision fields are incomplete or unknown"
            )
        for name in (
            "new_kernel_generation_allowed",
            "adapter_generation_required",
            "adapter_generation_allowed",
            "semantic_audit_required",
            "source_build_required",
            "activation_allowed",
        ):
            if not isinstance(data[name], bool):
                raise OfficialResolutionError("decision boolean is invalid: %s" % name)
        if data["adapter_generation_allowed"] is not False:
            raise OfficialResolutionError("official discovery cannot allow adapter generation")
        for name in (
            "gap_id", "qualified_api", "gap_spec_sha256", "route", "reason",
        ):
            if not isinstance(data[name], str):
                raise OfficialResolutionError(
                    "decision text field is invalid: %s" % name
                )
        result = cls(
            gap_id=data["gap_id"],
            qualified_api=data["qualified_api"],
            gap_spec_sha256=data["gap_spec_sha256"],
            route=data["route"],
            reason=data["reason"],
            installed_cann=data["installed_cann"],
            ops_nn_source=data["ops_nn_source"],
            new_kernel_generation_allowed=data["new_kernel_generation_allowed"],
            adapter_generation_required=data["adapter_generation_required"],
            activation_allowed=data["activation_allowed"],
        )
        if (
            data["semantic_audit_required"] is not result.semantic_audit_required
            or data["source_build_required"] is not result.source_build_required
        ):
            raise OfficialResolutionError(
                "decision stage requirements contradict the selected route"
            )
        return result


@dataclass(frozen=True)
class OfficialOperatorResolutionReport:
    """Content-bound, local-only route receipt for a GapPreparation."""

    source_path: str
    source_sha256: str
    target_soc: str
    cann_version: str
    cann_root: str
    ops_nn_root: str
    cann_osl_accepted: bool
    status: str
    decisions: Tuple[OfficialGapResolution, ...]
    installed_inventory: Optional[Mapping[str, Any]] = None
    schema_version: int = OFFICIAL_RESOLUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != OFFICIAL_RESOLUTION_SCHEMA_VERSION
        ):
            raise OfficialResolutionError("unsupported official-resolution schema")
        _bounded_text(self.source_path, "source path", 4096)
        if not Path(self.source_path).is_absolute():
            raise OfficialResolutionError("resolution source path must be absolute")
        if not _is_sha256(self.source_sha256):
            raise OfficialResolutionError("resolution source digest is invalid")
        _bounded_text(self.target_soc, "target SoC", 128)
        _bounded_text(self.cann_version, "CANN version", 128)
        _bounded_text(self.cann_root, "CANN root", 4096)
        if not Path(self.cann_root).is_absolute():
            raise OfficialResolutionError("resolution CANN root must be absolute")
        if self.ops_nn_root:
            _bounded_text(self.ops_nn_root, "ops-nn root", 4096)
            if not Path(self.ops_nn_root).is_absolute():
                raise OfficialResolutionError("resolution ops-nn root must be absolute")
        if not isinstance(self.cann_osl_accepted, bool):
            raise OfficialResolutionError("license acceptance must be boolean")
        if self.status not in {
            "no_gaps",
            "official_candidate_found",
            "new_kernel_generation_allowed",
            "custom_operator_generation_allowed",
            "mixed_routes",
            "blocked",
        }:
            raise OfficialResolutionError("invalid resolution report status")
        if not isinstance(self.decisions, tuple) or any(
            not isinstance(item, OfficialGapResolution) for item in self.decisions
        ):
            raise OfficialResolutionError("official decisions have an invalid type")
        if len(self.decisions) > 32:
            raise OfficialResolutionError("too many official gap decisions")
        identities = [(item.gap_id, item.qualified_api) for item in self.decisions]
        if len(set(identities)) != len(identities):
            raise OfficialResolutionError("official gap decisions are duplicated")
        for item in self.decisions:
            installed = item.installed_cann
            if installed.get("qualified_api") not in {None, item.qualified_api}:
                raise OfficialResolutionError(
                    "installed evidence is bound to another API"
                )
            if installed.get("expected_cann_version") not in {
                None, self.cann_version,
            }:
                raise OfficialResolutionError(
                    "installed evidence is bound to another CANN version"
                )
            installed_soc = installed.get("expected_target_soc")
            if installed_soc is not None and (
                _soc_identity(installed_soc) != _soc_identity(self.target_soc)
            ):
                raise OfficialResolutionError(
                    "installed evidence is bound to another target SoC"
                )
            source = item.ops_nn_source
            if source is not None and isinstance(source.get("requested"), Mapping):
                requested = source["requested"]
                if (
                    requested.get("qualified_api") != item.qualified_api
                    or requested.get("cann_version") != self.cann_version
                    or _soc_identity(requested.get("target_soc"))
                    != _soc_identity(self.target_soc)
                ):
                    raise OfficialResolutionError(
                        "source evidence is bound to another lookup"
                    )
        if self.installed_inventory is not None:
            inventory = _provider_payload(
                {"status": NO_MATCH, "inventory": self.installed_inventory},
                "installed inventory",
            )["inventory"]
            if not isinstance(inventory, Mapping):
                raise OfficialResolutionError("installed inventory is invalid")
            object.__setattr__(self, "installed_inventory", inventory)
        expected_status = _aggregate_status(self.decisions)
        if self.status != expected_status:
            raise OfficialResolutionError(
                "resolution report status contradicts its decisions"
            )

    @property
    def model_kernel_generation_allowed(self) -> bool:
        return bool(self.decisions) and all(
            item.new_kernel_generation_allowed for item in self.decisions
        )

    def _body(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_type": "official_operator_resolution",
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
            "cann_root": self.cann_root,
            "ops_nn_root": self.ops_nn_root,
            "cann_osl_accepted": self.cann_osl_accepted,
            "status": self.status,
            "model_kernel_generation_allowed": self.model_kernel_generation_allowed,
            "activation_allowed": False,
            "network_access_performed": False,
            "system_write_performed": False,
            "installed_inventory": (
                dict(self.installed_inventory)
                if self.installed_inventory is not None else None
            ),
            "decisions": [item.to_dict() for item in self.decisions],
        }

    @property
    def resolution_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._body())).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        result = self._body()
        result["resolution_id"] = self.resolution_id
        return result

    def validate_preparation(self, preparation: GapPreparation) -> None:
        """Reject a stale or tampered receipt before any model access."""

        if not isinstance(preparation, GapPreparation):
            raise OfficialResolutionError("resolution requires a GapPreparation")
        source_path = str(Path(preparation.source_path).absolute())
        source_sha256 = hashlib.sha256(
            preparation.source.encode("utf-8")
        ).hexdigest()
        if (
            self.source_path != source_path
            or self.source_sha256 != source_sha256
        ):
            raise OfficialResolutionError(
                "official resolution is not bound to the current source"
            )
        expected = {
            (spec.gap_id, spec.qualified_api): gap_spec_sha256(spec)
            for spec in preparation.specs
        }
        actual = {
            (item.gap_id, item.qualified_api): item.gap_spec_sha256
            for item in self.decisions
        }
        if expected != actual:
            raise OfficialResolutionError(
                "official resolution is not bound to the current GapSpecs"
            )
        if any(
            spec.target_soc != self.target_soc
            or spec.cann_version != self.cann_version
            for spec in preparation.specs
        ):
            raise OfficialResolutionError(
                "official resolution target does not match the current GapSpecs"
            )
        specs_by_key = {
            (spec.gap_id, spec.qualified_api): spec
            for spec in preparation.specs
        }
        for decision in self.decisions:
            if decision.route != CUSTOM_OPERATOR_ROUTE:
                continue
            spec = specs_by_key.get((decision.gap_id, decision.qualified_api))
            if spec is None or not is_custom_generation_operator_gap(
                preparation.source, spec
            ):
                raise OfficialResolutionError(
                    "custom route is not bound to an explicit custom OperatorSpec"
                )
            if not _custom_generation_proof(decision.installed_cann):
                raise OfficialResolutionError(
                    "custom route is missing its non-equivalence proof"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialOperatorResolutionReport":
        expected = {
            "schema_version",
            "report_type",
            "source_path",
            "source_sha256",
            "target_soc",
            "cann_version",
            "cann_root",
            "ops_nn_root",
            "cann_osl_accepted",
            "status",
            "model_kernel_generation_allowed",
            "activation_allowed",
            "network_access_performed",
            "system_write_performed",
            "installed_inventory",
            "decisions",
            "resolution_id",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise OfficialResolutionError(
                "official-resolution report fields are incomplete or unknown"
            )
        if data.get("report_type") != "official_operator_resolution":
            raise OfficialResolutionError("official-resolution report type is invalid")
        for name, expected_value in (
            ("activation_allowed", False),
            ("network_access_performed", False),
            ("system_write_performed", False),
        ):
            if data.get(name) is not expected_value:
                raise OfficialResolutionError("unsafe report claim: %s" % name)
        if not isinstance(data.get("cann_osl_accepted"), bool):
            raise OfficialResolutionError("license acceptance must be boolean")
        if (
            not isinstance(data.get("schema_version"), int)
            or isinstance(data.get("schema_version"), bool)
        ):
            raise OfficialResolutionError("official-resolution schema is invalid")
        for name in (
            "source_path", "source_sha256", "target_soc", "cann_version",
            "cann_root", "ops_nn_root", "status", "resolution_id",
        ):
            if not isinstance(data.get(name), str):
                raise OfficialResolutionError(
                    "official-resolution text field is invalid: %s" % name
                )
        if not isinstance(data.get("model_kernel_generation_allowed"), bool):
            raise OfficialResolutionError("model permission must be boolean")
        if data.get("installed_inventory") is not None and not isinstance(
            data.get("installed_inventory"), Mapping
        ):
            raise OfficialResolutionError("installed inventory is invalid")
        raw_decisions = data.get("decisions")
        if not isinstance(raw_decisions, list):
            raise OfficialResolutionError("official decisions must be a list")
        report = cls(
            schema_version=data["schema_version"],
            source_path=data["source_path"],
            source_sha256=data["source_sha256"],
            target_soc=data["target_soc"],
            cann_version=data["cann_version"],
            cann_root=data["cann_root"],
            ops_nn_root=data["ops_nn_root"],
            cann_osl_accepted=data["cann_osl_accepted"],
            status=data["status"],
            decisions=tuple(
                OfficialGapResolution.from_dict(item) for item in raw_decisions
            ),
            installed_inventory=data.get("installed_inventory"),
        )
        if data.get("model_kernel_generation_allowed") is not (
            report.model_kernel_generation_allowed
        ):
            raise OfficialResolutionError(
                "model permission contradicts official gap decisions"
            )
        if data.get("resolution_id") != report.resolution_id:
            raise OfficialResolutionError(
                "official-resolution receipt digest does not match its content"
            )
        return report


def _aggregate_status(decisions: Sequence[OfficialGapResolution]) -> str:
    if not decisions:
        return "no_gaps"
    routes = {item.route for item in decisions}
    if BLOCKED_ROUTE in routes:
        return "blocked"
    if routes == {NEW_KERNEL_ROUTE}:
        return "new_kernel_generation_allowed"
    if routes == {CUSTOM_OPERATOR_ROUTE}:
        return "custom_operator_generation_allowed"
    if routes.issubset({INSTALLED_ADAPTER_ROUTE, OFFICIAL_SOURCE_ROUTE}):
        return "official_candidate_found"
    return "mixed_routes"


def _without_inventory(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], object]:
    result = dict(payload)
    inventory = result.pop("inventory", None)
    return result, inventory


def _blocked_provider_payload(status: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": "ops_nn_source",
        "status": status,
        "reason": reason,
        "candidates": [],
        "activation_allowed": False,
    }


InventoryLoader = Callable[[str], InstalledCannInventory]
InstalledResolver = Callable[..., Any]
SourceResolverFactory = Callable[[str], Any]


def resolve_official_operators(
    preparation: GapPreparation,
    *,
    cann_root: str,
    ops_nn_root: str = "",
    accept_cann_osl: bool = False,
    allow_custom_operator_generation: bool = False,
    inventory_loader: InventoryLoader = inventory_installed_cann,
    installed_resolver: InstalledResolver = resolve_installed_aclnn_from_inventory,
    source_resolver_factory: SourceResolverFactory = OpsNnSourceResolver,
) -> OfficialOperatorResolutionReport:
    """Resolve all prepared gaps with installed-CANN-first precedence.

    ``ops_nn_root`` is consulted only after an installed ``no_match`` (or an
    explicitly opted-in, catalog-unreviewed custom OperatorSpec) and only when
    ``accept_cann_osl`` is true.  Any inability to prove a clean miss is a
    block, except for that explicit custom route, which records
    ``official_equivalence=not_proven`` and remains draft-only.
    """

    if not isinstance(preparation, GapPreparation):
        raise OfficialResolutionError("official resolution requires local preparation")
    if not isinstance(cann_root, str) or not cann_root.strip():
        raise OfficialResolutionError("an explicit CANN root is required")
    if not isinstance(ops_nn_root, str):
        raise OfficialResolutionError("ops-nn root must be a path string")
    if not isinstance(accept_cann_osl, bool):
        raise OfficialResolutionError("license acceptance must be boolean")
    if not isinstance(allow_custom_operator_generation, bool):
        raise OfficialResolutionError(
            "custom operator generation opt-in must be boolean"
        )
    specs = tuple(preparation.specs)
    source_path = str(Path(preparation.source_path).absolute())
    source_digest = hashlib.sha256(preparation.source.encode("utf-8")).hexdigest()
    if not specs:
        return OfficialOperatorResolutionReport(
            source_path=source_path,
            source_sha256=source_digest,
            target_soc="not-applicable",
            cann_version="not-applicable",
            cann_root=str(Path(cann_root).absolute()),
            ops_nn_root=(str(Path(ops_nn_root).absolute()) if ops_nn_root else ""),
            cann_osl_accepted=accept_cann_osl,
            status="no_gaps",
            decisions=(),
        )
    target_soc = specs[0].target_soc
    cann_version = specs[0].cann_version
    if any(
        spec.target_soc != target_soc or spec.cann_version != cann_version
        for spec in specs
    ):
        raise OfficialResolutionError("GapSpecs do not share one target identity")

    # A family label such as ``Ascend910B`` cannot prove the exact B1/B2/B3/B4
    # hardware identity.  Fail before reading even the local toolkit: negative
    # artifact evidence is meaningful only for a catalog-audited exact SoC.
    canonical_soc = canonical_target_soc(target_soc)
    audited_socs = {
        item
        for candidate in ACLNN_ADAPTER_CANDIDATES
        for item in candidate.audited_target_socs
    }
    if canonical_soc not in audited_socs:
        family = canonical_soc == "ascend910b"
        reason_code = (
            "target_soc_family_not_exact"
            if family else "target_soc_not_audited"
        )
        reason = (
            "target SoC is a family label; provide an exact identity such as Ascend910B4"
            if family else
            "target SoC is not an exact identity covered by the installed ACLNN catalog"
        )
        decisions = tuple(
            OfficialGapResolution(
                gap_id=spec.gap_id,
                qualified_api=spec.qualified_api,
                gap_spec_sha256=gap_spec_sha256(spec),
                route=BLOCKED_ROUTE,
                reason=reason,
                installed_cann={
                    "schema_version": 1,
                    "status": AMBIGUOUS,
                    "qualified_api": spec.qualified_api,
                    "expected_cann_version": spec.cann_version,
                    "expected_target_soc": spec.target_soc,
                    "reason_code": reason_code,
                    "reason": reason,
                    "adapter_candidate": False,
                    "semantic_equivalence": "unverified",
                    "catalog_revision": CATALOG_REVISION,
                },
                ops_nn_source=None,
                new_kernel_generation_allowed=False,
                adapter_generation_required=False,
            )
            for spec in specs
        )
        return OfficialOperatorResolutionReport(
            source_path=source_path,
            source_sha256=source_digest,
            target_soc=target_soc,
            cann_version=cann_version,
            cann_root=str(Path(cann_root).absolute()),
            ops_nn_root=(
                str(Path(ops_nn_root).absolute()) if ops_nn_root else ""
            ),
            cann_osl_accepted=accept_cann_osl,
            status="blocked",
            decisions=decisions,
            installed_inventory=None,
        )

    inventory: Optional[InstalledCannInventory]
    inventory_error: Optional[str] = None
    try:
        inventory = inventory_loader(cann_root)
        if not isinstance(inventory, InstalledCannInventory):
            raise OfficialResolutionError("CANN inventory loader returned a wrong type")
    except (CannInventoryError, OSError, ValueError, OfficialResolutionError) as exc:
        inventory = None
        inventory_error = "installed CANN inventory failed safely: %s" % type(exc).__name__

    source_resolver = None
    decisions = []
    for spec in specs:
        custom_opt_in = allow_custom_operator_generation and (
            is_custom_generation_operator_gap(preparation.source, spec)
        )
        if inventory is None:
            installed_payload = {
                "schema_version": 1,
                "status": ERROR,
                "qualified_api": spec.qualified_api,
                "expected_cann_version": spec.cann_version,
                "reason_code": "cann_inventory_error",
                "reason": inventory_error,
                "adapter_candidate": False,
                "semantic_equivalence": "unverified",
            }
            decisions.append(OfficialGapResolution(
                gap_id=spec.gap_id,
                qualified_api=spec.qualified_api,
                gap_spec_sha256=gap_spec_sha256(spec),
                route=BLOCKED_ROUTE,
                reason=inventory_error or "installed CANN inventory failed",
                installed_cann=installed_payload,
                ops_nn_source=None,
                new_kernel_generation_allowed=False,
                adapter_generation_required=False,
            ))
            continue
        try:
            installed_result = installed_resolver(
                spec.qualified_api,
                inventory=inventory,
                expected_cann_version=spec.cann_version,
                expected_target_soc=spec.target_soc,
            )
            installed_payload, result_inventory = _without_inventory(
                installed_result.to_dict()
            )
            if result_inventory is not None and result_inventory != inventory.to_dict():
                raise OfficialResolutionError(
                    "installed resolver returned another inventory"
                )
        except (AttributeError, TypeError, ValueError, OfficialResolutionError) as exc:
            installed_payload = {
                "schema_version": 1,
                "status": ERROR,
                "qualified_api": spec.qualified_api,
                "expected_cann_version": spec.cann_version,
                "reason_code": "installed_resolver_error",
                "reason": "installed resolver failed safely: %s" % type(exc).__name__,
                "adapter_candidate": False,
                "semantic_equivalence": "unverified",
            }
        installed_status = installed_payload.get("status")
        if installed_status == EXACT_CANDIDATE:
            decisions.append(OfficialGapResolution(
                gap_id=spec.gap_id,
                qualified_api=spec.qualified_api,
                gap_spec_sha256=gap_spec_sha256(spec),
                route=INSTALLED_ADAPTER_ROUTE,
                reason=(
                    "installed ACLNN artifacts match an audited adapter mapping; "
                    "semantic qualification and configured-model adapter generation remain required"
                ),
                installed_cann=installed_payload,
                ops_nn_source=None,
                new_kernel_generation_allowed=False,
                adapter_generation_required=True,
            ))
            continue
        # An unreviewed ``operator.*`` API may enter the explicit custom lane,
        # but only after its identity was proven from the source AST.  All
        # ordinary gaps and all other ambiguous/error statuses retain the
        # historical fail-closed branch below.
        custom_unreviewed = custom_opt_in and (
            installed_payload.get("status") == AMBIGUOUS
            and installed_payload.get("reason_code") == "catalog_unreviewed_api"
        )
        if not _installed_clean_no_match(installed_payload) and not custom_unreviewed:
            decisions.append(OfficialGapResolution(
                gap_id=spec.gap_id,
                qualified_api=spec.qualified_api,
                gap_spec_sha256=gap_spec_sha256(spec),
                route=BLOCKED_ROUTE,
                reason="installed CANN lookup did not produce a clean miss",
                installed_cann=installed_payload,
                ops_nn_source=None,
                new_kernel_generation_allowed=False,
                adapter_generation_required=False,
            ))
            continue

        if not accept_cann_osl:
            source_payload = _blocked_provider_payload(
                INCOMPATIBLE,
                "CANN OSL v2 acceptance is required before ops-nn source use",
            )
        elif not ops_nn_root:
            source_payload = _blocked_provider_payload(
                ERROR,
                "an explicit local ops-nn checkout is required after an installed miss",
            )
        else:
            try:
                if source_resolver is None:
                    source_resolver = source_resolver_factory(ops_nn_root)
                source_result = source_resolver.resolve(
                    cann_version=spec.cann_version,
                    target_soc=spec.target_soc,
                    op_name=spec.api_name,
                    qualified_api=spec.qualified_api,
                )
                source_payload = source_result.to_dict()
            except (
                AttributeError, OSError, RuntimeError, TypeError, ValueError
            ) as exc:
                source_payload = _blocked_provider_payload(
                    ERROR,
                    "ops-nn source resolver failed safely: %s" % type(exc).__name__,
                )
        source_status = source_payload.get("status")
        custom_route_allowed = custom_unreviewed and _custom_route_evidence(
            installed_payload, source_payload
        )
        if _source_clean_no_match(source_payload):
            if custom_route_allowed:
                # Keep the ambiguity visible.  A clean source miss does not
                # turn the unreviewed installed catalog into a clean miss.
                route = CUSTOM_OPERATOR_ROUTE
                reason = (
                    "explicit custom OperatorSpec opt-in; installed ACLNN "
                    "catalog is unreviewed for this operator, so official "
                    "equivalence is not proven"
                )
                installed_payload = dict(installed_payload)
                installed_payload["custom_generation_proof"] = {
                    "operator_spec_provenance": "custom_generation_candidate",
                    "operator_spec_route": "ascendc_registry_candidate",
                    "official_equivalence": "not_proven",
                }
                allow_kernel = True
                needs_adapter = False
            else:
                route = NEW_KERNEL_ROUTE
                reason = (
                    "installed CANN and the version-pinned official source both "
                    "returned a clean no_match"
                )
                allow_kernel = True
                needs_adapter = False
        elif (
            source_status in {EXACT_CANDIDATE, AMBIGUOUS}
            and _source_has_candidates(source_payload)
        ):
            if _installed_clean_no_match(installed_payload):
                route = OFFICIAL_SOURCE_ROUTE
                reason = (
                    "version-pinned official source contains a candidate; semantic "
                    "audit and isolated build must pass before adapter generation can be planned"
                )
                allow_kernel = False
                needs_adapter = False
            else:
                # A positive official source candidate must never be hidden by
                # the custom opt-in.  Keep the result blocked until a human
                # resolves the official candidate/installed ambiguity.
                route = BLOCKED_ROUTE
                reason = (
                    "official source contains a candidate while installed "
                    "ACLNN coverage is unreviewed; custom generation is not "
                    "allowed to override an official candidate"
                )
                allow_kernel = False
                needs_adapter = False
        else:
            if custom_route_allowed:
                # No source result (for example no accepted local checkout) is
                # also not a clean negative.  The explicit custom flag permits
                # a *draft-only* experiment while the receipt records that
                # official equivalence remains unproven.
                route = CUSTOM_OPERATOR_ROUTE
                reason = (
                    "explicit custom OperatorSpec opt-in; official catalog "
                    "lookup is unreviewed and no official equivalence proof "
                    "was available"
                )
                installed_payload = dict(installed_payload)
                installed_payload["custom_generation_proof"] = {
                    "operator_spec_provenance": "custom_generation_candidate",
                    "operator_spec_route": "ascendc_registry_candidate",
                    "official_equivalence": "not_proven",
                }
                allow_kernel = True
                needs_adapter = False
            else:
                route = BLOCKED_ROUTE
                reason = "official source lookup did not produce a clean miss or usable candidate"
                allow_kernel = False
                needs_adapter = False
        decisions.append(OfficialGapResolution(
            gap_id=spec.gap_id,
            qualified_api=spec.qualified_api,
            gap_spec_sha256=gap_spec_sha256(spec),
            route=route,
            reason=reason,
            installed_cann=installed_payload,
            ops_nn_source=source_payload,
            new_kernel_generation_allowed=allow_kernel,
            adapter_generation_required=needs_adapter,
        ))

    inventory_payload = inventory.to_dict() if inventory is not None else None
    return OfficialOperatorResolutionReport(
        source_path=source_path,
        source_sha256=source_digest,
        target_soc=target_soc,
        cann_version=cann_version,
        cann_root=(inventory.root if inventory is not None else str(Path(cann_root).absolute())),
        ops_nn_root=(str(Path(ops_nn_root).absolute()) if ops_nn_root else ""),
        cann_osl_accepted=accept_cann_osl,
        status=_aggregate_status(decisions),
        decisions=tuple(decisions),
        installed_inventory=inventory_payload,
    )


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OfficialResolutionError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_constant(value):
    raise OfficialResolutionError("non-standard JSON constant: %s" % value)


def _bounded_parse_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if not digits or len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise ValueError("JSON integer exceeds the receipt limit")
    return int(value, 10)


def _open_receipt_parent(path: str, *, create: bool) -> Tuple[int, str]:
    """Open a receipt parent by directory fd without following symlinks."""

    if (
        not isinstance(path, str)
        or not path
        or len(path) > 4096
        or "\x00" in path
    ):
        raise OfficialResolutionError("resolution report path is invalid")
    target = Path(path)
    name = target.name
    if not name or name in {".", ".."}:
        raise OfficialResolutionError("resolution report path has no safe filename")
    parts = target.parts
    if target.is_absolute():
        start = target.anchor
        ancestors = parts[1:-1]
    else:
        start = "."
        ancestors = parts[:-1]
    if any(part in {"", ".", ".."} for part in ancestors):
        raise OfficialResolutionError(
            "resolution report path contains an unsafe ancestor"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not nofollow or not directory:
        raise OfficialResolutionError(
            "secure resolution report path traversal is unavailable"
        )
    flags = os.O_RDONLY | directory | nofollow | cloexec
    current = -1
    try:
        current = os.open(start, flags)
        for part in ancestors:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            try:
                following = os.open(part, flags, dir_fd=current)
            except OSError as exc:
                raise OfficialResolutionError(
                    "resolution report path contains an unsafe ancestor"
                ) from exc
            os.close(current)
            current = following
        return current, name
    except Exception:
        if current >= 0:
            os.close(current)
        raise


def _read_receipt_descriptor(descriptor: int) -> Tuple[bytes, os.stat_result]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise OfficialResolutionError("resolution report must be a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise OfficialResolutionError(
            "resolution report permissions must be exactly 0600"
        )
    if metadata.st_size < 2 or metadata.st_size > _MAX_REPORT_BYTES:
        raise OfficialResolutionError("resolution report size is invalid")
    chunks = []
    remaining = _MAX_REPORT_BYTES + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > _MAX_REPORT_BYTES:
        raise OfficialResolutionError("resolution report exceeds the size limit")
    after = os.fstat(descriptor)
    if (
        metadata.st_dev != after.st_dev
        or metadata.st_ino != after.st_ino
        or metadata.st_size != after.st_size
        or metadata.st_mtime_ns != after.st_mtime_ns
        or metadata.st_ctime_ns != after.st_ctime_ns
    ):
        raise OfficialResolutionError(
            "resolution report changed while it was being read"
        )
    return payload, metadata


def _open_existing_receipt(parent: int, name: str) -> int:
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(name, flags, dir_fd=parent)
    except (OSError, TypeError, ValueError) as exc:
        raise OfficialResolutionError(
            "refusing an existing non-regular resolution report"
        ) from exc


def load_official_resolution_report(path: str) -> OfficialOperatorResolutionReport:
    """Load a bounded, regular, digest-verified route receipt."""

    parent = -1
    descriptor = -1
    try:
        parent, name = _open_receipt_parent(path, create=False)
        descriptor = _open_existing_receipt(parent, name)
        payload, metadata = _read_receipt_descriptor(descriptor)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise OfficialResolutionError(
                "resolution report changed while it was being read"
            )
    except OfficialResolutionError:
        raise
    except OSError as exc:
        raise OfficialResolutionError(
            "cannot read official-resolution report: %s" % type(exc).__name__
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent >= 0:
            os.close(parent)
    try:
        data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_int=_bounded_parse_int,
        )
    except OfficialResolutionError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise OfficialResolutionError(
            "resolution report is not bounded strict UTF-8 JSON"
        ) from exc
    try:
        return OfficialOperatorResolutionReport.from_dict(data)
    except OfficialResolutionError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise OfficialResolutionError(
            "official-resolution report has invalid field types"
        ) from exc


def write_official_resolution_report(
    report: OfficialOperatorResolutionReport,
    path: str,
    *,
    replace: bool = False,
) -> None:
    """Publish a private receipt through a symlink-safe directory descriptor."""

    if not isinstance(report, OfficialOperatorResolutionReport):
        raise OfficialResolutionError("cannot write an invalid resolution report")
    payload = (
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_REPORT_BYTES:
        raise OfficialResolutionError("resolution report exceeds the size limit")
    parent = -1
    descriptor = -1
    temporary = ""
    try:
        parent, name = _open_receipt_parent(path, create=True)
        for _attempt in range(128):
            temporary = ".%s-%s.tmp" % (name, secrets.token_hex(16))
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                descriptor = os.open(
                    temporary,
                    flags,
                    0o600,
                    dir_fd=parent,
                )
                break
            except FileExistsError:
                temporary = ""
        else:
            raise OfficialResolutionError(
                "cannot allocate a private resolution report temporary file"
            )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if replace:
                try:
                    existing = os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    existing = None
                if existing is not None and not stat.S_ISREG(existing.st_mode):
                    raise OfficialResolutionError(
                        "refusing to replace a non-regular resolution report"
                    )
                if (
                    existing is not None
                    and stat.S_IMODE(existing.st_mode) != 0o600
                ):
                    raise OfficialResolutionError(
                        "resolution report permissions must be exactly 0600"
                    )
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                temporary = ""
            else:
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing_descriptor = _open_existing_receipt(parent, name)
                    try:
                        existing_payload, existing = _read_receipt_descriptor(
                            existing_descriptor
                        )
                        current = os.stat(
                            name,
                            dir_fd=parent,
                            follow_symlinks=False,
                        )
                    finally:
                        os.close(existing_descriptor)
                    if (current.st_dev, current.st_ino) != (
                        existing.st_dev,
                        existing.st_ino,
                    ):
                        raise OfficialResolutionError(
                            "existing resolution report changed while inspected"
                        )
                    if existing_payload != payload:
                        raise OfficialResolutionError(
                            "refusing to overwrite an existing resolution report"
                        )
                    return
                os.unlink(temporary, dir_fd=parent)
                temporary = ""
            os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except OSError:
                    pass
    except OfficialResolutionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise OfficialResolutionError(
            "cannot write official-resolution report: %s" % type(exc).__name__
        ) from exc
    finally:
        if parent >= 0:
            os.close(parent)


__all__ = [
    "BLOCKED_ROUTE",
    "INSTALLED_ADAPTER_ROUTE",
    "NEW_KERNEL_ROUTE",
    "OFFICIAL_SOURCE_ROUTE",
    "OfficialGapResolution",
    "OfficialOperatorResolutionReport",
    "OfficialResolutionError",
    "gap_spec_sha256",
    "load_official_resolution_report",
    "resolve_official_operators",
    "write_official_resolution_report",
]
