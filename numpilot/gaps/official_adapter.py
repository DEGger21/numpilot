"""Prepare fail-closed model inputs for an installed official ACLNN candidate.

This module is the fail-closed bridge between official-first resolution and a
reviewed route-aware AsNumPy adapter planner.  It validates an ``official_binary``
``OperatorCandidateArtifact`` against the canonical ``GapSpec`` and matching
``OfficialGapResolution``, then emits a deterministic manifest and adapter
contract.  It does not inspect AsNumPy, invoke a model, build code, write an
integration tree, import AsNumPy, or activate a capability.

An installed ACLNN candidate needs an existing-module integration plan rather
than a custom OPP.  The prepared manifest therefore keeps
``model_invocation_allowed`` false until the separately inspected, hash-bound
context plan is attached.  Attaching that plan opens only the model gate; it
does not build, import, execute, integrate, qualify, or activate anything.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Dict, Mapping, Sequence, Tuple

from numpilot.asnumpy_adapter_context import (
    AsnumpyAdapterContextPlan,
    EXISTING_MODULE_MODE,
    OFFICIAL_ACLNN_ARCHITECTURE,
)

from numpilot.gaps.installed_cann import (
    ACLNN_ADAPTER_CANDIDATES,
    CATALOG_REVISION,
    EXACT_CANDIDATE,
    canonical_target_soc,
)
from numpilot.gaps.models import GapSpec
from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    OfficialGapResolution,
    OfficialOperatorResolutionReport,
    gap_spec_sha256,
)
from numpilot.gaps.recovery import (
    OperatorCandidateArtifact,
    OperatorCandidateOrigin,
    operator_candidate_sha256,
    validate_operator_candidate_binding,
)


OFFICIAL_ADAPTER_INPUT_SCHEMA_VERSION = 1
OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY = (
    "official_aclnn_existing_module"
)

OFFICIAL_CANDIDATE_MAPPING_REF = "official_candidate_mapping"
OFFICIAL_SELECTED_HEADER_REF = "official_selected_header"
OFFICIAL_SELECTED_LIBRARY_REF = "official_selected_library"
OFFICIAL_REQUIRED_SYMBOLS_REF = "official_required_symbols"
OFFICIAL_SEMANTIC_REVIEW_REF = "official_semantic_review"

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_CONTRACT_BYTES = 512 * 1024
_MAX_PATH_CHARS = 4096
_SHA256_ZERO = "0" * 64
_INSTALLED_FIELDS = frozenset({
    "schema_version",
    "status",
    "qualified_api",
    "expected_cann_version",
    "expected_target_soc",
    "reason_code",
    "reason",
    "adapter_candidate",
    "semantic_equivalence",
    "required_next_gate",
    "catalog_revision",
    "candidate",
    "candidate_mapping_sha256",
    "selected_header",
    "selected_library",
    "required_symbols",
    "missing_symbols",
    "alternatives",
    "provenance",
})
_HEADER_FIELDS = frozenset({
    "relative_path", "sha256", "size", "declared_symbols",
})
_LIBRARY_FIELDS = frozenset({
    "relative_path",
    "sha256",
    "size",
    "probe_tool",
    "aclnn_symbol_count",
    "aclnn_symbols_sha256",
})


class OfficialAdapterInputError(RuntimeError):
    """Official candidate evidence cannot safely drive adapter planning."""


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
        raise OfficialAdapterInputError(
            "official adapter value is not canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_object(
    value: object,
    fields: Sequence[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise OfficialAdapterInputError("%s must be an object" % label)
    expected = frozenset(fields)
    actual = frozenset(value)
    if actual != expected:
        raise OfficialAdapterInputError(
            "%s fields are incomplete or unknown" % label
        )
    return value


def _text(value: object, label: str, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise OfficialAdapterInputError("invalid %s" % label)
    return value


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == _SHA256_ZERO
    ):
        raise OfficialAdapterInputError("invalid %s SHA-256" % label)
    return value


def _exact_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OfficialAdapterInputError("invalid %s" % label)
    return value


def _string_list(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = 64,
) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) < minimum
        or len(value) > maximum
    ):
        raise OfficialAdapterInputError("invalid %s" % label)
    result = tuple(_text(item, label, 512) for item in value)
    if len(set(result)) != len(result):
        raise OfficialAdapterInputError("duplicated %s" % label)
    return result


def _relative_path(value: object, label: str) -> str:
    path_value = _text(value, label, 512)
    if "\\" in path_value:
        raise OfficialAdapterInputError("invalid %s" % label)
    path = PurePosixPath(path_value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != path_value
    ):
        raise OfficialAdapterInputError("unsafe %s" % label)
    return path_value


def _strict_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OfficialAdapterInputError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise OfficialAdapterInputError("non-standard JSON constant: %s" % value)


def _absolute_input_path(value: os.PathLike[str] | str, label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise OfficialAdapterInputError("invalid %s path" % label) from exc
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw) > _MAX_PATH_CHARS
        or "\x00" in raw
    ):
        raise OfficialAdapterInputError("invalid %s path" % label)
    try:
        return Path(os.path.abspath(raw))
    except (OSError, ValueError) as exc:
        raise OfficialAdapterInputError("invalid %s path" % label) from exc


def _read_regular_bounded_json(
    value: os.PathLike[str] | str,
    label: str,
    maximum: int = _MAX_JSON_BYTES,
) -> Mapping[str, Any]:
    """Read JSON through no-follow directory FDs and verify one stable inode."""

    path = _absolute_input_path(value, label)
    parts = path.parts
    if len(parts) < 2 or not path.name:
        raise OfficialAdapterInputError("invalid %s path" % label)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_fd = -1
    descriptor = -1
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for part in parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(path.name, file_flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OfficialAdapterInputError("%s must be a regular file" % label)
        if before.st_size <= 0 or before.st_size > maximum:
            raise OfficialAdapterInputError("%s exceeds its size bounds" % label)
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            len(payload) != before.st_size
            or len(payload) > maximum
            or identity_before != identity_after
        ):
            raise OfficialAdapterInputError("%s changed while being read" % label)
        os.lseek(descriptor, 0, os.SEEK_SET)
        verification = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            verification.append(chunk)
            remaining -= len(chunk)
        if b"".join(verification) != payload:
            raise OfficialAdapterInputError("%s changed while being read" % label)
    except OfficialAdapterInputError:
        raise
    except OSError as exc:
        raise OfficialAdapterInputError("cannot safely read %s" % label) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except OfficialAdapterInputError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OfficialAdapterInputError("%s is not strict JSON" % label) from exc
    if not isinstance(parsed, dict):
        raise OfficialAdapterInputError("%s must contain a JSON object" % label)
    return parsed


def _canonical_gap_spec(value: Mapping[str, Any]) -> GapSpec:
    try:
        spec = GapSpec.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise OfficialAdapterInputError("invalid canonical GapSpec") from exc
    if _canonical_bytes(spec.to_dict()) != _canonical_bytes(dict(value)):
        raise OfficialAdapterInputError("GapSpec is not in canonical schema form")
    return spec


def _canonical_candidate(value: Mapping[str, Any]) -> OperatorCandidateArtifact:
    try:
        candidate = OperatorCandidateArtifact.from_dict(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise OfficialAdapterInputError("invalid operator candidate") from exc
    if _canonical_bytes(candidate.to_dict()) != _canonical_bytes(dict(value)):
        raise OfficialAdapterInputError(
            "operator candidate is not in canonical schema form"
        )
    return candidate


def _canonical_resolution(value: Mapping[str, Any]) -> OfficialGapResolution:
    try:
        resolution = OfficialGapResolution.from_dict(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise OfficialAdapterInputError("invalid official gap resolution") from exc
    if _canonical_bytes(resolution.to_dict()) != _canonical_bytes(dict(value)):
        raise OfficialAdapterInputError(
            "official gap resolution is not in canonical schema form"
        )
    return resolution


def _selected_resolution(
    value: Mapping[str, Any], gap_id: str
) -> OfficialGapResolution:
    if value.get("report_type") == "official_operator_resolution":
        try:
            report = OfficialOperatorResolutionReport.from_dict(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise OfficialAdapterInputError(
                "invalid official resolution report"
            ) from exc
        matches = tuple(item for item in report.decisions if item.gap_id == gap_id)
        if len(matches) != 1:
            raise OfficialAdapterInputError(
                "official resolution report does not contain one matching decision"
            )
        return matches[0]
    return _canonical_resolution(value)


def _validate_installed_resolution(
    candidate: OperatorCandidateArtifact,
    spec: GapSpec,
    resolution: OfficialGapResolution,
) -> Dict[str, Any]:
    if candidate.origin_kind is not OperatorCandidateOrigin.OFFICIAL_BINARY:
        raise OfficialAdapterInputError(
            "official adapter input requires an official_binary candidate"
        )
    if (
        resolution.route != INSTALLED_ADAPTER_ROUTE
        or resolution.new_kernel_generation_allowed is not False
        or resolution.adapter_generation_required is not True
        or resolution.activation_allowed is not False
        or resolution.ops_nn_source is not None
    ):
        raise OfficialAdapterInputError(
            "resolution does not authorize an installed ACLNN adapter strategy"
        )
    if resolution.gap_spec_sha256 != gap_spec_sha256(spec):
        raise OfficialAdapterInputError("resolution GapSpec binding hash drift")
    try:
        validate_operator_candidate_binding(
            candidate,
            gap_spec=spec.to_dict(),
            resolution_decision=resolution.to_dict(),
        )
    except RuntimeError as exc:
        raise OfficialAdapterInputError(
            "operator candidate source/route binding drift"
        ) from exc

    installed = _strict_object(
        resolution.installed_cann, _INSTALLED_FIELDS, "installed ACLNN decision"
    )
    if (
        installed.get("schema_version") != 1
        or installed.get("status") != EXACT_CANDIDATE
        or installed.get("adapter_candidate") is not True
        or installed.get("semantic_equivalence") != "unverified"
        or installed.get("required_next_gate")
        != "generate_and_qualify_asnumpy_adapter"
        or installed.get("catalog_revision") != CATALOG_REVISION
        or installed.get("provenance")
        != "installed_cann_toolkit_read_only_inventory"
        or installed.get("reason_code") != "header_and_dynamic_symbols_present"
        or installed.get("qualified_api") != candidate.qualified_api
        or installed.get("expected_cann_version") != candidate.cann_version
        or installed.get("expected_target_soc") != candidate.target_soc
        or installed.get("missing_symbols") != []
    ):
        raise OfficialAdapterInputError("installed ACLNN decision is inconsistent")
    _text(installed.get("reason"), "installed ACLNN reason", 1024)
    _string_list(installed.get("alternatives"), "installed alternatives")

    raw_mapping = installed.get("candidate")
    if not isinstance(raw_mapping, Mapping):
        raise OfficialAdapterInputError("installed ACLNN candidate mapping is absent")
    catalog_matches = tuple(
        item
        for item in ACLNN_ADAPTER_CANDIDATES
        if item.candidate_id == raw_mapping.get("candidate_id")
    )
    if len(catalog_matches) != 1:
        raise OfficialAdapterInputError("installed candidate is not in the audit catalog")
    catalog = catalog_matches[0]
    if dict(raw_mapping) != catalog.to_dict():
        raise OfficialAdapterInputError("installed candidate mapping drift")
    mapping_sha256 = _digest(
        installed.get("candidate_mapping_sha256"), "candidate mapping"
    )
    if mapping_sha256 != catalog.mapping_sha256:
        raise OfficialAdapterInputError("installed candidate mapping hash drift")
    if (
        catalog.qualified_api != candidate.qualified_api
        or candidate.cann_version not in catalog.audited_cann_versions
        or canonical_target_soc(candidate.target_soc)
        not in catalog.audited_target_socs
    ):
        raise OfficialAdapterInputError("candidate is outside its audited target")

    required_symbols = _string_list(
        installed.get("required_symbols"),
        "required symbols",
        minimum=1,
    )
    if required_symbols != catalog.required_symbols:
        raise OfficialAdapterInputError("required ACLNN symbol set drift")

    header = _strict_object(
        installed.get("selected_header"), _HEADER_FIELDS, "selected header"
    )
    header_path = _relative_path(header.get("relative_path"), "selected header path")
    header_sha256 = _digest(header.get("sha256"), "selected header")
    _exact_int(header.get("size"), "selected header size", 1)
    declared_symbols = _string_list(
        header.get("declared_symbols"), "declared header symbols", minimum=1
    )
    if (
        PurePosixPath(header_path).name != catalog.header_name
        or not set(required_symbols).issubset(declared_symbols)
    ):
        raise OfficialAdapterInputError("selected header does not prove the ACLNN ABI")

    library = _strict_object(
        installed.get("selected_library"), _LIBRARY_FIELDS, "selected library"
    )
    library_path = _relative_path(
        library.get("relative_path"), "selected library path"
    )
    library_sha256 = _digest(library.get("sha256"), "selected library")
    _exact_int(library.get("size"), "selected library size", 1)
    _text(library.get("probe_tool"), "selected library probe", 512)
    symbol_count = _exact_int(
        library.get("aclnn_symbol_count"), "selected library symbol count", 1
    )
    _digest(library.get("aclnn_symbols_sha256"), "selected library symbols")
    if (
        PurePosixPath(library_path).name not in catalog.library_priority
        or symbol_count < len(required_symbols)
    ):
        raise OfficialAdapterInputError("selected library does not prove the ACLNN ABI")

    semantic_review = tuple(catalog.semantic_review)
    refs = tuple(candidate.provenance_refs) + tuple(candidate.candidate_refs)
    refs_by_name: Dict[str, Any] = {}
    for ref in refs:
        if ref.name in refs_by_name:
            raise OfficialAdapterInputError(
                "operator candidate repeats an official evidence reference"
            )
        refs_by_name[ref.name] = ref
    expected_refs = {
        OFFICIAL_CANDIDATE_MAPPING_REF: mapping_sha256,
        OFFICIAL_SELECTED_HEADER_REF: header_sha256,
        OFFICIAL_SELECTED_LIBRARY_REF: library_sha256,
        OFFICIAL_REQUIRED_SYMBOLS_REF: _sha256({
            "required_symbols": list(required_symbols),
        }),
        OFFICIAL_SEMANTIC_REVIEW_REF: _sha256({
            "semantic_review": list(semantic_review),
        }),
    }
    for name, digest in expected_refs.items():
        ref = refs_by_name.get(name)
        if ref is None or ref.sha256 != digest:
            raise OfficialAdapterInputError(
                "operator candidate does not bind %s" % name
            )

    return {
        "candidate_mapping": catalog.to_dict(),
        "candidate_mapping_sha256": mapping_sha256,
        "selected_header": dict(header),
        "selected_library": dict(library),
        "required_symbols": list(required_symbols),
        "semantic_review": list(semantic_review),
    }


@dataclass(frozen=True)
class OfficialCandidateAdapterInput:
    """Deterministic official-binary adapter input with an explicit model gate."""

    _manifest_json: str
    adapter_contract: str

    @property
    def operator_manifest(self) -> Dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def manifest(self) -> Dict[str, Any]:
        """Alias used by recovery receipt adapters."""

        return self.operator_manifest

    @property
    def strategy(self) -> str:
        return str(self.operator_manifest["strategy"])

    @property
    def model_invocation_allowed(self) -> bool:
        return bool(self.operator_manifest["model_invocation_allowed"])

    @property
    def adapter_contract_sha256(self) -> str:
        return hashlib.sha256(self.adapter_contract.encode("utf-8")).hexdigest()

    @property
    def selected_library(self) -> Dict[str, Any]:
        """Return the validated selected-library receipt from the contract."""

        contract = json.loads(self.adapter_contract)
        official = contract["official_binary"]
        return dict(official["selected_library"])

    def require_model_ready(self) -> None:
        """Fail until a route-aware existing-module context plan is attached."""

        if not self.model_invocation_allowed:
            raise OfficialAdapterInputError(
                "official ACLNN adapter requires a validated existing-module "
                "AsNumPy context plan; custom-op planners are incompatible"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_manifest": self.operator_manifest,
            "adapter_contract": self.adapter_contract,
        }


def bind_official_adapter_context(
    prepared: OfficialCandidateAdapterInput,
    context_plan: AsnumpyAdapterContextPlan,
    *,
    api_name: str,
) -> OfficialCandidateAdapterInput:
    """Bind a reviewed existing-module plan and open only the model gate.

    The plan is content-bound and path-bounded by ``asnumpy_adapter_context``.
    This function performs no model, filesystem, build, import, execution, or
    activation action; it merely proves that the official candidate now has a
    compatible AsNumPy engineering plan.
    """

    if not isinstance(prepared, OfficialCandidateAdapterInput):
        raise OfficialAdapterInputError("official adapter input must be typed")
    if not isinstance(context_plan, AsnumpyAdapterContextPlan):
        raise OfficialAdapterInputError("official adapter context plan must be typed")
    if (
        not isinstance(api_name, str)
        or not api_name
        or prepared.operator_manifest.get("qualified_api") != "numpy." + api_name
    ):
        raise OfficialAdapterInputError("official adapter context API mismatch")
    if (
        context_plan.architecture != OFFICIAL_ACLNN_ARCHITECTURE
        or context_plan.mode != EXISTING_MODULE_MODE
        or not context_plan.context
        or not context_plan.expected_overlay_paths
        or any("custom_ops" in path for path in context_plan.expected_overlay_paths)
    ):
        raise OfficialAdapterInputError(
            "official candidate requires an existing-module AsNumPy context plan"
        )
    manifest = prepared.operator_manifest
    if (
        manifest.get("strategy") != OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY
        or manifest.get("origin_kind") != OperatorCandidateOrigin.OFFICIAL_BINARY.value
        or manifest.get("activation_allowed") is not False
    ):
        raise OfficialAdapterInputError("official adapter input strategy drift")
    plan_manifest = context_plan.manifest()
    if plan_manifest.get("plan_sha256") != context_plan.plan_sha256:
        raise OfficialAdapterInputError("official adapter context digest drift")
    selected_library = prepared.selected_library
    library_binding = plan_manifest.get("official_library_binding")
    if (
        context_plan.schema_version != 3
        or not isinstance(library_binding, Mapping)
        or library_binding.get("selected_library_relative_path")
        != selected_library.get("relative_path")
        or library_binding.get("selected_library_sha256")
        != selected_library.get("sha256")
    ):
        raise OfficialAdapterInputError(
            "official adapter context does not bind the selected library to "
            "the reviewed AsNumPy link graph"
        )
    manifest.update({
        "context_plan_status": "prepared",
        "required_next_gate": "invoke_configured_model_for_official_adapter",
        "model_invocation_allowed": True,
        "asnumpy_adapter_context_plan": plan_manifest,
    })
    rebound = OfficialCandidateAdapterInput(
        _manifest_json=_canonical_bytes(manifest).decode("utf-8"),
        adapter_contract=prepared.adapter_contract,
    )
    rebound.require_model_ready()
    return rebound


def prepare_official_candidate_adapter_input(
    candidate: OperatorCandidateArtifact,
    gap_spec: GapSpec,
    resolution: OfficialGapResolution,
) -> OfficialCandidateAdapterInput:
    """Validate one installed ACLNN candidate and prepare deterministic inputs.

    The returned compatibility ``artifact_id`` is explicitly an alias for the
    route-neutral ``candidate_id``.  The manifest has a distinct artifact type,
    origin and strategy and therefore cannot be mistaken for a remote model
    operator artifact.
    """

    if not isinstance(candidate, OperatorCandidateArtifact):
        raise OfficialAdapterInputError("candidate must be typed")
    if not isinstance(gap_spec, GapSpec):
        raise OfficialAdapterInputError("GapSpec must be typed")
    if not isinstance(resolution, OfficialGapResolution):
        raise OfficialAdapterInputError("resolution must be typed")
    try:
        candidate = OperatorCandidateArtifact.from_dict(candidate.to_dict())
        gap_spec = GapSpec.from_dict(gap_spec.to_dict())
        resolution = OfficialGapResolution.from_dict(resolution.to_dict())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise OfficialAdapterInputError("official adapter typed input drift") from exc

    official = _validate_installed_resolution(candidate, gap_spec, resolution)
    candidate_digest = operator_candidate_sha256(candidate)
    spec_digest = gap_spec_sha256(gap_spec)
    decision_value = resolution.to_dict()
    decision_digest = _sha256(decision_value)
    if (
        candidate.gap_spec_ref.sha256 != spec_digest
        or candidate.resolution_decision_ref.sha256 != decision_digest
    ):
        raise OfficialAdapterInputError("candidate evidence hashes drifted")
    if (
        gap_spec.qualified_api != "numpy." + gap_spec.api_name
        or gap_spec.asnumpy_api != "asnumpy." + gap_spec.api_name
        or gap_spec.qualified_api != candidate.qualified_api
    ):
        raise OfficialAdapterInputError("GapSpec API identity is inconsistent")

    contract = {
        "schema_version": OFFICIAL_ADAPTER_INPUT_SCHEMA_VERSION,
        "contract_type": "official_binary_asnumpy_adapter",
        "strategy": OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY,
        "source_candidate": {
            "artifact_id": candidate.candidate_id,
            "candidate_id": candidate.candidate_id,
            "candidate_sha256": candidate_digest,
            "origin_kind": candidate.origin_kind.value,
            "license": candidate.license.to_dict(),
        },
        "api": {
            "gap_id": gap_spec.gap_id,
            "api_name": gap_spec.api_name,
            "qualified_api": gap_spec.qualified_api,
            "asnumpy_api": gap_spec.asnumpy_api,
            "signature_variants": [
                dict(item) for item in gap_spec.signature_variants
            ],
            "shape_rules": list(gap_spec.shape_rules),
            "dtype_rules": list(gap_spec.dtype_rules),
            "semantic_source": gap_spec.semantic_source,
            "reference_cases": [
                dict(item) for item in gap_spec.reference_cases
            ],
        },
        "target": {
            "soc": candidate.target_soc,
            "cann_version": candidate.cann_version,
        },
        "evidence": {
            "gap_spec_sha256": spec_digest,
            "resolution_decision_sha256": decision_digest,
            "candidate_mapping_sha256": official[
                "candidate_mapping_sha256"
            ],
        },
        "official_binary": official,
        "permissions": {
            "reuse_installed_aclnn": True,
            "generate_new_kernel": False,
            "generate_custom_opp": False,
            "activation_allowed": False,
        },
        "activation_allowed": False,
    }
    contract_payload = _canonical_bytes(contract) + b"\n"
    if len(contract_payload) > _MAX_CONTRACT_BYTES:
        raise OfficialAdapterInputError("official adapter contract is too large")
    contract_text = contract_payload.decode("utf-8")
    contract_digest = hashlib.sha256(contract_payload).hexdigest()

    manifest = {
        "schema_version": OFFICIAL_ADAPTER_INPUT_SCHEMA_VERSION,
        "artifact_type": "official_binary_adapter_input",
        "artifact_id": candidate.candidate_id,
        "artifact_id_semantics": "operator_candidate_id_compatibility_alias",
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate_digest,
        "origin_kind": candidate.origin_kind.value,
        "strategy": OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY,
        "gap_id": gap_spec.gap_id,
        "qualified_api": gap_spec.qualified_api,
        "asnumpy_api": gap_spec.asnumpy_api,
        "target_soc": candidate.target_soc,
        "cann_version": candidate.cann_version,
        "gap_spec_sha256": spec_digest,
        "resolution_decision_sha256": decision_digest,
        "adapter_contract_sha256": contract_digest,
        "context_plan_status": "required_not_prepared",
        "required_next_gate": "prepare_official_existing_module_context_plan",
        "model_invocation_allowed": False,
        "new_kernel_generation_allowed": False,
        "custom_opp_generation_allowed": False,
        "activation_allowed": False,
    }
    return OfficialCandidateAdapterInput(
        _manifest_json=_canonical_bytes(manifest).decode("utf-8"),
        adapter_contract=contract_text,
    )


def load_official_candidate_adapter_input(
    candidate_path: os.PathLike[str] | str,
    gap_spec_path: os.PathLike[str] | str,
    resolution_path: os.PathLike[str] | str,
) -> OfficialCandidateAdapterInput:
    """Safely load three bounded receipts and prepare an inert adapter input.

    ``resolution_path`` may contain either one canonical
    ``OfficialGapResolution`` or its complete canonical parent report.
    """

    candidate_value = _read_regular_bounded_json(
        candidate_path, "operator candidate"
    )
    spec_value = _read_regular_bounded_json(gap_spec_path, "GapSpec")
    resolution_value = _read_regular_bounded_json(
        resolution_path, "official resolution"
    )
    candidate = _canonical_candidate(candidate_value)
    spec = _canonical_gap_spec(spec_value)
    resolution = _selected_resolution(resolution_value, candidate.gap_id)
    return prepare_official_candidate_adapter_input(
        candidate, spec, resolution
    )


def load_operator_candidate_artifact(
    candidate_path: os.PathLike[str] | str,
) -> OperatorCandidateArtifact:
    """Safely load one canonical route-neutral operator candidate file."""

    return _canonical_candidate(
        _read_regular_bounded_json(candidate_path, "operator candidate")
    )


__all__ = [
    "OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY",
    "OFFICIAL_ADAPTER_INPUT_SCHEMA_VERSION",
    "OFFICIAL_CANDIDATE_MAPPING_REF",
    "OFFICIAL_REQUIRED_SYMBOLS_REF",
    "OFFICIAL_SELECTED_HEADER_REF",
    "OFFICIAL_SELECTED_LIBRARY_REF",
    "OFFICIAL_SEMANTIC_REVIEW_REF",
    "OfficialAdapterInputError",
    "OfficialCandidateAdapterInput",
    "bind_official_adapter_context",
    "load_official_candidate_adapter_input",
    "load_operator_candidate_artifact",
    "prepare_official_candidate_adapter_input",
]
