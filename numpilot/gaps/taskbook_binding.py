"""Bind a requirements taskbook to the candidate it actually describes.

Internal taskbook validity is necessary but cannot authorize a different gap.
These checks precede any new provider request; they do not qualify an operator
or retroactively alter receipts from an earlier generation.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from numpilot.gap_requirements import build_gap_taskbook, persist_gap_taskbook, validate_gap_taskbook


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("taskbook binding JSON contains a duplicate key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("taskbook binding JSON contains a non-finite constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _canonical_spec(taskbook):
    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.official_resolution import gap_spec_sha256

    value = validate_gap_taskbook(taskbook)
    if value["mode"] != "gap_spec_requirements":
        raise ValueError("taskbook binding requires a canonical GapSpec")
    spec = GapSpec.from_dict(value["gap_spec"])
    digest = gap_spec_sha256(spec)
    if value["gap_spec_sha256"] != digest:
        raise ValueError("taskbook canonical GapSpec hash mismatch")
    return value, spec, digest


def validate_adapter_taskbook(taskbook, *, operator_artifact, adapter_contract):
    """Check every supplied identity, including official and legacy contracts."""
    value, spec, spec_sha = _canonical_spec(taskbook)
    if not isinstance(operator_artifact, Mapping):
        raise ValueError("taskbook binding requires an operator artifact object")
    if not isinstance(adapter_contract, str) or len(adapter_contract.encode("utf-8")) > 512 * 1024:
        raise ValueError("taskbook binding requires a bounded adapter contract")
    contract = _strict_json(adapter_contract)
    if not isinstance(contract, Mapping):
        raise ValueError("taskbook binding requires an adapter contract object")
    expected = {"gap_id": spec.gap_id, "api_name": spec.api_name,
                "qualified_api": spec.qualified_api, "asnumpy_api": spec.asnumpy_api,
                "source_sha256": spec.source_sha256, "gap_source_sha256": spec.source_sha256,
                "gap_spec_sha256": spec_sha, "target_soc": spec.target_soc,
                "cann_version": spec.cann_version}

    def check(mapping, label, fields=expected):
        if not isinstance(mapping, Mapping):
            raise ValueError("taskbook binding object is invalid: " + label)
        for key, expected_value in fields.items():
            if key in mapping and mapping[key] != expected_value:
                raise ValueError("taskbook binding mismatch: " + label + "." + key)

    if operator_artifact.get("gap_id") != spec.gap_id:
        raise ValueError("taskbook binding mismatch: operator_artifact.gap_id")
    for mapping, label in ((operator_artifact, "operator_artifact"), (contract, "adapter_contract")):
        check(mapping, label)
        for key in ("api", "evidence"):
            if key in mapping:
                check(mapping[key], label + "." + key)
        if "target" in mapping:
            check(mapping["target"], label + ".target", {"soc": spec.target_soc, "cann_version": spec.cann_version})

    api = contract.get("api", {})
    operator = contract.get("operator", {})
    if "operator" in contract:
        check(operator, "adapter_contract.operator", {
            **expected, "source_api": spec.qualified_api, "public_api": spec.asnumpy_api})
    # The custom-operator draft generator writes a draft-stage operator
    # manifest whose identity lives under api.numpy / api.asnumpy and
    # names.python_wrapper_callable.  Validate that shape exactly instead of
    # silently accepting an unbound or conflicting producer.
    draft_names = contract.get("names")
    draft_identity_present = (
        isinstance(api, Mapping) and ("numpy" in api or "asnumpy" in api)
    ) or isinstance(draft_names, Mapping)
    if draft_identity_present:
        check(api, "adapter_contract.api", {
            "numpy": spec.qualified_api, "asnumpy": spec.asnumpy_api})
        check(draft_names, "adapter_contract.names", {
            "python_wrapper_callable": spec.api_name,
            "python_public_api": spec.asnumpy_api})
    has_api_identity = (any(key in contract for key in ("api_name", "qualified_api", "asnumpy_api"))
                        or any(key in api for key in ("api_name", "qualified_api", "asnumpy_api"))
                        or any(key in operator for key in ("source_api", "public_api"))
                        or (
                            isinstance(api, Mapping)
                            and api.get("numpy") == spec.qualified_api
                            and api.get("asnumpy") == spec.asnumpy_api
                            and isinstance(draft_names, Mapping)
                            and draft_names.get("python_wrapper_callable") == spec.api_name
                        ))
    if not has_api_identity:
        raise ValueError("taskbook binding adapter API identity is missing")
    for mapping, label in ((contract, "adapter_contract"), (operator, "adapter_contract.operator")):
        if "name_mapping" in mapping:
            names = mapping["name_mapping"]
            if not isinstance(names, Mapping) or not isinstance(names.get("python"), Mapping):
                raise ValueError("taskbook binding Python name mapping is invalid")
            check(names["python"], label + ".name_mapping.python", {
                "public_symbol": spec.api_name, "public_qualified_name": spec.asnumpy_api})
    return value


def prepare_spec_taskbook(spec, workspace):
    """Reuse full observed requirements, or persist a new spec-bound request."""
    from numpilot.gaps.official_resolution import gap_spec_sha256

    workspace = Path(workspace)
    path = workspace / "GAP_TASKBOOK.json"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("taskbook input must be a bounded regular file")
        taskbook = _strict_json(path.read_text(encoding="utf-8"))
        value, _bound_spec, digest = _canonical_spec(taskbook)
        if digest != gap_spec_sha256(spec):
            raise ValueError("taskbook binding differs from the requested GapSpec")
    else:
        value = build_gap_taskbook(spec)
    # Also verifies an existing Markdown representation and the parent path.
    # This never silently rebuilds a source-observed book without its facts.
    persist_gap_taskbook(workspace, value)
    return value
