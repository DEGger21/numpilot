"""Fail-closed validator for the official adapter model gate.

This module validates a context-bound ``official_candidate_gap`` hand-off.  It
reconstructs the inert adapter input from the hand-off/preflight receipts,
rebuilds the reviewed AsNumpy existing-module context plan, and compares every
hash-bound output before declaring the next model gate ready.  It never calls a
model, writes an overlay, imports AsNumpy, or creates RecoveryStore state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from numpilot.gaps.official_candidate_preflight import (
    OfficialCandidatePreflightError,
    prepare_official_adapter_input_from_preflight,
)
from numpilot.asnumpy_adapter_context import plan_official_aclnn_adapter_context
from numpilot.gaps.official_adapter import bind_official_adapter_context

SCHEMA = "numpilot.official-adapter-context-gate.v1"
STATUS = "ready_for_official_adapter_model"
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_REQUIRED_SIDE_EFFECTS = (
    "model_invocation_performed",
    "network_access_performed",
    "system_write_performed",
    "asnumpy_import_performed",
    "build_performed",
    "execution_performed",
    "activation_performed",
)


class OfficialAdapterContextGateError(RuntimeError):
    """The context-bound official adapter input is stale or unsafe."""


def _read_json(path: str, label: str) -> Mapping[str, Any]:
    target = Path(path).absolute()
    try:
        if not target.is_file() or target.is_symlink():
            raise OfficialAdapterContextGateError("%s must be a regular file" % label)
        if target.stat().st_size > _MAX_RECEIPT_BYTES:
            raise OfficialAdapterContextGateError("%s exceeds size limit" % label)
        value = json.loads(target.read_text(encoding="utf-8"))
    except OfficialAdapterContextGateError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise OfficialAdapterContextGateError("cannot read %s" % label) from exc
    if not isinstance(value, Mapping):
        raise OfficialAdapterContextGateError("%s must be a JSON object" % label)
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_official_adapter_context(
    bound_receipt_path: str,
    handoff_path: str,
    preflight_path: str,
    source_path: str,
    official_resolution_path: str,
) -> Mapping[str, Any]:
    """Rebuild and validate a context-bound official adapter input."""

    bound = _read_json(bound_receipt_path, "context-bound receipt")
    if set(bound) != {
        "schema",
        "source",
        "reference_root",
        "operator_manifest",
        "adapter_contract",
        "context_plan",
        "side_effects",
    }:
        raise OfficialAdapterContextGateError("context-bound receipt fields are invalid")
    if bound.get("schema") != "numpilot.official-adapter-context-bound.v1":
        raise OfficialAdapterContextGateError("unsupported context-bound receipt schema")
    side_effects = bound.get("side_effects")
    if not isinstance(side_effects, Mapping) or set(side_effects) != set(_REQUIRED_SIDE_EFFECTS):
        raise OfficialAdapterContextGateError("context-bound side-effect fields are invalid")
    if any(value is not False for value in side_effects.values()):
        raise OfficialAdapterContextGateError("context-bound receipt claims a side effect")
    source_abs = str(Path(source_path).absolute())
    if bound.get("source") != source_abs:
        raise OfficialAdapterContextGateError("context-bound source path drift")
    reference_root = bound.get("reference_root")
    if not isinstance(reference_root, str) or not reference_root:
        raise OfficialAdapterContextGateError("context-bound reference root is invalid")
    manifest = bound.get("operator_manifest")
    contract = bound.get("adapter_contract")
    context_plan = bound.get("context_plan")
    if not isinstance(manifest, Mapping) or not isinstance(contract, Mapping) or not isinstance(context_plan, Mapping):
        raise OfficialAdapterContextGateError("context-bound payload sections are invalid")
    if manifest.get("model_invocation_allowed") is not True or manifest.get(
        "required_next_gate"
    ) != "invoke_configured_model_for_official_adapter":
        raise OfficialAdapterContextGateError("context-bound model gate is not open")
    qualified_api = manifest.get("qualified_api")
    if not isinstance(qualified_api, str) or not qualified_api.startswith("numpy."):
        raise OfficialAdapterContextGateError("context-bound API identity is invalid")
    api_name = qualified_api.removeprefix("numpy.")
    try:
        inert = prepare_official_adapter_input_from_preflight(
            handoff_path, preflight_path, source_path, official_resolution_path
        )
        selected_library = inert.selected_library
        plan = plan_official_aclnn_adapter_context(
            reference_root,
            api_name,
            selected_library_relative_path=selected_library["relative_path"],
            selected_library_sha256=selected_library["sha256"],
        )
        expected = bind_official_adapter_context(inert, plan, api_name=api_name)
    except (OfficialCandidatePreflightError, OSError, ValueError, TypeError, RuntimeError) as exc:
        raise OfficialAdapterContextGateError("cannot rebuild context-bound adapter input") from exc
    if expected.operator_manifest != dict(manifest):
        raise OfficialAdapterContextGateError("context-bound operator manifest drift")
    try:
        bound_contract = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected_contract = json.dumps(
            json.loads(expected.adapter_contract), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise OfficialAdapterContextGateError("context-bound adapter contract is invalid") from exc
    if bound_contract != expected_contract:
        raise OfficialAdapterContextGateError("context-bound adapter contract drift")
    if dict(context_plan) != expected.operator_manifest.get("asnumpy_adapter_context_plan"):
        raise OfficialAdapterContextGateError("context-bound context plan drift")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "source": source_abs,
        "reference_root": reference_root,
        "qualified_api": qualified_api,
        "candidate_id": str(manifest.get("candidate_id", "")),
        "context_plan_sha256": str(context_plan.get("plan_sha256", "")),
        "adapter_contract_sha256": expected.adapter_contract_sha256,
        "model_invocation_performed": False,
        "network_access_performed": False,
        "system_write_performed": False,
        "asnumpy_import_performed": False,
        "build_performed": False,
        "execution_performed": False,
        "activation_performed": False,
        "next_action": "invoke_configured_model_for_official_adapter",
    }


__all__ = [
    "SCHEMA",
    "STATUS",
    "OfficialAdapterContextGateError",
    "validate_official_adapter_context",
]
