"""Inspectable, immutable requirements handed from Planner to Gap Filler.

A taskbook is a request, never a capability certificate. GapSpec remains the
single semantic specification: taskbook sections reference its fields instead
of maintaining another copy of dtype, shape, or route rules. No source is run.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from numpilot.gaps.models import GapSpec

SCHEMA = "numpilot.gap-taskbook"
REF_SCHEMA = "numpilot.gap-taskbook-ref"
JSON_NAME = "GAP_TASKBOOK.json"
MARKDOWN_NAME = "GAP_TASKBOOK.md"
_MAX_BYTES = 4 * 1024 * 1024
_PROFILE_FIELDS = ("input_dtype", "input_shape", "input_rank", "axis", "layout",
                   "broadcast", "output_dtype", "output_shape", "return_structure")


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value) -> str:
    return _digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _seal(value: dict) -> dict:
    value["taskbook_id"] = "gap-taskbook-" + _canonical_digest(value)
    return value


def _literal(node):
    # ast.literal_eval is deliberately restricted further to JSON metadata.
    try:
        value = ast.literal_eval(node)
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return {"status": "unknown", "expression": ast.unparse(node)}
    if not isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
        return {"status": "unknown", "expression": ast.unparse(node)}
    return {"status": "static_literal", "value": json.loads(json.dumps(value)),
            "expression": ast.unparse(node)}


def _call_arguments(expression: str) -> dict:
    try:
        call = ast.parse(expression, mode="eval").body
        if not isinstance(call, ast.Call):
            raise ValueError("not a call")
        return {
            "status": "static_syntax_only",
            "positional": [_literal(node) for node in call.args],
            "keywords": [{"name": item.arg, **_literal(item.value)} for item in call.keywords],
        }
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return {"status": "unknown", "reason": "call expression unavailable or truncated",
                "positional": [], "keywords": []}


def _unknown_profiles() -> dict:
    return {field: {"status": "unknown", "evidence": None} for field in _PROFILE_FIELDS}


_OBSERVED_PROFILE_FIELDS = frozenset(
    {"callsite", "signature", "operands", "unprofiled_operands", "output"}
)
_TENSOR_PROFILE_FIELDS = frozenset({"dtype", "shape", "rank", "c_contiguous"})
_OPERAND_PROFILE_FIELDS = _TENSOR_PROFILE_FIELDS | {"name"}


def _observed_matches(spec, observed) -> list:
    """Return observed callsites bound to this exact GapSpec, or raise."""

    if observed is None:
        return []
    if getattr(observed, "source_sha256", None) != spec.source_sha256:
        raise ValueError("observed inputs do not match the GapSpec source SHA-256")
    payload = observed.to_dict()
    raw_calls = payload.get("callsites")
    if not isinstance(raw_calls, list):
        raise ValueError("observed inputs have no callsite list")
    writable = [
        (index, item) for index, item in enumerate(raw_calls)
        if isinstance(item, dict)
        and item.get("qualified_api") == spec.qualified_api
    ]
    matched = []
    by_line = {}
    for index, item in writable:
        by_line.setdefault(item.get("lineno"), []).append((index, item))
    for site in spec.callsites:
        candidates = by_line.get(site.lineno, [])
        if len(candidates) == 1:
            matched.append(candidates[0][1])
            continue
        for index, item in candidates:
            signature = {
                "positional_args": item.get("positional_args"),
                "keyword_args": sorted(item.get("keyword_args") or []),
            }
            if any(
                int(variant.get("positional_args", -1)) == signature["positional_args"]
                and sorted(variant.get("keyword_args") or []) == signature["keyword_args"]
                for variant in spec.signature_variants
            ):
                matched.append(item)
                break
    return matched


def _observed_call_entry(item) -> dict:
    return {
        "callsite": {"lineno": int(item["lineno"]), "col_offset": None},
        "signature": {
            "positional_args": int(item["positional_args"]),
            "keyword_args": list(item.get("keyword_args") or []),
        },
        "operands": [dict(value) for value in item.get("inputs") or []],
        "unprofiled_operands": list(item.get("unprofiled_operands") or []),
        "output": dict(item["output"]) if item.get("output") is not None else None,
    }


def _observed_facts(entries) -> dict:
    """Summarize observed operands into the taskbook's fixed profile fields."""

    facts = _unknown_profiles()
    if not entries:
        return facts
    operands = [operand for entry in entries for operand in entry["operands"]]
    outputs = [entry["output"] for entry in entries if entry["output"] is not None]
    dtypes = sorted({operand["dtype"] for operand in operands})
    shapes = [tuple(operand["shape"]) for operand in operands]
    ranks = {operand["rank"] for operand in operands}
    contiguous = [operand["c_contiguous"] for operand in operands]
    evidence = "observed-inputs.json#callsites"
    if dtypes:
        facts["input_dtype"] = {
            "status": "observed" if len(dtypes) == 1 else "observed_multi_dtype",
            "value": dtypes[0] if len(dtypes) == 1 else dtypes,
            "evidence": evidence,
        }
    if shapes:
        uniform = len(set(shapes)) == 1
        facts["input_shape"] = {
            "status": "observed" if uniform else "observed_per_callsite",
            "value": list(shapes[0]) if uniform else [list(shape) for shape in shapes],
            "evidence": evidence,
        }
    if ranks:
        rank_values = sorted(ranks)
        facts["input_rank"] = {
            "status": "observed" if len(rank_values) == 1 else "observed_per_callsite",
            "value": rank_values[0] if len(rank_values) == 1 else rank_values,
            "evidence": evidence,
        }
    if contiguous:
        facts["layout"] = {
            "status": "observed",
            "value": "c_contiguous" if all(contiguous) else "observed_mixed",
            "evidence": evidence,
        }
    if shapes:
        facts["broadcast"] = {
            "status": "observed",
            "value": "no_broadcast_observed" if len(set(shapes)) == 1 else "multi_shape_observed",
            "evidence": evidence,
        }
    if outputs:
        output_dtypes = sorted({item["dtype"] for item in outputs})
        output_shapes = [tuple(item["shape"]) for item in outputs]
        facts["output_dtype"] = {
            "status": "observed" if len(output_dtypes) == 1 else "observed_per_callsite",
            "value": output_dtypes[0] if len(output_dtypes) == 1 else output_dtypes,
            "evidence": evidence,
        }
        facts["output_shape"] = {
            "status": "observed" if len(set(output_shapes)) == 1 else "observed_per_callsite",
            "value": list(output_shapes[0]) if len(set(output_shapes)) == 1 else [list(shape) for shape in output_shapes],
            "evidence": evidence,
        }
        facts["return_structure"] = {"status": "observed", "value": "single_tensor",
                                     "evidence": evidence}
    return facts


def _validate_observed_contract(contract: Mapping) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("taskbook call_contract must be an object")
    profiles = contract.get("observed_profiles")
    if not isinstance(profiles, list):
        raise ValueError("taskbook observed_profiles must be a list")
    if len(profiles) > 64:
        raise ValueError("taskbook observed_profiles exceeds the bound")
    for entry in profiles:
        if not isinstance(entry, Mapping) or frozenset(entry) != _OBSERVED_PROFILE_FIELDS:
            raise ValueError("taskbook observed profile fields are invalid")
        site = entry["callsite"]
        if (not isinstance(site, Mapping) or frozenset(site) != {"lineno", "col_offset"}
                or isinstance(site["lineno"], bool) or not isinstance(site["lineno"], int)):
            raise ValueError("taskbook observed callsite is invalid")
        signature = entry["signature"]
        if (not isinstance(signature, Mapping)
                or frozenset(signature) != {"positional_args", "keyword_args"}
                or isinstance(signature["positional_args"], bool)
                or not isinstance(signature["positional_args"], int)
                or not isinstance(signature["keyword_args"], list)):
            raise ValueError("taskbook observed signature is invalid")
        for operand in entry["operands"]:
            if not isinstance(operand, Mapping) or frozenset(operand) != _OPERAND_PROFILE_FIELDS:
                raise ValueError("taskbook observed operand fields are invalid")
            if not isinstance(operand["name"], str) or not operand["name"]:
                raise ValueError("taskbook observed operand name is invalid")
            if not isinstance(operand["dtype"], str) or not operand["dtype"]:
                raise ValueError("taskbook observed operand dtype is invalid")
            if (not isinstance(operand["shape"], list)
                    or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
                           for size in operand["shape"])
                    or operand["rank"] != len(operand["shape"])
                    or not isinstance(operand["c_contiguous"], bool)):
                raise ValueError("taskbook observed operand values are invalid")
        output = entry["output"]
        if output is not None:
            if not isinstance(output, Mapping) or frozenset(output) != _TENSOR_PROFILE_FIELDS:
                raise ValueError("taskbook observed output fields are invalid")
            if (not isinstance(output["shape"], list)
                    or output["rank"] != len(output["shape"])):
                raise ValueError("taskbook observed output values are invalid")
    facts = contract.get("tensor_facts")
    if not isinstance(facts, Mapping) or frozenset(facts) != frozenset(_PROFILE_FIELDS):
        raise ValueError("taskbook tensor_facts fields are invalid")
    for field, fact in facts.items():
        if not isinstance(fact, Mapping) or "status" not in fact:
            raise ValueError("taskbook tensor fact is invalid: %s" % field)
    if profiles and contract.get("observation_status") == "observed":
        if (facts["input_dtype"].get("status") != "observed"
                or facts["input_dtype"].get("value") != "float32"):
            raise ValueError("taskbook observed input dtype contradicts float32 scope")


def _scope() -> dict:
    return {
        "status": "requested_unverified",
        "input_dtype": "float32",
        "dtype_source": "project_policy_not_observed_profile",
        "output_dtype": "NumPy oracle determines output; comparison results may be bool",
        "other_input_dtypes": "reject_or_defer_without_implicit_cast",
        "unsupported_signatures": "explicit_rejection_no_host_fallback",
        "unknown_shape_policy": "generator must declare a bounded support scope and rejection cases; qualification requires measured profiles",
        "qualification_scope": "only the tested dtype/shape/layout/signature profiles; no global API claim",
    }


def _semantics(reference: str) -> dict:
    return {
        "authority_ref": reference,
        "status": "requires_oracle_validation",
        "checklist": ["input and output dtype, dtype promotion, scalar materialization",
                      "shape, axis normalization, broadcasting, layout and return structure",
                      "NaN, positive/negative infinity, positive/negative zero and signbit",
                      "out/where/casting and dtype keywords: preserve semantics or reject explicitly",
                      "empty input, invalid axes/shapes, unsupported dtypes and exceptions"],
        "do_not": ["silently cast other inputs to float32", "drop unsupported keyword arguments",
                   "replace a failed NumPy contract with a looser assertion"],
    }


def _integration(target_soc=None, cann_version=None) -> dict:
    return {
        "target_soc": target_soc,
        "cann_version": cann_version,
        "strategy": "inspect pinned official CANN/ACLNN support first; custom Ascend C is a candidate when official reuse is not qualified",
        "source_boundary": "reviewed isolated copy of bundled AsNumpy source; Python public API, pybind, native implementation and tests must agree",
        "mapping_required": ["public asnumpy API", "Python wrapper", "pybind symbol",
                             "native implementation", "ACLNN GetWorkspaceSize and execute symbols"],
        "custom_layout": ".asc or op_host/op_kernel is acceptable when the required two-stage ACLNN interface is exposed",
        "dependency_refs": ["gap-spec.json", "verified AsNumpy source manifest and adapter context plan",
                            "verified embedded CANNBot skill bundle", "official operator resolution evidence"],
        "forbidden": ["implicit writes to system CANN or shared AsNumpy checkout",
                      "silent CPU/NumPy fallback", "unverified package activation"],
    }


def _acceptance(oracle: dict) -> dict:
    return {
        "status": "pending",
        "oracle": oracle,
        "tolerance": {"status": "unknown", "required": "record dtype-specific rtol/atol before numerical qualification; compare boolean outputs exactly and preserve oracle special values/signbit"},
        "gates": [
            {"id": "build", "owner": "deterministic_validator", "requires": "isolated build/import and public API binding checks"},
            {"id": "unit_and_boundary", "owner": "deterministic_validator", "requires": "NumPy oracle, admitted profiles, unsupported dtype/shape/kwargs, repeated input lifecycle"},
            {"id": "numerical", "owner": "deterministic_validator", "requires": "saved original inputs, expected/actual outputs, explicit tolerances, NaN/Inf/signed-zero checks"},
            {"id": "native_execution", "owner": "deterministic_validator", "requires": "same source/wheel/runtime identity, native call evidence and real NPU profiler; no host fallback"},
            {"id": "whole_program", "owner": "deterministic_validator", "requires": "original program versus migrated program, actual callsite profiles and all runtime guards"},
            {"id": "performance", "owner": "deterministic_validator", "requires": "separate performance receipt: one CPU core vs one NPU, timing scope, warmup, synchronization and raw samples; correctness alone is no speedup claim"},
            {"id": "activation", "owner": "deterministic_validator", "requires": "all required qualification gates and capability rescan; generated code remains unverified until then"},
        ],
        "planner_self_certification": False,
    }


def _responsibility(next_role="gap_filler") -> dict:
    return {"planner": "organize evidence, requested scope and unresolved requirements",
            "next_role": next_role,
            "generation_owner": "gap_filler",
            "generation_actor": "configured model or external Codex/manual developer",
            "verification_owner": "deterministic_validator",
            "on_failure": "retain taskbook and failed evidence; revise candidate, do not advertise completion"}


def _dtype_conflicts(spec: GapSpec, call_contract: list) -> list:
    conflicts = []
    for rule in spec.dtype_rules:
        match = re.fullmatch(r"(?:dtype\s*=\s*)?(float\d+|int\d+|uint\d+|complex\d+|bool)", rule.strip())
        if match and match.group(1) != "float32":
            conflicts.append("GapSpec declares %s" % rule)
    for site in call_contract:
        for keyword in site["arguments"]["keywords"]:
            if keyword["name"] != "dtype":
                continue
            value = keyword.get("value", keyword.get("expression", ""))
            name = str(value).split(".")[-1]
            if name != "float32":
                conflicts.append("callsite dtype keyword requires resolution: %s" % value)
    return sorted(set(conflicts))


def build_gap_taskbook(spec: GapSpec, *, source: str | None = None,
                       observed=None) -> dict:
    """Make a requirements artifact from a canonical GapSpec.

    ``observed`` is an optional, hash-bound input-observation receipt produced
    by the Planner's explicit observation stage.  It records the concrete
    dtype/shape that reached each gap callsite while the original NumPy
    program ran on the CPU; it never claims device execution.
    """
    from numpilot.gaps.models import GapSpec

    if not isinstance(spec, GapSpec):
        raise TypeError("taskbook requires GapSpec")
    spec = GapSpec.from_dict(spec.to_dict())
    if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", spec.api_name)
            or spec.qualified_api.split(".")[-1] != spec.api_name
            or spec.asnumpy_api != "asnumpy." + spec.api_name
            or any(site.qualified_api != spec.qualified_api for site in spec.callsites)):
        raise ValueError("taskbook API names do not match the GapSpec")
    if source is not None and _digest(source.encode("utf-8")) != spec.source_sha256:
        raise ValueError("taskbook source does not match GapSpec SHA-256")
    source_calls = {}
    if source is not None:
        tree = ast.parse(source)
        source_calls = {(node.lineno, node.col_offset): node for node in ast.walk(tree)
                        if isinstance(node, ast.Call)}
    calls = []
    for site in spec.callsites:
        expression = site.call_expr
        if source is not None:
            node = source_calls.get((site.lineno, site.col_offset))
            if node is None:
                raise ValueError("taskbook callsite does not exist in bound source")
            expression = ast.unparse(node)
        calls.append({"lineno": site.lineno, "col_offset": site.col_offset,
                      "arguments": _call_arguments(expression)})
    conflicts = _dtype_conflicts(spec, calls)
    observed_status = getattr(observed, "status", None) if observed is not None else None
    matched = _observed_matches(spec, observed)
    usable = observed_status in {"observed", "unsupported_dtype"}
    observed_entries = [_observed_call_entry(item) for item in matched] if usable else []
    observed_facts = _observed_facts(observed_entries)
    if observed_entries and observed_status == "unsupported_dtype":
        conflicts = list(conflicts) + [
            "observed operand dtype is outside the float32 development scope"
        ]
    oracle = {"status": "inputs_required", "semantic_source_ref": "#/gap_spec/semantic_source",
              "reference_cases_ref": "#/gap_spec/reference_cases", "whole_program_ref": "#/source",
              "input_materialization": "save reproducible inputs for each requested profile; no inputs have been executed by Planner"}
    if observed_entries:
        oracle["status"] = "inputs_observed_cpu"
        oracle["input_materialization"] = (
            "observed-inputs.json records the concrete operands that reached the gap "
            "callsite while the original NumPy program ran on the CPU; the device "
            "region was not executed by AsNumpy and remains unverified"
        )
    if spec.qualified_api.startswith("numpy."):
        oracle["python"] = ("import numpy as np\n\ndef numpy_oracle(*args, **kwargs):\n"
                            "    return np.%s(*args, **kwargs)\n" % spec.api_name)
    else:
        oracle["status"] = "declared_operator_oracle_required"
        oracle["python"] = None
    support_scope = {**_scope(), "signatures_ref": "#/gap_spec/signature_variants",
                     "shape_rules_ref": "#/gap_spec/shape_rules"}
    if observed_entries:
        support_scope["dtype_source"] = "observed_cpu_execution"
    value = {
        "schema": SCHEMA, "schema_version": 1, "mode": "gap_spec_requirements",
        "status": "requirements_blocked" if conflicts else "requirements_prepared_unverified",
        "source": {"path": spec.source_path, "sha256": spec.source_sha256,
                   "content_checked": source is not None},
        "gap_spec": spec.to_dict(), "gap_spec_sha256": _canonical_digest(spec.to_dict()),
        "gap": {"target_ref": "#/gap_spec/qualified_api", "asnumpy_target_ref": "#/gap_spec/asnumpy_api",
                "callsites_ref": "#/gap_spec/callsites", "route_ref": "#/gap_spec/route",
                "route_decisions_ref": "#/gap_spec/route_decisions"},
        "call_contract": {"signature_variants_ref": "#/gap_spec/signature_variants",
                          "callsites": calls, "observed_profiles": observed_entries,
                          "tensor_facts": observed_facts,
                          "observation_status": observed_status or "not_requested",
                          "shape_rules_ref": "#/gap_spec/shape_rules", "dtype_rules_ref": "#/gap_spec/dtype_rules"},
        "semantics": _semantics("#/gap_spec/semantic_source"),
        "support_scope": support_scope,
        "integration": _integration(spec.target_soc, spec.cann_version),
        "acceptance": _acceptance(oracle), "responsibility": _responsibility(),
        "generation": {"allowed": not conflicts, "blockers": conflicts,
                       "claim_level": "unverified_candidate_only"},
        "unresolved": [field for field in _PROFILE_FIELDS
                       if observed_facts[field]["status"] == "unknown"]
                      + ["oracle_inputs", "numerical_tolerances",
                         "candidate_support_and_rejection_profiles"],
    }
    return _seal(value)


def build_planning_requirements(analysis, gap_plan) -> tuple:
    """Static planning summaries; no invented GapSpec or hardware target."""
    requirements = []
    for task in gap_plan.blocking:
        uses = [use for use in analysis.op_uses
                if use.name == task.name and use.lineno in task.call_sites and not use.supported]
        requirements.append({
            "name": task.name, "target": task.target,
            "category": task.category, "reason": task.reason,
            "callsites": [{"qualified_api": use.qualified_name, "lineno": use.lineno,
                           "col_offset": use.col_offset, "call_expr": use.call_expr,
                           "positional_args": use.positional_args, "keyword_args": list(use.keyword_args),
                           "arguments": _call_arguments(use.call_expr or "")}
                          for use in uses],
            "route": task.route, "next_role": task.route,
            "status": "requirements_incomplete", "requested_scope": _scope(),
            "tensor_facts": _unknown_profiles(),
            "unresolved": ["canonical GapSpec", "source SHA-256", "target_soc", "cann_version",
                           "oracle_inputs", "numerical_tolerances", *_PROFILE_FIELDS],
        })
    return tuple(requirements)


def build_planning_taskbook(analysis, gap_plan, *, source: str | None = None) -> dict:
    """Persistable analysis-stage taskbook when no canonical GapSpec exists yet."""
    value = {
        "schema": SCHEMA, "schema_version": 1, "mode": "analysis_requirements",
        "status": "requirements_incomplete", "source": {"path": analysis.source_path,
            "sha256": _digest(source.encode("utf-8")) if source is not None else None,
            "content_checked": source is not None},
        "requirements": list(build_planning_requirements(analysis, gap_plan)),
        "analysis_migratable": bool(analysis.migratable),
        "semantics": _semantics("original source NumPy calls; canonical GapSpec pending"),
        "support_scope": _scope(), "integration": _integration(),
        "acceptance": _acceptance({"status": "canonical_gap_spec_and_inputs_required",
                                   "whole_program_ref": "#/source"}),
        "responsibility": _responsibility(),
        "generation": {"allowed": False, "blockers": ["canonical GapSpec and target environment required"],
                       "claim_level": "none"},
        "unresolved": ["canonical GapSpec", "target_soc", "cann_version", "oracle_inputs",
                       "numerical_tolerances", *_PROFILE_FIELDS],
    }
    if source is not None:
        for requirement in value["requirements"]:
            requirement["unresolved"].remove("source SHA-256")
    if not value["requirements"] and not analysis.migratable:
        value["unresolved"].append("analysis rejected the program without a routed gap; diagnostics must resolve before generation")
    return _seal(value)


def validate_gap_taskbook(taskbook: Mapping) -> dict:
    """Check schema, content identity and GapSpec binding; not qualification."""
    if not isinstance(taskbook, Mapping):
        raise ValueError("taskbook must be an object")
    value = json.loads(_json(dict(taskbook)))
    if (value.get("schema") != SCHEMA or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("mode") not in {"gap_spec_requirements", "analysis_requirements"}):
        raise ValueError("unsupported taskbook schema")
    identity = value.pop("taskbook_id", None)
    if identity != "gap-taskbook-" + _canonical_digest(value):
        raise ValueError("taskbook content identity mismatch")
    value["taskbook_id"] = identity
    if value["mode"] == "gap_spec_requirements":
        from numpilot.gaps.models import GapSpec

        spec = GapSpec.from_dict(value["gap_spec"])
        if (value.get("gap_spec_sha256") != _canonical_digest(spec.to_dict())
                or value.get("source", {}).get("sha256") != spec.source_sha256
                or value.get("source", {}).get("path") != spec.source_path):
            raise ValueError("taskbook GapSpec/source binding mismatch")
        conflicts = _dtype_conflicts(spec, value["call_contract"]["callsites"])
        observed = value["call_contract"].get("observed_profiles")
        if observed:
            conflicts = list(conflicts)
            if value["call_contract"].get("observation_status") == "unsupported_dtype":
                conflicts.append(
                    "observed operand dtype is outside the float32 development scope"
                )
        if (value.get("generation") != {"allowed": not conflicts, "blockers": sorted(set(conflicts)),
                                       "claim_level": "unverified_candidate_only"}
                or value.get("status") != ("requirements_blocked" if conflicts else "requirements_prepared_unverified")):
            raise ValueError("taskbook generation scope does not match declared dtype requirements")
        _validate_observed_contract(value["call_contract"])
        if observed and value["call_contract"].get("observation_status") not in {
            "observed", "unsupported_dtype", "profile_missing",
            "nondeterministic", "observation_failed",
        }:
            raise ValueError("taskbook observation status is invalid")
    elif value.get("generation", {}).get("allowed") is not False:
        raise ValueError("analysis-only taskbook cannot authorize generation")
    if (value.get("support_scope", {}).get("input_dtype") != "float32"
            or value.get("acceptance", {}).get("status") != "pending"
            or value.get("acceptance", {}).get("planner_self_certification") is not False):
        raise ValueError("taskbook must remain a float32 request without qualification claims")
    return value


def render_gap_taskbook_markdown(taskbook: Mapping) -> str:
    """Render the same facts for a person; JSON is the machine-readable copy."""
    value = validate_gap_taskbook(taskbook)
    spec = value.get("gap_spec", {})

    def inline(item):
        return str(item).replace("`", "\\`").replace("\n", " ").replace("|", "\\|")

    lines = ["# 算子缺口生成任务书 / Operator gap taskbook", "",
             "- Target: `%s`" % inline(spec.get("qualified_api", "analysis requirements")),
             "- Status: `%s`" % value["status"], "- Taskbook ID: `%s`" % value["taskbook_id"],
             "- Source: `%s`" % inline(value["source"]["path"]),
             "- Source SHA-256: `%s`" % value["source"]["sha256"], "",
             "本任务书在模型调用前留存，可交给外部 Codex 或开发者继续。JSON 文件保存机器可读的同一份需求。",
             "Planner 整理需求；Gap Filler 生成候选；确定性验证执行验收。此任务书不是能力或性能证书。", "",
             "## 1. 缺口来源、源码位置与原因", ""]
    sites = spec.get("callsites", [])
    if spec:
        lines += ["- 接入目标：`%s` → `%s`。" % (inline(spec["qualified_api"]), inline(spec["asnumpy_api"])),
                  "- Gap ID：`%s`。" % spec["gap_id"],
                  "- 既有路线：`%s`；route_decisions 为已有决策，不能被任务书重新解释为已验证。" % inline(spec["route"])]
    else:
        for requirement in value["requirements"]:
            lines += ["- `%s`：%s；下一角色 `%s`。" % (inline(requirement["name"]), inline(requirement["reason"]), inline(requirement["next_role"]))]
            sites.extend(requirement["callsites"])
    lines += [""]
    for site in sites:
        lines += ["- 行 %s，列 %s：`%s`；原因：%s。" % (site["lineno"], site["col_offset"], inline(site.get("call_expr")), inline(site.get("reason", "见上述缺口原因")))]
    lines += ["", "## 2. 已知参数与未确认事实", "",
              "静态字面量只说明源码写了什么，不代表已经运行或取得张量 profile。", ""]
    calls = value.get("call_contract", {}).get("callsites", [])
    if not spec:
        calls = [site for requirement in value["requirements"] for site in requirement["callsites"]]
    for call in calls:
        lines += ["- 行 %s，列 %s：" % (call["lineno"], call["col_offset"])]
        args = call["arguments"]
        for index, arg in enumerate(args["positional"]):
            lines += ["  - 参数 %d：`%s`（%s）。" % (index, inline(arg.get("expression", "unknown")), arg["status"])]
        for arg in args["keywords"]:
            lines += ["  - `%s=%s`（%s）。" % (inline(arg["name"]), inline(arg.get("expression", "unknown")), arg["status"])]
        if args["status"] == "unknown":
            lines += ["  - 调用表达式不完整，需回到绑定源码确认参数。"]
    observed = value.get("call_contract", {}).get("observed_profiles") or []
    observation_status = value.get("call_contract", {}).get("observation_status", "not_requested")
    lines += ["", "- 未确认：%s。" % ", ".join(value["unresolved"])]
    if observed:
        lines += ["- 已观测运行时 profiles：`%s`。观测方式是在 CPU 上执行原 NumPy 程序并记录调用点实参，"
                  "源码 SHA-256 与任务书绑定。" % observation_status, "",
                  "| 调用点 | 参数名 | dtype | shape | rank | C 连续 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for entry in observed:
            site = "行 %s `%s`" % (entry["callsite"]["lineno"],
                                   inline(spec.get("qualified_api", "?")))
            for operand in entry["operands"]:
                lines += ["| %s | `%s` | `%s` | `%s` | %s | %s |" % (
                    site, inline(operand["name"]), inline(operand["dtype"]),
                    inline(list(operand["shape"])), operand["rank"], operand["c_contiguous"])]
            output = entry["output"]
            if output is not None:
                lines += ["| %s | 输出 | `%s` | `%s` | %s | %s |" % (
                    site, inline(output["dtype"]), inline(list(output["shape"])),
                    output["rank"], output["c_contiguous"])]
        lines += ["", "观测只覆盖 CPU 上实际执行到的调用点；设备区域在迁移验收前未被 AsNumpy 执行。"]
    else:
        if observation_status not in {"not_requested", None}:
            lines += ["- 已观测运行时 profiles：无（观测状态 `%s`）。不能据此声称已知输入，"
                      "重试观测或手工记录实际输入后重新提交。" % inline(observation_status)]
        else:
            lines += ["- 已观测运行时 profiles：无。请记录实际输入、输出和调用位置，再提交资格验证。"]
    lines += ["",
              "## 3. NumPy 语义", "",
              "语义来源：`%s`。" % inline(spec.get("semantic_source", "原始源码；GapSpec 尚未建立")), ""]
    lines += ["- " + item for item in value["semantics"]["checklist"]]
    dtype_line = ("- 输入 dtype：**float32**（已观测，见第 2 节）。" if observed
                  else "- 请求输入 dtype：**float32**。这是本批开发政策，不是已观测 dtype。")
    lines += ["", "## 4. 本批最小支持范围与拒绝范围", "",
              dtype_line,
              "- 输出 dtype/返回结构服从 NumPy oracle；比较接口可以返回 bool，不能强转为 float32。",
              "- 其他输入 dtype 显式拒绝或留待后续，不能静默 cast，也不能静默回退 CPU。",
              "- 调用签名和 shape/dtype 声明只引用原 GapSpec；未知 rank/shape/layout 不等于任意范围已支持。",
              "- 生成者须明确本候选支持哪些 profiles、拒绝哪些输入，并分别写正反测试。最终资格只覆盖实测范围。"]
    if observed:
        lines += ["- 本候选只需支持第 2 节已观测的输入 dtype/shape/rank/连续性组合；这是针对目标输入的"
                  "定向特化算子，范围外输入必须显式拒绝，不做泛化承诺。"]
    if observed:
        oracle_line = ("Oracle 状态：`%s`；输入已在 CPU 上由原程序实际执行并记录（见第 2 节），"
                       "设备区域仍须由确定性验证对拍。"
                       % value["acceptance"]["oracle"]["status"])
    else:
        oracle_line = ("Oracle 状态：`%s`；输入仍需从原程序/声明参考样例获取并落盘，Planner 没有执行源程序。"
                       % value["acceptance"]["oracle"]["status"])
    lines += ["",
              "## 5. 官方优先与源码接入边界", "",
              "- 目标 SoC：`%s`；CANN：`%s`。未提供时先补齐，不能自行猜测。" % (inline(value["integration"]["target_soc"]), inline(value["integration"]["cann_version"])),
              "- " + value["integration"]["strategy"], "- " + value["integration"]["source_boundary"],
              "- " + value["integration"]["custom_layout"],
              "- 必须记录公开 API → Python wrapper → pybind → native → ACLNN 两段式符号映射。",
              "- 接入依据引用：%s。" % "; ".join(value["integration"]["dependency_refs"]),
              "- 禁止隐式改系统 CANN/共享 AsNumpy、CPU fallback 或激活未验证包。", "",
              "## 6. 可执行 Oracle 与验收", "",
              oracle_line, ""]
    oracle_code = value["acceptance"]["oracle"].get("python")
    if oracle_code:
        lines += ["```python", oracle_code.rstrip(), "```", ""]
    lines += ["误差要求：" + value["acceptance"]["tolerance"]["required"], "",
              "| 门禁 | 必需证据 |", "| --- | --- |"]
    lines += ["| %s | %s |" % (gate["id"], inline(gate["requires"])) for gate in value["acceptance"]["gates"]]
    lines += ["", "所有门禁当前均 pending，由确定性验证执行；生成成功不等于测试通过。性能单独验收，不能从正确性推断加速比。", "",
              "## 交接与生成准入", "",
              "- 下一责任角色：`%s`；生成角色：`gap_filler`（配置模型、外部 Codex 或手工开发均可）。" % value["responsibility"]["next_role"],
              "- 允许生成未验证候选：`%s`。" % value["generation"]["allowed"],
              "- 生成阻塞项：%s。" % ("; ".join(value["generation"]["blockers"]) or "无；所有未知项仍阻塞资格发布"),
              "- 失败时保留任务书和错误证据，修改候选后重新验证；不得用 Planner 自认证替代测试。", ""]
    if spec:
        lines += ["## 唯一规范快照 / Authoritative GapSpec", "",
                  "JSON 中 `#/gap_spec/...` 均指向以下快照，避免维护第二份 shape/dtype/route 规则。", "",
                  "```json", _json(spec).decode().rstrip(), "```", ""]
    return "\n".join(lines)


def _existing(path: Path, expected: bytes) -> bool:
    if not os.path.lexists(path):
        return False
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise ValueError("taskbook artifact must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            actual = handle.read(_MAX_BYTES + 1)
        if actual != expected:
            raise ValueError("refusing to replace different taskbook content: %s" % path)
        return True
    finally:
        os.close(fd)


def persist_gap_taskbook(workspace, taskbook: Mapping) -> dict:
    """Write JSON and Markdown once; identical content is safe to reuse."""
    value = validate_gap_taskbook(taskbook)
    root = Path(workspace).absolute()
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError("taskbook workspace must not contain symlinks")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    payloads = {JSON_NAME: _json(value), MARKDOWN_NAME: render_gap_taskbook_markdown(value).encode("utf-8")}
    if any(len(payload) > _MAX_BYTES for payload in payloads.values()):
        raise ValueError("taskbook exceeds the size limit")
    existing = {name: _existing(root / name, payload) for name, payload in payloads.items()}
    for name, payload in payloads.items():
        if existing[name]:
            continue
        try:
            fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            _existing(root / name, payload)
            continue
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return {"schema": REF_SCHEMA, "schema_version": 1, "taskbook_id": value["taskbook_id"],
            "json": {"path": str(root / JSON_NAME), "sha256": _digest(payloads[JSON_NAME])},
            "markdown": {"path": str(root / MARKDOWN_NAME), "sha256": _digest(payloads[MARKDOWN_NAME])}}
