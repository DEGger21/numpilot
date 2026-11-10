"""Batch runner for the 150-case benchmark contract.

Each benchmark case is a Python module with ``build_inputs()`` returning a
``dict[str, numpy.ndarray]`` and ``forward(**inputs)`` returning named arrays.
The runner:

* statically refuses the ``J_out_of_subset`` group without executing it;
* lowers documented equivalent NumPy spellings for analysis/migration while
  keeping the original program as the verification oracle;
* runs the six-role orchestrator (Analyzer → ... → Verifier) per case;
* writes a sanitized, machine-readable summary with per-case evidence.

It never treats a gap-route stop as a pass: only numerical verification of the
original program against the migrated program counts as ``migrated``.

``--capability-overlay`` adds an overlay-scoped admission path: the official
qualification overlay must validate against the live qualified runtime, every
overlay callsite needs an exact tensor profile.  Shim runs verify semantics
with explicit stand-ins.  NPU runs use the admitted runtime and preserve a
hash-bound callsite qualification; they never promote the global runtime
audit or imply performance qualification.  An optional baseline API certificate
adds only independently verified exact profiles while preserving the official
overlay and global runtime identity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from numpilot.agents.analysis import AnalysisAgent
from numpilot.backends.base import select_backend
from numpilot.benchmark_lowering import lower_benchmark_program
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.orchestrator import Orchestrator
from numpilot.operator_artifacts import OperatorArtifactStore
from numpilot.performance_evidence import PerformanceEvidenceCatalog

REFUSAL_GROUP = "J_out_of_subset"
GROUP_ORDER = (
    "A_direct_simple",
    "B_direct_semantics",
    "C_direct_structure",
    "D1_partial_low_kernel",
    "D2_partial_low_official",
    "E_partial_high",
    "F_partial_consent",
    "G_partial_dedup",
    "H_gap_official",
    "I_gap_model",
    "J_out_of_subset",
    "K_gap_all",
)
EXPECTED_TOTAL = 150
SCOPE_SCHEMA = "numpilot.benchmark-scope"
SCOPE_VERSION = 1
SCOPE_CATEGORIES = (
    "covered_compute",
    "contract_limited",
    "single_gap",
    "official_gap",
    "multi_gap",
    "out_of_scope",
)


@dataclass
class CaseOutcome:
    group: str
    name: str
    status: str
    reason: str = ""
    route: str = ""
    engine: str = ""
    source_sha256: str = ""
    analysis_source_sha256: str = ""
    rewrites: dict = field(default_factory=dict)
    expected_rewrites: dict = field(default_factory=dict)
    rewrite_integrity: bool | None = None
    unsupported: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    lowering: list = field(default_factory=list)
    lowering_notes: list = field(default_factory=list)
    max_rel_diff: float | None = None
    numerically_consistent: bool | None = None
    comparisons: dict = field(default_factory=dict)
    failure_reason: str | None = None
    qualification_failure_reason: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    artifact_inputs_executed: bool = False
    out_dir: str = ""
    expected_route: str = ""
    expected_gap_kind: str = ""
    expected_source: str = ""
    verification_plan: list = field(default_factory=list)
    verdict: str = ""
    observed_outcome: str = ""
    analysis_admitted: bool | None = None
    migration_completed: bool = False
    numerical_verified: bool = False
    strict_npu_qualified: bool | None = None
    performance_qualified: bool | None = None
    performance_evidence: dict | None = None
    runtime_identity_status: str | None = None
    runtime_identity_sha256: str | None = None
    runtime_audit_id: str | None = None
    runtime_qualification_id: str = ""
    runtime_qualification_scope: dict | None = None
    receipt_path: str = ""
    operator_ref_path: str = ""
    overlay_ops: list = field(default_factory=list)
    overlay_profile_callsites: int = 0
    overlay_attempted_callsites: int = 0
    overlay_notes: list = field(default_factory=list)
    scope_name: str = ""
    scope_category: str = ""
    scope_source: str = ""
    scope_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "route": self.route,
            "engine": self.engine,
            "source_sha256": self.source_sha256,
            "analysis_source_sha256": self.analysis_source_sha256,
            "rewrites": dict(self.rewrites),
            "expected_rewrites": dict(self.expected_rewrites),
            "rewrite_integrity": self.rewrite_integrity,
            "unsupported": list(self.unsupported),
            "blockers": list(self.blockers),
            "lowering": list(self.lowering),
            "lowering_notes": list(self.lowering_notes),
            "max_rel_diff": self.max_rel_diff,
            "numerically_consistent": self.numerically_consistent,
            "comparisons": self.comparisons,
            "failure_reason": self.failure_reason,
            "qualification_failure_reason": self.qualification_failure_reason,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "artifact_inputs_executed": self.artifact_inputs_executed,
            "out_dir": self.out_dir,
            "expected_route": self.expected_route,
            "expected_gap_kind": self.expected_gap_kind,
            "expected_source": self.expected_source,
            "verification_plan": list(self.verification_plan),
            "verdict": self.verdict,
            "observed_outcome": self.observed_outcome,
            "analysis_admitted": self.analysis_admitted,
            "migration_completed": self.migration_completed,
            "numerical_verified": self.numerical_verified,
            "strict_npu_qualified": self.strict_npu_qualified,
            "performance_qualified": self.performance_qualified,
            "performance_evidence": self.performance_evidence,
            "runtime_identity_status": self.runtime_identity_status,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "runtime_audit_id": self.runtime_audit_id,
            "runtime_qualification_id": self.runtime_qualification_id,
            "runtime_qualification_scope": self.runtime_qualification_scope,
            "receipt_path": self.receipt_path,
            "operator_ref_path": self.operator_ref_path,
            "overlay_ops": list(self.overlay_ops),
            "overlay_profile_callsites": self.overlay_profile_callsites,
            "overlay_attempted_callsites": self.overlay_attempted_callsites,
            "overlay_notes": list(self.overlay_notes),
            "scope": {
                "name": self.scope_name,
                "category": self.scope_category,
                "source": self.scope_source,
                "rationale": self.scope_reason,
            },
        }


def discover_cases(cases_root: Path):
    root = Path(cases_root)
    if (root / "cases").is_dir():
        root = root / "cases"
    cases = []
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        for case_dir in sorted(group_dir.iterdir()):
            program = case_dir / "program.py"
            if program.is_file():
                cases.append((group_dir.name, case_dir.name, program))
    return cases


EXPECTATIONS_SCHEMA = "numpilot.benchmark-expectations"
RECEIPT_SCHEMA = "numpilot.benchmark-case-receipt"


def load_expectations(cases_root):
    """Load the case-route manifest next to (or above) the cases directory."""

    root = Path(cases_root)
    for candidate in (
        root / "expectations.json",
        root.parent / "expectations.json",
    ):
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("schema") != EXPECTATIONS_SCHEMA:
            raise ValueError(
                "unexpected benchmark expectations schema in %s" % candidate
            )
        payload["__path__"] = str(candidate)
        return payload
    return None


def load_scope(scope):
    """Load and validate an auditable view over the immutable case corpus."""

    if scope is None:
        return None
    if isinstance(scope, (str, Path)):
        path = Path(scope)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["__path__"] = str(path)
    elif isinstance(scope, dict):
        payload = dict(scope)
    else:
        raise TypeError("benchmark scope must be a path or mapping")
    if payload.get("schema") != SCOPE_SCHEMA:
        raise ValueError("unexpected benchmark scope schema")
    if payload.get("version") != SCOPE_VERSION:
        raise ValueError("unsupported benchmark scope version")
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise ValueError("benchmark scope name is required")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("benchmark scope groups must be a non-empty mapping")
    invalid = {
        group: category
        for group, category in groups.items()
        if category not in SCOPE_CATEGORIES
    }
    if invalid:
        raise ValueError("benchmark scope has invalid group categories: %r" % invalid)
    cases = payload.get("cases", {})
    if not isinstance(cases, dict):
        raise ValueError("benchmark scope cases must be a mapping")
    for case, entry in cases.items():
        if not isinstance(entry, dict) or entry.get("category") not in SCOPE_CATEGORIES:
            raise ValueError("benchmark scope case category is invalid: %s" % case)
    include = payload.get("include_categories")
    if include is None:
        include = [category for category in SCOPE_CATEGORIES if category != "out_of_scope"]
    if (
        not isinstance(include, list)
        or not include
        or any(category not in SCOPE_CATEGORIES for category in include)
    ):
        raise ValueError("benchmark scope include_categories is invalid")
    payload["include_categories"] = list(dict.fromkeys(include))
    return payload


def resolve_scope(scope, group, name):
    """Return the scope category binding for one discovered case."""

    if scope is None:
        return {}
    case_entry = scope.get("cases", {}).get(name, {})
    category = case_entry.get("category", scope.get("groups", {}).get(group))
    if category not in SCOPE_CATEGORIES:
        raise ValueError("benchmark scope has no category for %s/%s" % (group, name))
    return {
        "name": scope["name"],
        "category": category,
        "source": "case_override" if "category" in case_entry else "group_default",
        "rationale": case_entry.get(
            "rationale", scope.get("group_rationales", {}).get(group, "")
        ),
    }


def resolve_expectation(expectations, group, name):
    """Merge the group default with the per-case override for one case."""

    if not expectations:
        return {}
    default_verification = (
        expectations.get("defaults", {}).get("verification", [])
    )
    group_entry = expectations.get("groups", {}).get(group, {})
    case_entry = expectations.get("cases", {}).get(name, {})
    merged = {}
    source = ""
    for key in ("route", "gap_kind", "verification"):
        if key in case_entry:
            merged[key] = case_entry[key]
            source = "case_override"
        elif key in group_entry:
            merged[key] = group_entry[key]
            source = source or "group_default"
    merged["rationale"] = case_entry.get(
        "rationale", group_entry.get("rationale", "")
    )
    if "verification" not in merged:
        merged["verification"] = list(default_verification)
    merged["source"] = source or "missing"
    return merged


def _observed_outcome(outcome):
    if outcome.qualification_failure_reason:
        return "runtime_qualification_failure"
    return {
        "migrated": "migrated",
        "refused": "rejected",
        "unsupported": "gap_stop",
        "failed": "numerical_failure",
        "j_false_accept": "false_accept",
        "invalid_contract": "invalid_contract",
        "error": "error",
    }.get(outcome.status, "unknown")


def _classify_verdict(outcome):
    """Decide whether the observed outcome honours the audited route."""

    route = outcome.expected_route
    if not route:
        expected = (
            "refused" if outcome.group == REFUSAL_GROUP else "migrated"
        )
        return "honored" if outcome.status == expected else "unexpected"
    if route == "direct":
        return "honored" if outcome.status == "migrated" else "unexpected"
    if route == "gap":
        if outcome.status == "unsupported":
            return "honored"
        if outcome.status == "migrated":
            return "improved"
        return "unexpected"
    if route == "audit":
        # An audit expectation must never be counted as a direct migration;
        # stopping at the gap or failing numerical comparison is safe.
        return (
            "honored"
            if outcome.status in {"failed", "unsupported"}
            else "unexpected"
        )
    if route == "reject":
        return "honored" if outcome.status == "refused" else "unexpected"
    return "unexpected"


def _write_receipt(case_out, outcome, expectation):
    payload = {
        "schema": RECEIPT_SCHEMA,
        "version": 1,
        "group": outcome.group,
        "case": outcome.name,
        "scope": {
            "name": outcome.scope_name,
            "category": outcome.scope_category,
            "source": outcome.scope_source,
            "rationale": outcome.scope_reason,
        },
        "expected": {
            "route": outcome.expected_route,
            "gap_kind": outcome.expected_gap_kind,
            "source": outcome.expected_source,
            "verification": list(outcome.verification_plan),
            "rationale": expectation.get("rationale", ""),
        },
        "observed": {
            "status": outcome.status,
            "outcome": outcome.observed_outcome,
            "mainline_route": outcome.route,
            "analysis_admitted": outcome.analysis_admitted,
            "migration_completed": outcome.migration_completed,
            "numerical_verified": outcome.numerical_verified,
            "strict_npu_qualified": outcome.strict_npu_qualified,
            "performance_qualified": outcome.performance_qualified,
            "artifact_inputs_executed": outcome.artifact_inputs_executed,
            "overlay_ops": list(outcome.overlay_ops),
            "overlay_profile_callsites": outcome.overlay_profile_callsites,
            "overlay_attempted_callsites": outcome.overlay_attempted_callsites,
            "runtime_identity_status": outcome.runtime_identity_status,
            "runtime_identity_sha256": outcome.runtime_identity_sha256,
            "runtime_audit_id": outcome.runtime_audit_id,
            "runtime_qualification_id": outcome.runtime_qualification_id,
            "runtime_qualification_scope": outcome.runtime_qualification_scope,
            "overlay_notes": list(outcome.overlay_notes),
            "reason": outcome.reason,
            "failure_reason": outcome.failure_reason,
            "qualification_failure_reason": outcome.qualification_failure_reason,
            "max_rel_diff": outcome.max_rel_diff,
            "rewrites": dict(outcome.rewrites),
            "lowering": list(outcome.lowering),
            "lowering_notes": list(outcome.lowering_notes),
        },
        "artifacts": {
            "source_sha256": outcome.source_sha256,
            "analysis_source_sha256": outcome.analysis_source_sha256,
            "out_dir": outcome.out_dir,
        },
        "verdict": outcome.verdict,
    }
    (Path(case_out) / "receipt.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _finish(outcome, case_out, expectation):
    expectation = expectation or {}
    outcome.scope_name = expectation.get("scope_name", "")
    outcome.scope_category = expectation.get("scope_category", "")
    outcome.scope_source = expectation.get("scope_source", "")
    outcome.scope_reason = expectation.get("scope_reason", "")
    outcome.expected_route = expectation.get("route", "")
    outcome.expected_gap_kind = expectation.get("gap_kind", "")
    outcome.expected_source = expectation.get("source", "")
    outcome.verification_plan = list(expectation.get("verification", []))
    outcome.observed_outcome = _observed_outcome(outcome)
    outcome.verdict = _classify_verdict(outcome)
    outcome.receipt_path = "receipt.json"
    _write_receipt(case_out, outcome, expectation)
    return outcome


def _operator_ref_status(outcome):
    if outcome.status == "refused":
        return "rejected"
    if outcome.status == "unsupported":
        return "unsupported"
    if outcome.status in {"failed", "error", "invalid_contract"}:
        return "failed"
    return "not_applicable"


def _runtime_identity_snapshot(backend):
    if getattr(backend, "name", "") != "npu":
        return {
            "status": "not_applicable",
            "identity_sha256": None,
            "audit_id": None,
        }
    try:
        identity = backend.runtime_identity()
        payload = identity.to_dict() if hasattr(identity, "to_dict") else {}
        return {
            "status": payload.get("status"),
            "identity_sha256": payload.get("identity_sha256"),
            "audit_id": payload.get("audit_id"),
            "asnumpy_version": payload.get("asnumpy_version"),
            "cann_version": payload.get("cann_version"),
            "soc_model": payload.get("soc_model"),
            "python_version": payload.get("python_version"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "identity_sha256": None,
            "audit_id": type(exc).__name__,
        }


def _record_operator_ref(store, out_root, outcome, *, backend, runtime_identity):
    """Record case and runtime qualification references without inventing a candidate."""
    if not outcome.out_dir:
        return
    case_out = Path(out_root) / outcome.out_dir
    sample_id = "%s/%s" % (outcome.group, outcome.name)
    backend_name = getattr(backend, "name", type(backend).__name__)
    strict = outcome.strict_npu_qualified
    if backend_name != "npu":
        strict = None
    runtime_receipt = {
        "sample_id": sample_id,
        "source_sha256": outcome.source_sha256,
        "backend": backend_name,
        "runtime_identity_status": runtime_identity.get("status"),
        "runtime_identity_sha256": runtime_identity.get("identity_sha256"),
        "audit_id": runtime_identity.get("audit_id"),
        "asnumpy_version": runtime_identity.get("asnumpy_version"),
        "cann_version": runtime_identity.get("cann_version"),
        "soc_model": runtime_identity.get("soc_model"),
        "python_version": runtime_identity.get("python_version"),
        "strict_npu_qualified": strict,
        "numerical_verified": bool(outcome.numerical_verified),
        "performance_qualified": outcome.performance_qualified,
        "performance_evidence": outcome.performance_evidence,
        "runtime_qualification_scope": outcome.runtime_qualification_scope,
        "reason": outcome.reason or "runtime qualification gate completed",
    }
    runtime_ref = store.put_runtime_qualification(sample_id, runtime_receipt)
    outcome.runtime_identity_status = runtime_identity.get("status")
    outcome.runtime_identity_sha256 = runtime_identity.get("identity_sha256")
    outcome.runtime_audit_id = runtime_identity.get("audit_id")
    outcome.runtime_qualification_id = runtime_ref.artifact_id
    runtime_case_ref = {
        "schema": "numpilot.benchmark-runtime-qualification-ref",
        "schema_version": 1,
        "sample_id": sample_id,
        "runtime_qualification_id": runtime_ref.artifact_id,
        "store": "operator-artifacts",
        "receipt": str(
            Path("operator-artifacts")
            / "runtime-qualifications"
            / (runtime_ref.artifact_id + ".json")
        ),
        "strict_npu_qualified": strict,
        "numerical_verified": bool(outcome.numerical_verified),
        "performance_qualified": outcome.performance_qualified,
        "runtime_identity_status": outcome.runtime_identity_status,
        "runtime_identity_sha256": outcome.runtime_identity_sha256,
        "runtime_qualification_scope": outcome.runtime_qualification_scope,
    }
    case_out.mkdir(parents=True, exist_ok=True)
    (case_out / "runtime-qualification.json").write_text(
        json.dumps(runtime_case_ref, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    binding = store.record_sample_without_operator(
        sample_id,
        status=_operator_ref_status(outcome),
        route=outcome.route or outcome.expected_route,
        reason=outcome.reason or "no generated operator artifact in this run",
        runtime_qualification_id=runtime_ref.artifact_id,
        strict_npu_qualified=strict,
        performance_qualified=outcome.performance_qualified,
    )
    ref = {
        "schema": "numpilot.operator-sample-ref",
        "schema_version": 1,
        "sample_id": sample_id,
        "status": _operator_ref_status(outcome),
        "operator_store": "operator-artifacts",
        "sample_binding": str(binding.relative_to(Path(out_root))),
        "runtime_qualification_ref": "runtime-qualification.json",
        "runtime_qualification_id": runtime_ref.artifact_id,
        "strict_npu_qualified": strict,
        "performance_qualified": outcome.performance_qualified,
        "operator_id": None,
        "candidate_id": None,
        "qualification_id": None,
        "runtime_qualification_scope": outcome.runtime_qualification_scope,
        "activation_id": None,
        "reason": outcome.reason or "no generated operator artifact in this run",
    }
    (case_out / "operator-ref.json").write_text(
        json.dumps(ref, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outcome.operator_ref_path = "operator-ref.json"


def _overlay_description(overlay, backend):
    if overlay is None:
        return None
    description = overlay.to_dict()
    if getattr(backend, "name", "") == "npu":
        description = {**description, "execution": "strict_npu_mainline"}
    return description


def _select_case_ids(case_ids, discovered):
    if case_ids is None:
        return None
    selected = set()
    known = {group + "/" + name for group, name, _ in discovered}
    for value in case_ids:
        if (not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9_]+/[A-Za-z0-9_]+", value)):
            raise ValueError("case IDs must be exact group/name identifiers")
        if value in selected:
            raise ValueError("duplicate case ID: " + value)
        if value not in known:
            raise ValueError("unknown case ID: " + value)
        selected.add(value)
    if not selected:
        raise ValueError("case IDs must not be empty")
    return selected


def _validate_scoped_corpus(cases, expectations):
    """An official NPU benchmark claim uses the unchanged repository corpus."""
    corpus = Path(__file__).resolve().parent.parent / "benchmark/cases"
    for group, name, program in cases:
        original = corpus / group / name / "program.py"
        if not original.is_file() or original.read_bytes() != Path(program).read_bytes():
            raise ValueError("scoped NPU source differs from the canonical benchmark: " + group + "/" + name)
    canonical = load_expectations(corpus)
    actual = {key: value for key, value in (expectations or {}).items() if key != "__path__"}
    expected = {key: value for key, value in (canonical or {}).items() if key != "__path__"}
    if actual != expected:
        raise ValueError("scoped NPU qualification requires unchanged benchmark expectations")


def _persist_scoped_qualification(case_out, *, sample_id, source, analysis_source,
                                  result, profile_result, overlay, runtime_before,
                                  runtime_after, base_manifest):
    from numpilot.scoped_qualification import build_scoped_qualification

    if not overlay.overlay_path or not overlay.evidence_root:
        raise ValueError("official overlay has no bound evidence paths")
    generated = result.get("migrated_source")
    if not isinstance(generated, str) or not generated:
        raise ValueError("mainline generated source is unavailable")
    # Preserve the actual artifact executed by the verifier, not a reconstructed rewrite.
    if (Path(case_out) / "migrated.py").read_text(encoding="utf-8") != generated:
        raise ValueError("persisted migration does not match verified generated source")
    evidence_dir = Path(case_out) / "scoped-qualification"
    evidence_dir.mkdir(mode=0o700, exist_ok=False)

    def write(name, payload):
        with (evidence_dir / name).open("xb") as stream:
            stream.write(payload)
        (evidence_dir / name).chmod(0o600)

    for name, text in (("original.py", source), ("analysis_source.py", analysis_source),
                       ("migrated.py", generated)):
        write(name, text.encode("utf-8"))
    profiles = {
        "matched_callsites": profile_result.matched_callsites,
        "attempted_callsites": profile_result.attempted_callsites,
        "notes": list(profile_result.notes),
        "profiles": profile_result.profiles.to_dict() if profile_result.profiles else None,
    }
    payloads = {
        "profiles.json": profiles,
        "overlay-admission.json": _overlay_description(overlay, overlay.resolver),
        "runtime-before.json": runtime_before,
        "runtime-after.json": runtime_after,
        "base-manifest.json": base_manifest,
        "validation-evidence.json": result.get("validation_evidence") or {},
        "mainline-result.json": {key: result.get(key) for key in (
            "status", "engine", "rewrite_integrity", "rewrites", "expected_rewrites",
            "source_sha256", "analysis_source_sha256",
        )},
    }
    for name, value in payloads.items():
        write(name, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                allow_nan=False) + "\n").encode("utf-8"))
    baseline_kwargs = {}
    if overlay.baseline_admission is not None:
        baseline_kwargs = {
            "baseline_certificate_path": overlay.baseline_certificate_path,
            "baseline_evidence_root": overlay.baseline_evidence_root,
        }
    return build_scoped_qualification(
        evidence_dir, sample_id=sample_id,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        analysis_source_sha256=hashlib.sha256(analysis_source.encode("utf-8")).hexdigest(),
        overlay_path=overlay.overlay_path,
        overlay_evidence_root=overlay.evidence_root,
        expected_runtime_identity=runtime_after,
        **baseline_kwargs,
    )


def run_suite(
    cases_root,
    out_dir,
    *,
    backend,
    engine,
    groups=None,
    case_ids=None,
    limit=None,
    on_case=None,
    expectations=None,
    overlay=None,
    performance_evidence_root=None,
    scope=None,
    scope_categories=None,
):
    """Run every selected case and return the sanitized summary dict."""

    scope = load_scope(scope) if scope is not None else None
    if scope_categories is not None:
        if scope is None:
            raise ValueError("scope_categories requires a benchmark scope")
        categories = list(dict.fromkeys(scope_categories))
        if not categories or any(category not in SCOPE_CATEGORIES for category in categories):
            raise ValueError("scope_categories contains an unknown category")
        scope = dict(scope)
        scope["include_categories"] = categories
    selected = set(groups) if groups else None
    discovered = discover_cases(cases_root)
    exact_cases = _select_case_ids(case_ids, discovered)
    scope_entries = {}
    if scope is not None:
        for group, name, _program in discovered:
            scope_entries[(group, name)] = resolve_scope(scope, group, name)
        include_categories = set(scope["include_categories"])
    else:
        include_categories = None
    cases = [
        case for case in discovered
        if (selected is None or case[0] in selected)
        and (exact_cases is None or case[0] + "/" + case[1] in exact_cases)
        and (
            include_categories is None
            or scope_entries[(case[0], case[1])]["category"] in include_categories
        )
    ]
    if limit is not None:
        cases = cases[:limit]
    if exact_cases is not None and not cases:
        raise ValueError("case selection is empty after groups/scope/limit filters")

    if expectations is None:
        expectations = load_expectations(cases_root)
    if overlay is not None and getattr(backend, "name", "") == "npu":
        _validate_scoped_corpus(cases, expectations)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    artifact_store = OperatorArtifactStore(out_root / "operator-artifacts")
    runtime_identity = _runtime_identity_snapshot(backend)
    performance_catalog = (
        PerformanceEvidenceCatalog(performance_evidence_root)
        if performance_evidence_root is not None
        else None
    )
    outcomes = []
    started = time.time()
    for index, (group, name, program) in enumerate(cases, 1):
        expectation = resolve_expectation(expectations, group, name)
        if scope is not None:
            scope_entry = scope_entries[(group, name)]
            expectation = dict(expectation)
            expectation.update({
                "scope_name": scope_entry["name"],
                "scope_category": scope_entry["category"],
                "scope_source": scope_entry["source"],
                "scope_reason": scope_entry["rationale"],
            })
        outcome = run_case(
            group,
            name,
            program,
            out_root=out_root,
            backend=backend,
            engine=engine,
            expectation=expectation,
            overlay=overlay,
        )
        runtime_identity = _runtime_identity_snapshot(backend)
        if performance_catalog is not None:
            resolution = performance_catalog.resolve(
                "%s/%s" % (outcome.group, outcome.name),
                source_sha256=outcome.source_sha256,
                backend=getattr(backend, "name", type(backend).__name__),
                strict_npu_qualified=outcome.strict_npu_qualified,
                numerical_verified=outcome.numerical_verified,
                runtime_identity_sha256=runtime_identity.get("identity_sha256"),
                scoped_qualification=outcome.runtime_qualification_scope,
            )
            outcome.performance_qualified = resolution.qualified
            outcome.performance_evidence = resolution.ref
            if outcome.out_dir:
                _write_receipt(
                    out_root / outcome.out_dir,
                    outcome,
                    expectation,
                )
        _record_operator_ref(artifact_store, out_root, outcome, backend=backend, runtime_identity=runtime_identity)
        if outcome.out_dir:
            _write_receipt(out_root / outcome.out_dir, outcome, expectation)
        outcomes.append(outcome)
        if on_case is not None:
            on_case(outcome, index, len(cases))

    summary = build_summary(
        outcomes, backend=backend, engine=engine, overlay=overlay, scope=scope
    )
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary["generated_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    summary["operator_artifact_store"] = "operator-artifacts"
    summary["operator_artifacts"] = artifact_store.summary()
    write_summary(out_root, summary)
    return summary


def run_case(
    group, name, program, *, out_root, backend, engine, expectation=None,
    overlay=None,
):
    program = Path(program)
    try:
        source = program.read_text(encoding="utf-8")
    except OSError as exc:
        return CaseOutcome(
            group=group,
            name=name,
            status="error",
            reason="cannot read program.py",
            exception_type=type(exc).__name__,
            exception_message=_sanitize(exc),
        )

    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    case_out = out_root / "cases" / group / name
    case_out.mkdir(parents=True, exist_ok=True)
    relative_out = str(Path("cases") / group / name)

    structure_error = _contract_structure_error(source)
    if structure_error is not None:
        return _finish(
            CaseOutcome(
                group=group,
                name=name,
                status="invalid_contract",
                reason=structure_error,
                source_sha256=source_sha256,
                out_dir=relative_out,
            ),
            case_out,
            expectation,
        )

    if group == REFUSAL_GROUP:
        return _finish(
            _run_refusal_case(
                group, name, source, source_sha256, case_out, relative_out
            ),
            case_out,
            expectation,
        )

    lowered = lower_benchmark_program(source)
    (case_out / "analysis_source.py").write_text(lowered.source, encoding="utf-8")
    (case_out / "lowering.json").write_text(
        json.dumps(lowered.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    profiles = None
    profile_result = None
    runtime_before = None
    runtime_after = None
    base_manifest = None
    scoped_npu = overlay is not None and getattr(backend, "name", "") == "npu"
    overlay_ops = []
    overlay_notes = []
    overlay_profile_callsites = 0
    overlay_attempted_callsites = 0
    if overlay is not None:
        from numpilot.benchmark_overlay import build_analysis_profiles

        profile_result = build_analysis_profiles(lowered.source, overlay)
        profiles = profile_result.profiles
        overlay_ops = list(overlay.operators)
        overlay_notes = list(profile_result.notes)
        overlay_profile_callsites = profile_result.matched_callsites
        overlay_attempted_callsites = profile_result.attempted_callsites

    try:
        if scoped_npu:
            if backend is not overlay.resolver:
                raise ValueError("NPU overlay must execute with its qualified resolver")
            runtime_before = backend.runtime_identity().to_dict()
            base_manifest = backend._raw_capability_manifest(overlay.operators).to_dict()
            if runtime_before.get("identity_sha256") != overlay.runtime_identity_sha256:
                raise ValueError("live runtime identity does not match the official overlay")
        orchestrator = Orchestrator(
            backend=backend,
            engine=engine,
            threshold=1_000_000,
            analysis_profiles=profiles,
        )
        result = orchestrator.run_source(
            lowered.source,
            "program.py",
            out_dir=str(case_out),
            verification_source=source,
            strict_input_types=True,
        )
        if scoped_npu:
            runtime_after = backend.runtime_identity().to_dict()
    except Exception as exc:  # a single case must not abort the sweep
        return _finish(
            CaseOutcome(
                group=group,
                name=name,
                status="error",
                reason="orchestrator raised",
                source_sha256=source_sha256,
                analysis_source_sha256=hashlib.sha256(
                    lowered.source.encode("utf-8")
                ).hexdigest(),
                lowering=[item.to_dict() for item in lowered.rewrites],
                lowering_notes=list(lowered.notes),
                exception_type=type(exc).__name__,
                exception_message=_sanitize(exc),
                out_dir=relative_out,
                strict_npu_qualified=False if getattr(backend, "name", "") == "npu" else None,
                runtime_identity_status=(runtime_before or {}).get("status"),
                runtime_identity_sha256=(runtime_before or {}).get("identity_sha256"),
                runtime_audit_id=(runtime_before or {}).get("audit_id"),
                overlay_ops=overlay_ops,
                overlay_profile_callsites=overlay_profile_callsites,
                overlay_attempted_callsites=overlay_attempted_callsites,
                overlay_notes=[
                    _sanitize(note) for note in overlay_notes
                ],
            ),
            case_out,
            expectation,
        )

    report = result.get("report")
    status = _map_status(result.get("status"))
    analysis = result.get("analysis")
    blockers = _blockers(analysis)
    evidence = result.get("validation_evidence") or {}
    provenance = evidence.get("backend_provenance") or {}
    numerical_verified = bool(
        getattr(report, "numerically_consistent", False)
    )
    if getattr(backend, "name", "") == "npu":
        strict_npu_qualified = bool(
            status == "migrated"
            and provenance.get("strict_npu") is True
            and provenance.get("ok") is True
        )
    else:
        strict_npu_qualified = None
    qualification_scope = None
    scope_error = ""
    if (status == "failed" and getattr(backend, "name", "") == "npu"
            and getattr(report, "exception_phase", None) == "migrated.capabilities"
            and getattr(report, "exception_type", None) in ("BackendCapabilityError", "ValueError")
            and provenance.get("ok") is False
            and provenance.get("runtime_execution_ok") is False):
        scope_error = "runtime capability preflight rejected: " + _sanitize(
            getattr(report, "exception_message", None)
            or "required APIs lack qualified NPU provenance"
        )
    if scoped_npu and status == "migrated":
        try:
            if runtime_before != runtime_after:
                raise ValueError("runtime identity changed during scoped migration")
            if program.read_text(encoding="utf-8") != source:
                raise ValueError("original benchmark source changed during execution")
            qualification_scope = _persist_scoped_qualification(
                case_out, sample_id="%s/%s" % (group, name),
                source=source, analysis_source=lowered.source, result=result,
                profile_result=profile_result, overlay=overlay,
                runtime_before=runtime_before, runtime_after=runtime_after,
                base_manifest=base_manifest,
            )
            strict_npu_qualified = True
        except Exception as exc:
            strict_npu_qualified = False
            qualification_scope = None
            status = "failed"
            scope_error = "scoped NPU qualification failed: " + _sanitize(exc)
    return _finish(
        CaseOutcome(
            group=group,
            name=name,
            status=status,
            reason=scope_error or _reason_for(result, blockers),
            route=result.get("route") or "",
            engine=result.get("engine") or "",
            source_sha256=source_sha256,
            analysis_source_sha256=result.get("analysis_source_sha256", ""),
            rewrites=dict(result.get("rewrites") or {}),
            expected_rewrites=dict(result.get("expected_rewrites") or {}),
            rewrite_integrity=result.get("rewrite_integrity"),
            unsupported=list(result.get("unsupported") or []),
            blockers=blockers,
            lowering=[item.to_dict() for item in lowered.rewrites],
            lowering_notes=list(lowered.notes),
            max_rel_diff=getattr(report, "max_rel_diff", None),
            numerically_consistent=getattr(
                report, "numerically_consistent", None
            ),
            comparisons=evidence.get("comparisons") or {},
            failure_reason=getattr(report, "failure_reason", None),
            qualification_failure_reason=scope_error or None,
            exception_type=getattr(report, "exception_type", None),
            exception_message=_sanitize(
                getattr(report, "exception_message", None) or ""
            ) or None,
            artifact_inputs_executed=bool(
                evidence.get("artifact_inputs_executed", False)
            ),
            out_dir=relative_out,
            analysis_admitted=(
                bool(analysis.migratable)
                if analysis is not None else None
            ),
            migration_completed=status in {"migrated", "failed"},
            numerical_verified=numerical_verified,
            strict_npu_qualified=strict_npu_qualified,
            performance_qualified=None,
            runtime_qualification_scope=qualification_scope,
            runtime_identity_status=(runtime_after or {}).get("status"),
            runtime_identity_sha256=(runtime_after or {}).get("identity_sha256"),
            runtime_audit_id=(runtime_after or {}).get("audit_id"),
            overlay_ops=overlay_ops,
            overlay_profile_callsites=overlay_profile_callsites,
            overlay_attempted_callsites=overlay_attempted_callsites,
            overlay_notes=[_sanitize(note) for note in overlay_notes],
        ),
        case_out,
        expectation,
    )


def _run_refusal_case(group, name, source, source_sha256, case_out, relative_out):
    """J cases are never executed; static refusal is the expected outcome."""

    try:
        analysis = AnalysisAgent().analyze(
            source,
            "program.py",
            strict_input_types=True,
        )
    except Exception as exc:
        return CaseOutcome(
            group=group,
            name=name,
            status="error",
            reason="refusal analysis raised",
            source_sha256=source_sha256,
            exception_type=type(exc).__name__,
            exception_message=_sanitize(exc),
            out_dir=relative_out,
        )

    blockers = _blockers(analysis)
    admitted = sorted({op.name for op in analysis.op_uses if op.supported})
    refused = not analysis.migratable
    status = "refused" if refused else "j_false_accept"
    reason = (
        "; ".join(blockers)
        if refused and blockers
        else "static analysis admitted a refusal-only program: "
        + ", ".join(admitted)
        if not refused
        else "no supported compute surface admitted"
    )
    payload = {
        "group": group,
        "name": name,
        "status": status,
        "migratable": analysis.migratable,
        "admitted_ops": admitted,
        "unsupported": list(analysis.unsupported),
        "blockers": blockers,
        "source_sha256": source_sha256,
        "execution_performed": False,
    }
    (case_out / "refusal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return CaseOutcome(
        group=group,
        name=name,
        status=status,
        reason=reason,
        source_sha256=source_sha256,
        analysis_source_sha256=source_sha256,
        unsupported=list(analysis.unsupported),
        blockers=blockers,
        out_dir=relative_out,
        analysis_admitted=bool(analysis.migratable),
        migration_completed=False,
        numerical_verified=False,
        strict_npu_qualified=False,
        performance_qualified=None,
    )


def _contract_structure_error(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return "syntax error: %s" % exc
    builds = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_inputs"
    ]
    forwards = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(builds) != 1:
        return "contract requires exactly one build_inputs, found %d" % len(builds)
    if len(forwards) != 1:
        return "contract requires exactly one forward, found %d" % len(forwards)
    imports_numpy = any(
        isinstance(node, ast.Import)
        and any(alias.name == "numpy" for alias in node.names)
        for node in tree.body
    )
    if not imports_numpy:
        return "contract requires a top-level import numpy"
    return None


def _map_status(status):
    return {
        "pass": "migrated",
        "unsupported": "unsupported",
        "fail": "failed",
    }.get(status, "unknown")


def _blockers(analysis):
    if analysis is None:
        return []
    blockers = []
    seen = set()
    for op in getattr(analysis, "op_uses", ()):
        if getattr(op, "supported", True):
            continue
        text = "%s@L%s: %s" % (
            getattr(op, "name", "?"),
            getattr(op, "lineno", 0),
            getattr(op, "reason", "") or "blocked",
        )
        if text not in seen:
            seen.add(text)
            blockers.append(_sanitize(text))
    return blockers


def _reason_for(result, blockers):
    if result.get("status") == "unsupported":
        return "; ".join(blockers) or "gap route: unsupported"
    report = result.get("report")
    reason = getattr(report, "failure_reason", None)
    if reason:
        return _sanitize(reason)
    if result.get("status") == "pass":
        return "numerically verified against the original NumPy program"
    return ""


_PRIVATE_HOME_RE = re.compile(r"/(?:home|Users)/[^/\s]+(?:/work)?(?=/|\b)")


def _sanitize(value):
    """Redact user home directories without embedding a machine identity."""

    return _PRIVATE_HOME_RE.sub("<private>", str(value))


def build_summary(outcomes, *, backend, engine, overlay=None, scope=None):
    group_counts = {}
    for outcome in outcomes:
        counts = group_counts.setdefault(outcome.group, {
            "total": 0, "migrated": 0, "refused": 0, "unsupported": 0,
            "failed": 0, "error": 0, "j_false_accept": 0,
            "invalid_contract": 0, "unknown": 0,
        })
        counts["total"] += 1
        counts[outcome.status] = counts.get(outcome.status, 0) + 1

    status_counts = {}
    for outcome in outcomes:
        status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1

    expected = 0
    improved = 0
    unexpected = 0
    route_counts = {}
    for outcome in outcomes:
        route = outcome.expected_route or "<legacy>"
        counts = route_counts.setdefault(route, {
            "total": 0, "honored": 0, "improved": 0, "unexpected": 0,
        })
        counts["total"] += 1
        if outcome.verdict == "honored":
            expected += 1
            counts["honored"] += 1
        elif outcome.verdict == "improved":
            improved += 1
            counts["improved"] += 1
        else:
            unexpected += 1
            counts["unexpected"] += 1

    scope_summary = None
    if scope is not None:
        category_counts = {}
        for outcome in outcomes:
            category = outcome.scope_category or "<unknown>"
            category_counts[category] = category_counts.get(category, 0) + 1
        scope_summary = {
            "schema": scope["schema"],
            "version": scope["version"],
            "name": scope["name"],
            "path": scope.get("__path__"),
            "include_categories": list(scope["include_categories"]),
            "category_counts": dict(sorted(category_counts.items())),
        }
    return {
        "status": "pass" if unexpected == 0 else "incomplete",
        "backend": getattr(backend, "name", type(backend).__name__),
        "engine": getattr(engine, "name", type(engine).__name__),
        "overlay": _overlay_description(overlay, backend),
        "scoped_npu_qualified": sum(
            outcome.strict_npu_qualified is True and outcome.runtime_qualification_scope is not None
            for outcome in outcomes
        ),
        "runtime_identity_statuses": sorted({
            outcome.runtime_identity_status for outcome in outcomes if outcome.runtime_identity_status
        }),
        "global_runtime_audit_promoted": False,
        "scope": scope_summary,
        "total": len(outcomes),
        "selected_total": len(outcomes),
        "corpus_expected_total": EXPECTED_TOTAL,
        "expected_total": len(outcomes) if scope is not None else EXPECTED_TOTAL,
        "expected": expected,
        "improved": improved,
        "unexpected": unexpected,
        "status_counts": dict(sorted(status_counts.items())),
        "route_counts": route_counts,
        "group_counts": group_counts,
        "performance_qualified": sum(
            outcome.performance_qualified is True for outcome in outcomes
        ),
        "cases": [outcome.to_dict() for outcome in outcomes],
    }


def write_summary(out_root, summary):
    out_root = Path(out_root)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = [
        "group\tcase\tstatus\texpected_route\tverdict\troute\trewrites"
        "\tunsupported\treason",
    ]
    for case in summary["cases"]:
        rows.append("\t".join([
            case["group"],
            case["name"],
            case["status"],
            case["expected_route"],
            case["verdict"],
            case["route"],
            ",".join(sorted((case["rewrites"] or {}).keys())),
            ",".join(case["unsupported"]),
            (case["reason"] or "").replace("\t", " ").replace("\n", " "),
        ]))
    (out_root / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_root / "SUMMARY.md").write_text(_render_markdown(summary), encoding="utf-8")


def _render_markdown(summary):
    lines = [
        "# NumPilot benchmark suite",
        "",
        "- status: `%s`" % summary["status"],
        "- backend: `%s` · engine: `%s`" % (summary["backend"], summary["engine"]),
        "- selected cases completed: %d · original corpus: %d cases" % (
            summary["total"], summary.get("corpus_expected_total", EXPECTED_TOTAL),
        ),
        "- scoped strict-NPU qualified: %d · global runtime status: %s" % (
            summary.get("scoped_npu_qualified", 0),
            ", ".join("`%s`" % value for value in summary.get("runtime_identity_statuses", [])) or "not recorded",
        ),
        "- global runtime audit promoted: no · performance qualified: %d" % summary.get("performance_qualified", 0),
        "- route honoured: %d · gap closed early: %d · unexpected: %d" % (
            summary["expected"], summary.get("improved", 0),
            summary["unexpected"],
        ),
        "- generated: %s" % summary.get("generated_at_utc", ""),
        "- operator artifacts: %s" % json.dumps(summary.get("operator_artifacts", {}), ensure_ascii=False, sort_keys=True),
        "",
        "## Expected-route conformance",
        "",
        "| route | total | honoured | improved | unexpected |",
        "|---|---:|---:|---:|---:|",
    ]
    for route in ("direct", "gap", "audit", "reject", "<legacy>"):
        counts = summary.get("route_counts", {}).get(route)
        if not counts:
            continue
        lines.append(
            "| %s | %d | %d | %d | %d |" % (
                route, counts["total"], counts["honored"],
                counts["improved"], counts["unexpected"],
            )
        )
    lines.extend([
        "",
        "## Group results",
        "",
        "| group | total | migrated | refused | unsupported | failed | error | false accept |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group in GROUP_ORDER + tuple(
        sorted(set(summary["group_counts"]) - set(GROUP_ORDER))
    ):
        counts = summary["group_counts"].get(group)
        if not counts:
            continue
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %d | %d |" % (
                group, counts["total"], counts["migrated"], counts["refused"],
                counts["unsupported"], counts["failed"], counts["error"],
                counts["j_false_accept"],
            )
        )
    lines.extend(["", "## Cases that are not migrated", ""])
    lines.append("| group | case | status | expected | verdict | reason |")
    lines.append("|---|---|---|---|---|---|")
    for case in summary["cases"]:
        if case["status"] == "migrated":
            continue
        reason = (case["reason"] or "").replace("|", "\\|")
        lines.append(
            "| %s | %s | %s | %s | %s | %s |" % (
                case["group"], case["name"], case["status"],
                case["expected_route"] or "-", case["verdict"] or "-", reason,
            )
        )
    lines.extend([
        "",
        "A `direct` expectation is honoured only by a numerically verified",
        "migration. A `gap` expectation is honoured by a clean gap stop and",
        "recorded as `improved` when the gap was closed by a documented",
        "equivalence. An `audit` expectation must never be counted as a direct",
        "migration. `refused` is the expected static outcome for",
        "`J_out_of_subset`; those programs are never executed. Every case also",
        "writes `receipt.json` with its expected route, verification plan,",
        "observed dimensions and artifact hashes.",
        "",
    ])
    return "\n".join(lines)


def default_cases_root():
    env = os.environ.get("NUMPILOT_BENCHMARK_CASES")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "benchmark" / "cases"
    if candidate.is_dir():
        return candidate
    return Path.cwd() / "benchmark" / "cases"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the benchmark contract through NumPilot and write evidence.",
    )
    parser.add_argument(
        "--root",
        default=str(default_cases_root()),
        help="benchmark cases root (default: repository benchmark/cases)",
    )
    parser.add_argument("--out", required=True, help="evidence output directory")
    parser.add_argument(
        "--backend", choices=("shim", "npu"), default="shim",
        help="execution backend (default: shim; npu requires audited runtime or exact official overlay scope)",
    )
    parser.add_argument(
        "--engine", choices=("ast",), default="ast",
        help="migration engine for the deterministic sweep",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="optional group directory names to run",
    )
    parser.add_argument("--cases", nargs="+", default=None,
                        help="exact group/name case IDs; intersected with groups and scope")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scope",
        default=None,
        help="auditable benchmark scope manifest (default: all discovered cases)",
    )
    parser.add_argument(
        "--scope-categories",
        nargs="+",
        choices=SCOPE_CATEGORIES,
        default=None,
        help="run only selected categories from --scope",
    )
    parser.add_argument(
        "--capability-overlay",
        default=None,
        help="official capability overlay JSON (shim semantics or exact scoped NPU qualification)",
    )
    parser.add_argument(
        "--qualification-evidence-root",
        default=None,
        help="evidence root that authenticates --capability-overlay",
    )
    parser.add_argument(
        "--baseline-api-certificate", default=None,
        help="independent baseline API/profile certificate; requires NPU plus official overlay",
    )
    parser.add_argument(
        "--baseline-api-evidence-root", default=None,
        help="evidence root authenticating the baseline API certificate",
    )
    parser.add_argument(
        "--performance-evidence-root",
        default=None,
        help="manifest root of hash-bound per-case script benchmark receipts",
    )
    parser.add_argument(
        "--expectations",
        default=None,
        help="route manifest (default: benchmark/expectations.json)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="write the machine summary to stdout as well",
    )
    args = parser.parse_args(argv)

    if bool(args.capability_overlay) != bool(args.qualification_evidence_root):
        parser.error(
            "--capability-overlay and --qualification-evidence-root "
            "must be supplied together"
        )
    if bool(args.baseline_api_certificate) != bool(args.baseline_api_evidence_root):
        parser.error("--baseline-api-certificate and --baseline-api-evidence-root must be supplied together")
    if args.baseline_api_certificate and (not args.capability_overlay or args.backend != "npu"):
        parser.error("baseline API admission requires --backend npu and --capability-overlay")
    overlay = None
    if args.capability_overlay:
        from numpilot.backends.shim import ShimBackend
        from numpilot.benchmark_overlay import load_benchmark_overlay

        try:
            baseline_kwargs = {}
            if args.baseline_api_certificate:
                baseline_kwargs = {
                    "baseline_api_certificate": args.baseline_api_certificate,
                    "baseline_api_evidence_root": args.baseline_api_evidence_root,
                }
            overlay = load_benchmark_overlay(
                args.capability_overlay,
                args.qualification_evidence_root,
                **baseline_kwargs,
            )
        except Exception as exc:
            parser.error(
                "cannot admit capability overlay: %s" % _sanitize(exc)
            )
        backend = (overlay.resolver if args.backend == "npu" else ShimBackend(
            extra_ops=overlay.operators,
            analysis_profile_resolver=overlay.resolver,
        ))
    else:
        backend = select_backend(args.backend)
    engine = AstRewriteEngine()
    if args.expectations is not None:
        expectations = load_expectations(Path(args.expectations).parent)
    else:
        expectations = load_expectations(args.root)
    if args.scope_categories and args.scope is None:
        parser.error("--scope-categories requires --scope")
    try:
        scope = load_scope(args.scope) if args.scope is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error("cannot load benchmark scope: %s" % _sanitize(exc))
    try:
        summary = run_suite(
            args.root,
            args.out,
            backend=backend,
            engine=engine,
            groups=args.groups,
            case_ids=args.cases,
            limit=args.limit,
            expectations=expectations,
            overlay=overlay,
            performance_evidence_root=args.performance_evidence_root,
            scope=scope,
            scope_categories=args.scope_categories,
            on_case=lambda outcome, index, total: print(
                "[bench] %d/%d %s/%s -> %s (%s)" % (
                    index, total, outcome.group, outcome.name, outcome.status,
                    outcome.verdict or "legacy",
                ),
                file=sys.stderr,
            ),
        )
    except (ValueError, OSError) as exc:
        parser.error("cannot run benchmark selection: %s" % _sanitize(exc))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(
            "[bench] status=%s total=%d honored=%d improved=%d "
            "unexpected=%d counts=%s"
            % (
                summary["status"], summary["total"], summary["expected"],
                summary.get("improved", 0), summary["unexpected"],
                summary["status_counts"],
            )
        )
        print("[bench] summary=%s" % (Path(args.out) / "summary.json"))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
