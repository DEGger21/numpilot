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
    receipt_path: str = ""

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
            "receipt_path": self.receipt_path,
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
            "reason": outcome.reason,
            "failure_reason": outcome.failure_reason,
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
    outcome.expected_route = expectation.get("route", "")
    outcome.expected_gap_kind = expectation.get("gap_kind", "")
    outcome.expected_source = expectation.get("source", "")
    outcome.verification_plan = list(expectation.get("verification", []))
    outcome.observed_outcome = _observed_outcome(outcome)
    outcome.verdict = _classify_verdict(outcome)
    outcome.receipt_path = "receipt.json"
    _write_receipt(case_out, outcome, expectation)
    return outcome


def run_suite(
    cases_root,
    out_dir,
    *,
    backend,
    engine,
    groups=None,
    limit=None,
    on_case=None,
    expectations=None,
):
    """Run every selected case and return the sanitized summary dict."""

    selected = set(groups) if groups else None
    cases = [
        case for case in discover_cases(cases_root)
        if selected is None or case[0] in selected
    ]
    if limit is not None:
        cases = cases[:limit]

    if expectations is None:
        expectations = load_expectations(cases_root)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    started = time.time()
    for index, (group, name, program) in enumerate(cases, 1):
        outcome = run_case(
            group,
            name,
            program,
            out_root=out_root,
            backend=backend,
            engine=engine,
            expectation=resolve_expectation(expectations, group, name),
        )
        outcomes.append(outcome)
        if on_case is not None:
            on_case(outcome, index, len(cases))

    summary = build_summary(outcomes, backend=backend, engine=engine)
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary["generated_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    write_summary(out_root, summary)
    return summary


def run_case(group, name, program, *, out_root, backend, engine, expectation=None):
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

    try:
        orchestrator = Orchestrator(
            backend=backend,
            engine=engine,
            threshold=1_000_000,
        )
        result = orchestrator.run_source(
            lowered.source,
            "program.py",
            out_dir=str(case_out),
            verification_source=source,
            strict_input_types=True,
        )
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
    return _finish(
        CaseOutcome(
            group=group,
            name=name,
            status=status,
            reason=_reason_for(result, blockers),
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


def build_summary(outcomes, *, backend, engine):
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

    return {
        "status": "pass" if unexpected == 0 else "incomplete",
        "backend": getattr(backend, "name", type(backend).__name__),
        "engine": getattr(engine, "name", type(engine).__name__),
        "total": len(outcomes),
        "expected_total": EXPECTED_TOTAL,
        "expected": expected,
        "improved": improved,
        "unexpected": unexpected,
        "status_counts": dict(sorted(status_counts.items())),
        "route_counts": route_counts,
        "group_counts": group_counts,
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
        "- cases: %d/%d (expected %d)" % (
            summary["total"], summary["expected_total"], summary["expected_total"],
        ),
        "- route honoured: %d · gap closed early: %d · unexpected: %d" % (
            summary["expected"], summary.get("improved", 0),
            summary["unexpected"],
        ),
        "- generated: %s" % summary.get("generated_at_utc", ""),
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
        help="execution backend (default: shim; npu requires audited runtime)",
    )
    parser.add_argument(
        "--engine", choices=("ast",), default="ast",
        help="migration engine for the deterministic sweep",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="optional group directory names to run",
    )
    parser.add_argument("--limit", type=int, default=None)
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

    backend = select_backend(args.backend)
    engine = AstRewriteEngine()
    if args.expectations is not None:
        expectations = load_expectations(Path(args.expectations).parent)
    else:
        expectations = load_expectations(args.root)
    summary = run_suite(
        args.root,
        args.out,
        backend=backend,
        engine=engine,
        groups=args.groups,
        limit=args.limit,
        expectations=expectations,
        on_case=lambda outcome, index, total: print(
            "[bench] %d/%d %s/%s -> %s (%s)" % (
                index, total, outcome.group, outcome.name, outcome.status,
                outcome.verdict or "legacy",
            ),
            file=sys.stderr,
        ),
    )
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
