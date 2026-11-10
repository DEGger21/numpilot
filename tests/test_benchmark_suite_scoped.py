"""Suite wiring and corpus boundaries for official callsite-scoped NPU runs.

Device execution and the scope verifier are isolated here; their independent
truth checks have dedicated tests. No NPU is used by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from numpilot import benchmark_suite as suite
from numpilot.backends.shim import ShimBackend
from numpilot.benchmark_overlay import BenchmarkOverlay
from numpilot.engines.ast_rewrite import AstRewriteEngine

CASE_ID = "I_gap_model/I006_concatenate_axis"
REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "benchmark/cases"
SOURCE = (CORPUS / CASE_ID / "program.py").read_text()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Backend:
    name = "npu"

    def __init__(self):
        self.identity = {"status": "audit_required", "identity_sha256": "b" * 64,
                         "audit_id": "unaudited-runtime", "issues": ["version requires audit"]}

    def runtime_identity(self):
        return SimpleNamespace(to_dict=lambda: dict(self.identity))

    def _raw_capability_manifest(self, operators):
        return SimpleNamespace(to_dict=lambda: {"operators": list(operators), "audit_id": "unaudited-runtime"})


def overlay(backend):
    return BenchmarkOverlay(
        qualification_id="official-aclnn-" + "1" * 20,
        overlay_sha256="a" * 64, runtime_identity_sha256="b" * 64,
        operators=("concatenate",), qualified_names=("numpy.concatenate",), resolver=backend,
        overlay_path="/bound/overlay.json", evidence_root="/bound/evidence",
    )


@pytest.fixture
def case_run(tmp_path, monkeypatch):
    backend = Backend()
    admitted = overlay(backend)
    program = tmp_path / "program.py"
    program.write_text(SOURCE)
    calls = []
    state = {"mutation": None, "scope_error": None}
    scope_ref = {"schema": "numpilot.scoped-runtime-qualification-ref", "schema_version": 1,
                 "identity_scope": "official_overlay_callsite", "runtime_identity_status": "audit_required",
                 "global_runtime_audit_promoted": False, "runtime_identity_sha256": "b" * 64,
                 "qualification_id": admitted.qualification_id, "sample_id": CASE_ID}

    class Orchestrator:
        def __init__(self, **kwargs):
            assert kwargs["backend"] is backend
            calls.append(("orchestrator", kwargs))

        def run_source(self, source, filename, **kwargs):
            generated = "# synthetic verified artifact for suite wiring\n"
            (Path(kwargs["out_dir"]) / "migrated.py").write_text(generated)
            result = {"status": "pass", "engine": "ast", "route": "direct", "rewrite_integrity": True,
                      "rewrites": {"concatenate": 1}, "expected_rewrites": {"concatenate": 1},
                      "source_sha256": sha(kwargs["verification_source"]), "analysis_source_sha256": sha(source),
                      "migrated_source": generated,
                      "report": SimpleNamespace(numerically_consistent=True),
                      "analysis": SimpleNamespace(migratable=True),
                      "validation_evidence": {"backend": "npu", "artifact_inputs_executed": True,
                                              "backend_provenance": {"strict_npu": True, "ok": True}}}
            if state["mutation"]:
                state["mutation"](backend, program, result)
            return result

    def build(evidence_dir, **kwargs):
        calls.append(("scope", Path(evidence_dir), kwargs))
        if state["scope_error"]:
            raise ValueError(state["scope_error"])
        return dict(scope_ref)

    from numpilot import scoped_qualification
    monkeypatch.setattr(scoped_qualification, "build_scoped_qualification", build)
    monkeypatch.setattr(suite, "Orchestrator", Orchestrator)

    def run():
        group, name = CASE_ID.split("/")
        return suite.run_case(group, name, program, out_root=tmp_path / "run", backend=backend,
                              engine=AstRewriteEngine(), overlay=admitted,
                              expectation={"route": "gap", "source": "case_override"})

    return SimpleNamespace(run=run, state=state, calls=calls, backend=backend, overlay=admitted,
                           program=program, root=tmp_path / "run", scope_ref=scope_ref)


def test_scoped_npu_case_persists_exact_evidence_without_promoting_global_identity(case_run):
    outcome = case_run.run()
    assert outcome.status == "migrated" and outcome.verdict == "improved"
    assert outcome.strict_npu_qualified is True and outcome.performance_qualified is None
    assert outcome.runtime_identity_status == "audit_required"
    assert outcome.runtime_qualification_scope == case_run.scope_ref
    scope_call = [call for call in case_run.calls if call[0] == "scope"][0]
    evidence, kwargs = scope_call[1:]
    assert set(path.name for path in evidence.iterdir()) == {
        "original.py", "analysis_source.py", "migrated.py", "profiles.json", "overlay-admission.json",
        "runtime-before.json", "runtime-after.json", "validation-evidence.json", "mainline-result.json",
        "base-manifest.json",
    }
    assert (evidence / "original.py").read_text() == SOURCE
    assert kwargs["source_sha256"] == sha(SOURCE)
    assert kwargs["expected_runtime_identity"] == case_run.backend.identity
    profiles = json.loads((evidence / "profiles.json").read_text())
    assert profiles["attempted_callsites"] == profiles["matched_callsites"] == 1
    assert profiles["profiles"]["callsites"][0]["profile"]["inputs"][0]["shape"] == [3, 4, 5]
    observed = json.loads((case_run.root / outcome.out_dir / "receipt.json").read_text())["observed"]
    assert observed["runtime_identity_status"] == "audit_required"
    assert observed["runtime_qualification_scope"] == case_run.scope_ref
    summary = suite.build_summary([outcome], backend=case_run.backend, engine=AstRewriteEngine(), overlay=case_run.overlay)
    assert summary["scoped_npu_qualified"] == 1 and summary["global_runtime_audit_promoted"] is False
    assert summary["overlay"]["execution"] == "strict_npu_mainline"
    markdown = suite._render_markdown(summary)
    assert "selected cases completed: 1" in markdown and "original corpus: 150" in markdown
    assert "scoped strict-NPU qualified: 1" in markdown and "audit_required" in markdown


def test_scope_rejection_cannot_be_reported_as_strict_success(case_run):
    case_run.state["scope_error"] = "mixed unqualified add call outside official overlay"
    outcome = case_run.run()
    assert outcome.status == "failed" and outcome.strict_npu_qualified is False
    assert outcome.runtime_qualification_scope is None and "mixed unqualified" in outcome.reason
    assert outcome.verdict == "unexpected"
    assert outcome.numerical_verified is True
    assert outcome.observed_outcome == "runtime_qualification_failure"
    observed = json.loads((case_run.root / outcome.out_dir / "receipt.json").read_text())["observed"]
    assert observed["outcome"] == "runtime_qualification_failure"
    assert observed["numerical_verified"] is True and observed["failure_reason"] is None


def test_wrong_initial_runtime_never_executes_mainline(case_run):
    case_run.backend.identity["identity_sha256"] = "c" * 64
    outcome = case_run.run()
    assert outcome.status == "error" and outcome.strict_npu_qualified is False
    assert not case_run.calls


@pytest.mark.parametrize("mutation", [
    lambda backend, program, result: backend.identity.update(identity_sha256="c" * 64),
    lambda backend, program, result: program.write_text(SOURCE + "\n# drift\n"),
])
def test_runtime_or_source_drift_prevents_scope_issue(case_run, mutation):
    case_run.state["mutation"] = mutation
    outcome = case_run.run()
    assert outcome.status == "failed" and outcome.strict_npu_qualified is False
    assert not any(call[0] == "scope" for call in case_run.calls)


def test_scoped_runtime_reference_is_forwarded_to_artifact_store(case_run):
    outcome = case_run.run()
    captured = []

    class Store:
        def put_runtime_qualification(self, sample, receipt):
            captured.append(receipt)
            return SimpleNamespace(artifact_id="d" * 64)

        def record_sample_without_operator(self, sample, **kwargs):
            return case_run.root / "operator-artifacts/samples/reference.json"

    suite._record_operator_ref(Store(), case_run.root, outcome, backend=case_run.backend,
                               runtime_identity=case_run.backend.identity)
    assert captured[0]["runtime_qualification_scope"] == case_run.scope_ref
    assert captured[0]["runtime_identity_status"] == "audit_required"
    ref = json.loads((case_run.root / outcome.out_dir / "runtime-qualification.json").read_text())
    assert ref["runtime_identity_status"] == "audit_required"
    assert ref["runtime_qualification_scope"] == case_run.scope_ref


@pytest.mark.parametrize("ids,match", [
    ([CASE_ID, CASE_ID], "duplicate"), (["I_gap_model/missing"], "unknown"),
    (["../program.py"], "exact group/name"), (["/absolute/path"], "exact group/name"),
    (["I_gap_model/../I006_concatenate_axis"], "exact group/name"), ([], "must not be empty"),
])
def test_case_selector_rejects_ambiguous_or_unsafe_ids(ids, match):
    with pytest.raises(ValueError, match=match):
        suite._select_case_ids(ids, suite.discover_cases(CORPUS))


def test_exact_case_selector_preserves_original_expectation(tmp_path):
    summary = suite.run_suite(CORPUS, tmp_path / "run", backend=ShimBackend(), engine=AstRewriteEngine(),
                              case_ids=[CASE_ID])
    assert summary["total"] == 1
    assert summary["cases"][0]["name"] == "I006_concatenate_axis"
    assert summary["cases"][0]["source_sha256"] == sha(SOURCE)
    assert summary["cases"][0]["expected_route"] == "gap"
    assert summary["cases"][0]["runtime_qualification_scope"] is None
    receipt = json.loads((tmp_path / "run/cases" / CASE_ID / "receipt.json").read_text())
    assert receipt["observed"]["runtime_qualification_id"]


def test_case_selector_empty_intersection_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="empty after"):
        suite.run_suite(CORPUS, tmp_path / "run", backend=ShimBackend(), engine=AstRewriteEngine(),
                        case_ids=[CASE_ID], groups=["A_direct_simple"])


def test_replacement_corpus_cannot_borrow_canonical_case_identity(tmp_path):
    program = tmp_path / CASE_ID / "program.py"
    program.parent.mkdir(parents=True)
    program.write_text(SOURCE + "\n# replacement\n")
    backend = Backend()
    with pytest.raises(ValueError, match="differs from the canonical"):
        suite.run_suite(tmp_path, tmp_path / "out", backend=backend, engine=AstRewriteEngine(),
                        overlay=overlay(backend), case_ids=[CASE_ID], expectations=suite.load_expectations(CORPUS))


def test_changed_expectations_cannot_mask_a_scoped_improvement(tmp_path):
    expectations = suite.load_expectations(CORPUS)
    expectations["defaults"]["verification"] = []
    backend = Backend()
    with pytest.raises(ValueError, match="unchanged benchmark expectations"):
        suite.run_suite(CORPUS, tmp_path / "run", backend=backend, engine=AstRewriteEngine(),
                        overlay=overlay(backend), case_ids=[CASE_ID], expectations=expectations)


def test_cli_npu_overlay_uses_qualified_resolver(monkeypatch, tmp_path, capsys):
    from numpilot import benchmark_overlay
    backend = Backend()
    admitted = overlay(backend)
    monkeypatch.setattr(benchmark_overlay, "load_benchmark_overlay", lambda *args: admitted)
    called = []

    def run(*args, **kwargs):
        called.append(kwargs)
        return {"status": "pass", "total": 0, "unexpected": 0}

    monkeypatch.setattr(suite, "run_suite", run)
    suite.main(["--backend", "npu", "--capability-overlay", "/bound/overlay.json",
                "--qualification-evidence-root", "/bound/evidence", "--cases", CASE_ID,
                "--out", str(tmp_path), "--json"])
    assert called[0]["backend"] is backend and called[0]["overlay"] is admitted
    assert called[0]["case_ids"] == [CASE_ID]
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_performance_catalog_receives_current_scoped_qualification(tmp_path, monkeypatch):
    backend = Backend()
    scope = {"identity_scope": "official_overlay_callsite", "receipt_sha256": "d" * 64}
    outcome = suite.CaseOutcome(group="I_gap_model", name="I006_concatenate_axis", status="migrated",
                                source_sha256=sha(SOURCE), numerical_verified=True,
                                strict_npu_qualified=True, runtime_qualification_scope=scope)
    observed = []

    class Catalog:
        def __init__(self, root):
            pass

        def resolve(self, sample_id, **kwargs):
            observed.append((sample_id, kwargs))
            return SimpleNamespace(qualified=False, ref=None)

    monkeypatch.setattr(suite, "PerformanceEvidenceCatalog", Catalog)
    monkeypatch.setattr(suite, "run_case", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(suite, "_record_operator_ref", lambda *args, **kwargs: None)
    summary = suite.run_suite(CORPUS, tmp_path / "run", backend=backend, engine=AstRewriteEngine(),
                              case_ids=[CASE_ID], overlay=overlay(backend),
                              performance_evidence_root=tmp_path / "performance")
    assert observed[0][0] == CASE_ID
    assert observed[0][1]["scoped_qualification"] == scope
    assert observed[0][1]["runtime_identity_sha256"] == "b" * 64
    assert summary["cases"][0]["performance_qualified"] is False
    assert summary["performance_qualified"] == 0


def test_cli_invalid_selector_reports_a_concise_argument_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        suite.main(["--backend", "shim", "--cases", "I_gap_model/missing", "--out", str(tmp_path)])
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "unknown case ID" in error and "Traceback" not in error


def _preflight_rejection(backend, program, result):
    result["status"] = "fail"
    result["report"] = SimpleNamespace(
        numerically_consistent=False, exception_phase="migrated.capabilities",
        exception_type="BackendCapabilityError",
        exception_message="base add API lacks qualified NPU provenance",
    )
    result["validation_evidence"]["artifact_inputs_executed"] = False
    result["validation_evidence"]["backend_provenance"] = {
        "backend": "npu", "strict_npu": False, "ok": False,
        "non_strict": ["add"], "runtime_execution_ok": False,
    }


@pytest.mark.parametrize("exception_type", ["BackendCapabilityError", "ValueError"])
def test_capability_preflight_rejection_is_qualification_failure_without_numerical_claim(case_run, exception_type):
    def mutation(backend, program, result):
        _preflight_rejection(backend, program, result)
        result["report"].exception_type = exception_type
        if exception_type == "ValueError":
            result["report"].exception_message = "runtime target snapshot is invalid"
    case_run.state["mutation"] = mutation
    outcome = case_run.run()
    assert outcome.status == "failed" and outcome.verdict == "unexpected"
    assert outcome.observed_outcome == "runtime_qualification_failure"
    assert outcome.qualification_failure_reason.startswith("runtime capability preflight rejected")
    assert outcome.numerical_verified is False and outcome.strict_npu_qualified is False
    assert outcome.runtime_qualification_scope is None and not outcome.artifact_inputs_executed
    assert not any(call[0] == "scope" for call in case_run.calls)
    summary = suite.build_summary([outcome], backend=case_run.backend, engine=AstRewriteEngine(), overlay=case_run.overlay)
    assert summary["status_counts"] == {"failed": 1} and summary["unexpected"] == 1
    assert summary["improved"] == 0 and summary["expected"] == 0
    observed = json.loads((case_run.root / outcome.out_dir / "receipt.json").read_text())["observed"]
    assert observed["outcome"] == "runtime_qualification_failure"
    assert observed["numerical_verified"] is False


@pytest.mark.parametrize("changed_field", ["phase", "exception", "execution", "provenance"])
def test_unrelated_failures_do_not_borrow_preflight_classification(case_run, changed_field):
    def mutation(backend, program, result):
        _preflight_rejection(backend, program, result)
        if changed_field == "phase":
            result["report"].exception_phase = "migrated.forward"
            result["report"].exception_type = "ValueError"
        elif changed_field == "exception":
            result["report"].exception_type = "TypeError"
        elif changed_field == "execution":
            result["validation_evidence"]["backend_provenance"]["runtime_execution_ok"] = True
        else:
            result["validation_evidence"]["backend_provenance"]["ok"] = True
    case_run.state["mutation"] = mutation
    outcome = case_run.run()
    assert outcome.status == "failed" and outcome.verdict == "unexpected"
    assert outcome.observed_outcome == "numerical_failure"
    assert outcome.qualification_failure_reason is None
