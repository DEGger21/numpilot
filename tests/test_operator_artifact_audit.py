import json
from pathlib import Path

from numpilot.operator_artifact_audit import audit_run, audit_store
from numpilot.operator_artifacts import OperatorArtifactStore, OperatorKey


def key():
    return OperatorKey("add_one", "float32 x+1", "custom", "Ascend910B4", "8.1", "a" * 64)


def test_audit_empty_no_operator_bindings(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    store.record_sample_without_operator("A_direct_simple/A001", status="not_applicable")
    report = audit_store(tmp_path / "run")
    assert report["ok"] is True
    assert report["counts"]["samples"] == 1


def test_audit_detects_candidate_source_tampering(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    ref = store.put_candidate(key(), "candidate", {"status": "generated_unverified"})
    Path(ref.source_path).write_text("tampered")
    report = audit_store(tmp_path / "run")
    assert report["ok"] is False
    assert any("source hash mismatch" in item for item in report["errors"])


def _make_run(tmp_path, *, count=2):
    run = tmp_path / "run"
    store = OperatorArtifactStore(run / "operator-artifacts")
    cases = run / "cases" / "A_direct_simple"
    refs = []
    for index in range(count):
        case = cases / f"A{index:03d}"
        case.mkdir(parents=True)
        sample_id = f"A_direct_simple/A{index:03d}"
        binding = store.record_sample_without_operator(sample_id, status="not_applicable")
        relative_binding = str(binding.relative_to(run))
        ref = {
            "schema": "numpilot.operator-sample-ref",
            "schema_version": 1,
            "sample_id": sample_id,
            "status": "not_applicable",
            "operator_store": "operator-artifacts",
            "sample_binding": relative_binding,
            "operator_id": None,
            "candidate_id": None,
            "qualification_id": None,
            "activation_id": None,
        }
        (case / "operator-ref.json").write_text(json.dumps(ref), encoding="utf-8")
        (case / "receipt.json").write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
        refs.append(ref)
    (run / "summary.json").write_text(
        json.dumps({
            "total": count,
            "operator_artifacts": {"sample_bindings": count},
        }),
        encoding="utf-8",
    )
    return run


def test_audit_run_cross_checks_refs_and_summary(tmp_path):
    run = _make_run(tmp_path)
    report = audit_run(run, expected_total=2)
    assert report["ok"] is True
    assert report["counts"] == {
        "operator_refs": 2,
        "sample_bindings": 2,
        "receipts": 2,
        "summary_total": 2,
    }


def test_audit_run_rejects_missing_ref_and_wrong_total(tmp_path):
    run = _make_run(tmp_path)
    ref_path = next((run / "cases").rglob("operator-ref.json"))
    ref_path.unlink()
    report = audit_run(run, expected_total=2)
    assert report["ok"] is False
    assert any("sample binding has no operator ref" in item for item in report["errors"])

    summary = json.loads((run / "summary.json").read_text())
    summary["total"] = 1
    (run / "summary.json").write_text(json.dumps(summary))
    report = audit_run(run, expected_total=2)
    assert any("expected total" in item for item in report["errors"])


def test_audit_run_detects_operator_ref_binding_mismatch(tmp_path):
    run = _make_run(tmp_path)
    ref_path = next((run / "cases").rglob("operator-ref.json"))
    payload = json.loads(ref_path.read_text())
    payload["status"] = "unsupported"
    ref_path.write_text(json.dumps(payload))
    report = audit_run(run, expected_total=2)
    assert report["ok"] is False
    assert any("ref/binding mismatch status" in item for item in report["errors"])


def test_audit_run_accepts_runtime_qualification_ref(tmp_path):
    run = _make_run(tmp_path, count=1)
    store = OperatorArtifactStore(run / "operator-artifacts")
    runtime = store.put_runtime_qualification(
        "A_direct_simple/A000",
        {
            "backend": "npu",
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": "b" * 64,
            "strict_npu_qualified": True,
            "numerical_verified": True,
            "performance_qualified": None,
        },
    )
    binding_path = next((run / "operator-artifacts" / "samples").glob("*.json"))
    binding = json.loads(binding_path.read_text())
    binding["runtime_qualification_id"] = runtime.artifact_id
    binding["strict_npu_qualified"] = True
    binding["performance_qualified"] = None
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    ref_path = next((run / "cases").rglob("operator-ref.json"))
    ref = json.loads(ref_path.read_text())
    ref["runtime_qualification_id"] = runtime.artifact_id
    ref["strict_npu_qualified"] = True
    ref["performance_qualified"] = None
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    report = audit_run(run, expected_total=1)
    assert report["ok"] is True
    assert report["store"]["counts"]["runtime_qualifications"] == 1



def test_audit_rejects_forged_performance_flag(tmp_path):
    run = _make_run(tmp_path, count=1)
    store = OperatorArtifactStore(run / "operator-artifacts")
    runtime = store.put_runtime_qualification(
        "A_direct_simple/A000",
        {
            "backend": "npu",
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": "b" * 64,
            "strict_npu_qualified": True,
            "numerical_verified": True,
            "performance_qualified": None,
        },
    )
    runtime_path = run / "operator-artifacts" / "runtime-qualifications" / f"{runtime.artifact_id}.json"
    payload = json.loads(runtime_path.read_text())
    payload["performance_qualified"] = True
    payload["receipt"]["performance_qualified"] = True
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")
    binding_path = next((run / "operator-artifacts" / "samples").glob("*.json"))
    binding = json.loads(binding_path.read_text())
    binding["runtime_qualification_id"] = runtime.artifact_id
    binding["strict_npu_qualified"] = True
    binding["performance_qualified"] = True
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    ref_path = next((run / "cases").rglob("operator-ref.json"))
    ref = json.loads(ref_path.read_text())
    ref["runtime_qualification_id"] = runtime.artifact_id
    ref["strict_npu_qualified"] = True
    ref["performance_qualified"] = True
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    report = audit_run(run, expected_total=1)
    assert report["ok"] is False
    assert any("performance evidence" in item for item in report["errors"])
