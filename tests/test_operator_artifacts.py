from pathlib import Path

import pytest

from numpilot.operator_artifacts import (
    OperatorArtifactStore,
    OperatorArtifactStoreError,
    OperatorKey,
)


def _key():
    return OperatorKey(
        operator_name="add_one",
        semantic_signature="float32 contiguous rank>=1 x+1",
        gap_kind="custom_operator",
        target_soc="Ascend910B4",
        cann_version="8.1.RC1",
        runtime_identity_sha256="a" * 64,
    )


def test_store_preserves_candidate_qualification_activation_and_sample_binding(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    key = _key()
    candidate = store.put_candidate(
        key,
        "// Ascend C candidate\n",
        {
            "status": "generated_unverified",
            "model": "configured-model",
            "compiled": False,
            "activated": False,
            "artifact_id": "draft-abc123",
        },
    )
    assert candidate.kind == "candidate"
    assert candidate.source_artifact_id == "draft-abc123"
    assert Path(candidate.source_path).read_text() == "// Ascend C candidate\n"

    # Replaying the same candidate is content-addressed and does not create a
    # second operator record.
    replay = store.put_candidate(
        key,
        "// Ascend C candidate\n",
        {"status": "generated_unverified", "model": "configured-model"},
    )
    assert replay.artifact_id != candidate.artifact_id  # receipt is part of provenance

    qualification = store.put_qualification(
        candidate.operator_id,
        candidate.artifact_id,
        {
            "candidate_id": candidate.artifact_id,
            "qualification_id": "qualification-legacy",
            "status": "qualified",
            "interface_available": True,
            "compiled": True,
            "numerically_verified": True,
        },
    )
    activation = store.put_activation(
        candidate.operator_id,
        qualification.artifact_id,
        {
            "qualification_id": qualification.artifact_id,
            "activation_id": "activation-legacy",
            "status": "activated",
            "activated": True,
            "upstream_modified": False,
        },
    )
    binding = store.bind_sample(
        "B014/add-one",
        candidate.operator_id,
        candidate_id=candidate.artifact_id,
        qualification_id=qualification.artifact_id,
        activation_id=activation.artifact_id,
        status="activated",
        callsite="program.py:4",
    )
    payload = __import__("json").loads(binding.read_text())
    assert payload["sample_id"] == "B014/add-one"
    assert payload["status"] == "activated"
    assert payload["activation_id"] == activation.artifact_id
    summary = store.summary()
    assert summary["sample_bindings"] == 1
    assert summary["unique_operators"] == 1
    assert summary["statuses"] == {"activated": 1}
    assert qualification.source_artifact_id == "qualification-legacy"
    assert activation.source_artifact_id == "activation-legacy"
    verification = store.verify()
    assert verification["ok"] is True
    assert verification["sample_bindings"] == 1
    gc = store.gc_dry_run()
    assert any(item["artifact_id"] == replay.artifact_id for item in gc["orphaned"])
    assert gc["delete_performed"] is False
    assert (tmp_path / "run" / "store.json").is_file()


def test_store_enforces_dependency_order_and_activation_boundary(tmp_path):
    store = OperatorArtifactStore(tmp_path)
    key = _key()
    candidate = store.put_candidate(key, "candidate", {"status": "generated_unverified"})
    with pytest.raises(OperatorArtifactStoreError, match="artifact receipt is missing"):
        store.put_activation(candidate.operator_id, "b" * 64, {"status": "activated", "activated": True, "upstream_modified": False})
    failed = store.put_qualification(
        candidate.operator_id,
        candidate.artifact_id,
        {"status": "failed", "interface_available": False},
    )
    with pytest.raises(OperatorArtifactStoreError, match="requires a qualified"):
        store.put_activation(candidate.operator_id, failed.artifact_id, {"status": "activated", "activated": True, "upstream_modified": False})


def test_store_rejects_sensitive_receipt_fields(tmp_path):
    store = OperatorArtifactStore(tmp_path)
    with pytest.raises(OperatorArtifactStoreError, match="sensitive field"):
        store.put_candidate(_key(), "candidate", {"status": "generated_unverified", "api_token": "secret"})


def test_store_verify_and_gc_report_unreferenced_candidate_without_deleting(tmp_path):
    store = OperatorArtifactStore(tmp_path)
    candidate = store.put_candidate(_key(), "candidate", {"status": "generated_unverified"})
    verification = store.verify()
    assert verification["ok"] is True
    gc = store.gc_dry_run()
    assert gc["dry_run"] is True
    assert gc["delete_performed"] is False
    assert any(item["artifact_id"] == candidate.artifact_id for item in gc["orphaned"])


def test_store_records_runtime_qualification_without_fake_operator(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    failed = store.put_runtime_qualification(
        "A_direct_simple/A001",
        {
            "backend": "npu",
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": "a" * 64,
            "strict_npu_qualified": False,
            "numerical_verified": False,
            "performance_qualified": None,
            "reason": "capability gate failed",
        },
    )
    store.record_sample_without_operator(
        "A_direct_simple/A001",
        status="failed",
        runtime_qualification_id=failed.artifact_id,
        strict_npu_qualified=False,
        performance_qualified=None,
    )
    summary = store.summary()
    assert summary["runtime_qualifications"] == 1
    assert summary["strict_npu_qualified"] == 0
    assert summary["performance_valid"] == 0
    assert store.verify()["ok"] is True



def test_store_rejects_unbound_strict_runtime_qualification(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    with pytest.raises(OperatorArtifactStoreError, match="matched NPU identity"):
        store.put_runtime_qualification(
            "A_direct_simple/A001",
            {
                "backend": "npu",
                "runtime_identity_status": "audit_required",
                "strict_npu_qualified": True,
                "numerical_verified": True,
            },
        )



def test_store_rejects_performance_without_strict_npu_and_evidence(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    base = {
        "backend": "npu",
        "runtime_identity_status": "matched",
        "runtime_identity_sha256": "a" * 64,
        "strict_npu_qualified": False,
        "numerical_verified": True,
        "performance_qualified": True,
    }
    with pytest.raises(OperatorArtifactStoreError, match="strict NPU"):
        store.put_runtime_qualification("A_direct_simple/A001", base)

    base["strict_npu_qualified"] = True
    with pytest.raises(OperatorArtifactStoreError, match="evidence reference"):
        store.put_runtime_qualification("A_direct_simple/A001", base)


def test_store_accepts_hash_bound_performance_evidence_only_after_strict_gate(tmp_path):
    store = OperatorArtifactStore(tmp_path / "run")
    source_sha = "b" * 64
    runtime_sha = "a" * 64
    receipt = store.put_runtime_qualification(
        "A_direct_simple/A001",
        {
            "backend": "npu",
            "source_sha256": source_sha,
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": runtime_sha,
            "strict_npu_qualified": True,
            "numerical_verified": True,
            "performance_qualified": True,
            "performance_evidence": {
                "schema": "numpilot.performance-evidence-ref",
                "schema_version": 1,
                "status": "passed",
                "performance_valid": True,
                "receipt_sha256": "c" * 64,
                "source_sha256": source_sha,
                "runtime_identity_sha256": runtime_sha,
            },
        },
    )
    store.record_sample_without_operator(
        "A_direct_simple/A001",
        status="not_applicable",
        runtime_qualification_id=receipt.artifact_id,
        strict_npu_qualified=True,
        performance_qualified=True,
    )
    summary = store.summary()
    assert summary["performance_valid"] == 1
    assert store.verify()["ok"] is True
