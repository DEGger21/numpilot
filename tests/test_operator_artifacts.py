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
