"""Integration-transaction and repaired-adapter recovery state transitions.

These tests replace only the isolated build/gate producer.  All state-machine,
receipt, binding, idempotency, and repair-budget behavior is exercised for
real; the compiled AsNumpy gate itself is validated separately by executing
the real transaction.
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from numpilot.gaps import integration
from numpilot.gaps.adapter_recovery import (
    AdapterRecoveryError,
    ingest_repaired_adapter_candidate,
)
from numpilot.gaps.integration import (
    INTEGRATION_RECEIPT_FILENAME,
    IntegrationError,
    run_official_adapter_integration,
)
from numpilot.gaps.provider import _adapter_artifact_id
from numpilot.gaps.recovery import (
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryPolicyProfile,
    recovery_policy_for_profile,
)
from tests.test_adapter_recovery import (
    _adapter_output,
    _candidate_ready_fixture,
    _remote_receipt,
)


def _canonical(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o600)


def _failure_transaction(out, candidate, adapter, adapter_ref, *, summary):
    diagnostic = {
        "schema_version": 1,
        "record_type": "numpilot_integration_failure_diagnostic",
        "transaction_id": "int-test-%s" % adapter_ref.sha256[:8],
        "gap_id": candidate.gap_id,
        "candidate_id": candidate.candidate_id,
        "adapter_artifact_id": adapter.report["adapter_artifact_id"],
        "adapter_report_sha256": adapter_ref.sha256,
        "phase": "gate",
        "failure": {"summary": summary},
        "repairs_used": 0,
        "repairs_remaining": 3,
        "manual_review_required": False,
    }
    diagnostic_payload = _canonical(diagnostic)
    _write(out / "diagnostics/failure.json", diagnostic_payload)
    return {
        "schema_version": 1,
        "record_type": "numpilot_integration_transaction",
        "transaction_id": diagnostic["transaction_id"],
        "gap_id": candidate.gap_id,
        "candidate_id": candidate.candidate_id,
        "adapter_artifact_id": adapter.report["adapter_artifact_id"],
        "adapter_report_sha256": adapter_ref.sha256,
        "status": "failed",
        "failure": {
            "phase": "gate",
            "summary": summary,
            "diagnostic_relative": "diagnostics/failure.json",
            "diagnostic_sha256": _sha256(diagnostic_payload),
            "manual_review_required": False,
        },
        "side_effects": {
            "subprocess_performed": True,
            "network_access_performed": False,
            "system_write_performed": True,
            "model_invocation_performed": False,
            "asnumpy_import_performed": True,
            "build_performed": True,
            "execution_performed": True,
            "verification_performed": False,
            "activation_performed": False,
        },
    }


def _success_transaction(candidate, adapter, adapter_ref):
    return {
        "schema_version": 1,
        "record_type": "numpilot_integration_transaction",
        "transaction_id": "int-test-%s" % adapter_ref.sha256[:8],
        "gap_id": candidate.gap_id,
        "candidate_id": candidate.candidate_id,
        "adapter_artifact_id": adapter.report["adapter_artifact_id"],
        "adapter_report_sha256": adapter_ref.sha256,
        "status": "passed",
        "failure": None,
        "side_effects": {
            "subprocess_performed": True,
            "network_access_performed": False,
            "system_write_performed": True,
            "model_invocation_performed": False,
            "asnumpy_import_performed": True,
            "build_performed": True,
            "execution_performed": True,
            "verification_performed": False,
            "activation_performed": False,
        },
    }


def _publish(out, transaction):
    transaction = dict(transaction)
    transaction["transaction_sha256"] = integration._transaction_sha256(transaction)
    _write(out / INTEGRATION_RECEIPT_FILENAME, _canonical(transaction))
    return transaction


def _fake_producer(status, summary="gate failed"):
    def produce(
        out,
        reference_root,
        candidate,
        adapter,
        adapter_ref,
        adapter_root,
        python,
        python_sha256,
        timeout,
        gap,
    ):
        out.mkdir(mode=0o700)
        if status == "passed":
            transaction = _success_transaction(candidate, adapter, adapter_ref)
        else:
            transaction = _failure_transaction(
                out, candidate, adapter, adapter_ref, summary=summary
            )
        transaction = _publish(out, transaction)
        return transaction, out / INTEGRATION_RECEIPT_FILENAME

    return produce


def _full_local_fixture(tmp_path):
    state_dir, status, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(
            tmp_path,
            policy=recovery_policy_for_profile(
                RecoveryPolicyProfile.FULL_LOCAL_RECOVERY
            ),
        )
    )
    adapter_root, report = _adapter_output(tmp_path, candidate, prepared)
    from numpilot.gaps.adapter_recovery import ingest_official_adapter_candidate

    adapt_result = ingest_official_adapter_candidate(
        state_dir, candidate_path, adapter_root
    )
    plan_reference = Path(
        report["asnumpy_adapter_context_plan"]["reference_root"]
    )
    return (
        state_dir,
        candidate_path,
        candidate,
        adapter_root,
        adapt_result.adapter_ref,
        plan_reference,
    )


def _latest_failure_sha(status):
    from numpilot.gaps.adapter_recovery import _latest_failure_ref

    return _latest_failure_ref(status.state.gaps[0].refs).sha256


def _repaired_output(
    parent_root,
    destination,
    diagnostic_sha256,
    repaired_relative,
    new_content,
    response_text,
):
    parent_root = Path(parent_root)
    parent_report_path = parent_root / "asnumpy-adapter-report.json"
    parent_payload = parent_report_path.read_bytes()
    parent_report = json.loads(parent_payload.decode("utf-8"))
    shutil.copytree(parent_root / "candidate", destination / "candidate")
    target = destination / "candidate" / repaired_relative
    old_payload = target.read_bytes()
    new_payload = new_content.encode("utf-8")
    target.write_bytes(new_payload)
    target.chmod(0o600)
    files = dict(parent_report["files_sha256"])
    files[repaired_relative] = _sha256(new_payload)
    report = {
        "schema_version": 4,
        "status": "repaired_unverified",
        "source_artifact_id": parent_report["source_artifact_id"],
        "model": parent_report["model"],
        "provider": parent_report["provider"],
        "protocol": parent_report["protocol"],
        "response_model_aliases": list(parent_report["response_model_aliases"]),
        "configuration_source": parent_report["configuration_source"],
        "parent_adapter_artifact_id": parent_report["adapter_artifact_id"],
        "parent_adapter_report_sha256": _sha256(parent_payload),
        "source_adapter_root": str(parent_root),
        "repair_profile": "compile",
        "attempt_count": parent_report["attempt_count"] + 1,
        "paths": [repaired_relative],
        "before_sha256": {repaired_relative: _sha256(old_payload)},
        "after_sha256": {repaired_relative: _sha256(new_payload)},
        "files_sha256": files,
        "diagnostic_sha256": diagnostic_sha256,
        "asnumpy_context_sha256": dict(parent_report["asnumpy_context_sha256"]),
        "asnumpy_adapter_context_plan": parent_report[
            "asnumpy_adapter_context_plan"
        ],
        "skill_bundle_sha256": parent_report["skill_bundle_sha256"],
        "integration_skill_files": list(parent_report["integration_skill_files"]),
        "response_sha256": [
            hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        ],
        "remote_receipts": [_remote_receipt(response_text)],
        "actual_remote_call": True,
        "gates": dict(parent_report["gates"]),
    }
    report["adapter_artifact_id"] = _adapter_artifact_id(report)
    _write(
        destination / "asnumpy-adapter-report.json",
        _canonical(report),
    )
    return destination, report


def _run(state_dir, candidate_path, adapter_root, reference, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    return run_official_adapter_integration(
        state_dir,
        candidate_path,
        adapter_root,
        reference,
        out,
        python_executable=sys.executable,
        timeout_seconds=60,
    )


def test_successful_transaction_advances_and_replays_idempotently(
    tmp_path, monkeypatch
):
    (
        state_dir,
        candidate_path,
        candidate,
        adapter_root,
        adapter_ref,
        reference_root,
    ) = _full_local_fixture(tmp_path)
    monkeypatch.setattr(
        integration, "_produce_transaction", _fake_producer("passed")
    )
    out = tmp_path / "integration" / "attempt-pass"

    result = _run(
        state_dir, candidate_path, adapter_root, reference_root, out
    )

    assert result.status == "passed"
    assert result.appended_revisions == 1
    assert result.recovery.state.phase is RecoveryPhase.INTEGRATED_CANDIDATE
    gap = result.recovery.state.gaps[0]
    assert gap.phase is RecoveryPhase.INTEGRATED_CANDIDATE
    assert gap.outcome is RecoveryOutcome.READY
    assert gap.attempt_count == 0
    receipt = next(
        item
        for item in result.recovery.state.stage_receipts
        if item.phase is RecoveryPhase.INTEGRATED_CANDIDATE
    )
    assert receipt.ref == result.transaction_ref
    assert receipt.verdict.value == "satisfied"
    effects = result.recovery.state.event_side_effects.to_dict()
    assert effects["build_performed"] is True
    assert effects["execution_performed"] is True
    assert effects["model_invocation_performed"] is False
    assert effects["activation_performed"] is False

    history = result.recovery.history_length
    replay = _run(
        state_dir, candidate_path, adapter_root, reference_root, out
    )

    assert replay.status == "passed"
    assert replay.appended_revisions == 0
    assert replay.recovery.history_length == history
    assert replay.recovery.state.phase is RecoveryPhase.INTEGRATED_CANDIDATE
    assert replay.transaction_ref == result.transaction_ref


def test_failed_transaction_preserves_candidate_and_appends_once(
    tmp_path, monkeypatch
):
    (
        state_dir,
        candidate_path,
        candidate,
        adapter_root,
        adapter_ref,
        reference_root,
    ) = _full_local_fixture(tmp_path)
    candidate_payload = candidate_path.read_bytes()
    adapter_payload = (adapter_root / "asnumpy-adapter-report.json").read_bytes()
    monkeypatch.setattr(
        integration, "_produce_transaction", _fake_producer("failed")
    )
    out = tmp_path / "integration" / "attempt-fail"

    result = _run(
        state_dir, candidate_path, adapter_root, reference_root, out
    )

    assert result.status == "failed"
    assert result.diagnostic_ref is not None
    assert result.appended_revisions == 1
    assert result.manual_review_required is False
    gap = result.recovery.state.gaps[0]
    assert gap.phase is RecoveryPhase.ADAPTER_CANDIDATE
    assert gap.outcome is RecoveryOutcome.FAILED_ATTEMPT
    assert gap.attempt_count == 1
    assert any(
        ref.name == "integration_failure_1" and ref == result.diagnostic_ref
        for ref in gap.refs
    )
    receipt = next(
        item
        for item in result.recovery.state.stage_receipts
        if item.phase is RecoveryPhase.INTEGRATED_CANDIDATE
    )
    assert receipt.verdict.value == "failed"
    assert receipt.ref == result.diagnostic_ref
    assert candidate_path.read_bytes() == candidate_payload
    assert (
        adapter_root / "asnumpy-adapter-report.json"
    ).read_bytes() == adapter_payload

    replay = _run(
        state_dir, candidate_path, adapter_root, reference_root, out
    )
    assert replay.appended_revisions == 0
    assert replay.recovery.state.gaps[0].attempt_count == 1
    assert replay.diagnostic_ref == result.diagnostic_ref


def test_repaired_adapter_resumes_same_phase_and_integrates(
    tmp_path, monkeypatch
):
    (
        state_dir,
        candidate_path,
        candidate,
        adapter_root,
        adapter_ref,
        reference_root,
    ) = _full_local_fixture(tmp_path)
    monkeypatch.setattr(
        integration, "_produce_transaction", _fake_producer("failed")
    )
    first = _run(
        state_dir,
        candidate_path,
        adapter_root,
        reference_root,
        tmp_path / "integration" / "attempt-1",
    )
    assert first.recovery.state.gaps[0].attempt_count == 1
    receipts_before_repair = first.recovery.state.stage_receipts

    repaired_root, repaired_report = _repaired_output(
        adapter_root,
        tmp_path / "repaired-1",
        _latest_failure_sha(first.recovery),
        "csrc/linalg/product.cpp",
        "// repaired ownership fixture\n",
        "repaired adapter response 1",
    )
    repair = ingest_repaired_adapter_candidate(
        state_dir, candidate_path, repaired_root
    )

    assert repair.appended_revisions == 1
    assert repair.repairs_used == 1
    assert repair.parent_adapter_ref == adapter_ref
    assert repair.diagnostic_ref.sha256 == _latest_failure_sha(first.recovery)
    gap = repair.recovery.state.gaps[0]
    assert gap.phase is RecoveryPhase.ADAPTER_CANDIDATE
    assert gap.outcome is RecoveryOutcome.READY
    assert gap.attempt_count == 1
    assert repair.adapter_ref.name == "asnumpy_adapter_candidate_repair_1"
    assert repair.adapter_ref.sha256 == _sha256(
        (repaired_root / "asnumpy-adapter-report.json").read_bytes()
    )
    assert repair.recovery.state.stage_receipts == receipts_before_repair
    replay_repair = ingest_repaired_adapter_candidate(
        state_dir, candidate_path, repaired_root
    )
    assert replay_repair.appended_revisions == 0
    assert replay_repair.adapter_ref == repair.adapter_ref

    monkeypatch.setattr(
        integration, "_produce_transaction", _fake_producer("passed")
    )
    integrated = _run(
        state_dir,
        candidate_path,
        repaired_root,
        reference_root,
        tmp_path / "integration" / "attempt-2",
    )
    assert integrated.status == "passed"
    assert integrated.recovery.state.phase is RecoveryPhase.INTEGRATED_CANDIDATE
    assert integrated.recovery.state.gaps[0].attempt_count == 1


def test_repair_budget_blocks_after_three_rounds(tmp_path, monkeypatch):
    (
        state_dir,
        candidate_path,
        candidate,
        adapter_root,
        adapter_ref,
        reference_root,
    ) = _full_local_fixture(tmp_path)
    monkeypatch.setattr(
        integration, "_produce_transaction", _fake_producer("failed")
    )
    result = _run(
        state_dir,
        candidate_path,
        adapter_root,
        reference_root,
        tmp_path / "integration" / "attempt-0",
    )
    parent_root = adapter_root
    parent_ref = adapter_ref
    for round_index in (1, 2, 3):
        repaired_root, _report = _repaired_output(
            parent_root,
            tmp_path / ("repaired-%d" % round_index),
            _latest_failure_sha(result.recovery),
            "csrc/linalg/product.cpp",
            "// repair round %d\n" % round_index,
            "repair response %d" % round_index,
        )
        repair = ingest_repaired_adapter_candidate(
            state_dir, candidate_path, repaired_root
        )
        assert repair.repairs_used == round_index
        parent_root = repaired_root
        parent_ref = repair.adapter_ref
        result = _run(
            state_dir,
            candidate_path,
            repaired_root,
            reference_root,
            tmp_path / "integration" / ("attempt-%d" % round_index),
        )

    assert result.status == "failed"
    assert result.manual_review_required is True
    gap = result.recovery.state.gaps[0]
    assert gap.phase is RecoveryPhase.ADAPTER_CANDIDATE
    assert gap.outcome is RecoveryOutcome.BLOCKED
    assert gap.attempt_count == 4
    assert "MANUAL_REVIEW_REQUIRED" in gap.reason

    fourth_root, _report = _repaired_output(
        parent_root,
        tmp_path / "repaired-4",
        _latest_failure_sha(result.recovery),
        "csrc/linalg/product.cpp",
        "// repair round 4\n",
        "repair response 4",
    )
    with pytest.raises(AdapterRecoveryError) as caught:
        ingest_repaired_adapter_candidate(
            state_dir, candidate_path, fourth_root
        )
    assert caught.value.code == "repair_budget_exhausted"
    assert result.recovery.next_action == "inspect_blocker"
    blocked_replay = _run(
        state_dir,
        candidate_path,
        parent_root,
        reference_root,
        tmp_path / "integration" / "attempt-3",
    )
    assert blocked_replay.appended_revisions == 0
    assert blocked_replay.recovery.state.gaps[0].outcome is RecoveryOutcome.BLOCKED


def test_integration_requires_full_local_recovery_authorization(
    tmp_path, monkeypatch
):
    state_dir, status, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, report = _adapter_output(tmp_path, candidate, prepared)
    from numpilot.gaps.adapter_recovery import ingest_official_adapter_candidate

    ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    reference_root = Path(report["asnumpy_adapter_context_plan"]["reference_root"])
    monkeypatch.setattr(
        integration, "_produce_transaction", _fake_producer("passed")
    )
    with pytest.raises(IntegrationError) as caught:
        _run(
            state_dir,
            candidate_path,
            adapter_root,
            reference_root,
            tmp_path / "integration" / "attempt",
        )
    assert caught.value.code == "authorization_missing"
