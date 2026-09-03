import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess

import pytest

from numpilot.gaps.recovery import (
    RecoveryError,
    RecoveryPhase,
    RecoveryPolicy,
    RecoveryPolicyProfile,
    RecoveryStore,
    StageReceiptKind,
    StageReceiptScope,
    StageReceiptVerdict,
)
from numpilot.gaps.recovery_ingest import (
    RecoveryIngestError,
    ingest_official_recovery,
    recovery_status,
)
from tests.fixture_support import recovery_fixture_root


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "examples" / "ops_nn_matmul_inference.py"
EVIDENCE = recovery_fixture_root()
MIGRATION = EVIDENCE / "pre-integration-gap-v2" / "script-migration-report.json"
GAP_REQUEST = EVIDENCE / "pre-integration-gap-v2" / "gap-request.json"
OFFICIAL = EVIDENCE / "official-resolution-linkage-v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copied_receipts(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir(mode=0o700)
    migration = inputs / "script-migration-report.json"
    gap_request = inputs / "gap-request.json"
    official = inputs / "official-resolution.json"
    for source, target in (
        (MIGRATION, migration),
        (GAP_REQUEST, gap_request),
        (OFFICIAL, official),
    ):
        shutil.copyfile(source, target)
        target.chmod(0o600)
    return migration, gap_request, official


def _mutate(path: Path, change) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _ingest(
    state_dir: Path,
    *,
    source: Path = SOURCE,
    migration: Path = MIGRATION,
    gap_request: Path = GAP_REQUEST,
    official: Path = OFFICIAL,
    policy=None,
    profile=None,
):
    return ingest_official_recovery(
        source,
        migration,
        gap_request,
        official,
        state_dir,
        policy=policy,
        profile=profile,
    )


def test_public_matmul_receipts_reach_resolved_without_side_effect_claims(tmp_path):
    state_dir = tmp_path / "recovery"
    result = _ingest(state_dir)

    assert result.created is True
    assert result.appended_revisions == 3
    assert result.recovery.history_length == 4
    assert result.recovery.state.phase is RecoveryPhase.RESOLVED
    assert result.recovery.next_action == "materialize_operator_candidate"
    assert result.recovery.state.run.run_id.startswith("recovery-")
    assert result.recovery.state.source.sha256 == _sha256(SOURCE)
    assert result.recovery.state.runtime.backend == "npu"
    assert result.recovery.state.target.target_soc == "Ascend910B4"
    assert result.recovery.state.target.cann_version == "9.0.0"
    assert not result.recovery.state.side_effects.any_performed

    rendered = result.to_dict()
    assert rendered["receipt_ingestion_only"] is True
    assert rendered["current_environment_recheck"] is False
    assert not any(rendered["side_effects_performed"].values())
    assert rendered == result.to_dict()
    assert {item["name"] for item in rendered["input_receipts"]} == {
        "script_migration_report",
        "gap_request",
        "official_resolution",
    }

    history = RecoveryStore(state_dir).load_history()
    assert [state.phase for state, _digest in history] == [
        RecoveryPhase.CREATED,
        RecoveryPhase.ANALYZED,
        RecoveryPhase.GAP_PREPARED,
        RecoveryPhase.RESOLVED,
    ]
    for index, (state, _digest) in enumerate(history[1:], start=1):
        receipt = next(
            item for item in state.stage_receipts if item.phase is state.phase
        )
        assert receipt.parent_state_sha256 == history[index - 1][1]
        assert receipt.scope is StageReceiptScope.RUN
        assert receipt.verdict is StageReceiptVerdict.SATISFIED
    assert {item.kind for item in history[-1][0].stage_receipts} == {
        StageReceiptKind.ANALYSIS_REPORT,
        StageReceiptKind.GAP_SPEC,
        StageReceiptKind.OFFICIAL_RESOLUTION_DECISION,
    }
    gap = result.recovery.state.gaps[0]
    assert {item.name for item in gap.refs} == {
        "gap_spec",
        "resolution_decision",
    }
    gap_request_value = json.loads(GAP_REQUEST.read_text(encoding="utf-8"))
    official_value = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    assert next(item for item in gap.refs if item.name == "gap_spec").sha256 == (
        _canonical_sha256(gap_request_value["gap_specs"][0])
    )
    assert next(
        item for item in gap.refs if item.name == "resolution_decision"
    ).sha256 == _canonical_sha256(official_value["decisions"][0])
    stage_digests = {
        item.kind: item.ref.sha256
        for item in result.recovery.state.stage_receipts
    }
    assert stage_digests == {
        StageReceiptKind.ANALYSIS_REPORT: _sha256(MIGRATION),
        StageReceiptKind.GAP_SPEC: _sha256(GAP_REQUEST),
        StageReceiptKind.OFFICIAL_RESOLUTION_DECISION: _sha256(OFFICIAL),
    }
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((state_dir / "states").stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in (state_dir / "states").glob("*.json")
    )


def test_ingestion_is_idempotent_and_status_rechecks_the_full_chain(tmp_path):
    state_dir = tmp_path / "recovery"
    first = _ingest(state_dir)
    second = _ingest(state_dir)

    assert second.created is False
    assert second.appended_revisions == 0
    assert second.recovery == first.recovery
    assert recovery_status(state_dir) == first.recovery
    assert sorted(path.name for path in (state_dir / "states").iterdir()) == [
        "000001.json",
        "000002.json",
        "000003.json",
        "000004.json",
    ]


def test_partial_publication_resumes_from_the_last_valid_stage(tmp_path, monkeypatch):
    state_dir = tmp_path / "recovery"
    original = RecoveryStore.append
    calls = {"count": 0}

    def fail_second_append(self, event):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RecoveryError("injected interruption")
        return original(self, event)

    monkeypatch.setattr(RecoveryStore, "append", fail_second_append)
    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(state_dir)
    assert failure.value.code == "state_append_failed"
    assert recovery_status(state_dir).state.phase is RecoveryPhase.ANALYZED
    assert recovery_status(state_dir).history_length == 2

    monkeypatch.setattr(RecoveryStore, "append", original)
    resumed = _ingest(state_dir)
    assert resumed.created is False
    assert resumed.appended_revisions == 2
    assert resumed.recovery.state.phase is RecoveryPhase.RESOLVED
    assert resumed.recovery.history_length == 4


@pytest.mark.parametrize(
    "target,change,expected_code",
    [
        (
            "migration",
            lambda value: value.__setitem__("schema_version", 1),
            "invalid_migration_report",
        ),
        (
            "migration",
            lambda value: value["analysis"].__setitem__("backend", "shim"),
            "invalid_migration_report",
        ),
        (
            "migration",
            lambda value: value["analysis"].__setitem__("unknown", True),
            "invalid_migration_report",
        ),
        (
            "migration",
            lambda value: value["analysis"].__setitem__("audit_id", "other-audit"),
            "receipt_binding_mismatch",
        ),
        (
            "migration",
            lambda value: value["qualified_analysis"].__setitem__(
                "runtime_identity_sha256", "0" * 64
            ),
            "invalid_migration_report",
        ),
        (
            "gap_request",
            lambda value: value.__setitem__("audit_id", "other-audit"),
            "receipt_binding_mismatch",
        ),
        (
            "gap_request",
            lambda value: value.__setitem__("activation_performed", True),
            "invalid_gap_request",
        ),
        (
            "official",
            lambda value: value.__setitem__("activation_allowed", True),
            "invalid_official_resolution",
        ),
        (
            "official",
            lambda value: value["decisions"][0].__setitem__(
                "activation_allowed", True
            ),
            "invalid_official_resolution",
        ),
    ],
)
def test_tampered_receipts_fail_before_recovery_creation(
    tmp_path, target, change, expected_code
):
    migration, gap_request, official = _copied_receipts(tmp_path)
    selected = {
        "migration": migration,
        "gap_request": gap_request,
        "official": official,
    }[target]
    _mutate(selected, change)
    state_dir = tmp_path / "recovery"

    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(
            state_dir,
            migration=migration,
            gap_request=gap_request,
            official=official,
        )
    assert failure.value.code == expected_code
    assert not state_dir.exists()


def test_source_and_receipt_sibling_bindings_fail_closed(tmp_path):
    migration, gap_request, official = _copied_receipts(tmp_path)
    with pytest.raises(RecoveryIngestError):
        _ingest(
            tmp_path / "wrong-source-state",
            source=REPO / "examples" / "big_gemm.py",
            migration=migration,
            gap_request=gap_request,
            official=official,
        )

    detached = tmp_path / "detached-gap.json"
    shutil.copyfile(gap_request, detached)
    detached.chmod(0o600)
    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(
            tmp_path / "detached-state",
            migration=migration,
            gap_request=detached,
            official=official,
        )
    assert failure.value.code == "receipt_binding_mismatch"


def test_duplicate_json_and_symlinked_inputs_are_rejected(tmp_path):
    migration, gap_request, official = _copied_receipts(tmp_path)
    migration.write_bytes(b'{"schema_version":2,"schema_version":2}\n')
    migration.chmod(0o600)
    with pytest.raises(RecoveryIngestError) as duplicate:
        _ingest(
            tmp_path / "duplicate-state",
            migration=migration,
            gap_request=gap_request,
            official=official,
        )
    assert duplicate.value.code == "invalid_json"

    migration.unlink()
    migration.symlink_to(MIGRATION)
    with pytest.raises(RecoveryIngestError) as symlink:
        _ingest(
            tmp_path / "symlink-state",
            migration=migration,
            gap_request=gap_request,
            official=official,
        )
    assert symlink.value.code == "unsafe_input"


def test_fifo_input_is_rejected_without_blocking(tmp_path):
    migration, gap_request, official = _copied_receipts(tmp_path)
    migration.unlink()
    os.mkfifo(migration, 0o600)

    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(
            tmp_path / "fifo-state",
            migration=migration,
            gap_request=gap_request,
            official=official,
        )
    assert failure.value.code == "unsafe_input"


def test_existing_state_policy_drift_is_rejected(tmp_path):
    state_dir = tmp_path / "recovery"
    first = _ingest(state_dir)
    assert first.recovery.state.policy.activation_allowed is False
    changed = RecoveryPolicy(
        fallback_policy="reject",
        activation_allowed=True,
    )
    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(state_dir, policy=changed)
    assert failure.value.code == "state_binding_mismatch"
    assert recovery_status(state_dir) == first.recovery


def test_public_named_profiles_freeze_exact_policy_without_claiming_effects(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("http_proxy", "http://proxy-sentinel.invalid:7897")
    monkeypatch.setenv("https_proxy", "http://proxy-sentinel.invalid:7897")
    monkeypatch.setenv(
        "NUMPILOT_LLM_CONFIG_FILE", "/private/config-sentinel/llm.json"
    )
    candidate = _ingest(
        tmp_path / "candidate-state", profile="candidate-generation"
    )
    candidate_policy = candidate.recovery.state.policy
    assert candidate_policy == RecoveryPolicy(
        fallback_policy="reject",
        subprocess_allowed=True,
        network_allowed=True,
        model_allowed=True,
    )
    assert not candidate.recovery.state.side_effects.any_performed

    full = _ingest(
        tmp_path / "full-state",
        profile=RecoveryPolicyProfile.FULL_LOCAL_RECOVERY,
    )
    full_policy = full.recovery.state.policy
    assert full_policy.official_first is True
    assert all(
        (
            full_policy.subprocess_allowed,
            full_policy.network_allowed,
            full_policy.system_write_allowed,
            full_policy.model_allowed,
            full_policy.asnumpy_import_allowed,
            full_policy.build_allowed,
            full_policy.execution_allowed,
            full_policy.activation_allowed,
        )
    )
    assert not full.recovery.state.side_effects.any_performed

    serialized = json.dumps(
        (candidate.recovery.state.to_dict(), full.recovery.state.to_dict()),
        ensure_ascii=False,
        sort_keys=True,
    )
    assert "proxy-sentinel" not in serialized
    assert "config-sentinel" not in serialized


@pytest.mark.parametrize(
    ("initial_profile", "changed_profile"),
    (
        (None, "candidate-generation"),
        ("candidate-generation", "full-local-recovery"),
    ),
)
def test_existing_state_rejects_named_profile_drift(
    tmp_path, initial_profile, changed_profile
):
    state_dir = tmp_path / "recovery"
    first = _ingest(state_dir, profile=initial_profile)

    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(state_dir, profile=changed_profile)
    assert failure.value.code == "state_binding_mismatch"
    assert recovery_status(state_dir) == first.recovery


def test_default_and_explicit_receipt_only_profile_are_idempotent(tmp_path):
    state_dir = tmp_path / "recovery"
    first = _ingest(state_dir)
    replay = _ingest(state_dir, profile="receipt-only")

    assert replay.created is False
    assert replay.appended_revisions == 0
    assert replay.recovery == first.recovery


def test_named_profile_and_custom_policy_are_mutually_exclusive(tmp_path):
    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(
            tmp_path / "recovery",
            profile="receipt-only",
            policy=RecoveryPolicy(fallback_policy="reject"),
        )
    assert failure.value.code == "policy_conflict"
    assert not (tmp_path / "recovery").exists()


@pytest.mark.parametrize("profile", ("", "unknown", False, 1))
def test_invalid_explicit_named_profile_is_not_coerced(tmp_path, profile):
    with pytest.raises(RecoveryIngestError) as failure:
        _ingest(tmp_path / "recovery", profile=profile)
    assert failure.value.code == "invalid_policy_profile"
    assert not (tmp_path / "recovery").exists()


def test_invalid_policy_and_run_id_are_not_coerced_to_defaults(tmp_path):
    with pytest.raises(RecoveryIngestError) as policy_failure:
        ingest_official_recovery(
            SOURCE,
            MIGRATION,
            GAP_REQUEST,
            OFFICIAL,
            tmp_path / "policy-state",
            policy=False,
        )
    assert policy_failure.value.code == "invalid_policy"

    with pytest.raises(RecoveryIngestError) as run_failure:
        ingest_official_recovery(
            SOURCE,
            MIGRATION,
            GAP_REQUEST,
            OFFICIAL,
            tmp_path / "run-state",
            run_id="",
        )
    assert run_failure.value.code == "invalid_run_id"


def test_status_detects_intermediate_state_tampering(tmp_path):
    state_dir = tmp_path / "recovery"
    _ingest(state_dir)
    second = state_dir / "states" / "000002.json"
    value = json.loads(second.read_text(encoding="utf-8"))
    value["reason"] = "tampered"
    second.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    second.chmod(0o600)

    with pytest.raises(RecoveryError):
        recovery_status(state_dir)


def test_ingestion_does_not_cross_external_side_effect_boundaries(
    tmp_path, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "receipt ingestion crossed an external side-effect boundary"
        )

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "asnumpy" or name.startswith("asnumpy."):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    result = _ingest(tmp_path / "recovery")
    assert result.recovery.state.phase is RecoveryPhase.RESOLVED
    assert not result.recovery.state.side_effects.any_performed
