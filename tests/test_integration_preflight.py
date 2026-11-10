import sys

from numpilot.gaps.integration import preflight_official_adapter_integration
from numpilot.gaps.recovery import (
    RecoveryPhase,
    RecoveryPolicyProfile,
    RecoveryStore,
    recovery_policy_for_profile,
)
from numpilot.gaps.adapter_recovery import ingest_codex_adapter_candidate
from tests.test_adapter_recovery import _candidate_ready_fixture, _codex_adapter_output
from tests.test_gap_integration import _full_local_fixture


def test_preflight_ready_is_read_only(tmp_path):
    state_dir, candidate_path, _candidate, adapter_root, _ref, reference = (
        _full_local_fixture(tmp_path)
    )
    before = RecoveryStore(state_dir).status()
    result = preflight_official_adapter_integration(
        state_dir,
        candidate_path,
        adapter_root,
        reference,
        python_executable=sys.executable,
    )
    after = RecoveryStore(state_dir).status()
    assert result.status == "ready"
    assert result.code == "ready_for_integration"
    assert result.next_action == "integrate_candidate"
    assert result.generation_mode == "remote_llm"
    assert result.missing_policy == ()
    assert after == before
    assert result.to_dict()["side_effects"]["build_performed"] is False


def test_preflight_receipt_only_local_codex_is_blocked(tmp_path):
    state_dir, _before, candidate, candidate_path, prepared = _candidate_ready_fixture(
        tmp_path,
        policy=recovery_policy_for_profile(RecoveryPolicyProfile.CANDIDATE_GENERATION),
    )
    adapter_root, report = _codex_adapter_output(tmp_path, candidate, prepared)
    ingest_codex_adapter_candidate(state_dir, candidate_path, adapter_root)
    reference = report["asnumpy_adapter_context_plan"]["reference_root"]
    before = RecoveryStore(state_dir).status()
    result = preflight_official_adapter_integration(
        state_dir,
        candidate_path,
        adapter_root,
        reference,
        python_executable=sys.executable,
    )
    after = RecoveryStore(state_dir).status()
    assert result.status == "blocked"
    assert result.code == "authorization_missing"
    assert result.next_action == "authorize_full_local_recovery"
    assert result.generation_mode == "injected_test_double"
    assert "build_allowed" in result.missing_policy
    assert "activation_allowed" in result.missing_policy
    assert result.to_dict()["side_effects"]["execution_performed"] is False
    assert after == before
    assert after.state.phase is RecoveryPhase.ADAPTER_CANDIDATE


def test_preflight_cli_emits_blocked_json_and_exit_two(monkeypatch, capsys, tmp_path):
    import numpilot.__main__ as cli
    import numpilot.gaps.integration as integration

    payload = {
        "record_type": "numpilot_integration_preflight",
        "status": "blocked",
        "code": "authorization_missing",
        "next_action": "authorize_full_local_recovery",
        "missing_policy": ["build_allowed"],
    }

    class Result:
        status = "blocked"
        gap_id = "concatenate-test"
        code = "authorization_missing"
        generation_mode = "injected_test_double"
        next_action = "authorize_full_local_recovery"
        missing_policy = ("build_allowed",)

        @staticmethod
        def to_dict():
            return payload

    monkeypatch.setattr(
        integration,
        "preflight_official_adapter_integration",
        lambda *args, **kwargs: Result(),
    )
    rc = cli.main([
        "recover-gaps",
        "preflight-integrate-candidate",
        "--state-dir", str(tmp_path / "state"),
        "--candidate", str(tmp_path / "candidate.json"),
        "--adapter-output", str(tmp_path / "adapter"),
        "--asnumpy-reference", str(tmp_path / "asnumpy"),
        "--json",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err == ""
    import json
    assert json.loads(captured.out) == payload
