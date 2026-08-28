import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpilot.__main__ as cli
from numpilot.gaps.recovery import RecoveryError
import numpilot.gaps.recovery_ingest as recovery_ingest_api
from numpilot.gaps.recovery_ingest import RecoveryIngestError


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "examples" / "ops_nn_matmul_inference.py"
EVIDENCE = REPO / "tests" / "fixtures" / "ops-nn-recovery"
MIGRATION = EVIDENCE / "pre-integration-gap-v2" / "script-migration-report.json"
GAP_REQUEST = EVIDENCE / "pre-integration-gap-v2" / "gap-request.json"
OFFICIAL = EVIDENCE / "official-resolution-linkage-v2.json"


def _ingest_args(state_dir: Path, *extra: str):
    return [
        "recover-gaps",
        "ingest-official",
        str(SOURCE),
        "--migration-report",
        str(MIGRATION),
        "--gap-request",
        str(GAP_REQUEST),
        "--official-resolution-report",
        str(OFFICIAL),
        "--state-dir",
        str(state_dir),
        *extra,
    ]


def test_ingest_and_status_json_are_clean_and_share_the_verified_state(
    tmp_path, capsys
):
    state_dir = tmp_path / "recovery"
    rc = cli.main(_ingest_args(
        state_dir, "--run-id", "cli-recovery-001", "--json"
    ))

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    ingest = json.loads(captured.out)
    assert ingest["record_type"] == "numpilot_official_recovery_ingest"
    assert ingest["created"] is True
    assert ingest["appended_revisions"] == 3
    assert ingest["recovery"]["state"]["run"]["run_id"] == "cli-recovery-001"
    assert ingest["recovery"]["state"]["phase"] == "resolved"
    assert ingest["recovery"]["state"]["policy"]["model_allowed"] is False

    rc = cli.main([
        "recover-gaps", "status", "--state-dir", str(state_dir), "--json"
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    status = json.loads(captured.out)
    assert status == ingest["recovery"]


def test_human_output_is_a_concise_chinese_summary(tmp_path, capsys):
    state_dir = tmp_path / "recovery"
    assert cli.main(_ingest_args(state_dir)) == 0
    ingest_output = capsys.readouterr()
    assert ingest_output.err == ""
    assert "缺口恢复" in ingest_output.out
    assert "阶段=resolved" in ingest_output.out
    assert "本次追加=3" in ingest_output.out

    assert cli.main([
        "recover-gaps", "status", "--state-dir", str(state_dir)
    ]) == 0
    status_output = capsys.readouterr()
    assert status_output.err == ""
    assert "结果=ready" in status_output.out
    assert "下一步=materialize_operator_candidate" in status_output.out


def test_ingest_error_uses_stderr_and_exit_two(monkeypatch, capsys, tmp_path):
    def reject(*_args, **_kwargs):
        raise RecoveryIngestError("invalid_gap_request", "receipt mismatch")

    monkeypatch.setattr(recovery_ingest_api, "ingest_official_recovery", reject)
    assert cli.main(_ingest_args(tmp_path / "state", "--json")) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid_gap_request" in captured.err
    assert "receipt mismatch" in captured.err


def test_status_error_uses_stderr_and_exit_two(monkeypatch, capsys, tmp_path):
    def reject(_state_dir):
        raise RecoveryError("broken recovery chain")

    monkeypatch.setattr(recovery_ingest_api, "recovery_status", reject)
    assert cli.main([
        "recover-gaps",
        "status",
        "--state-dir",
        str(tmp_path / "state"),
        "--json",
    ]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "broken recovery chain" in captured.err


def test_recovery_cli_exposes_only_typed_recovery_workflows(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["recover-gaps", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "ingest-official" in help_text
    assert "materialize-candidates" in help_text
    assert "ingest-adapter" in help_text
    assert "status" in help_text
    assert "advance" not in help_text
    assert "side-effect" not in help_text
    assert "policy" not in help_text


def test_ingest_adapter_cli_replays_published_receipt_as_json(
    monkeypatch, capsys, tmp_path
):
    import numpilot.gaps.adapter_recovery as adapter_recovery_api

    payload = {
        "record_type": "numpilot_official_adapter_recovery",
        "appended_revisions": 1,
    }
    result = SimpleNamespace(
        gap_id="matmul-test",
        recovery=SimpleNamespace(
            state=SimpleNamespace(phase=SimpleNamespace(value="adapter_candidate"))
        ),
        appended_revisions=1,
        adapter_output_root=str(tmp_path / "adapter"),
        to_dict=lambda: payload,
    )
    observed = []

    def ingest(state_dir, candidate, adapter_output):
        observed.append((state_dir, candidate, adapter_output))
        return result

    monkeypatch.setattr(
        adapter_recovery_api, "ingest_official_adapter_candidate", ingest
    )

    rc = cli.main([
        "recover-gaps",
        "ingest-adapter",
        "--state-dir", str(tmp_path / "state"),
        "--candidate", str(tmp_path / "candidate.json"),
        "--adapter-output", str(tmp_path / "adapter"),
        "--json",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out) == payload
    assert observed == [(
        str(tmp_path / "state"),
        str(tmp_path / "candidate.json"),
        str(tmp_path / "adapter"),
    )]


def test_ingest_adapter_cli_reports_stable_error_code(
    monkeypatch, capsys, tmp_path
):
    import numpilot.gaps.adapter_recovery as adapter_recovery_api
    from numpilot.gaps.adapter_recovery import AdapterRecoveryError

    def reject(*_args):
        raise AdapterRecoveryError("adapter_hash_mismatch", "drift")

    monkeypatch.setattr(
        adapter_recovery_api, "ingest_official_adapter_candidate", reject
    )

    rc = cli.main([
        "recover-gaps",
        "ingest-adapter",
        "--state-dir", str(tmp_path / "state"),
        "--candidate", str(tmp_path / "candidate.json"),
        "--adapter-output", str(tmp_path / "adapter"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "adapter_hash_mismatch" in captured.err


def test_materialize_cli_keeps_json_clean_and_reports_no_model_call(
    monkeypatch, capsys, tmp_path
):
    import numpilot.gaps.candidate_materializer as materializer_api

    payload = {
        "record_type": "numpilot_candidate_materialization",
        "side_effects_performed": {
            "model_invocation_performed": False,
            "network_access_performed": False,
        },
    }
    result = SimpleNamespace(
        candidates=(object(),),
        recovery=SimpleNamespace(
            state=SimpleNamespace(phase=SimpleNamespace(value="candidate_ready"))
        ),
        appended_revisions=1,
        output_dir=str(tmp_path / "candidates"),
        to_dict=lambda: payload,
    )
    monkeypatch.setattr(
        materializer_api,
        "materialize_official_binary_candidates",
        lambda _state, _out: result,
    )

    rc = cli.main([
        "recover-gaps",
        "materialize-candidates",
        "--state-dir",
        str(tmp_path / "state"),
        "--out",
        str(tmp_path / "candidates"),
        "--json",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out) == payload


def test_materialize_cli_error_has_stable_code(monkeypatch, capsys, tmp_path):
    import numpilot.gaps.candidate_materializer as materializer_api
    from numpilot.gaps.candidate_materializer import CandidateMaterializationError

    def reject(_state, _out):
        raise CandidateMaterializationError("fresh_resolution_drift", "drift")

    monkeypatch.setattr(
        materializer_api, "materialize_official_binary_candidates", reject
    )

    rc = cli.main([
        "recover-gaps",
        "materialize-candidates",
        "--state-dir",
        str(tmp_path / "state"),
        "--out",
        str(tmp_path / "candidates"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "fresh_resolution_drift" in captured.err


def test_ingest_cli_freezes_named_candidate_generation_authorization(
    tmp_path, capsys
):
    state_dir = tmp_path / "candidate-recovery"

    assert cli.main(_ingest_args(
        state_dir,
        "--authorization-profile",
        "candidate-generation",
        "--json",
    )) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    policy = json.loads(captured.out)["recovery"]["state"]["policy"]
    assert policy["subprocess_allowed"] is True
    assert policy["network_allowed"] is True
    assert policy["model_allowed"] is True
    assert policy["system_write_allowed"] is False
    assert policy["build_allowed"] is False
    assert policy["activation_allowed"] is False
