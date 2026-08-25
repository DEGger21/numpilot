import json

import pytest

from numpilot.__main__ import main


def _plan():
    return {
        "mode": "read-only-plan",
        "python": {"version": "3.10.0", "ok": True},
        "venv": {"active": True},
        "cann": {"toolkit_directory": True, "opp_directory": True},
        "npu_smi": {"available": True},
        "request": {
            "asnumpy_mode": None,
            "use_cannbot": False,
            "cannbot_license_accepted": False,
        },
        "planned_actions": [
            {"action": "inspect_environment", "network": False},
        ],
        "issues": ["select an AsNumpy mode"],
    }


def test_bootstrap_defaults_to_read_only_json_plan(monkeypatch, capsys):
    import numpilot.bootstrap as module

    monkeypatch.setattr(module, "build_bootstrap_plan", lambda **_kwargs: _plan())
    monkeypatch.setattr(
        module,
        "apply_bootstrap",
        lambda **_kwargs: pytest.fail("default bootstrap attempted apply"),
    )

    assert main(["bootstrap", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "read-only-plan"


def test_bootstrap_apply_forwards_explicit_supply_chain_inputs(
    monkeypatch, capsys, tmp_path
):
    import numpilot.bootstrap as module

    observed = {}

    def fake_apply(**kwargs):
        observed.update(kwargs)
        return {
            "status": "ready",
            "ready": True,
            "wheel": {"sha256": "a" * 64},
        }

    monkeypatch.setattr(module, "apply_bootstrap", fake_apply)
    wheel = tmp_path / "asnumpy.whl"
    receipt = tmp_path / "receipt.json"

    assert main(
        [
            "bootstrap",
            "--apply",
            "--asnumpy-wheel",
            str(wheel),
            "--asnumpy-wheel-sha256",
            "a" * 64,
            "--with-cannbot",
            "--accept-cannbot-license",
            "--receipt",
            str(receipt),
        ]
    ) == 0

    assert observed["receipt_path"] == receipt
    assert observed["asnumpy_wheel"] == wheel
    assert observed["expected_wheel_sha256"] == "a" * 64
    assert observed["use_cannbot"] is True
    assert observed["accept_cannbot_license"] is True
    assert "status=ready" in capsys.readouterr().out


def test_bootstrap_apply_failure_returns_two_and_names_receipt(
    monkeypatch, capsys, tmp_path
):
    import numpilot.bootstrap as module

    def fail(**_kwargs):
        raise module.BootstrapError("wheel SHA-256 mismatch")

    monkeypatch.setattr(module, "apply_bootstrap", fail)
    receipt = tmp_path / "receipt.json"
    code = main(
        [
            "bootstrap",
            "--apply",
            "--build-asnumpy",
            "--receipt",
            str(receipt),
        ]
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "wheel SHA-256 mismatch" in error
    assert str(receipt) in error


def test_bootstrap_rejects_two_asnumpy_modes():
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "bootstrap",
                "--apply",
                "--asnumpy-wheel",
                "/tmp/asnumpy.whl",
                "--build-asnumpy",
            ]
        )
    assert raised.value.code == 2


def test_bootstrap_verify_prints_asnumpy_and_cannbot_skill_smokes(
    monkeypatch, capsys
):
    import numpilot.bootstrap as module

    def fake_verify(**kwargs):
        assert kwargs["use_cannbot"] is True
        assert kwargs["accept_cannbot_license"] is True
        return {
            "network_requested_by_numpilot": False,
            "asnumpy": {
                "status": "passed",
                "shape": [1000, 1000],
                "dtype": "float32",
            },
            "cannbot": {
                "status": "passed",
                "question": "现在写 Ascend C 算子有哪些能参考的 Skill？",
                "skills": [
                    {"phase": "工程模板", "name": "ascendc-registry-invoke-template"}
                ],
            },
        }

    monkeypatch.setattr(module, "verify_bootstrap_installation", fake_verify)
    assert main(
        [
            "bootstrap",
            "--verify",
            "--with-cannbot",
            "--accept-cannbot-license",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "AsNumpy" in output
    assert "ap.ones" in output
    assert "现在写 Ascend C" in output
    assert "ascendc-registry-invoke-template" in output
    assert "network_requested_by_numpilot=false" in output


def test_bootstrap_verify_rejects_install_arguments(capsys):
    assert main(
        [
            "bootstrap",
            "--verify",
            "--asnumpy-wheel",
            "/tmp/asnumpy.whl",
            "--asnumpy-wheel-sha256",
            "a" * 64,
        ]
    ) == 2
    assert "不接受 AsNumpy" in capsys.readouterr().err
