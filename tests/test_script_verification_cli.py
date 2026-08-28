import json
from pathlib import Path

import numpilot.__main__ as cli
import numpilot.script_verification as script_verification


class _Receipt:
    def __init__(self, status="passed", path="/published/verification.json"):
        self.status = status
        self.path = path
        self._payload = {
            "schema_version": 1,
            "status": status,
            "execution_boundary": {
                "subprocess_isolation": True,
                "os_sandbox": False,
                "explicit_execution": True,
            },
            "evidence_boundary": {
                "numerical_only": True,
                "strict_npu_verified": False,
                "performance_valid": False,
            },
        }

    def to_dict(self):
        return dict(self._payload)


def _base_args(tmp_path):
    source = tmp_path / "source.py"
    report = tmp_path / "script-migration-report.json"
    migrated = tmp_path / "migrated.py"
    output = tmp_path / "script-verification.json"
    source.write_text("result = 1\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")
    migrated.write_text("result = 1\n", encoding="utf-8")
    return source, report, migrated, output


def test_verify_script_cli_maps_arguments_and_renders_passed_boundaries(
    tmp_path, monkeypatch, capsys
):
    source, report, migrated, output = _base_args(tmp_path)
    published = tmp_path / "canonical" / "verification.json"
    captured = {}

    def fake_verify(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Receipt("passed", str(published.absolute()))

    monkeypatch.setattr(script_verification, "verify_script", fake_verify)

    rc = cli.main([
        "verify-script",
        str(source),
        "--migration-report", str(report),
        "--migrated", str(migrated),
        "--output", "scores",
        "--output", "labels",
        "--out", str(output),
        "--timeout", "12.5",
        "--max-output-bytes", "4096",
        "--source-python", "/opt/source/bin/python",
        "--migrated-python", "/opt/migrated/bin/python",
    ])

    assert rc == 0
    assert captured["args"] == (
        str(source),
        str(report),
        ["scores", "labels"],
        str(output),
    )
    assert captured["kwargs"] == {
        "migrated_path": str(migrated),
        "timeout_seconds": 12.5,
        "max_output_bytes": 4096,
        "source_python_executable": "/opt/source/bin/python",
        "migrated_python_executable": "/opt/migrated/bin/python",
    }
    rendered = capsys.readouterr().out
    assert "status=passed" in rendered
    assert "scores, labels" in rendered
    assert (
        "subprocess_isolation=true · os_sandbox=false · "
        "explicit_execution=true"
    ) in rendered
    assert (
        "numerical_only=true · strict_npu_verified=false · "
        "performance_valid=false"
    ) in rendered
    assert str(published.absolute()) in rendered


def test_verify_script_cli_returned_failed_receipt_returns_one(
    tmp_path, monkeypatch, capsys
):
    source, report, _migrated, output = _base_args(tmp_path)
    monkeypatch.setattr(
        script_verification,
        "verify_script",
        lambda *_args, **_kwargs: _Receipt("failed"),
    )

    rc = cli.main([
        "verify-script", str(source),
        "--migration-report", str(report),
        "--output", "result",
        "--out", str(output),
    ])

    assert rc == 1
    assert "status=failed" in capsys.readouterr().out


def test_verify_script_cli_auditable_error_returns_one(
    tmp_path, monkeypatch, capsys
):
    source, report, _migrated, output = _base_args(tmp_path)
    receipt = _Receipt("failed", str(output.absolute()))

    def fail_with_receipt(*_args, **_kwargs):
        raise script_verification.ScriptVerificationError(
            "numerical_mismatch",
            receipt=receipt,
            phase="comparison",
        )

    monkeypatch.setattr(
        script_verification, "verify_script", fail_with_receipt
    )

    rc = cli.main([
        "verify-script", str(source),
        "--migration-report", str(report),
        "--output", "result",
        "--out", str(output),
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "numerical_mismatch" in captured.err
    assert str(Path(output).absolute()) in captured.out
    assert "execution_not_admitted_or_receipt_not_published" not in captured.err


def test_verify_script_cli_preflight_error_without_receipt_returns_two(
    tmp_path, monkeypatch, capsys
):
    source, report, _migrated, output = _base_args(tmp_path)

    def fail_before_receipt(*_args, **_kwargs):
        raise script_verification.ScriptVerificationError("invalid_report")

    monkeypatch.setattr(
        script_verification, "verify_script", fail_before_receipt
    )

    rc = cli.main([
        "verify-script", str(source),
        "--migration-report", str(report),
        "--output", "result",
        "--out", str(output),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "invalid_report" in captured.err
    assert "execution_not_admitted_or_receipt_not_published" in captured.err
    assert captured.out == ""


def test_verify_script_cli_real_invalid_report_returns_two_without_receipt(
    tmp_path, capsys
):
    source, report, _migrated, output = _base_args(tmp_path)

    rc = cli.main([
        "verify-script", str(source),
        "--migration-report", str(report),
        "--output", "result",
        "--out", str(output),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "invalid_report" in captured.err
    assert "execution_not_admitted_or_receipt_not_published" in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_verify_script_cli_json_is_one_valid_receipt(
    tmp_path, monkeypatch, capsys
):
    source, report, _migrated, output = _base_args(tmp_path)
    receipt = _Receipt("passed")
    monkeypatch.setattr(
        script_verification,
        "verify_script",
        lambda *_args, **_kwargs: receipt,
    )

    rc = cli.main([
        "verify-script", str(source),
        "--migration-report", str(report),
        "--output", "result",
        "--out", str(output),
        "--json",
    ])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == receipt.to_dict()
