import copy
import hashlib
import json
import signal
import stat
import sys
from pathlib import Path

import pytest

import numpilot.script_verification as verification
from numpilot.script_verification import (
    ScriptVerificationError,
    verify_script,
)


SOURCE = """\
import numpy as np
result = np.arange(6, dtype=np.float32).reshape(2, 3)
flag = bool(result.size == 6)
print("source stdout is not evidence", result)
"""


MIGRATED = """\
import numpy as np
result = np.arange(6, dtype=np.float32).reshape(2, 3)
flag = bool(result.size == 6)
print("migrated stdout can be different")
"""


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _migration(tmp_path, source=SOURCE, migrated=MIGRATED):
    source_path = tmp_path / "case.py"
    source_path.write_text(source, encoding="utf-8")
    migration_dir = tmp_path / "migration"
    migration_dir.mkdir()
    migrated_path = migration_dir / "migrated.py"
    migrated_path.write_text(migrated, encoding="utf-8")
    source_hash = _sha256(source_path.read_bytes())
    migrated_hash = _sha256(migrated_path.read_bytes())
    report = {
        "schema_version": 2,
        "status": "migrated_strict_npu",
        "mode": "script",
        "analysis": {
            "schema_version": 1,
            "mode": "script",
            "scan": {
                "schema_version": 1,
                "source_path": str(source_path.absolute()),
                "source_sha256": source_hash,
                "ok": True,
                "issues": [],
            },
        },
        "fallback_accepted": False,
        "strict_npu": True,
        "performance_valid": False,
        "execution_performed": False,
        "numerical_verification": "not_run",
        "generated_by": "configured_model",
        "source_sha256": source_hash,
        "migrated_sha256": migrated_hash,
        "ast_audit": "passed",
        "published_files": ["migrated.py"],
    }
    report_path = migration_dir / "script-migration-report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source_path, migrated_path, report_path


def _failed_receipt(error):
    assert error.receipt is not None
    assert error.receipt.status == "failed"
    persisted = json.loads(
        open(error.receipt.path, encoding="utf-8").read()
    )
    assert persisted == error.receipt.to_dict()
    return persisted


def test_verify_script_passes_with_explicit_no_pickle_output_contract(
    tmp_path, monkeypatch
):
    source, _, report = _migration(tmp_path)
    report_before = report.read_bytes()
    out = tmp_path / "receipts" / "verification.json"
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-worker")

    receipt = verify_script(source, report, ["result", "flag"], out)

    assert receipt.status == "passed"
    assert receipt.path == str(out.absolute())
    assert receipt.sha256 == _sha256(out.read_bytes())
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    payload = receipt.to_dict()
    assert payload["numerically_consistent"] is True
    assert payload["comparisons"]["result"]["consistent"] is True
    assert payload["comparisons"]["flag"]["consistent"] is True
    assert payload["contract"]["transport"] == (
        "json_manifest_and_npy_allow_pickle_false"
    )
    assert payload["contract"]["stdout_is_result_channel"] is False
    assert payload["strict_npu_verified"] is False
    assert payload["performance_valid"] is False
    assert payload["claims"]["os_sandbox"] is False
    assert payload["claims"]["strict_npu_provenance_verified"] is False
    assert payload["executions"]["source"]["pid"] != (
        payload["executions"]["migrated"]["pid"]
    )
    assert payload["executions"]["source"]["worker_sha256"] == (
        payload["executions"]["migrated"]["worker_sha256"]
    )
    assert len(
        payload["executions"]["source"]["python_executable_sha256"]
    ) == 64
    assert payload["executions"]["source"]["total_data_bytes"] == 25
    numpy_runtime = payload["executions"]["source"][
        "loaded_runtime_modules"
    ]["numpy"]
    assert numpy_runtime["version"]
    assert numpy_runtime["path"]
    assert numpy_runtime["file_hash_status"] == "hashed"
    assert len(numpy_runtime["file_sha256"]) == 64
    assert payload["executions"]["source"][
        "removed_sensitive_environment_variables"
    ] >= 1
    for execution in payload["executions"].values():
        assert all("value" not in item for item in execution["outputs"])
    assert report.read_bytes() == report_before


def test_python_invocation_path_preserves_venv_style_symlink(tmp_path):
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)

    identity = verification._python_executable(executable)
    assert identity.invocation_path == str(executable.absolute())
    assert identity.realpath == str(executable.resolve())
    assert len(identity.file_sha256) == 64
    assert identity.file_size == executable.resolve().stat().st_size


def test_keyboard_interrupt_terminates_the_private_worker_group(
    tmp_path, monkeypatch
):
    class _InterruptedProcess:
        pid = 424242

        def __init__(self):
            self.returncode = None
            self.wait_count = 0

        def wait(self, timeout=None):
            self.wait_count += 1
            if self.wait_count == 1:
                raise KeyboardInterrupt()
            self.returncode = -signal.SIGKILL
            return self.returncode

        def kill(self):
            self.returncode = -signal.SIGKILL

    process = _InterruptedProcess()
    killed = []
    monkeypatch.setattr(
        verification.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(
        verification.os,
        "killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )
    identity = verification._python_executable(sys.executable)

    with pytest.raises(KeyboardInterrupt):
        verification._run_worker(
            root=tmp_path,
            label="interrupted",
            worker_path=tmp_path / "worker.py",
            script_path=tmp_path / "script.py",
            script_sha256="1" * 64,
            worker_sha256="2" * 64,
            output_names=("result",),
            python_identity=identity,
            timeout_seconds=1.0,
            max_output_bytes=1024,
        )

    assert killed == [(process.pid, signal.SIGKILL)]
    assert process.wait_count == 2


def test_invalid_report_is_preflight_and_never_publishes_null_hash_receipt(
    tmp_path
):
    source, _, report = _migration(tmp_path)
    report.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "invalid-report-receipt.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "invalid_report"
    assert caught.value.receipt is None
    assert not out.exists()


def test_report_source_path_must_be_absolute_before_receipt(tmp_path):
    source, _, report = _migration(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["analysis"]["scan"]["source_path"] = source.name
    report.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    out = tmp_path / "relative-source-path.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "invalid_report"
    assert caught.value.receipt is None
    assert not out.exists()


def test_runtime_identity_is_frozen_before_user_code_mutates_sys(tmp_path):
    mutating = """\
import sys
import numpy as np
result = np.arange(3, dtype=np.float32)
sys.executable = "/tampered/python"
sys.prefix = "/tampered/prefix"
sys.base_prefix = "/tampered/base"
"""
    source, _, report = _migration(
        tmp_path, source=mutating, migrated=mutating
    )
    identity = verification._python_executable(sys.executable)

    receipt = verify_script(
        source, report, ["result"], tmp_path / "frozen-runtime.json"
    ).to_dict()

    for execution in receipt["executions"].values():
        assert execution["python_executable_realpath"] == identity.realpath
        assert execution["python_executable_sha256"] == identity.file_sha256
        assert execution["python_executable_size"] == identity.file_size
        assert execution["python_prefix"] != "/tampered/prefix"
        assert execution["python_base_prefix"] != "/tampered/base"

    changed = copy.deepcopy(receipt)
    changed["executions"]["source"]["python_prefix"] = "/another/runtime"
    assert verification._verification_id(changed) != receipt["verification_id"]


def test_cumulative_data_limit_stops_before_second_output_file(tmp_path):
    root = tmp_path / "worker-run"
    root.mkdir()
    script = root / "two-outputs.py"
    script.write_text(
        "import numpy as np\n"
        "first = np.zeros((150,), dtype=np.float32)\n"
        "second = np.zeros((150,), dtype=np.float32)\n",
        encoding="utf-8",
    )
    worker = Path(verification.__file__).with_name(
        "_script_verification_worker.py"
    )
    identity = verification._python_executable(sys.executable)

    with pytest.raises(ScriptVerificationError) as caught:
        verification._run_worker(
            root=root,
            label="bounded",
            worker_path=worker,
            script_path=script,
            script_sha256=_sha256(script.read_bytes()),
            worker_sha256=_sha256(worker.read_bytes()),
            output_names=("first", "second"),
            python_identity=identity,
            timeout_seconds=10.0,
            max_output_bytes=1024,
        )

    assert caught.value.code == "output_too_large"
    output_dir = root / "bounded" / "outputs"
    assert (output_dir / "output-0000.npy").is_file()
    assert not (output_dir / "output-0001.npy").exists()


@pytest.mark.parametrize("target", ["source", "migrated"])
def test_hash_drift_from_migration_report_fails_before_execution(tmp_path, target):
    source, migrated, report = _migration(tmp_path)
    path = source if target == "source" else migrated
    path.write_text(path.read_text() + "\n# drift\n", encoding="utf-8")
    out = tmp_path / (target + "-drift-receipt.json")

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    expected = (
        "source_hash_mismatch" if target == "source" else "migrated_hash_mismatch"
    )
    assert caught.value.code == expected
    assert caught.value.receipt is None
    assert not out.exists()


def test_numerical_error_publishes_failed_receipt(tmp_path):
    migrated = """\
import numpy as np
result = np.arange(6, dtype=np.float32).reshape(2, 3) + np.float32(1)
"""
    source, _, report = _migration(tmp_path, migrated=migrated)
    out = tmp_path / "numerical-failure.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "numerical_mismatch"
    receipt = _failed_receipt(caught.value)
    comparison = receipt["comparisons"]["result"]
    assert comparison["consistent"] is False
    assert comparison["failure_path"] == "$"
    assert receipt["strict_npu_verified"] is False
    assert receipt["performance_valid"] is False


def test_missing_output_is_not_inferred_from_stdout(tmp_path):
    migrated = """\
import numpy as np
other = np.arange(6, dtype=np.float32).reshape(2, 3)
print("result=[0, 1, 2, 3, 4, 5]")
"""
    source, _, report = _migration(tmp_path, migrated=migrated)
    out = tmp_path / "missing-output.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "missing_output"
    receipt = _failed_receipt(caught.value)
    assert receipt["failure"]["code"] == "missing_output"
    assert "migrated" not in receipt["executions"]


@pytest.mark.parametrize(
    "source_text",
    [
        """\
import numpy as np
result = np.asarray([object()], dtype=object)
""",
        """\
import numpy as np
result = np.asarray(["not-numeric"], dtype=np.str_)
""",
    ],
    ids=["object", "string"],
)
def test_non_numeric_dtype_is_rejected_without_pickle(tmp_path, source_text):
    source, _, report = _migration(
        tmp_path, source=source_text, migrated=source_text
    )
    out = tmp_path / "object-output.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "unsupported_dtype"
    receipt = _failed_receipt(caught.value)
    assert receipt["failure"]["code"] == "unsupported_dtype"


def test_timeout_kills_worker_and_publishes_no_process_output(tmp_path):
    slow = """\
import time
import numpy as np
time.sleep(5)
result = np.asarray([1.0], dtype=np.float32)
"""
    source, _, report = _migration(tmp_path, source=slow, migrated=slow)
    out = tmp_path / "timeout.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(
            source, report, ["result"], out, timeout_seconds=0.1
        )

    assert caught.value.code == "execution_timeout"
    receipt = _failed_receipt(caught.value)
    assert receipt["failure"]["code"] == "execution_timeout"
    rendered = json.dumps(receipt)
    assert "stdout" not in rendered or "stdout_is_result_channel" in rendered


def test_script_exception_is_structured_without_exception_message(tmp_path):
    migrated = """\
raise RuntimeError("private exception detail must not enter the receipt")
"""
    source, _, report = _migration(tmp_path, migrated=migrated)
    out = tmp_path / "exception.json"

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "script_exception"
    receipt = _failed_receipt(caught.value)
    assert receipt["failure"]["exception_type"] == "RuntimeError"
    assert "private exception detail" not in json.dumps(receipt)


def test_receipt_is_write_once_and_existing_bytes_are_preserved(tmp_path):
    source, _, report = _migration(tmp_path)
    out = tmp_path / "verification.json"
    first = verify_script(source, report, ["result"], out)
    before = out.read_bytes()

    with pytest.raises(ScriptVerificationError) as caught:
        verify_script(source, report, ["result"], out)

    assert caught.value.code == "receipt_exists"
    assert caught.value.receipt is None
    assert out.read_bytes() == before
    assert first.sha256 == _sha256(before)
