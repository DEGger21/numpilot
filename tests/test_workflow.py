import json
import hashlib
from pathlib import Path

import pytest

from numpilot import __main__ as cli
from numpilot import workflow


def test_workflow_rejects_existing_output_without_touching_it(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\nresult = np.ones((2, 2))\n", encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out)

    assert caught.value.code == "output_exists"
    assert list(out.iterdir()) == []


def test_workflow_rejects_dotdot_path_that_would_cross_symlink(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    # ``linked/../run`` is normalized to ``tmp_path/run`` by Python, but the
    # kernel follows ``linked`` first; admission must reject the lexical
    # symlink instead of relying on the normalized spelling.
    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, linked / ".." / "run")

    assert caught.value.code == "unsafe_path"
    assert not (outside / "run").exists()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_workflow_wraps_pathlike_fspath_errors_before_creating_run(
    tmp_path, error_type
):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    class ExplodingPath:
        def __fspath__(self):
            raise error_type("path conversion failed")

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, ExplodingPath())

    assert caught.value.code == "invalid_path"
    assert not out.exists()


def test_workflow_rejects_unicode_format_in_output_path_before_creating_run(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run\u202eoutput"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out)

    assert caught.value.code == "invalid_path"
    assert not out.exists()


def test_workflow_stops_at_first_failed_stage_and_cleans_temps(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\nresult = np.ones((2, 2))\n", encoding="utf-8")
    out = tmp_path / "run"

    def fake_stage(name, argv, root, *, cwd, timeout_seconds):
        stdout = root / (name + ".stdout.log")
        stderr = root / (name + ".stderr.log")
        workflow._write_private(stdout, b"")
        workflow._write_private(stderr, b"blocked by test\n")
        return workflow.WorkflowStage(
            name=name,
            argv=tuple(argv),
            returncode=1,
            status="failed",
            elapsed_seconds=0.01,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=workflow._sha256(b""),
            stderr_sha256=workflow._sha256(b"blocked by test\n"),
        )

    monkeypatch.setattr(workflow, "_run_stage", fake_stage)
    result = workflow.run_ordinary_workflow(source, out)

    assert result["status"] == "blocked"
    assert result["stage_count"] == 1
    assert result["claims"] == {
        "model_generated": False,
        "numerically_verified": False,
        "strict_npu_performance_verified": False,
    }
    report_path = out / "workflow-report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["workflow_report"] == str(report_path)
    assert not list(out.glob("*.tmp"))
    assert not list(out.glob(".*.tmp"))


def test_workflow_classifies_model_gate_as_blocked(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\nresult = np.ones((2, 2))\n", encoding="utf-8")
    out = tmp_path / "run"

    def fake_stage(name, argv, root, *, cwd, timeout_seconds):
        del argv, cwd, timeout_seconds
        stdout = root / (name + ".stdout.log")
        stderr = root / (name + ".stderr.log")
        workflow._write_private(stdout, b"")
        workflow._write_private(stderr, b"configured model unavailable\n")
        return workflow.WorkflowStage(
            name=name,
            argv=("python",),
            returncode=4,
            status="failed",
            elapsed_seconds=0.01,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=workflow._sha256(b""),
            stderr_sha256=workflow._sha256(b"configured model unavailable\n"),
        )

    monkeypatch.setattr(workflow, "_run_stage", fake_stage)
    result = workflow.run_ordinary_workflow(source, out)

    assert result["status"] == "blocked"
    assert result["next_action"].startswith("inspect the failed stage")


def test_workflow_validates_output_names_before_creating_run(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError, match="outputs"):
        workflow.run_ordinary_workflow(source, out, outputs=("result", "bad-name"))

    assert not out.exists()


def test_workflow_validates_python_before_creating_run(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError, match="python executable"):
        workflow.run_ordinary_workflow(source, out, python_executable="")

    assert not out.exists()


@pytest.mark.parametrize("value", [None, [], {}])
def test_workflow_rejects_non_string_backend_before_creating_run(tmp_path, value):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out, backend=value)

    assert caught.value.code == "invalid_backend"
    assert not out.exists()


@pytest.mark.parametrize("value", [None, [], {}])
def test_workflow_rejects_non_string_fallback_policy_before_creating_run(
    tmp_path, value
):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out, fallback_policy=value)

    assert caught.value.code == "invalid_policy"
    assert not out.exists()


@pytest.mark.parametrize("value", [None, "result", b"result", 123])
def test_workflow_rejects_invalid_outputs_container_before_creating_run(
    tmp_path, value
):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out, outputs=value)

    assert caught.value.code == "invalid_outputs"
    assert not out.exists()


def test_workflow_matches_downstream_python_identifier_contract(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out, outputs=("1result",))

    assert caught.value.code == "invalid_outputs"
    assert not out.exists()


@pytest.mark.parametrize("argument", ["benchmark", "skip_env_check"])
def test_workflow_rejects_non_boolean_switches_before_creating_run(
    tmp_path, argument
):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out, **{argument: "yes"})

    assert caught.value.code in {"invalid_benchmark", "invalid_environment"}
    assert not out.exists()


def test_workflow_bounds_output_iterable_before_creating_run(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(
            source,
            out,
            outputs=iter(["result"] * (workflow.MAX_OUTPUT_NAMES + 1)),
        )

    assert caught.value.code == "invalid_outputs"
    assert not out.exists()


@pytest.mark.parametrize(
    ("argument", "value", "code"),
    [
        ("target_soc", None, "invalid_target_soc"),
        ("target_soc", "Ascend910B4\n", "invalid_target_soc"),
        (
            "target_soc",
            "x" * (workflow.MAX_TARGET_TEXT_LENGTH + 1),
            "invalid_target_soc",
        ),
        ("cann_version", None, "invalid_cann_version"),
        ("cann_version", "9.0.0\x00", "invalid_cann_version"),
        (
            "cann_version",
            "x" * (workflow.MAX_TARGET_TEXT_LENGTH + 1),
            "invalid_cann_version",
        ),
        ("python_executable", "\x1bpython", "invalid_python"),
        (
            "python_executable",
            "x" * (workflow.MAX_OPTION_TEXT_LENGTH + 1),
            "invalid_python",
        ),
    ],
)
def test_workflow_bounds_text_options_before_creating_run(
    tmp_path, argument, value, code
):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out, **{argument: value})

    assert caught.value.code == code
    assert not out.exists()


def test_workflow_rejects_oversized_source_before_creating_run(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"x" * (workflow.MAX_SOURCE_BYTES + 1))
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError) as caught:
        workflow.run_ordinary_workflow(source, out)

    assert caught.value.code == "source_too_large"
    assert not out.exists()


def test_source_hash_uses_bounded_descriptor_reads(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    payload = b"result = 1\n"
    source.write_bytes(payload)

    # The admission hash must not fall back to Path.read_bytes(), whose
    # allocation is unbounded if a concurrent writer grows the source after
    # the initial stat.
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("source hash used an unbounded read"),
    )

    assert workflow._source_hash(source) == hashlib.sha256(payload).hexdigest()


def test_source_hash_rejects_growth_beyond_bound(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"x")
    real_read = workflow.os.read

    def oversized_read(fd, size):
        # Return one byte more than requested to model a racing/hostile read
        # implementation; the helper must fail closed before another read.
        return real_read(fd, size) + b"x" * (workflow.MAX_SOURCE_BYTES + 1)

    monkeypatch.setattr(workflow.os, "read", oversized_read)
    assert workflow._source_hash(source) is None


def test_workflow_shim_lane_keeps_analysis_advisory_and_never_benchmarks(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\nresult = np.ones((2, 2))\n", encoding="utf-8")
    out = tmp_path / "run"
    calls = []
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    def fake_stage(name, argv, root, *, cwd, timeout_seconds):
        calls.append((name, tuple(argv)))
        argv = tuple(argv)
        stdout = root / (name + ".stdout.log")
        stderr = root / (name + ".stderr.log")
        workflow._write_private(stdout, b"")
        workflow._write_private(stderr, b"")
        output = Path(argv[argv.index("--out") + 1])
        if name == "01-analyze":
            workflow._write_private(
                output,
                json.dumps({"source_sha256": source_sha}).encode("utf-8"),
            )
        elif name == "02-migrate":
            output.mkdir(mode=0o700)
            workflow._write_private(
                output / "script-migration-report.json",
                json.dumps(
                    {"source_sha256": source_sha, "actual_remote_call": True}
                ).encode("utf-8"),
            )
        elif name == "03-verify":
            workflow._write_private(
                output,
                json.dumps(
                    {"status": "passed", "source": {"sha256": source_sha}}
                ).encode("utf-8"),
            )
        return workflow.WorkflowStage(
            name=name,
            argv=argv,
            returncode=0,
            status="passed",
            elapsed_seconds=0.01,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=workflow._sha256(b""),
            stderr_sha256=workflow._sha256(b""),
        )

    monkeypatch.setattr(workflow, "_run_stage", fake_stage)
    result = workflow.run_ordinary_workflow(
        source,
        out,
        backend="shim",
        fallback_policy="accept",
    )

    assert result["status"] == "passed"
    assert [name for name, _ in calls] == ["01-analyze", "02-migrate", "03-verify"]
    assert "--require-direct" not in calls[0][1]


def test_workflow_blocks_when_source_changes_between_stages(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\nresult = np.ones((2, 2))\n", encoding="utf-8")
    out = tmp_path / "run"
    initial_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []

    def fake_stage(name, argv, root, *, cwd, timeout_seconds):
        del cwd, timeout_seconds
        calls.append(name)
        argv = tuple(argv)
        stdout = root / (name + ".stdout.log")
        stderr = root / (name + ".stderr.log")
        workflow._write_private(stdout, b"")
        workflow._write_private(stderr, b"")
        output = Path(argv[argv.index("--out") + 1])
        if name == "01-analyze":
            workflow._write_private(
                output,
                json.dumps({"source_sha256": initial_sha}).encode("utf-8"),
            )
            source.write_text(
                "import numpy as np\nresult = np.zeros((2, 2))\n",
                encoding="utf-8",
            )
        return workflow.WorkflowStage(
            name=name,
            argv=argv,
            returncode=0,
            status="passed",
            elapsed_seconds=0.01,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=workflow._sha256(b""),
            stderr_sha256=workflow._sha256(b""),
        )

    monkeypatch.setattr(workflow, "_run_stage", fake_stage)
    result = workflow.run_ordinary_workflow(
        source, out, backend="shim", fallback_policy="accept"
    )

    assert calls == ["01-analyze"]
    assert result["status"] == "blocked"
    assert result["source_changed"] is True
    assert result["source_guard"]["status"] == "blocked"
    assert result["claims"] == {
        "model_generated": False,
        "numerically_verified": False,
        "strict_npu_performance_verified": False,
    }


def test_workflow_rejects_shim_benchmark_before_creating_output(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError, match="backend=npu"):
        workflow.run_ordinary_workflow(source, out, backend="shim", benchmark=True)

    assert not out.exists()


def test_workflow_rejects_too_few_benchmark_samples_before_creating_output(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"

    with pytest.raises(workflow.WorkflowError, match="at least 7"):
        workflow.run_ordinary_workflow(
            source,
            out,
            backend="npu",
            benchmark=True,
            samples=6,
        )

    assert not out.exists()


def test_workflow_log_tail_replaces_terminal_controls(tmp_path):
    log = tmp_path / "stage.stderr.log"
    log.write_text("\x1b[31mBAD\x1b[0m\r\nsafe\u202e text\n", encoding="utf-8")

    rendered = cli._workflow_log_tail(log)

    assert "\x1b" not in rendered
    assert "\r" not in rendered
    assert "\u202e" not in rendered
    assert "BAD" in rendered
    assert "safe" in rendered


def test_cli_run_does_not_create_default_parent_before_admission(tmp_path, monkeypatch):
    demo_root = tmp_path / "demo"
    missing = tmp_path / "missing.py"
    monkeypatch.setattr(cli, "default_demo_dir", lambda: demo_root)

    # Invalid source admission is handled as a stable CLI error, and should
    # not leave an empty default runs/demo directory behind.
    assert cli.main(["run", str(missing), "--backend", "shim"]) == 2
    assert not demo_root.exists()


def test_cli_run_wraps_default_output_path_resolution_error(monkeypatch, capsys):
    def fail_default_root():
        raise RuntimeError("private home details")

    monkeypatch.setattr(cli, "default_demo_dir", fail_default_root)
    args = type("Args", (), {"out": None})()

    assert cli._cmd_run(args) == 2
    captured = capsys.readouterr()
    assert "invalid_path" in captured.err
    assert "private home details" not in captured.err
    assert captured.out == ""


def test_cli_run_wraps_explicit_output_path_resolution_error(capsys):
    # Direct callers can supply an object whose fspath conversion fails even
    # though argparse itself normally gives us a string.
    class ExplodingPath:
        def __fspath__(self):
            raise OSError("sensitive path details")

    args = type("Args", (), {"out": ExplodingPath()})()

    assert cli._cmd_run(args) == 2
    captured = capsys.readouterr()
    assert "invalid_path" in captured.err
    assert "sensitive path details" not in captured.err
    assert captured.out == ""


def test_workflow_does_not_read_receipt_symlink_outside_output(tmp_path, monkeypatch):
    """A child-produced receipt symlink must fail closed without leaking data."""

    source = tmp_path / "source.py"
    source.write_text("result = 1\n", encoding="utf-8")
    out = tmp_path / "run"
    secret = tmp_path / "secret.json"
    secret.write_text(
        json.dumps(
            {
                "source_sha256": "leaked-secret",
                "actual_remote_call": True,
                "token": "should-not-appear",
            }
        ),
        encoding="utf-8",
    )

    def fake_stage(name, argv, root, *, cwd, timeout_seconds):
        del argv, cwd, timeout_seconds
        stdout = root / (name + ".stdout.log")
        stderr = root / (name + ".stderr.log")
        workflow._write_private(stdout, b"")
        workflow._write_private(stderr, b"")
        if name == "01-analyze":
            (root / "qualified-analysis.json").symlink_to(secret)
        return workflow.WorkflowStage(
            name=name,
            argv=("python",),
            returncode=0,
            status="passed",
            elapsed_seconds=0.01,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=workflow._sha256(b""),
            stderr_sha256=workflow._sha256(b""),
        )

    monkeypatch.setattr(workflow, "_run_stage", fake_stage)
    result = workflow.run_ordinary_workflow(
        source, out, backend="shim", fallback_policy="accept"
    )

    assert result["status"] == "blocked"
    assert result["source_guard"]["status"] == "blocked"
    report_text = (out / "workflow-report.json").read_text(encoding="utf-8")
    assert "leaked-secret" not in report_text
    assert "should-not-appear" not in report_text
    assert result["receipts"]["analysis"] is None


def _write_replay_fixture(root: Path) -> Path:
    root.mkdir(mode=0o700)
    analysis = root / "qualified-analysis.json"
    analysis.write_text(
        json.dumps({"schema_version": 1, "status": "DIRECT"}) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": workflow.WORKFLOW_SCHEMA_VERSION,
        "record_type": "numpilot_ordinary_workflow",
        "status": "passed",
        "source_path": str(root / "source.py"),
        "source_sha256": "a" * 64,
        "stages": [{"name": "01-analyze", "status": "passed"}],
        "receipts": {"analysis": str(analysis), "migration": None, "verification": None, "benchmark": None},
        "claims": {"numerically_verified": False},
    }
    (root / "workflow-report.json").write_text(
        json.dumps(report) + "\n", encoding="utf-8"
    )
    return root


def test_replay_workflow_is_read_only_and_validates_receipts(tmp_path):
    run = _write_replay_fixture(tmp_path / "run")
    before = (run / "workflow-report.json").read_bytes()

    result = workflow.replay_workflow(run)

    assert result["record_type"] == "numpilot_workflow_replay"
    assert result["receipt_integrity"]["status"] == "passed"
    assert result["replay_mode"] == "read_only"
    assert result["execution_performed"] is False
    assert result["actual_remote_call"] is False
    assert (run / "workflow-report.json").read_bytes() == before
    assert not (run / "replay.json").exists()


def test_replay_cli_emits_machine_readable_summary(tmp_path, capsys):
    run = _write_replay_fixture(tmp_path / "run")

    assert cli.main(["replay", str(run), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["receipt_integrity"]["status"] == "passed"
    assert payload["execution_performed"] is False


def test_replay_workflow_fails_closed_when_receipt_escapes_run(tmp_path):
    run = _write_replay_fixture(tmp_path / "run")
    report_path = run / "workflow-report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["receipts"]["analysis"] = str(tmp_path / "outside.json")
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = workflow.replay_workflow(run)

    assert result["receipt_integrity"]["status"] == "failed"
    assert "escapes the run directory" in result["receipt_integrity"]["errors"][0]
