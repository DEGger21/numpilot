# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, side-effect-free tests for the interactive migration CLI gate."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpilot.__main__ as cli
from numpilot.interactive_migration import (
    InteractiveMigrationError,
    InterpretationResult,
    read_source_snapshot,
)
from numpilot.qualified_analysis import analyze_qualified_script
from numpilot.script_migration import scan_numpy_script


class _TtyInput(io.StringIO):
    def isatty(self):  # pragma: no cover - exercised through input()
        return True


def _args(source, **overrides):
    values = dict(
        file=str(source),
        source_option=None,
        backend="shim",
        fallback_policy="reject",
        target_soc="unknown",
        cann_version="unknown",
        outputs=None,
        out=None,
        benchmark=False,
        warmup=1,
        samples=7,
        core=None,
        timeout=30.0,
        python=None,
        skip_env_check=True,
        interpret_max_tokens=1800,
        yes=False,
        execute_source=False,
        non_interactive=False,
        cannbot_gap_draft=False,
        accept_cannbot_license=False,
        show_cannbot_skill="none",
        cannbot_skill_root=None,
        cann_root="",
        cannbot_max_tokens=4096,
        cannbot_timeout=30.0,
        skill_path=None,
        llm_config=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _interpretation(source):
    snapshot = read_source_snapshot(source)
    text = "# 迁移前解读\n- 尚未验证"
    return InterpretationResult(
        status="interpreted_unverified",
        source_path=snapshot.path,
        source_sha256=snapshot.sha256,
        source_bytes=snapshot.bytes,
        source_lines=snapshot.lines,
        model="glm-test",
        provider="glm",
        protocol="openai-chat-completions",
        response_model="glm-test",
        receipt_metadata={},
        display_text=text,
        display_format="markdown",
        display_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        actual_remote_call=True,
        elapsed_ms=3,
    )


def test_source_arg_requires_explicit_path_without_a_tty(monkeypatch):
    args = _args("/tmp/not-used.py", file=None)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO())
    with pytest.raises(ValueError, match="非交互"):
        cli._interactive_source_arg(args)


def test_source_arg_prompts_only_on_tty_and_rejects_duplicates(monkeypatch, tmp_path):
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    args = _args(source, file=None)
    monkeypatch.setattr(cli.sys, "stdin", _TtyInput(str(source) + "\n"))
    assert cli._interactive_source_arg(args) == str(source)

    with pytest.raises(ValueError, match="provided once"):
        cli._interactive_source_arg(_args(source, source_option=str(source)))


def test_final_summary_uses_receipts_for_path_scale_and_speedup(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "migration").mkdir(parents=True)
    migrated = run / "migration" / "migrated.py"
    migrated.write_text("import asnumpy as ap\nresult = ap.ones(4)\n", encoding="utf-8")
    workflow = run / "workflow-report.json"
    workflow.write_text("{}\n", encoding="utf-8")
    (run / "script-verification-report.json").write_text(
        '{"status":"passed","comparisons":{"result":{"n_elements":4,"max_abs_diff":0.0,"max_rel_diff":0.0}}}\n',
        encoding="utf-8",
    )
    (run / "script-benchmark-report.json").write_text(
        '{"performance":{"performance_valid":true,"cpu_median_ms":10.0,"npu_median_ms":4.0,"median_speedup":2.5,"conservative_cpu_p10_over_npu_p90":2.0,"median_speedup_bootstrap_95ci":[2.0,3.0]},"workers":{"npu":{"outputs":[{"elements":4,"data_bytes":16,"shape":[4],"dtype":"float32"}]}}}\n',
        encoding="utf-8",
    )
    result = {
        "workflow_report": str(workflow),
        "status": "passed",
        "completed_stage_count": 4,
        "stage_count": 4,
        "stages": [],
    }
    interpretation = SimpleNamespace(
        status="interpreted_unverified", report_path=str(tmp_path / "interpretation.json")
    )

    cli._interactive_final_summary(
        result,
        session_root=tmp_path,
        interpretation=interpretation,
        gap_result={"status": "not_requested"},
    )
    output = capsys.readouterr().out
    assert str(migrated) in output
    assert "[运行验证] status=passed" in output
    assert "speedup=2.50x" in output
    assert "elements=4" in output
    assert "当前规模/当前设备" in output


def test_final_summary_does_not_publish_speedup_when_benchmark_invalid(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    workflow = run / "workflow-report.json"
    workflow.write_text("{}\n", encoding="utf-8")
    (run / "script-benchmark-report.json").write_text(
        '{"performance":{"performance_valid":false,"median_speedup":99.0}}\n',
        encoding="utf-8",
    )
    cli._interactive_final_summary(
        {"workflow_report": str(workflow), "status": "blocked", "stages": []},
        session_root=tmp_path,
        interpretation=SimpleNamespace(status="interpreted_unverified", report_path="x"),
        gap_result={"status": "not_requested"},
    )
    output = capsys.readouterr().out
    assert "performance_valid=false" in output
    assert "未形成可宣称的加速比" in output
    assert "speedup=99" not in output


def test_final_summary_keeps_boundary_when_valid_flag_has_malformed_metrics(
    tmp_path, capsys
):
    """A damaged benchmark receipt must not hide the remaining audit facts."""

    run = tmp_path / "run"
    run.mkdir()
    workflow = run / "workflow-report.json"
    workflow.write_text("{}\n", encoding="utf-8")
    (run / "script-benchmark-report.json").write_text(
        '{"performance":{"performance_valid":true,"median_speedup":"not-a-number"}}\n',
        encoding="utf-8",
    )
    cli._interactive_final_summary(
        {"workflow_report": str(workflow), "status": "blocked", "stages": []},
        session_root=tmp_path,
        interpretation=SimpleNamespace(
            status="interpreted_unverified", report_path="interpretation.json"
        ),
        gap_result={"status": "not_requested"},
    )
    output = capsys.readouterr().out
    assert "数值字段不完整" in output
    assert "[解读边界]" in output


def test_final_summary_ignores_malformed_worker_shape_without_crashing(tmp_path, capsys):
    """A non-object NPU worker entry cannot become a scale claim."""

    run = tmp_path / "run"
    run.mkdir()
    workflow = run / "workflow-report.json"
    workflow.write_text("{}\n", encoding="utf-8")
    (run / "script-benchmark-report.json").write_text(
        '{"performance":{"performance_valid":true,"cpu_median_ms":10,"npu_median_ms":5,"median_speedup":2,"conservative_cpu_p10_over_npu_p90":1.5},"workers":{"npu":[]}}\n',
        encoding="utf-8",
    )
    cli._interactive_final_summary(
        {"workflow_report": str(workflow), "status": "passed", "stages": []},
        session_root=tmp_path,
        interpretation=SimpleNamespace(
            status="interpreted_unverified", report_path="interpretation.json"
        ),
        gap_result={"status": "not_requested"},
    )
    output = capsys.readouterr().out
    assert "speedup=2.00x" in output
    assert "[规模]" in output
    assert "elements=None" in output


def test_interactive_command_decline_never_calls_workflow(monkeypatch, tmp_path, capsys):
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    args = _args(source)
    monkeypatch.setattr(cli.sys, "stdin", _TtyInput("n\n"))
    snapshot = read_source_snapshot(source)
    interpretation = _interpretation(source)
    calls = []

    monkeypatch.setattr(
        cli,
        "_interactive_runtime_analysis",
        lambda _snapshot, _args: ({"imports": [], "decision": "DIRECT"}, SimpleNamespace(ok=True, uses=(), issues=()), SimpleNamespace(unsupported=())),
    )
    monkeypatch.setattr(cli, "_interactive_render_analysis", lambda *args: [])
    monkeypatch.setattr(cli, "_interactive_gap_request", lambda *args: (None, ()))
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: object())
    monkeypatch.setattr(cli, "interpret_source_with_configured_model", lambda *args, **kwargs: interpretation, raising=False)
    # The command imports the helper inside its body, so patch the module
    # attribute rather than relying on an HTTP transport.
    import numpilot.interactive_migration as interactive_module
    monkeypatch.setattr(interactive_module, "interpret_source_with_configured_model", lambda *args, **kwargs: interpretation)
    monkeypatch.setattr(interactive_module, "read_source_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(interactive_module, "write_interpretation_report", lambda path, _result: Path(path))
    monkeypatch.setattr(interactive_module, "failure_report_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_interactive_private_json", lambda path, payload, **kwargs: calls.append((Path(path).name, payload)) or Path(path))
    monkeypatch.setattr(cli, "run_ordinary_workflow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow must not run")), raising=False)
    import numpilot.workflow as workflow_module
    monkeypatch.setattr(workflow_module, "run_ordinary_workflow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow must not run")))

    rc = cli._cmd_interactive_migrate(args)

    assert rc == 0
    assert any(name == "confirmation.json" and payload["status"] == "declined" for name, payload in calls)
    assert "未开始迁移" in capsys.readouterr().out


def test_interactive_command_interpretation_failure_stops_before_confirmation(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    args = _args(source)
    snapshot = read_source_snapshot(source)
    monkeypatch.setattr(cli.sys, "stdin", _TtyInput("y\n"))
    monkeypatch.setattr(
        cli,
        "_interactive_runtime_analysis",
        lambda _snapshot, _args: ({"imports": [], "decision": "DIRECT"}, SimpleNamespace(ok=True, uses=(), issues=()), SimpleNamespace(unsupported=())),
    )
    monkeypatch.setattr(cli, "_interactive_render_analysis", lambda *args: [])
    monkeypatch.setattr(cli, "_interactive_gap_request", lambda *args: (None, ()))
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: object())
    import numpilot.interactive_migration as interactive_module
    monkeypatch.setattr(interactive_module, "read_source_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(
        interactive_module,
        "interpret_source_with_configured_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(InteractiveMigrationError("model failed")),
    )
    monkeypatch.setattr(cli, "_interactive_private_json", lambda *args, **kwargs: Path(args[0]))
    monkeypatch.setattr(cli, "run_ordinary_workflow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow must not run")), raising=False)
    import numpilot.workflow as workflow_module
    monkeypatch.setattr(workflow_module, "run_ordinary_workflow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow must not run")))

    assert cli._cmd_interactive_migrate(args) == 4
    assert "未进入迁移/执行阶段" in capsys.readouterr().err


def test_noninteractive_without_yes_stops_before_model_call(
    monkeypatch, tmp_path, capsys
):
    """A redirected CLI must fail closed before touching configured LLM state."""

    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    args = _args(source, non_interactive=True, yes=False)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO())
    snapshot = read_source_snapshot(source)
    monkeypatch.setattr(
        cli,
        "_interactive_runtime_analysis",
        lambda _snapshot, _args: (
            {"imports": [], "decision": "DIRECT"},
            SimpleNamespace(ok=True, uses=(), issues=()),
            SimpleNamespace(unsupported=()),
        ),
    )
    monkeypatch.setattr(cli, "_interactive_render_analysis", lambda *args: [])
    monkeypatch.setattr(cli, "_interactive_gap_request", lambda *args: (None, ()))
    import numpilot.interactive_migration as interactive_module

    monkeypatch.setattr(interactive_module, "read_source_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: (_ for _ in ()).throw(AssertionError("model/config must not be touched")),
    )
    monkeypatch.setattr(
        interactive_module,
        "interpret_source_with_configured_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("remote interpretation must not run")
        ),
    )

    assert cli._cmd_interactive_migrate(args) == 2
    captured = capsys.readouterr()
    assert "--yes" in captured.err


def test_registered_add_one_is_rendered_as_framework_gap(tmp_path, capsys):
    source = tmp_path / "add_one.py"
    source.write_text(
        "import numpy as np\nx = np.ones(4, dtype=np.float32)\nresult = np.add(x, 1.0)\n",
        encoding="utf-8",
    )
    snapshot = read_source_snapshot(source)
    scan = SimpleNamespace(ok=False, uses=(), issues=())
    classification = SimpleNamespace(decisions=())

    gaps = cli._interactive_render_analysis(
        snapshot,
        {"imports": [], "decision": "BLOCKED", "calls": []},
        scan,
        classification,
    )

    assert len(gaps) == 1
    assert gaps[0]["kind"] == "registered_signature_gap"
    output = capsys.readouterr().out
    assert "asnumpy.add_one" in output
    assert "Ascend C" in output
    assert "asnumpy.ones_like + asnumpy.add" in output


def test_add_one_draft_launches_installed_package_helper(monkeypatch, tmp_path):
    source = Path(__file__).parents[1] / "examples" / "custom_operator_add_one.py"
    snapshot = read_source_snapshot(source)
    args = _args(
        source,
        cannbot_gap_draft=True,
        accept_cannbot_license=True,
        cannbot_skill_root=str(tmp_path / "pinned-cannbot"),
        cann_version="9.0.0",
        target_soc="Ascend910B4",
    )
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if command[1:4] == ["-m", "numpilot", "resolve-gaps"]:
            resolution = Path(command[command.index("--out") + 1])
            resolution.parent.mkdir(parents=True, exist_ok=True)
            resolution.write_text(
                json.dumps(
                    {
                        "status": "custom_operator_generation_allowed",
                        "model_kernel_generation_allowed": True,
                    }
                ),
                encoding="utf-8",
            )
        else:
            output = Path(command[command.index("--out") + 1])
            output.mkdir(parents=True)
            (output / "operator-generation-report.json").write_text(
                json.dumps(
                    {
                        "status": "generated_unverified",
                        "actual_remote_call": True,
                        "generated_file": "operator-draft.cpp",
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli._interactive_cannbot_route(snapshot, None, args, tmp_path)

    assert result["status"] == "generated_unverified"
    helper = commands[1]
    assert helper[1:3] == ["-m", "numpilot.demo_llm_operator_generation"]
    assert "--skill-bundle-sha256" in helper


def test_generic_unintegrated_gap_stops_before_workflow(monkeypatch, tmp_path):
    source = tmp_path / "case.py"
    source.write_text("result = 1\n", encoding="utf-8")
    args = _args(
        source,
        out=str(tmp_path / "session"),
        yes=True,
        non_interactive=True,
        cannbot_gap_draft=True,
        accept_cannbot_license=True,
    )
    snapshot = read_source_snapshot(source)
    interpretation = _interpretation(source)
    monkeypatch.setattr(
        cli,
        "_interactive_runtime_analysis",
        lambda *_args: (
            {"imports": [], "decision": "GAP_REQUIRED", "qualified_apis": []},
            SimpleNamespace(ok=False, uses=(), issues=()),
            SimpleNamespace(unsupported=(object(),)),
        ),
    )
    monkeypatch.setattr(cli, "_interactive_render_analysis", lambda *_args: [object()])
    monkeypatch.setattr(
        cli,
        "_interactive_static_source_admission",
        lambda *_args: {"ok": True, "route": "test", "issues": []},
    )
    monkeypatch.setattr(cli, "_interactive_gap_request", lambda *_args: (tmp_path / "gap.json", (object(),)))
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: object())
    monkeypatch.setattr(
        cli,
        "_interactive_cannbot_route",
        lambda *_args: {
            "status": "generated_unverified",
            "actual_remote_call": True,
            "claim_boundary": {"interface_available": False, "activated": False},
        },
    )
    monkeypatch.setattr(cli, "_interactive_publish_reports", lambda **_kwargs: (Path("a"), Path("b")))
    import numpilot.interactive_migration as interactive_module
    import numpilot.workflow as workflow_module

    monkeypatch.setattr(interactive_module, "read_source_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(interactive_module, "interpret_source_with_configured_model", lambda *_args, **_kwargs: interpretation)
    monkeypatch.setattr(interactive_module, "write_interpretation_report", lambda path, _result: Path(path))
    monkeypatch.setattr(
        workflow_module,
        "run_ordinary_workflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("workflow must not run")),
    )

    assert cli._cmd_interactive_migrate(args) == 1


def test_registered_add_one_uses_normalized_source_for_workflow(monkeypatch, tmp_path):
    source = tmp_path / "case.py"
    source.write_text(
        "import numpy as np\nx=np.ones(4,dtype=np.float32)\nresult=np.add(x,1.0)\n",
        encoding="utf-8",
    )
    normalized = tmp_path / "normalized.py"
    normalized.write_text(
        "import numpy as np\nx=np.ones(4,dtype=np.float32)\n"
        "result=np.add(x,np.ones_like(x,dtype=np.float32))\n",
        encoding="utf-8",
    )
    args = _args(
        source,
        out=str(tmp_path / "session"),
        yes=True,
        non_interactive=True,
        cannbot_gap_draft=True,
        accept_cannbot_license=True,
    )
    original_snapshot = read_source_snapshot(source)
    normalized_snapshot = read_source_snapshot(normalized)
    qualified = analyze_qualified_script(
        original_snapshot.text, original_snapshot.path
    )
    scan = scan_numpy_script(
        original_snapshot.text,
        original_snapshot.path,
        qualified_analysis=qualified,
    )
    interpretation = _interpretation(source)
    calls = []
    monkeypatch.setattr(
        cli,
        "_interactive_runtime_analysis",
        lambda *_args: (
            qualified.to_dict(),
            scan,
            SimpleNamespace(decisions=(), unsupported=()),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_interactive_render_analysis",
        lambda *_args: [{"kind": "registered_signature_gap"}],
    )
    monkeypatch.setattr(cli, "_interactive_gap_request", lambda *_args: (None, ()))
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: object())
    monkeypatch.setattr(
        cli,
        "_interactive_complete_registered_add_one",
        lambda *_args: (
            {
                "status": "qualified_composite_interface",
                "claim_boundary": {"interface_available": True, "activated": True},
            },
            normalized_snapshot,
        ),
    )
    monkeypatch.setattr(cli, "_interactive_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_interactive_publish_reports", lambda **_kwargs: (Path("a"), Path("b")))
    import numpilot.interactive_migration as interactive_module
    import numpilot.workflow as workflow_module

    monkeypatch.setattr(interactive_module, "read_source_snapshot", lambda _path: original_snapshot)
    monkeypatch.setattr(interactive_module, "interpret_source_with_configured_model", lambda *_args, **_kwargs: interpretation)
    monkeypatch.setattr(interactive_module, "write_interpretation_report", lambda path, _result: Path(path))

    def fake_workflow(source_path, out, **_kwargs):
        calls.append((source_path, out))
        return {"status": "passed", "stages": [], "workflow_report": None}

    monkeypatch.setattr(workflow_module, "run_ordinary_workflow", fake_workflow)

    assert cli._cmd_interactive_migrate(args) == 0
    assert calls == [(normalized_snapshot.path, tmp_path / "session" / "workflow")]


@pytest.mark.parametrize(
    "unsafe_statement",
    (
        "import os",
        "open({marker!r}, 'w').write('executed')",
        "np = object()",
        "marker_value: exec(\"open({marker!r}, 'w').write('executed')\")",
        "annotated: open({marker!r}, 'w') = np.ones(4, dtype=np.float32)",
    ),
    ids=(
        "dangerous-import",
        "side-effecting-call",
        "numpy-rebinding",
        "annotation-exec",
        "annotation-open",
    ),
)
def test_unsafe_registered_add_one_stops_before_model_and_execution(
    monkeypatch, tmp_path, capsys, unsafe_statement
):
    marker = tmp_path / "must-not-exist"
    statement = unsafe_statement.format(marker=str(marker))
    source = tmp_path / "unsafe-add-one.py"
    source.write_text(
        "import numpy as np\n"
        + statement
        + "\nx = np.ones(4, dtype=np.float32)\n"
        + "result = np.add(x, 1.0)\n",
        encoding="utf-8",
    )
    snapshot = read_source_snapshot(source)
    qualified = analyze_qualified_script(snapshot.text, snapshot.path)
    scan = scan_numpy_script(
        snapshot.text,
        snapshot.path,
        qualified_analysis=qualified,
    )
    args = _args(
        source,
        out=str(tmp_path / "session"),
        yes=True,
        non_interactive=True,
        cannbot_gap_draft=True,
        accept_cannbot_license=True,
    )
    monkeypatch.setattr(
        cli,
        "_interactive_runtime_analysis",
        lambda *_args: (
            qualified.to_dict(),
            scan,
            SimpleNamespace(decisions=(), unsupported=()),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: (_ for _ in ()).throw(
            AssertionError("static rejection must precede model/config access")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_interactive_publish_reports",
        lambda **_kwargs: (Path("a"), Path("b")),
    )

    assert cli._cmd_interactive_migrate(args) == 2
    assert not marker.exists()
    assert "不会读取模型配置、生成算子或执行用户源码" in capsys.readouterr().err


def test_add_one_source_execution_needs_exact_nearby_confirmation(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.py"
    normalized = tmp_path / "normalized.py"
    source.write_text("result = 1\n", encoding="utf-8")
    normalized.write_text("result = 1\n", encoding="utf-8")
    source_snapshot = read_source_snapshot(source)
    normalized_snapshot = read_source_snapshot(normalized)
    args = _args(source)

    declined_run = tmp_path / "declined"
    declined_run.mkdir(mode=0o700)
    monkeypatch.setattr(cli.sys, "stdin", _TtyInput("y\n"))
    assert not cli._interactive_confirm_add_one_source_execution(
        source_snapshot,
        normalized_snapshot,
        args,
        declined_run,
    )

    accepted_run = tmp_path / "accepted"
    accepted_run.mkdir(mode=0o700)
    monkeypatch.setattr(cli.sys, "stdin", _TtyInput("execute\n"))
    assert cli._interactive_confirm_add_one_source_execution(
        source_snapshot,
        normalized_snapshot,
        args,
        accepted_run,
    )


def test_noninteractive_yes_does_not_authorize_source_execution(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.py"
    normalized = tmp_path / "normalized.py"
    source.write_text("result = 1\n", encoding="utf-8")
    normalized.write_text("result = 1\n", encoding="utf-8")
    source_snapshot = read_source_snapshot(source)
    normalized_snapshot = read_source_snapshot(normalized)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO())

    declined_run = tmp_path / "yes-only"
    declined_run.mkdir(mode=0o700)
    assert not cli._interactive_confirm_add_one_source_execution(
        source_snapshot,
        normalized_snapshot,
        _args(source, yes=True, non_interactive=True),
        declined_run,
    )

    accepted_run = tmp_path / "explicit-execution"
    accepted_run.mkdir(mode=0o700)
    assert cli._interactive_confirm_add_one_source_execution(
        source_snapshot,
        normalized_snapshot,
        _args(
            source,
            yes=True,
            execute_source=True,
            non_interactive=True,
        ),
        accepted_run,
    )
