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

"""Side-effect-free tests for the persistent ``numpilot`` terminal."""

from __future__ import annotations

import io
import re
from types import SimpleNamespace

import numpilot.__main__ as cli
from numpilot.repl import (
    check_model_status,
    normalize_repl_command,
    read_report,
    render_startup_menu,
    run_repl,
    start_new_migration,
)


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_repl_normalizes_human_commands_to_framework_commands():
    assert normalize_repl_command(("check", "llm")) == ("check-llm",)
    assert normalize_repl_command(("numpilot", "check", "llm")) == (
        "check-llm",
    )
    assert normalize_repl_command(("doctor", "--backend", "npu")) == (
        "doctor",
        "--backend",
        "npu",
    )

    migration = normalize_repl_command(("migrate", "/tmp/example.py"))
    assert migration[:2] == ("interactive-migrate", "/tmp/example.py")
    assert "--cannbot-gap-draft" in migration
    assert migration[-2:] == ("--show-cannbot-skill", "full")
    assert "--accept-cannbot-license" not in migration


def test_repl_keeps_session_alive_across_commands(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("NUMPILOT_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    source = _Tty("check llm\nmigrate '/tmp/demo case.py' --no-benchmark\nexit\n")
    output = _Tty()
    diagnostic = _Tty()
    calls: list[tuple[str, ...]] = []

    def dispatch(arguments):
        call = tuple(arguments)
        calls.append(call)
        return 4 if call[0] == "check-llm" else 0

    assert run_repl(
        "1.2.3",
        dispatch,
        input_stream=source,
        output_stream=output,
        error_stream=diagnostic,
    ) == 0

    assert calls[0] == ("check-llm",)
    assert calls[1][:2] == ("interactive-migrate", "/tmp/demo case.py")
    raw = output.getvalue()
    assert "\x1b[" in raw
    rendered = _ANSI.sub("", raw)
    assert "NumPilot" in rendered
    assert "v1.2.3" in rendered
    assert "NumPy → AsNumpy" in rendered
    assert rendered.count("numpilot ›") == 3
    assert "已退出 NumPilot" in rendered
    assert "exit_code=4" in diagnostic.getvalue()


def test_repl_help_and_bad_quote_do_not_end_session(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    source = _Tty("help\n'bad\nexit\n")
    output = _Tty()
    diagnostic = _Tty()

    assert run_repl(
        "test",
        lambda _args: (_ for _ in ()).throw(AssertionError("not dispatched")),
        input_stream=source,
        output_stream=output,
        error_stream=diagnostic,
    ) == 0
    assert "check llm" in output.getvalue()
    assert "migrate" in output.getvalue()
    assert "指令解析失败" in diagnostic.getvalue()
    assert "\x1b" not in output.getvalue()
    assert "\x1b" not in diagnostic.getvalue()


def test_repl_does_not_echo_unexpected_exception_details(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    source = _Tty("boom\nexit\n")
    output = _Tty()
    diagnostic = _Tty()

    def dispatch(_arguments):
        raise RuntimeError("auth_token=must-not-reach-terminal")

    assert run_repl(
        "test",
        dispatch,
        input_stream=source,
        output_stream=output,
        error_stream=diagnostic,
    ) == 0
    assert "RuntimeError" in diagnostic.getvalue()
    assert "must-not-reach-terminal" not in diagnostic.getvalue()
    assert "已退出 NumPilot" in output.getvalue()


def test_empty_cli_argv_starts_repl_when_both_streams_are_ttys(monkeypatch):
    input_stream = _Tty("exit\n")
    output = _Tty()
    diagnostic = _Tty()
    monkeypatch.setattr(cli.sys, "stdin", input_stream)
    monkeypatch.setattr(cli.sys, "stdout", output)
    monkeypatch.setattr(cli.sys, "stderr", diagnostic)
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main([]) == 0
    rendered = output.getvalue()
    assert "NumPy → AsNumpy" in rendered
    assert "numpilot ›" in rendered
    assert "已退出 NumPilot" in rendered


def test_one_shot_check_llm_phrase_uses_canonical_handler(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "_cmd_check_llm",
        lambda args: calls.append(args.cmd) or 7,
    )

    assert cli.main(["check", "llm"]) == 7
    assert calls == ["check-llm"]


def test_modelcheck_alias_and_startup_menu_show_unchecked_status(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert normalize_repl_command(("/modelcheck",)) == ("__modelcheck__",)
    output = _Tty()
    render_startup_menu(
        "1.2.3",
        {"state": "未检查", "detail": "运行 /modelcheck"},
        stream=output,
    )
    rendered = _ANSI.sub("", output.getvalue())
    assert "NumPilot 启动页" in rendered
    assert "未检查" in rendered
    assert "新建迁移" in rendered
    assert "加载迁移记忆" in rendered


def test_check_model_status_reports_available_and_unavailable(monkeypatch):
    import numpilot.llm_check as llm_check

    monkeypatch.setattr(
        llm_check,
        "run_llm_connectivity_check",
        lambda: SimpleNamespace(configured_model="model-x", protocol="openai-chat-completions"),
    )
    output = _Tty()
    assert check_model_status(stream=output)["state"] == "模型可用"
    assert "模型可用" in output.getvalue()

    def fail():
        raise llm_check.LlmConnectivityCheckError("not_configured")

    monkeypatch.setattr(llm_check, "run_llm_connectivity_check", fail)
    output = _Tty()
    assert check_model_status(stream=output)["state"] == "模型不可用"
    assert "模型不可用" in output.getvalue()


def test_start_new_migration_requires_absolute_numpy_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output = _Tty()
    diagnostic = _Tty()
    calls = []

    result = start_new_migration(
        source=_Tty("relative.py\n"),
        output=output,
        diagnostic=diagnostic,
        dispatch=lambda args: calls.append(args) or 0,
    )
    assert result == 0 and not calls
    assert "绝对路径" in diagnostic.getvalue()

    plain = tmp_path / "plain.py"
    plain.write_text("result = 1\n", encoding="utf-8")
    diagnostic = _Tty()
    start_new_migration(
        source=_Tty(str(plain) + "\n"),
        output=output,
        diagnostic=diagnostic,
        dispatch=lambda args: calls.append(args) or 0,
    )
    assert not calls
    assert "不是 NumPy 程序" in diagnostic.getvalue()

    numpy_source = tmp_path / "numpy_case.py"
    numpy_source.write_text("import numpy as np\nresult = np.ones(4)\n", encoding="utf-8")
    calls = []
    start_new_migration(
        source=_Tty(str(numpy_source) + "\n"),
        output=_Tty(),
        diagnostic=_Tty(),
        dispatch=lambda args: calls.append(args) or 0,
    )
    assert calls and calls[0][:2] == ("interactive-migrate", str(numpy_source))
    assert "--out" in calls[0]
    assert any(part.startswith(str(tmp_path / "runs" / "demo-")) for part in calls[0])


def test_slash_commands_are_first_class():
    assert normalize_repl_command(("/help",)) == ("__repl_help__",)
    assert normalize_repl_command(("/exit",)) == ("__repl_exit__",)
    assert normalize_repl_command(("/modelcheck",)) == ("__modelcheck__",)
    migration = normalize_repl_command(("/migrate", "/tmp/example.py"))
    assert migration[:2] == ("interactive-migrate", "/tmp/example.py")
    assert "--cannbot-gap-draft" in migration

    output = _Tty()
    from numpilot.repl import render_repl_help

    render_repl_help(stream=output)
    rendered = _ANSI.sub("", output.getvalue())
    assert "/help" in rendered
    assert "/resume" in rendered
    assert "/modelcheck" in rendered
    assert "/readreport" in rendered
    assert "runs/demo-" in rendered


def test_readreport_renders_latest_markdown(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_COLOR", "1")
    session = tmp_path / "runs" / "demo-20260101-000000"
    session.mkdir(parents=True)
    (session / "migration-report.json").write_text(
        '{"status":"passed","source":{"path":"/workspace/real.py"},'
        '"migration":{},"workflow":{}}\n',
        encoding="utf-8",
    )
    (session / "migration-report.md").write_text(
        "# NumPilot 迁移报告\n\n- 状态：passed\n- 加速比：3.0 x\n",
        encoding="utf-8",
    )
    output = _Tty()
    assert read_report(stream=output) == 0
    rendered = output.getvalue()
    assert "NumPilot 迁移报告" in rendered
    assert "加速比：3.0 x" in rendered


def test_readreport_without_history_is_actionable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_COLOR", "1")
    output = _Tty()
    assert read_report(stream=output) == 1
    assert "没有找到已完成的迁移记录" in output.getvalue()
