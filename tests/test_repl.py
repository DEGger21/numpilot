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

import numpilot.__main__ as cli
from numpilot.repl import normalize_repl_command, run_repl


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
