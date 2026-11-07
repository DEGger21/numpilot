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

import io
import json
import re
import sys

import pytest
from rich.console import Console

from numpilot import __version__
from numpilot import __main__ as cli
from numpilot import cli_ui
from numpilot.cli_ui import (
    OperatorGenerationTimeline,
    flow_print,
    is_interactive,
    render_home,
    terminal_status,
)
from numpilot.env_check import ERROR, OK, WARN, EnvReport
from numpilot.paths import state_home


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TtyBuffer(io.StringIO):
    def isatty(self):
        return True


def _enable_rich_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("NUMPILOT_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


def _force_console_width(monkeypatch, width):
    monkeypatch.setattr(
        cli_ui,
        "_console",
        lambda stream: Console(
            file=stream,
            highlight=False,
            markup=False,
            soft_wrap=False,
            width=width,
        ),
    )


def test_interactive_detection_honors_terminal_opt_outs():
    stream = TtyBuffer()

    assert is_interactive(stream, {})
    assert not is_interactive(stream, {"NO_COLOR": "1"})
    assert not is_interactive(stream, {"TERM": "dumb"})
    assert not is_interactive(stream, {"NUMPILOT_PLAIN": "1"})
    assert not is_interactive(io.StringIO(), {})


def test_flow_print_adds_semantic_color_without_changing_text(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()
    messages = (
        ("── NumPilot 交互迁移入口 ──", "title"),
        ("[阶段 1/5] 已读取稳定源码", "phase"),
        ("[状态] workflow=passed", "success"),
        ("[算子草稿] generated_unverified", "warning"),
        ("[迁移流水线] failed", "error"),
        ("[性能] speedup=6.64x", "metric"),
        ("[完整报告] /tmp/report.json", "artifact"),
    )

    for message, tone in messages:
        assert flow_print(message, stream=output, tone=tone)

    raw = output.getvalue()
    assert "\x1b[" in raw
    assert _ANSI.sub("", raw) == "".join(message + "\n" for message, _ in messages)
    for color in (31, 32, 33, 35, 36, 94):
        assert re.search(rf"\x1b\[[0-9;]*{color}m", raw)


@pytest.mark.parametrize(
    "environment",
    (
        {"NO_COLOR": "1"},
        {"NUMPILOT_PLAIN": "1"},
        {"TERM": "dumb"},
    ),
)
def test_flow_print_opt_outs_are_exact_plain_text(monkeypatch, environment):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("NUMPILOT_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    output = TtyBuffer()
    message = "[状态] workflow=passed"

    assert not flow_print(message, stream=output, tone="success")
    assert output.getvalue() == message + "\n"
    assert "\x1b" not in output.getvalue()


def test_flow_print_sanitizes_terminal_control_in_tty(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()

    flow_print(
        "[源码] safe\x1b]8;;https://invalid.example\x07forged\x1b[31m",
        stream=output,
        tone="artifact",
    )

    raw = output.getvalue()
    assert "\x1b]" not in raw
    assert "\x07" not in raw
    rendered = _ANSI.sub("", raw)
    assert "safe" in rendered
    assert "forged" in rendered
    assert "�" in rendered


def test_operator_generation_timeline_plain_fallback_is_stable():
    output = io.StringIO()
    timeline = OperatorGenerationTimeline(
        "add_one", route="CANNBot Skill → Ascend C", stream=output, enabled=False
    )

    timeline.start()
    timeline.update("route", "done", "official-first resolution accepted")
    timeline.update("generation", "running", "model is writing candidate")
    timeline.fail("generation", "candidate gate blocked")
    timeline.finish("candidate remains blocked")

    rendered = output.getvalue()
    assert "[算子流程]" in rendered
    assert "CANNBot Skill → Ascend C" in rendered
    assert "模型写算子" in rendered
    assert "candidate gate blocked" in rendered
    assert "\x1b" not in rendered
    assert timeline.states["route"] == "done"
    assert timeline.states["generation"] == "blocked"
    with pytest.raises(ValueError, match="invalid operator generation transition"):
        timeline.update("route", "running")
    with pytest.raises(ValueError, match="invalid operator generation transition"):
        timeline.update("generation", "done")


def test_operator_generation_timeline_rich_panel_shows_all_gates(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()
    _force_console_width(monkeypatch, 110)
    timeline = OperatorGenerationTimeline(
        "add_one", route="CANNBot Skill → Ascend C", stream=output, enabled=True
    )

    timeline.start()
    timeline.update("route", "done", "official-first resolution accepted")
    timeline.update("skill", "done", "bundle verified")
    timeline.update("context", "done", "bounded official context")
    timeline.update("generation", "running", "model is writing candidate")

    rendered = _ANSI.sub("", output.getvalue())
    assert "NumPilot · 算子生成 · add_one" in rendered
    assert "CANNBot Skill → Ascend C" in rendered
    for stage in ("缺口路由", "CANNBot Skill", "官方上下文", "模型写算子", "候选校验", "草稿落盘"):
        assert stage in rendered
    assert "model is writing candidate" in rendered
    assert "\x1b" not in rendered


def test_empty_argv_renders_product_home_only_on_tty(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)

    assert cli.main([]) == 0
    rendered = _ANSI.sub("", output.getvalue())
    assert "NumPilot" in rendered
    assert "╔" in rendered
    assert __version__ in rendered
    assert "numpilot doctor --backend npu" in rendered
    assert "numpilot migrate <script.py>" in rendered
    # Rich may wrap long command cells on a normal 80-column terminal.
    assert "numpilot run <script.py>" in rendered
    assert "--benchmark" in rendered


def test_empty_argv_keeps_argparse_error_when_redirected(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 2


def test_empty_argv_no_color_keeps_plain_product_home(monkeypatch):
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main([]) == 0
    assert output.getvalue().startswith("NumPilot v")
    assert "numpilot migrate <script.py>" in output.getvalue()
    assert "numpilot run <script.py> --benchmark" in output.getvalue()
    assert "\x1b" not in output.getvalue()


def test_paths_get_is_exact_scalar_even_on_tty(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)

    assert cli.main(["paths", "--get", "state"]) == 0
    assert output.getvalue() == str(state_home()) + "\n"


def test_paths_json_remains_one_clean_document_on_tty(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)

    assert cli.main(["paths", "--json"]) == 0
    payload = json.loads(output.getvalue())
    assert payload["state"] == str(state_home())
    assert "\x1b" not in output.getvalue()
    assert "\r" not in output.getvalue()


def test_bare_paths_uses_table_on_tty(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)

    assert cli.main(["paths"]) == 0
    rendered = _ANSI.sub("", output.getvalue())
    assert "NumPilot 用户目录" in rendered
    assert "bootstrap_receipt" in rendered
    assert "llm_config" in rendered


def test_no_color_keeps_bare_paths_plain(monkeypatch):
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main(["paths"]) == 0
    assert output.getvalue().startswith("NumPilot 路径（本命令不会创建目录）:")
    assert "\x1b" not in output.getvalue()


def test_compact_diagnostic_rejects_impossible_width_and_preserves_counts():
    with pytest.raises(ValueError, match="max_chars"):
        cli_ui.compact_diagnostic("detail", "long", max_chars=2)

    detail = ";".join(
        (
            "冲突 toolkit=/opt/a,/opt/b",
            "devlib=/opt/c",
            "重复=/opt/d,/opt/e",
            "同根别名=/opt/f",
            "extra=" + ("x" * 200),
        )
    )
    rendered = cli_ui.compact_diagnostic(
        "CANN 动态库路径", detail, max_chars=180
    )
    assert "toolkit 冲突 2 处" in rendered
    assert "重复 2 处" in rendered
    assert "numpilot doctor --json" in rendered

    for width in range(8, 30):
        bounded = cli_ui.compact_diagnostic("detail", "A" * 500, max_chars=width)
        assert len(bounded) <= width
        bounded_cann = cli_ui.compact_diagnostic(
            "CANN 动态库路径", detail, max_chars=width
        )
        assert len(bounded_cann) <= width


def test_compact_diagnostic_strips_terminal_controls():
    rendered = cli_ui.compact_diagnostic(
        "detail", "safe\x1b[31m\r\nforged=PASS"
    )
    assert "\x1b" not in rendered
    assert "�" in rendered
    assert "forged=PASS" in rendered


@pytest.mark.parametrize(
    "status, expected_rc, expected_summary",
    [
        (OK, 0, "环境检查通过"),
        (WARN, 0, "1 个警告"),
        (ERROR, 1, "1 个错误"),
    ],
)
def test_doctor_tty_table_preserves_exit_contract(
    monkeypatch, status, expected_rc, expected_summary
):
    _enable_rich_terminal(monkeypatch)
    report = EnvReport("npu")
    report.add("AsNumpy", status, "safe detail")
    monkeypatch.setattr(cli.Env, "check", staticmethod(lambda backend: report))
    output = TtyBuffer()
    diagnostic = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", diagnostic)

    assert cli.main(["doctor", "--backend", "npu"]) == expected_rc
    rendered = _ANSI.sub("", output.getvalue())
    progress = _ANSI.sub("", diagnostic.getvalue())
    expected_marker = {OK: "✓", WARN: "⚠", ERROR: "✗"}[status]
    assert expected_marker in progress
    assert "环境自检" in rendered
    assert "╭" in rendered
    assert "AsNumpy" in rendered
    assert "safe detail" in rendered
    assert expected_summary in rendered


def test_render_home_is_inert_for_non_tty():
    output = io.StringIO()

    assert not render_home("1.2.3", stream=output)
    assert output.getvalue() == ""


def test_terminal_status_has_durable_tty_outcomes(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    diagnostic = TtyBuffer()
    monkeypatch.setattr(sys, "stderr", diagnostic)

    with terminal_status("成功阶段"):
        pass
    with terminal_status("警告阶段") as progress:
        progress.warning()
    with terminal_status("失败阶段") as progress:
        progress.error()

    rendered = _ANSI.sub("", diagnostic.getvalue())
    assert "✓ 成功阶段" in rendered
    assert "⚠ 警告阶段" in rendered
    assert "✗ 失败阶段" in rendered
    assert re.search(r"成功阶段\s+\d+\.\ds", rendered)


def test_terminal_status_marks_exception_and_reraises(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    diagnostic = TtyBuffer()
    monkeypatch.setattr(sys, "stderr", diagnostic)

    with pytest.raises(RuntimeError, match="sentinel"):
        with terminal_status("抛出异常"):
            raise RuntimeError("sentinel")

    assert "✗ 抛出异常" in _ANSI.sub("", diagnostic.getvalue())


def test_terminal_status_sanitizes_dynamic_terminal_controls(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    diagnostic = TtyBuffer()
    monkeypatch.setattr(sys, "stderr", diagnostic)

    with terminal_status(
        "backend=npu\x1b[2J"
        "\x1b]8;;https://invalid.example\x07forged"
        "\x1b]0;title\x1b\\tail"
    ):
        pass

    raw = diagnostic.getvalue()
    assert "\x1b[2J" not in raw
    assert "\x1b]" not in raw
    assert "\x07" not in raw
    rendered = _ANSI.sub("", raw)
    assert "backend=npu" in rendered
    assert "forged" in rendered
    assert "tail" in rendered
    assert "�" in rendered


@pytest.mark.parametrize("environment", ({}, {"NO_COLOR": "1"}))
def test_terminal_status_is_silent_without_interactive_tty(
    monkeypatch, environment
):
    diagnostic = io.StringIO() if not environment else TtyBuffer()
    monkeypatch.setattr(sys, "stderr", diagnostic)
    monkeypatch.delenv("NO_COLOR", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with terminal_status("不应渲染"):
        pass

    assert diagnostic.getvalue() == ""


def test_doctor_narrow_tty_keeps_diagnostic_detail(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    _force_console_width(monkeypatch, 30)
    report = EnvReport("npu")
    report.add("AsNumpy", ERROR, "DETAIL_SENTINEL")
    monkeypatch.setattr(cli.Env, "check", staticmethod(lambda backend: report))
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    assert cli.main(["doctor", "--backend", "npu"]) == 1
    rendered = _ANSI.sub("", output.getvalue())
    assert "DETAIL_SENTINEL" in rendered
    assert "AsNumpy" in rendered


@pytest.mark.parametrize("width", (30, 100))
def test_doctor_tty_sanitizes_environment_text_in_all_layouts(
    monkeypatch, width
):
    _enable_rich_terminal(monkeypatch)
    _force_console_width(monkeypatch, width)
    report = EnvReport("npu\x1b]0;BT\x07")
    report.add(
        "AsNumpy\x1b[2JNM",
        ERROR,
        "D0\x1b]8;;x\x07D1\x1b]0;y\x1b\\D2",
    )
    output = TtyBuffer()

    assert cli_ui.render_doctor(report, stream=output)

    raw = output.getvalue()
    assert "\x1b[2J" not in raw
    assert "\x1b]" not in raw
    assert "\x07" not in raw
    rendered = _ANSI.sub("", raw)
    assert "BT" in rendered
    assert "NM" in rendered
    assert "D0" in rendered
    assert "D1" in rendered
    assert "D2" in rendered
    assert "�" in rendered
    # Remaining CSI bytes are emitted by Rich for the intended styles.
    assert "\x1b[" in raw


def test_paths_narrow_tty_keeps_full_path_value(monkeypatch):
    _enable_rich_terminal(monkeypatch)
    _force_console_width(monkeypatch, 30)
    monkeypatch.setattr(
        cli,
        "path_summary",
        lambda: {"state": "/PATH_VALUE_SENTINEL"},
    )
    output = TtyBuffer()
    monkeypatch.setattr(sys, "stdout", output)

    assert cli.main(["paths"]) == 0
    rendered = _ANSI.sub("", output.getvalue())
    assert "/PATH_VALUE_SENTINEL" in rendered


@pytest.mark.parametrize("width", (30, 100))
def test_paths_tty_sanitizes_names_and_values_in_all_layouts(
    monkeypatch, width
):
    _enable_rich_terminal(monkeypatch)
    _force_console_width(monkeypatch, width)
    output = TtyBuffer()
    paths = {
        "state\x1b[2JNM": "/P0\x1b]8;;x\x07P1",
        "config": "/P2\x1b]0;y\x1b\\P3",
    }

    assert cli_ui.render_paths(paths, stream=output)

    raw = output.getvalue()
    assert "\x1b[2J" not in raw
    assert "\x1b]" not in raw
    assert "\x07" not in raw
    rendered = _ANSI.sub("", raw)
    assert "NM" in rendered
    assert "P0" in rendered
    assert "P1" in rendered
    assert "P2" in rendered
    assert "P3" in rendered
    assert "�" in rendered
    # Names and paths retain their Rich emphasis/dim presentation.
    assert "\x1b[" in raw
