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

"""TTY-aware presentation helpers for NumPilot's human CLI.

Dynamic output is deliberately limited to real terminals.  Redirected output,
machine-readable commands, ``NO_COLOR``, and ``TERM=dumb`` keep using the
stable plain-text renderers owned by each command.
"""

from contextlib import contextmanager
import os
import re
import sys
import time
import unicodedata

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_STATUS_STYLE = {
    "ok": ("✓", "green"),
    "warn": ("⚠", "yellow"),
    "error": ("✗", "red"),
}

_READINESS_STYLE = {
    "pass": ("✓", "green"),
    "partial": ("~", "yellow"),
    "warn": ("!", "yellow"),
    "blocked": ("✗", "red"),
    "info": ("i", "cyan"),
}


def _terminal_safe(value):
    """Return one printable line suitable for a human terminal.

    Environment probes and local evidence metadata can contain terminal
    control characters (for example an injected ESC in a library path).  Rich
    ``Text`` disables markup parsing, but it intentionally does not strip raw
    ANSI bytes; sanitize at this presentation boundary instead.
    """

    rendered = str(value)
    rendered = rendered.replace("\r\n", "\n")
    safe = []
    for character in rendered:
        if character in {"\t", "\n", "\r"}:
            safe.append("�")
        elif (
            not character.isprintable()
            or unicodedata.category(character).startswith("C")
        ):
            safe.append("�")
        else:
            safe.append(character)
    return "".join(safe) or "—"


def _fit_terminal_text(rendered, max_chars):
    """Bound a sanitized diagnostic while retaining a useful JSON hint."""

    if len(rendered) <= max_chars:
        return rendered
    suffix = "…（详情请运行 --json）"
    # Keep the result bounded even when a caller requests a very small
    # diagnostic width.  In particular, ``text[-0:]`` means the *whole*
    # string in Python, so an arithmetic slice here must never receive zero.
    budget = max_chars - len(suffix) - 1  # one code point for the middle ellipsis
    if budget <= 0:
        return suffix[:max_chars]
    head = budget // 2
    tail = budget - head
    return (
        rendered[:head]
        + "…"
        + (rendered[-tail:] if tail else "")
        + suffix
    )


def compact_diagnostic(name, detail, *, max_chars=240):
    """Make a human-facing diagnostic bounded without weakening its meaning.

    Environment probes can return a complete PATH/LD_LIBRARY_PATH inventory.
    That is useful in JSON receipts, but rendering every absolute path in a
    60-column terminal hides the actual pass/fail result.  Keep counts and
    the severity class visible and direct the operator to the machine-readable
    command for the full list.  Short details are returned unchanged so this
    helper is safe to use for ordinary component diagnostics as well.
    """

    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars < 8
    ):
        raise ValueError("max_chars must be an integer >= 8")
    if detail is None:
        return "—"
    rendered = _terminal_safe(detail)
    if len(rendered) <= max_chars:
        return rendered

    if name == "CANN 动态库路径":
        # ``Env`` emits semicolon-separated groups such as
        # ``重复=PATH[1]=...,PATH[2]=...``.  Count entries without exposing
        # long host-specific paths in the visual table.
        groups = []
        for label, display in (
            ("冲突 toolkit", "toolkit 冲突"),
            ("devlib", "devlib"),
            ("重复", "重复"),
            ("同根别名", "同根别名"),
        ):
            match = re.search(
                rf"(?:^|;)\s*{re.escape(label)}=([^;]*)", rendered
            )
            if match is None:
                continue
            values = [item for item in match.group(1).split(",") if item]
            groups.append(f"{display} {len(values)} 处")
        if groups:
            summary = (
                "发现 " + "、".join(groups)
                + "；完整路径请运行 `numpilot doctor --json`"
            )
            return _fit_terminal_text(summary, max_chars)

    # Preserve both ends: the beginning normally contains the error class,
    # while the tail often contains the concrete component that failed.
    return _fit_terminal_text(rendered, max_chars)


class _StatusOutcome:
    """Mutable outcome returned by :func:`terminal_status`."""

    def __init__(self):
        self.value = "ok"

    def warning(self):
        if self.value != "error":
            self.value = "warn"

    def error(self):
        self.value = "error"


def _is_terminal(stream):
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def is_interactive(stream=None, environment=None):
    """Return whether rich, dynamic output is appropriate for ``stream``."""

    output = sys.stdout if stream is None else stream
    values = os.environ if environment is None else environment
    if (
        "NO_COLOR" in values
        or values.get("TERM", "").lower() == "dumb"
        or values.get("NUMPILOT_PLAIN", "") == "1"
    ):
        return False
    return _is_terminal(output)


def _console(stream):
    return Console(
        file=stream,
        highlight=False,
        markup=False,
        soft_wrap=False,
    )


_FLOW_TONE_STYLES = {
    "title": ("bold cyan", "bold cyan"),
    "phase": ("bold cyan", None),
    "model": ("bold magenta", None),
    "operator": ("bold bright_magenta", None),
    "success": ("bold green", None),
    "warning": ("bold yellow", None),
    "error": ("bold red", "red"),
    "progress": ("bold yellow", "dim"),
    "artifact": ("bold bright_blue", "dim bright_blue"),
    "metric": ("bold magenta", None),
    "info": ("bold cyan", None),
    "prompt": ("bold cyan", "bold cyan"),
    "code": ("cyan", "dim cyan"),
}


def _flow_label_tone(label):
    """Classify one framework-owned leading label without reading its body."""

    name = label.strip("[] ")
    if name in {"✓", "PASS"}:
        return "success"
    if name in {"✗", "FAIL"}:
        return "error"
    if name in {"?", "!"}:
        return "warning"
    if name.startswith("阶段"):
        return "phase"
    if name.startswith("算子接入"):
        return "operator"
    if name.startswith(("CANNBot", "模型", "LLM")):
        return "model"
    if name in {
        "确认",
        "缺口",
        "扫描提示",
        "解析边界",
        "模型输入边界",
        "解读边界",
        "口径",
        "迁移门禁",
    }:
        return "warning"
    if name == "进行中":
        return "progress"
    if name in {"性能", "规模"}:
        return "metric"
    if name in {
        "会话目录",
        "会话回执",
        "缺口回执",
        "框架命令",
        "迁移后代码",
        "完整报告",
        "机器回执",
    }:
        return "artifact"
    return "info"


def _styled_flow_text(message, tone=None):
    """Build safe Rich text for one human flow message.

    Only framework-owned leading labels receive semantic color.  The body is
    never parsed as Rich markup, and raw terminal controls are removed before
    rendering so source/model-derived values cannot inject ANSI or OSC data.
    """

    rendered = Text()
    lines = str(message).split("\n")
    for index, raw_line in enumerate(lines):
        if raw_line:
            safe_line = _terminal_safe(raw_line)
            if safe_line.startswith("──") and safe_line.endswith("──"):
                rendered.append(safe_line, style="bold cyan")
            else:
                match = re.match(r"^(\[[^\]]{1,64}\])(.*)$", safe_line)
                selected = tone
                if match is not None:
                    selected = selected or _flow_label_tone(match.group(1))
                    label_style, body_style = _FLOW_TONE_STYLES.get(
                        selected, ("bold cyan", None)
                    )
                    rendered.append(match.group(1), style=label_style)
                    rendered.append(match.group(2), style=body_style)
                else:
                    label_style, body_style = _FLOW_TONE_STYLES.get(
                        selected, (None, None)
                    )
                    rendered.append(safe_line, style=body_style or label_style)
        if index < len(lines) - 1:
            rendered.append("\n")
    return rendered


def flow_print(message="", *, stream=None, tone=None, end="\n", flush=False):
    """Print one colorized human-flow line while preserving plain contracts.

    Color is emitted only to a real interactive terminal.  Redirected output,
    ``NO_COLOR``, ``TERM=dumb``, and ``NUMPILOT_PLAIN=1`` remain byte-stable
    plain text, which keeps automation, tests, and persisted receipts clean.
    """

    output = sys.stdout if stream is None else stream
    if not is_interactive(output):
        print(message, file=output, end=end, flush=flush)
        return False
    console = _console(output)
    console.print(
        _styled_flow_text(message, tone=tone),
        end=end,
        soft_wrap=True,
    )
    if flush:
        output.flush()
    return True


@contextmanager
def terminal_status(message, *, enabled=True):
    """Show a durable spinner/result pair on stderr only for a real TTY."""

    stream = sys.stderr
    outcome = _StatusOutcome()
    if not enabled or not is_interactive(stream):
        yield outcome
        return

    console = _console(stream)
    safe_message = _terminal_safe(message)
    started = time.monotonic()
    try:
        with console.status(Text(safe_message, style="bold cyan"), spinner="dots"):
            yield outcome
    except BaseException:
        elapsed = time.monotonic() - started
        console.print(
            Text.assemble(
                ("✗ ", "bold red"),
                (safe_message, "red"),
                (f"  {elapsed:.1f}s", "dim"),
            )
        )
        raise
    else:
        elapsed = time.monotonic() - started
        marker, style = _STATUS_STYLE[outcome.value]
        console.print(
            Text.assemble(
                (marker + " ", "bold " + style),
                (
                    safe_message,
                    style if outcome.value != "ok" else "white",
                ),
                (f"  {elapsed:.1f}s", "dim"),
            )
        )


def _render_home_compact(console, rows):
    for target, command, detail in rows:
        console.print(
            Text.assemble(
                (target, "bold cyan"),
                "\n  ",
                (command, "bold"),
                "\n  ",
                (detail, "dim"),
            )
        )
        console.print()


def _render_doctor_compact(console, report):
    for item in report.items:
        marker, style = _STATUS_STYLE.get(item.status, ("?", "white"))
        console.print(
            Text.assemble(
                (marker + " ", "bold " + style),
                (_terminal_safe(item.name), "bold"),
                "\n  ",
                (compact_diagnostic(item.name, item.detail), "dim"),
            )
        )


def _render_paths_compact(console, paths, groups):
    active_group = None
    for name, path in paths.items():
        group = groups.get(name, "其他")
        if group != active_group:
            if active_group is not None:
                console.print()
            console.print(Text(_terminal_safe(group), style="bold cyan"))
            active_group = group
        console.print(
            Text.assemble(
                ("  " + _terminal_safe(name), "bold"),
                "\n    ",
                (_terminal_safe(path or "—"), "dim"),
            )
        )


def render_home(version, *, stream=None):
    """Render the product landing page; return ``False`` outside a TTY."""

    output = sys.stdout if stream is None else stream
    if not _is_terminal(output):
        return False
    if not is_interactive(output):
        print("NumPilot v%s" % version, file=output)
        print("从 NumPy 程序到可审计的 Ascend NPU 执行", file=output)
        print("", file=output)
        print("  检查环境  numpilot doctor --backend npu", file=output)
        print("  分析程序  numpilot analyze <script.py>", file=output)
        print("  开始迁移  numpilot migrate <script.py>", file=output)
        print("  一键演示  numpilot run <script.py> --benchmark", file=output)
        print("  比赛自检  numpilot competition-readiness --strict", file=output)
        print("  查看路径  numpilot paths", file=output)
        print("", file=output)
        print("运行 numpilot --help 查看完整命令。", file=output)
        return True
    console = _console(output)
    title = Text.assemble(
        ("NumPilot", "bold cyan"),
        (f"  v{version}", "dim"),
        "\n",
        ("从 NumPy 程序到可审计的 Ascend NPU 执行", "white"),
    )
    console.print(Panel.fit(title, border_style="cyan", box=box.DOUBLE))

    rows = (
        (
            "检查环境",
            "numpilot doctor --backend npu",
            "验证 Python、CANN、NPU 与 AsNumpy",
        ),
        (
            "分析程序",
            "numpilot analyze <script.py>",
            "只读扫描，不执行源码或调用模型",
        ),
        (
            "开始迁移",
            "numpilot migrate <script.py>",
            "使用已配置模型生成并审计迁移产物",
        ),
        (
            "一键演示",
            "numpilot run <script.py> --benchmark",
            "分析→模型迁移→数值验证→可选性能门",
        ),
        (
            "比赛自检",
            "numpilot competition-readiness --strict",
            "核对 CodeArts、华为云、视频与发布材料",
        ),
        (
            "查看路径",
            "numpilot paths",
            "定位配置、回执、缓存和运行产物",
        ),
    )
    # A six-row command table becomes cramped on ordinary 80-column terminals;
    # use the vertical narrative until there is enough room for all command
    # names and their explanations to remain scan-friendly.
    if console.width < 96:
        _render_home_compact(console, rows)
        console.print(
            Text.assemble(
                ("提示  ", "bold green"),
                "运行 numpilot --help 查看完整命令。",
            )
        )
        return True

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("目标", style="cyan", no_wrap=True)
    table.add_column("命令", style="bold")
    table.add_column("说明")
    for row in rows:
        table.add_row(*row)
    console.print(table)
    console.print(
        Text.assemble(
            ("提示  ", "bold green"),
            "运行 numpilot --help 查看完整命令；设置 NO_COLOR=1 可使用纯文本。",
        )
    )
    return True


def render_doctor(report, *, stream=None):
    """Render an environment report as a readable TTY table."""

    output = sys.stdout if stream is None else stream
    if not is_interactive(output):
        return False
    console = _console(output)
    if console.width < 64:
        console.print(
            Panel.fit(
                Text(
                    "环境自检 · backend=" + _terminal_safe(report.backend),
                    style="bold cyan",
                ),
                border_style="cyan",
            )
        )
        _render_doctor_compact(console, report)
    else:
        table = Table(
            title="环境自检 · backend=" + _terminal_safe(report.backend),
            box=box.ROUNDED,
            header_style="bold",
            title_style="bold cyan",
            expand=True,
        )
        table.add_column("状态", width=6, no_wrap=True)
        table.add_column("组件", style="bold", no_wrap=True)
        table.add_column("详情", overflow="fold")
        for item in report.items:
            marker, style = _STATUS_STYLE.get(item.status, ("?", "white"))
            table.add_row(
                Text(marker, style=f"bold {style}"),
                Text(_terminal_safe(item.name)),
                Text(compact_diagnostic(item.name, item.detail), style="dim"),
            )
        console.print(table)

    if report.errors:
        summary = Text(
            f"未通过 · {len(report.errors)} 个错误 · {len(report.warnings)} 个警告",
            style="bold red",
        )
        border = "red"
    elif report.warnings:
        summary = Text(
            f"可继续 · {len(report.warnings)} 个警告建议处理",
            style="bold yellow",
        )
        border = "yellow"
    else:
        summary = Text("环境检查通过 · 可以进入 NumPilot 工作流", style="bold green")
        border = "green"
    console.print(Panel.fit(summary, border_style=border))
    return True


def render_readiness(report, *, stream=None):
    """Render the competition audit without changing its JSON/plain contract."""

    output = sys.stdout if stream is None else stream
    if not is_interactive(output):
        return False
    console = _console(output)
    status = str(report.overall_status)
    _, status_style = _READINESS_STYLE.get(status, ("?", "white"))
    console.print(
        Panel.fit(
            Text.assemble(
                ("NumPilot 比赛就绪度", "bold cyan"),
                "\n",
                (status.upper(), "bold " + status_style),
                (
                    "  · 截止 %s · 剩余 %s 天"
                    % (report.deadline["date"], report.deadline["days_remaining"]),
                    "dim",
                ),
            ),
            border_style=status_style,
        )
    )

    if console.width < 80:
        for item in report.checks:
            marker, style = _READINESS_STYLE.get(item.status, ("?", "white"))
            console.print(
                Text.assemble(
                    (marker + " ", "bold " + style),
                    (item.check_id + " ", "bold"),
                    item.title,
                    (" · " + item.status, style),
                )
            )
    else:
        table = Table(box=box.ROUNDED, header_style="bold", expand=True)
        table.add_column("状态", width=6, no_wrap=True)
        table.add_column("ID", style="bold", width=6, no_wrap=True)
        table.add_column("维度", width=12, no_wrap=True)
        table.add_column("检查项")
        table.add_column("结论", width=10, no_wrap=True)
        labels = {
            "completeness": "完整性",
            "innovation": "创新性",
            "authenticity": "真实性",
        }
        for item in report.checks:
            marker, style = _READINESS_STYLE.get(item.status, ("?", "white"))
            table.add_row(
                Text(marker, style="bold " + style),
                item.check_id,
                labels.get(item.dimension, item.dimension),
                item.title,
                Text(item.status, style=style),
            )
        console.print(table)

    if report.blockers:
        console.print(
            Panel.fit(
                Text(
                    "关键未完成 %d 项；运行 --json 查看完整证据与动作。"
                    % len(report.blockers),
                    style="bold red",
                ),
                border_style="red",
            )
        )
        actions = [
            compact_diagnostic("next_action", action, max_chars=180)
            for action in tuple(report.next_actions)[:3]
            if action
        ]
        if actions:
            console.print(
                Panel(
                    Text("\n".join("→ " + action for action in actions)),
                    title="建议下一步（前 3 项）",
                    border_style="yellow",
                )
            )
    else:
        console.print(
            Panel.fit(
                Text("关键材料已通过本地合同检查。", style="bold green"),
                border_style="green",
            )
        )
    return True


def render_paths(paths, *, stream=None):
    """Render all user-owned locations without abbreviating their values."""

    output = sys.stdout if stream is None else stream
    if not is_interactive(output):
        return False
    console = _console(output)
    groups = {
        "persistent_home": "运行时",
        "state": "运行时",
        "config": "运行时",
        "cache": "运行时",
        "runs": "产物",
        "reports": "产物",
        "benchmark": "产物",
        "wizard": "产物",
        "gaps": "产物",
        "demo": "产物",
        "bootstrap_receipt": "供应链",
        "llm_config": "配置",
        "asnumpy_cache_base": "供应链",
        "asnumpy_cache": "供应链",
        "cannbot_cache_base": "供应链",
        "cannbot_cache": "供应链",
    }
    if console.width < 64:
        console.print(
            Panel.fit(
                Text("NumPilot 用户目录 · 本命令不会创建文件", style="bold cyan"),
                border_style="cyan",
            )
        )
        _render_paths_compact(console, paths, groups)
        return True
    table = Table(
        title="NumPilot 用户目录 · 本命令不会创建文件",
        box=box.ROUNDED,
        header_style="bold",
        title_style="bold cyan",
        expand=True,
    )
    table.add_column("类别", style="cyan", no_wrap=True)
    table.add_column("名称", style="bold", no_wrap=True)
    table.add_column("路径", overflow="fold")
    for name, path in paths.items():
        table.add_row(
            groups.get(name, "其他"),
            Text(_terminal_safe(name)),
            Text(_terminal_safe(path or "—"), style="dim"),
        )
    console.print(table)
    return True
