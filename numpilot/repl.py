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

"""Persistent human terminal for NumPilot.

The normal argparse commands remain the machine/automation interface.  This
module adds a deliberately small TTY-only command loop for the product entry
point (``numpilot`` with no arguments).  It never implements migration or LLM
logic itself: commands are normalized and dispatched back to the framework's
ordinary CLI handlers so terminal users and direct CLI callers share the same
gates and receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import shlex
import sys
import unicodedata

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from numpilot.cli_ui import flow_print, is_interactive


_PROMPT = "numpilot › "
_LOGO = r"""
    _   __                ____  _ __      __
   / | / /_  ______ ___  / __ \(_) /___  / /_
  /  |/ / / / / __ `__ \/ /_/ / / / __ \/ __/
 / /|  / /_/ / / / / / / ____/ / / /_/ / /_
/_/ |_/\__,_/_/ /_/ /_/_/   /_/\____/\__/
""".strip("\n")


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def can_start_repl(input_stream=None, output_stream=None) -> bool:
    """Return whether a safe two-way terminal session can be started."""

    source = sys.stdin if input_stream is None else input_stream
    sink = sys.stdout if output_stream is None else output_stream
    return _is_tty(source) and _is_tty(sink)


def _safe_line(value, *, limit: int = 240) -> str:
    """Bound terminal diagnostics and remove control/bidirectional bytes."""

    rendered = str(value).replace("\r", " ").replace("\n", " ")
    safe: list[str] = []
    for character in rendered:
        category = unicodedata.category(character)
        if (
            character.isprintable()
            and not category.startswith("C")
            and category not in {"Cf"}
        ):
            safe.append(character)
        else:
            safe.append("�")
        if len(safe) >= limit:
            break
    return "".join(safe) or "unknown"


def render_repl_banner(version: str, *, stream=None) -> None:
    """Render the interactive product mark without changing command state."""

    output = sys.stdout if stream is None else stream
    subtitle = "NumPy → AsNumpy · Ascend NPU 可审计迁移"
    version_text = _safe_line(version, limit=64)
    if not is_interactive(output):
        print(_LOGO, file=output)
        print(f"NumPilot v{version_text}  |  {subtitle}", file=output)
        return

    console = Console(
        file=output,
        highlight=False,
        markup=False,
        soft_wrap=False,
    )
    body = Text(_LOGO, style="bold cyan")
    body.append("\n")
    body.append(f"v{version_text}  |  {subtitle}", style="white")
    console.print(Panel.fit(body, border_style="cyan", box=box.DOUBLE))


def render_repl_help(*, stream=None) -> None:
    """Show the intentionally small vocabulary of the persistent session."""

    output = sys.stdout if stream is None else stream
    flow_print("\n可用指令", stream=output, tone="title")
    print("  check llm          向 llm.json 配置的模型发送最小测试消息", file=output)
    print("  migrate            开始全量迁移；下一步会询问源码绝对路径", file=output)
    print("  migrate /abs/x.py  已知路径时直接开始同一流程", file=output)
    print("  help               再次显示帮助", file=output)
    print("  exit               退出 NumPilot", file=output)
    print(
        "\n也可直接输入 doctor、analyze、paths 等 `numpilot --help` 中的子命令。",
        file=output,
    )


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def normalize_repl_command(tokens: Sequence[str]) -> tuple[str, ...]:
    """Translate conversational shell commands to the canonical CLI surface.

    ``migrate`` inside the product shell intentionally means the complete
    interactive workflow, not the lower-level historical ``migrate`` command.
    CANNBot context display is enabled here, while license acceptance remains
    an explicit framework-owned prompt/flag and is never inferred by the REPL.
    """

    command = [str(item) for item in tokens]
    if command and command[0] == "numpilot":
        command.pop(0)
    if not command:
        return ()

    lowered = [item.lower() for item in command]
    if lowered[:2] == ["check", "llm"]:
        return ("check-llm", *command[2:])
    if lowered[0] in {"check-llm", "check_llm"}:
        return ("check-llm", *command[1:])

    migration_arguments: list[str] | None = None
    if lowered[0] in {"migrate", "migration"}:
        migration_arguments = command[1:]
    elif lowered[:2] in (["start", "migration"], ["start", "migrate"]):
        migration_arguments = command[2:]
    elif command[0] in {"开始迁移", "迁移"}:
        migration_arguments = command[1:]

    if migration_arguments is not None:
        routed = ["interactive-migrate", *migration_arguments]
        if not _has_option(routed, "--cannbot-gap-draft"):
            routed.append("--cannbot-gap-draft")
        if not _has_option(routed, "--show-cannbot-skill"):
            routed.extend(("--show-cannbot-skill", "full"))
        return tuple(routed)

    if lowered[0] == "help":
        if len(command) == 1:
            return ("__repl_help__",)
        help_target = normalize_repl_command(command[1:])
        if not help_target or help_target[0] == "__repl_help__":
            return ("__repl_help__",)
        return (*help_target, "--help")
    if lowered[0] in {"exit", "quit", "q"} or command[0] in {"退出", "结束"}:
        return ("__repl_exit__",)
    return tuple(command)


def run_repl(
    version: str,
    dispatch: Callable[[Sequence[str]], int | None],
    *,
    input_stream=None,
    output_stream=None,
    error_stream=None,
) -> int:
    """Run the TTY command loop and keep it alive across individual tasks."""

    source = sys.stdin if input_stream is None else input_stream
    output = sys.stdout if output_stream is None else output_stream
    diagnostic = sys.stderr if error_stream is None else error_stream
    if not can_start_repl(source, output):
        raise RuntimeError("NumPilot interactive shell requires a TTY input and output")

    render_repl_banner(version, stream=output)
    flow_print(
        "已进入交互会话。输入 `migrate` 开始迁移，或输入 `check llm` 检查模型连通性。",
        stream=output,
        tone="info",
    )
    print("输入 `help` 查看完整的会话指令。\n", file=output)

    while True:
        try:
            flow_print(
                _PROMPT,
                stream=output,
                tone="prompt",
                end="",
                flush=True,
            )
            line = source.readline()
        except KeyboardInterrupt:
            flow_print(
                "\n已取消当前输入；会话仍在运行。",
                stream=output,
                tone="warning",
            )
            continue
        if line == "":
            flow_print("\n已退出 NumPilot。", stream=output, tone="success")
            return 0
        raw = line.strip()
        if not raw:
            continue
        try:
            tokens = shlex.split(raw)
        except ValueError as exc:
            flow_print(
                "指令解析失败：%s" % _safe_line(exc),
                stream=diagnostic,
                tone="error",
            )
            continue

        arguments = normalize_repl_command(tokens)
        if not arguments:
            continue
        if arguments == ("__repl_exit__",):
            flow_print("已退出 NumPilot。", stream=output, tone="success")
            return 0
        if arguments == ("__repl_help__",):
            render_repl_help(stream=output)
            continue

        try:
            result = dispatch(arguments)
            return_code = 0 if result is None else int(result)
        except KeyboardInterrupt:
            flow_print(
                "\n已中止当前任务；会话仍在运行。",
                stream=diagnostic,
                tone="warning",
            )
            continue
        except SystemExit as exc:
            # argparse uses SystemExit for --help as well as invalid input.
            try:
                return_code = int(exc.code or 0)
            except (TypeError, ValueError):
                return_code = 2
        except Exception as exc:  # keep a long-running product shell usable
            flow_print(
                "命令失败：%s；为避免泄露本机配置，交互层不展开异常正文。"
                % type(exc).__name__,
                stream=diagnostic,
                tone="error",
            )
            continue
        if return_code:
            flow_print(
                "当前任务未完成（exit_code=%d）；可修正后继续输入指令。"
                % return_code,
                stream=diagnostic,
                tone="warning",
            )


__all__ = [
    "can_start_repl",
    "normalize_repl_command",
    "render_repl_banner",
    "render_repl_help",
    "run_repl",
]
