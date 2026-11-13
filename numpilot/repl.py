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
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
import time
import unicodedata

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from numpilot.cli_ui import flow_print, is_interactive


_PROMPT = "numpilot › "
SLASH_COMMANDS = (
    ("/help", "显示本帮助与交互说明"),
    ("/new", "新建迁移；输入 NumPy 程序绝对路径"),
    ("/resume", "加载最近的迁移记忆（也可选编号）"),
    ("/modelcheck", "手动真实检查 llm.json 模型；状态栏显示可用/不可用"),
    ("/readreport", "在终端展示最近一次迁移报告；可加编号或会话路径"),
    ("/doctor", "检查 CANN、NPU、Python 与 AsNumpy"),
    ("/status", "只显示当前模型状态"),
    ("/migrate /abs/x.py", "已知绝对路径时直接开始迁移"),
    ("/exit", "退出 NumPilot"),
)
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
        color_system=None if os.getenv("NO_COLOR") else "auto",
    )
    body = Text(_LOGO, style="bold cyan")
    body.append("\n")
    body.append(f"v{version_text}  |  {subtitle}", style="white")
    console.print(Panel.fit(body, border_style="cyan", box=box.DOUBLE))


def render_repl_help(*, stream=None) -> None:
    """Show slash commands, menu options, and the status/session contract."""

    output = sys.stdout if stream is None else stream
    flow_print("\nNumPilot 交互指令", stream=output, tone="title")
    for command, description in SLASH_COMMANDS:
        print("  %-20s %s" % (command, description), file=output)
    print("  1-5                  等价于 /new、/resume、/modelcheck、/doctor、/exit", file=output)
    print("  check llm            等价于 /modelcheck 的框架检查命令", file=output)
    print(
        "\n每次迁移都会新建 runs/demo-时间戳/，并落盘：source/ 源码副本、"
        "migration-plan.md/json、migration-report.md/json、workflow/ 回执与日志。",
        file=output,
    )
    print(
        "状态规范：待命 → 分析中 → 响应中 → 生成中 → 校验中 → 落盘中 → 完成；"
        "模型不可用时显示回退，流程阻断时显示阻断或失败。",
        file=output,
    )
    print(
        "\n也可直接输入 analyze、paths 等 `numpilot --help` 中的子命令。",
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
    if lowered[0] in {"/modelcheck", "modelcheck", "检查模型"}:
        return ("__modelcheck__",)
    if lowered[:2] == ["check", "llm"]:
        return ("check-llm", *command[2:])
    if lowered[0] in {"check-llm", "check_llm"}:
        return ("check-llm", *command[1:])

    migration_arguments: list[str] | None = None
    if lowered[0] in {"migrate", "migration", "/migrate", "/migration"}:
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

    if lowered[0] in {"help", "/help"}:
        if len(command) == 1:
            return ("__repl_help__",)
        help_target = normalize_repl_command(command[1:])
        if not help_target or help_target[0] == "__repl_help__":
            return ("__repl_help__",)
        return (*help_target, "--help")
    if lowered[0] in {"exit", "quit", "q", "/exit", "/quit"} or command[0] in {"退出", "结束"}:
        return ("__repl_exit__",)
    return tuple(command)


def render_model_status(status, *, stream=None) -> None:
    """Render the single-line model availability status bar."""

    output = sys.stdout if stream is None else stream
    state = _safe_line(status.get("state", "未检查"), limit=32)
    detail = _safe_line(status.get("detail", ""), limit=180)
    if os.getenv("NO_COLOR") or not is_interactive(output):
        print(
            "[模型状态] %s%s" % (state, (" · " + detail) if detail else ""),
            file=output,
        )
        return
    tone = {
        "模型可用": "success",
        "模型不可用": "error",
        "检查中": "progress",
    }.get(state, "warning")
    flow_print(
        "[模型状态] %s%s" % (state, (" · " + detail) if detail else ""),
        stream=output,
        tone=tone,
    )


def check_model_status(*, stream=None) -> dict:
    """Manually probe llm.json through the framework's real connectivity check."""

    output = sys.stdout if stream is None else stream
    flow_print(
        "[模型状态] 检查中 · 正在向 llm.json 配置的模型发送轻量探测",
        stream=output,
        tone="progress",
    )
    try:
        from numpilot.llm_check import (
            LlmConnectivityCheckError,
            run_llm_connectivity_check,
        )

        result = run_llm_connectivity_check()
    except LlmConnectivityCheckError as exc:
        status = {"state": "模型不可用", "detail": str(exc)}
    except Exception as exc:
        status = {"state": "模型不可用", "detail": type(exc).__name__}
    else:
        status = {
            "state": "模型可用",
            "detail": "%s · %s" % (result.configured_model, result.protocol),
        }
    render_model_status(status, stream=output)
    return status


def migration_sessions(*, limit: int = 8) -> list[Path]:
    """Return recent session folders that contain a final migration report."""

    def collect(root: Path) -> list[Path]:
        found = []
        if not root.is_dir():
            return found
        try:
            children = list(root.iterdir())
        except OSError:
            return found
        for child in children:
            if not child.is_dir() or child.is_symlink():
                continue
            if not (child / "migration-report.json").is_file():
                continue
            summary = _session_summary(child)
            if not summary:
                continue
            source_path = str(summary.get("source", ""))
            # Test fixtures and scratch copies must not appear as user memory.
            if source_path.startswith("/tmp/") or "/pytest-of-" in source_path:
                continue
            # Keep completed memories and failed sessions that reached the
            # new plan artifact; early confirmation scratch sessions stay out.
            if summary.get("status") != "passed" and not (child / "migration-plan.md").is_file():
                continue
            found.append(child)
        return found

    sessions = collect(Path.cwd() / "runs")
    if not sessions:
        try:
            from numpilot.paths import default_demo_dir

            sessions = collect(default_demo_dir())
        except Exception:
            sessions = []
    sessions.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return sessions[:limit]


def _session_summary(session: Path) -> dict:
    report_path = session / "migration-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    source = report.get("source") if isinstance(report.get("source"), dict) else {}
    migration = report.get("migration") if isinstance(report.get("migration"), dict) else {}
    workflow = report.get("workflow") if isinstance(report.get("workflow"), dict) else {}
    performance = workflow.get("performance") if isinstance(workflow.get("performance"), dict) else {}
    return {
        "source": source.get("path") or "—",
        "status": report.get("status") or workflow.get("status") or "unknown",
        "migrated": migration.get("migrated_path") or "—",
        "api_count": len(migration.get("api_rewrites") or []),
        "gaps": len(report.get("gap_completions") or []),
        "cpu_ms": performance.get("cpu_median_ms"),
        "npu_ms": performance.get("npu_median_ms"),
        "speedup": performance.get("speedup"),
        "report": str(
            session / "migration-report.md"
            if (session / "migration-report.md").is_file()
            else report_path
        ),
        "plan": str(session / "migration-plan.md"),
    }


def load_history(*, stream=None, error_stream=None, input_stream=None) -> None:
    """List and load the summary of one previous migration memory."""

    output = sys.stdout if stream is None else stream
    diagnostic = sys.stderr if error_stream is None else error_stream
    source_stream = sys.stdin if input_stream is None else input_stream
    sessions = migration_sessions()
    if not sessions:
        flow_print(
            "[历史迁移] 没有找到已完成的迁移记录。",
            stream=output,
            tone="warning",
        )
        return
    flow_print("\n[历史迁移] 最近记录", stream=output, tone="title")
    for index, session in enumerate(sessions, 1):
        summary = _session_summary(session)
        stamp = datetime.fromtimestamp(session.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        speed = summary.get("speedup")
        speed_text = "%.2fx" % speed if isinstance(speed, (int, float)) else "—"
        print(
            "  %d) %s  %-8s  %-36s  %s"
            % (
                index,
                stamp,
                summary.get("status", "unknown"),
                Path(str(summary.get("source", "—"))).name,
                speed_text,
            ),
            file=output,
        )
    flow_print("选择编号加载迁移记忆（其他键返回）：", stream=output, tone="prompt", end="", flush=True)
    try:
        raw = source_stream.readline().strip()
    except KeyboardInterrupt:
        return
    print("", file=output)
    if not raw.isdigit() or not 1 <= int(raw) <= len(sessions):
        return
    session = sessions[int(raw) - 1]
    summary = _session_summary(session)
    if not summary:
        flow_print("[历史迁移] 该记录无法读取。", stream=diagnostic, tone="error")
        return
    flow_print("\n── 迁移记忆 ──", stream=output, tone="title")
    flow_print("[源程序] %s" % summary["source"], stream=output, tone="artifact")
    flow_print("[迁移后] %s" % summary["migrated"], stream=output, tone="artifact")
    flow_print(
        "[状态] %s · API=%s 组 · 缺口=%s"
        % (summary["status"], summary["api_count"], summary["gaps"]),
        stream=output,
        tone="success" if summary["status"] == "passed" else "warning",
    )
    if isinstance(summary.get("speedup"), (int, float)):
        flow_print(
            "[性能] CPU=%s ms · NPU=%s ms · speedup=%.2fx"
            % (summary.get("cpu_ms"), summary.get("npu_ms"), summary["speedup"]),
            stream=output,
            tone="metric",
        )
    flow_print("[报告] %s" % summary["report"], stream=output, tone="artifact")
    if Path(summary["plan"]).is_file():
        flow_print("[计划] %s" % summary["plan"], stream=output, tone="artifact")


def read_report(*, stream=None, error_stream=None, argument=None) -> int:
    """Render the latest (or selected) migration report in the terminal."""

    output = sys.stdout if stream is None else stream
    diagnostic = sys.stderr if error_stream is None else error_stream
    sessions = migration_sessions()
    if not sessions:
        flow_print(
            "[迁移报告] 没有找到已完成的迁移记录；请先执行 /new。",
            stream=output,
            tone="warning",
        )
        return 1

    selected = None
    raw = (argument or "").strip()
    if raw:
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(sessions):
                selected = sessions[index - 1]
            else:
                flow_print(
                    "[迁移报告] 编号超出范围；请先运行 /resume 查看列表。",
                    stream=diagnostic,
                    tone="error",
                )
                return 1
        else:
            candidate = Path(raw).expanduser()
            if candidate.is_file() and candidate.name == "migration-report.md":
                selected = candidate.parent
            elif candidate.is_dir() and (candidate / "migration-report.md").is_file():
                selected = candidate
            else:
                flow_print(
                    "[迁移报告] 未找到指定的 migration-report.md。",
                    stream=diagnostic,
                    tone="error",
                )
                return 1
    if selected is None:
        selected = sessions[0]
    report_path = selected / "migration-report.md"
    if not report_path.is_file():
        flow_print(
            "[迁移报告] 该会话没有 migration-report.md：%s" % selected,
            stream=diagnostic,
            tone="error",
        )
        return 1
    try:
        report_text = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        flow_print(
            "[迁移报告] 读取失败：%s" % type(exc).__name__,
            stream=diagnostic,
            tone="error",
        )
        return 1
    if len(report_text) > 256 * 1024:
        report_text = report_text[:256 * 1024] + "\n\n> [报告过长，已截断]\n"
    if os.getenv("NO_COLOR") or not is_interactive(output):
        print(report_text, file=output)
        return 0
    console = Console(
        file=output,
        highlight=False,
        markup=False,
        color_system="auto",
    )
    console.print(Panel(
        Markdown(report_text),
        title="NumPilot · 迁移报告",
        subtitle=str(selected),
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    return 0


def start_new_migration(*, source, output, diagnostic, dispatch) -> int:
    """Prompt for an absolute NumPy path and dispatch a fresh session."""

    flow_print(
        "\n[新建迁移] 请输入待迁移 NumPy 程序的绝对路径（输入 cancel 取消）：",
        stream=output,
        tone="prompt",
        end="",
        flush=True,
    )
    try:
        raw = source.readline()
    except KeyboardInterrupt:
        return 0
    if raw == "":
        return 0
    print("", file=output)
    path_text = raw.strip()
    if not path_text or path_text.lower() in {"cancel", "q", "quit"}:
        flow_print("[新建迁移] 已取消。", stream=output, tone="warning")
        return 0
    if not os.path.isabs(path_text):
        flow_print(
            "[拒绝] 请输入绝对路径，例如 /path/to/program.py。",
            stream=diagnostic,
            tone="error",
        )
        return 0
    try:
        from numpilot.interactive_migration import (
            looks_like_numpy_program,
            read_source_snapshot,
        )

        snapshot = read_source_snapshot(path_text)
    except Exception as exc:
        flow_print(
            "[拒绝] 无法读取目标文件：%s" % type(exc).__name__,
            stream=diagnostic,
            tone="error",
        )
        return 0
    if not looks_like_numpy_program(snapshot.text):
        flow_print(
            "[拒绝] 目标程序不是 NumPy 程序：未检测到 `import numpy` 或 NumPy 调用。",
            stream=diagnostic,
            tone="error",
        )
        return 0
    runs = Path.cwd() / "runs"
    try:
        runs.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        flow_print(
            "[新建迁移] 无法创建 runs 目录：%s" % type(exc).__name__,
            stream=diagnostic,
            tone="error",
        )
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs / ("demo-" + stamp)
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs / ("demo-%s-%d" % (stamp, suffix))
    flow_print("[新建迁移] 会话目录：%s" % run_dir, stream=output, tone="artifact")
    arguments = normalize_repl_command([
        "migrate", snapshot.path,
        "--out", str(run_dir),
        "--backend", "npu",
        "--benchmark",
    ])
    return dispatch(arguments)


def render_startup_menu(version: str, model_status: dict, *, stream=None) -> None:
    """Render the product landing menu and its model status bar."""

    output = sys.stdout if stream is None else stream
    state = _safe_line(model_status.get("state", "未检查"), limit=32)
    detail = _safe_line(model_status.get("detail", ""), limit=180)
    status_style = {
        "模型可用": "bold green",
        "模型不可用": "bold red",
        "未检查": "bold yellow",
    }.get(state, "bold yellow")
    console = Console(
        file=output,
        highlight=False,
        markup=False,
        color_system=(
            None
            if os.getenv("NO_COLOR") or not is_interactive(output)
            else "auto"
        ),
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("1  新建迁移", "输入 NumPy 程序绝对路径，创建新的会话目录并落盘全部产物")
    table.add_row("2  加载迁移记忆", "查看最近一次或指定历史迁移的报告与计划")
    table.add_row("3  检查模型", "等价于 /modelcheck；真实请求 llm.json 配置的模型")
    table.add_row("4  环境自检", "检查 CANN、NPU、Python 与 AsNumpy")
    table.add_row("5  退出", "离开 NumPilot")
    header = Text()
    header.append("[模型状态] ", style="bold cyan")
    header.append(state, style=status_style)
    if detail:
        header.append(" · " + detail, style="dim")
    console.print(Panel(
        Group(header, Text(""), table),
        title="NumPilot 启动页",
        subtitle="v%s · NumPy → AsNumpy · Ascend NPU" % _safe_line(version, limit=64),
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print(Text(
        "输入 1-5 选择，或直接输入 /modelcheck、migrate、doctor 等命令。",
        style="dim",
    ))


def run_repl(
    version: str,
    dispatch: Callable[[Sequence[str]], int | None],
    *,
    input_stream=None,
    output_stream=None,
    error_stream=None,
) -> int:
    """Run the startup menu and keep the product shell alive across tasks."""

    source = sys.stdin if input_stream is None else input_stream
    output = sys.stdout if output_stream is None else output_stream
    diagnostic = sys.stderr if error_stream is None else error_stream
    if not can_start_repl(source, output):
        raise RuntimeError("NumPilot interactive shell requires a TTY input and output")

    render_repl_banner(version, stream=output)
    model_status = {
        "state": "未检查",
        "detail": "运行 /modelcheck 检查 llm.json 中配置的模型",
    }
    render_model_status(model_status, stream=output)
    flow_print(
        "已进入 NumPilot 交互会话。输入 /modelcheck 真实检查模型；"
        "输入 1 开始新的迁移。",
        stream=output,
        tone="info",
    )

    def dispatch_once(arguments, *, announce=True):
        try:
            result = dispatch(arguments)
            code = 0 if result is None else int(result)
        except KeyboardInterrupt:
            flow_print(
                "\n已中止当前任务；会话仍在运行。",
                stream=diagnostic,
                tone="warning",
            )
            return 0
        except SystemExit as exc:
            try:
                code = int(exc.code or 0)
            except (TypeError, ValueError):
                code = 2
        except Exception as exc:
            flow_print(
                "命令失败：%s；为避免泄露本机配置，交互层不展开异常正文。"
                % type(exc).__name__,
                stream=diagnostic,
                tone="error",
            )
            return 0
        if announce and code:
            flow_print(
                "当前任务未完成（exit_code=%d）；可修正后继续。" % code,
                stream=diagnostic,
                tone="warning",
            )
        return code

    while True:
        render_startup_menu(version, model_status, stream=output)
        flow_print(
            "numpilot › 选择 [1-5] 或输入命令：",
            stream=output,
            tone="prompt",
            end="",
            flush=True,
        )
        try:
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
        lowered = raw.lower()

        if lowered in {"help", "/help", "?", "菜单", "menu", "/menu"}:
            render_repl_help(stream=output)
            continue
        if lowered == "/readreport" or lowered.startswith("/readreport "):
            argument = raw[len("/readreport"):].strip()
            read_report(
                stream=output,
                error_stream=diagnostic,
                argument=argument,
            )
            continue
        if lowered in {"/status", "status"}:
            render_model_status(model_status, stream=output)
            continue
        if lowered in {"1", "new", "/new", "新建迁移", "新建"}:
            start_new_migration(
                source=source,
                output=output,
                diagnostic=diagnostic,
                dispatch=lambda args: dispatch_once(args),
            )
            continue
        if lowered in {"2", "history", "/resume", "/history", "resume", "加载历史迁移", "历史", "迁移记忆"}:
            load_history(stream=output, error_stream=diagnostic, input_stream=source)
            continue
        if lowered in {"3", "/modelcheck", "/check", "modelcheck", "检查模型", "check model"}:
            model_status = check_model_status(stream=output)
            continue
        if lowered in {"4", "/doctor", "doctor", "环境自检"}:
            dispatch_once(["doctor", "--backend", "npu"])
            continue
        if lowered in {"5", "/exit", "/quit", "exit", "quit", "q", "退出", "结束"}:
            flow_print("已退出 NumPilot。", stream=output, tone="success")
            return 0

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
        if arguments == ("__modelcheck__",):
            model_status = check_model_status(stream=output)
            continue
        if arguments and arguments[0] in {"check-llm", "check_llm"}:
            code = dispatch_once(arguments, announce=False)
            model_status = {
                "state": "模型可用" if code == 0 else "模型不可用",
                "detail": "check-llm exit_code=%d" % code,
            }
            render_model_status(model_status, stream=output)
            if code:
                flow_print(
                    "当前任务未完成（exit_code=%d）；可修正后继续。" % code,
                    stream=diagnostic,
                    tone="warning",
                )
            continue
        if arguments == ("__repl_exit__",):
            flow_print("已退出 NumPilot。", stream=output, tone="success")
            return 0
        if arguments == ("__repl_help__",):
            render_repl_help(stream=output)
            continue
        dispatch_once(arguments)


__all__ = [
    "can_start_repl",
    "check_model_status",
    "load_history",
    "migration_sessions",
    "normalize_repl_command",
    "read_report",
    "render_model_status",
    "render_repl_banner",
    "render_repl_help",
    "render_startup_menu",
    "run_repl",
    "start_new_migration",
]
