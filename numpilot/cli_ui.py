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
stable plain-text renderers owned by each command.  ``rich`` is optional; when
it is unavailable every helper degrades to plain text so core commands remain
usable in a minimal AsNumpy/NPU runtime.
"""

from contextlib import contextmanager
import os
import re
import sys
import time
import unicodedata

try:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError as exc:
    # ``rich`` is an optional presentation dependency.  Core commands must
    # keep working in a minimal AsNumpy runtime where only NumPy and the NPU
    # stack are installed, so a missing rich package degrades to the plain
    # text renderers owned by each command instead of aborting at import.
    if exc.name != "rich" and not (exc.name or "").startswith("rich."):
        raise
    box = Console = Group = Panel = Table = Text = None
    _RICH_AVAILABLE = False
else:
    _RICH_AVAILABLE = True


def rich_available():
    """Return whether the optional ``rich`` presentation layer is importable."""

    return _RICH_AVAILABLE


_STATUS_STYLE = {
    "ok": ("✓", "green"),
    "warn": ("⚠", "yellow"),
    "error": ("✗", "red"),
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
    if not _RICH_AVAILABLE:
        return False
    if (
        "NO_COLOR" in values
        or values.get("TERM", "").lower() == "dumb"
        or values.get("NUMPILOT_PLAIN", "") == "1"
    ):
        return False
    return _is_terminal(output)


def _console(stream):
    if not _RICH_AVAILABLE:
        raise RuntimeError("rich is not available")
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

    if not _RICH_AVAILABLE:
        return _terminal_safe(message)

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


class OperatorGenerationTimeline:
    """Render a bounded, human-readable operator-generation timeline.

    The timeline is presentation-only. It never changes the workflow state or
    writes receipts, and it degrades to stable plain text when Rich or a TTY
    is unavailable. Each update is a durable snapshot so a user can see the
    CANNBot route, skill verification, prompt assembly, model generation,
    validation, and artifact write as separate facts.
    """

    _STEPS = (
        ("route", "缺口路由"),
        ("skill", "CANNBot Skill"),
        ("context", "官方上下文"),
        ("generation", "模型写算子"),
        ("validation", "候选校验"),
        ("artifact", "草稿落盘"),
    )
    _STATE = {
        "pending": ("○", "dim"),
        "running": ("◉", "yellow"),
        "done": ("✓", "green"),
        "blocked": ("!", "red"),
        "failed": ("✗", "red"),
        "skipped": ("–", "dim"),
    }
    _ALLOWED_TRANSITIONS = {
        "pending": {"pending", "running", "done", "blocked", "failed", "skipped"},
        "running": {"running", "done", "blocked", "failed", "skipped"},
        "done": {"done"},
        "blocked": {"blocked"},
        "failed": {"failed"},
        "skipped": {"skipped"},
    }

    def __init__(self, operator, *, route="CANNBot", stream=None, enabled=None):
        self.operator = _terminal_safe(operator)
        self.route = _terminal_safe(route)
        self.stream = sys.stdout if stream is None else stream
        self.enabled = (
            is_interactive(self.stream) if enabled is None else bool(enabled)
        )
        self._states = {key: "pending" for key, _label in self._STEPS}
        self._details = {}
        self._started = False

    @property
    def states(self):
        return dict(self._states)

    def start(self):
        self._started = True
        self._render("开始生成")

    def update(self, step, state="running", detail=""):
        keys = {key for key, _label in self._STEPS}
        if step not in keys:
            raise ValueError("unknown operator generation step")
        if state not in self._STATE:
            raise ValueError("unknown operator generation state")
        previous = self._states[step]
        if state not in self._ALLOWED_TRANSITIONS[previous]:
            raise ValueError(
                "invalid operator generation transition: %s -> %s"
                % (previous, state)
            )
        self._states[step] = state
        if detail:
            self._details[step] = _terminal_safe(detail)
        headline = self._step_summary(step, state)
        if detail:
            headline += " · " + self._details[step]
        self._render(headline)

    def finish(self, detail=""):
        # Finalization is presentation-only.  Preserve blocked/pending states
        # so a fail-closed gate can never be rendered as a false success.
        if detail:
            self._details["artifact"] = _terminal_safe(detail)
        states = set(self._states.values())
        if states & {"blocked", "failed"}:
            self._render("生成流程阻止")
            return
        if states & {"pending", "running"}:
            self._render("生成流程收口（仍有未执行阶段）")
            return
        self._render("生成流程完成")

    def fail(self, step="generation", detail=""):
        self.update(step, "blocked", detail or "流程被门禁阻止")

    def _step_summary(self, step, state):
        label = dict(self._STEPS)[step]
        marker, _style = self._STATE[state]
        return "%s %s" % (marker, label)

    def _render(self, headline):
        if not self._started:
            self._started = True
        if not self.enabled or not _RICH_AVAILABLE:
            flow_print(
                "[算子流程] %s · operator=%s · route=%s"
                % (headline, self.operator, self.route),
                stream=self.stream,
                tone="progress",
                flush=True,
            )
            return
        console = _console(self.stream)
        table = Table(
            box=box.SIMPLE_HEAVY,
            expand=False,
            show_header=False,
            padding=(0, 1),
        )
        table.add_column("state", width=2, justify="center")
        table.add_column("stage", style="bold cyan", min_width=14)
        table.add_column("detail", style="white", no_wrap=False)
        for key, label in self._STEPS:
            state = self._states[key]
            marker, style = self._STATE[state]
            detail = self._details.get(key, {
                "pending": "等待",
                "running": "正在处理",
                "done": "已完成",
                "blocked": "已阻止",
                "failed": "失败",
                "skipped": "跳过",
            }[state])
            table.add_row(
                Text(marker, style=style),
                Text(label, style="bold cyan"),
                Text(_terminal_safe(detail), style="white"),
            )
        title = "NumPilot · 算子生成 · %s" % self.operator
        subtitle = "%s  |  %s" % (self.route, _terminal_safe(headline))
        console.print(
            Panel(
                table,
                title=title,
                subtitle=subtitle,
                border_style="magenta",
                box=box.ROUNDED,
                expand=False,
            )
        )


_PROGRESS_STATES = {
    "pending": ("○", "待执行", "dim"),
    "running": ("●", "执行中", "bold cyan"),
    "completed": ("✓", "完成", "green"),
    "blocked": ("!", "已阻断", "yellow"),
    "failed": ("✗", "失败", "red"),
    "skipped": ("–", "未执行", "dim"),
}
_PROGRESS_MODES = {
    "static": "规则处理 · 无模型",
    "llm": "已配置模型 · 真实调用",
    "external": "外部代理 · 独立接入",
}


def _progress_message(value):
    """Bound framework labels and redact accidental credential-like values.

    Event producers must never pass raw provider responses or exceptions. This
    last presentation boundary additionally removes common credential forms;
    it is not a substitute for that fixed-message event contract.
    """
    value = _terminal_safe(value)
    value = re.sub(r"(?i)\b(?:api[_-]?key|auth[_-]?token|access[_-]?token|authorization|password|secret)\b\s*[:=]\s*(?:Bearer\s+\S+|\"[^\"]*\"|'[^']*'|\S+)", "[凭据已隐藏]", value)
    value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [已隐藏]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[凭据已隐藏]", value)
    return value[:180] + ("…" if len(value) > 180 else "")


class AgentProgress:
    """Observer-only live six-role view, with plain immediate stderr events.

    Verification and workflow stages have separate rows, never a seventh role.
    Model labels come from actual lifecycle events, not the selected engine or
    an optimistic roster. Pending roles stay pending when work is blocked.
    """

    def __init__(self, *, enabled=True, stream=None, dynamic=True, context=None):
        from numpilot.progress import ProgressEmitter
        from numpilot.roles import ROLE_DEFINITIONS
        self.stream = sys.stderr if stream is None else stream
        self.enabled = enabled
        self.dynamic = dynamic
        self.definitions = ROLE_DEFINITIONS
        self.rows = {item.role.value: self._new_row() for item in self.definitions}
        self.steps = {}
        self.emitter = ProgressEmitter(self.on_event)
        self.value = "ok"
        self._live = None
        self._started = None
        self._visible = False
        self.context = {}
        self.events = []
        self.title = "NumPilot · 迁移工作台"
        self.set_context(context or {})

    def set_context(self, context=None, **values):
        """Set user-visible run metadata without changing workflow state."""
        payload = dict(context or {})
        payload.update(values)
        allowed = {"source", "output", "backend", "engine", "benchmark", "mode", "apis", "gaps"}
        for key, value in payload.items():
            if key not in allowed or value is None:
                continue
            rendered = ("是" if value else "否") if isinstance(value, bool) else _progress_message(value)
            self.context[key] = rendered
        return self

    @staticmethod
    def _new_row():
        return {"state": "pending", "mode": None, "message": "尚未执行", "attempt": 1,
                "started": None, "elapsed": 0.0}

    def __enter__(self):
        self._started = time.monotonic()
        if self.enabled and self.dynamic and is_interactive(self.stream):
            from rich.live import Live
            self._live = Live(get_renderable=self._renderable, console=_console(self.stream),
                              refresh_per_second=4, redirect_stdout=False, redirect_stderr=False)
            self._live.start(refresh=True)
        return self

    def error(self):
        self.value = "error"

    def warning(self):
        if self.value != "error":
            self.value = "warn"

    def emit(self, *args, **kwargs):
        return self.emitter.emit(*args, **kwargs)

    def operation(self, *args, **kwargs):
        return self.emitter.operation(*args, **kwargs)

    def on_event(self, event):
        from numpilot.progress import RoleProgressEvent
        if isinstance(event, dict):
            event = RoleProgressEvent.from_dict(event)
        if not isinstance(event, RoleProgressEvent):
            raise ValueError("expected a role lifecycle event")
        # A workflow child sends structured events over its private inherited
        # descriptor. Raw child stderr remains in its own evidence log.
        forwarded = False
        try:
            from numpilot.progress_transport import emit_child_event
        except ModuleNotFoundError as exc:
            if exc.name != "numpilot.progress_transport":
                raise
        else:
            forwarded = emit_child_event(event)
        key = event.role.value if event.role is not None else event.component
        target = self.rows if event.role is not None else self.steps
        row = target.setdefault(key, self._new_row())
        now = time.monotonic()
        if event.state == "running" and row["state"] != "running":
            row["started"] = now
        if event.state != "running" and row["started"] is not None:
            row["elapsed"] = max(0.0, now - row["started"])
            row["started"] = None
        row.update(state=event.state, mode=event.execution_mode,
                   message=_progress_message(event.message), attempt=event.attempt)

        # The ordinary workflow emits a parent ``workflow`` gate around child
        # stages. Mirror those stage boundaries onto the user-facing agent row
        # so a failed early migration or a long verification cannot look like
        # an idle Migrator/Diagnostics agent. The original workflow row remains
        # visible as an independent system gate.
        workflow_role = None
        if event.role is None and event.component == "workflow":
            message = row["message"]
            if "迁移子进程" in message:
                workflow_role = "code_migrator"
            elif "执行原程序" in message or "性能测量" in message:
                workflow_role = "diagnostics"
            elif "分析源码" in message:
                workflow_role = "analyzer"
        if event.role is None and event.component == "verification":
            workflow_role = "diagnostics"
        if workflow_role is not None:
            agent_row = self.rows[workflow_role]
            now = time.monotonic()
            if event.state == "running" and agent_row["state"] != "running":
                agent_row["started"] = now
            if event.state != "running" and agent_row["started"] is not None:
                agent_row["elapsed"] = max(0.0, now - agent_row["started"])
                agent_row["started"] = None
            agent_row.update(state=event.state, mode=event.execution_mode,
                             message=row["message"], attempt=event.attempt)
        label = event.role.value if event.role is not None else event.component
        self.events.append("%s · %s · %s" % (
            label or "流程", _PROGRESS_STATES[event.state][1], row["message"]
        ))
        self.events = self.events[-5:]
        if event.state == "failed":
            self.error()
        elif event.state == "blocked":
            self.warning()
        if not self.enabled or forwarded:
            return
        self._visible = True
        if self._live is not None:
            self._live.refresh()
            return
        from numpilot.roles import role_definition
        label = role_definition(event.role).display_name if event.role is not None else (
            "验证步骤" if event.component == "verification" else "流水线")
        _mark, state, _style = _PROGRESS_STATES[event.state]
        mode = (_PROGRESS_MODES[event.execution_mode] if event.role is not None else
                "确定性验证" if event.component == "verification" else "独立流程")
        print("[%s] %s · %s · %s · %.1fs" % (
            label, state, mode, row["message"], self._elapsed(row)),
            file=self.stream, flush=True)

    def _elapsed(self, row):
        return (max(0.0, time.monotonic() - row["started"])
                if row["started"] is not None else row["elapsed"])

    def _renderable(self):
        width = self._live.console.width if self._live is not None else _console(self.stream).width
        compact = width < 110
        table = Table(box=box.SIMPLE, expand=True, padding=(0, 1))
        table.add_column("六角色", min_width=16, max_width=23 if compact else None)
        table.add_column("状态", no_wrap=True)
        if not compact:
            table.add_column("执行方式", min_width=12)
        table.add_column("当前活动", ratio=1)
        table.add_column("耗时", justify="right", no_wrap=True)
        compact_names = {
            "analyzer": "① 分析 Analyzer", "planner": "② 规划 Planner",
            "gap_filler": "③ 补全 Gap Filler", "code_migrator": "④ 迁移 Migrator",
            "diagnostics": "⑤ 诊断 Diagnostics", "reporter": "⑥ 报告 Reporter",
        }

        def add_row(name, row, *, independent=False):
            marker, label, style = _PROGRESS_STATES[row["state"]]
            mode = "独立步骤" if independent else _PROGRESS_MODES.get(row["mode"], "尚未执行")
            cells = [Text(name), Text(marker + " " + label, style=style)]
            if compact:
                detail = Text(row["message"])
                detail.append("\n" + mode, style="dim")
            else:
                cells.append(Text(mode, style="dim"))
                detail = Text(row["message"])
            cells.extend((detail, Text("%.1fs" % self._elapsed(row), style="dim")))
            table.add_row(*cells)

        for definition in self.definitions:
            key = definition.role.value
            add_row(compact_names[key] if compact else definition.display_name, self.rows[key])
        for key, row in list(self.steps.items()):
            add_row("验证步骤" if key == "verification" else "流水线", row, independent=True)
        total = max(0.0, time.monotonic() - self._started) if self._started is not None else 0.0
        context = Table.grid(expand=True, padding=(0, 1))
        context.add_column(style="bold cyan", no_wrap=True)
        context.add_column(ratio=1, overflow="ellipsis")
        context.add_column(style="bold cyan", no_wrap=True)
        context.add_column(ratio=1, overflow="ellipsis")
        labels = (("source", "源码"), ("output", "运行目录"),
                  ("backend", "后端"), ("engine", "迁移引擎"),
                  ("benchmark", "性能门"), ("mode", "模式"),
                  ("apis", "API 调用"), ("gaps", "缺口"))
        for offset in range(0, len(labels), 2):
            cells = []
            for key, label in labels[offset:offset + 2]:
                cells.extend((label, self.context.get(key, "—")))
            while len(cells) < 4:
                cells.extend(("", ""))
            context.add_row(*[Text(_terminal_safe(cell), no_wrap=True) for cell in cells])
        event_display = Text(style="dim")
        if self.events:
            for index, event in enumerate(self.events[-5:]):
                if index:
                    event_display.append("\n")
                event_display.append(_terminal_safe(event))
        else:
            event_display.append("等待工作流事件…")
        event_panel = Panel(event_display, title="最近事件", border_style="dim", padding=(0, 1))
        role_panel = Panel(
            table,
            title="六角色协作",
            border_style={"ok": "cyan", "warn": "yellow", "error": "red"}[self.value],
            padding=(0, 1),
        )
        return Panel(
            Group(
                Panel(context, title="任务上下文", border_style="blue", padding=(0, 1)),
                role_panel,
                event_panel,
            ),
            title=self.title,
            border_style={"ok": "cyan", "warn": "yellow", "error": "red"}[self.value],
            subtitle=Text("总耗时 %.1fs · 运行状态不替代验证回执" % total),
            box=box.ROUNDED,
        )

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.error()
        # A caught CLI error may return a nonzero code instead of raising.
        # Running rows must not remain spinning after either exit path.
        for key, row in list(self.rows.items()) + list(self.steps.items()):
            if row["state"] == "running":
                self.emit(key if key in self.rows else None,
                    "failed" if self.value == "error" else "blocked",
                    "任务未完成，请查看结果报告", execution_mode=row["mode"],
                    component=key)
        if self._live is not None:
            self._live.stop()
        return False


@contextmanager
def terminal_status(message, *, enabled=True):
    """Show a durable spinner/result pair on stderr only for a real TTY."""

    stream = sys.stderr
    outcome = _StatusOutcome()
    if not enabled or not _RICH_AVAILABLE or not is_interactive(stream):
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
