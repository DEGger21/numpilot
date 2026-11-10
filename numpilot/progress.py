"""Observer-only lifecycle events for the six roles and deterministic gates.

Messages are framework-owned activity labels, never provider output, exception
text, source code, credentials or evidence. Events describe work in progress;
the existing role records and validation receipts remain the result contracts.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

from numpilot.roles import Role


PROGRESS_SCHEMA = "numpilot.role-progress"
PROGRESS_STATES = frozenset(("running", "completed", "blocked", "failed", "skipped"))
EXECUTION_MODES = frozenset(("static", "llm", "external"))


@dataclass(frozen=True)
class RoleProgressEvent:
    """One immutable event; ``role=None`` represents a gate, never a new agent."""

    role: Role | None
    component: str
    state: str
    message: str
    attempt: int = 1
    execution_mode: str = "static"
    sequence: int = 1
    elapsed_seconds: float = 0.0

    def __post_init__(self):
        if self.role is not None:
            object.__setattr__(self, "role", Role(self.role))
        allowed_components = {self.role.value} if self.role is not None else {"verification", "workflow"}
        if self.component not in allowed_components:
            raise ValueError("progress component must identify its role or verification gate")
        if self.state not in PROGRESS_STATES:
            raise ValueError("unknown progress state")
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError("unknown progress execution mode")
        if self.component == "verification" and self.execution_mode != "static":
            raise ValueError("verification is a deterministic gate")
        if (not isinstance(self.message, str) or not 1 <= len(self.message) <= 512
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in self.message)):
            raise ValueError("progress requires a framework-owned activity message")
        if (type(self.attempt) is not int or not 1 <= self.attempt <= 10_000
                or type(self.sequence) is not int or not 1 <= self.sequence <= 10_000_000):
            raise ValueError("progress attempt and sequence must be positive integers")
        if (type(self.elapsed_seconds) not in (int, float)
                or not math.isfinite(self.elapsed_seconds) or not 0 <= self.elapsed_seconds <= 8_640_000):
            raise ValueError("progress elapsed time must be finite and non-negative")

    @property
    def schema(self):
        return PROGRESS_SCHEMA

    @property
    def schema_version(self):
        return 1

    @property
    def type(self):
        return "lifecycle"

    @property
    def kind(self):
        return "role" if self.role is not None else ("validation" if self.component == "verification" else "system")

    def to_dict(self):
        return {
            "schema": self.schema, "schema_version": self.schema_version,
            "type": self.type, "kind": self.kind,
            "role": self.role.value if self.role is not None else None,
            "component": self.component, "state": self.state,
            "message": self.message, "attempt": self.attempt,
            "execution_mode": self.execution_mode, "sequence": self.sequence,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data):
        """Strict transport parsing; an event conveys no qualification authority."""
        fields = {"schema", "schema_version", "type", "kind", "role", "component", "state",
                  "message", "attempt", "execution_mode", "sequence", "elapsed_seconds"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("invalid progress event fields")
        if (data["schema"] != PROGRESS_SCHEMA or type(data["schema_version"]) is not int
                or data["schema_version"] != 1 or data["type"] != "lifecycle"
                or (data["role"] is not None and type(data["role"]) is not str)
                or any(type(data[key]) is not str for key in
                       ("component", "state", "message", "execution_mode", "kind"))):
            raise ValueError("invalid progress event identity")
        event = cls(**{key: data[key] for key in
                       ("role", "component", "state", "message", "attempt",
                        "execution_mode", "sequence", "elapsed_seconds")})
        if event.kind != data["kind"]:
            raise ValueError("progress kind does not match its component")
        return event


def notify_observer(observer, value):
    """A broken progress display must not change migration or gate outcomes."""
    if observer is not None:
        try:
            observer(value)
        except Exception:
            # Observer failures have no place in numerical/evidence receipts.
            # KeyboardInterrupt/SystemExit remain process control signals.
            pass


class _Operation:
    def __init__(self):
        self.state = "completed"
        self.message = "处理完成"

    def finish(self, state="completed", message="处理完成"):
        if state not in PROGRESS_STATES - {"running"}:
            raise ValueError("operation requires a terminal progress state")
        self.state, self.message = state, message


class ProgressEmitter:
    """Allocate per-run sequence numbers and monotonic elapsed time."""

    def __init__(self, on_event=None):
        self.on_event = on_event
        self._started = time.monotonic()
        self._sequence = 0

    def emit(self, role, state, message, *, execution_mode="static", component=None, attempt=1):
        role = None if role is None else Role(role)
        self._sequence += 1
        event = RoleProgressEvent(
            role=role, component=component or (role.value if role is not None else "verification"),
            state=state, message=message, attempt=attempt, execution_mode=execution_mode,
            sequence=self._sequence, elapsed_seconds=max(0.0, time.monotonic() - self._started),
        )
        notify_observer(self.on_event, event)
        return event

    @contextmanager
    def operation(self, role, message, *, execution_mode="static", component=None, attempt=1):
        """Emit before execution, and exactly one terminal event even on errors."""
        kwargs = dict(execution_mode=execution_mode, component=component, attempt=attempt)
        self.emit(role, "running", message, **kwargs)
        operation = _Operation()
        try:
            yield operation
        except BaseException:
            self.emit(role, "failed", "执行未完成，请查看错误报告", **kwargs)
            raise
        else:
            self.emit(role, operation.state, operation.message, **kwargs)
