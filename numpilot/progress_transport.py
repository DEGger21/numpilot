"""Bounded observer-only progress transport for direct CLI child processes.

The dedicated pipe carries lifecycle events, never stdout, model text or gate
evidence. Invalid/dropped events cannot change migration results. The descriptor
is passed only to the direct CLI child, not its model or user-code workers.
"""
from __future__ import annotations

import json
import os
import select
import stat
import threading

from numpilot.progress import RoleProgressEvent, notify_observer

PROGRESS_FD_ENV = "NUMPILOT_PROGRESS_FD"
MAX_EVENT_BYTES = 4096


def emit_child_event(event):
    """Return True after one atomic pipe write, False for standalone CLI use."""
    raw = os.environ.get(PROGRESS_FD_ENV, "")
    if not raw.isascii() or not raw.isdecimal() or len(raw) > 8:
        return False
    descriptor = int(raw)
    if descriptor < 3:
        return False
    try:
        if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
            return False
        # Keep producer writes bounded and nonblocking even if a parent UI
        # stops draining. Atomic writes below PIPE_BUF never splice events.
        value = RoleProgressEvent.from_dict(event.to_dict())
        payload = (json.dumps(value.to_dict(), ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_EVENT_BYTES:
            return False
        os.set_blocking(descriptor, False)
        return os.write(descriptor, payload) == len(payload)
    except (OSError, TypeError, ValueError, AttributeError):
        return False


class ChildProgressPipe:
    """Drain a POSIX child pipe without delaying process cleanup or receipts."""

    def __init__(self, observer):
        self.observer = observer
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)
        self._stop = threading.Event()
        self._thread = None

    def popen_options(self, environment):
        environment[PROGRESS_FD_ENV] = str(self.write_fd)
        return {"pass_fds": (self.write_fd,)}

    def start(self):
        # Only the child retains the write end, so normal exit gives EOF.
        os.close(self.write_fd)
        self.write_fd = -1
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self):
        pending = bytearray()
        dropping = False
        while True:
            try:
                ready, _, _ = select.select([self.read_fd], [], [], 0.05)
                if not ready:
                    if self._stop.is_set():
                        return
                    continue
                chunk = os.read(self.read_fd, MAX_EVENT_BYTES)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            for piece in chunk.splitlines(keepends=True):
                ended = piece.endswith(b"\n")
                if dropping:
                    if ended:
                        dropping = False
                    continue
                pending.extend(piece)
                if len(pending) > MAX_EVENT_BYTES:
                    pending.clear()
                    dropping = not ended
                    continue
                if not ended:
                    continue
                try:
                    def pairs(items):
                        result = {}
                        for key, value in items:
                            if key in result:
                                raise ValueError("duplicate event field")
                            result[key] = value
                        return result
                    data = json.loads(bytes(pending), object_pairs_hook=pairs,
                                      parse_constant=lambda _v: (_ for _ in ()).throw(ValueError()))
                    event = RoleProgressEvent.from_dict(data)
                except (ValueError, TypeError, UnicodeError, KeyError):
                    pass
                else:
                    notify_observer(self.observer, event)
                pending.clear()

    def close(self):
        self._stop.set()
        if self.write_fd >= 0:
            os.close(self.write_fd)
            self.write_fd = -1
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        if self.read_fd >= 0:
            os.close(self.read_fd)
            self.read_fd = -1
