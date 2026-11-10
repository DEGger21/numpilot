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

"""One-command orchestration for the product-facing ordinary workflow.

The lower-level commands in :mod:`numpilot` intentionally remain independently
auditable.  This module only provides a small, process-isolated coordinator for
the human-facing ``numpilot run`` command:

``analyze -> migrate -> verify-script -> (optional benchmark-script)``.

Every stage writes its own receipt.  The coordinator never edits a receipt,
never retries a failed model request, and stops at the first non-zero stage.
The resulting ``workflow-report.json`` is a navigation/audit index, not a
replacement for the stage-specific evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


WORKFLOW_SCHEMA_VERSION = 1
MAX_STAGE_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_STAGE_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
# Keep the coordinator's admission limit aligned with the downstream
# verify/benchmark contracts, which each accept at most 128 named outputs.
MAX_OUTPUT_NAMES = 128
# Gap/target metadata contracts use the same 128-character bound.
MAX_TARGET_TEXT_LENGTH = 128
MAX_OPTION_TEXT_LENGTH = 4096


def _contains_control_text(value: str) -> bool:
    """Return whether text contains terminal-control or Unicode format chars."""

    return any(
        not character.isprintable()
        or unicodedata.category(character).startswith("C")
        for character in value
    )


class WorkflowError(RuntimeError):
    """A workflow request could not be admitted or published."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkflowStage:
    """A single child-process invocation and its bounded diagnostics."""

    name: str
    argv: tuple[str, ...]
    returncode: int | None
    status: str
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str
    stdout_sha256: str
    stderr_sha256: str
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            # argv contains no credentials; preserving it makes reproduction
            # possible while keeping the report independent of shell quoting.
            "argv": list(self.argv),
            "command": shlex.join(self.argv),
            "returncode": self.returncode,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "timed_out": self.timed_out,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_contains_symlink_before_normalization(rendered: str) -> bool:
    """Check lexical path components without collapsing ``..`` first.

    ``os.path.abspath``/``normpath`` erase a preceding ``link/..`` pair even
    though the kernel resolves the link before applying ``..``.  Walk the
    caller's spelling one component at a time and inspect each existing
    component with ``lstat`` semantics.  Missing components are allowed so a
    new output boundary can still be created; any existing symlink is rejected
    before the normalized path is used for publication.
    """

    try:
        expanded = os.path.expanduser(rendered)
        if os.path.isabs(expanded):
            current = Path(os.path.abspath(os.sep))
        else:
            current = Path(os.getcwd())
        for component in expanded.split(os.sep):
            if not component or component == ".":
                continue
            if component == "..":
                current = current.parent
                continue
            candidate = current / component
            if os.path.lexists(candidate) and candidate.is_symlink():
                return True
            current = candidate
    except (OSError, RuntimeError, TypeError, ValueError):
        # An uninspectable component is not safe to use as an evidence
        # boundary.  Callers treat this as a symlink/unsafe-path rejection.
        return True
    return False


def _safe_path(value: os.PathLike[str] | str, label: str) -> Path:
    try:
        rendered = os.fspath(value)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkflowError("invalid_path", "%s path is invalid" % label) from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or _contains_control_text(rendered)
        or len(rendered) > 4096
    ):
        raise WorkflowError("invalid_path", "%s path is invalid" % label)
    try:
        if _path_contains_symlink_before_normalization(rendered):
            raise WorkflowError(
                "unsafe_path", "%s path contains a symlink" % label
            )
        path = Path(os.path.abspath(os.path.expanduser(rendered)))
    except WorkflowError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkflowError("invalid_path", "%s path is invalid" % label) from exc
    # The pre-normalization walk above is the security check.  Keep this
    # second check for a race where a component is replaced between the walk
    # and path construction.
    current = path
    while True:
        try:
            if os.path.lexists(current) and current.is_symlink():
                raise WorkflowError(
                    "unsafe_path", "%s path contains a symlink" % label
                )
        except OSError as exc:
            raise WorkflowError("unsafe_path", "%s path is not inspectable" % label) from exc
        if current == current.parent:
            break
        current = current.parent
    return path


def _ensure_new_directory(path: Path) -> None:
    if os.path.lexists(path):
        raise WorkflowError("output_exists", "workflow output already exists: %s" % path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.mkdir(path, 0o700)
    except FileExistsError as exc:
        raise WorkflowError("output_exists", "workflow output already exists: %s" % path) from exc
    except OSError as exc:
        raise WorkflowError("output_unavailable", "cannot create workflow output") from exc


def _require_private_directory(path: Path) -> None:
    """Ensure a directory used for workflow logs is not group/world readable."""

    try:
        if path.is_symlink() or not path.is_dir():
            raise WorkflowError(
                "unsafe_path", "workflow receipt parent must be a real directory"
            )
        if os.name == "posix" and (path.stat().st_mode & 0o077):
            # Do not chmod a caller-owned directory behind their back (the
            # parent may be shared, e.g. /tmp); fail closed instead.
            raise WorkflowError(
                "insecure_output",
                "workflow receipt parent must have private permissions",
            )
    except OSError as exc:
        raise WorkflowError(
            "output_unavailable", "cannot inspect workflow receipt parent"
        ) from exc


def _write_private(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Write a bounded private file with no-clobber publication by default."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_directory(path.parent)
    fd = -1
    temporary = ""
    try:
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent))
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        temporary = ""
    except FileExistsError as exc:
        raise WorkflowError("receipt_exists", "workflow receipt already exists: %s" % path) from exc
    except OSError as exc:
        raise WorkflowError("publish_failed", "cannot publish workflow receipt") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _bounded_child_output(value: bytes) -> bytes:
    if len(value) <= MAX_STAGE_OUTPUT_BYTES:
        return value
    # Keep a deterministic truncation marker.  The complete diagnostic remains
    # outside the report only when the child itself was configured to emit it;
    # a bounded coordinator must never consume unbounded model/server output.
    marker = b"\n[NumPilot workflow: child output truncated]\n"
    return value[: MAX_STAGE_OUTPUT_BYTES - len(marker)] + marker


def _capture_pipe(stream: Any, holder: dict[str, bytes], key: str) -> None:
    """Drain one child pipe while retaining at most limit+1 bytes.

    Draining in a dedicated thread prevents a verbose child from blocking on a
    full pipe.  Bytes beyond the retained prefix are discarded, so the
    coordinator's memory use remains bounded even when a user script prints an
    unbounded diagnostic stream.
    """

    retained = bytearray()
    cap = MAX_STAGE_OUTPUT_BYTES + 1
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            if len(retained) < cap:
                retained.extend(chunk[: cap - len(retained)])
    except (OSError, ValueError, TypeError):
        # The parent may close a pipe while terminating a timed-out process.
        pass
    finally:
        holder[key] = bytes(retained)


def _kill_child_process(process: subprocess.Popen) -> None:
    """Terminate the process group created for one workflow stage."""

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _run_stage(
    name: str,
    argv: Sequence[str],
    root: Path,
    *,
    cwd: Path,
    timeout_seconds: float,
    on_event=None,
) -> WorkflowStage:
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    progress_pipe = None
    try:
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": os.environ.copy(),
        }
        if os.name == "posix":
            # A stage may launch workers of its own.  Isolate them so a
            # timeout cannot leave a descendant running after the report says
            # the stage stopped.
            popen_kwargs["start_new_session"] = True
            from numpilot.progress_transport import ChildProgressPipe, PROGRESS_FD_ENV
            popen_kwargs["env"].pop(PROGRESS_FD_ENV, None)
            if on_event is not None:
                progress_pipe = ChildProgressPipe(on_event)
                popen_kwargs.update(progress_pipe.popen_options(popen_kwargs["env"]))
        process = subprocess.Popen(
            list(argv),
            **popen_kwargs,
        )
        if progress_pipe is not None:
            progress_pipe.start()
        captured: dict[str, bytes] = {}
        readers = [
            threading.Thread(
                target=_capture_pipe,
                args=(process.stdout, captured, "stdout"),
                daemon=True,
            ),
            threading.Thread(
                target=_capture_pipe,
                args=(process.stderr, captured, "stderr"),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_child_process(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # A platform-specific process implementation may not honour
                # the group kill immediately; close pipes and make one final
                # best-effort kill before joining bounded reader threads.
                _kill_child_process(process)
        except BaseException:
            # Ctrl-C/SystemExit must not leave a model or user-script child
            # running after the coordinator itself unwinds.  The stage is not
            # published as a receipt in this path; callers retain the original
            # exception after the process group has been reaped best-effort.
            _kill_child_process(process)
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                _kill_child_process(process)
            raise
        finally:
            for pipe in (process.stdout, process.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
            for reader in readers:
                reader.join(timeout=5)
        returncode = process.returncode
        stdout = _bounded_child_output(captured.get("stdout", b""))
        stderr = _bounded_child_output(captured.get("stderr", b""))
    except OSError as exc:
        timed_out = False
        stdout = b""
        stderr = ("child launch failed: %s" % type(exc).__name__).encode("ascii")
        returncode = None
    finally:
        if progress_pipe is not None:
            progress_pipe.close()
    elapsed = round(max(0.0, time.monotonic() - started), 6)
    stdout_path = root / ("%s.stdout.log" % name)
    stderr_path = root / ("%s.stderr.log" % name)
    _write_private(stdout_path, stdout)
    _write_private(stderr_path, stderr)
    if timed_out:
        status = "timeout"
    elif returncode == 0:
        status = "passed"
    elif returncode is None:
        status = "launch_failed"
    else:
        status = "failed"
    return WorkflowStage(
        name=name,
        argv=tuple(str(item) for item in argv),
        returncode=returncode,
        status=status,
        elapsed_seconds=elapsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_sha256=_sha256(stdout),
        stderr_sha256=_sha256(stderr),
        timed_out=timed_out,
    )


def _safe_output_file(path: Path, root: Path) -> bool:
    """Return whether an output artifact stays inside *root*.

    Stage processes write the receipts that the coordinator later consumes.
    Treat those files as untrusted: a generated script/model response could
    otherwise replace a receipt with a symlink and make the parent read an
    arbitrary file (and copy selected fields into ``workflow-report.json``).
    Keep the check lexical and reject every existing symlink component.
    """

    try:
        candidate = Path(os.path.abspath(os.fspath(path)))
        boundary = Path(os.path.abspath(os.fspath(root)))
        candidate.relative_to(boundary)
        current = candidate
        while True:
            if os.path.lexists(current) and current.is_symlink():
                return False
            if current == current.parent:
                break
            current = current.parent
        return candidate.is_file() and not candidate.is_symlink()
    except (OSError, TypeError, ValueError, RuntimeError):
        return False


def _load_json(path: Path, *, root: Path) -> Mapping[str, Any] | None:
    """Load a coordinator receipt only when it is a private regular file."""

    if not _safe_output_file(path, root):
        return None
    try:
        # Receipts are child-produced data.  Read a bounded prefix rather
        # than ``Path.read_text()`` so a buggy/malicious stage cannot make the
        # coordinator allocate unbounded memory before JSON parsing.
        with path.open("rb") as handle:
            payload = handle.read(MAX_RECEIPT_BYTES + 1)
        if len(payload) > MAX_RECEIPT_BYTES:
            return None
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return None
    return value if isinstance(value, Mapping) else None


def _validate_timeout(value: object) -> float:
    if isinstance(value, bool):
        raise WorkflowError("invalid_timeout", "workflow timeout is invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowError("invalid_timeout", "workflow timeout is invalid") from exc
    if not 1.0 <= result <= MAX_STAGE_TIMEOUT_SECONDS:
        raise WorkflowError("invalid_timeout", "workflow timeout must be 1..86400 seconds")
    return result


def _validate_bounded_text(
    value: object,
    *,
    label: str,
    code: str,
    maximum: int = MAX_OPTION_TEXT_LENGTH,
    pathlike: bool = False,
) -> str:
    """Validate one caller-controlled text option before child admission."""

    rendered: object = value
    if pathlike:
        try:
            rendered = os.fspath(value)  # type: ignore[arg-type]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WorkflowError(code, "%s is invalid" % label) from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > maximum
        or any(
            not character.isprintable()
            or unicodedata.category(character).startswith("C")
            for character in rendered
        )
    ):
        raise WorkflowError(code, "%s is invalid" % label)
    return rendered


def _source_hash(path: Path) -> str | None:
    """Hash a stable regular source file while retaining a hard read bound.

    A preliminary ``stat`` followed by ``Path.read_bytes`` is not sufficient:
    a concurrent writer can grow a file after the size check and make the
    coordinator allocate an unbounded buffer.  Hash through a single file
    descriptor in bounded chunks, then compare descriptor metadata before and
    after the read so a source replacement/drift is rejected.
    """

    descriptor = -1
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
            return None

        hasher = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_SOURCE_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SOURCE_BYTES:
                return None
            hasher.update(chunk)

        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            return None
        return hasher.hexdigest()
    except (OSError, TypeError, ValueError):
        return None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def run_ordinary_workflow(
    source: os.PathLike[str] | str,
    out: os.PathLike[str] | str,
    *,
    backend: str = "npu",
    fallback_policy: str = "reject",
    target_soc: str = "unknown",
    cann_version: str = "unknown",
    outputs: Iterable[str] = ("result",),
    benchmark: bool = False,
    warmup: int = 3,
    samples: int = 20,
    core: int | None = None,
    skip_env_check: bool = False,
    timeout_seconds: float = 3600.0,
    python_executable: str | None = None,
    on_event=None,
) -> dict[str, Any]:
    """Run the ordinary configured-model chain in separate child processes.

    The function is intentionally explicit: ordinary configured-model artifacts are
    always generated by ``engine=llm`` and the migration command's own
    fail-closed checks.  ``backend=shim`` is accepted for diagnostics, but its
    receipts remain truthful and cannot become a strict-NPU claim.
    """

    source_path = _safe_path(source, "source")
    if not source_path.is_file() or source_path.is_symlink():
        raise WorkflowError("invalid_source", "source must be a regular file")
    try:
        source_size = source_path.stat().st_size
    except OSError as exc:
        raise WorkflowError("invalid_source", "source cannot be inspected") from exc
    if source_size > MAX_SOURCE_BYTES:
        raise WorkflowError(
            "source_too_large",
            "source exceeds the %d-byte workflow limit" % MAX_SOURCE_BYTES,
        )
    initial_source_sha = _source_hash(source_path)
    if initial_source_sha is None:
        raise WorkflowError(
            "source_changed",
            "source changed while it was being admitted; retry with a stable file",
        )
    root = _safe_path(out, "workflow output")
    timeout = _validate_timeout(timeout_seconds)
    # These options eventually become child-process argv values.  Check their
    # type before membership tests so an unhashable caller value (for example
    # a list/dict supplied through the Python API) is reported as the same
    # stable WorkflowError as any other invalid backend/policy.
    if not isinstance(backend, str) or backend not in ("npu", "shim"):
        raise WorkflowError("invalid_backend", "backend must be npu or shim")
    if not isinstance(fallback_policy, str) or fallback_policy not in (
        "ask",
        "accept",
        "reject",
    ):
        raise WorkflowError("invalid_policy", "fallback policy is invalid")

    # ``outputs`` is a public Python API boundary, so do not let ``None``, a
    # bare string (which would otherwise be iterated character-by-character),
    # or a hostile/non-iterable object leak TypeError.  Consume at most the
    # bounded number of names to make an infinite generator harmless.
    if outputs is None or isinstance(outputs, (str, bytes, bytearray)):
        raise WorkflowError("invalid_outputs", "outputs must be non-empty Python names")
    try:
        output_iterator = iter(outputs)
    except Exception as exc:
        raise WorkflowError(
            "invalid_outputs",
            "outputs must be an iterable of non-empty Python names",
        ) from exc
    collected_names: list[object] = []
    try:
        for index, name in enumerate(output_iterator):
            if index >= MAX_OUTPUT_NAMES:
                raise WorkflowError(
                    "invalid_outputs",
                    "outputs must contain at most %d names" % MAX_OUTPUT_NAMES,
                )
            collected_names.append(name)
    except WorkflowError:
        raise
    except Exception as exc:
        raise WorkflowError(
            "invalid_outputs",
            "outputs must be an iterable of non-empty Python names",
        ) from exc
    names = tuple(collected_names)
    if not names or any(
        not isinstance(name, str)
        or not name
        or len(name) > 128
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
            for character in name
        )
        for name in names
    ):
        raise WorkflowError("invalid_outputs", "outputs must be non-empty Python names")
    if len(set(names)) != len(names):
        raise WorkflowError("invalid_outputs", "outputs must not contain duplicates")
    if any(not name.isidentifier() for name in names):
        raise WorkflowError("invalid_outputs", "outputs must be valid Python names")

    # Target metadata and the interpreter path are copied into child argv.
    # Keep them bounded and terminal-safe, and perform all checks before the
    # output boundary is created below.
    target_soc_value = _validate_bounded_text(
        target_soc,
        label="target_soc",
        code="invalid_target_soc",
        maximum=MAX_TARGET_TEXT_LENGTH,
    )
    cann_version_value = _validate_bounded_text(
        cann_version,
        label="cann_version",
        code="invalid_cann_version",
        maximum=MAX_TARGET_TEXT_LENGTH,
    )
    if not isinstance(benchmark, bool):
        raise WorkflowError("invalid_benchmark", "benchmark must be a boolean")
    if not isinstance(skip_env_check, bool):
        raise WorkflowError(
            "invalid_environment", "skip_env_check must be a boolean"
        )
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 1000:
        raise WorkflowError("invalid_benchmark", "warmup is invalid")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 10000:
        raise WorkflowError("invalid_benchmark", "samples is invalid")
    # Keep the coordinator's admission contract aligned with
    # ``benchmark-script``.  Letting a one-command run create an output
    # boundary only to fail later in the benchmark child is both surprising
    # and leaves an avoidable failed artifact behind.  Seven samples is the
    # minimum needed by the percentile/CI performance gate.
    if benchmark and samples < 7:
        raise WorkflowError(
            "invalid_benchmark",
            "benchmark samples must be at least 7",
        )
    if benchmark and backend != "npu":
        raise WorkflowError(
            "invalid_benchmark",
            "strict-NPU benchmark requires backend=npu; shim runs are numerical diagnostics only",
        )
    if core is not None and (isinstance(core, bool) or not isinstance(core, int) or core < 0):
        raise WorkflowError("invalid_benchmark", "core is invalid")

    py = _validate_bounded_text(
        sys.executable if python_executable is None else python_executable,
        label="python executable",
        code="invalid_python",
        pathlike=True,
    )

    # Admit all caller-controlled options before creating the evidence
    # boundary.  A typo in ``--output`` (or any other argument) must leave no
    # misleading, empty run directory behind and must not consume a path that
    # could have been used for a later retry.
    _ensure_new_directory(root)

    common = [py, "-m", "numpilot"]
    source_text = str(source_path)
    migration_dir = root / "migration"
    analysis_receipt = root / "qualified-analysis.json"
    verification_receipt = root / "script-verification-report.json"
    benchmark_receipt = root / "script-benchmark-report.json"
    stages: list[WorkflowStage] = []
    source_guard_checks: list[dict[str, Any]] = [
        {"stage": "admission", "sha256": initial_source_sha}
    ]
    source_changed = False
    from numpilot.progress import ProgressEmitter, notify_observer
    progress_events = []
    active_events = {}
    progress_lock = threading.RLock()

    def observe(event):
        with progress_lock:
            if len(progress_events) < 2048:
                progress_events.append(event.to_dict())
            if event.state == "running":
                active_events[event.component] = event
            else:
                active_events.pop(event.component, None)
            notify_observer(on_event, event)

    progress = ProgressEmitter(observe if on_event is not None else None)

    def relay(event):
        # A single parent clock/sequence orders children from several stages.
        with progress_lock:
            progress.emit(event.role, event.state, event.message,
                          execution_mode=event.execution_mode,
                          component=event.component, attempt=event.attempt)

    def close_active(state):
        with progress_lock:
            for event in list(active_events.values()):
                if event.component != "workflow":
                    progress.emit(event.role, state, "子进程已结束，请以阶段回执为准",
                                  execution_mode=event.execution_mode,
                                  component=event.component, attempt=event.attempt)

    def save_progress():
        if on_event is not None:
            payload = "".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n"
                              for e in progress_events).encode("utf-8")
            _write_private(root / "workflow-progress.jsonl", payload)

    def run_stage_checked(
        name: str, argv: Sequence[str]
    ) -> tuple[WorkflowStage, bool]:
        """Run a child and verify the source did not drift at its boundary."""

        labels = {"01-analyze": "分析源码并检查迁移准入",
                  "02-migrate": "迁移子进程运行中，等待其真实角色状态",
                  "03-verify": "执行原程序与迁移程序的独立数值验证",
                  "04-benchmark": "执行独立性能测量"}
        progress.emit(None, "running", labels[name], component="workflow")
        extra = {"on_event": relay} if on_event is not None else {}
        try:
            stage = _run_stage(
                name, argv, root, cwd=project_root, timeout_seconds=timeout, **extra,
            )
        except BaseException:
            close_active("failed")
            progress.emit(None, "failed", "流程中断，子进程已停止", component="workflow")
            save_progress()
            raise
        stages.append(stage)
        observed = _source_hash(source_path)
        source_guard_checks.append({"stage": name, "sha256": observed})
        state = ("completed" if stage.status == "passed" and observed == initial_source_sha
                 else "blocked" if stage.returncode in {1, 2, 3, 4} or observed != initial_source_sha
                 else "failed")
        close_active(state)
        progress.emit(None, state, "阶段结束，回执与日志已保存", component="workflow")
        return stage, observed == initial_source_sha

    # Running from the package/repository root keeps ``python -m numpilot``
    # usable directly from a source checkout (before an editable install),
    # while all user artifacts remain in the private output directory.
    project_root = Path(__file__).resolve().parents[1]

    analysis_argv = common + [
        "analyze", source_text, "--json", "--out", str(analysis_receipt)
    ]
    # The shim lane is an explicitly non-NPU diagnostic.  It still records the
    # real qualification result, but must not be rejected merely because the
    # machine has no matched Ascend identity; the following migration stage
    # applies the caller's explicit fallback policy and the final verification
    # receipt keeps strict_npu=false.  The default NPU lane keeps
    # the hard DIRECT gate.
    if backend == "npu":
        analysis_argv.insert(-2, "--require-direct")
    stage, source_ok = run_stage_checked("01-analyze", analysis_argv)
    if not source_ok:
        source_changed = True
        final_status = "blocked"
    elif stage.status != "passed":
        # 1/2 are analysis admission failures and 4 is the configured-model
        # gate; all are actionable prerequisites rather than coordinator
        # crashes.  Keep launch failures/timeouts as hard failures.
        final_status = "blocked" if stages[-1].returncode in {1, 2, 4} else "failed"
    else:
        migrate_argv = common + [
            "migrate", source_text, "--mode", "script", "--engine", "llm",
            "--backend", backend, "--fallback-policy", fallback_policy,
            "--target-soc", target_soc_value, "--cann-version", cann_version_value,
            "--out", str(migration_dir),
        ]
        if skip_env_check:
            migrate_argv.append("--skip-env-check")
        stage, source_ok = run_stage_checked("02-migrate", migrate_argv)
        if not source_ok:
            source_changed = True
            final_status = "blocked"
        elif stage.status != "passed":
            final_status = "blocked" if stages[-1].returncode in {1, 2, 3, 4} else "failed"
        else:
            verify_argv = common + [
                "verify-script", source_text,
                "--migration-report", str(migration_dir / "script-migration-report.json"),
                "--out", str(verification_receipt), "--json",
            ]
            for name in names:
                verify_argv.extend(["--output", name])
            stage, source_ok = run_stage_checked("03-verify", verify_argv)
            if not source_ok:
                source_changed = True
                final_status = "blocked"
            elif stage.status != "passed":
                final_status = "blocked" if stages[-1].returncode in {1, 2, 4} else "failed"
            elif benchmark:
                benchmark_argv = common + [
                    "benchmark-script", source_text,
                    "--migration-report", str(migration_dir / "script-migration-report.json"),
                    "--out", str(benchmark_receipt), "--warmup", str(warmup),
                    "--samples", str(samples), "--json",
                ]
                if core is not None:
                    benchmark_argv.extend(["--core", str(core)])
                for name in names:
                    benchmark_argv.extend(["--output", name])
                stage, source_ok = run_stage_checked("04-benchmark", benchmark_argv)
                if not source_ok:
                    source_changed = True
                    final_status = "blocked"
                else:
                    final_status = "passed" if stage.status == "passed" else (
                        "blocked" if stage.returncode in {1, 2, 4} else "failed"
                    )
            else:
                final_status = "passed"

    final_source_sha = _source_hash(source_path)
    source_guard_checks.append({"stage": "final", "sha256": final_source_sha})
    if final_source_sha != initial_source_sha:
        source_changed = True
        final_status = "blocked"

    # Receipts are independently produced by child commands.  A passing child
    # must bind the same source digest that was admitted by the coordinator;
    # otherwise a source replacement between stages could leave a green-looking
    # report whose final ``source_sha256`` describes a different file.
    migration_value = _load_json(
        migration_dir / "script-migration-report.json", root=root
    )
    verification_value = _load_json(verification_receipt, root=root)
    benchmark_value = _load_json(benchmark_receipt, root=root)

    def receipt_source_digest(label: str, value: Mapping[str, Any] | None) -> str | None:
        if not isinstance(value, Mapping):
            return None
        if label in {"analysis", "migration"}:
            raw = value.get("source_sha256")
        elif label == "verification":
            source_block = value.get("source")
            raw = source_block.get("sha256") if isinstance(source_block, Mapping) else None
        elif label == "benchmark":
            binding = value.get("binding")
            source_block = binding.get("source") if isinstance(binding, Mapping) else None
            raw = source_block.get("sha256") if isinstance(source_block, Mapping) else None
        else:
            raw = None
        return raw if isinstance(raw, str) else None

    receipt_values: dict[str, Mapping[str, Any] | None] = {
        "analysis": _load_json(analysis_receipt, root=root),
        "migration": migration_value,
        "verification": verification_value,
        "benchmark": benchmark_value,
    }
    receipt_source_hashes = {
        label: receipt_source_digest(label, value)
        for label, value in receipt_values.items()
        if value is not None
    }
    passed_stage_to_receipt = {
        "01-analyze": "analysis",
        "02-migrate": "migration",
        "03-verify": "verification",
        "04-benchmark": "benchmark",
    }
    receipt_mismatch = any(
        stage.status == "passed"
        and receipt_source_hashes.get(passed_stage_to_receipt.get(stage.name, ""))
        != initial_source_sha
        for stage in stages
    )
    if receipt_mismatch:
        final_status = "blocked"

    report: dict[str, Any] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "record_type": "numpilot_ordinary_workflow",
        "created_at": _utc_now(),
        "status": final_status,
        "source_path": str(source_path),
        "source_sha256": initial_source_sha,
        "source_sha256_final": final_source_sha,
        "source_changed": bool(source_changed),
        "source_guard": {
            "status": "blocked" if source_changed or receipt_mismatch else "passed",
            "checks": source_guard_checks,
            "receipt_hashes": receipt_source_hashes,
            "receipt_hashes_match": not receipt_mismatch,
        },
        "backend": backend,
        "engine": "llm",
        "fallback_policy": fallback_policy,
        "outputs": list(names),
        "benchmark_requested": bool(benchmark),
        "stage_count": len(stages),
        "completed_stage_count": sum(stage.status == "passed" for stage in stages),
        "stages": [stage.to_dict() for stage in stages],
        "receipts": {
            "analysis": str(analysis_receipt)
            if _safe_output_file(analysis_receipt, root)
            else None,
            "migration": str(migration_dir / "script-migration-report.json")
            if _safe_output_file(migration_dir / "script-migration-report.json", root)
            else None,
            "verification": str(verification_receipt)
            if _safe_output_file(verification_receipt, root)
            else None,
            "benchmark": str(benchmark_receipt)
            if _safe_output_file(benchmark_receipt, root)
            else None,
        },
        "next_action": (
            "source changed or a stage receipt bound a different source; inspect the guard and rerun with a fresh output directory"
            if source_changed or receipt_mismatch
            else "complete"
            if final_status == "passed"
            else "inspect the failed stage stderr and resume with a fresh output directory"
        ),
        "claims": {
            "model_generated": False,
            "numerically_verified": False,
            "strict_npu_performance_verified": False,
        },
    }
    # Derive claims only from child receipts; the coordinator itself never
    # upgrades a status.  This keeps a shim/diagnostic run from being marketed
    # as an NPU result.
    stage_statuses = {stage.name: stage.status for stage in stages}
    # A receipt is evidence, but the coordinator's child exit status is the
    # authoritative outcome for this run.  Do not surface a positive claim
    # from a receipt that a failed/blocked child happened to leave behind.
    report["claims"]["model_generated"] = bool(
        stage_statuses.get("02-migrate") == "passed"
        and migration_value
        and migration_value.get("actual_remote_call") is True
    )
    report["claims"]["numerically_verified"] = bool(
        stage_statuses.get("03-verify") == "passed"
        and verification_value
        and verification_value.get("status") == "passed"
    )
    report_path = root / "workflow-report.json"
    report["workflow_report"] = str(report_path)
    performance_value = (
        benchmark_value.get("performance")
        if isinstance(benchmark_value, Mapping)
        else None
    )
    report["claims"]["strict_npu_performance_verified"] = bool(
        stage_statuses.get("04-benchmark") == "passed"
        and benchmark_value
        and benchmark_value.get("status") == "passed"
        and isinstance(performance_value, Mapping)
        and performance_value.get("performance_valid") is True
    )
    if source_changed or receipt_mismatch:
        # A child may have produced a valid receipt for an earlier source, but
        # it is not safe to advertise that fact from this workflow run once the
        # source binding is no longer closed.
        report["claims"] = {
            "model_generated": False,
            "numerically_verified": False,
            "strict_npu_performance_verified": False,
        }
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    from numpilot.roles import Role
    with progress.operation(Role.REPORTER, "汇总子进程回执并写入工作流报告"):
        _write_private(report_path, payload)
    progress.emit(None, "completed" if final_status == "passed" else "blocked" if final_status == "blocked" else "failed",
                  "流程结束，请查看工作流报告与下一步建议", component="workflow")
    save_progress()
    return report


def _bounded_sha256_file(path: Path) -> str | None:
    """Hash one bounded regular file without allocating its whole contents."""

    try:
        hasher = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_RECEIPT_BYTES:
                    return None
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def replay_workflow(run_dir: os.PathLike[str] | str) -> dict[str, Any]:
    """Read-only integrity summary for a completed ordinary workflow.

    Replay deliberately does not execute the source, invoke a model, import
    AsNumpy, or run an NPU stage. It validates the immutable workflow index
    and the stage receipts that the original coordinator published, making it
    safe for the terminal and sidebar adapters to use the same evidence.
    """

    root = _safe_path(run_dir, "workflow run")
    if not root.is_dir() or root.is_symlink():
        raise WorkflowError("invalid_run", "workflow run must be a regular directory")
    _require_private_directory(root)

    report_path = root / "workflow-report.json"
    report = _load_json(report_path, root=root)
    if report is None:
        raise WorkflowError("invalid_report", "workflow-report.json is invalid or missing")
    if report.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise WorkflowError("invalid_report", "unsupported workflow report schema")
    if report.get("record_type") != "numpilot_ordinary_workflow":
        raise WorkflowError("invalid_report", "unexpected workflow report type")

    errors: list[str] = []
    receipts_value = report.get("receipts")
    receipts = receipts_value if isinstance(receipts_value, Mapping) else {}
    if not isinstance(receipts_value, Mapping):
        errors.append("receipts is not an object")

    receipt_summary: dict[str, dict[str, Any]] = {}
    for label, raw_path in sorted(receipts.items()):
        if raw_path is None:
            receipt_summary[str(label)] = {"present": False, "status": "not_run"}
            continue
        if not isinstance(raw_path, str) or not raw_path:
            errors.append("receipt path for %s is invalid" % label)
            receipt_summary[str(label)] = {"present": False, "status": "invalid"}
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not _safe_output_file(candidate, root):
            errors.append("receipt path for %s escapes the run directory" % label)
            receipt_summary[str(label)] = {
                "present": False,
                "status": "invalid",
                "path": str(candidate),
            }
            continue
        payload = _load_json(candidate, root=root)
        digest = _bounded_sha256_file(candidate)
        if payload is None or digest is None:
            errors.append("receipt for %s is unreadable" % label)
            receipt_summary[str(label)] = {
                "present": True,
                "status": "invalid",
                "path": str(candidate),
            }
            continue
        receipt_summary[str(label)] = {
            "present": True,
            "status": str(payload.get("status", "published")),
            "path": str(candidate),
            "sha256": digest,
        }

    stages_value = report.get("stages")
    stages = stages_value if isinstance(stages_value, list) else []
    if not isinstance(stages_value, list):
        errors.append("stages is not an array")
    stage_summary: list[dict[str, Any]] = []
    stage_receipt = {
        "01-analyze": "analysis",
        "02-migrate": "migration",
        "03-verify": "verification",
        "04-benchmark": "benchmark",
    }
    for stage in stages:
        if not isinstance(stage, Mapping):
            errors.append("stage entry is not an object")
            continue
        name = stage.get("name")
        status = stage.get("status")
        if not isinstance(name, str) or not isinstance(status, str):
            errors.append("stage entry has invalid name or status")
            continue
        stage_summary.append({"name": name, "status": status})
        receipt_label = stage_receipt.get(name)
        if status == "passed" and receipt_label:
            receipt = receipt_summary.get(receipt_label)
            if not receipt or receipt.get("status") == "invalid" or not receipt.get("present"):
                errors.append("passed stage %s has no valid receipt" % name)

    report_digest = _bounded_sha256_file(report_path)
    if report_digest is None:
        raise WorkflowError("invalid_report", "workflow report is unreadable")

    return {
        "schema_version": 1,
        "record_type": "numpilot_workflow_replay",
        "run_dir": str(root),
        "workflow_report": str(report_path),
        "workflow_report_sha256": report_digest,
        "workflow_status": report.get("status", "unknown"),
        "source": {
            "path": report.get("source_path"),
            "sha256": report.get("source_sha256"),
        },
        "stages": stage_summary,
        "receipts": receipt_summary,
        "receipt_integrity": {
            "status": "passed" if not errors else "failed",
            "errors": errors,
        },
        "claims": dict(report.get("claims")) if isinstance(report.get("claims"), Mapping) else {},
        "replay_mode": "read_only",
        "execution_performed": False,
        "actual_remote_call": False,
    }
