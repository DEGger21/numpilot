"""Fail-closed numerical verification for ordinary migrated scripts.

``verify_script`` binds an immutable verification receipt to the source file,
the migration report, and its sibling ``migrated.py``.  It executes snapshots
of the original NumPy script and migrated script in separate ``python -I``
subprocesses.  The caller must name every result variable explicitly; stdout is
discarded and is never interpreted as evidence.  Values cross the process
boundary only as bounded, numeric ``.npy`` files loaded with
``allow_pickle=False``.

The MVP output contract accepts one numeric-or-boolean array/scalar leaf per
name; nested containers, strings, structured dtypes, and object dtypes are
rejected.  Because workers use Python isolated mode (``-I``), AsNumPy and every
other runtime dependency must be installed in the explicitly selected Python
interpreter; ``PYTHONPATH`` injection is intentionally unavailable.

Subprocess isolation provides state separation and a killable timeout.  It is
not an OS sandbox: user code can still access resources permitted to the user
account.  The receipt therefore proves only numerical consistency of the named
outputs.  It never claims strict-NPU provenance or valid performance evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from numpilot.comparison import compare_outputs


_MAX_SCRIPT_BYTES = 4 * 1024 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_MAX_PYTHON_EXECUTABLE_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_ALLOWED_OUTPUT_BYTES = 1024 * 1024 * 1024
_MAX_OUTPUT_NAMES = 128
_MIGRATED_STATUSES = frozenset({
    "migrated_strict_npu",
    "migrated_scoped_npu",
    "migrated_with_fallback",
})
_WORKER_CODES = frozenset({
    "input_read_failed",
    "input_too_large",
    "invalid_contract",
    "invalid_input_file",
    "invalid_output_directory",
    "missing_output",
    "numpy_unavailable",
    "output_conversion_failed",
    "output_serialization_failed",
    "output_too_large",
    "python_identity_mismatch",
    "script_exception",
    "script_hash_mismatch",
    "unexpected_output_file",
    "unsupported_dtype",
    "worker_error_metadata_too_large",
    "worker_hash_mismatch",
    "worker_internal_error",
})
_STABLE_MESSAGES = {
    "invalid_argument": "verification arguments are invalid",
    "invalid_output_path": "verification receipt path is invalid",
    "receipt_exists": "verification receipt already exists",
    "output_aliases_input": "verification receipt path aliases an input",
    "input_not_regular": "verification input must be a regular non-symlink file",
    "input_too_large": "verification input exceeds the size limit",
    "input_read_failed": "verification input could not be read",
    "invalid_report": "migration report contract is invalid",
    "source_hash_mismatch": "source file does not match the migration report",
    "source_path_mismatch": "source path does not match the migration report",
    "migrated_hash_mismatch": "migrated.py does not match the migration report",
    "input_drift": "a hash-bound input changed during verification",
    "python_unavailable": "verification Python executable is unavailable",
    "python_identity_mismatch": "verification Python identity changed before execution",
    "execution_timeout": "script execution exceeded the verification timeout",
    "execution_failed": "script subprocess did not produce valid evidence",
    "invalid_worker_output": "script subprocess output contract is invalid",
    "missing_output": "script did not define every contracted output variable",
    "unsupported_dtype": "a contracted output is not a numeric or boolean value",
    "output_too_large": "serialized verification output exceeds the size limit",
    "script_exception": "script raised an exception during verification",
    "worker_hash_mismatch": "verification worker does not match its bound digest",
    "numerical_mismatch": "migrated outputs are not numerically consistent",
    "receipt_publish_failed": "verification receipt could not be published",
    "internal_error": "verification failed closed due to an internal error",
}


class ScriptVerificationError(RuntimeError):
    """Stable failure raised by :func:`verify_script`.

    ``receipt`` is populated when execution or comparison reached a state that
    could be recorded at the requested write-once path.  Callers can therefore
    distinguish an auditable verification failure from a pre-publication usage
    or destination error without parsing the message.
    """

    def __init__(
        self,
        code: str,
        message: Optional[str] = None,
        *,
        phase: str = "preflight",
        exception_type: Optional[str] = None,
        receipt: Optional["ScriptVerificationReceipt"] = None,
    ) -> None:
        stable_message = message or _STABLE_MESSAGES.get(
            code, _STABLE_MESSAGES["internal_error"]
        )
        super().__init__(stable_message)
        self.code = code
        self.phase = phase
        self.exception_type = exception_type
        self.receipt = receipt


@dataclass(frozen=True)
class ScriptVerificationReceipt:
    """Published, content-addressed numerical verification receipt."""

    path: str
    sha256: str
    payload: Mapping[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-compatible receipt payload."""

        return json.loads(json.dumps(self.payload, ensure_ascii=False))


@dataclass(frozen=True)
class _BoundInputs:
    source_path: Path
    source_payload: bytes
    source_sha256: str
    report_path: Path
    report_payload: bytes
    report_sha256: str
    report: Mapping[str, Any]
    migrated_path: Path
    migrated_payload: bytes
    migrated_sha256: str


@dataclass(frozen=True)
class _ExecutionResult:
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    removed_secret_variables: int


@dataclass(frozen=True)
class _PythonExecutableIdentity:
    invocation_path: str
    realpath: str
    file_sha256: str
    file_size: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_path(value: os.PathLike[str] | str, *, code: str) -> Path:
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise ScriptVerificationError(code) from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or "\x00" in rendered
        or len(rendered) > 4096
    ):
        raise ScriptVerificationError(code)
    return Path(os.path.abspath(os.path.expanduser(rendered)))


def _read_regular(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(str(path), flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ScriptVerificationError("input_not_regular")
        if metadata.st_size > limit:
            raise ScriptVerificationError("input_too_large")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise ScriptVerificationError("input_too_large")
        return payload
    except ScriptVerificationError:
        raise
    except OSError as exc:
        code = "input_not_regular" if path.is_symlink() else "input_read_failed"
        raise ScriptVerificationError(code, exception_type=type(exc).__name__) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_report(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScriptVerificationError(
            "invalid_report", exception_type=type(exc).__name__
        ) from exc
    if not isinstance(value, dict):
        raise ScriptVerificationError("invalid_report")
    return value


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _load_bound_inputs(
    source_path: Path,
    report_path: Path,
    migrated_path_value: os.PathLike[str] | str | None,
) -> _BoundInputs:
    source_payload = _read_regular(source_path, _MAX_SCRIPT_BYTES)
    report_payload = _read_regular(report_path, _MAX_REPORT_BYTES)
    try:
        source_payload.decode("utf-8")
    except UnicodeError as exc:
        raise ScriptVerificationError(
            "invalid_report", exception_type=type(exc).__name__
        ) from exc
    report = _parse_report(report_payload)

    expected_migrated_path = report_path.parent / "migrated.py"
    if migrated_path_value is None:
        migrated_path = expected_migrated_path
    else:
        migrated_path = _absolute_path(
            migrated_path_value, code="invalid_argument"
        )
        if _normal_path(migrated_path) != _normal_path(expected_migrated_path):
            raise ScriptVerificationError("invalid_report")
    migrated_payload = _read_regular(migrated_path, _MAX_SCRIPT_BYTES)
    try:
        migrated_payload.decode("utf-8")
    except UnicodeError as exc:
        raise ScriptVerificationError(
            "invalid_report", exception_type=type(exc).__name__
        ) from exc

    source_sha256 = _sha256(source_payload)
    migrated_sha256 = _sha256(migrated_payload)
    if (
        report.get("schema_version") != 2
        or report.get("mode") != "script"
        or report.get("status") not in _MIGRATED_STATUSES
        or report.get("ast_audit") != "passed"
        or report.get("execution_performed") is not False
        or report.get("numerical_verification") != "not_run"
        or report.get("performance_valid") is not False
        or report.get("published_files") != ["migrated.py"]
        or not _is_sha256(report.get("source_sha256"))
        or not _is_sha256(report.get("migrated_sha256"))
    ):
        raise ScriptVerificationError("invalid_report")
    if report["source_sha256"] != source_sha256:
        raise ScriptVerificationError("source_hash_mismatch")
    if report["migrated_sha256"] != migrated_sha256:
        raise ScriptVerificationError("migrated_hash_mismatch")

    analysis = report.get("analysis")
    scan = analysis.get("scan") if isinstance(analysis, dict) else None
    if (
        not isinstance(scan, dict)
        or scan.get("ok") is not True
        or scan.get("source_sha256") != source_sha256
        or not isinstance(scan.get("source_path"), str)
    ):
        raise ScriptVerificationError("invalid_report")
    scan_source_path = scan.get("source_path")
    if (
        not isinstance(scan_source_path, str)
        or not scan_source_path
        or scan_source_path != scan_source_path.strip()
        or "\x00" in scan_source_path
        or len(scan_source_path) > 4096
        or not os.path.isabs(scan_source_path)
    ):
        raise ScriptVerificationError("invalid_report")
    report_source_path = Path(os.path.abspath(scan_source_path))
    if _normal_path(report_source_path) != _normal_path(source_path):
        raise ScriptVerificationError("source_path_mismatch")
    qualified = report.get("qualified_analysis")
    if isinstance(qualified, dict) and qualified.get("source_sha256") != source_sha256:
        raise ScriptVerificationError("invalid_report")

    return _BoundInputs(
        source_path=source_path,
        source_payload=source_payload,
        source_sha256=source_sha256,
        report_path=report_path,
        report_payload=report_payload,
        report_sha256=_sha256(report_payload),
        report=report,
        migrated_path=migrated_path,
        migrated_payload=migrated_payload,
        migrated_sha256=migrated_sha256,
    )


def _validate_output_names(output_names: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(output_names, (str, bytes)):
        raise ScriptVerificationError("invalid_argument")
    try:
        names = tuple(output_names)
    except TypeError as exc:
        raise ScriptVerificationError("invalid_argument") from exc
    if (
        not names
        or len(names) > _MAX_OUTPUT_NAMES
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or not name.isidentifier()
            or len(name) > 128
            for name in names
        )
    ):
        raise ScriptVerificationError("invalid_argument")
    return names


def _validate_limits(timeout_seconds: float, max_output_bytes: int) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.05 <= float(timeout_seconds) <= 3600.0
        or isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or not 1024 <= max_output_bytes <= _MAX_ALLOWED_OUTPUT_BYTES
    ):
        raise ScriptVerificationError("invalid_argument")


def _has_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if os.path.lexists(current) and current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _prepare_receipt_path(
    out_path: Path, input_paths: Sequence[Path]
) -> None:
    if any(
        _normal_path(out_path) == _normal_path(input_path)
        for input_path in input_paths
    ):
        raise ScriptVerificationError("output_aliases_input")
    if os.path.lexists(out_path):
        raise ScriptVerificationError("receipt_exists")
    parent = out_path.parent
    if _has_symlink_component(parent):
        raise ScriptVerificationError("invalid_output_path")
    try:
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise ScriptVerificationError(
            "invalid_output_path", exception_type=type(exc).__name__
        ) from exc
    if parent.is_symlink() or not parent.is_dir():
        raise ScriptVerificationError("invalid_output_path")


def _python_file_identity(path: Path) -> Tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > _MAX_PYTHON_EXECUTABLE_BYTES
        ):
            raise ScriptVerificationError("python_unavailable")
        digest = hashlib.sha256()
        total = 0
        while total <= _MAX_PYTHON_EXECUTABLE_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    _MAX_PYTHON_EXECUTABLE_BYTES + 1 - total,
                ),
            )
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if (
            total > _MAX_PYTHON_EXECUTABLE_BYTES
            or total != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ScriptVerificationError("python_unavailable")
        return total, digest.hexdigest()
    except ScriptVerificationError:
        raise
    except OSError as exc:
        raise ScriptVerificationError(
            "python_unavailable", exception_type=type(exc).__name__
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _python_executable(
    value: os.PathLike[str] | str | None,
) -> _PythonExecutableIdentity:
    selected = sys.executable if value is None else value
    path = _absolute_path(selected, code="python_unavailable")
    resolved = Path(os.path.realpath(str(path)))
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ScriptVerificationError("python_unavailable")
    # Preserve the invocation path.  Resolving ``.venv/bin/python`` to its base
    # binary changes Python's prefix discovery and can silently lose the target
    # venv (including its AsNumPy installation).
    file_size, file_sha256 = _python_file_identity(resolved)
    return _PythonExecutableIdentity(
        invocation_path=str(path),
        realpath=str(resolved),
        file_sha256=file_sha256,
        file_size=file_size,
    )


def _write_private(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _is_sensitive_environment_key(key: str) -> bool:
    upper = key.upper()
    if upper.startswith(("NUMPILOT_LLM_", "OPENAI_", "ANTHROPIC_")):
        return True
    if upper in {
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "HUAWEICLOUD_SDK_AK",
        "HUAWEICLOUD_SDK_SK",
    }:
        return True
    return any(
        marker in upper
        for marker in (
            "API_KEY",
            "ACCESS_KEY",
            "AUTH_TOKEN",
            "PRIVATE_KEY",
            "PASSWORD",
            "SECRET",
        )
    ) or upper.endswith(("_TOKEN", "_CREDENTIALS"))


def _worker_environment(home: Path) -> Tuple[Dict[str, str], int]:
    environment = {}
    removed = 0
    for key, value in os.environ.items():
        if _is_sensitive_environment_key(key):
            removed += 1
            continue
        environment[key] = value
    environment.update({
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    })
    return environment, removed


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=5)
    except BaseException:
        try:
            process.kill()
        except BaseException:
            pass
        try:
            process.wait(timeout=5)
        except BaseException:
            pass


def _load_json_file(path: Path, limit: int, code: str) -> Mapping[str, Any]:
    try:
        raw = _read_regular(path, limit)
        value = json.loads(raw.decode("utf-8"))
    except ScriptVerificationError as exc:
        raise ScriptVerificationError(code, phase="execution_evidence") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScriptVerificationError(
            code, phase="execution_evidence", exception_type=type(exc).__name__
        ) from exc
    if not isinstance(value, dict):
        raise ScriptVerificationError(code, phase="execution_evidence")
    return value


def _worker_failure(output_dir: Path, returncode: int) -> ScriptVerificationError:
    error_path = output_dir / "error.json"
    if not error_path.is_file() or error_path.is_symlink():
        return ScriptVerificationError(
            "execution_failed", phase="script_execution"
        )
    value = _load_json_file(error_path, 16 * 1024, "execution_failed")
    worker_code = value.get("code")
    phase = value.get("phase")
    exception_type = value.get("exception_type")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "failed"
        or worker_code not in _WORKER_CODES
        or not isinstance(phase, str)
        or (exception_type is not None and not isinstance(exception_type, str))
    ):
        return ScriptVerificationError(
            "execution_failed", phase="script_execution"
        )
    public_code = worker_code if worker_code in _STABLE_MESSAGES else "execution_failed"
    return ScriptVerificationError(
        public_code,
        phase=phase,
        exception_type=exception_type,
    )


def _load_worker_success(
    output_dir: Path,
    *,
    script_sha256: str,
    worker_sha256: str,
    python_identity: _PythonExecutableIdentity,
    output_names: Tuple[str, ...],
    max_output_bytes: int,
) -> Tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json_file(
        manifest_path, _MAX_MANIFEST_BYTES, "invalid_worker_output"
    )
    entries = manifest.get("outputs")
    runtime_modules = manifest.get("loaded_runtime_modules")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("script_sha256") != script_sha256
        or manifest.get("worker_sha256") != worker_sha256
        or manifest.get("python_invocation_path")
        != python_identity.invocation_path
        or not isinstance(manifest.get("pid"), int)
        or isinstance(manifest.get("pid"), bool)
        or manifest.get("pid") <= 0
        or manifest.get("output_names") != list(output_names)
        or not isinstance(manifest.get("python_executable"), str)
        or manifest.get("python_executable_realpath")
        != python_identity.realpath
        or manifest.get("python_executable_sha256")
        != python_identity.file_sha256
        or manifest.get("python_executable_size")
        != python_identity.file_size
        or not isinstance(manifest.get("python_prefix"), str)
        or not isinstance(manifest.get("python_base_prefix"), str)
        or not isinstance(manifest.get("python_version"), str)
        or not isinstance(runtime_modules, dict)
        or "numpy" not in runtime_modules
        or any(
            name not in {"numpy", "asnumpy", "asnumpy._core"}
            or not isinstance(evidence, dict)
            or evidence.get("file_hash_status")
            not in {"hashed", "too_large", "unavailable"}
            or (
                evidence.get("file_hash_status") == "hashed"
                and not _is_sha256(evidence.get("file_sha256"))
            )
            or (
                evidence.get("path") is not None
                and not isinstance(evidence.get("path"), str)
            )
            or (
                evidence.get("realpath") is not None
                and not isinstance(evidence.get("realpath"), str)
            )
            for name, evidence in runtime_modules.items()
        )
        or not isinstance(entries, list)
        or len(entries) != len(output_names)
    ):
        raise ScriptVerificationError(
            "invalid_worker_output", phase="execution_evidence"
        )

    expected_files = {"manifest.json"}
    arrays: Dict[str, np.ndarray] = {}
    total_data_bytes = 0
    total_serialized_bytes = 0
    for index, (name, entry) in enumerate(zip(output_names, entries)):
        expected_filename = "output-%04d.npy" % index
        if (
            not isinstance(entry, dict)
            or entry.get("name") != name
            or entry.get("filename") != expected_filename
            or not _is_sha256(entry.get("sha256"))
            or not isinstance(entry.get("serialized_bytes"), int)
            or isinstance(entry.get("serialized_bytes"), bool)
            or not isinstance(entry.get("data_bytes"), int)
            or isinstance(entry.get("data_bytes"), bool)
            or not isinstance(entry.get("dtype"), str)
            or not isinstance(entry.get("shape"), list)
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in entry.get("shape", [])
            )
        ):
            raise ScriptVerificationError(
                "invalid_worker_output", phase="execution_evidence"
            )
        expected_files.add(expected_filename)
        payload = _read_regular(output_dir / expected_filename, max_output_bytes)
        if (
            len(payload) != entry["serialized_bytes"]
            or _sha256(payload) != entry["sha256"]
        ):
            raise ScriptVerificationError(
                "invalid_worker_output", phase="execution_evidence"
            )
        total_serialized_bytes += len(payload)
        if total_serialized_bytes > max_output_bytes:
            raise ScriptVerificationError(
                "output_too_large", phase="execution_evidence"
            )
        try:
            # Parse the exact bytes whose length and hash were validated above;
            # never reopen the worker pathname after validation.
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise ScriptVerificationError(
                "invalid_worker_output",
                phase="execution_evidence",
                exception_type=type(exc).__name__,
            ) from exc
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.hasobject
            or array.dtype.kind not in "biufc"
            or array.dtype.str != entry["dtype"]
            or list(array.shape) != entry["shape"]
            or int(array.nbytes) != entry["data_bytes"]
        ):
            raise ScriptVerificationError(
                "invalid_worker_output", phase="execution_evidence"
            )
        arrays[name] = array
        total_data_bytes += int(array.nbytes)
        if total_data_bytes > max_output_bytes:
            raise ScriptVerificationError(
                "output_too_large", phase="execution_evidence"
            )

    try:
        actual_files = {item.name for item in output_dir.iterdir()}
    except OSError as exc:
        raise ScriptVerificationError(
            "invalid_worker_output",
            phase="execution_evidence",
            exception_type=type(exc).__name__,
        ) from exc
    if actual_files != expected_files:
        raise ScriptVerificationError(
            "invalid_worker_output", phase="execution_evidence"
        )
    if (
        manifest.get("total_data_bytes") != total_data_bytes
        or total_data_bytes > max_output_bytes
        or manifest.get("total_serialized_bytes") != total_serialized_bytes
        or total_serialized_bytes > max_output_bytes
    ):
        raise ScriptVerificationError(
            "invalid_worker_output", phase="execution_evidence"
        )
    return manifest, arrays


def _run_worker(
    *,
    root: Path,
    label: str,
    worker_path: Path,
    script_path: Path,
    script_sha256: str,
    worker_sha256: str,
    output_names: Tuple[str, ...],
    python_identity: _PythonExecutableIdentity,
    timeout_seconds: float,
    max_output_bytes: int,
) -> _ExecutionResult:
    side = root / label
    output_dir = side / "outputs"
    home = side / "home"
    for directory in (side, output_dir, home):
        directory.mkdir(mode=0o700)
    contract = {
        "schema_version": 1,
        "script_sha256": script_sha256,
        "worker_sha256": worker_sha256,
        "python_invocation_path": python_identity.invocation_path,
        "python_invocation_realpath": python_identity.realpath,
        "python_invocation_sha256": python_identity.file_sha256,
        "python_invocation_size": python_identity.file_size,
        "output_names": list(output_names),
        "max_script_bytes": _MAX_SCRIPT_BYTES,
        "max_output_bytes": max_output_bytes,
    }
    contract_path = side / "contract.json"
    contract_payload = (
        json.dumps(contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    _write_private(contract_path, contract_payload)
    environment, removed = _worker_environment(home)
    command = [
        python_identity.invocation_path,
        "-I",
        str(worker_path),
        "--script",
        str(script_path),
        "--contract",
        str(contract_path),
        "--out",
        str(output_dir),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(side),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise ScriptVerificationError(
            "execution_failed",
            phase=label + ":process_start",
            exception_type=type(exc).__name__,
        ) from exc
    try:
        try:
            process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            raise ScriptVerificationError(
                "execution_timeout", phase=label + ":script_execution"
            ) from exc
    finally:
        # This runs for timeout, KeyboardInterrupt/SystemExit, unexpected wait
        # failures, and successful completion.  It also removes descendants
        # left in the worker's private process group.
        _terminate_process_group(process)
    if process.returncode != 0:
        failure = _worker_failure(output_dir, process.returncode)
        failure.phase = label + ":" + failure.phase
        raise failure
    manifest, arrays = _load_worker_success(
        output_dir,
        script_sha256=script_sha256,
        worker_sha256=worker_sha256,
        python_identity=python_identity,
        output_names=output_names,
        max_output_bytes=max_output_bytes,
    )
    return _ExecutionResult(
        manifest=manifest,
        arrays=arrays,
        removed_secret_variables=removed,
    )


def _assert_inputs_unchanged(inputs: _BoundInputs) -> None:
    current = (
        _sha256(_read_regular(inputs.source_path, _MAX_SCRIPT_BYTES)),
        _sha256(_read_regular(inputs.report_path, _MAX_REPORT_BYTES)),
        _sha256(_read_regular(inputs.migrated_path, _MAX_SCRIPT_BYTES)),
    )
    expected = (
        inputs.source_sha256,
        inputs.report_sha256,
        inputs.migrated_sha256,
    )
    if current != expected:
        raise ScriptVerificationError("input_drift", phase="hash_recheck")


def _execution_summary(result: _ExecutionResult) -> Dict[str, Any]:
    manifest = result.manifest
    return {
        "status": "completed",
        "script_sha256": manifest["script_sha256"],
        "worker_sha256": manifest["worker_sha256"],
        "pid": manifest["pid"],
        "python_invocation_path": manifest["python_invocation_path"],
        "python_executable": manifest["python_executable"],
        "python_executable_realpath": manifest["python_executable_realpath"],
        "python_executable_sha256": manifest["python_executable_sha256"],
        "python_executable_size": manifest["python_executable_size"],
        "python_prefix": manifest["python_prefix"],
        "python_base_prefix": manifest["python_base_prefix"],
        "python_version": manifest["python_version"],
        "loaded_runtime_modules": {
            name: dict(evidence)
            for name, evidence in manifest["loaded_runtime_modules"].items()
        },
        "output_names": list(manifest["output_names"]),
        "total_data_bytes": manifest["total_data_bytes"],
        "total_serialized_bytes": manifest["total_serialized_bytes"],
        "outputs": [dict(item) for item in manifest["outputs"]],
        "removed_sensitive_environment_variables": (
            result.removed_secret_variables
        ),
        "stdout_used_as_evidence": False,
        "stderr_used_as_evidence": False,
    }


def _verification_id(payload: Mapping[str, Any]) -> str:
    identity = {
        "receipt_type": payload["receipt_type"],
        "status": payload["status"],
        "source_sha256": (payload.get("source") or {}).get("sha256"),
        "migration_report_sha256": (payload.get("migration") or {}).get(
            "report_sha256"
        ),
        "migrated_sha256": (payload.get("migration") or {}).get(
            "migrated_sha256"
        ),
        "contract": payload.get("contract"),
        "executions": payload.get("executions"),
        "comparisons": payload.get("comparisons"),
        "failure": payload.get("failure"),
    }
    encoded = json.dumps(
        identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def _base_receipt(
    *,
    status: str,
    started_at: str,
    finished_at: str,
    source_path: Path,
    report_path: Path,
    output_names: Tuple[str, ...],
    timeout_seconds: float,
    max_output_bytes: int,
    inputs: _BoundInputs,
    executions: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    failure: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "ordinary_script_numerical_verification",
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "source": {
            "path": str(source_path),
            "sha256": inputs.source_sha256,
        },
        "migration": {
            "report_path": str(report_path),
            "report_sha256": inputs.report_sha256,
            "report_status": inputs.report.get("status"),
            "migrated_path": str(inputs.migrated_path),
            "migrated_sha256": inputs.migrated_sha256,
        },
        "contract": {
            "output_names": list(output_names),
            "timeout_seconds_per_script": float(timeout_seconds),
            "max_total_serialized_bytes_per_script": max_output_bytes,
            "transport": "json_manifest_and_npy_allow_pickle_false",
            "stdout_is_result_channel": False,
        },
        "executions": dict(executions),
        "comparisons": dict(comparisons),
        "numerically_consistent": status == "passed",
        "strict_npu_verified": False,
        "performance_valid": False,
        "execution_boundary": {
            "subprocess_isolation": True,
            "os_sandbox": False,
            "explicit_execution": True,
            "stdout_used_as_evidence": False,
            "stderr_used_as_evidence": False,
        },
        "evidence_boundary": {
            "numerical_only": True,
            "strict_npu_verified": False,
            "performance_valid": False,
            "named_numeric_or_boolean_leaves_only": True,
        },
        "claims": {
            "named_outputs_numerically_consistent": status == "passed",
            "strict_npu_provenance_verified": False,
            "performance_verified": False,
            "subprocess_isolation": True,
            "os_sandbox": False,
            "scope": "only the explicitly named output variables",
        },
        "failure": dict(failure) if failure else None,
    }
    payload["verification_id"] = _verification_id(payload)
    return payload


def _publish_receipt(
    path: Path, payload: Mapping[str, Any]
) -> ScriptVerificationReceipt:
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ScriptVerificationError("receipt_publish_failed")
    parent = path.parent
    descriptor = -1
    temporary = ""
    published = False
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + path.name + "-", suffix=".tmp", dir=str(parent)
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            published = True
        except FileExistsError as exc:
            raise ScriptVerificationError("receipt_exists") from exc
        os.unlink(temporary)
        temporary = ""
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(str(parent), directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except ScriptVerificationError:
        raise
    except OSError as exc:
        error = ScriptVerificationError(
            "receipt_publish_failed", exception_type=type(exc).__name__
        )
        if published:
            error.receipt = ScriptVerificationReceipt(
                path=str(path), sha256=_sha256(raw), payload=dict(payload)
            )
        raise error from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return ScriptVerificationReceipt(
        path=str(path), sha256=_sha256(raw), payload=dict(payload)
    )


def verify_script(
    source_path: os.PathLike[str] | str,
    migration_report_path: os.PathLike[str] | str,
    output_names: Sequence[str],
    out_path: os.PathLike[str] | str,
    *,
    migrated_path: os.PathLike[str] | str | None = None,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    source_python_executable: os.PathLike[str] | str | None = None,
    migrated_python_executable: os.PathLike[str] | str | None = None,
) -> ScriptVerificationReceipt:
    """Verify explicitly named outputs and publish one write-once receipt.

    The migration report must be a schema-v2 ordinary-script report emitted by
    NumPilot.  Its top-level and nested source hashes/path must match
    ``source_path`` and its ``migrated_sha256`` must match the sibling
    ``migrated.py``.  ``migrated_path`` is only an explicit assertion of that
    sibling path; alternate artifacts are rejected.

    Every named global must resolve to one numeric-or-boolean scalar/array (or
    an AsNumPy value exposing ``to_numpy``).  Containers and object/string
    dtypes are outside this receipt schema and fail closed.

    A failed execution or numerical comparison publishes a ``failed`` receipt
    and raises :class:`ScriptVerificationError` with ``error.receipt`` set.
    Destination/argument failures and source/report/hash/path admission failures
    leave ``receipt`` as ``None``.  Existing receipts are never replaced.
    """

    names = _validate_output_names(output_names)
    _validate_limits(timeout_seconds, max_output_bytes)
    source = _absolute_path(source_path, code="invalid_argument")
    report_path = _absolute_path(
        migration_report_path, code="invalid_argument"
    )
    receipt_path = _absolute_path(out_path, code="invalid_output_path")
    _prepare_receipt_path(receipt_path, (source, report_path))
    source_python = _python_executable(source_python_executable)
    migrated_python = (
        source_python
        if migrated_python_executable is None
        else _python_executable(migrated_python_executable)
    )
    # No receipt is published until the source, report, and migrated artifact
    # form one complete, validated binding.  This prevents hash-null rejection
    # receipts from being mistaken for content-bound evidence.
    inputs = _load_bound_inputs(source, report_path, migrated_path)
    if _normal_path(receipt_path) == _normal_path(inputs.migrated_path):
        raise ScriptVerificationError("output_aliases_input")

    started_at = _utc_now()
    executions: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    try:
        _assert_inputs_unchanged(inputs)

        worker_source = Path(__file__).with_name(
            "_script_verification_worker.py"
        )
        worker_payload = _read_regular(worker_source, _MAX_SCRIPT_BYTES)
        worker_sha256 = _sha256(worker_payload)
        with tempfile.TemporaryDirectory(prefix="numpilot-script-verify-") as raw_root:
            root = Path(raw_root)
            os.chmod(root, 0o700)
            worker_snapshot = root / "worker.py"
            source_snapshot = root / "source.py"
            migrated_snapshot = root / "migrated.py"
            _write_private(worker_snapshot, worker_payload)
            _write_private(source_snapshot, inputs.source_payload)
            _write_private(migrated_snapshot, inputs.migrated_payload)

            source_result = _run_worker(
                root=root,
                label="source",
                worker_path=worker_snapshot,
                script_path=source_snapshot,
                script_sha256=inputs.source_sha256,
                worker_sha256=worker_sha256,
                output_names=names,
                python_identity=source_python,
                timeout_seconds=float(timeout_seconds),
                max_output_bytes=max_output_bytes,
            )
            executions["source"] = _execution_summary(source_result)
            _assert_inputs_unchanged(inputs)
            migrated_result = _run_worker(
                root=root,
                label="migrated",
                worker_path=worker_snapshot,
                script_path=migrated_snapshot,
                script_sha256=inputs.migrated_sha256,
                worker_sha256=worker_sha256,
                output_names=names,
                python_identity=migrated_python,
                timeout_seconds=float(timeout_seconds),
                max_output_bytes=max_output_bytes,
            )
            executions["migrated"] = _execution_summary(migrated_result)
            for name in names:
                comparison = compare_outputs(
                    source_result.arrays[name], migrated_result.arrays[name]
                )
                comparisons[name] = asdict(comparison)
            if not all(item["consistent"] for item in comparisons.values()):
                raise ScriptVerificationError(
                    "numerical_mismatch", phase="numerical_comparison"
                )
        _assert_inputs_unchanged(inputs)
        success = _base_receipt(
            status="passed",
            started_at=started_at,
            finished_at=_utc_now(),
            source_path=source,
            report_path=report_path,
            output_names=names,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            inputs=inputs,
            executions=executions,
            comparisons=comparisons,
            failure=None,
        )
        return _publish_receipt(receipt_path, success)
    except ScriptVerificationError as exc:
        if exc.code in {
            "receipt_exists",
            "invalid_output_path",
            "receipt_publish_failed",
        }:
            raise
        failure_payload = _base_receipt(
            status="failed",
            started_at=started_at,
            finished_at=_utc_now(),
            source_path=source,
            report_path=report_path,
            output_names=names,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            inputs=inputs,
            executions=executions,
            comparisons=comparisons,
            failure={
                "code": exc.code,
                "phase": exc.phase,
                "exception_type": exc.exception_type,
                "message": _STABLE_MESSAGES.get(
                    exc.code, _STABLE_MESSAGES["internal_error"]
                ),
            },
        )
        receipt = _publish_receipt(receipt_path, failure_payload)
        raise ScriptVerificationError(
            exc.code,
            phase=exc.phase,
            exception_type=exc.exception_type,
            receipt=receipt,
        ) from exc
    except Exception as exc:
        failure_payload = _base_receipt(
            status="failed",
            started_at=started_at,
            finished_at=_utc_now(),
            source_path=source,
            report_path=report_path,
            output_names=names,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            inputs=inputs,
            executions=executions,
            comparisons=comparisons,
            failure={
                "code": "internal_error",
                "phase": "internal",
                "exception_type": type(exc).__name__,
                "message": _STABLE_MESSAGES["internal_error"],
            },
        )
        receipt = _publish_receipt(receipt_path, failure_payload)
        raise ScriptVerificationError(
            "internal_error",
            phase="internal",
            exception_type=type(exc).__name__,
            receipt=receipt,
        ) from exc


__all__ = [
    "ScriptVerificationError",
    "ScriptVerificationReceipt",
    "verify_script",
]
