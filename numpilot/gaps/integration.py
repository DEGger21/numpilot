"""Isolated integration transaction for a reviewed official AsNumPy adapter.

The transaction is deliberately local and deterministic.  It copies the
pinned AsNumpy work tree into a private directory, applies exactly the overlay
files bound by the adapter report, builds a wheel without build isolation,
installs that wheel into a private runtime directory, and executes a
NumPilot-owned API correctness gate against a NumPy oracle.  A model is never
consulted, and the shared AsNumpy installation is never modified.

On success the producer appends one satisfied ``integration_transaction``
receipt and advances the gap to ``INTEGRATED_CANDIDATE``.  On failure it
preserves every candidate artifact, publishes a bounded diagnostic, records a
``FAILED_ATTEMPT`` without advancing, and marks the gap ``BLOCKED`` once the
three-repair budget has been consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import signal
import stat
import subprocess
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple
import zipfile

from numpilot.gaps.adapter_recovery import (
    AdapterRecoveryError,
    INTEGRATION_FAILURE_REF_PREFIX,
    _ADAPTER_REF_NAME,
    _ADAPTER_REPAIR_REF_PREFIX,
    _ADAPTER_STATUS_GENERATED,
    _ADAPTER_STATUS_REPAIRED,
    _MAX_REPAIR_ROUNDS,
    _PHASE_INDEX,
    _absolute_path,
    _candidate_gap,
    _gap_for_candidate,
    _latest_adapter_ref,
    _load_adapter_output,
    _official_adapter_input,
    _repair_count,
    _snapshot_regular_file,
)
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    RecoveryError,
    RecoveryEvent,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryStatus,
    RecoveryStore,
    SideEffectTruth,
    StageReceiptKind,
    StageReceiptScope,
    StageReceiptVerdict,
    create_stage_receipt,
)


INTEGRATION_SCHEMA_VERSION = 1
INTEGRATION_RECEIPT_FILENAME = "integration-transaction.json"
DIAGNOSTIC_FILENAME = "failure.json"
INTEGRATION_REF_NAME = "integration_transaction"
MAX_APPEND_ATTEMPTS = 256

_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_LOG_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_FILES = 50_000
_MAX_SOURCE_BYTES = 1024 * 1024 * 1024
_MAX_WHEEL_ENTRY_BYTES = 512 * 1024 * 1024
_MAX_WHEEL_BYTES = 1536 * 1024 * 1024
_MAX_PYTHON_BYTES = 512 * 1024 * 1024
_MAX_DETAIL_CHARS = 2048
_MAX_CASE_DETAIL_CHARS = 512
_MAX_DIAGNOSTIC_TAIL = 8192
_MAX_TIMEOUT_SECONDS = 3600.0
_DEFAULT_TIMEOUT_SECONDS = 1800.0

_IGNORED_DIRECTORIES = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
})
_IGNORED_SUFFIXES = (".egg-info",)
_OMITTED_PYTHON_ENVIRONMENT = frozenset({
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
})


class IntegrationError(RecoveryError):
    """The isolated integration transaction failed closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


INTEGRATION_PREFLIGHT_SCHEMA_VERSION = 1
_REQUIRED_INTEGRATION_POLICY = (
    "subprocess_allowed",
    "system_write_allowed",
    "asnumpy_import_allowed",
    "build_allowed",
    "execution_allowed",
    "activation_allowed",
)


@dataclass(frozen=True)
class IntegrationPreflightResult:
    """Read-only admission result for the isolated integration producer."""

    status: str
    code: str
    next_action: str
    gap_id: str
    candidate_id: str
    candidate_sha256: str
    adapter_artifact_id: str
    adapter_report_sha256: str
    generation_mode: str
    provider: str
    model: str
    configuration_source: str
    reference_root: str
    context_plan_sha256: str
    python_path: str
    python_sha256: str
    policy: Mapping[str, Any]
    required_policy: Tuple[str, ...]
    missing_policy: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": INTEGRATION_PREFLIGHT_SCHEMA_VERSION,
            "record_type": "numpilot_integration_preflight",
            "status": self.status,
            "code": self.code,
            "next_action": self.next_action,
            "gap_id": self.gap_id,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "adapter_artifact_id": self.adapter_artifact_id,
            "adapter_report_sha256": self.adapter_report_sha256,
            "adapter_profile": {
                "generation_mode": self.generation_mode,
                "provider": self.provider,
                "model": self.model,
                "configuration_source": self.configuration_source,
            },
            "reference_root": self.reference_root,
            "context_plan_sha256": self.context_plan_sha256,
            "python": {
                "path": self.python_path,
                "sha256": self.python_sha256,
            },
            "policy": dict(self.policy),
            "required_policy": list(self.required_policy),
            "missing_policy": list(self.missing_policy),
            "side_effects": {
                "subprocess_performed": False,
                "system_write_performed": False,
                "asnumpy_import_performed": False,
                "build_performed": False,
                "execution_performed": False,
                "verification_performed": False,
                "activation_performed": False,
                "model_invocation_performed": False,
                "network_access_performed": False,
            },
        }


@dataclass(frozen=True)
class IntegrationTransactionResult:
    """Deterministic result of one integration producer invocation."""

    recovery: RecoveryStatus
    gap_id: str
    transaction_id: str
    status: str
    out_root: str
    appended_revisions: int
    manual_review_required: bool
    transaction_ref: EvidenceRef
    diagnostic_ref: EvidenceRef | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": INTEGRATION_SCHEMA_VERSION,
            "record_type": "numpilot_integration_transaction_result",
            "gap_id": self.gap_id,
            "transaction_id": self.transaction_id,
            "status": self.status,
            "out_root": self.out_root,
            "appended_revisions": self.appended_revisions,
            "manual_review_required": self.manual_review_required,
            "transaction_ref": self.transaction_ref.to_dict(),
            "diagnostic_ref": (
                None
                if self.diagnostic_ref is None
                else self.diagnostic_ref.to_dict()
            ),
            "model_invoked_by_producer": False,
            "activation_performed": False,
            "recovery": self.recovery.to_dict(),
        }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path, maximum: int = _MAX_LOG_BYTES) -> str:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise IntegrationError(
                    "artifact_too_large", "integration artifact exceeds its bound"
                )
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    try:
        details = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise IntegrationError(
            "artifact_missing", "%s is unavailable" % label
        ) from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_size <= 0
        or details.st_size > maximum
    ):
        raise IntegrationError("artifact_invalid", "%s is invalid" % label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IntegrationError(
            "artifact_missing", "%s is unavailable" % label
        ) from exc
    if len(payload) != details.st_size:
        raise IntegrationError("artifact_changed", "%s changed while read" % label)
    return payload


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, secrets.token_hex(8)))
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise IntegrationError(
                    "output_write_failed", "short integration artifact write"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _write_private_json(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = _canonical(value) + b"\n"
    _write_private(path, payload)
    return payload


def _read_private_json(path: Path, label: str) -> Mapping[str, Any]:
    payload = _read_bounded(path, _MAX_JSON_BYTES, label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrationError("artifact_invalid", "%s is not JSON" % label) from exc
    if not isinstance(value, dict):
        raise IntegrationError("artifact_invalid", "%s must be an object" % label)
    return value


def _is_ignored(name: str) -> bool:
    return name in _IGNORED_DIRECTORIES or name.endswith(_IGNORED_SUFFIXES)


def _copy_ignore(_directory: str, names: Sequence[str]) -> list[str]:
    return [name for name in names if _is_ignored(name)]


def _reject_unsafe_source(root: Path, label: str) -> None:
    try:
        root_stat = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise IntegrationError(
            "source_unavailable", "%s is unavailable" % label
        ) from exc
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise IntegrationError("unsafe_source", "%s must be a real directory" % label)
    entries = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if not _is_ignored(name)]
        filenames = [name for name in filenames if not _is_ignored(name)]
        entries += len(dirnames) + len(filenames)
        if entries > _MAX_SOURCE_FILES * 4:
            raise IntegrationError("source_too_large", "%s is too large" % label)
        for name in dirnames + filenames:
            path = Path(directory) / name
            if os.path.islink(path):
                raise IntegrationError(
                    "unsafe_source", "%s contains a symbolic link" % label
                )
            details = os.stat(path, follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                continue
            if not stat.S_ISREG(details.st_mode):
                raise IntegrationError(
                    "unsafe_source", "%s contains a non-regular file" % label
                )


def _tree_hashes(root: Path, label: str) -> Tuple[Dict[str, str], int, int]:
    hashes: Dict[str, str] = {}
    total = 0
    count = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not _is_ignored(name))
        for name in sorted(filenames):
            if _is_ignored(name):
                continue
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if os.path.islink(path):
                raise IntegrationError(
                    "unsafe_source", "%s contains a symbolic link" % label
                )
            details = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(details.st_mode):
                raise IntegrationError(
                    "unsafe_source", "%s contains a non-regular file" % label
                )
            if details.st_size > _MAX_SOURCE_BYTES:
                raise IntegrationError("source_too_large", "%s is too large" % label)
            count += 1
            if count > _MAX_SOURCE_FILES:
                raise IntegrationError("source_too_large", "%s is too large" % label)
            total += details.st_size
            if total > _MAX_SOURCE_BYTES:
                raise IntegrationError("source_too_large", "%s is too large" % label)
            hashes[relative] = _sha256_file(path, _MAX_SOURCE_BYTES)
    if not hashes:
        raise IntegrationError("source_unavailable", "%s is empty" % label)
    return hashes, total, count


def _tree_digest(hashes: Mapping[str, str]) -> str:
    return _sha256(_canonical(hashes))


def _git_identity(reference_root: Path, timeout: float) -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout, 60.0),
        )
        status = subprocess.run(
            ["git", "-C", str(reference_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout, 60.0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError(
            "source_unavailable", "cannot inspect the pinned AsNumpy revision"
        ) from exc
    revision = head.stdout.strip()
    dirty = status.stdout.strip()
    if (
        head.returncode != 0
        or status.returncode != 0
        or not revision
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise IntegrationError(
            "source_unavailable", "AsNumpy source must be a Git work tree"
        )
    if dirty:
        raise IntegrationError(
            "source_dirty", "AsNumpy source work tree must be clean"
        )
    return revision


def _overlay_bytes(
    adapter_root: Path, files: Mapping[str, str]
) -> Dict[str, bytes]:
    payloads: Dict[str, bytes] = {}
    candidate_root = adapter_root / "candidate"
    for relative in sorted(files):
        path = candidate_root / PurePosixPath(relative)
        payload = _read_bounded(
            path, _MAX_JSON_BYTES, "adapter overlay file %s" % relative
        )
        if _sha256(payload) != files[relative]:
            raise IntegrationError(
                "adapter_overlay_drift",
                "adapter overlay file differs from its report",
            )
        payloads[relative] = payload
    return payloads


def _apply_overlay(
    source_root: Path,
    payloads: Mapping[str, bytes],
) -> Dict[str, str]:
    before: Dict[str, str] = {}
    for relative, payload in payloads.items():
        path = source_root / PurePosixPath(relative)
        if not path.is_file() or path.is_symlink():
            raise IntegrationError(
                "adapter_overlay_target_missing",
                "adapter overlay target is missing from the pinned source",
            )
        previous = _read_bounded(
            path, _MAX_SOURCE_BYTES, "pin source file %s" % relative
        )
        before[relative] = _sha256(previous)
        path.write_bytes(payload)
        os.chmod(path, 0o600)
    return before


def _source_snapshot(
    reference_root: Path,
    target: Path,
    timeout: float,
    *,
    binding: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, str], int]:
    try:
        revision = _git_identity(reference_root, timeout)
    except IntegrationError as exc:
        if exc.code not in {"source_dirty", "source_unavailable"}:
            raise
        # Author-maintained source bundles can deliberately include the matmul
        # patch without a new upstream Git commit. Copy only files whose full
        # contents match the verified component manifest, never a dirty tree.
        from numpilot.gaps.bundled_source import snapshot_bundled_source

        try:
            return snapshot_bundled_source(reference_root, target, binding=binding)
        except (OSError, ValueError) as bundle_error:
            raise IntegrationError("source_binding_mismatch", str(bundle_error)) from bundle_error
    _reject_unsafe_source(reference_root, "pinned AsNumpy source")
    reference_hashes, total, count = _tree_hashes(
        reference_root, "pinned AsNumpy source"
    )
    if target.exists():
        raise IntegrationError("output_exists", "integration staging already exists")
    shutil.copytree(
        reference_root,
        target,
        symlinks=False,
        ignore=_copy_ignore,
    )
    copied_hashes, copied_total, copied_count = _tree_hashes(
        target, "isolated AsNumpy source"
    )
    if (
        copied_hashes != reference_hashes
        or copied_total != total
        or copied_count != count
    ):
        raise IntegrationError(
            "source_copy_drift", "isolated source copy differs from the pinned tree"
        )
    if binding is not None:
        binding.update({"kind": "clean_git", "revision": revision})
    return revision, reference_hashes, total


def _build_environment() -> Dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _OMITTED_PYTHON_ENVIRONMENT
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["LC_ALL"] = environment.get("LC_ALL") or "C.UTF-8"
    return environment


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    log_path: Path,
    label: str,
) -> Tuple[int, str]:
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            try:
                process = subprocess.run(
                    list(command),
                    cwd=str(cwd),
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = -int(signal.SIGKILL)
            except OSError as exc:
                raise IntegrationError(
                    "subprocess_failed", "%s could not be started" % label
                ) from exc
            handle.flush()
            os.fsync(handle)
    finally:
        os.close(descriptor)
    if log_path.stat().st_size > _MAX_LOG_BYTES:
        payload = _read_bounded(log_path, _MAX_LOG_BYTES, "%s log" % label)
        _write_private(log_path, payload)
    return returncode, _sha256_file(log_path)


def _python_identity(python: Path) -> str:
    try:
        details = os.stat(python, follow_symlinks=True)
    except OSError as exc:
        raise IntegrationError(
            "invalid_python", "integration Python is unavailable"
        ) from exc
    if not stat.S_ISREG(details.st_mode):
        raise IntegrationError(
            "invalid_python", "integration Python must be a regular file"
        )
    return _sha256_file(python, _MAX_PYTHON_BYTES)


def _site_paths(python: Path, timeout: float) -> Tuple[str, ...]:
    command = [
        str(python),
        "-c",
        (
            "import json,sysconfig;"
            "print(json.dumps([sysconfig.get_paths()['purelib'],"
            "sysconfig.get_paths()['platlib']]))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout, 60.0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError(
            "invalid_python", "cannot inspect the integration interpreter"
        ) from exc
    if completed.returncode != 0:
        raise IntegrationError(
            "invalid_python", "cannot inspect the integration interpreter"
        )
    try:
        paths = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IntegrationError(
            "invalid_python", "integration interpreter returned invalid paths"
        ) from exc
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(item, str) or not os.path.isabs(item) for item in paths)
    ):
        raise IntegrationError(
            "invalid_python", "integration interpreter returned invalid paths"
        )
    result = []
    for item in paths:
        if item not in result and os.path.isdir(item):
            result.append(item)
    if not result:
        raise IntegrationError(
            "invalid_python", "integration interpreter has no site-packages"
        )
    return tuple(result)


def _runtime_environment(site: Path, site_paths: Sequence[str]) -> Dict[str, str]:
    environment = _build_environment()
    entries = [str(site)]
    for item in site_paths:
        if item not in entries:
            entries.append(item)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _extract_wheel(wheel: Path, site: Path) -> Tuple[str, str]:
    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as exc:
        raise IntegrationError(
            "invalid_wheel", "built wheel is not a ZIP archive"
        ) from exc
    total = 0
    with archive:
        names = set()
        core_names = []
        package_init = None
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or "." in path.parts
                or name in names
            ):
                raise IntegrationError(
                    "invalid_wheel", "built wheel contains an unsafe path"
                )
            names.add(name)
            if name.endswith("/"):
                continue
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise IntegrationError(
                    "invalid_wheel", "built wheel contains a symbolic link"
                )
            try:
                payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise IntegrationError(
                    "invalid_wheel", "built wheel entry is unreadable"
                ) from exc
            total += len(payload)
            if (
                len(payload) > _MAX_WHEEL_ENTRY_BYTES
                or total > _MAX_WHEEL_BYTES
            ):
                raise IntegrationError(
                    "invalid_wheel", "built wheel entry is too large"
                )
            destination = site / path
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_private(destination, payload)
            if (
                len(path.parts) == 2
                and path.parts[0] == "asnumpy"
                and path.parts[1].startswith("_core")
                and path.parts[1].endswith(".so")
            ):
                core_names.append(name)
            if name == "asnumpy/__init__.py":
                package_init = name
        if not core_names or len(core_names) != 1 or package_init is None:
            raise IntegrationError(
                "invalid_wheel", "built wheel lacks the AsNumpy runtime surface"
            )
        core_relative = core_names[0]
        core_sha256 = _sha256((site / core_relative).read_bytes())
        return core_relative, core_sha256


_GATE_PROFILES = {
    "numpy.concatenate": {
        "api": "numpy.concatenate",
        "minimum_oracle_comparisons": 5,
        "minimum_cases": 10,
        "candidate_tests": "tests/asnumpy_tests/creation_tests/test_basic.py",
    },
    "numpy.cbrt": {
        "api": "numpy.cbrt",
        "minimum_oracle_comparisons": 10,
        "minimum_cases": 10,
        "candidate_tests": (
            "tests/asnumpy_tests/math_tests/test_exponents_and_logarithms.py"
        ),
    },
}


def _gate_profile_for_api(qualified_api: object) -> Dict[str, Any]:
    """Return the deterministic gate profile bound to one gap API."""

    if not isinstance(qualified_api, str) or qualified_api not in _GATE_PROFILES:
        raise IntegrationError(
            "unsupported_gate_api",
            "no deterministic integration gate profile is registered for %r"
            % (qualified_api,),
        )
    return dict(_GATE_PROFILES[qualified_api])


def _gate_failure_detail(result: Mapping[str, Any]) -> str:
    cases = result.get("cases")
    if not isinstance(cases, list):
        return "gate result has no case list"
    failed = [
        item
        for item in cases
        if isinstance(item, dict) and item.get("status") != "passed"
    ]
    if not failed:
        error = result.get("error")
        if isinstance(error, dict):
            return "%s: %s" % (
                str(error.get("type", "GateError"))[:64],
                str(error.get("message", ""))[:_MAX_DETAIL_CHARS],
            )
        return "gate did not report a passing result"
    return "; ".join(
        "%s: %s"
        % (
            str(item.get("name", "case"))[:128],
            str(item.get("detail", ""))[:_MAX_CASE_DETAIL_CHARS],
        )
        for item in failed[:8]
    )


def _sanitize(text: object, replacements: Sequence[Tuple[str, str]]) -> str:
    value = str(text)
    for needle, replacement in replacements:
        if needle:
            value = value.replace(needle, replacement)
    return value[:_MAX_DETAIL_CHARS]


def _sanitized_tail(path: Path, replacements: Sequence[Tuple[str, str]]) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _MAX_DIAGNOSTIC_TAIL))
            payload = handle.read(_MAX_DIAGNOSTIC_TAIL)
    except OSError:
        return ""
    return _sanitize(payload.decode("utf-8", "replace"), replacements)


def _normalize_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrationError("invalid_timeout", "integration timeout is invalid")
    timeout = float(value)
    if not (0 < timeout <= _MAX_TIMEOUT_SECONDS):
        raise IntegrationError("invalid_timeout", "integration timeout is invalid")
    return timeout


def _failure_ref_name(attempt: int) -> str:
    return "%s%d" % (INTEGRATION_FAILURE_REF_PREFIX, attempt)


def _failure_refs(gap: GapRecoveryState) -> Tuple[EvidenceRef, ...]:
    return tuple(
        ref
        for ref in gap.refs
        if ref.name.startswith(INTEGRATION_FAILURE_REF_PREFIX)
    )


def _stored_failure_ref(
    gap: GapRecoveryState, locator: str, sha256: str
) -> EvidenceRef:
    matches = tuple(
        ref
        for ref in _failure_refs(gap)
        if ref.locator == locator and ref.sha256 == sha256
    )
    if len(matches) != 1:
        raise IntegrationError(
            "failure_record_missing",
            "integration failure record is not bound to exactly one gap reference",
        )
    return matches[0]


def _existing_success(
    status: RecoveryStatus, gap_id: str, ref: EvidenceRef
) -> bool:
    receipts = tuple(
        item
        for item in status.state.stage_receipts
        if item.phase is RecoveryPhase.INTEGRATED_CANDIDATE
        and item.kind is StageReceiptKind.INTEGRATION_TRANSACTION
        and item.scope is StageReceiptScope.GAPS
        and item.covered_gap_ids == (gap_id,)
        and item.verdict is StageReceiptVerdict.SATISFIED
        and item.ref == ref
    )
    return len(receipts) == 1


def _side_effects(transaction: Mapping[str, Any]) -> SideEffectTruth:
    effects = transaction.get("side_effects")
    if not isinstance(effects, Mapping):
        return SideEffectTruth()
    return SideEffectTruth(
        subprocess_performed=bool(effects.get("subprocess_performed")),
        network_access_performed=bool(effects.get("network_access_performed")),
        system_write_performed=bool(effects.get("system_write_performed")),
        model_invocation_performed=bool(effects.get("model_invocation_performed")),
        asnumpy_import_performed=bool(effects.get("asnumpy_import_performed")),
        build_performed=bool(effects.get("build_performed")),
        execution_performed=bool(effects.get("execution_performed")),
        verification_performed=bool(effects.get("verification_performed")),
        activation_performed=bool(effects.get("activation_performed")),
    )


def _transaction_sha256(transaction: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in transaction.items()
        if key != "transaction_sha256"
    }
    return _sha256(_canonical(body))


def _load_transaction(out_root: Path) -> Tuple[Mapping[str, Any], Path, str]:
    path = out_root / INTEGRATION_RECEIPT_FILENAME
    transaction = _read_private_json(path, "integration transaction")
    if (
        transaction.get("schema_version") != INTEGRATION_SCHEMA_VERSION
        or transaction.get("record_type") != "numpilot_integration_transaction"
        or transaction.get("status") not in {"passed", "failed"}
        or transaction.get("transaction_sha256") != _transaction_sha256(transaction)
    ):
        raise IntegrationError(
            "transaction_invalid", "integration transaction is not self-consistent"
        )
    if transaction.get("status") == "passed":
        if transaction.get("failure") is not None:
            raise IntegrationError(
                "transaction_invalid", "passed transaction carries a failure"
            )
    elif not isinstance(transaction.get("failure"), Mapping):
        raise IntegrationError(
            "transaction_invalid", "failed transaction has no failure record"
        )
    return transaction, path, _sha256_file(path)


def _verify_transaction_artifacts(
    out_root: Path, transaction: Mapping[str, Any]
) -> None:
    def check(relative: object, digest: object, label: str) -> None:
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise IntegrationError(
                "transaction_invalid",
                "integration transaction %s is invalid" % label,
            )
        path = out_root / PurePosixPath(relative)
        if path.is_symlink() or not path.is_file():
            raise IntegrationError(
                "transaction_invalid",
                "integration transaction %s is missing" % label,
            )
        if _sha256_file(path) != digest:
            raise IntegrationError(
                "transaction_invalid",
                "integration transaction %s drifted" % label,
            )

    build = transaction.get("build")
    if isinstance(build, Mapping) and build.get("status") == "passed":
        check(build.get("wheel_relative"), build.get("wheel_sha256"), "wheel")
        check(build.get("log_relative"), build.get("log_sha256"), "build log")
    gate = transaction.get("gate")
    if isinstance(gate, Mapping):
        check(gate.get("result_relative"), gate.get("result_sha256"), "gate result")
        check(gate.get("log_relative"), gate.get("log_sha256"), "gate log")
        check(gate.get("script_relative"), gate.get("script_sha256"), "gate script")
    tests = transaction.get("candidate_tests")
    if isinstance(tests, Mapping):
        check(tests.get("log_relative"), tests.get("log_sha256"), "test log")
    failure = transaction.get("failure")
    if isinstance(failure, Mapping):
        check(
            failure.get("diagnostic_relative"),
            failure.get("diagnostic_sha256"),
            "failure diagnostic",
        )


def _append_success(
    store: RecoveryStore,
    transaction: Mapping[str, Any],
    transaction_ref: EvidenceRef,
    adapter_ref: EvidenceRef,
    gap_id: str,
) -> Tuple[RecoveryStatus, int]:
    for _attempt in range(MAX_APPEND_ATTEMPTS):
        try:
            current = store.status()
        except RecoveryError as exc:
            raise IntegrationError(
                "state_unavailable", "cannot reload the complete recovery chain"
            ) from exc
        if _existing_success(current, gap_id, transaction_ref):
            return current, 0
        gap = _gap_for_candidate(current, gap_id)
        if (
            gap.phase is not RecoveryPhase.ADAPTER_CANDIDATE
            or gap.outcome is not RecoveryOutcome.READY
        ):
            raise IntegrationError(
                "state_phase_mismatch",
                "gap is not ready for its integration transaction",
            )
        if _latest_adapter_ref(gap.refs) != adapter_ref:
            raise IntegrationError(
                "adapter_state_drift",
                "integration transaction does not target the current adapter",
            )
        next_gaps = tuple(
            GapRecoveryState(
                gap_id=item.gap_id,
                phase=(
                    RecoveryPhase.INTEGRATED_CANDIDATE
                    if item.gap_id == gap_id
                    else item.phase
                ),
                outcome=item.outcome,
                attempt_count=item.attempt_count,
                refs=(
                    tuple(item.refs) + (transaction_ref,)
                    if item.gap_id == gap_id
                    else item.refs
                ),
                reason=item.reason,
            )
            for item in current.state.gaps
        )
        global_phase = min(
            next_gaps, key=lambda item: _PHASE_INDEX[item.phase]
        ).phase
        receipt = create_stage_receipt(
            phase=RecoveryPhase.INTEGRATED_CANDIDATE,
            kind=StageReceiptKind.INTEGRATION_TRANSACTION,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(gap_id,),
            binding=current.state.binding,
            parent_state_sha256=current.state_sha256,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=transaction_ref,
        )
        event = RecoveryEvent.bound_to(
            current.state,
            phase=global_phase,
            outcome=RecoveryOutcome.READY,
            gaps=next_gaps,
            stage_receipts=(receipt,),
            side_effects=_side_effects(transaction),
            reason=(
                "isolated AsNumpy integration built, loaded, and passed the "
                "deterministic NumPy-oracle API gate; qualification and "
                "activation remain pending"
            ),
        )
        try:
            appended = store.append(event)
        except RecoveryError:
            continue
        return appended, 1
    raise IntegrationError(
        "state_append_failed",
        "recovery state changed too often to append integration",
    )


def _append_failure(
    store: RecoveryStore,
    transaction: Mapping[str, Any],
    diagnostic_locator: str,
    diagnostic_sha256: str,
    adapter_ref: EvidenceRef,
    gap_id: str,
) -> Tuple[RecoveryStatus, int, bool]:
    for _attempt in range(MAX_APPEND_ATTEMPTS):
        try:
            current = store.status()
        except RecoveryError as exc:
            raise IntegrationError(
                "state_unavailable", "cannot reload the complete recovery chain"
            ) from exc
        gap = _gap_for_candidate(current, gap_id)
        existing = tuple(
            ref
            for ref in _failure_refs(gap)
            if ref.locator == diagnostic_locator
            and ref.sha256 == diagnostic_sha256
        )
        if existing:
            if len(existing) != 1:
                raise IntegrationError(
                    "failure_record_ambiguous",
                    "integration failure record is ambiguous",
                )
            return current, 0, False
        if (
            gap.phase is not RecoveryPhase.ADAPTER_CANDIDATE
            or gap.outcome is not RecoveryOutcome.READY
        ):
            raise IntegrationError(
                "state_phase_mismatch",
                "gap is not ready for its integration failure record",
            )
        if _latest_adapter_ref(gap.refs) != adapter_ref:
            raise IntegrationError(
                "adapter_state_drift",
                "integration failure does not target the current adapter",
            )
        attempt = gap.attempt_count + 1
        ref = EvidenceRef(
            name=_failure_ref_name(attempt),
            locator=diagnostic_locator,
            sha256=diagnostic_sha256,
        )
        failure = transaction.get("failure")
        summary = failure.get("summary") if isinstance(failure, Mapping) else ""
        next_gaps = tuple(
            GapRecoveryState(
                gap_id=item.gap_id,
                phase=item.phase,
                outcome=(
                    RecoveryOutcome.FAILED_ATTEMPT
                    if item.gap_id == gap_id
                    else item.outcome
                ),
                attempt_count=(
                    item.attempt_count + 1
                    if item.gap_id == gap_id
                    else item.attempt_count
                ),
                refs=(
                    tuple(item.refs) + (ref,)
                    if item.gap_id == gap_id
                    else item.refs
                ),
                reason=(
                    str(summary)[:2048]
                    if item.gap_id == gap_id
                    else item.reason
                ),
            )
            for item in current.state.gaps
        )
        receipt = create_stage_receipt(
            phase=RecoveryPhase.INTEGRATED_CANDIDATE,
            kind=StageReceiptKind.INTEGRATION_TRANSACTION,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(gap_id,),
            binding=current.state.binding,
            parent_state_sha256=current.state_sha256,
            verdict=StageReceiptVerdict.FAILED,
            ref=ref,
        )
        event = RecoveryEvent.bound_to(
            current.state,
            phase=current.state.phase,
            outcome=RecoveryOutcome.FAILED_ATTEMPT,
            gaps=next_gaps,
            stage_receipts=(receipt,),
            side_effects=_side_effects(transaction),
            reason=(
                "isolated integration attempt failed; candidate artifacts were "
                "preserved and a bounded diagnostic was published"
            ),
        )
        try:
            appended = store.append(event)
        except RecoveryError:
            continue
        return appended, 1, True
    raise IntegrationError(
        "state_append_failed",
        "recovery state changed too often to append failure",
    )


def _append_blocked_if_exhausted(
    store: RecoveryStore, gap_id: str
) -> Tuple[RecoveryStatus, int]:
    for _attempt in range(MAX_APPEND_ATTEMPTS):
        try:
            current = store.status()
        except RecoveryError as exc:
            raise IntegrationError(
                "state_unavailable", "cannot reload the complete recovery chain"
            ) from exc
        gap = _gap_for_candidate(current, gap_id)
        if gap.outcome is RecoveryOutcome.BLOCKED:
            return current, 0
        if (
            gap.outcome is not RecoveryOutcome.FAILED_ATTEMPT
            or _repair_count(gap.refs) < _MAX_REPAIR_ROUNDS
        ):
            return current, 0
        next_gaps = tuple(
            GapRecoveryState(
                gap_id=item.gap_id,
                phase=item.phase,
                outcome=(
                    RecoveryOutcome.BLOCKED
                    if item.gap_id == gap_id
                    else item.outcome
                ),
                attempt_count=item.attempt_count,
                refs=item.refs,
                reason=(
                    "automatic repair budget exhausted after %d rounds; "
                    "MANUAL_REVIEW_REQUIRED" % _MAX_REPAIR_ROUNDS
                    if item.gap_id == gap_id
                    else item.reason
                ),
            )
            for item in current.state.gaps
        )
        event = RecoveryEvent.bound_to(
            current.state,
            phase=current.state.phase,
            outcome=RecoveryOutcome.BLOCKED,
            gaps=next_gaps,
            reason="integration repair budget exhausted; MANUAL_REVIEW_REQUIRED",
        )
        try:
            appended = store.append(event)
        except RecoveryError:
            continue
        return appended, 1
    raise IntegrationError(
        "state_append_failed",
        "recovery state changed too often to append block",
    )


def preflight_official_adapter_integration(
    state_dir: os.PathLike[str] | str,
    candidate_path: os.PathLike[str] | str,
    adapter_output_root: os.PathLike[str] | str,
    asnumpy_reference: os.PathLike[str] | str,
    *,
    python_executable: os.PathLike[str] | str | None = None,
) -> IntegrationPreflightResult:
    """Validate integration admission without writes, subprocesses, or imports."""

    python = _absolute_path(
        sys.executable if python_executable is None else python_executable,
        "integration Python",
    )
    if not python.is_file():
        raise IntegrationError(
            "invalid_python", "integration Python must be a regular file"
        )
    python_sha256 = _python_identity(python)
    reference_root = _absolute_path(asnumpy_reference, "AsNumpy reference root")
    if reference_root.is_symlink() or not reference_root.is_dir():
        raise IntegrationError(
            "invalid_reference", "AsNumpy reference root must be a real directory"
        )

    try:
        status = RecoveryStore(state_dir).status()
    except RecoveryError as exc:
        raise IntegrationError(
            "state_unavailable", "cannot verify the complete recovery chain"
        ) from exc
    candidate_file = _absolute_path(candidate_path, "operator candidate")
    adapter_root = _absolute_path(adapter_output_root, "adapter output root")
    candidate, gap, candidate_before = _candidate_gap(status, candidate_file)
    if gap.phase is not RecoveryPhase.ADAPTER_CANDIDATE or gap.outcome is not RecoveryOutcome.READY:
        raise IntegrationError(
            "state_phase_mismatch",
            "integration preflight requires a ready ADAPTER_CANDIDATE gap",
        )
    adapter_ref = _latest_adapter_ref(gap.refs)
    if adapter_ref.locator != str(adapter_root / "asnumpy-adapter-report.json"):
        raise IntegrationError(
            "adapter_binding_mismatch",
            "adapter output is not the current recovery adapter",
        )
    expected_status = (
        _ADAPTER_STATUS_GENERATED
        if adapter_ref.name == _ADAPTER_REF_NAME
        else _ADAPTER_STATUS_REPAIRED
    )
    if not (
        adapter_ref.name == _ADAPTER_REF_NAME
        or adapter_ref.name.startswith(_ADAPTER_REPAIR_REF_PREFIX)
    ):
        raise IntegrationError(
            "adapter_binding_mismatch", "adapter reference name is not recognized"
        )
    prepared = _official_adapter_input(status, candidate)
    try:
        adapter = _load_adapter_output(
            adapter_root,
            candidate,
            prepared,
            accepted_statuses=(expected_status,),
            ref_name=adapter_ref.name,
            accepted_generation_modes=("remote_llm", "injected_test_double"),
        )
    except AdapterRecoveryError as exc:
        raise IntegrationError(exc.code, str(exc)) from exc
    if adapter.report_ref != adapter_ref:
        raise IntegrationError(
            "adapter_binding_mismatch",
            "adapter report differs from the current recovery adapter",
        )
    context_plan = adapter.report.get("asnumpy_adapter_context_plan")
    if not isinstance(context_plan, Mapping) or context_plan.get(
        "reference_root"
    ) != str(reference_root):
        raise IntegrationError(
            "reference_mismatch",
            "integration reference root differs from the adapter context plan",
        )
    candidate_after = _snapshot_regular_file(
        candidate_file, _MAX_JSON_BYTES, "operator candidate"
    )
    if candidate_after != candidate_before:
        raise IntegrationError(
            "candidate_drift",
            "operator candidate changed during integration preflight",
        )

    policy = status.state.policy.to_dict()
    missing = tuple(
        name for name in _REQUIRED_INTEGRATION_POLICY if not policy.get(name, False)
    )
    blocked = bool(missing)
    return IntegrationPreflightResult(
        status="blocked" if blocked else "ready",
        code="authorization_missing" if blocked else "ready_for_integration",
        next_action=(
            "authorize_full_local_recovery" if blocked else "integrate_candidate"
        ),
        gap_id=candidate.gap_id,
        candidate_id=candidate.candidate_id,
        candidate_sha256=candidate_before.sha256,
        adapter_artifact_id=str(adapter.report["adapter_artifact_id"]),
        adapter_report_sha256=adapter_ref.sha256,
        generation_mode=str(adapter.report.get("generation_mode", "remote_llm")),
        provider=str(adapter.report["provider"]),
        model=str(adapter.report["model"]),
        configuration_source=str(adapter.report["configuration_source"]),
        reference_root=str(reference_root),
        context_plan_sha256=str(context_plan["plan_sha256"]),
        python_path=str(python),
        python_sha256=python_sha256,
        policy=policy,
        required_policy=_REQUIRED_INTEGRATION_POLICY,
        missing_policy=missing,
    )


def run_official_adapter_integration(
    state_dir: os.PathLike[str] | str,
    candidate_path: os.PathLike[str] | str,
    adapter_output_root: os.PathLike[str] | str,
    asnumpy_reference: os.PathLike[str] | str,
    out_root: os.PathLike[str] | str,
    *,
    python_executable: os.PathLike[str] | str | None = None,
    timeout_seconds: object = _DEFAULT_TIMEOUT_SECONDS,
) -> IntegrationTransactionResult:
    """Perform or replay one isolated integration transaction."""

    timeout = _normalize_timeout(timeout_seconds)
    python = _absolute_path(
        sys.executable if python_executable is None else python_executable,
        "integration Python",
    )
    if not python.is_file():
        raise IntegrationError(
            "invalid_python", "integration Python must be a regular file"
        )
    python_sha256 = _python_identity(python)
    reference_root = _absolute_path(asnumpy_reference, "AsNumpy reference root")
    out = _absolute_path(out_root, "integration output root")
    if out.parent.is_symlink() or not out.parent.is_dir():
        raise IntegrationError(
            "unsafe_output", "integration output parent must be a real directory"
        )

    store = RecoveryStore(state_dir)
    try:
        status = store.status()
    except RecoveryError as exc:
        raise IntegrationError(
            "state_unavailable", "cannot verify the complete recovery chain"
        ) from exc
    policy = status.state.policy
    for name in (
        "subprocess_allowed",
        "system_write_allowed",
        "asnumpy_import_allowed",
        "build_allowed",
        "execution_allowed",
    ):
        if not getattr(policy, name):
            raise IntegrationError(
                "authorization_missing",
                "frozen recovery policy does not allow %s" % name,
            )
    if not policy.activation_allowed:
        raise IntegrationError(
            "authorization_missing",
            "full-local-recovery authorization is required for integration",
        )

    candidate_file = _absolute_path(candidate_path, "operator candidate")
    adapter_root = _absolute_path(adapter_output_root, "adapter output root")
    candidate, gap, candidate_before = _candidate_gap(status, candidate_file)
    adapter_ref = _latest_adapter_ref(gap.refs)
    if adapter_ref.locator != str(adapter_root / "asnumpy-adapter-report.json"):
        raise IntegrationError(
            "adapter_binding_mismatch",
            "adapter output is not the current recovery adapter",
        )
    expected_status = (
        _ADAPTER_STATUS_GENERATED
        if adapter_ref.name == _ADAPTER_REF_NAME
        else _ADAPTER_STATUS_REPAIRED
    )
    if not (
        adapter_ref.name == _ADAPTER_REF_NAME
        or adapter_ref.name.startswith(_ADAPTER_REPAIR_REF_PREFIX)
    ):
        raise IntegrationError(
            "adapter_binding_mismatch", "adapter reference name is not recognized"
        )
    prepared = _official_adapter_input(status, candidate)
    try:
        adapter = _load_adapter_output(
            adapter_root,
            candidate,
            prepared,
            accepted_statuses=(expected_status,),
            ref_name=adapter_ref.name,
            accepted_generation_modes=("remote_llm", "injected_test_double"),
        )
    except AdapterRecoveryError as exc:
        raise IntegrationError(exc.code, str(exc)) from exc
    if adapter.report_ref != adapter_ref:
        raise IntegrationError(
            "adapter_binding_mismatch",
            "adapter report differs from the current recovery adapter",
        )
    context_plan = adapter.report.get("asnumpy_adapter_context_plan")
    if not isinstance(context_plan, Mapping) or context_plan.get(
        "reference_root"
    ) != str(reference_root):
        raise IntegrationError(
            "reference_mismatch",
            "integration reference root differs from the adapter context plan",
        )
    if adapter.report.get("status") == _ADAPTER_STATUS_REPAIRED and (
        adapter.report.get("parent_adapter_artifact_id")
        == adapter.report.get("adapter_artifact_id")
        or adapter.report.get("source_artifact_id") != candidate.candidate_id
    ):
        raise IntegrationError(
            "adapter_binding_mismatch",
            "repaired adapter does not derive from the bound candidate",
        )
    candidate_after = _snapshot_regular_file(
        candidate_file, _MAX_JSON_BYTES, "operator candidate"
    )
    if candidate_after != candidate_before:
        raise IntegrationError(
            "candidate_drift",
            "operator candidate changed during integration validation",
        )

    if out.exists():
        transaction, transaction_path, transaction_payload_sha = _load_transaction(out)
        _verify_transaction_artifacts(out, transaction)
        if transaction.get("gap_id") != candidate.gap_id:
            raise IntegrationError(
                "transaction_invalid", "integration transaction binds another gap"
            )
        if (
            transaction.get("candidate_id") != candidate.candidate_id
            or transaction.get("adapter_report_sha256") != adapter_ref.sha256
            or transaction.get("adapter_artifact_id")
            != adapter.report.get("adapter_artifact_id")
        ):
            raise IntegrationError(
                "transaction_invalid", "integration transaction binds another adapter"
            )
    else:
        if (
            gap.phase is not RecoveryPhase.ADAPTER_CANDIDATE
            or gap.outcome is not RecoveryOutcome.READY
        ):
            raise IntegrationError(
                "state_phase_mismatch",
                "a new integration attempt requires a ready ADAPTER_CANDIDATE gap",
            )
        transaction, transaction_path = _produce_transaction(
            out,
            reference_root,
            candidate,
            adapter,
            adapter_ref,
            adapter_root,
            python,
            python_sha256,
            timeout,
            gap,
        )
        transaction_payload_sha = _sha256_file(transaction_path)

    transaction_ref = EvidenceRef(
        name=INTEGRATION_REF_NAME,
        locator=str(transaction_path),
        sha256=transaction_payload_sha,
    )
    diagnostic_ref = None
    if transaction.get("status") == "passed":
        if transaction.get("transaction_sha256") != _transaction_sha256(transaction):
            raise IntegrationError(
                "transaction_invalid", "integration transaction identity drifted"
            )
        appended_status, appended = _append_success(
            store, transaction, transaction_ref, adapter_ref, candidate.gap_id
        )
    else:
        failure = transaction.get("failure")
        if not isinstance(failure, Mapping):
            raise IntegrationError(
                "transaction_invalid", "failed transaction has no failure record"
            )
        diagnostic_path = out / PurePosixPath(
            str(failure.get("diagnostic_relative"))
        )
        diagnostic_payload = _read_bounded(
            diagnostic_path, _MAX_JSON_BYTES, "integration failure diagnostic"
        )
        diagnostic_sha256 = _sha256(diagnostic_payload)
        if diagnostic_sha256 != failure.get("diagnostic_sha256"):
            raise IntegrationError(
                "transaction_invalid", "integration failure diagnostic drifted"
            )
        appended_status, appended, recorded_now = _append_failure(
            store,
            transaction,
            str(diagnostic_path),
            diagnostic_sha256,
            adapter_ref,
            candidate.gap_id,
        )
        current_gap = _gap_for_candidate(appended_status, candidate.gap_id)
        if recorded_now or current_gap.outcome is RecoveryOutcome.FAILED_ATTEMPT:
            blocked_status, blocked_appended = _append_blocked_if_exhausted(
                store, candidate.gap_id
            )
            if blocked_appended:
                appended_status = blocked_status
            appended += blocked_appended
        final_gap = _gap_for_candidate(appended_status, candidate.gap_id)
        diagnostic_ref = _stored_failure_ref(
            final_gap, str(diagnostic_path), diagnostic_sha256
        )
    final_gap = _gap_for_candidate(appended_status, candidate.gap_id)
    return _result(
        appended_status,
        transaction,
        out,
        transaction_ref,
        diagnostic_ref,
        appended,
        final_gap.outcome is RecoveryOutcome.BLOCKED,
    )


def _result(
    status: RecoveryStatus,
    transaction: Mapping[str, Any],
    out_root: Path,
    transaction_ref: EvidenceRef,
    diagnostic_ref: EvidenceRef | None,
    appended: int,
    blocked: bool,
) -> IntegrationTransactionResult:
    failure = transaction.get("failure")
    manual = blocked or bool(
        isinstance(failure, Mapping) and failure.get("manual_review_required")
    )
    return IntegrationTransactionResult(
        recovery=status,
        gap_id=str(transaction["gap_id"]),
        transaction_id=str(transaction["transaction_id"]),
        status=str(transaction["status"]),
        out_root=str(out_root),
        appended_revisions=appended,
        manual_review_required=manual,
        transaction_ref=transaction_ref,
        diagnostic_ref=diagnostic_ref,
    )


def _produce_transaction(
    out: Path,
    reference_root: Path,
    candidate,
    adapter,
    adapter_ref: EvidenceRef,
    adapter_root: Path,
    python: Path,
    python_sha256: str,
    timeout: float,
    gap: GapRecoveryState,
) -> Tuple[Mapping[str, Any], Path]:
    stage = out.with_name(".%s-%s.stage" % (out.name, secrets.token_hex(8)))
    try:
        stage.mkdir(mode=0o700)
        source_root = stage / "source"
        source_binding: Dict[str, Any] = {}
        revision, reference_hashes, reference_total = _source_snapshot(
            reference_root, source_root, timeout, binding=source_binding
        )
        overlay = _overlay_bytes(adapter_root, adapter.report["files_sha256"])
        overlay_before = _apply_overlay(source_root, overlay)
        applied, applied_total, applied_count = _tree_hashes(
            source_root, "isolated AsNumpy source"
        )
        for relative, digest in adapter.report["files_sha256"].items():
            if applied.get(relative) != digest:
                raise IntegrationError(
                    "adapter_overlay_drift",
                    "isolated overlay differs from the adapter report",
                )

        replacements = (
            (str(stage), "<transaction>"),
            (str(reference_root), "<asnumpy-reference>"),
            (str(adapter_root), "<adapter>"),
            (os.path.expanduser("~"), "<home>"),
        )
        environment = _build_environment()
        environment["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASNUMPY"] = (
            "0.4.1.post0+numpilot.%s" % adapter_ref.sha256[:8]
        )
        dist = stage / "dist"
        dist.mkdir(mode=0o700)
        build_returncode, build_log_sha = _run_logged(
            [
                str(python),
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(dist),
            ],
            cwd=source_root,
            environment=environment,
            timeout=timeout,
            log_path=stage / "logs/build.log",
            label="AsNumpy wheel build",
        )
        wheels = tuple(
            sorted(
                path
                for path in dist.iterdir()
                if path.is_file() and path.suffix == ".whl"
            )
        )
        build: Dict[str, Any] = {
            "status": "failed",
            "returncode": build_returncode,
            "log_relative": "logs/build.log",
            "log_sha256": build_log_sha,
            "wheel_relative": None,
            "wheel_sha256": None,
            "wheel_filename": None,
        }
        wheel = wheels[0] if len(wheels) == 1 else None
        if build_returncode != 0 or wheel is None:
            failure_phase = "build"
            failure_detail = (
                "AsNumpy wheel build failed with exit code %d" % build_returncode
                if build_returncode != 0
                else "AsNumpy wheel build did not produce exactly one wheel"
            )
        else:
            build.update({
                "status": "passed",
                "wheel_relative": "dist/%s" % wheel.name,
                "wheel_sha256": _sha256_file(wheel),
                "wheel_filename": wheel.name,
            })
            failure_phase = ""
            failure_detail = ""

        site = stage / "runtime/site"
        site.mkdir(mode=0o700, parents=True)
        core_relative = ""
        core_sha256 = ""
        package_version = ""
        if wheel is not None:
            core_relative, core_sha256 = _extract_wheel(wheel, site)
            dist_infos = tuple(site.glob("asnumpy-*.dist-info"))
            if len(dist_infos) != 1:
                raise IntegrationError(
                    "invalid_wheel",
                    "built wheel has no unique distribution metadata",
                )
            package_version = dist_infos[0].name

        gate_profile = _gate_profile_for_api(candidate.qualified_api)
        gate_source = Path(__file__).with_name("integration_gate.py")
        gate_payload = _read_bounded(
            gate_source, _MAX_JSON_BYTES, "integration gate script"
        )
        if not gate_payload.startswith(b"#!/usr/bin/env python3"):
            raise IntegrationError(
                "invalid_gate_script", "integration gate script is invalid"
            )
        gate_dir = stage / "gate"
        gate_dir.mkdir(mode=0o700)
        gate_path = gate_dir / "gate.py"
        _write_private(gate_path, gate_payload)
        gate_sha = _sha256(gate_payload)
        gate_result_path = gate_dir / "result.json"
        gate_log_path = stage / "logs/gate.log"
        site_paths = _site_paths(python, timeout)
        runtime_environment = _runtime_environment(site, site_paths)
        gate_returncode = -1
        gate_result: Mapping[str, Any] = {}
        gate: Dict[str, Any] = {
            "status": "failed",
            "returncode": gate_returncode,
            "result_relative": "gate/result.json",
            "result_sha256": "",
            "log_relative": "logs/gate.log",
            "log_sha256": "",
            "script_relative": "gate/gate.py",
            "script_sha256": gate_sha,
            "oracle_comparisons": 0,
            "cases": 0,
        }
        tests: Dict[str, Any] = {
            "status": "not_run",
            "returncode": None,
            "log_relative": "logs/candidate-tests.log",
            "log_sha256": "",
        }
        gate_ran = False
        if wheel is not None and not failure_phase:
            gate_ran = True
            gate_returncode, gate_log_sha = _run_logged(
                [
                    str(python),
                    "-S",
                    str(gate_path),
                    "--result",
                    str(gate_result_path),
                    "--expected-root",
                    str(site),
                    "--api",
                    gate_profile["api"],
                ],
                cwd=gate_dir,
                environment=runtime_environment,
                timeout=timeout,
                log_path=gate_log_path,
                label="AsNumpy API gate",
            )
            gate["returncode"] = gate_returncode
            gate["log_sha256"] = gate_log_sha
            if gate_result_path.is_file():
                gate_result = _read_private_json(
                    gate_result_path, "integration gate result"
                )
                gate["result_sha256"] = _sha256_file(gate_result_path)
            cases = gate_result.get("cases")
            gate["cases"] = len(cases) if isinstance(cases, list) else 0
            comparisons = gate_result.get("oracle_comparisons")
            gate["oracle_comparisons"] = (
                comparisons
                if type(comparisons) is int and comparisons >= 0
                else 0
            )
            gate_passed = (
                gate_returncode == 0
                and gate_result.get("status") == "passed"
                and gate_result.get("api") == gate_profile["api"]
                and gate_result.get("asnumpy_file_under_expected_root") is True
                and gate["oracle_comparisons"]
                >= gate_profile["minimum_oracle_comparisons"]
                and gate["cases"] >= gate_profile["minimum_cases"]
            )
            gate["status"] = "passed" if gate_passed else "failed"
            if not gate_passed:
                failure_phase = "gate"
                failure_detail = _sanitize(
                    "deterministic API gate failed: %s"
                    % _gate_failure_detail(gate_result),
                    replacements,
                )
            if gate_passed:
                tests_returncode, tests_log_sha = _run_logged(
                    [
                        str(python),
                        "-S",
                        "-m",
                        "pytest",
                        "-c",
                        os.devnull,
                        "-p",
                        "no:cacheprovider",
                        gate_profile["candidate_tests"],
                        "-q",
                    ],
                    cwd=source_root,
                    environment=runtime_environment,
                    timeout=timeout,
                    log_path=stage / "logs/candidate-tests.log",
                    label="candidate AsNumpy %s tests" % gate_profile["api"],
                )
                tests.update({
                    "status": "passed" if tests_returncode == 0 else "failed",
                    "returncode": tests_returncode,
                    "log_sha256": tests_log_sha,
                })
                if tests_returncode != 0:
                    failure_phase = "candidate_tests"
                    failure_detail = (
                        "candidate concatenate tests failed with exit code %d"
                        % tests_returncode
                    )

        repairs_used = _repair_count(gap.refs)
        side_effects = {
            "subprocess_performed": True,
            "network_access_performed": False,
            "system_write_performed": True,
            "model_invocation_performed": False,
            "asnumpy_import_performed": bool(gate_result.get("asnumpy_file")),
            "build_performed": True,
            "execution_performed": gate_ran,
            "verification_performed": False,
            "activation_performed": False,
        }
        status_value = "passed" if not failure_phase else "failed"
        identity = {
            "gap_id": candidate.gap_id,
            "candidate_id": candidate.candidate_id,
            "adapter_artifact_id": adapter.report.get("adapter_artifact_id"),
            "adapter_report_sha256": adapter_ref.sha256,
            "source_revision": revision,
            "source_binding": source_binding,
            "source_tree_sha256": _tree_digest(reference_hashes),
            "overlay_sha256": dict(adapter.report["files_sha256"]),
            "wheel_sha256": build.get("wheel_sha256"),
            "gate_result_sha256": gate.get("result_sha256") or None,
            "status": status_value,
        }
        transaction_id = "int-%s" % _sha256(_canonical(identity))[:24]

        failed_cases = []
        raw_cases = gate_result.get("cases")
        if isinstance(raw_cases, list):
            for item in raw_cases:
                if isinstance(item, dict) and item.get("status") != "passed":
                    failed_cases.append({
                        "name": _sanitize(item.get("name", "case"), replacements)[
                            :128
                        ],
                        "detail": _sanitize(
                            item.get("detail", ""), replacements
                        )[:_MAX_CASE_DETAIL_CHARS],
                    })
                    if len(failed_cases) >= 8:
                        break
        if failure_phase == "build":
            tail_source = stage / "logs/build.log"
        elif failure_phase == "candidate_tests":
            tail_source = stage / "logs/candidate-tests.log"
        else:
            tail_source = stage / "logs/gate.log"
        diagnostic: Dict[str, Any] = {
            "schema_version": INTEGRATION_SCHEMA_VERSION,
            "record_type": "numpilot_integration_failure_diagnostic",
            "transaction_id": transaction_id,
            "gap_id": candidate.gap_id,
            "candidate_id": candidate.candidate_id,
            "adapter_artifact_id": adapter.report.get("adapter_artifact_id"),
            "adapter_report_sha256": adapter_ref.sha256,
            "phase": failure_phase,
            "failure": {
                "summary": _sanitize(failure_detail, replacements),
                "build_returncode": build_returncode,
                "gate_returncode": gate.get("returncode"),
                "candidate_tests_returncode": tests.get("returncode"),
                "gate_failed_cases": failed_cases,
                "gate_completed_cases": gate_result.get("completed_cases"),
                "stderr_tail": _sanitized_tail(tail_source, replacements),
            },
            "repairs_used": repairs_used,
            "repairs_remaining": max(0, _MAX_REPAIR_ROUNDS - repairs_used),
            "manual_review_required": repairs_used >= _MAX_REPAIR_ROUNDS,
        }
        diagnostic_path = stage / "diagnostics" / DIAGNOSTIC_FILENAME
        diagnostic_payload = _write_private_json(diagnostic_path, diagnostic)
        diagnostic_sha = _sha256(diagnostic_payload)

        transaction: Dict[str, Any] = {
            "schema_version": INTEGRATION_SCHEMA_VERSION,
            "record_type": "numpilot_integration_transaction",
            "transaction_id": transaction_id,
            "gap_id": candidate.gap_id,
            "candidate_id": candidate.candidate_id,
            "adapter_artifact_id": adapter.report.get("adapter_artifact_id"),
            "adapter_report_sha256": adapter_ref.sha256,
            "adapter_ref": adapter_ref.to_dict(),
            "source": {
                "binding": source_binding,
                "revision": revision,
                "tree_sha256": _tree_digest(reference_hashes),
                "file_count": len(reference_hashes),
                "total_bytes": reference_total,
                "overlay_before_sha256": overlay_before,
                "overlay_sha256": dict(adapter.report["files_sha256"]),
                "applied_file_count": applied_count,
                "applied_total_bytes": applied_total,
            },
            "build": build,
            "runtime": {
                "site_relative": "runtime/site",
                "core_relative": core_relative,
                "core_sha256": core_sha256,
                "distribution": package_version,
                "python_sha256": python_sha256,
                "site_path_count": len(site_paths),
            },
            "gate": gate,
            "candidate_tests": tests,
            "status": status_value,
            "failure": None
            if status_value == "passed"
            else {
                "phase": failure_phase,
                "summary": _sanitize(failure_detail, replacements),
                "diagnostic_relative": "diagnostics/%s" % DIAGNOSTIC_FILENAME,
                "diagnostic_sha256": diagnostic_sha,
                "manual_review_required": repairs_used >= _MAX_REPAIR_ROUNDS,
            },
            "side_effects": side_effects,
        }
        transaction["transaction_sha256"] = _transaction_sha256(transaction)
        _write_private_json(stage / INTEGRATION_RECEIPT_FILENAME, transaction)
        os.replace(stage, out)
        transaction_path = out / INTEGRATION_RECEIPT_FILENAME
        return transaction, transaction_path
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "IntegrationError",
    "IntegrationPreflightResult",
    "IntegrationTransactionResult",
    "preflight_official_adapter_integration",
    "run_official_adapter_integration",
]
