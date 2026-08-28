"""Materialize verified installed-CANN candidates into recovery receipts.

This module implements one deliberately narrow recovery transition:
``RESOLVED -> CANDIDATE_READY`` for the ``official_binary`` route.  It does
not generate an AsNumPy adapter, build code, invoke a model, access the
network, or activate a capability.

The stored official-resolution receipt is not treated as current environment
truth.  Before publishing anything, the materializer reconstructs the bound
GapSpecs, runs the official resolver again, compares the complete
``resolution_id``, and independently re-hashes the selected header and shared
library beneath the anchored CANN root.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

from numpilot.gaps.installed_cann import (
    ACLNN_ADAPTER_CANDIDATES,
    CATALOG_REVISION,
    EXACT_CANDIDATE,
)
from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    NEW_KERNEL_ROUTE,
    OFFICIAL_SOURCE_ROUTE,
    OfficialGapResolution,
    OfficialOperatorResolutionReport,
    OfficialResolutionError,
    load_official_resolution_report,
    resolve_official_operators,
)
from numpilot.gaps.official_adapter import (
    OFFICIAL_CANDIDATE_MAPPING_REF,
    OFFICIAL_REQUIRED_SYMBOLS_REF,
    OFFICIAL_SELECTED_HEADER_REF,
    OFFICIAL_SELECTED_LIBRARY_REF,
    OFFICIAL_SEMANTIC_REVIEW_REF,
)
from numpilot.gaps.pipeline import (
    GapCompletionError,
    load_script_gap_request,
    prepare_gap_completion,
)
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    LicenseBinding,
    OperatorCandidateArtifact,
    OperatorCandidateOrigin,
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
    create_operator_candidate,
    create_stage_receipt,
    operator_candidate_sha256,
    validate_operator_candidate_binding,
)


CANDIDATE_MATERIALIZATION_SCHEMA_VERSION = 1
EXTERNAL_INSTALLED_CANN_TERMS_ID = (
    "LicenseRef-External-Installed-CANN-Terms"
)

_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_GAP_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_OFFICIAL_REPORT_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 2 * 1024 * 1024
_MAX_LIBRARY_BYTES = 512 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_LINK_SETTLE_ATTEMPTS = 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_GAP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_INSTALLED_FIELDS = frozenset({
    "schema_version",
    "status",
    "qualified_api",
    "expected_cann_version",
    "expected_target_soc",
    "reason_code",
    "reason",
    "adapter_candidate",
    "semantic_equivalence",
    "required_next_gate",
    "catalog_revision",
    "candidate",
    "candidate_mapping_sha256",
    "selected_header",
    "selected_library",
    "required_symbols",
    "missing_symbols",
    "alternatives",
    "provenance",
})
_HEADER_FIELDS = frozenset({
    "relative_path", "sha256", "size", "declared_symbols",
})
_LIBRARY_FIELDS = frozenset({
    "relative_path",
    "sha256",
    "size",
    "probe_tool",
    "aclnn_symbol_count",
    "aclnn_symbols_sha256",
})


class CandidateMaterializationError(RecoveryError):
    """The candidate producer failed closed with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MaterializedOperatorCandidate:
    gap_id: str
    qualified_api: str
    candidate: OperatorCandidateArtifact
    artifact_ref: EvidenceRef
    installed_manifest_ref: EvidenceRef
    license_ref: EvidenceRef

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "qualified_api": self.qualified_api,
            "candidate": self.candidate.to_dict(),
            "candidate_canonical_sha256": operator_candidate_sha256(
                self.candidate
            ),
            "artifact_ref": self.artifact_ref.to_dict(),
            "installed_manifest_ref": self.installed_manifest_ref.to_dict(),
            "license_ref": self.license_ref.to_dict(),
        }


@dataclass(frozen=True)
class CandidateMaterializationResult:
    recovery: RecoveryStatus
    output_dir: str
    fresh_resolution_id: str
    candidates: Tuple[MaterializedOperatorCandidate, ...]
    appended_revisions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
            "record_type": "numpilot_candidate_materialization",
            "route": OperatorCandidateOrigin.OFFICIAL_BINARY.value,
            "output_dir": self.output_dir,
            "fresh_resolution_id": self.fresh_resolution_id,
            "fresh_environment_recheck": True,
            "appended_revisions": self.appended_revisions,
            "side_effects_performed": {
                "subprocess_performed": True,
                "network_access_performed": False,
                "system_write_performed": False,
                "model_invocation_performed": False,
                "asnumpy_import_performed": False,
                "build_performed": False,
                "execution_performed": False,
                "verification_performed": False,
                "activation_performed": False,
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "recovery": self.recovery.to_dict(),
        }


@dataclass(frozen=True)
class _FileSnapshot:
    locator: str
    relative_path: str
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def evidence_ref(self, name: str) -> EvidenceRef:
        return EvidenceRef(name=name, locator=self.locator, sha256=self.sha256)


@dataclass(frozen=True)
class _InstalledCandidateEvidence:
    candidate_mapping: Mapping[str, Any]
    header_evidence: Mapping[str, Any]
    library_evidence: Mapping[str, Any]
    header: _FileSnapshot
    library: _FileSnapshot


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise CandidateMaterializationError(
            "invalid_candidate_evidence",
            "candidate evidence is not canonical JSON",
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_file_bytes(value: object) -> bytes:
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise CandidateMaterializationError(
            "invalid_candidate_evidence",
            "candidate artifact is not strict JSON",
        ) from exc
    if not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise CandidateMaterializationError(
            "candidate_artifact_too_large",
            "candidate artifact exceeds its bounded size",
        )
    return payload


def _find_ref(refs: Sequence[EvidenceRef], name: str) -> EvidenceRef:
    matches = tuple(item for item in refs if item.name == name)
    if len(matches) != 1:
        raise CandidateMaterializationError(
            "missing_recovery_reference",
            "recovery state requires exactly one %s reference" % name,
        )
    return matches[0]


def _path_from_ref(ref: EvidenceRef, label: str) -> Path:
    if "#" in ref.locator:
        raise CandidateMaterializationError(
            "invalid_recovery_reference",
            "%s must refer to a whole local file" % label,
        )
    path = Path(ref.locator)
    if not path.is_absolute() or Path(os.path.normpath(str(path))) != path:
        raise CandidateMaterializationError(
            "invalid_recovery_reference",
            "%s path must be absolute and normalized" % label,
        )
    return path


def _hash_open_file(
    descriptor: int,
    maximum: int,
    label: str,
) -> Tuple[str, int, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
        raise CandidateMaterializationError(
            "unsafe_candidate_file", "%s must be a non-empty regular file" % label
        )
    if before.st_size > maximum:
        raise CandidateMaterializationError(
            "candidate_file_too_large", "%s exceeds its size limit" % label
        )

    digests = []
    sizes = []
    for _pass in range(2):
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise CandidateMaterializationError(
                    "candidate_file_too_large",
                    "%s exceeds its size limit" % label,
                )
            digest.update(chunk)
        digests.append(digest.hexdigest())
        sizes.append(total)
    after = os.fstat(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(set(digests)) != 1 or len(set(sizes)) != 1:
        raise CandidateMaterializationError(
            "candidate_file_drift", "%s changed while it was hashed" % label
        )
    return digests[0], sizes[0], after


def _snapshot_regular_path(path: Path, maximum: int, label: str) -> _FileSnapshot:
    if not path.is_absolute() or Path(os.path.normpath(str(path))) != path:
        raise CandidateMaterializationError(
            "unsafe_input", "%s path must be absolute and normalized" % label
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise CandidateMaterializationError(
            "secure_path_unavailable",
            "this platform cannot traverse bound inputs safely",
        )
    directory_flags = (
        os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = -1
    descriptor = -1
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for component in path.parts[1:-1]:
            following = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = following
        descriptor = os.open(path.name, file_flags, dir_fd=directory_fd)
        digest, size, details = _hash_open_file(descriptor, maximum, label)
        current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (details.st_dev, details.st_ino):
            raise CandidateMaterializationError(
                "input_drift", "%s changed while it was read" % label
            )
        return _FileSnapshot(
            locator=str(path),
            relative_path=path.name,
            sha256=digest,
            size=size,
            device=details.st_dev,
            inode=details.st_ino,
            mtime_ns=details.st_mtime_ns,
            ctime_ns=details.st_ctime_ns,
        )
    except CandidateMaterializationError:
        raise
    except OSError as exc:
        raise CandidateMaterializationError(
            "unsafe_input", "cannot read %s safely" % label
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence", "%s path is invalid" % label
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence", "%s path is unsafe" % label
        )
    return relative


def _canonical_real_directory(value: str, label: str) -> Path:
    try:
        supplied = Path(value)
        if not supplied.is_absolute() or Path(os.path.normpath(str(supplied))) != supplied:
            raise ValueError
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "%s must be a real canonical directory" % label,
        ) from exc
    if supplied != resolved or supplied.is_symlink() or not resolved.is_dir():
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "%s must be a real canonical directory" % label,
        )
    return resolved


def _snapshot_beneath(
    root_value: str,
    relative_value: object,
    *,
    expected_sha256: object,
    expected_size: object,
    maximum: int,
    label: str,
) -> _FileSnapshot:
    root = _canonical_real_directory(root_value, "CANN root")
    relative = _safe_relative_path(relative_value, label)
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence", "%s digest is invalid" % label
        )
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or expected_size > maximum
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence", "%s size is invalid" % label
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise CandidateMaterializationError(
            "secure_path_unavailable",
            "this platform cannot anchor installed CANN evidence safely",
        )
    directory_flags = os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    current_fd = -1
    file_fd = -1
    try:
        current_fd = os.open(str(root), directory_flags)
        for component in relative.parts[:-1]:
            following = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = following
        file_flags = (
            os.O_RDONLY
            | os.O_NONBLOCK
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_fd = os.open(relative.name, file_flags, dir_fd=current_fd)
        digest, size, details = _hash_open_file(file_fd, maximum, label)
    except CandidateMaterializationError:
        raise
    except OSError as exc:
        raise CandidateMaterializationError(
            "unsafe_candidate_file", "cannot open %s beneath CANN root" % label
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if current_fd >= 0:
            os.close(current_fd)
    if digest != expected_sha256 or size != expected_size:
        raise CandidateMaterializationError(
            "candidate_file_hash_mismatch",
            "%s differs from the fresh official inventory" % label,
        )
    return _FileSnapshot(
        locator=str(root.joinpath(*relative.parts)),
        relative_path=relative.as_posix(),
        sha256=digest,
        size=size,
        device=details.st_dev,
        inode=details.st_ino,
        mtime_ns=details.st_mtime_ns,
        ctime_ns=details.st_ctime_ns,
    )


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if os.path.lexists(current) and current.is_symlink():
            raise CandidateMaterializationError(
                "unsafe_output", "candidate output contains a symlink"
            )
        if current == current.parent:
            return
        current = current.parent


def _prepare_output_path(value: os.PathLike[str] | str) -> Path:
    try:
        path = Path(value)
    except TypeError as exc:
        raise CandidateMaterializationError(
            "unsafe_output", "candidate output path is invalid"
        ) from exc
    if (
        not path.is_absolute()
        or Path(os.path.normpath(str(path))) != path
        or len(str(path)) > 4096
    ):
        raise CandidateMaterializationError(
            "unsafe_output", "candidate output path must be absolute and normalized"
        )
    _reject_symlink_components(path)
    return path


def _ensure_private_directory(path: Path) -> None:
    _reject_symlink_components(path)
    if not os.path.lexists(path):
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise CandidateMaterializationError(
                "unsafe_output", "candidate output parent must be a real directory"
            )
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise CandidateMaterializationError(
                "output_write_failed", "cannot create candidate output directory"
            ) from exc
    try:
        details = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(details.st_mode):
            raise CandidateMaterializationError(
                "unsafe_output", "candidate output is not a directory"
            )
        if stat.S_IMODE(details.st_mode) != 0o700:
            os.chmod(path, 0o700, follow_symlinks=False)
            details = os.stat(path, follow_symlinks=False)
        if stat.S_IMODE(details.st_mode) != 0o700:
            raise CandidateMaterializationError(
                "unsafe_output", "candidate output directory must be mode 0700"
            )
    except CandidateMaterializationError:
        raise
    except OSError as exc:
        raise CandidateMaterializationError(
            "unsafe_output", "cannot validate candidate output directory"
        ) from exc


def _read_existing_private_file(
    directory_fd: int, name: str, maximum: int
) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        details = os.fstat(descriptor)
        # A successful publisher creates the destination with a hard link and
        # immediately removes its private temporary name.  An identical
        # concurrent publisher can observe the destination during that tiny
        # two-link interval.  Wait only for that known transient state; a
        # persistent or larger link count remains an unsafe collision.
        for _attempt in range(_LINK_SETTLE_ATTEMPTS):
            if details.st_nlink != 2:
                break
            time.sleep(0.001)
            details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
            or details.st_size < 1
            or details.st_size > maximum
        ):
            raise CandidateMaterializationError(
                "unsafe_output", "existing candidate artifact is not private"
            )
        chunks = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise CandidateMaterializationError(
                    "unsafe_output", "existing candidate artifact is truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
            details.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if identity_before != identity_after:
            raise CandidateMaterializationError(
                "unsafe_output", "existing candidate artifact changed while read"
            )
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_private_file(path: Path, payload: bytes) -> None:
    if not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise CandidateMaterializationError(
            "candidate_artifact_too_large", "candidate artifact is invalid"
        )
    _ensure_private_directory(path.parent)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = -1
    descriptor = -1
    temporary = ""
    linked = False
    try:
        directory_fd = os.open(str(path.parent), directory_flags)
        temporary = ".candidate-%s.tmp" % secrets.token_hex(16)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise CandidateMaterializationError(
                    "output_write_failed", "short candidate artifact write"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            existing = _read_existing_private_file(
                directory_fd, path.name, _MAX_ARTIFACT_BYTES
            )
            if existing != payload:
                raise CandidateMaterializationError(
                    "output_collision",
                    "refusing to overwrite a different candidate artifact",
                )
        os.unlink(temporary, dir_fd=directory_fd)
        temporary = ""
        if linked:
            os.fsync(directory_fd)
    except CandidateMaterializationError:
        raise
    except OSError as exc:
        raise CandidateMaterializationError(
            "output_write_failed", "cannot publish candidate artifact"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary and directory_fd >= 0:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
        if directory_fd >= 0:
            os.close(directory_fd)


def _validate_installed_decision(
    decision: OfficialGapResolution,
    report: OfficialOperatorResolutionReport,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    payload = decision.installed_cann
    candidate = payload.get("candidate")
    selected_header = payload.get("selected_header")
    selected_library = payload.get("selected_library")
    if (
        decision.route != INSTALLED_ADAPTER_ROUTE
        or frozenset(payload) != _INSTALLED_FIELDS
        or payload.get("schema_version") != 1
        or payload.get("status") != EXACT_CANDIDATE
        or payload.get("adapter_candidate") is not True
        or payload.get("semantic_equivalence") != "unverified"
        or payload.get("provenance")
        != "installed_cann_toolkit_read_only_inventory"
        or payload.get("catalog_revision") != CATALOG_REVISION
        or payload.get("qualified_api") != decision.qualified_api
        or payload.get("expected_cann_version") != report.cann_version
        or payload.get("expected_target_soc") != report.target_soc
        or payload.get("reason_code") != "header_and_dynamic_symbols_present"
        or payload.get("required_next_gate")
        != "generate_and_qualify_asnumpy_adapter"
        or payload.get("missing_symbols") != []
        or not isinstance(payload.get("alternatives"), list)
        or not isinstance(candidate, Mapping)
        or not isinstance(selected_header, Mapping)
        or not isinstance(selected_library, Mapping)
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "installed CANN decision is incomplete or inconsistent",
        )
    if (
        frozenset(selected_header) != _HEADER_FIELDS
        or frozenset(selected_library) != _LIBRARY_FIELDS
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "installed CANN selected artifact schema is invalid",
        )
    declared_symbols = selected_header.get("declared_symbols")
    required_symbols = payload.get("required_symbols")
    if (
        not isinstance(declared_symbols, list)
        or not isinstance(required_symbols, list)
        or any(not isinstance(item, str) or not item for item in declared_symbols)
        or any(not isinstance(item, str) or not item for item in required_symbols)
        or len(set(declared_symbols)) != len(declared_symbols)
        or len(set(required_symbols)) != len(required_symbols)
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "installed CANN symbol evidence is invalid",
        )
    reviewed = next(
        (
            item
            for item in ACLNN_ADAPTER_CANDIDATES
            if item.candidate_id == candidate.get("candidate_id")
        ),
        None,
    )
    if (
        reviewed is None
        or candidate != reviewed.to_dict()
        or payload.get("candidate_mapping_sha256") != reviewed.mapping_sha256
        or reviewed.qualified_api != decision.qualified_api
        or report.cann_version not in reviewed.audited_cann_versions
        or "".join(ch for ch in report.target_soc.casefold() if ch.isalnum())
        not in reviewed.audited_target_socs
        or required_symbols != list(reviewed.required_symbols)
        or sorted(declared_symbols)
        != sorted(reviewed.required_symbols)
        or Path(str(selected_header.get("relative_path", ""))).name
        != reviewed.header_name
        or Path(str(selected_library.get("relative_path", ""))).name
        not in reviewed.library_priority
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "installed CANN candidate does not match the reviewed catalog",
        )
    if (
        not isinstance(selected_library.get("probe_tool"), str)
        or not selected_library["probe_tool"]
        or not isinstance(selected_library.get("aclnn_symbol_count"), int)
        or isinstance(selected_library.get("aclnn_symbol_count"), bool)
        or selected_library["aclnn_symbol_count"] < len(reviewed.required_symbols)
        or not isinstance(selected_library.get("aclnn_symbols_sha256"), str)
        or not _SHA256.fullmatch(selected_library["aclnn_symbols_sha256"])
    ):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence",
            "installed CANN library symbol evidence is invalid",
        )
    return candidate, selected_header, selected_library


def _materialize_one(
    *,
    output_root: Path,
    report: OfficialOperatorResolutionReport,
    decision: OfficialGapResolution,
    gap_spec: Mapping[str, Any],
    gap_spec_ref: EvidenceRef,
    decision_ref: EvidenceRef,
    official_report_ref: EvidenceRef,
    installed_evidence: _InstalledCandidateEvidence,
) -> MaterializedOperatorCandidate:
    candidate_mapping = installed_evidence.candidate_mapping
    header_evidence = installed_evidence.header_evidence
    library_evidence = installed_evidence.library_evidence
    header = installed_evidence.header
    library = installed_evidence.library

    gap_dir = output_root / "gaps" / decision.gap_id
    if not _SAFE_GAP_ID.fullmatch(decision.gap_id):
        raise CandidateMaterializationError(
            "invalid_official_binary_evidence", "candidate gap id is unsafe"
        )
    _ensure_private_directory(gap_dir)

    terms_path = gap_dir / "external-installed-cann-terms.json"
    terms_value = {
        "schema_version": CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
        "record_type": "external_installed_cann_terms_status",
        "identifier": EXTERNAL_INSTALLED_CANN_TERMS_ID,
        "accepted": False,
        "cann_root": report.cann_root,
        "cann_version": report.cann_version,
        "target_soc": report.target_soc,
        "gap_id": decision.gap_id,
        "qualified_api": decision.qualified_api,
        "ops_nn_cann_osl_applied_to_installed_binary": False,
        "activation_allowed": False,
    }
    terms_payload = _json_file_bytes(terms_value)
    _publish_private_file(terms_path, terms_payload)
    terms_ref = EvidenceRef(
        name="external_installed_cann_terms",
        locator=str(terms_path),
        sha256=hashlib.sha256(terms_payload).hexdigest(),
    )
    license_binding = LicenseBinding(
        identifier=EXTERNAL_INSTALLED_CANN_TERMS_ID,
        accepted=False,
        ref=terms_ref,
    )

    header_ref = header.evidence_ref(OFFICIAL_SELECTED_HEADER_REF)
    library_ref = library.evidence_ref(OFFICIAL_SELECTED_LIBRARY_REF)
    installed_manifest_path = gap_dir / "installed-cann-candidate.json"
    installed_manifest = {
        "schema_version": CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
        "artifact_type": "installed_cann_binary_candidate",
        "origin_kind": OperatorCandidateOrigin.OFFICIAL_BINARY.value,
        "gap_id": decision.gap_id,
        "qualified_api": decision.qualified_api,
        "target_soc": report.target_soc,
        "cann_version": report.cann_version,
        "cann_root": report.cann_root,
        "fresh_resolution_id": report.resolution_id,
        "official_resolution_ref": official_report_ref.to_dict(),
        "resolution_decision_ref": decision_ref.to_dict(),
        "gap_spec_ref": gap_spec_ref.to_dict(),
        "catalog_revision": CATALOG_REVISION,
        "candidate_mapping": dict(candidate_mapping),
        "candidate_mapping_sha256": decision.installed_cann[
            "candidate_mapping_sha256"
        ],
        "required_symbols": list(candidate_mapping["required_symbols"]),
        "selected_header": {
            "ref": header_ref.to_dict(),
            "relative_path": header.relative_path,
            "size": header.size,
            "declared_symbols": list(header_evidence["declared_symbols"]),
        },
        "selected_library": {
            "ref": library_ref.to_dict(),
            "relative_path": library.relative_path,
            "size": library.size,
            "probe_tool": library_evidence["probe_tool"],
            "aclnn_symbol_count": library_evidence["aclnn_symbol_count"],
            "aclnn_symbols_sha256": library_evidence["aclnn_symbols_sha256"],
        },
        "license": license_binding.to_dict(),
        "semantic_equivalence": "unverified",
        "build_performed": False,
        "activation_allowed": False,
    }
    installed_payload = _json_file_bytes(installed_manifest)
    _publish_private_file(installed_manifest_path, installed_payload)
    installed_ref = EvidenceRef(
        name="installed_cann_manifest",
        locator=str(installed_manifest_path),
        sha256=hashlib.sha256(installed_payload).hexdigest(),
    )
    mapping_ref = EvidenceRef(
        name=OFFICIAL_CANDIDATE_MAPPING_REF,
        locator=str(installed_manifest_path) + "#candidate_mapping",
        sha256=decision.installed_cann["candidate_mapping_sha256"],
    )
    required_symbols_ref = EvidenceRef(
        name=OFFICIAL_REQUIRED_SYMBOLS_REF,
        locator=str(installed_manifest_path) + "#required_symbols",
        sha256=_canonical_sha256({
            "required_symbols": list(candidate_mapping["required_symbols"]),
        }),
    )
    semantic_review_ref = EvidenceRef(
        name=OFFICIAL_SEMANTIC_REVIEW_REF,
        locator=str(installed_manifest_path) + "#candidate_mapping/semantic_review",
        sha256=_canonical_sha256({
            "semantic_review": list(candidate_mapping["semantic_review"]),
        }),
    )

    candidate = create_operator_candidate(
        origin_kind=OperatorCandidateOrigin.OFFICIAL_BINARY,
        gap_id=decision.gap_id,
        qualified_api=decision.qualified_api,
        gap_spec_ref=gap_spec_ref,
        resolution_decision_ref=decision_ref,
        target_soc=report.target_soc,
        cann_version=report.cann_version,
        license=license_binding,
        provenance_refs=(
            mapping_ref,
            semantic_review_ref,
            installed_ref,
            official_report_ref,
        ),
        candidate_refs=(header_ref, library_ref, required_symbols_ref),
    )
    validate_operator_candidate_binding(
        candidate,
        gap_spec=gap_spec,
        resolution_decision=decision.to_dict(),
    )
    candidate_path = gap_dir / "operator-candidate.json"
    candidate_payload = _json_file_bytes(candidate.to_dict())
    _publish_private_file(candidate_path, candidate_payload)
    candidate_ref = EvidenceRef(
        name="operator_candidate",
        locator=str(candidate_path),
        sha256=hashlib.sha256(candidate_payload).hexdigest(),
    )
    return MaterializedOperatorCandidate(
        gap_id=decision.gap_id,
        qualified_api=decision.qualified_api,
        candidate=candidate,
        artifact_ref=candidate_ref,
        installed_manifest_ref=installed_ref,
        license_ref=terms_ref,
    )


def _preflight_installed_candidate(
    decision: OfficialGapResolution,
    report: OfficialOperatorResolutionReport,
) -> _InstalledCandidateEvidence:
    candidate_mapping, header_evidence, library_evidence = (
        _validate_installed_decision(decision, report)
    )
    header = _snapshot_beneath(
        report.cann_root,
        header_evidence.get("relative_path"),
        expected_sha256=header_evidence.get("sha256"),
        expected_size=header_evidence.get("size"),
        maximum=_MAX_HEADER_BYTES,
        label="installed CANN header",
    )
    library = _snapshot_beneath(
        report.cann_root,
        library_evidence.get("relative_path"),
        expected_sha256=library_evidence.get("sha256"),
        expected_size=library_evidence.get("size"),
        maximum=_MAX_LIBRARY_BYTES,
        label="installed CANN library",
    )
    return _InstalledCandidateEvidence(
        candidate_mapping=candidate_mapping,
        header_evidence=header_evidence,
        library_evidence=library_evidence,
        header=header,
        library=library,
    )


def _validate_route_support(report: OfficialOperatorResolutionReport) -> None:
    for decision in report.decisions:
        if decision.route == OFFICIAL_SOURCE_ROUTE:
            raise CandidateMaterializationError(
                "official_source_not_supported",
                "official-source materialization requires a pinned source snapshot and semantic gate",
            )
        if decision.route == NEW_KERNEL_ROUTE:
            raise CandidateMaterializationError(
                "remote_llm_not_supported",
                "remote model candidates require a separately authorized producer",
            )
        if decision.route != INSTALLED_ADAPTER_ROUTE:
            raise CandidateMaterializationError(
                "unsupported_resolution_route",
                "resolution route cannot produce an official binary candidate",
            )


def _verify_input_ref(ref: EvidenceRef, maximum: int, label: str) -> Path:
    path = _path_from_ref(ref, label)
    snapshot = _snapshot_regular_path(path, maximum, label)
    if snapshot.sha256 != ref.sha256:
        raise CandidateMaterializationError(
            "input_hash_mismatch", "%s differs from the recovery binding" % label
        )
    return path


def _candidate_index(
    output_root: Path,
    report: OfficialOperatorResolutionReport,
    candidates: Sequence[MaterializedOperatorCandidate],
) -> EvidenceRef:
    value = {
        "schema_version": CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
        "record_type": "operator_candidate_index",
        "origin_kind": OperatorCandidateOrigin.OFFICIAL_BINARY.value,
        "fresh_resolution_id": report.resolution_id,
        "activation_allowed": False,
        "candidates": [
            {
                "gap_id": item.gap_id,
                "qualified_api": item.qualified_api,
                "candidate_id": item.candidate.candidate_id,
                "artifact_ref": item.artifact_ref.to_dict(),
            }
            for item in sorted(candidates, key=lambda item: item.gap_id)
        ],
    }
    path = output_root / "operator-candidate-index.json"
    payload = _json_file_bytes(value)
    _publish_private_file(path, payload)
    return EvidenceRef(
        name="operator_candidate_index",
        locator=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _status_matches_candidates(
    status: RecoveryStatus,
    candidates: Sequence[MaterializedOperatorCandidate],
    index_ref: EvidenceRef,
) -> bool:
    if (
        status.state.phase is not RecoveryPhase.CANDIDATE_READY
        or status.state.outcome is not RecoveryOutcome.READY
    ):
        return False
    try:
        if _find_ref(status.state.refs, "operator_candidate_index") != index_ref:
            return False
    except CandidateMaterializationError:
        return False
    expected = {item.gap_id: item for item in candidates}
    if set(expected) != {item.gap_id for item in status.state.gaps}:
        return False
    receipts = tuple(
        item
        for item in status.state.stage_receipts
        if item.phase is RecoveryPhase.CANDIDATE_READY
        and item.kind is StageReceiptKind.OPERATOR_CANDIDATE
        and item.verdict is StageReceiptVerdict.SATISFIED
    )
    covered = set()
    for gap in status.state.gaps:
        if gap.phase is not RecoveryPhase.CANDIDATE_READY:
            return False
        try:
            ref = _find_ref(gap.refs, "operator_candidate")
        except CandidateMaterializationError:
            return False
        if ref != expected[gap.gap_id].artifact_ref:
            return False
        matching = tuple(
            item
            for item in receipts
            if item.scope is StageReceiptScope.GAPS
            and item.covered_gap_ids == (gap.gap_id,)
            and item.ref == ref
        )
        if len(matching) != 1:
            return False
        covered.add(gap.gap_id)
    return covered == set(expected)


def materialize_official_binary_candidates(
    state_dir: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> CandidateMaterializationResult:
    """Freshly verify and publish official-binary candidates, then append once.

    The recovery binding must already authorize subprocess use.  This check is
    made before the fresh resolver runs and before the output path is created.
    The function is write-once and idempotent; concurrent identical producers
    either win the compare-and-append or verify the winner's exact state.
    """

    store = RecoveryStore(state_dir)
    try:
        status = store.status()
    except RecoveryError as exc:
        raise CandidateMaterializationError(
            "state_unavailable", "cannot verify the recovery state"
        ) from exc
    state = status.state
    if state.phase not in {RecoveryPhase.RESOLVED, RecoveryPhase.CANDIDATE_READY}:
        raise CandidateMaterializationError(
            "state_phase_mismatch",
            "candidate materialization requires RESOLVED or CANDIDATE_READY",
        )
    if state.outcome is not RecoveryOutcome.READY:
        raise CandidateMaterializationError(
            "state_not_ready", "recovery state is not ready for materialization"
        )
    if not state.policy.subprocess_allowed:
        raise CandidateMaterializationError(
            "subprocess_not_authorized",
            "fresh official resolution requires frozen subprocess authorization",
        )

    gap_request_ref = _find_ref(state.refs, "gap_request")
    official_ref = _find_ref(state.refs, "official_resolution")
    gap_request_path = _verify_input_ref(
        gap_request_ref, _MAX_GAP_REQUEST_BYTES, "gap request"
    )
    official_path = _verify_input_ref(
        official_ref, _MAX_OFFICIAL_REPORT_BYTES, "official resolution"
    )
    source_snapshot = _snapshot_regular_path(
        Path(state.source.path), _MAX_SOURCE_BYTES, "recovery source"
    )
    if source_snapshot.sha256 != state.source.sha256:
        raise CandidateMaterializationError(
            "input_hash_mismatch", "source differs from the recovery binding"
        )

    try:
        selection = load_script_gap_request(
            str(gap_request_path),
            state.source.path,
            target_soc=state.target.target_soc,
            cann_version=state.target.cann_version,
        )
        if selection.blocked:
            raise CandidateMaterializationError(
                "unsupported_gap_bundle",
                "candidate materialization requires generatable gaps only",
            )
        preparation = prepare_gap_completion(
            state.source.path,
            target_soc=state.target.target_soc,
            cann_version=state.target.cann_version,
            requested_specs=selection.specs,
            requested_blocked=selection.blocked,
        )
        stored_report = load_official_resolution_report(str(official_path))
        stored_report.validate_preparation(preparation)
    except CandidateMaterializationError:
        raise
    except (GapCompletionError, OfficialResolutionError, OSError, ValueError) as exc:
        raise CandidateMaterializationError(
            "invalid_bound_receipt",
            "cannot reconstruct the bound gap and official receipts",
        ) from exc
    _validate_route_support(stored_report)

    try:
        fresh_report = resolve_official_operators(
            preparation,
            cann_root=stored_report.cann_root,
            ops_nn_root=stored_report.ops_nn_root,
            accept_cann_osl=stored_report.cann_osl_accepted,
        )
        if not isinstance(fresh_report, OfficialOperatorResolutionReport):
            raise OfficialResolutionError("fresh resolver returned a wrong type")
        fresh_report.validate_preparation(preparation)
    except (OfficialResolutionError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CandidateMaterializationError(
            "fresh_resolution_failed", "fresh official resolution failed safely"
        ) from exc
    if fresh_report.resolution_id != stored_report.resolution_id:
        raise CandidateMaterializationError(
            "fresh_resolution_drift",
            "current official evidence differs from the stored resolution",
        )
    _validate_route_support(fresh_report)

    # Re-check immutable producer inputs after the subprocess boundary and
    # before creating the output tree.
    _verify_input_ref(gap_request_ref, _MAX_GAP_REQUEST_BYTES, "gap request")
    _verify_input_ref(official_ref, _MAX_OFFICIAL_REPORT_BYTES, "official resolution")
    source_after = _snapshot_regular_path(
        Path(state.source.path), _MAX_SOURCE_BYTES, "recovery source"
    )
    if source_after.sha256 != state.source.sha256:
        raise CandidateMaterializationError(
            "input_hash_mismatch", "source changed during fresh resolution"
        )

    specs = {spec.gap_id: spec for spec in selection.specs}
    decisions = {item.gap_id: item for item in fresh_report.decisions}
    state_gaps = {item.gap_id: item for item in state.gaps}
    if set(specs) != set(decisions) or set(specs) != set(state_gaps):
        raise CandidateMaterializationError(
            "candidate_coverage_mismatch",
            "candidate producer coverage differs from recovery state",
        )

    bindings = {}
    installed_evidence = {}
    for gap_id in sorted(specs):
        gap_state = state_gaps[gap_id]
        if gap_state.phase not in {
            RecoveryPhase.RESOLVED,
            RecoveryPhase.CANDIDATE_READY,
        }:
            raise CandidateMaterializationError(
                "state_phase_mismatch", "gap is not at a candidate materialization phase"
            )
        gap_spec_ref = _find_ref(gap_state.refs, "gap_spec")
        decision_ref = _find_ref(gap_state.refs, "resolution_decision")
        spec_value = specs[gap_id].to_dict()
        decision_value = decisions[gap_id].to_dict()
        if (
            _canonical_sha256(spec_value) != gap_spec_ref.sha256
            or _canonical_sha256(decision_value) != decision_ref.sha256
        ):
            raise CandidateMaterializationError(
                "candidate_binding_mismatch",
                "GapSpec or resolution decision differs from recovery state",
            )
        bindings[gap_id] = (gap_spec_ref, decision_ref, spec_value)
        installed_evidence[gap_id] = _preflight_installed_candidate(
            decisions[gap_id], fresh_report
        )

    # Only now may private output publication begin: every route, binding,
    # selected file, and current content hash has already passed preflight.
    output_root = _prepare_output_path(output_dir)
    _ensure_private_directory(output_root)
    _ensure_private_directory(output_root / "gaps")
    materialized = []
    for gap_id in sorted(specs):
        gap_spec_ref, decision_ref, spec_value = bindings[gap_id]
        materialized.append(
            _materialize_one(
                output_root=output_root,
                report=fresh_report,
                decision=decisions[gap_id],
                gap_spec=spec_value,
                gap_spec_ref=gap_spec_ref,
                decision_ref=decision_ref,
                official_report_ref=official_ref,
                installed_evidence=installed_evidence[gap_id],
            )
        )
    candidates = tuple(materialized)
    index_ref = _candidate_index(output_root, fresh_report, candidates)

    if _status_matches_candidates(status, candidates, index_ref):
        return CandidateMaterializationResult(
            recovery=status,
            output_dir=str(output_root),
            fresh_resolution_id=fresh_report.resolution_id,
            candidates=candidates,
            appended_revisions=0,
        )
    if state.phase is not RecoveryPhase.RESOLVED or any(
        item.phase is not RecoveryPhase.RESOLVED for item in state.gaps
    ):
        raise CandidateMaterializationError(
            "candidate_state_mismatch",
            "existing candidate state does not match the materialized artifacts",
        )

    by_gap = {item.gap_id: item for item in candidates}
    next_gaps = tuple(
        GapRecoveryState(
            gap_id=gap.gap_id,
            phase=RecoveryPhase.CANDIDATE_READY,
            outcome=RecoveryOutcome.READY,
            attempt_count=gap.attempt_count,
            refs=tuple(gap.refs) + (by_gap[gap.gap_id].artifact_ref,),
            reason="",
        )
        for gap in state.gaps
    )
    receipts = tuple(
        create_stage_receipt(
            phase=RecoveryPhase.CANDIDATE_READY,
            kind=StageReceiptKind.OPERATOR_CANDIDATE,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(item.gap_id,),
            binding=state.binding,
            parent_state_sha256=status.state_sha256,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=item.artifact_ref,
        )
        for item in candidates
    )
    event = RecoveryEvent.bound_to(
        state,
        phase=RecoveryPhase.CANDIDATE_READY,
        outcome=RecoveryOutcome.READY,
        refs=(index_ref,),
        gaps=next_gaps,
        stage_receipts=receipts,
        side_effects=SideEffectTruth(subprocess_performed=True),
        reason=(
            "fresh installed-CANN candidates were content-bound; semantic, "
            "adapter, build, execution, and activation gates remain pending"
        ),
    )
    try:
        final_status = store.append(event)
        appended = 1
    except RecoveryError as exc:
        try:
            concurrent = store.status()
        except RecoveryError as reload_exc:
            raise CandidateMaterializationError(
                "state_append_failed", "cannot append or reload candidate state"
            ) from reload_exc
        if not _status_matches_candidates(concurrent, candidates, index_ref):
            raise CandidateMaterializationError(
                "state_append_failed", "candidate state changed concurrently"
            ) from exc
        final_status = concurrent
        appended = 0
    return CandidateMaterializationResult(
        recovery=final_status,
        output_dir=str(output_root),
        fresh_resolution_id=fresh_report.resolution_id,
        candidates=candidates,
        appended_revisions=appended,
    )


__all__ = [
    "CANDIDATE_MATERIALIZATION_SCHEMA_VERSION",
    "EXTERNAL_INSTALLED_CANN_TERMS_ID",
    "CandidateMaterializationError",
    "CandidateMaterializationResult",
    "MaterializedOperatorCandidate",
    "materialize_official_binary_candidates",
]
