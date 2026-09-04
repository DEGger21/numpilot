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

"""Run-scoped, receipt-bound AsNumpy compatibility extensions.

This module deliberately keeps two facts separate:

* CANNBot may have produced an unverified Ascend C *draft*; and
* NumPilot may qualify a small composition of already audited AsNumpy APIs for
  one run.

The latter never activates the former and never modifies the installed
AsNumpy package.  A positive ``interface_available`` claim is emitted only by
the built-in subprocess runner after real AsNumpy execution and a NumPy oracle
comparison.  Injected runners are useful for offline contract tests, but can
never publish that claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


EXTENSION_SCHEMA_VERSION = 1
QUALIFICATION_SCHEMA_VERSION = 1
WORKER_SCHEMA_VERSION = 1
RUNTIME_ATTESTATION_SCHEMA_VERSION = 1
EXTENSION_MODULE_NAME = "numpilot_run_add_one"
EXTENSION_FILENAME = "add_one_extension.py"
MANIFEST_FILENAME = "runtime-extension-manifest.json"
QUALIFICATION_FILENAME = "runtime-extension-qualification.json"
WORKER_EVIDENCE_FILENAME = "worker-evidence.json"
PROVENANCE = "composite_npu"
SCOPE = "run"
INTERFACE = "asnumpy.add_one"
CONSTITUENTS = ("asnumpy.ones_like", "asnumpy.add")
CONSTITUENT_KEYS = ("ones_like", "add")
CONSTRAINTS = ("dtype=float32", "rank>=1", "contiguous=true", "numel>=1")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_DRAFT_BYTES = 256 * 1024
_MAX_MODULE_BYTES = 64 * 1024
_MAX_TIMEOUT_SECONDS = 3600.0


class RuntimeExtensionError(RuntimeError):
    """A runtime extension failed a local or execution gate."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class MaterializedRuntimeExtension:
    root: str
    module_path: str
    module_sha256: str
    manifest_path: str
    manifest_sha256: str
    cannbot_report_path: str
    cannbot_report_sha256: str
    cannbot_draft_path: str
    cannbot_draft_sha256: str
    interface: str = INTERFACE
    provenance: str = PROVENANCE
    scope: str = SCOPE


@dataclass(frozen=True)
class QualificationInvocation:
    """Bound inputs exposed to an offline test runner.

    The environment used by the production subprocess is intentionally not
    exposed here: it can contain machine-specific paths and must never become
    part of a test receipt.
    """

    command: Tuple[str, ...]
    cwd: str
    timeout_seconds: float
    contract_path: str
    contract_sha256: str
    evidence_path: str


@dataclass(frozen=True)
class RunnerObservation:
    returncode: int
    timed_out: bool = False
    evidence: Optional[Mapping[str, Any]] = None
    error_type: Optional[str] = None


@dataclass(frozen=True)
class RuntimeExtensionQualification:
    report_path: str
    report_sha256: str
    status: str
    interface_available: bool
    provenance: str
    scope: str
    upstream_modified: bool
    cannbot_draft_activated: bool
    numerical_cases: int
    max_abs_error: Optional[float]


Runner = Callable[[QualificationInvocation], RunnerObservation]


_EXTENSION_SOURCE = '''# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
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

"""NumPilot run-scoped AsNumpy add_one composition.

This module is mechanically materialized by NumPilot.  It does not contain or
execute the associated generated_unverified CANNBot Ascend C draft.
"""

INTERFACE = "asnumpy.add_one"
PROVENANCE = "composite_npu"
SCOPE = "run"
UPSTREAM_MODIFIED = False
CANNBOT_DRAFT_ACTIVATED = False
CONSTITUENTS = ("asnumpy.ones_like", "asnumpy.add")


def activate(asnumpy_module):
    """Attach the composition to one imported AsNumpy module object."""

    if not callable(getattr(asnumpy_module, "ones_like", None)):
        raise RuntimeError("asnumpy.ones_like is unavailable")
    if not callable(getattr(asnumpy_module, "add", None)):
        raise RuntimeError("asnumpy.add is unavailable")
    existing = asnumpy_module.__dict__.get("add_one")
    if existing is not None:
        if getattr(existing, "__numpilot_run_extension__", False):
            return existing
        raise RuntimeError("asnumpy.add_one already exists outside this run extension")

    def add_one(x):
        one = asnumpy_module.ones_like(x, dtype=x.dtype)
        return asnumpy_module.add(x, one, dtype=x.dtype)

    add_one.__name__ = "add_one"
    add_one.__module__ = "asnumpy"
    add_one.__numpilot_run_extension__ = True
    add_one.__numpilot_provenance__ = PROVENANCE
    add_one.__numpilot_constituents__ = CONSTITUENTS
    setattr(asnumpy_module, "add_one", add_one)
    return add_one
'''


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeExtensionError(
            "invalid_report", "runtime extension report is not serializable"
        ) from exc


def _runtime_attestation_issues(
    identity_value: object, capability_receipt: object
) -> Tuple[str, ...]:
    """Revalidate the existing runtime/capability evidence as one contract.

    Merely importing ``asnumpy`` and receiving an ``asnumpy.ndarray`` is not an
    NPU provenance proof: a compatible package may be a shim or may contain a
    host fallback.  Reuse NumPilot's hash-bound runtime identity and strict
    capability receipt instead of creating a weaker, extension-local notion of
    "NPU".
    """

    issues = []
    try:
        from numpilot.capabilities import RuntimeIdentity, strict_receipt_issues

        identity = RuntimeIdentity.from_dict(identity_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ("runtime_identity_invalid",)
    if identity.status != "matched":
        issues.append("runtime_identity_not_matched")
    if not isinstance(capability_receipt, Mapping):
        issues.append("capability_receipt_missing")
        return tuple(issues)
    if strict_receipt_issues(capability_receipt, CONSTITUENT_KEYS):
        issues.append("constituent_capability_not_strict_npu")
    if capability_receipt.get("backend") != "npu":
        issues.append("capability_backend_not_npu")
    if capability_receipt.get("backend_version") != identity.asnumpy_version:
        issues.append("capability_runtime_version_mismatch")
    if capability_receipt.get("audit_id") != identity.audit_id:
        issues.append("capability_runtime_audit_mismatch")
    if capability_receipt.get("runtime_target_integrity_ok") is not True:
        issues.append("runtime_target_integrity_unverified")
    if capability_receipt.get("profile_status") not in {"not_required", "matched"}:
        issues.append("runtime_profile_unverified")
    environment = capability_receipt.get("environment")
    if not isinstance(environment, Mapping):
        issues.append("capability_environment_missing")
    else:
        expected = {
            "cann_version": identity.cann_version,
            "soc_version": identity.soc_model,
            "python_version": identity.python_version,
        }
        if any(environment.get(name) != value for name, value in expected.items()):
            issues.append("capability_environment_mismatch")
        cann_home = environment.get("cann_home")
        if (
            not isinstance(cann_home, str)
            or os.path.realpath(cann_home) != os.path.realpath(identity.cann_home)
        ):
            issues.append("capability_environment_mismatch")
        module_path = environment.get("module_path")
        if (
            not isinstance(module_path, str)
            or os.path.realpath(module_path) != os.path.realpath(identity.module_path)
        ):
            issues.append("capability_module_path_mismatch")
    return tuple(sorted(set(issues)))


def _runtime_attestation_record(
    identity_value: Optional[Mapping[str, Any]],
    capability_receipt: Optional[Mapping[str, Any]],
    issues: Sequence[str],
) -> Mapping[str, Any]:
    normalized_issues = tuple(sorted(set(str(item) for item in issues if item)))
    verified = not normalized_issues
    identity_hash = (
        identity_value.get("identity_sha256")
        if isinstance(identity_value, Mapping)
        else None
    )
    capability_hash = (
        _sha256(_json_bytes(capability_receipt))
        if isinstance(capability_receipt, Mapping)
        else None
    )
    return {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "verified_strict_npu" if verified else "audit_required",
        "verified": verified,
        "required_ops": sorted(CONSTITUENT_KEYS),
        "runtime_identity": dict(identity_value) if identity_value is not None else None,
        "runtime_identity_sha256": identity_hash,
        "capability_receipt": (
            dict(capability_receipt) if capability_receipt is not None else None
        ),
        "capability_receipt_sha256": capability_hash,
        "issues": list(normalized_issues),
    }


def _prepare_runtime_attestation(asnumpy_module: Any) -> Tuple[Mapping[str, Any], Any]:
    """Prepare an existing NumPilot strict-NPU guard for the two constituents."""

    identity_value: Optional[Mapping[str, Any]] = None
    capability_receipt: Optional[Mapping[str, Any]] = None
    guard = None
    try:
        from numpilot.backends.npu import NpuBackend

        backend = NpuBackend(module=asnumpy_module)
        identity = backend.runtime_identity()
        identity_value = identity.to_dict()
    except Exception:
        return (
            _runtime_attestation_record(
                identity_value, capability_receipt, ("runtime_identity_unavailable",)
            ),
            None,
        )
    if identity.status != "matched":
        return (
            _runtime_attestation_record(
                identity_value,
                capability_receipt,
                ("runtime_identity_not_matched",),
            ),
            None,
        )
    try:
        guard = backend.prepare_profile_guard(CONSTITUENT_KEYS)
        capability_receipt = guard.to_dict()
    except Exception as exc:
        validation = getattr(exc, "validation", None)
        to_dict = getattr(validation, "to_dict", None)
        if callable(to_dict):
            try:
                candidate = to_dict()
            except Exception:
                candidate = None
            if isinstance(candidate, Mapping):
                capability_receipt = candidate
        return (
            _runtime_attestation_record(
                identity_value,
                capability_receipt,
                ("constituent_capability_unavailable",),
            ),
            None,
        )
    issues = _runtime_attestation_issues(identity_value, capability_receipt)
    record = _runtime_attestation_record(identity_value, capability_receipt, issues)
    return record, guard if record.get("verified") is True else None


def _validate_runtime_attestation(
    value: object, *, require_verified: bool
) -> Tuple[str, Optional[str]]:
    """Validate a worker attestation and return its two stable hashes."""

    fields = {
        "schema_version",
        "status",
        "verified",
        "required_ops",
        "runtime_identity",
        "runtime_identity_sha256",
        "capability_receipt",
        "capability_receipt_sha256",
        "issues",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime attestation contract is invalid"
        )
    issues = value.get("issues")
    if (
        value.get("schema_version") != RUNTIME_ATTESTATION_SCHEMA_VERSION
        or value.get("required_ops") != sorted(CONSTITUENT_KEYS)
        or not isinstance(issues, list)
        or any(not isinstance(item, str) or not item for item in issues)
    ):
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime attestation contract is invalid"
        )
    identity_value = value.get("runtime_identity")
    capability_receipt = value.get("capability_receipt")
    if identity_value is None:
        if value.get("runtime_identity_sha256") is not None:
            raise RuntimeExtensionError(
                "invalid_worker_evidence", "runtime attestation identity hash is invalid"
            )
        identity_hash = ""
    else:
        try:
            from numpilot.capabilities import RuntimeIdentity

            identity = RuntimeIdentity.from_dict(identity_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeExtensionError(
                "invalid_worker_evidence", "runtime attestation identity is invalid"
            ) from exc
        identity_hash = identity.identity_sha256
        if value.get("runtime_identity_sha256") != identity_hash:
            raise RuntimeExtensionError(
                "invalid_worker_evidence", "runtime attestation identity hash changed"
            )
    if capability_receipt is None:
        if value.get("capability_receipt_sha256") is not None:
            raise RuntimeExtensionError(
                "invalid_worker_evidence", "runtime capability hash is invalid"
            )
        capability_hash = None
    elif isinstance(capability_receipt, Mapping):
        capability_hash = _sha256(_json_bytes(capability_receipt))
        if value.get("capability_receipt_sha256") != capability_hash:
            raise RuntimeExtensionError(
                "invalid_worker_evidence", "runtime capability hash changed"
            )
    else:
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime capability receipt is invalid"
        )
    if require_verified:
        if (
            value.get("status") != "verified_strict_npu"
            or value.get("verified") is not True
            or issues
            or not identity_hash
            or capability_hash is None
            or _runtime_attestation_issues(identity_value, capability_receipt)
        ):
            raise RuntimeExtensionError(
                "invalid_worker_evidence", "runtime is not proven strict NPU"
            )
    elif (
        value.get("status") != "audit_required"
        or value.get("verified") is not False
        or not issues
    ):
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime downgrade evidence is invalid"
        )
    return identity_hash, capability_hash


def _read_regular(path: Path, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeExtensionError("invalid_input", "%s must be a regular file" % label)
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
            raise RuntimeExtensionError("invalid_input", "%s is empty or oversized" % label)
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(payload) > limit:
            raise RuntimeExtensionError("invalid_input", "%s is oversized" % label)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeExtensionError("input_changed", "%s changed while being read" % label)
        return payload
    except RuntimeExtensionError:
        raise
    except OSError as exc:
        raise RuntimeExtensionError("invalid_input", "cannot read %s" % label) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    payload = _read_regular(path, _MAX_JSON_BYTES, label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeExtensionError("invalid_input", "%s is not valid JSON" % label) from exc
    if not isinstance(value, dict):
        raise RuntimeExtensionError("invalid_input", "%s must contain an object" % label)
    return value


def _new_private_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeExtensionError("invalid_output", "%s must be absolute" % label)
    if os.path.lexists(path):
        raise RuntimeExtensionError("output_exists", "%s must not exist" % label)
    try:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise RuntimeExtensionError("invalid_output", "%s parent is unsafe" % label)
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
    except RuntimeExtensionError:
        raise
    except OSError as exc:
        raise RuntimeExtensionError("invalid_output", "cannot create %s" % label) from exc
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(str(path), flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeExtensionError("write_failed", "cannot publish runtime extension file") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_cannbot_draft(
    report_path: Path, draft_path: Path
) -> Tuple[Mapping[str, Any], bytes, bytes]:
    report_payload = _read_regular(report_path, _MAX_JSON_BYTES, "CANNBot report")
    try:
        report = json.loads(report_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeExtensionError(
            "invalid_cannbot_report", "CANNBot report is not valid JSON"
        ) from exc
    if not isinstance(report, dict):
        raise RuntimeExtensionError(
            "invalid_cannbot_report", "CANNBot report must contain an object"
        )
    claims = report.get("claims")
    skill_gate = report.get("cannbot_skill_gate")
    draft_kind = report.get("draft_kind")
    if (
        report.get("status") != "generated_unverified"
        or report.get("candidate_status") != "draft"
        or report.get("generation_route") != "cannbot_registry_invoke_mini_gap"
        or report.get("cannbot_route") is not True
        or report.get("actual_remote_call") is not True
        or not isinstance(claims, dict)
        or claims.get("model_generated") is not True
        or any(
            claims.get(name) is not False
            for name in (
                "compiled",
                "unit_tested",
                "numerically_verified",
                "activated",
                "strict_npu_performance_verified",
            )
        )
        or not isinstance(skill_gate, dict)
        or skill_gate.get("status") != "pass"
        or skill_gate.get("content_embedded_for_remote_model") is not True
        or draft_kind not in (None, "ascendc_kernel_fragment")
        or (
            draft_kind == "ascendc_kernel_fragment"
            and (
                report.get("operator_language") != "ascend_c"
                or report.get("standalone_compilation_unit") is not False
            )
        )
    ):
        raise RuntimeExtensionError(
            "invalid_cannbot_report",
            "report is not a receipt-bound generated_unverified CANNBot add_one draft",
        )
    draft_payload = _read_regular(draft_path, _MAX_DRAFT_BYTES, "CANNBot draft")
    expected_name = report.get("generated_file")
    expected_hash = report.get("generated_file_sha256")
    if (
        not isinstance(expected_name, str)
        or Path(expected_name).name != expected_name
        or report_path.parent / expected_name != draft_path
        or not isinstance(expected_hash, str)
        or not _SHA256.fullmatch(expected_hash)
        or _sha256(draft_payload) != expected_hash
    ):
        raise RuntimeExtensionError(
            "cannbot_binding_mismatch", "CANNBot report and draft do not match"
        )
    try:
        draft_text = draft_payload.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeExtensionError(
            "invalid_cannbot_draft", "CANNBot draft is not UTF-8"
        ) from exc
    legacy_scalar = re.search(
        r"\b(?:DemoAddOne|add_one)\s*\(\s*float\s+\w+\s*\)", draft_text
    )
    ascendc_helper = re.search(
        r"\b(?:__aicore__\s+inline|inline\s+__aicore__)\s+void\s+"
        r"add_one\s*\(",
        draft_text,
    )
    ascendc_adds = re.findall(
        r"\bAscendC::Adds\s*\((.*?)\)\s*;", draft_text, flags=re.DOTALL
    )
    valid_ascendc = (
        ascendc_helper is not None
        and draft_text.count("AscendC::LocalTensor<float>") >= 2
        and re.search(r"\buint32_t\s+count\b", draft_text) is not None
        and len(ascendc_adds) == 1
        and re.search(
            r"(?m)^\s*#\s*include\b|\b(?:class|__global__|main)\b", draft_text
        )
        is None
        and legacy_scalar is None
    )
    if valid_ascendc:
        arguments = [item.strip() for item in ascendc_adds[0].split(",")]
        valid_ascendc = (
            len(arguments) == 4
            and re.fullmatch(r"1(?:\.0*)?[fF]?", arguments[2]) is not None
            and arguments[3] == "count"
        )
    if (
        "UNVERIFIED_DRAFT" not in draft_text
        or (
            draft_kind == "ascendc_kernel_fragment"
            and not valid_ascendc
        )
        or (draft_kind is None and legacy_scalar is None)
    ):
        raise RuntimeExtensionError(
            "invalid_cannbot_draft", "CANNBot draft is not the bounded add_one draft"
        )
    return report, report_payload, draft_payload


def materialize_add_one_extension(
    cannbot_report: os.PathLike[str] | str,
    cannbot_draft: os.PathLike[str] | str,
    out: os.PathLike[str] | str,
) -> MaterializedRuntimeExtension:
    """Materialize an unqualified, run-scoped composite ``asnumpy.add_one``.

    The generated module is deterministic framework code.  The supplied model
    draft is only a hash-bound provenance input and is explicitly not executed.
    """

    report_path = Path(cannbot_report).absolute()
    draft_path = Path(cannbot_draft).absolute()
    report, report_payload, draft_payload = _validate_cannbot_draft(
        report_path, draft_path
    )
    root = _new_private_directory(Path(out).absolute(), "extension output")
    module_payload = _EXTENSION_SOURCE.encode("utf-8")
    module_path = root / EXTENSION_FILENAME
    _write_exclusive(module_path, module_payload)
    module_sha = _sha256(module_payload)
    manifest = {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "record_type": "numpilot_run_scoped_runtime_extension",
        "status": "materialized_unqualified",
        "interface": INTERFACE,
        "implementation": "asnumpy_composite",
        "provenance": PROVENANCE,
        "scope": SCOPE,
        "constituents": list(CONSTITUENTS),
        "constraints": list(CONSTRAINTS),
        "extension_module": EXTENSION_FILENAME,
        "extension_module_sha256": module_sha,
        "cannbot": {
            "generation_route": report["generation_route"],
            "draft_kind": report.get("draft_kind", "legacy_scalar_cpp_function"),
            "operator_language": report.get("operator_language", "cpp"),
            "report_path": str(report_path),
            "report_sha256": _sha256(report_payload),
            "draft_path": str(draft_path),
            "draft_sha256": _sha256(draft_payload),
            "draft_status": "generated_unverified",
        },
        "interface_available": False,
        "upstream_modified": False,
        "cannbot_draft_activated": False,
        "claims": {
            "materialized": True,
            "numerically_verified": False,
            "npu_execution_verified": False,
            "performance_verified": False,
        },
    }
    manifest_payload = _json_bytes(manifest)
    manifest_path = root / MANIFEST_FILENAME
    _write_exclusive(manifest_path, manifest_payload)
    return MaterializedRuntimeExtension(
        root=str(root),
        module_path=str(module_path),
        module_sha256=module_sha,
        manifest_path=str(manifest_path),
        manifest_sha256=_sha256(manifest_payload),
        cannbot_report_path=str(report_path),
        cannbot_report_sha256=_sha256(report_payload),
        cannbot_draft_path=str(draft_path),
        cannbot_draft_sha256=_sha256(draft_payload),
    )


def load_materialized_add_one_extension(
    manifest_path: os.PathLike[str] | str,
) -> MaterializedRuntimeExtension:
    """Reload and fully revalidate a materialized extension receipt."""

    path = Path(manifest_path).absolute()
    payload = _read_regular(path, _MAX_JSON_BYTES, "runtime extension manifest")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeExtensionError(
            "invalid_manifest", "runtime extension manifest is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeExtensionError("invalid_manifest", "runtime extension manifest is invalid")
    cannbot = value.get("cannbot")
    claims = value.get("claims")
    if (
        value.get("schema_version") != EXTENSION_SCHEMA_VERSION
        or value.get("record_type") != "numpilot_run_scoped_runtime_extension"
        or value.get("status") != "materialized_unqualified"
        or value.get("interface") != INTERFACE
        or value.get("implementation") != "asnumpy_composite"
        or value.get("provenance") != PROVENANCE
        or value.get("scope") != SCOPE
        or value.get("constituents") != list(CONSTITUENTS)
        or value.get("constraints") != list(CONSTRAINTS)
        or value.get("extension_module") != EXTENSION_FILENAME
        or value.get("interface_available") is not False
        or value.get("upstream_modified") is not False
        or value.get("cannbot_draft_activated") is not False
        or not isinstance(claims, dict)
        or claims.get("materialized") is not True
        or claims.get("numerically_verified") is not False
        or claims.get("npu_execution_verified") is not False
        or claims.get("performance_verified") is not False
        or not isinstance(cannbot, dict)
        or cannbot.get("draft_status") != "generated_unverified"
    ):
        raise RuntimeExtensionError("invalid_manifest", "runtime extension manifest contract is invalid")
    module_path = path.parent / EXTENSION_FILENAME
    module_payload = _read_regular(module_path, _MAX_MODULE_BYTES, "runtime extension module")
    module_sha = _sha256(module_payload)
    if module_sha != value.get("extension_module_sha256"):
        raise RuntimeExtensionError("extension_changed", "runtime extension module hash mismatch")
    report_path = Path(str(cannbot.get("report_path", ""))).absolute()
    draft_path = Path(str(cannbot.get("draft_path", ""))).absolute()
    _, report_payload, draft_payload = _validate_cannbot_draft(report_path, draft_path)
    if (
        _sha256(report_payload) != cannbot.get("report_sha256")
        or _sha256(draft_payload) != cannbot.get("draft_sha256")
    ):
        raise RuntimeExtensionError("cannbot_binding_mismatch", "CANNBot inputs changed")
    return MaterializedRuntimeExtension(
        root=str(path.parent),
        module_path=str(module_path),
        module_sha256=module_sha,
        manifest_path=str(path),
        manifest_sha256=_sha256(payload),
        cannbot_report_path=str(report_path),
        cannbot_report_sha256=_sha256(report_payload),
        cannbot_draft_path=str(draft_path),
        cannbot_draft_sha256=_sha256(draft_payload),
    )


def _qualification_contract(
    extension: MaterializedRuntimeExtension, evidence_path: Path, nonce: str
) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "numpilot_runtime_extension_worker_contract",
        "nonce": nonce,
        "manifest_path": extension.manifest_path,
        "manifest_sha256": extension.manifest_sha256,
        "extension_module_path": extension.module_path,
        "extension_module_sha256": extension.module_sha256,
        "evidence_path": str(evidence_path),
        "interface": INTERFACE,
        "provenance": PROVENANCE,
        "scope": SCOPE,
        "constraints": list(CONSTRAINTS),
    }


def _sanitize_worker_environment(home: Path) -> Mapping[str, str]:
    environment = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if (
            upper.startswith(("NUMPILOT_LLM_", "OPENAI_", "ANTHROPIC_"))
            or any(marker in upper for marker in ("API_KEY", "AUTH_TOKEN", "PASSWORD", "SECRET"))
            or upper.endswith(("_TOKEN", "_CREDENTIALS"))
        ):
            continue
        environment[key] = value
    environment.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return MappingProxyType(environment)


def _default_runner(invocation: QualificationInvocation) -> RunnerObservation:
    home = Path(invocation.cwd) / "home"
    home.mkdir(mode=0o700)
    # HOME is intentionally replaced for the worker, but the audited runtime
    # identity may rely on the caller's private bootstrap receipt.  Freeze that
    # exact path before sanitizing HOME so the read-only identity verifier sees
    # the same receipt without exposing any model credential or user cache.
    try:
        from numpilot.paths import default_bootstrap_receipt

        bootstrap_receipt = str(default_bootstrap_receipt().absolute())
    except Exception:
        bootstrap_receipt = ""
    environment = dict(_sanitize_worker_environment(home))
    if bootstrap_receipt and "NUMPILOT_BOOTSTRAP_RECEIPT" not in environment:
        environment["NUMPILOT_BOOTSTRAP_RECEIPT"] = bootstrap_receipt
    process: Optional[subprocess.Popen[Any]] = None
    try:
        process = subprocess.Popen(
            invocation.command,
            cwd=invocation.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        try:
            process.wait(timeout=invocation.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            return RunnerObservation(returncode=-1, timed_out=True, error_type="TimeoutExpired")
    except OSError as exc:
        return RunnerObservation(returncode=-1, error_type=type(exc).__name__)
    if process is None:
        return RunnerObservation(returncode=-1, error_type="ProcessUnavailable")
    evidence = None
    if process.returncode == 0:
        try:
            evidence = _read_json(Path(invocation.evidence_path), "worker evidence")
        except RuntimeExtensionError:
            evidence = None
    return RunnerObservation(returncode=int(process.returncode), evidence=evidence)


def _validate_worker_evidence(
    evidence: Mapping[str, Any], *, contract_sha256: str, extension: MaterializedRuntimeExtension
) -> Tuple[int, float, str, str]:
    counts = evidence.get("constituent_dispatches")
    max_abs = evidence.get("max_abs_error")
    cases = evidence.get("numerical_cases")
    before = evidence.get("runtime_attestation_before")
    after = evidence.get("runtime_attestation_after")
    if (
        evidence.get("schema_version") != WORKER_SCHEMA_VERSION
        or evidence.get("record_type") != "numpilot_runtime_extension_worker_evidence"
        or evidence.get("status") != "passed"
        or evidence.get("contract_sha256") != contract_sha256
        or evidence.get("manifest_sha256") != extension.manifest_sha256
        or evidence.get("extension_module_sha256") != extension.module_sha256
        or evidence.get("interface") != INTERFACE
        or evidence.get("interface_available") is not True
        or evidence.get("provenance") != PROVENANCE
        or evidence.get("scope") != SCOPE
        or evidence.get("upstream_modified") is not False
        or evidence.get("cannbot_draft_activated") is not False
        or evidence.get("backend") != "npu"
        or evidence.get("npu_execution_performed") is not True
        or evidence.get("no_fallback_provenance_verified") is not True
        or evidence.get("runtime_identity_stable") is not True
        or evidence.get("result_type") != "asnumpy.ndarray"
        or evidence.get("constraints") != list(CONSTRAINTS)
        or not isinstance(counts, dict)
        or set(counts) != set(CONSTITUENTS)
        or any(
            isinstance(counts[name], bool)
            or not isinstance(counts[name], int)
            or counts[name] < 1
            for name in CONSTITUENTS
        )
        or isinstance(cases, bool)
        or not isinstance(cases, int)
        or cases < 1
        or isinstance(max_abs, bool)
        or not isinstance(max_abs, (int, float))
        or not math.isfinite(float(max_abs))
        or float(max_abs) < 0.0
    ):
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime extension worker evidence is invalid"
        )
    before_identity, before_capability = _validate_runtime_attestation(
        before, require_verified=True
    )
    after_identity, after_capability = _validate_runtime_attestation(
        after, require_verified=True
    )
    if (
        before_identity != after_identity
        or before_capability != after_capability
        or before_capability is None
    ):
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime attestation changed during execution"
        )
    return cases, float(max_abs), after_identity, after_capability


def _validate_runtime_audit_required_evidence(
    evidence: Mapping[str, Any], *, contract_sha256: str, extension: MaterializedRuntimeExtension
) -> Tuple[str, Optional[str]]:
    """Accept a truthful non-execution record without promoting it to NPU."""

    counts = evidence.get("constituent_dispatches")
    if (
        evidence.get("schema_version") != WORKER_SCHEMA_VERSION
        or evidence.get("record_type") != "numpilot_runtime_extension_worker_evidence"
        or evidence.get("status") != "runtime_audit_required"
        or evidence.get("contract_sha256") != contract_sha256
        or evidence.get("manifest_sha256") != extension.manifest_sha256
        or evidence.get("extension_module_sha256") != extension.module_sha256
        or evidence.get("interface") != INTERFACE
        or evidence.get("interface_available") is not False
        or evidence.get("provenance") != PROVENANCE
        or evidence.get("scope") != SCOPE
        or evidence.get("upstream_modified") is not False
        or evidence.get("cannbot_draft_activated") is not False
        or evidence.get("backend") != "unverified"
        or evidence.get("npu_execution_performed") is not False
        or evidence.get("no_fallback_provenance_verified") is not False
        or evidence.get("runtime_identity_stable") is not False
        or evidence.get("result_type") is not None
        or evidence.get("constraints") != list(CONSTRAINTS)
        or not isinstance(counts, dict)
        or set(counts) != set(CONSTITUENTS)
        or any(type(counts[name]) is not int or counts[name] != 0 for name in CONSTITUENTS)
        or type(evidence.get("numerical_cases")) is not int
        or evidence.get("numerical_cases") != 0
        or evidence.get("max_abs_error") is not None
        or evidence.get("runtime_attestation_after") is not None
    ):
        raise RuntimeExtensionError(
            "invalid_worker_evidence", "runtime audit downgrade evidence is invalid"
        )
    return _validate_runtime_attestation(
        evidence.get("runtime_attestation_before"), require_verified=False
    )


def qualify_add_one_extension(
    extension: MaterializedRuntimeExtension | os.PathLike[str] | str,
    out: os.PathLike[str] | str,
    *,
    python_executable: os.PathLike[str] | str | None = None,
    timeout_seconds: float = 300.0,
    runner: Optional[Runner] = None,
) -> RuntimeExtensionQualification:
    """Qualify the composition in an isolated real-runtime subprocess.

    Supplying ``runner`` selects an explicitly non-qualifying test seam.  Even
    perfectly shaped injected evidence is recorded as ``test_evidence_only``
    with ``interface_available=false``.
    """

    if isinstance(extension, MaterializedRuntimeExtension):
        materialized = load_materialized_add_one_extension(extension.manifest_path)
        if materialized.manifest_sha256 != extension.manifest_sha256:
            raise RuntimeExtensionError("extension_changed", "materialized extension changed")
    else:
        materialized = load_materialized_add_one_extension(extension)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.05 <= float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS
    ):
        raise RuntimeExtensionError("invalid_timeout", "qualification timeout is invalid")
    py = Path(sys.executable if python_executable is None else python_executable).absolute()
    if py.is_symlink():
        resolved = py.resolve(strict=True)
    else:
        resolved = py
    if not resolved.is_file() or not os.access(py, os.X_OK):
        raise RuntimeExtensionError("invalid_python", "qualification Python is unavailable")

    root = _new_private_directory(Path(out).absolute(), "qualification output")
    evidence_path = root / WORKER_EVIDENCE_FILENAME
    nonce = secrets.token_hex(16)
    contract = _qualification_contract(materialized, evidence_path, nonce)
    contract_payload = _json_bytes(contract)
    contract_path = root / "worker-contract.json"
    _write_exclusive(contract_path, contract_payload)
    contract_sha = _sha256(contract_payload)
    invocation = QualificationInvocation(
        command=(
            str(py),
            "-I",
            str(Path(__file__).resolve()),
            "--qualification-worker",
            str(contract_path),
        ),
        cwd=str(root),
        timeout_seconds=float(timeout_seconds),
        contract_path=str(contract_path),
        contract_sha256=contract_sha,
        evidence_path=str(evidence_path),
    )
    selected_runner = _default_runner if runner is None else runner
    trusted_execution = runner is None
    try:
        observation = selected_runner(invocation)
    except Exception as exc:
        observation = RunnerObservation(returncode=-1, error_type=type(exc).__name__)
    if not isinstance(observation, RunnerObservation):
        observation = RunnerObservation(returncode=-1, error_type="InvalidRunnerResult")

    cases = 0
    max_abs_error: Optional[float] = None
    evidence_valid = False
    audit_required_evidence_valid = False
    runtime_identity_sha256 = ""
    capability_receipt_sha256: Optional[str] = None
    if observation.returncode == 0 and not observation.timed_out and isinstance(
        observation.evidence, Mapping
    ):
        try:
            if observation.evidence.get("status") == "passed":
                (
                    cases,
                    max_abs_error,
                    runtime_identity_sha256,
                    capability_receipt_sha256,
                ) = _validate_worker_evidence(
                    observation.evidence,
                    contract_sha256=contract_sha,
                    extension=materialized,
                )
                evidence_valid = True
            elif observation.evidence.get("status") == "runtime_audit_required":
                (
                    runtime_identity_sha256,
                    capability_receipt_sha256,
                ) = _validate_runtime_audit_required_evidence(
                    observation.evidence,
                    contract_sha256=contract_sha,
                    extension=materialized,
                )
                audit_required_evidence_valid = True
        except RuntimeExtensionError:
            evidence_valid = False
            audit_required_evidence_valid = False
            cases = 0
            max_abs_error = None
            runtime_identity_sha256 = ""
            capability_receipt_sha256 = None

    interface_available = bool(trusted_execution and evidence_valid)
    status = (
        "qualified_for_run"
        if interface_available
        else "runtime_audit_required"
        if trusted_execution and audit_required_evidence_valid
        else "test_evidence_only"
        if (evidence_valid or audit_required_evidence_valid) and not trusted_execution
        else "failed"
    )
    trusted_evidence_payload = None
    trusted_evidence_sha256 = None
    trusted_record_valid = bool(
        trusted_execution and (evidence_valid or audit_required_evidence_valid)
    )
    if trusted_record_valid:
        try:
            trusted_evidence_payload = _read_regular(
                evidence_path, _MAX_JSON_BYTES, "worker evidence"
            )
            trusted_evidence_sha256 = _sha256(trusted_evidence_payload)
        except RuntimeExtensionError:
            interface_available = False
            status = "failed"
            evidence_valid = False
            audit_required_evidence_valid = False
            trusted_record_valid = False
            cases = 0
            max_abs_error = None
            runtime_identity_sha256 = ""
            capability_receipt_sha256 = None
    report = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "record_type": "numpilot_runtime_extension_qualification",
        "status": status,
        "interface": INTERFACE,
        "interface_available": interface_available,
        "provenance": PROVENANCE,
        "scope": SCOPE,
        "constituents": list(CONSTITUENTS),
        "constraints": list(CONSTRAINTS),
        "upstream_modified": False,
        "cannbot_draft_activated": False,
        "manifest_path": materialized.manifest_path,
        "manifest_sha256": materialized.manifest_sha256,
        "extension_module_path": materialized.module_path,
        "extension_module_sha256": materialized.module_sha256,
        "cannbot_report_sha256": materialized.cannbot_report_sha256,
        "cannbot_draft_sha256": materialized.cannbot_draft_sha256,
        "runner": "builtin_subprocess" if trusted_execution else "injected_test_runner",
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "worker_evidence_path": str(evidence_path) if trusted_record_valid else None,
        "worker_evidence_sha256": trusted_evidence_sha256,
        "subprocess_returncode": observation.returncode,
        "timed_out": bool(observation.timed_out),
        "error_type": observation.error_type,
        "worker_evidence_valid": evidence_valid,
        "runtime_audit_evidence_valid": audit_required_evidence_valid,
        "runtime_identity_status": (
            "matched"
            if interface_available
            else "audit_required"
            if audit_required_evidence_valid
            else "unavailable"
        ),
        "runtime_identity_sha256": runtime_identity_sha256 or None,
        "constituent_capability_receipt_sha256": capability_receipt_sha256,
        "npu_execution_performed": interface_available,
        "no_fallback_provenance_verified": interface_available,
        "numerical_cases": cases if interface_available else 0,
        "max_abs_error": max_abs_error if interface_available else None,
        "claims": {
            "numerically_verified": interface_available,
            "npu_execution_verified": interface_available,
            "performance_verified": False,
            "cannbot_draft_compiled": False,
            "cannbot_draft_activated": False,
        },
    }
    report_payload = _json_bytes(report)
    report_path = root / QUALIFICATION_FILENAME
    _write_exclusive(report_path, report_payload)
    return RuntimeExtensionQualification(
        report_path=str(report_path),
        report_sha256=_sha256(report_payload),
        status=status,
        interface_available=interface_available,
        provenance=PROVENANCE,
        scope=SCOPE,
        upstream_modified=False,
        cannbot_draft_activated=False,
        numerical_cases=cases if interface_available else 0,
        max_abs_error=max_abs_error if interface_available else None,
    )


def activate_qualified_add_one_extension(
    qualification_report: os.PathLike[str] | str, asnumpy_module: Any
) -> Callable[..., Any]:
    """Activate a qualified composite interface on one AsNumpy module object."""

    report_path = Path(qualification_report).absolute()
    report = _read_json(report_path, "runtime extension qualification")
    claims = report.get("claims")
    if (
        report.get("schema_version") != QUALIFICATION_SCHEMA_VERSION
        or report.get("record_type") != "numpilot_runtime_extension_qualification"
        or report.get("status") != "qualified_for_run"
        or report.get("interface") != INTERFACE
        or report.get("interface_available") is not True
        or report.get("provenance") != PROVENANCE
        or report.get("scope") != SCOPE
        or report.get("upstream_modified") is not False
        or report.get("cannbot_draft_activated") is not False
        or report.get("runner") != "builtin_subprocess"
        or report.get("worker_evidence_valid") is not True
        or report.get("runtime_audit_evidence_valid") is not False
        or report.get("runtime_identity_status") != "matched"
        or not isinstance(report.get("runtime_identity_sha256"), str)
        or not _SHA256.fullmatch(report["runtime_identity_sha256"])
        or not isinstance(
            report.get("constituent_capability_receipt_sha256"), str
        )
        or not _SHA256.fullmatch(
            report["constituent_capability_receipt_sha256"]
        )
        or report.get("npu_execution_performed") is not True
        or report.get("no_fallback_provenance_verified") is not True
        or report.get("subprocess_returncode") != 0
        or report.get("timed_out") is not False
        or not isinstance(claims, dict)
        or claims.get("numerically_verified") is not True
        or claims.get("npu_execution_verified") is not True
        or claims.get("performance_verified") is not False
        or claims.get("cannbot_draft_compiled") is not False
        or claims.get("cannbot_draft_activated") is not False
    ):
        raise RuntimeExtensionError(
            "not_qualified", "runtime extension is not qualified for activation"
        )
    manifest_path = Path(str(report.get("manifest_path", ""))).absolute()
    materialized = load_materialized_add_one_extension(manifest_path)
    if (
        materialized.manifest_sha256 != report.get("manifest_sha256")
        or materialized.module_sha256 != report.get("extension_module_sha256")
        or materialized.cannbot_report_sha256 != report.get("cannbot_report_sha256")
        or materialized.cannbot_draft_sha256 != report.get("cannbot_draft_sha256")
    ):
        raise RuntimeExtensionError("extension_changed", "qualified extension inputs changed")
    contract_path = Path(str(report.get("contract_path", ""))).absolute()
    contract_payload = _read_regular(
        contract_path, _MAX_JSON_BYTES, "qualification worker contract"
    )
    contract_sha = _sha256(contract_payload)
    evidence_path = Path(str(report.get("worker_evidence_path", ""))).absolute()
    evidence_payload = _read_regular(
        evidence_path, _MAX_JSON_BYTES, "qualification worker evidence"
    )
    if (
        contract_sha != report.get("contract_sha256")
        or _sha256(evidence_payload) != report.get("worker_evidence_sha256")
    ):
        raise RuntimeExtensionError(
            "qualification_changed", "qualification subprocess evidence changed"
        )
    try:
        evidence = json.loads(evidence_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeExtensionError(
            "qualification_changed", "qualification subprocess evidence is invalid"
        ) from exc
    if not isinstance(evidence, dict):
        raise RuntimeExtensionError(
            "qualification_changed", "qualification subprocess evidence is invalid"
        )
    cases, max_abs, identity_hash, capability_hash = _validate_worker_evidence(
        evidence,
        contract_sha256=contract_sha,
        extension=materialized,
    )
    if (
        identity_hash != report.get("runtime_identity_sha256")
        or capability_hash
        != report.get("constituent_capability_receipt_sha256")
        or report.get("numerical_cases") != cases
        or isinstance(report.get("max_abs_error"), bool)
        or not isinstance(report.get("max_abs_error"), (int, float))
        or float(report["max_abs_error"]) != max_abs
    ):
        raise RuntimeExtensionError(
            "qualification_changed", "qualification attestation binding changed"
        )

    # Re-attest the supplied module immediately before activation.  A passed
    # record for some other AsNumpy build, CANN installation, SoC or callable
    # snapshot must never authorize this module object.
    live_attestation, live_guard = _prepare_runtime_attestation(asnumpy_module)
    live_identity, live_capability = _validate_runtime_attestation(
        live_attestation, require_verified=True
    )
    if (
        live_guard is None
        or live_identity != identity_hash
        or live_capability != capability_hash
    ):
        raise RuntimeExtensionError(
            "runtime_changed", "qualified strict-NPU runtime identity changed"
        )
    spec = importlib.util.spec_from_file_location(EXTENSION_MODULE_NAME, materialized.module_path)
    if spec is None or spec.loader is None:
        raise RuntimeExtensionError("activation_failed", "cannot load runtime extension module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        activate = getattr(module, "activate")
        interface = activate(asnumpy_module)
    except Exception as exc:
        raise RuntimeExtensionError("activation_failed", "runtime extension activation failed") from exc
    if not callable(interface) or asnumpy_module.__dict__.get("add_one") is not interface:
        raise RuntimeExtensionError("activation_failed", "runtime extension interface is unavailable")
    return interface


def _publish_worker_evidence(
    contract_path: Path,
    evidence_path: Path,
    evidence: Mapping[str, Any],
) -> bool:
    """Create the bound worker record without following or replacing a path."""

    if evidence_path.parent != contract_path.parent or os.path.lexists(evidence_path):
        return False
    try:
        _write_exclusive(evidence_path, _json_bytes(evidence))
    except RuntimeExtensionError:
        return False
    return True


def _runtime_audit_required_worker_evidence(
    *,
    contract_payload: bytes,
    materialized: MaterializedRuntimeExtension,
    attestation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build a truthful zero-execution downgrade record."""

    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "record_type": "numpilot_runtime_extension_worker_evidence",
        "status": "runtime_audit_required",
        "contract_sha256": _sha256(contract_payload),
        "manifest_sha256": materialized.manifest_sha256,
        "extension_module_sha256": materialized.module_sha256,
        "interface": INTERFACE,
        "interface_available": False,
        "provenance": PROVENANCE,
        "scope": SCOPE,
        "constituent_dispatches": {name: 0 for name in CONSTITUENTS},
        "constraints": list(CONSTRAINTS),
        "backend": "unverified",
        "npu_execution_performed": False,
        "no_fallback_provenance_verified": False,
        "runtime_identity_stable": False,
        "runtime_attestation_before": dict(attestation),
        "runtime_attestation_after": None,
        "result_type": None,
        "numerical_cases": 0,
        "max_abs_error": None,
        "upstream_modified": False,
        "cannbot_draft_activated": False,
    }


def _qualification_worker(contract_path: Path) -> int:
    """Execute the numerical/NPU gate.  Called only through ``python -I``."""

    # Isolated mode intentionally ignores cwd, PYTHONPATH and the user site.
    # Admit only the package root containing this already selected worker so
    # its capability/identity verifier is the same NumPilot source as the
    # qualification entry point, including in a source checkout.
    package_root = str(Path(__file__).resolve().parent.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    contract_payload = _read_regular(contract_path, _MAX_JSON_BYTES, "worker contract")
    try:
        contract = json.loads(contract_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        return 2
    if not isinstance(contract, dict):
        return 2
    evidence_path = Path(str(contract.get("evidence_path", ""))).absolute()
    try:
        materialized = load_materialized_add_one_extension(contract["manifest_path"])
        if (
            materialized.manifest_sha256 != contract.get("manifest_sha256")
            or materialized.module_path != contract.get("extension_module_path")
            or materialized.module_sha256 != contract.get("extension_module_sha256")
            or contract.get("interface") != INTERFACE
            or contract.get("provenance") != PROVENANCE
            or contract.get("scope") != SCOPE
            or contract.get("constraints") != list(CONSTRAINTS)
        ):
            return 2

        import numpy as np
        try:
            import asnumpy
        except Exception:
            attestation = _runtime_attestation_record(
                None, None, ("runtime_identity_unavailable",)
            )
            evidence = _runtime_audit_required_worker_evidence(
                contract_payload=contract_payload,
                materialized=materialized,
                attestation=attestation,
            )
            return 0 if _publish_worker_evidence(
                contract_path, evidence_path, evidence
            ) else 2

        attestation_before, guard = _prepare_runtime_attestation(asnumpy)
        if guard is None:
            evidence = _runtime_audit_required_worker_evidence(
                contract_payload=contract_payload,
                materialized=materialized,
                attestation=attestation_before,
            )
            return 0 if _publish_worker_evidence(
                contract_path, evidence_path, evidence
            ) else 2

        runtime_targets = guard.runtime_targets
        if (
            not isinstance(runtime_targets, Mapping)
            or set(runtime_targets) != set(CONSTITUENT_KEYS)
            or any(not callable(runtime_targets[name]) for name in CONSTITUENT_KEYS)
        ):
            attestation = _runtime_attestation_record(
                attestation_before.get("runtime_identity"),
                attestation_before.get("capability_receipt"),
                ("runtime_target_snapshot_unavailable",),
            )
            evidence = _runtime_audit_required_worker_evidence(
                contract_payload=contract_payload,
                materialized=materialized,
                attestation=attestation,
            )
            return 0 if _publish_worker_evidence(
                contract_path, evidence_path, evidence
            ) else 2

        dispatches = {CONSTITUENTS[0]: 0, CONSTITUENTS[1]: 0}

        def counted_ones_like(*args: Any, **kwargs: Any) -> Any:
            guard.validate_call("ones_like", args, kwargs)
            dispatches[CONSTITUENTS[0]] += 1
            return runtime_targets["ones_like"](*args, **kwargs)

        def counted_add(*args: Any, **kwargs: Any) -> Any:
            guard.validate_call("add", args, kwargs)
            dispatches[CONSTITUENTS[1]] += 1
            return runtime_targets["add"](*args, **kwargs)

        # Qualify against a private proxy.  The installed module is never
        # monkey-patched during measurement, so the guard can continuously
        # verify the exact audited callables captured above.
        class _QualificationProxy:
            pass

        proxy = _QualificationProxy()
        proxy.ones_like = counted_ones_like
        proxy.add = counted_add
        spec = importlib.util.spec_from_file_location(EXTENSION_MODULE_NAME, materialized.module_path)
        if spec is None or spec.loader is None:
            return 2
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        interface = module.activate(proxy)
        if proxy.__dict__.get("add_one") is not interface:
            return 2

        cases = (
            np.array([-2.0, 0.0, 3.5], dtype=np.float32),
            np.linspace(-8.0, 8.0, 257, dtype=np.float32),
            np.linspace(-3.0, 5.0, 1024, dtype=np.float32).reshape(32, 32),
        )
        max_abs = 0.0
        for host in cases:
            device = asnumpy.ndarray.from_numpy(np.ascontiguousarray(host))
            actual = interface(device)
            if type(actual) is not asnumpy.ndarray:
                return 2
            result = actual.to_numpy()
            expected = np.add(host, np.float32(1.0))
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
            max_abs = max(
                max_abs,
                float(np.max(np.abs(result.astype(np.float64) - expected.astype(np.float64)))),
            )
        attestation_after, after_guard = _prepare_runtime_attestation(asnumpy)
        before_identity, before_capability = _validate_runtime_attestation(
            attestation_before, require_verified=True
        )
        after_identity, after_capability = _validate_runtime_attestation(
            attestation_after, require_verified=True
        )
        if (
            after_guard is None
            or before_identity != after_identity
            or before_capability != after_capability
            or before_capability is None
        ):
            return 1
        evidence = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "record_type": "numpilot_runtime_extension_worker_evidence",
            "status": "passed",
            "contract_sha256": _sha256(contract_payload),
            "manifest_sha256": materialized.manifest_sha256,
            "extension_module_sha256": materialized.module_sha256,
            "interface": INTERFACE,
            "interface_available": True,
            "provenance": PROVENANCE,
            "scope": SCOPE,
            "constituent_dispatches": dispatches,
            "constraints": list(CONSTRAINTS),
            "backend": "npu",
            "npu_execution_performed": True,
            "no_fallback_provenance_verified": True,
            "runtime_identity_stable": True,
            "runtime_attestation_before": attestation_before,
            "runtime_attestation_after": attestation_after,
            "result_type": "asnumpy.ndarray",
            "numerical_cases": len(cases),
            "max_abs_error": max_abs,
            "upstream_modified": False,
            "cannbot_draft_activated": False,
        }
        return 0 if _publish_worker_evidence(
            contract_path, evidence_path, evidence
        ) else 2
    except Exception:
        return 1


def _main(argv: Tuple[str, ...]) -> int:
    if len(argv) == 2 and argv[0] == "--qualification-worker":
        return _qualification_worker(Path(argv[1]).absolute())
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(tuple(sys.argv[1:])))


__all__ = [
    "CONSTITUENTS",
    "CONSTRAINTS",
    "INTERFACE",
    "PROVENANCE",
    "SCOPE",
    "MaterializedRuntimeExtension",
    "QualificationInvocation",
    "RunnerObservation",
    "RuntimeExtensionError",
    "RuntimeExtensionQualification",
    "activate_qualified_add_one_extension",
    "load_materialized_add_one_extension",
    "materialize_add_one_extension",
    "qualify_add_one_extension",
]
