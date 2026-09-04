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

"""Registered, functionality-first recovery for ``np.add(array, 1)``.

This is intentionally a very small gap provider used by the interactive
migration flow.  The current pinned AsNumpy binding accepts two arrays for
``add`` and rejects a scalar second operand.  NumPilot can therefore recognize
the exact scalar-one call as an ``add_one`` contract, ask the configured model
to produce an isolated CANNBot/Ascend-C draft, and lower the *executable*
functionality-first version to the already qualified ``ones_like + add`` NPU
composition.  The generated custom-kernel draft remains unverified and is not
silently activated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional, Sequence

from numpilot import script_verification as _script_verification
from numpilot.comparison import compare_outputs


MAX_SOURCE_BYTES = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TARGET = re.compile(r"^[A-Za-z0-9_.+-]{1,128}$")
_TARGET_API_CONTRACT = "asnumpy.add_one"
_LOWERED_PROGRAM_APIS = ("numpy.ones_like", "numpy.add")
_EXPECTED_MIGRATED_PROGRAM_APIS = ("asnumpy.ones_like", "asnumpy.add")


class AddOneGapError(RuntimeError):
    """The registered add-one gap did not satisfy its narrow contract."""


class AddOneLoweringVerificationError(AddOneGapError):
    """The source-to-normalized numerical gate failed closed.

    ``receipt`` is populated only when all three inputs were admitted and the
    failure happened during execution, drift checking, or comparison.  Pure
    preflight failures do not publish a receipt.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        receipt: Optional[_script_verification.ScriptVerificationReceipt] = None,
    ) -> None:
        self.code = code
        self.receipt = receipt
        super().__init__(message)


@dataclass(frozen=True, order=True)
class AddOneCall:
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    input_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "end_lineno": self.end_lineno,
            "end_col_offset": self.end_col_offset,
            "input_name": self.input_name,
            "source_api": "numpy.add",
            "source_signature": "array,float_scalar_one",
            "target_api": "asnumpy.add_one",
        }


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _is_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and float(node.value) == 1.0
    )


def detect_add_one_calls(source: str) -> tuple[AddOneCall, ...]:
    """Detect only direct ``np.add(NAME, 1)`` calls without executing code."""

    if not isinstance(source, str):
        raise AddOneGapError("source must be text")
    try:
        encoded = source.encode("utf-8")
    except UnicodeError as exc:
        raise AddOneGapError("source must be UTF-8") from exc
    if len(encoded) > MAX_SOURCE_BYTES:
        raise AddOneGapError("source exceeds the add-one gap limit")
    try:
        tree = ast.parse(source, type_comments=True)
    except SyntaxError as exc:
        raise AddOneGapError("source has invalid Python syntax") from exc
    calls: list[AddOneCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            _attribute_path(node.func) != ("np", "add")
            or len(node.args) != 2
            or node.keywords
            or not isinstance(node.args[0], ast.Name)
            or not _is_one(node.args[1])
        ):
            continue
        location = (
            getattr(node, "lineno", None),
            getattr(node, "col_offset", None),
            getattr(node, "end_lineno", None),
            getattr(node, "end_col_offset", None),
        )
        if any(not isinstance(value, int) for value in location):
            raise AddOneGapError("add-one call has no stable AST location")
        calls.append(AddOneCall(*location, input_name=node.args[0].id))
    return tuple(sorted(calls))


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _atomic_private(path: Path, payload: bytes) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise AddOneGapError("output path must be fresh and absolute")
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise AddOneGapError("output parent is invalid")
        if os.name == "posix":
            os.chmod(path.parent, 0o700)
    except OSError as exc:
        raise AddOneGapError("output parent is unavailable") from exc
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent)
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = ""
        return path
    except OSError as exc:
        raise AddOneGapError("output publication failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _validated_target(value: str, label: str) -> str:
    rendered = str(value or "")
    if _SAFE_TARGET.fullmatch(rendered) is None:
        raise AddOneGapError("%s is invalid" % label)
    return rendered


def materialize_add_one_operator_spec(
    *,
    original_source_path: str,
    original_source: str,
    calls: Sequence[AddOneCall],
    out_dir: Path,
    target_soc: str,
    cann_version: str,
) -> tuple[Path, Path]:
    """Write the static OperatorSpec passed to the existing CANNBot resolver."""

    source_path = Path(original_source_path)
    if not source_path.is_absolute() or not calls:
        raise AddOneGapError("add-one operator contract requires a bound source")
    detected = detect_add_one_calls(original_source)
    if tuple(calls) != detected:
        raise AddOneGapError("add-one call list does not match the source")
    target = _validated_target(target_soc, "target SoC")
    cann = _validated_target(cann_version, "CANN version")
    original_sha = _source_hash(original_source)
    contract_source = '''# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
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

"""Generated inert OperatorSpec for a source-bound add-one gap."""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="add_one",
    version="1.elementwise",
    numpy_oracle="add_one_oracle",
    target_api="asnumpy.add_one",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=({"positional_args": 1, "keyword_args": ()},),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "same_shape", "contiguous"),
    source="numpilot://registered-gap/add-one",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc=%s,
    cann_version=%s,
)
def add_one_oracle(x):
    return np.add(x, np.float32(1.0))


def build_inputs():
    return (np.array([-2.0, 0.0, 3.5], dtype=np.float32),)


def forward(x):
    return add_one_oracle(x)
''' % (json.dumps(target), json.dumps(cann))
    root = Path(out_dir).absolute()
    contract_path = _atomic_private(
        root / "add-one-operator-spec.py", contract_source.encode("utf-8")
    )
    contract_sha = hashlib.sha256(contract_source.encode("utf-8")).hexdigest()
    relation = {
        "schema_version": 1,
        "record_type": "numpilot_registered_add_one_gap",
        "status": "operator_spec_materialized",
        "original_source_path": str(source_path),
        "original_source_sha256": original_sha,
        "operator_spec_path": str(contract_path),
        "operator_spec_sha256": contract_sha,
        "source_api": "numpy.add",
        "rejected_signature": "array,float_scalar_one",
        "target_api": "asnumpy.add_one",
        "semantic_contract": "float32 contiguous y = x + 1.0",
        "target_soc": target,
        "cann_version": cann,
        "callsites": [item.to_dict() for item in calls],
        "claims": {
            "source_gap_detected": True,
            "operator_spec_materialized": True,
            "operator_generated": False,
            "operator_compiled": False,
            "interface_available": False,
        },
    }
    relation_path = _atomic_private(
        root / "add-one-gap-relation.json",
        (json.dumps(relation, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return contract_path, relation_path


def _byte_offsets(source: str) -> tuple[list[bytes], list[int]]:
    lines = source.encode("utf-8").splitlines(keepends=True)
    if not lines:
        lines = [b""]
    starts: list[int] = []
    total = 0
    for line in lines:
        starts.append(total)
        total += len(line)
    return lines, starts


def _lower_add_one_payload(
    source: str, calls: Sequence[AddOneCall]
) -> bytes:
    """Return the exact registered lowering without touching the filesystem."""

    if tuple(calls) != detect_add_one_calls(source):
        raise AddOneGapError("add-one call list does not match the source")
    payload = source.encode("utf-8")
    _lines, starts = _byte_offsets(source)
    replacements: list[tuple[int, int, bytes]] = []
    for item in calls:
        try:
            start = starts[item.lineno - 1] + item.col_offset
            end = starts[item.end_lineno - 1] + item.end_col_offset
        except (IndexError, TypeError) as exc:
            raise AddOneGapError("add-one location is outside the source") from exc
        original = payload[start:end]
        expected_prefix = ("np.add(" + item.input_name + ",").encode("utf-8")
        if not original.replace(b" ", b"").startswith(
            expected_prefix.replace(b" ", b"")
        ):
            raise AddOneGapError("add-one source segment changed")
        replacement = (
            "np.add(%s, np.ones_like(%s, dtype=np.float32))"
            % (item.input_name, item.input_name)
        ).encode("utf-8")
        replacements.append((start, end, replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        payload = payload[:start] + replacement + payload[end:]
    try:
        normalized = payload.decode("utf-8")
        ast.parse(normalized, type_comments=True)
    except (UnicodeError, SyntaxError) as exc:
        raise AddOneGapError("lowered source is invalid") from exc
    if detect_add_one_calls(normalized):
        raise AddOneGapError("lowered source still contains scalar add-one calls")
    return payload


def _analyze_effective_script(source_path: str, source: str):
    """Run both inert ordinary-script analyzers over one effective source."""

    try:
        from numpilot.qualified_analysis import (
            QualifiedAnalysisError,
            analyze_qualified_script,
        )
        from numpilot.script_migration import (
            ScriptMigrationError,
            scan_numpy_script,
        )

        qualified = analyze_qualified_script(source, source_path)
        scan = scan_numpy_script(
            source,
            source_path,
            qualified_analysis=qualified,
        )
    except (QualifiedAnalysisError, ScriptMigrationError) as exc:
        raise AddOneGapError("add-one static admission failed") from exc
    return qualified, scan


def validate_add_one_source_admission(
    *,
    source_path: str,
    source: str,
    calls: Sequence[AddOneCall],
    scanner_issues: Sequence[str] | None = None,
) -> str:
    """Return the exact lowering only when its effective script is inertly safe.

    The ordinary scanner necessarily rejects ``np.add(array, 1)`` because that
    signature is not an AsNumpy call contract.  The registered route may waive
    only that one issue: it builds the exact in-memory lowering, then requires
    both static analyzers to accept the resulting script without issues.  Since
    the lowering changes no other AST surface, imports, arbitrary calls,
    side-effecting methods, alias rebinding, and similar unsafe constructs stay
    visible and fail closed before either source is executed.
    """

    if not Path(source_path).is_absolute():
        raise AddOneGapError("add-one admission requires an absolute source path")
    payload = _lower_add_one_payload(source, calls)
    normalized = payload.decode("utf-8")
    source_qualified, source_scan = _analyze_effective_script(
        source_path, source
    )
    normalized_qualified, normalized_scan = _analyze_effective_script(
        source_path, normalized
    )
    source_issues = tuple(source_scan.issues)
    expected_prefixes = tuple(
        "AsNumpy call contract rejected numpy.add at line %d:" % item.lineno
        for item in calls
    )
    supplied_issues = (
        tuple(str(item) for item in scanner_issues)
        if scanner_issues is not None
        else source_issues
    )
    expected_gap_only = (
        not source_scan.ok
        and bool(source_issues)
        and source_issues == supplied_issues
        and all(
            any(issue.startswith(prefix) for prefix in expected_prefixes)
            for issue in source_issues
        )
    )
    effective_issues = tuple(normalized_qualified.issues) + tuple(
        normalized_scan.issues
    )
    if (
        tuple(source_qualified.issues)
        or not expected_gap_only
        or not normalized_scan.ok
        or effective_issues
    ):
        issues = (
            tuple(source_qualified.issues)
            + source_issues
            + effective_issues
        )
        raise AddOneGapError(
            "add-one static admission rejected the effective script: %s"
            % (str(issues[0]) if issues else "scanner did not pass")
        )
    return normalized


def lower_add_one_calls(
    *,
    source_path: str,
    source: str,
    calls: Sequence[AddOneCall],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Lower exact scalar-one calls to the qualified two-array composition."""

    if (
        not Path(source_path).is_absolute()
        or tuple(calls) != detect_add_one_calls(source)
    ):
        raise AddOneGapError("lowering input is not source-bound")
    normalized = validate_add_one_source_admission(
        source_path=source_path,
        source=source,
        calls=calls,
    )
    payload = normalized.encode("utf-8")
    root = Path(out_dir).absolute()
    normalized_path = _atomic_private(root / "normalized-source.py", payload)
    normalized_sha = hashlib.sha256(payload).hexdigest()
    report = {
        "schema_version": 1,
        "record_type": "numpilot_add_one_functionality_lowering",
        "status": "lowered_unverified",
        "source_path": source_path,
        "source_sha256": _source_hash(source),
        "normalized_source_path": str(normalized_path),
        "normalized_source_sha256": normalized_sha,
        # ``target_api`` is retained as the legacy name of the materialized
        # interface contract.  The explicit fields below prevent consumers
        # from mistaking that contract for an API invoked by this lowering.
        "target_api": _TARGET_API_CONTRACT,
        "target_api_contract": _TARGET_API_CONTRACT,
        "active_route": "COMPOSITE_NPU",
        "constituents": list(_EXPECTED_MIGRATED_PROGRAM_APIS),
        "lowered_program_apis": list(_LOWERED_PROGRAM_APIS),
        "expected_migrated_program_apis": list(
            _EXPECTED_MIGRATED_PROGRAM_APIS
        ),
        "migrated_program_uses_target_api": False,
        "interface_activation_required_by_lowered_program": False,
        "callsites": [item.to_dict() for item in calls],
        "numeric_equivalence": "not_run",
        "cannbot_custom_kernel_activated": False,
    }
    report_path = _atomic_private(
        root / "add-one-lowering-report.json",
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return normalized_path, report_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _lowering_verification_message(code: str) -> str:
    return {
        "confirmation_required": (
            "add-one lowering verification requires explicit user confirmation"
        ),
        "invalid_lowering_report": "add-one lowering report contract is invalid",
        "input_drift": "a hash-bound lowering input changed during verification",
        "numerical_mismatch": "add-one lowering changed a contracted output",
        "internal_error": "add-one lowering verification failed closed",
    }.get(code, "add-one lowering verification failed: %s" % code)


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _read_lowering_inputs(
    *,
    source_path: Path,
    normalized_path: Path,
    report_path: Path,
) -> tuple[bytes, bytes, bytes, Mapping[str, Any]]:
    """Read and admit one source-bound lowering tuple."""

    try:
        source_payload = _script_verification._read_regular(
            source_path, MAX_SOURCE_BYTES
        )
        normalized_payload = _script_verification._read_regular(
            normalized_path, MAX_SOURCE_BYTES
        )
        report_payload = _script_verification._read_regular(
            report_path, MAX_SOURCE_BYTES
        )
        source_text = source_payload.decode("utf-8")
        normalized_text = normalized_payload.decode("utf-8")
        report = json.loads(report_payload.decode("utf-8"))
    except (
        _script_verification.ScriptVerificationError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise AddOneLoweringVerificationError(
            "invalid_lowering_report",
            _lowering_verification_message("invalid_lowering_report"),
        ) from exc
    source_sha = hashlib.sha256(source_payload).hexdigest()
    normalized_sha = hashlib.sha256(normalized_payload).hexdigest()
    try:
        source_calls = detect_add_one_calls(source_text)
        normalized_calls = detect_add_one_calls(normalized_text)
        validate_add_one_source_admission(
            source_path=str(source_path),
            source=source_text,
            calls=source_calls,
        )
        normalized_qualified, normalized_scan = _analyze_effective_script(
            str(normalized_path), normalized_text
        )
    except AddOneGapError as exc:
        raise AddOneLoweringVerificationError(
            "invalid_lowering_report",
            _lowering_verification_message("invalid_lowering_report"),
        ) from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("record_type")
        != "numpilot_add_one_functionality_lowering"
        or report.get("status") != "lowered_unverified"
        or not isinstance(report.get("source_path"), str)
        or not os.path.isabs(report["source_path"])
        or _normal_path(Path(report["source_path"])) != _normal_path(source_path)
        or report.get("source_sha256") != source_sha
        or not isinstance(report.get("normalized_source_path"), str)
        or not os.path.isabs(report["normalized_source_path"])
        or _normal_path(Path(report["normalized_source_path"]))
        != _normal_path(normalized_path)
        or report.get("normalized_source_sha256") != normalized_sha
        or report.get("target_api") != _TARGET_API_CONTRACT
        or report.get("target_api_contract") != _TARGET_API_CONTRACT
        or report.get("active_route") != "COMPOSITE_NPU"
        or report.get("constituents")
        != list(_EXPECTED_MIGRATED_PROGRAM_APIS)
        or report.get("lowered_program_apis") != list(_LOWERED_PROGRAM_APIS)
        or report.get("expected_migrated_program_apis")
        != list(_EXPECTED_MIGRATED_PROGRAM_APIS)
        or report.get("migrated_program_uses_target_api") is not False
        or report.get("interface_activation_required_by_lowered_program")
        is not False
        or report.get("callsites") != [item.to_dict() for item in source_calls]
        or report.get("numeric_equivalence") != "not_run"
        or report.get("cannbot_custom_kernel_activated") is not False
        or not source_calls
        or normalized_calls
        or tuple(normalized_qualified.issues)
        or not normalized_scan.ok
        or tuple(normalized_scan.issues)
    ):
        raise AddOneLoweringVerificationError(
            "invalid_lowering_report",
            _lowering_verification_message("invalid_lowering_report"),
        )
    return source_payload, normalized_payload, report_payload, report


def _lowering_inputs_unchanged(
    *,
    source_path: Path,
    normalized_path: Path,
    report_path: Path,
    source_sha256: str,
    normalized_sha256: str,
    report_sha256: str,
) -> None:
    try:
        current = (
            hashlib.sha256(
                _script_verification._read_regular(source_path, MAX_SOURCE_BYTES)
            ).hexdigest(),
            hashlib.sha256(
                _script_verification._read_regular(normalized_path, MAX_SOURCE_BYTES)
            ).hexdigest(),
            hashlib.sha256(
                _script_verification._read_regular(report_path, MAX_SOURCE_BYTES)
            ).hexdigest(),
        )
    except _script_verification.ScriptVerificationError as exc:
        raise AddOneLoweringVerificationError(
            "input_drift", _lowering_verification_message("input_drift")
        ) from exc
    if current != (source_sha256, normalized_sha256, report_sha256):
        raise AddOneLoweringVerificationError(
            "input_drift", _lowering_verification_message("input_drift")
        )


def _lowering_verification_receipt(
    *,
    status: str,
    started_at: str,
    source_path: Path,
    normalized_path: Path,
    report_path: Path,
    source_sha256: str,
    normalized_sha256: str,
    report_sha256: str,
    report: Mapping[str, Any],
    output_names: Sequence[str],
    timeout_seconds: float,
    max_output_bytes: int,
    executions: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    failure: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "numpilot_add_one_lowering_numerical_verification",
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "user_confirmation": {"required": True, "received": True},
        "source": {"path": str(source_path), "sha256": source_sha256},
        "normalized_source": {
            "path": str(normalized_path),
            "sha256": normalized_sha256,
        },
        "lowering": {
            "report_path": str(report_path),
            "report_sha256": report_sha256,
            "report_status": report.get("status"),
            "target_api": report.get("target_api"),
            "target_api_contract": report.get("target_api_contract"),
            "active_route": report.get("active_route"),
            "constituents": list(report.get("constituents") or ()),
            "lowered_program_apis": list(
                report.get("lowered_program_apis") or ()
            ),
            "expected_migrated_program_apis": list(
                report.get("expected_migrated_program_apis") or ()
            ),
            "migrated_program_uses_target_api": report.get(
                "migrated_program_uses_target_api"
            ),
            "interface_activation_required_by_lowered_program": report.get(
                "interface_activation_required_by_lowered_program"
            ),
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
            "explicit_user_confirmation": True,
            "stdout_used_as_evidence": False,
            "stderr_used_as_evidence": False,
        },
        "claims": {
            "source_to_normalized_outputs_consistent": status == "passed",
            "constituent_lowering_numerically_verified": status == "passed",
            "migrated_program_uses_add_one_interface": False,
            "add_one_interface_integration_verified": False,
            "scope": "only the explicitly named output variables",
            "strict_npu_provenance_verified": False,
            "performance_verified": False,
            "cannbot_custom_kernel_activated": False,
        },
        "failure": dict(failure) if failure else None,
    }
    identity = {
        "receipt_type": payload["receipt_type"],
        "status": status,
        "source_sha256": source_sha256,
        "normalized_source_sha256": normalized_sha256,
        "lowering_report_sha256": report_sha256,
        "contract": payload["contract"],
        "executions": payload["executions"],
        "comparisons": payload["comparisons"],
        "failure": payload["failure"],
    }
    payload["verification_id"] = hashlib.sha256(
        json.dumps(
            identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload


def verify_add_one_lowering(
    source_path: os.PathLike[str] | str,
    normalized_source_path: os.PathLike[str] | str,
    lowering_report_path: os.PathLike[str] | str,
    output_names: Sequence[str],
    out_path: os.PathLike[str] | str,
    *,
    user_confirmed: bool,
    python_executable: os.PathLike[str] | str | None = None,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 64 * 1024 * 1024,
) -> _script_verification.ScriptVerificationReceipt:
    """Prove original/normalized NumPy output equivalence after confirmation.

    This gate deliberately proves numerical equivalence only.  It neither
    executes nor activates the CANNBot draft, and it emits no NPU or
    performance claim.  ``user_confirmed`` must be the literal boolean
    ``True``; otherwise the function performs no filesystem access and starts
    no subprocess.

    Once all hash-bound inputs pass preflight, execution/comparison failures
    publish a write-once ``failed`` receipt and raise
    :class:`AddOneLoweringVerificationError`.  Preflight failures publish
    nothing.  An existing destination is never replaced.
    """

    if user_confirmed is not True:
        raise AddOneLoweringVerificationError(
            "confirmation_required",
            _lowering_verification_message("confirmation_required"),
        )
    try:
        names = _script_verification._validate_output_names(output_names)
        _script_verification._validate_limits(timeout_seconds, max_output_bytes)
        source = _script_verification._absolute_path(
            source_path, code="invalid_argument"
        )
        normalized = _script_verification._absolute_path(
            normalized_source_path, code="invalid_argument"
        )
        lowering_report = _script_verification._absolute_path(
            lowering_report_path, code="invalid_argument"
        )
        receipt_path = _script_verification._absolute_path(
            out_path, code="invalid_output_path"
        )
        _script_verification._prepare_receipt_path(
            receipt_path, (source, normalized, lowering_report)
        )
        python_identity = _script_verification._python_executable(python_executable)
    except _script_verification.ScriptVerificationError as exc:
        raise AddOneLoweringVerificationError(
            exc.code, _lowering_verification_message(exc.code)
        ) from exc

    source_payload, normalized_payload, report_payload, report = (
        _read_lowering_inputs(
            source_path=source,
            normalized_path=normalized,
            report_path=lowering_report,
        )
    )
    source_sha = hashlib.sha256(source_payload).hexdigest()
    normalized_sha = hashlib.sha256(normalized_payload).hexdigest()
    report_sha = hashlib.sha256(report_payload).hexdigest()
    started_at = _utc_now()
    executions: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}

    def unchanged() -> None:
        _lowering_inputs_unchanged(
            source_path=source,
            normalized_path=normalized,
            report_path=lowering_report,
            source_sha256=source_sha,
            normalized_sha256=normalized_sha,
            report_sha256=report_sha,
        )

    try:
        unchanged()
        worker_source = Path(_script_verification.__file__).with_name(
            "_script_verification_worker.py"
        )
        worker_payload = _script_verification._read_regular(
            worker_source, MAX_SOURCE_BYTES
        )
        worker_sha = hashlib.sha256(worker_payload).hexdigest()
        with tempfile.TemporaryDirectory(
            prefix="numpilot-add-one-lowering-verify-"
        ) as raw_root:
            root = Path(raw_root)
            os.chmod(root, 0o700)
            worker_snapshot = root / "worker.py"
            source_snapshot = root / "source.py"
            normalized_snapshot = root / "normalized.py"
            _script_verification._write_private(worker_snapshot, worker_payload)
            _script_verification._write_private(source_snapshot, source_payload)
            _script_verification._write_private(
                normalized_snapshot, normalized_payload
            )
            source_result = _script_verification._run_worker(
                root=root,
                label="source",
                worker_path=worker_snapshot,
                script_path=source_snapshot,
                script_sha256=source_sha,
                worker_sha256=worker_sha,
                output_names=names,
                python_identity=python_identity,
                timeout_seconds=float(timeout_seconds),
                max_output_bytes=max_output_bytes,
            )
            executions["source"] = _script_verification._execution_summary(
                source_result
            )
            unchanged()
            normalized_result = _script_verification._run_worker(
                root=root,
                label="normalized",
                worker_path=worker_snapshot,
                script_path=normalized_snapshot,
                script_sha256=normalized_sha,
                worker_sha256=worker_sha,
                output_names=names,
                python_identity=python_identity,
                timeout_seconds=float(timeout_seconds),
                max_output_bytes=max_output_bytes,
            )
            executions["normalized"] = _script_verification._execution_summary(
                normalized_result
            )
            for name in names:
                comparison = compare_outputs(
                    source_result.arrays[name], normalized_result.arrays[name]
                )
                comparisons[name] = asdict(comparison)
            if not all(value["consistent"] for value in comparisons.values()):
                raise AddOneLoweringVerificationError(
                    "numerical_mismatch",
                    _lowering_verification_message("numerical_mismatch"),
                )
        unchanged()
        payload = _lowering_verification_receipt(
            status="passed",
            started_at=started_at,
            source_path=source,
            normalized_path=normalized,
            report_path=lowering_report,
            source_sha256=source_sha,
            normalized_sha256=normalized_sha,
            report_sha256=report_sha,
            report=report,
            output_names=names,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            executions=executions,
            comparisons=comparisons,
            failure=None,
        )
        return _script_verification._publish_receipt(receipt_path, payload)
    except (AddOneLoweringVerificationError, _script_verification.ScriptVerificationError) as exc:
        code = exc.code
        if code in {"receipt_exists", "invalid_output_path", "receipt_publish_failed"}:
            raise AddOneLoweringVerificationError(
                code, _lowering_verification_message(code)
            ) from exc
        payload = _lowering_verification_receipt(
            status="failed",
            started_at=started_at,
            source_path=source,
            normalized_path=normalized,
            report_path=lowering_report,
            source_sha256=source_sha,
            normalized_sha256=normalized_sha,
            report_sha256=report_sha,
            report=report,
            output_names=names,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            executions=executions,
            comparisons=comparisons,
            failure={
                "code": code,
                "message": _lowering_verification_message(code),
                "exception_type": getattr(exc, "exception_type", None),
            },
        )
        try:
            receipt = _script_verification._publish_receipt(receipt_path, payload)
        except _script_verification.ScriptVerificationError as publish_exc:
            raise AddOneLoweringVerificationError(
                publish_exc.code, _lowering_verification_message(publish_exc.code)
            ) from publish_exc
        raise AddOneLoweringVerificationError(
            code, _lowering_verification_message(code), receipt=receipt
        ) from exc
    except Exception as exc:
        payload = _lowering_verification_receipt(
            status="failed",
            started_at=started_at,
            source_path=source,
            normalized_path=normalized,
            report_path=lowering_report,
            source_sha256=source_sha,
            normalized_sha256=normalized_sha,
            report_sha256=report_sha,
            report=report,
            output_names=names,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            executions=executions,
            comparisons=comparisons,
            failure={
                "code": "internal_error",
                "message": _lowering_verification_message("internal_error"),
                "exception_type": type(exc).__name__,
            },
        )
        try:
            receipt = _script_verification._publish_receipt(receipt_path, payload)
        except _script_verification.ScriptVerificationError as publish_exc:
            raise AddOneLoweringVerificationError(
                publish_exc.code, _lowering_verification_message(publish_exc.code)
            ) from publish_exc
        raise AddOneLoweringVerificationError(
            "internal_error",
            _lowering_verification_message("internal_error"),
            receipt=receipt,
        ) from exc


def add_one_gap_summary(calls: Sequence[AddOneCall]) -> dict[str, Any]:
    """Return the stable UI/report facts for detected registered calls."""

    return {
        "kind": "registered_signature_gap",
        "source_api": "numpy.add",
        "missing_interface": "asnumpy.add_one",
        "reason_code": "SCALAR_SECOND_OPERAND_UNSUPPORTED",
        "semantic_contract": "float32 contiguous y = x + 1.0",
        "candidate_operator": "elementwise Ascend C AddOne",
        "functionality_first_route": "asnumpy.ones_like + asnumpy.add",
        "target_api_contract": _TARGET_API_CONTRACT,
        "lowered_program_apis": list(_LOWERED_PROGRAM_APIS),
        "expected_migrated_program_apis": list(
            _EXPECTED_MIGRATED_PROGRAM_APIS
        ),
        "migrated_program_uses_target_api": False,
        "interface_activation_required_by_lowered_program": False,
        "callsites": [item.to_dict() for item in calls],
    }


__all__ = [
    "AddOneCall",
    "AddOneGapError",
    "AddOneLoweringVerificationError",
    "add_one_gap_summary",
    "detect_add_one_calls",
    "lower_add_one_calls",
    "materialize_add_one_operator_spec",
    "verify_add_one_lowering",
]
