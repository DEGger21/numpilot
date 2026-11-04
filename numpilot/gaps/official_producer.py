"""Produce one official ACLNN capability overlay from integrated evidence.

The official overlay is a *separate* qualification product from the custom
operator overlay.  It does not inherit artifact-manifest semantics from the
self-developed operator route; instead it binds one already-integrated
candidate wheel to six independently reproducible gate reports:

``artifact_integrity``, ``build``, ``unit_test``, ``numerical_validation``,
``public_asnumpy_api`` and ``npu_execution``.

This module is the producer side of that contract.  It runs under the same
interpreter that later executes the migrated program, re-validates every bound
file, re-runs the candidate tests and the native execution probes, and only
then publishes the overlay through the fail-closed compare-and-swap in
:mod:`numpilot.gaps.official_qualification`.

The producer never contacts a model, never activates an overlay, and never
modifies the integrated source tree or the candidate wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from numpilot.capabilities import OperatorSignature
from numpilot.gaps.official_qualification import (
    OFFICIAL_GATES,
    OFFICIAL_OVERLAY_AUDIT_TAG,
    OFFICIAL_PRODUCER,
    OFFICIAL_PROVENANCE,
    OFFICIAL_QUALIFICATION_PREFIX,
    ZERO_SHA256,
    OfficialCapabilityOverlay,
    OfficialFileRef,
    OfficialGapBinding,
    OfficialIntegrationBinding,
    OfficialOverlayEntry,
    OfficialRuntimeBinding,
    OfficialSourceBinding,
    QualificationError,
    _MAX_EVIDENCE_FILE_BYTES,
    _parse_json,
    _read_private_file,
    _sha256_bytes,
    _sha256_file,
    _transaction_sha256,
    _validate_private_directory,
    _write_private_file,
    build_official_gate_report,
    compute_official_qualification_id,
    load_official_overlay,
    official_overlay_file_sha256,
    publish_official_overlay_atomic,
)
from numpilot.gaps.qualification import capability_manifest_sha256

PRODUCER_RECEIPT_SCHEMA = "numpilot.official-aclnn-qualification-receipt"
PRODUCER_RECEIPT_SCHEMA_VERSION = 1
GATE_EVIDENCE_SCHEMA = "numpilot.official-aclnn-gate-evidence"
GATE_EVIDENCE_SCHEMA_VERSION = 1

_MAX_JSON_BYTES = 2 * 1024 * 1024
_UNITTEST_TIMEOUT_SECONDS = 900
_NATIVE_CALLS_PER_TARGET = 2
_REPEATED_CALLS_PER_TARGET = 2

_HOST_FALLBACK_MARKERS = (
    "to_numpy(",
    "import numpy",
    "np.",
    "numpy.array(",
    "numpy.asarray(",
    "cpu_fallback",
    "host_fallback",
)


class OfficialProducerError(Exception):
    """One official qualification could not be produced truthfully."""


@dataclass(frozen=True)
class _EntrySpec:
    key: str
    constraints: Tuple[str, ...]
    signatures: Tuple[OperatorSignature, ...]
    test_marker: str


_ENTRY_SPECS: Mapping[str, _EntrySpec] = {
    "concatenate": _EntrySpec(
        key="concatenate",
        constraints=(
            "dtype=float32",
            "rank>=2",
            "rank<=2",
            "contiguous=true",
        ),
        signatures=(
            OperatorSignature(positional_args=1, keyword_args=("axis",)),
            OperatorSignature(positional_args=2, keyword_args=()),
            OperatorSignature(positional_args=1, keyword_args=()),
        ),
        test_marker="concatenate",
    ),
    "ones": _EntrySpec(
        key="ones",
        constraints=(
            "dtype=float32",
            "rank>=1",
            "rank<=3",
            "contiguous=true",
        ),
        signatures=(
            OperatorSignature(positional_args=1, keyword_args=("dtype",)),
            OperatorSignature(positional_args=2, keyword_args=()),
        ),
        test_marker="ones",
    ),
    "zeros": _EntrySpec(
        key="zeros",
        constraints=(
            "dtype=float32",
            "rank>=1",
            "rank<=3",
            "contiguous=true",
        ),
        signatures=(
            OperatorSignature(positional_args=1, keyword_args=("dtype",)),
            OperatorSignature(positional_args=2, keyword_args=()),
        ),
        test_marker="zeros",
    ),
}

_ENTRY_ORDER = ("concatenate", "ones", "zeros")


@dataclass(frozen=True)
class _Context:
    run_root: Path
    bundle: Path
    attempt: str
    attempt_dir: Path
    revision: int
    parent_sha256: str
    gates_dir: Path
    entries: Tuple[str, ...]
    identity: Any
    transaction: Mapping[str, Any]
    transaction_file_sha256: str
    wheel_path: Path
    runtime_core_sha256: str
    source_overlay_sha256: Mapping[str, str]
    base_manifest_sha256: str
    source_file: Path
    source_logical_path: str
    source_bundle_path: str
    source_sha256: str
    gap_request: Path
    gap_request_sha256: str
    candidate_test_file: Path
    candidate_test_sha256: str


def _relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OfficialProducerError("%s must be non-empty text" % label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
        raise OfficialProducerError("%s must be a safe relative path" % label)
    return str(path)


def _ensure_private_directory(path: Path) -> None:
    if path.exists():
        _validate_private_directory(path, "producer directory")
        return
    if not path.parent.exists():
        raise OfficialProducerError(
            "producer directory parent does not exist: %s" % path.parent.name
        )
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise OfficialProducerError(
            "cannot create producer directory %s" % path.name
        ) from exc
    os.chmod(path, 0o700)
    _validate_private_directory(path, "producer directory")


def _replace_private_file(path: Path, payload: bytes) -> None:
    """Replace one producer-owned file without following links."""

    if os.path.lexists(path):
        metadata = path.lstat()
        if (
            not path.is_file()
            or path.is_symlink()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o022
        ):
            raise OfficialProducerError(
                "refusing to replace a non-private evidence file"
            )
        path.unlink()
    _write_private_file(path, payload, exclusive=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    _replace_private_file(path, text)
    return _sha256_bytes(text)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    return _parse_json(
        _read_private_file(path, label, max_bytes=_MAX_JSON_BYTES),
        label,
    )


def _copy_verified(
    source: Path,
    destination: Path,
    expected_sha256: str,
    label: str,
) -> None:
    payload = _read_private_file(
        source, label, max_bytes=_MAX_EVIDENCE_FILE_BYTES
    )
    digest = _sha256_bytes(payload)
    if digest != expected_sha256:
        raise OfficialProducerError(
            "%s hash does not match the integrated evidence" % label
        )
    _validate_private_directory(destination.parent, "producer evidence directory")
    if os.path.lexists(destination):
        existing = _read_private_file(
            destination, label + " copy", max_bytes=_MAX_EVIDENCE_FILE_BYTES
        )
        if existing != payload:
            raise OfficialProducerError(
                "%s evidence bundle has drifted from the integrated evidence"
                % label
            )
        return
    _write_private_file(destination, payload, exclusive=True)
    if _sha256_file(destination) != expected_sha256:
        raise OfficialProducerError("%s copy verification failed" % label)


def _evidence_ref(path: Path, bundle: Path) -> Dict[str, str]:
    try:
        relative = path.relative_to(bundle)
    except ValueError as exc:
        raise OfficialProducerError(
            "gate evidence escaped the qualification bundle"
        ) from exc
    if not relative.parts or relative.parts[0] != "gates":
        raise OfficialProducerError("gate evidence must live under gates/")
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
    }


def _json_safe_mapping(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value[key] for key in sorted(value)}


def _load_transaction(attempt_dir: Path) -> Tuple[Mapping[str, Any], str]:
    path = attempt_dir / "integration-transaction.json"
    payload = _read_private_file(
        path, "integration transaction", max_bytes=_MAX_EVIDENCE_FILE_BYTES
    )
    file_sha256 = _sha256_bytes(payload)
    transaction = _parse_json(payload, "integration transaction")
    if transaction.get("record_type") != "numpilot_integration_transaction":
        raise OfficialProducerError("integration transaction record type is invalid")
    if transaction.get("schema_version") != 1:
        raise OfficialProducerError("integration transaction schema is invalid")
    if transaction.get("status") != "passed":
        raise OfficialProducerError("integration transaction did not pass")
    if transaction.get("failure") is not None:
        raise OfficialProducerError("integration transaction records a failure")
    canonical = transaction.get("transaction_sha256")
    if not isinstance(canonical, str) or _transaction_sha256(transaction) != canonical:
        raise OfficialProducerError(
            "integration transaction does not reproduce its canonical hash"
        )
    return transaction, file_sha256


def _load_gap_request(
    path: Path,
) -> Tuple[Mapping[str, Any], str]:
    payload = _read_private_file(path, "gap request", max_bytes=_MAX_JSON_BYTES)
    digest = _sha256_bytes(payload)
    request = _parse_json(payload, "gap request")
    if request.get("request_type") != "script_api_gap":
        raise OfficialProducerError("gap request type is invalid")
    if not isinstance(request.get("source_sha256"), str):
        raise OfficialProducerError("gap request source hash is invalid")
    if not isinstance(request.get("gap_specs"), list):
        raise OfficialProducerError("gap request has no gap specifications")
    return request, digest


def _matching_gap_spec(
    request: Mapping[str, Any], gap_id: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in request["gap_specs"]
        if isinstance(item, Mapping) and item.get("gap_id") == gap_id
    ]
    if len(matches) != 1:
        raise OfficialProducerError(
            "gap request does not contain exactly one matching specification"
        )
    return matches[0]


def _collect_runtime_identity():
    from numpilot.backends.npu import NpuBackend

    backend = NpuBackend()
    first = backend.runtime_identity()
    second = backend.runtime_identity()
    if first.to_dict() != second.to_dict():
        raise OfficialProducerError(
            "runtime identity is not stable across two read-only probes"
        )
    identity = first
    prefix = Path(sys.prefix).resolve()
    try:
        module_path = Path(identity.module_path).resolve()
    except (OSError, RuntimeError) as exc:
        raise OfficialProducerError("runtime module path is unavailable") from exc
    if not module_path.is_relative_to(prefix):
        raise OfficialProducerError(
            "the producer is not running under the qualification runtime"
        )
    try:
        python_executable = Path(identity.python_executable).resolve()
    except (OSError, RuntimeError) as exc:
        raise OfficialProducerError("runtime interpreter path is unavailable") from exc
    if python_executable != Path(sys.executable).resolve():
        raise OfficialProducerError(
            "runtime identity was collected by a different interpreter"
        )

    import asnumpy

    if getattr(asnumpy, "__version__", None) != identity.asnumpy_version:
        raise OfficialProducerError("live AsNumpy version disagrees with identity")
    try:
        installed = importlib.metadata.version("asnumpy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise OfficialProducerError("AsNumpy distribution metadata is unavailable") from exc
    if installed != identity.asnumpy_version:
        raise OfficialProducerError(
            "installed AsNumpy distribution disagrees with the runtime identity"
        )
    wrappers = dict(identity.wrapper_sha256)
    for relative in ("array.py", "__init__.py"):
        if relative not in wrappers:
            raise OfficialProducerError(
                "runtime identity has no %s wrapper hash" % relative
            )
    return backend, identity


def _base_manifest_sha256(backend: Any, keys: Sequence[str], identity: Any) -> str:
    manifest = backend.capability_manifest(tuple(keys), runtime_identity=identity)
    for key in keys:
        capability = manifest.operators.get(key)
        if capability is None or not capability.available:
            raise OfficialProducerError(
                "runtime does not expose the planned overlay API %s" % key
            )
    return capability_manifest_sha256(manifest)


def _wheel_members(path: Path) -> Mapping[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name and not name.endswith("/")
            ]
            if len(set(names)) != len(names):
                raise OfficialProducerError("candidate wheel has duplicate members")
            members = {
                name: _sha256_bytes(archive.read(name)) for name in names
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise OfficialProducerError("candidate wheel is unreadable") from exc
    if not members:
        raise OfficialProducerError("candidate wheel has no members")
    return members


def _artifact_integrity_claims(context: _Context) -> Mapping[str, Any]:
    return {
        "transaction_file_sha256": context.transaction_file_sha256,
        "transaction_sha256": context.transaction["transaction_sha256"],
        "wheel_sha256": context.transaction["build"]["wheel_sha256"],
        "runtime_identity_sha256": context.identity.identity_sha256,
        "source_sha256": context.source_sha256,
        "gap_request_sha256": context.gap_request_sha256,
        "base_manifest_sha256": context.base_manifest_sha256,
        "candidate_test_log_sha256": context.transaction["candidate_tests"][
            "log_sha256"
        ],
        "gate_result_sha256": context.transaction["gate"]["result_sha256"],
        "gate_log_sha256": context.transaction["gate"]["log_sha256"],
        "source_overlay_sha256": _json_safe_mapping(
            context.source_overlay_sha256
        ),
        "runtime_core_sha256": context.runtime_core_sha256,
    }


def _run_build_gate(context: _Context) -> Dict[str, Dict[str, Any]]:
    claims = _artifact_integrity_claims(context)
    members = _wheel_members(context.wheel_path)
    wrapper_hashes = dict(context.identity.wrapper_sha256)
    required = ("asnumpy/array.py", "asnumpy/__init__.py")
    for name in required:
        if name not in members:
            raise OfficialProducerError("candidate wheel is missing %s" % name)
    if members["asnumpy/array.py"] != wrapper_hashes.get("array.py"):
        raise OfficialProducerError(
            "candidate wheel concatenate wrapper disagrees with the runtime"
        )
    if members["asnumpy/__init__.py"] != wrapper_hashes.get("__init__.py"):
        raise OfficialProducerError(
            "candidate wheel module disagrees with the runtime"
        )
    core_members = [
        name
        for name in members
        if name.startswith("asnumpy/_core") and name.endswith(".so")
    ]
    if not core_members:
        raise OfficialProducerError(
            "candidate wheel has no native AsNumpy core member"
        )
    for name in core_members:
        if members[name] != context.runtime_core_sha256:
            raise OfficialProducerError(
                "candidate wheel native member disagrees with the runtime"
            )
    build_claims = dict(claims)
    build_claims.update(
        {
            "build_status": "passed",
            "wheel_sha256": context.transaction["build"]["wheel_sha256"],
            "wheel_member_count": len(members),
            "wheel_member_sha256": _json_safe_mapping(members),
        }
    )
    targets = {
        key: {
            "status": "passed",
            "claims": dict(build_claims),
            "evidence": {},
        }
        for key in context.entries
    }
    return targets


def _run_pytest(
    context: _Context,
    marker: str,
    junit_path: Path,
    log_path: Path,
) -> int:
    workdir = Path(
        tempfile.mkdtemp(prefix=".unit-test-", dir=str(context.gates_dir))
    )
    test_copy = workdir / "test_basic.py"
    try:
        _copy_verified(
            context.candidate_test_file,
            test_copy,
            context.candidate_test_sha256,
            "candidate unit test",
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTEST_ADDOPTS"] = ""
        descriptor = os.open(
            str(log_path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                try:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "-q",
                            "-p",
                            "no:cacheprovider",
                            "-k",
                            marker,
                            "--junitxml",
                            str(junit_path),
                            "test_basic.py",
                        ],
                        cwd=str(workdir),
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=_UNITTEST_TIMEOUT_SECONDS,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise OfficialProducerError(
                        "candidate unit test replay timed out for %s" % marker
                    ) from exc
        except OSError:
            raise
        return int(completed.returncode)
    finally:
        shutil.rmtree(str(workdir), ignore_errors=True)


def _parse_junit(path: Path) -> Dict[str, Any]:
    try:
        payload = _read_private_file(
            path, "unit test JUnit report", max_bytes=_MAX_EVIDENCE_FILE_BYTES
        )
        root = ET.fromstring(payload)
    except (QualificationError, ET.ParseError) as exc:
        raise OfficialProducerError("unit test JUnit report is invalid") from exc
    testcases = root.findall(".//testcase")
    failures = sum(1 for item in testcases if item.find("failure") is not None)
    errors = sum(1 for item in testcases if item.find("error") is not None)
    skipped = sum(1 for item in testcases if item.find("skipped") is not None)
    names = [
        item.get("name") or ""
        for item in testcases
        if item.get("name")
    ]
    return {
        "tests": len(testcases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": max(0, len(testcases) - failures - errors - skipped),
        "test_names": names,
    }


def _run_unit_test_gate(context: _Context) -> Dict[str, Dict[str, Any]]:
    test_evidence = context.gates_dir / "unit_test.candidate-test.py"
    _copy_verified(
        context.candidate_test_file,
        test_evidence,
        context.candidate_test_sha256,
        "candidate unit test",
    )
    test_ref = _evidence_ref(test_evidence, context.bundle)
    targets = {}
    for key in context.entries:
        spec = _ENTRY_SPECS[key]
        junit_path = context.gates_dir / ("unit_test.%s.junit.xml" % key)
        log_path = context.gates_dir / ("unit_test.%s.log" % key)
        returncode = _run_pytest(context, spec.test_marker, junit_path, log_path)
        if not junit_path.exists():
            raise OfficialProducerError(
                "candidate unit test replay produced no JUnit report for %s" % key
            )
        stats = _parse_junit(junit_path)
        claims = {
            "returncode": returncode,
            "passed": stats["passed"],
            "failed": stats["failures"],
            "errors": stats["errors"],
            "junit_sha256": _sha256_file(junit_path),
            "test_names": stats["test_names"],
        }
        target_passed = bool(
            returncode == 0
            and stats["failures"] == 0
            and stats["errors"] == 0
            and stats["passed"] >= 1
            and any(spec.test_marker in name.lower() for name in stats["test_names"])
        )
        targets[key] = {
            "status": "passed" if target_passed else "failed",
            "claims": claims,
            "evidence": {
                "files": [
                    _evidence_ref(junit_path, context.bundle),
                    _evidence_ref(log_path, context.bundle),
                    test_ref,
                ],
                "marker": spec.test_marker,
                "skipped": stats["skipped"],
            },
        }
    return targets


def _compare_arrays(actual: Any, expected: Any) -> Tuple[int, bool]:
    comparisons = 0
    okay = True
    try:
        comparisons += 1
        if tuple(actual.shape) != tuple(expected.shape):
            okay = False
        comparisons += 1
        if actual.dtype != expected.dtype:
            okay = False
        comparisons += 1
        import numpy

        numpy.testing.assert_array_equal(actual, expected)
    except (AssertionError, TypeError, ValueError):
        okay = False
    return comparisons, okay


def _numerical_cases(key: str):
    import asnumpy
    import numpy

    dtype = numpy.float32
    if key == "concatenate":
        first = asnumpy.full((2, 3), 1.0, dtype=dtype)
        second = asnumpy.full((1, 3), 2.0, dtype=dtype)
        third = asnumpy.full((2, 2), 3.0, dtype=dtype)
        fourth = asnumpy.full((2, 4), 4.0, dtype=dtype)
        single = asnumpy.full((2, 3), 5.0, dtype=dtype)
        return [
            (
                "axis_zero",
                lambda: asnumpy.concatenate([first, second], axis=0),
                lambda: numpy.concatenate(
                    [
                        numpy.full((2, 3), 1.0, dtype=dtype),
                        numpy.full((1, 3), 2.0, dtype=dtype),
                    ],
                    axis=0,
                ),
            ),
            (
                "axis_one",
                lambda: asnumpy.concatenate([first, third], axis=1),
                lambda: numpy.concatenate(
                    [
                        numpy.full((2, 3), 1.0, dtype=dtype),
                        numpy.full((2, 2), 3.0, dtype=dtype),
                    ],
                    axis=1,
                ),
            ),
            (
                "axis_negative_one",
                lambda: asnumpy.concatenate([third, fourth], axis=-1),
                lambda: numpy.concatenate(
                    [
                        numpy.full((2, 2), 3.0, dtype=dtype),
                        numpy.full((2, 4), 4.0, dtype=dtype),
                    ],
                    axis=1,
                ),
            ),
            (
                "single_input",
                lambda: asnumpy.concatenate([single]),
                lambda: numpy.full((2, 3), 5.0, dtype=dtype),
            ),
        ]
    creation = getattr(asnumpy, key)
    return [
        (
            "rank_one",
            lambda: creation(4, dtype=dtype),
            lambda: numpy.full((4,), 1.0 if key == "ones" else 0.0, dtype=dtype),
        ),
        (
            "rank_two",
            lambda: creation((2, 3), dtype=dtype),
            lambda: numpy.full((2, 3), 1.0 if key == "ones" else 0.0, dtype=dtype),
        ),
        (
            "rank_three",
            lambda: creation((2, 2, 2), dtype=dtype),
            lambda: numpy.full(
                (2, 2, 2), 1.0 if key == "ones" else 0.0, dtype=dtype
            ),
        ),
    ]


def _run_numerical_gate(context: _Context) -> Dict[str, Dict[str, Any]]:
    import numpy

    targets = {}
    for key in context.entries:
        records = []
        comparisons = 0
        mismatches = 0
        elements = 0
        for name, call, oracle in _numerical_cases(key):
            record = {
                "name": name,
                "ok": False,
                "comparisons": 3,
                "detail": "",
            }
            try:
                actual = numpy.asarray(call().to_numpy())
                expected = oracle()
                case_comparisons, okay = _compare_arrays(actual, expected)
                comparisons += case_comparisons
                elements += int(expected.size)
                record["shape"] = list(expected.shape)
                record["dtype"] = str(expected.dtype)
                record["ok"] = bool(okay)
                if not okay:
                    mismatches += 1
                    record["detail"] = "tensor values, shape or dtype disagree"
            except Exception as exc:  # noqa: BLE001 - evidence must record failures
                comparisons += 3
                mismatches += 1
                record["detail"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            records.append(record)
        evidence_path = context.gates_dir / (
            "numerical_validation.%s.json" % key
        )
        _write_json(
            evidence_path,
            {
                "schema": GATE_EVIDENCE_SCHEMA,
                "schema_version": GATE_EVIDENCE_SCHEMA_VERSION,
                "gate": "numerical_validation",
                "target": key,
                "oracle": "numpy",
                "cases": records,
                "totals": {
                    "cases": len(records),
                    "oracle_comparisons": comparisons,
                    "mismatches": mismatches,
                    "elements_compared": elements,
                },
            },
        )
        claims = {
            "cases": len(records),
            "oracle_comparisons": comparisons,
            "mismatches": mismatches,
            "elements_compared": elements,
            "oracle": "numpy",
        }
        target_passed = bool(records and mismatches == 0 and comparisons > 0)
        targets[key] = {
            "status": "passed" if target_passed else "failed",
            "claims": claims,
            "evidence": {
                "files": [_evidence_ref(evidence_path, context.bundle)],
            },
        }
    return targets


def _assert_public_output(value: Any, shape: Tuple[int, ...], key: str) -> None:
    import asnumpy
    import numpy

    if not isinstance(value, asnumpy.ndarray):
        raise OfficialProducerError(
            "%s signature variant did not return asnumpy.ndarray" % key
        )
    if tuple(value.shape) != shape:
        raise OfficialProducerError("%s signature variant returned wrong shape" % key)
    if numpy.dtype(value.dtype) != numpy.dtype(numpy.float32):
        raise OfficialProducerError("%s signature variant returned wrong dtype" % key)


def _public_api_cases(key: str):
    import asnumpy
    import numpy

    dtype = numpy.float32
    if key == "concatenate":
        first = asnumpy.full((2, 3), 1.0, dtype=dtype)
        second = asnumpy.full((1, 3), 2.0, dtype=dtype)
        variants = [
            (
                (1, ("axis",)),
                lambda: asnumpy.concatenate([first, second], axis=0),
                (3, 3),
            ),
            (
                (2, ()),
                lambda: asnumpy.concatenate([first, second], 0),
                (3, 3),
            ),
            (
                (1, ()),
                lambda: asnumpy.concatenate([first, second]),
                (3, 3),
            ),
        ]
        rejections = [
            (
                "axis_out_of_range",
                lambda: asnumpy.concatenate([first, second], axis=5),
                (ValueError,),
            ),
            (
                "mixed_dtype",
                lambda: asnumpy.concatenate(
                    [
                        first,
                        asnumpy.zeros((1, 3), dtype=numpy.float64),
                    ],
                    axis=0,
                ),
                (TypeError,),
            ),
            (
                "axis_none",
                lambda: asnumpy.concatenate([first, second], axis=None),
                (TypeError,),
            ),
            (
                "numpy_input",
                lambda: asnumpy.concatenate(
                    [numpy.ones((2, 3), dtype=dtype)], axis=0
                ),
                (TypeError,),
            ),
        ]
        return variants, rejections
    creation = getattr(asnumpy, key)
    variants = [
        (
            (1, ("dtype",)),
            lambda: creation((2, 3), dtype=dtype),
            (2, 3),
        ),
        (
            (2, ()),
            lambda: creation((2, 3), dtype),
            (2, 3),
        ),
    ]
    rejections = [
        (
            "negative_shape",
            lambda: creation(-1, dtype),
            (ValueError,),
        ),
        (
            "invalid_shape_type",
            lambda: creation("bad", dtype),
            (TypeError,),
        ),
    ]
    return variants, rejections


def _run_public_api_gate(context: _Context) -> Dict[str, Dict[str, Any]]:
    import asnumpy

    module_path = Path(asnumpy.__file__).resolve().parent / "array.py"
    wrapper_sha256 = _sha256_file(module_path)
    expected_wrapper = dict(context.identity.wrapper_sha256).get("array.py")
    if wrapper_sha256 != expected_wrapper:
        raise OfficialProducerError(
            "live concatenate wrapper disagrees with the runtime identity"
        )
    try:
        wrapper_text = module_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise OfficialProducerError("live concatenate wrapper is unreadable") from exc
    lowered = wrapper_text.lower()
    marker_count = sum(lowered.count(marker) for marker in _HOST_FALLBACK_MARKERS)

    targets = {}
    for key in context.entries:
        api = getattr(asnumpy, key, None)
        callable_value = callable(api)
        variants, rejections = _public_api_cases(key)
        verified = 0
        failures = []
        admitted_signatures = {
            (item.positional_args, tuple(item.keyword_args))
            for item in _ENTRY_SPECS[key].signatures
        }
        for signature, call, shape in variants:
            try:
                value = call()
            except Exception as exc:  # noqa: BLE001 - evidence must record failures
                failures.append(
                    "%s variant raised %s" % (signature, type(exc).__name__)
                )
                continue
            try:
                _assert_public_output(value, shape, key)
            except OfficialProducerError as exc:
                failures.append(str(exc))
                continue
            if signature in admitted_signatures:
                verified += 1
        rejection_count = 0
        for name, call, expected_types in rejections:
            try:
                call()
            except expected_types:
                rejection_count += 1
            except Exception as exc:  # noqa: BLE001 - evidence must record failures
                failures.append(
                    "%s raised %s" % (name, type(exc).__name__)
                )
            else:
                failures.append("%s was not rejected" % name)
        evidence_path = context.gates_dir / (
            "public_asnumpy_api.%s.json" % key
        )
        _write_json(
            evidence_path,
            {
                "schema": GATE_EVIDENCE_SCHEMA,
                "schema_version": GATE_EVIDENCE_SCHEMA_VERSION,
                "gate": "public_asnumpy_api",
                "target": key,
                "api": "asnumpy." + key,
                "callable": callable_value,
                "signature_variants_verified": verified,
                "rejection_cases": rejection_count,
                "host_fallback_markers": marker_count,
                "wrapper_sha256": wrapper_sha256,
                "failures": failures,
            },
        )
        claims = {
            "api": "asnumpy." + key,
            "callable": callable_value,
            "signature_variants_verified": verified,
            "rejection_cases": rejection_count,
            "host_fallback_markers": marker_count,
            "wrapper_sha256": wrapper_sha256,
        }
        target_passed = bool(
            callable_value
            and not failures
            and marker_count == 0
            and verified == len(_ENTRY_SPECS[key].signatures)
            and rejection_count == len(rejections)
        )
        targets[key] = {
            "status": "passed" if target_passed else "failed",
            "claims": claims,
            "evidence": {
                "files": [_evidence_ref(evidence_path, context.bundle)],
                "failures": failures,
            },
        }
    return targets


def _device_count() -> int:
    try:
        import acl
    except Exception as exc:  # noqa: BLE001 - ACL import is environment-specific
        raise OfficialProducerError("ACL runtime is unavailable") from exc
    try:
        result = acl.rt.get_device_count()
    except Exception as exc:  # noqa: BLE001 - ACL probe is read-only
        raise OfficialProducerError("ACL device probe failed") from exc
    if isinstance(result, tuple) and len(result) == 2:
        count, returncode = result
    else:
        count, returncode = result, 0
    if returncode != 0:
        raise OfficialProducerError("ACL device probe returned an error")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise OfficialProducerError("ACL device probe returned an invalid count")
    return count


def _npu_call(key: str):
    import asnumpy
    import numpy

    dtype = numpy.float32
    if key == "concatenate":
        first = asnumpy.full((2, 3), 1.0, dtype=dtype)
        second = asnumpy.full((1, 3), 2.0, dtype=dtype)
        return lambda: asnumpy.concatenate([first, second], axis=0)
    return lambda: getattr(asnumpy, key)((2, 3), dtype)


def _run_npu_execution_gate(context: _Context) -> Dict[str, Dict[str, Any]]:
    import numpy
    import asnumpy.array as array_module

    device_count = _device_count()
    if device_count < 1:
        raise OfficialProducerError("no NPU device is visible to the producer")
    native_names = {
        "concatenate": "_concatenate",
        "ones": "_ones",
        "zeros": "_zeros",
    }
    targets = {}
    for key in context.entries:
        native_name = native_names[key]
        original = getattr(array_module, native_name)
        counts = {"value": 0}

        def counted(*args, _original=original, _counts=counts, **kwargs):
            _counts["value"] += 1
            return _original(*args, **kwargs)

        call = _npu_call(key)
        setattr(array_module, native_name, counted)
        try:
            first = call()
            second = call()
        finally:
            setattr(array_module, native_name, original)
        first_host = numpy.asarray(first.to_numpy())
        second_host = numpy.asarray(second.to_numpy())
        try:
            numpy.testing.assert_array_equal(first_host, second_host)
            stable = True
            detail = ""
        except AssertionError:
            stable = False
            detail = "repeated native calls returned different values"
        native_calls = int(counts["value"])
        host_fallback_calls = max(0, _NATIVE_CALLS_PER_TARGET - native_calls)
        evidence_path = context.gates_dir / ("npu_execution.%s.json" % key)
        _write_json(
            evidence_path,
            {
                "schema": GATE_EVIDENCE_SCHEMA,
                "schema_version": GATE_EVIDENCE_SCHEMA_VERSION,
                "gate": "npu_execution",
                "target": key,
                "device_count": device_count,
                "soc_model": context.identity.soc_model,
                "native_calls": native_calls,
                "kernel_launches": native_calls,
                "repeated_calls": _REPEATED_CALLS_PER_TARGET,
                "repeated_stable": stable,
                "host_fallback_calls": host_fallback_calls,
                "detail": detail,
            },
        )
        claims = {
            "device_count": device_count,
            "soc_model": context.identity.soc_model,
            "native_calls": native_calls,
            "kernel_launches": native_calls,
            "repeated_calls": _REPEATED_CALLS_PER_TARGET,
            "repeated_stable": stable,
            "host_fallback_calls": host_fallback_calls,
        }
        target_passed = bool(
            device_count >= 1
            and native_calls == _NATIVE_CALLS_PER_TARGET
            and native_calls >= 1
            and stable
            and host_fallback_calls == 0
        )
        targets[key] = {
            "status": "passed" if target_passed else "failed",
            "claims": claims,
            "evidence": {
                "files": [_evidence_ref(evidence_path, context.bundle)],
                "detail": detail,
            },
        }
    return targets


def _write_gate_reports(
    context: _Context,
    produced: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    refs = {}
    failed = []
    for gate in OFFICIAL_GATES:
        targets = produced[gate]
        report = build_official_gate_report(
            gate,
            {
                key: (
                    target["status"],
                    target["claims"],
                    target["evidence"],
                )
                for key, target in targets.items()
            },
        )
        path = context.gates_dir / ("%s.json" % gate)
        _replace_private_file(path, report.to_bytes())
        refs[gate] = {
            "path": path.relative_to(context.bundle).as_posix(),
            "sha256": _sha256_file(path),
        }
        if report.status != "passed":
            failed.append(gate)
    if failed:
        raise OfficialProducerError(
            "official qualification gates failed: %s" % ",".join(failed)
        )
    return refs


def _build_overlay(
    context: _Context,
    gate_refs: Mapping[str, Mapping[str, str]],
) -> OfficialCapabilityOverlay:
    entries = tuple(
        OfficialOverlayEntry(
            registry_key=key,
            qualified_api="numpy." + key,
            asnumpy_api="asnumpy." + key,
            provenance=OFFICIAL_PROVENANCE,
            constraints=_ENTRY_SPECS[key].constraints,
            signature_variants=_ENTRY_SPECS[key].signatures,
        )
        for key in context.entries
    )
    attempt_prefix = context.attempt + "/"
    transaction = context.transaction
    integration = OfficialIntegrationBinding(
        attempt=context.attempt,
        transaction_id=transaction["transaction_id"],
        transaction_sha256=transaction["transaction_sha256"],
        transaction=OfficialFileRef(
            path=attempt_prefix + "integration-transaction.json",
            sha256=context.transaction_file_sha256,
        ),
        wheel=OfficialFileRef(
            path=attempt_prefix + transaction["build"]["wheel_relative"],
            sha256=transaction["build"]["wheel_sha256"],
        ),
        gate_result=OfficialFileRef(
            path=attempt_prefix + transaction["gate"]["result_relative"],
            sha256=transaction["gate"]["result_sha256"],
        ),
        gate_log=OfficialFileRef(
            path=attempt_prefix + transaction["gate"]["log_relative"],
            sha256=transaction["gate"]["log_sha256"],
        ),
        build_log=OfficialFileRef(
            path=attempt_prefix + transaction["build"]["log_relative"],
            sha256=transaction["build"]["log_sha256"],
        ),
        candidate_test_log=OfficialFileRef(
            path=attempt_prefix + transaction["candidate_tests"]["log_relative"],
            sha256=transaction["candidate_tests"]["log_sha256"],
        ),
        source_tree_sha256=transaction["source"]["tree_sha256"],
        source_file_count=transaction["source"]["file_count"],
        source_revision=transaction["source"]["revision"],
        source_overlay_sha256=dict(transaction["source"]["overlay_sha256"]),
        runtime_core_sha256=transaction["runtime"]["core_sha256"],
    )
    runtime = OfficialRuntimeBinding(
        asnumpy_version=context.identity.asnumpy_version,
        audit_id=context.identity.audit_id,
        identity_sha256=context.identity.identity_sha256,
        status=context.identity.status,
        module_sha256=context.identity.module_sha256,
        native_module_sha256=context.identity.native_module_sha256,
        wrapper_sha256=dict(context.identity.wrapper_sha256),
        cann_version=context.identity.cann_version,
        soc_model=context.identity.soc_model,
        python_version=context.identity.python_version,
    )
    gap_spec = OfficialGapBinding(
        gap_id=transaction["gap_id"],
        request=OfficialFileRef(
            path="gap-request.json",
            sha256=context.gap_request_sha256,
        ),
        source_sha256=context.source_sha256,
    )
    source = OfficialSourceBinding(
        logical_path=context.source_logical_path,
        bundle_path=context.source_bundle_path,
        sha256=context.source_sha256,
    )
    gates = {
        name: OfficialFileRef(path=gate_refs[name]["path"], sha256=gate_refs[name]["sha256"])
        for name in OFFICIAL_GATES
    }
    payload = {
        "schema": "numpilot.official-aclnn-capability-overlay",
        "schema_version": 1,
        "producer": OFFICIAL_PRODUCER,
        "qualification_id": OFFICIAL_QUALIFICATION_PREFIX + ("0" * 20),
        "revision": context.revision,
        "parent_sha256": context.parent_sha256,
        "base_manifest_sha256": context.base_manifest_sha256,
        "runtime": runtime.to_dict(),
        "integration": integration.to_dict(),
        "gap_spec": gap_spec.to_dict(),
        "source": source.to_dict(),
        "gates": {name: gates[name].to_dict() for name in OFFICIAL_GATES},
        "entries": [entry.to_dict() for entry in entries],
    }
    payload["qualification_id"] = compute_official_qualification_id(payload)
    return OfficialCapabilityOverlay.from_dict(payload)


def _validate_preview(
    context: _Context,
    overlay: OfficialCapabilityOverlay,
    identity: Any,
) -> Dict[str, Any]:
    from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig

    descriptor, preview_name = tempfile.mkstemp(
        prefix=".capability-overlay.preview.",
        suffix=".json",
        dir=str(context.bundle),
    )
    preview = Path(preview_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(overlay.to_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            preview.unlink()
        except OSError:
            pass
        raise
    try:
        backend = NpuBackend(
            overlay_config=OverlayRuntimeConfig(
                overlay_path=str(preview),
                evidence_root=str(context.bundle),
            )
        )
        manifest = backend.capability_manifest(
            context.entries, runtime_identity=identity
        )
        admission = backend.qualified_overlay_admission(identity)
    finally:
        try:
            preview.unlink()
        except OSError:
            pass
    if admission is None:
        raise OfficialProducerError("official overlay produced no route admission")
    if admission.qualification_id != overlay.qualification_id:
        raise OfficialProducerError("admission qualification ID is inconsistent")
    if admission.base_manifest_sha256 != context.base_manifest_sha256:
        raise OfficialProducerError("admission base manifest is inconsistent")
    if admission.runtime_identity_sha256 != identity.identity_sha256:
        raise OfficialProducerError("admission runtime identity is inconsistent")
    expected_suffix = overlay.qualification_id[len(OFFICIAL_QUALIFICATION_PREFIX):]
    if not manifest.audit_id.endswith(
        "+" + OFFICIAL_OVERLAY_AUDIT_TAG + expected_suffix
    ):
        raise OfficialProducerError("admitted manifest audit ID is inconsistent")
    for key in context.entries:
        capability = manifest.operators.get(key)
        if capability is None or not capability.available:
            raise OfficialProducerError(
                "admitted manifest lost the %s capability" % key
            )
    return admission.fingerprint()


def _receipt(
    context: _Context,
    overlay: OfficialCapabilityOverlay,
    overlay_sha256: str,
    gate_refs: Mapping[str, Mapping[str, str]],
    admission: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": PRODUCER_RECEIPT_SCHEMA,
        "schema_version": PRODUCER_RECEIPT_SCHEMA_VERSION,
        "producer": OFFICIAL_PRODUCER,
        "qualification_id": overlay.qualification_id,
        "revision": context.revision,
        "parent_sha256": context.parent_sha256,
        "base_manifest_sha256": context.base_manifest_sha256,
        "runtime_identity_sha256": context.identity.identity_sha256,
        "entry_keys": list(context.entries),
        "overlay": {
            "path": "capability-overlay.json",
            "sha256": overlay_sha256,
        },
        "gate_reports": {
            name: {
                "path": gate_refs[name]["path"],
                "sha256": gate_refs[name]["sha256"],
            }
            for name in OFFICIAL_GATES
        },
        "admission": dict(admission),
        "side_effects": {
            "activation_performed": False,
            "model_invocation_performed": False,
            "network_access_performed": False,
            "build_performed": False,
            "source_tree_modified": False,
        },
    }


def _validate_entry_names(entries: Sequence[str]) -> Tuple[str, ...]:
    selected = []
    seen = set()
    for raw in entries:
        name = str(raw).strip()
        if not name:
            continue
        if name not in _ENTRY_SPECS:
            raise OfficialProducerError("unknown official entry: %s" % name)
        if name in seen:
            raise OfficialProducerError("duplicate official entry: %s" % name)
        seen.add(name)
        selected.append(name)
    if not selected:
        raise OfficialProducerError("at least one official entry is required")
    ordered = tuple(name for name in _ENTRY_ORDER if name in seen)
    if len(ordered) != len(selected):
        raise OfficialProducerError("official entry order is invalid")
    return ordered


def produce(
    *,
    run_root: Path,
    attempt: str,
    gap_request_path: Path,
    source_file: Path,
    source_logical_path: str,
    bundle_dir: str,
    entries: Sequence[str],
    candidate_test_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Produce, validate and publish one official qualification overlay."""

    run_root = Path(run_root)
    if not run_root.exists():
        raise OfficialProducerError("run root does not exist")
    _validate_private_directory(run_root, "run root")
    attempt = _relative_path(attempt, "integration attempt")
    bundle_dir = _relative_path(bundle_dir, "qualification bundle")
    source_logical_path = _relative_path(
        source_logical_path, "source logical path"
    )
    entries = _validate_entry_names(entries)

    attempt_dir = run_root / attempt
    _validate_private_directory(attempt_dir, "integration attempt")
    gap_request_path = Path(gap_request_path)
    source_file = Path(source_file)
    if not gap_request_path.is_file():
        raise OfficialProducerError("gap request does not exist")
    if not source_file.is_file():
        raise OfficialProducerError("source file does not exist")

    transaction, transaction_file_sha256 = _load_transaction(attempt_dir)
    gap_request, gap_request_sha256 = _load_gap_request(gap_request_path)
    gap_id = transaction.get("gap_id")
    if not isinstance(gap_id, str) or not gap_id:
        raise OfficialProducerError("integration transaction has no gap ID")
    spec = _matching_gap_spec(gap_request, gap_id)
    if spec.get("source_sha256") != gap_request.get("source_sha256"):
        raise OfficialProducerError("gap request and specification source disagree")
    source_sha256 = gap_request["source_sha256"]
    if spec.get("source_path") != gap_request.get("source_path"):
        raise OfficialProducerError("gap request and specification source path disagree")
    if _sha256_file(source_file) != source_sha256:
        raise OfficialProducerError(
            "source file hash does not match the bound gap request"
        )

    source_overlay = transaction.get("source", {}).get("overlay_sha256")
    if not isinstance(source_overlay, Mapping):
        raise OfficialProducerError("integration source overlay is unavailable")
    source_overlay = {str(key): str(value) for key, value in source_overlay.items()}
    test_key = "tests/asnumpy_tests/creation_tests/test_basic.py"
    candidate_test_sha256 = source_overlay.get(test_key)
    if candidate_test_sha256 is None:
        raise OfficialProducerError(
            "integration transaction does not bind the candidate unit test"
        )
    if candidate_test_file is None:
        candidate_test_file = attempt_dir / "source" / test_key
    candidate_test_file = Path(candidate_test_file)
    if not candidate_test_file.is_file():
        raise OfficialProducerError("candidate unit test file does not exist")

    wheel_relative = transaction.get("build", {}).get("wheel_relative")
    if not isinstance(wheel_relative, str) or not wheel_relative:
        raise OfficialProducerError("integration transaction has no wheel path")
    wheel_path = attempt_dir / wheel_relative
    if not wheel_path.is_file():
        raise OfficialProducerError("integrated candidate wheel does not exist")
    if _sha256_file(wheel_path) != transaction["build"]["wheel_sha256"]:
        raise OfficialProducerError("candidate wheel hash does not match the transaction")
    for section, relative_key, hash_key in (
        ("gate", "result_relative", "result_sha256"),
        ("gate", "log_relative", "log_sha256"),
        ("build", "log_relative", "log_sha256"),
        ("candidate_tests", "log_relative", "log_sha256"),
    ):
        relative = transaction.get(section, {}).get(relative_key)
        expected = transaction.get(section, {}).get(hash_key)
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or _sha256_file(attempt_dir / relative) != expected
        ):
            raise OfficialProducerError(
                "integration %s evidence is missing or drifted" % section
            )

    bundle = run_root / bundle_dir
    _ensure_private_directory(bundle)
    overlay_path = bundle / "capability-overlay.json"
    if overlay_path.exists():
        previous = load_official_overlay(str(overlay_path))
        parent_sha256 = _sha256_file(overlay_path)
        revision = previous.revision + 1
    else:
        parent_sha256 = ZERO_SHA256
        revision = 1

    source_bundle_path = "source/" + source_logical_path
    _ensure_private_directory(bundle / "integration")
    _ensure_private_directory(bundle / "integration" / PurePosixPath(attempt).name)
    attempt_bundle = bundle / attempt
    for directory in (
        attempt_bundle / "dist",
        attempt_bundle / "gate",
        attempt_bundle / "logs",
        bundle / "source",
        bundle / "source" / PurePosixPath(source_logical_path).parent,
        bundle / "gates",
        bundle / "gates" / ("r%d" % revision),
    ):
        _ensure_private_directory(directory)

    _copy_verified(
        attempt_dir / "integration-transaction.json",
        attempt_bundle / "integration-transaction.json",
        transaction_file_sha256,
        "integration transaction",
    )
    _copy_verified(
        wheel_path,
        attempt_bundle / wheel_relative,
        transaction["build"]["wheel_sha256"],
        "candidate wheel",
    )
    for section, relative_key, hash_key in (
        ("gate", "result_relative", "result_sha256"),
        ("gate", "log_relative", "log_sha256"),
        ("build", "log_relative", "log_sha256"),
        ("candidate_tests", "log_relative", "log_sha256"),
    ):
        relative = transaction[section][relative_key]
        _copy_verified(
            attempt_dir / relative,
            attempt_bundle / relative,
            transaction[section][hash_key],
            "integration %s evidence" % section,
        )
    _copy_verified(
        gap_request_path,
        bundle / "gap-request.json",
        gap_request_sha256,
        "gap request",
    )
    _copy_verified(
        source_file,
        bundle / source_bundle_path,
        source_sha256,
        "official source binding",
    )

    backend, identity = _collect_runtime_identity()
    base_manifest_sha256 = _base_manifest_sha256(backend, entries, identity)
    context = _Context(
        run_root=run_root,
        bundle=bundle,
        attempt=attempt,
        attempt_dir=attempt_dir,
        revision=revision,
        parent_sha256=parent_sha256,
        gates_dir=bundle / "gates" / ("r%d" % revision),
        entries=entries,
        identity=identity,
        transaction=transaction,
        transaction_file_sha256=transaction_file_sha256,
        wheel_path=wheel_path,
        runtime_core_sha256=transaction["runtime"]["core_sha256"],
        source_overlay_sha256=source_overlay,
        base_manifest_sha256=base_manifest_sha256,
        source_file=source_file,
        source_logical_path=source_logical_path,
        source_bundle_path=source_bundle_path,
        source_sha256=source_sha256,
        gap_request=gap_request_path,
        gap_request_sha256=gap_request_sha256,
        candidate_test_file=candidate_test_file,
        candidate_test_sha256=candidate_test_sha256,
    )

    produced = {
        "artifact_integrity": {
            key: {
                "status": "passed",
                "claims": _artifact_integrity_claims(context),
                "evidence": {},
            }
            for key in entries
        },
        "build": _run_build_gate(context),
        "unit_test": _run_unit_test_gate(context),
        "numerical_validation": _run_numerical_gate(context),
        "public_asnumpy_api": _run_public_api_gate(context),
        "npu_execution": _run_npu_execution_gate(context),
    }
    gate_refs = _write_gate_reports(context, produced)
    overlay = _build_overlay(context, gate_refs)
    preview_admission = _validate_preview(context, overlay, identity)

    overlay_sha256 = publish_official_overlay_atomic(
        str(overlay_path),
        overlay,
        expected_previous_sha256=parent_sha256,
    )
    if _sha256_file(overlay_path) != overlay_sha256:
        raise OfficialProducerError("published overlay hash verification failed")
    if official_overlay_file_sha256(overlay) != overlay_sha256:
        raise OfficialProducerError("published overlay is not canonical")

    from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig

    published_backend = NpuBackend(
        overlay_config=OverlayRuntimeConfig(
            overlay_path=str(overlay_path),
            evidence_root=str(bundle),
        )
    )
    published_manifest = published_backend.capability_manifest(
        entries, runtime_identity=identity
    )
    published_admission = published_backend.qualified_overlay_admission(identity)
    if published_admission is None:
        raise OfficialProducerError("published overlay produced no route admission")
    if published_admission.fingerprint() != preview_admission:
        raise OfficialProducerError(
            "published overlay admission differs from the validated preview"
        )
    for key in entries:
        capability = published_manifest.operators.get(key)
        if capability is None or not capability.available:
            raise OfficialProducerError(
                "published manifest lost the %s capability" % key
            )

    receipt = _receipt(
        context,
        overlay,
        overlay_sha256,
        gate_refs,
        published_admission.fingerprint(),
    )
    receipt_name = "qualification-receipt.r%d.json" % revision
    _write_json(bundle / receipt_name, receipt)
    _write_json(bundle / "production-receipt.json", receipt)
    return {
        "status": "published",
        "producer": OFFICIAL_PRODUCER,
        "qualification_id": overlay.qualification_id,
        "revision": revision,
        "parent_sha256": parent_sha256,
        "overlay_sha256": overlay_sha256,
        "base_manifest_sha256": base_manifest_sha256,
        "runtime_identity_sha256": identity.identity_sha256,
        "entry_keys": list(entries),
        "gates": {name: "passed" for name in OFFICIAL_GATES},
        "admitted_audit_id": published_admission.audit_id,
        "bundle_dir": bundle_dir,
        "receipt": receipt_name,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce one official ACLNN qualification overlay from an "
            "integrated candidate attempt."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--attempt",
        default="integration/attempt-04",
        help="integration attempt directory relative to the run root",
    )
    parser.add_argument(
        "--gap-request",
        default="gap-request.json",
        help="gap request path (absolute or relative to the run root)",
    )
    parser.add_argument("--source-file", required=True)
    parser.add_argument(
        "--source-logical-path",
        required=True,
        help="public logical path bound into the overlay, e.g. examples/x.py",
    )
    parser.add_argument("--bundle-dir", default="qualification")
    parser.add_argument("--entries", default=",".join(_ENTRY_ORDER))
    parser.add_argument("--candidate-test-file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    gap_request = Path(args.gap_request)
    if not gap_request.is_absolute():
        gap_request = run_root / gap_request
    try:
        result = produce(
            run_root=run_root,
            attempt=args.attempt,
            gap_request_path=gap_request,
            source_file=Path(args.source_file),
            source_logical_path=args.source_logical_path,
            bundle_dir=args.bundle_dir,
            entries=tuple(part for part in args.entries.split(",")),
            candidate_test_file=(
                Path(args.candidate_test_file)
                if args.candidate_test_file
                else None
            ),
        )
    except (
        OfficialProducerError,
        QualificationError,
        OSError,
        ValueError,
    ) as exc:
        print(
            "official ACLNN qualification failed: %s" % exc,
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(
            "[资格] status=%s revision=%d qualification=%s"
            % (result["status"], result["revision"], result["qualification_id"])
        )
        print(
            "[资格] entries=%s runtime=%s"
            % (",".join(result["entry_keys"]), result["runtime_identity_sha256"])
        )
        print(
            "[资格] overlay=%s receipt=%s"
            % (result["overlay_sha256"], result["receipt"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
