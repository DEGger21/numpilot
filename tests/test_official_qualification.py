"""Tests for the explicit official-ACLNN qualification adapter.

The fixture is synthetic on purpose: every hash, wheel, report and runtime
identity is built in a temporary directory, so the tests exercise the
fail-closed schema and publisher without touching a device, a model or the
real AsNumpy runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple
import zipfile

import pytest

from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    OperatorSignature,
    parse_profile_constraints,
)
from numpilot.gaps import official_producer as producer
from numpilot.gaps.official_qualification import (
    OFFICIAL_GATE_SCHEMA,
    OFFICIAL_OVERLAY_SCHEMA,
    OFFICIAL_PRODUCER,
    OFFICIAL_QUALIFICATION_PREFIX,
    OFFICIAL_SCHEMA_VERSION,
    ZERO_SHA256,
    OfficialCapabilityOverlay,
    OfficialGateReport,
    apply_official_overlay,
    build_official_gate_report,
    compute_official_qualification_id,
    load_official_overlay,
    publish_official_overlay_atomic,
)
from numpilot.gaps.qualification import (
    QualificationError,
    capability_manifest_sha256,
)
from numpilot.gaps.official_qualification import _transaction_sha256


ENTRY_KEYS = ("concatenate", "ones", "zeros")
CORE_MEMBER = "asnumpy/_core.cpython-312-aarch64-linux-gnu.so"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(data: Any) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    path.write_bytes(payload)
    os.chmod(path, mode)


def _write_json(path: Path, data: Any, *, mode: int = 0o600) -> None:
    _write_bytes(path, _json_bytes(data), mode=mode)


def _wheel_bytes(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


@dataclass(frozen=True)
class _Identity:
    asnumpy_version: str
    audit_id: str
    identity_sha256: str
    status: str
    module_sha256: str
    native_module_sha256: str
    wrapper_sha256: Mapping[str, str]
    cann_version: str
    soc_model: str
    python_version: str


@dataclass(frozen=True)
class Fixture:
    root: Path
    base: CapabilityManifest
    identity: _Identity
    overlay: OfficialCapabilityOverlay
    transaction: Mapping[str, Any]
    transaction_file_sha256: str
    wheel_path: Path
    wheel_sha256: str
    wheel_member_sha256: Mapping[str, str]
    source_path: Path
    source_sha256: str
    gap_request_path: Path
    gap_request_sha256: str
    candidate_test_path: Path
    candidate_test_sha256: str
    gate_refs: Mapping[str, Mapping[str, str]]
    runtime_core_sha256: str
    source_overlay_sha256: Mapping[str, str]


_ENTRY_CONSTRAINTS = {
    "concatenate": (
        "dtype=float32",
        "rank>=2",
        "rank<=2",
        "contiguous=true",
    ),
    "ones": (
        "dtype=float32",
        "rank>=1",
        "rank<=3",
        "contiguous=true",
    ),
    "zeros": (
        "dtype=float32",
        "rank>=1",
        "rank<=3",
        "contiguous=true",
    ),
}
_ENTRY_SIGNATURES = {
    "concatenate": (
        OperatorSignature(positional_args=1, keyword_args=("axis",)),
        OperatorSignature(positional_args=2, keyword_args=()),
        OperatorSignature(positional_args=1, keyword_args=()),
    ),
    "ones": (
        OperatorSignature(positional_args=1, keyword_args=("dtype",)),
        OperatorSignature(positional_args=2, keyword_args=()),
    ),
    "zeros": (
        OperatorSignature(positional_args=1, keyword_args=("dtype",)),
        OperatorSignature(positional_args=2, keyword_args=()),
    ),
}


def _entries() -> Tuple[Any, ...]:
    from numpilot.gaps.official_qualification import OfficialOverlayEntry

    return tuple(
        OfficialOverlayEntry(
            registry_key=key,
            qualified_api="numpy." + key,
            asnumpy_api="asnumpy." + key,
            provenance="native_npu",
            constraints=_ENTRY_CONSTRAINTS[key],
            signature_variants=_ENTRY_SIGNATURES[key],
        )
        for key in ENTRY_KEYS
    )


def _build_fixture(root: Path, *, revision: int = 1, parent: str = ZERO_SHA256) -> Fixture:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    attempt = "integration/attempt-04"
    attempt_dir = root / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(attempt_dir, 0o700)

    wrapper = b"class ndarray:\n    pass\n\ndef concatenate(values, axis=0):\n    raise NotImplementedError\n"
    init = b"from .array import *\n"
    module = b"# synthetic asnumpy package module\n"
    native = b"\x7fELF-synthetic-native-core"
    wrapper_sha = _sha256(wrapper)
    init_sha = _sha256(init)
    module_sha = _sha256(module)
    native_sha = _sha256(native)
    members = {
        "asnumpy/array.py": wrapper,
        "asnumpy/__init__.py": init,
        CORE_MEMBER: native,
        "asnumpy-0.4.1.dist-info/METADATA": b"Name: asnumpy\n",
    }
    member_sha256 = {name: _sha256(payload) for name, payload in members.items()}
    wheel_bytes = _wheel_bytes(members)
    wheel_path = attempt_dir / "dist/candidate.whl"
    _write_bytes(wheel_path, wheel_bytes)
    wheel_sha = _sha256(wheel_bytes)

    gate_result = _json_bytes({"gate": "passed"})
    gate_log = b"gate passed\n"
    build_log = b"build passed\n"
    candidate_log = b"candidate tests passed\n"
    gate_result_path = attempt_dir / "gate/result.json"
    gate_log_path = attempt_dir / "logs/gate.log"
    build_log_path = attempt_dir / "logs/build.log"
    candidate_log_path = attempt_dir / "logs/candidate-tests.log"
    _write_bytes(gate_result_path, gate_result)
    _write_bytes(gate_log_path, gate_log)
    _write_bytes(build_log_path, build_log)
    _write_bytes(candidate_log_path, candidate_log)

    source_text = (
        "import numpy as np\n"
        "x = np.ones((3, 2), dtype=np.float32)\n"
        "y = np.zeros((3, 4), dtype=np.float32)\n"
        "joined = np.concatenate((x, y), axis=1)\n"
        "print(joined)\n"
    )
    source_bytes = source_text.encode("utf-8")
    source_sha = _sha256(source_bytes)
    source_path = root / "source/examples/demo.py"
    _write_bytes(source_path, source_bytes)

    gap_id = "gap-concatenate-0001"
    gap_request = {
        "request_type": "script_api_gap",
        "source_path": "examples/demo.py",
        "source_sha256": source_sha,
        "gap_specs": [
            {
                "gap_id": gap_id,
                "source_sha256": source_sha,
                "source_path": "examples/demo.py",
                "qualified_api": "numpy.concatenate",
                "asnumpy_api": "asnumpy.concatenate",
                "signature_variants": [
                    {"positional_args": 1, "keyword_args": ["axis"]}
                ],
            }
        ],
    }
    gap_request_path = root / "gap-request.json"
    _write_json(gap_request_path, gap_request)
    gap_request_sha = _sha256(gap_request_path.read_bytes())

    source_overlay = {
        "src/asnumpy/array.py": wrapper_sha,
        "src/asnumpy/__init__.py": init_sha,
        "tests/asnumpy_tests/creation_tests/test_basic.py": _sha256(
            b"synthetic candidate test"
        ),
    }
    transaction: Dict[str, Any] = {
        "record_type": "numpilot_integration_transaction",
        "schema_version": 1,
        "status": "passed",
        "failure": None,
        "transaction_id": "tx-20261104-0001",
        "gap_id": gap_id,
        "build": {
            "status": "passed",
            "returncode": 0,
            "wheel_sha256": wheel_sha,
            "wheel_relative": "dist/candidate.whl",
            "wheel_filename": "candidate.whl",
            "log_relative": "logs/build.log",
            "log_sha256": _sha256(build_log),
        },
        "gate": {
            "status": "passed",
            "returncode": 0,
            "result_relative": "gate/result.json",
            "result_sha256": _sha256(gate_result),
            "log_relative": "logs/gate.log",
            "log_sha256": _sha256(gate_log),
        },
        "candidate_tests": {
            "status": "passed",
            "returncode": 0,
            "log_relative": "logs/candidate-tests.log",
            "log_sha256": _sha256(candidate_log),
        },
        "source": {
            "tree_sha256": _sha256(b"synthetic source tree"),
            "file_count": 1,
            "revision": "a" * 40,
            "overlay_sha256": source_overlay,
        },
        "runtime": {"core_sha256": native_sha},
    }
    transaction["transaction_sha256"] = _transaction_sha256(transaction)
    transaction_path = attempt_dir / "integration-transaction.json"
    _write_json(transaction_path, transaction)
    transaction_file_sha = _sha256(transaction_path.read_bytes())

    identity = _Identity(
        asnumpy_version="0.4.1.post0+numpilot.test",
        audit_id="unaudited-asnumpy-version",
        identity_sha256="1" * 64,
        status="audit_required",
        module_sha256=module_sha,
        native_module_sha256=native_sha,
        wrapper_sha256={"array.py": wrapper_sha, "__init__.py": init_sha},
        cann_version="9.2.0-beta.2",
        soc_model="Ascend910B3",
        python_version="3.12.14",
    )
    base = CapabilityManifest(
        backend="asnumpy",
        backend_version=identity.asnumpy_version,
        operators={
            key: OperatorCapability(
                name=key,
                available=True,
                provenance=ExecutionProvenance.UNKNOWN,
                evidence="runtime version has no provenance audit",
            )
            for key in ENTRY_KEYS
        },
        environment={
            "cann_version": identity.cann_version,
            "soc_model": identity.soc_model,
            "python_version": identity.python_version,
        },
        audit_id=identity.audit_id,
    )
    base_sha = capability_manifest_sha256(base)

    junit = (
        b'<?xml version="1.0"?><testsuites><testsuite name="candidate">'
        b'<testcase name="test_concatenate_contract"/>'
        b'<testcase name="test_ones_contract"/>'
        b'<testcase name="test_zeros_contract"/>'
        b"</testsuite></testsuites>"
    )
    junit_path = root / "gates/unit_test.junit.xml"
    _write_bytes(junit_path, junit)
    junit_sha = _sha256(junit)
    numerical_path = root / "gates/numerical_validation.evidence.json"
    public_path = root / "gates/public_asnumpy_api.evidence.json"
    npu_path = root / "gates/npu_execution.evidence.json"
    _write_json(numerical_path, {"oracle": "numpy", "cases": 3})
    _write_json(public_path, {"api": "asnumpy", "variants": 1})
    _write_json(npu_path, {"device": "Ascend910B3"})
    numerical_sha = _sha256(numerical_path.read_bytes())
    public_sha = _sha256(public_path.read_bytes())
    npu_sha = _sha256(npu_path.read_bytes())

    artifact_claims = {
        "transaction_file_sha256": transaction_file_sha,
        "transaction_sha256": transaction["transaction_sha256"],
        "wheel_sha256": wheel_sha,
        "runtime_identity_sha256": identity.identity_sha256,
        "source_sha256": source_sha,
        "gap_request_sha256": gap_request_sha,
        "base_manifest_sha256": base_sha,
        "candidate_test_log_sha256": _sha256(candidate_log),
        "gate_result_sha256": _sha256(gate_result),
        "gate_log_sha256": _sha256(gate_log),
        "source_overlay_sha256": source_overlay,
        "runtime_core_sha256": native_sha,
    }
    build_claims = {
        "build_status": "passed",
        "wheel_sha256": wheel_sha,
        "wheel_member_count": len(members),
        "wheel_member_sha256": member_sha256,
    }

    def unit_claims(key: str) -> Dict[str, Any]:
        return {
            "returncode": 0,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "junit_sha256": junit_sha,
            "test_names": ["test_%s_contract" % key],
        }

    def numerical_claims(key: str) -> Dict[str, Any]:
        return {
            "cases": 1,
            "oracle_comparisons": 1,
            "mismatches": 0,
            "elements_compared": 6,
            "oracle": "numpy",
        }

    def public_claims(key: str) -> Dict[str, Any]:
        return {
            "api": "asnumpy." + key,
            "callable": True,
            "signature_variants_verified": 1,
            "rejection_cases": 2,
            "host_fallback_markers": 0,
            "wrapper_sha256": wrapper_sha,
        }

    def npu_claims(key: str) -> Dict[str, Any]:
        return {
            "device_count": 1,
            "soc_model": identity.soc_model,
            "native_calls": 2,
            "kernel_launches": 2,
            "repeated_calls": 2,
            "repeated_stable": True,
            "host_fallback_calls": 0,
        }

    gate_claims = {
        "artifact_integrity": lambda key: dict(artifact_claims),
        "build": lambda key: dict(build_claims),
        "unit_test": unit_claims,
        "numerical_validation": numerical_claims,
        "public_asnumpy_api": public_claims,
        "npu_execution": npu_claims,
    }
    gate_evidence = {
        "artifact_integrity": {},
        "build": {},
        "unit_test": {
            "files": [
                {"path": "gates/unit_test.junit.xml", "sha256": junit_sha}
            ],
            "marker": "candidate",
        },
        "numerical_validation": {
            "files": [
                {
                    "path": "gates/numerical_validation.evidence.json",
                    "sha256": numerical_sha,
                }
            ]
        },
        "public_asnumpy_api": {
            "files": [
                {
                    "path": "gates/public_asnumpy_api.evidence.json",
                    "sha256": public_sha,
                }
            ]
        },
        "npu_execution": {
            "files": [
                {
                    "path": "gates/npu_execution.evidence.json",
                    "sha256": npu_sha,
                }
            ]
        },
    }

    gate_refs: Dict[str, Dict[str, str]] = {}
    for gate in (
        "artifact_integrity",
        "build",
        "unit_test",
        "numerical_validation",
        "public_asnumpy_api",
        "npu_execution",
    ):
        report = build_official_gate_report(
            gate,
            {
                key: (
                    "passed",
                    gate_claims[gate](key),
                    gate_evidence[gate],
                )
                for key in ENTRY_KEYS
            },
        )
        payload = report.to_bytes()
        path = root / ("gates/%s.json" % gate)
        _write_bytes(path, payload)
        gate_refs[gate] = {
            "path": "gates/%s.json" % gate,
            "sha256": _sha256(payload),
        }

    payload = {
        "schema": OFFICIAL_OVERLAY_SCHEMA,
        "schema_version": OFFICIAL_SCHEMA_VERSION,
        "producer": OFFICIAL_PRODUCER,
        "qualification_id": OFFICIAL_QUALIFICATION_PREFIX + ("0" * 20),
        "revision": revision,
        "parent_sha256": parent,
        "base_manifest_sha256": base_sha,
        "runtime": {
            "asnumpy_version": identity.asnumpy_version,
            "audit_id": identity.audit_id,
            "identity_sha256": identity.identity_sha256,
            "status": identity.status,
            "module_sha256": identity.module_sha256,
            "native_module_sha256": identity.native_module_sha256,
            "wrapper_sha256": dict(identity.wrapper_sha256),
            "cann_version": identity.cann_version,
            "soc_model": identity.soc_model,
            "python_version": identity.python_version,
        },
        "integration": {
            "attempt": attempt,
            "transaction_id": transaction["transaction_id"],
            "transaction_sha256": transaction["transaction_sha256"],
            "transaction": {
                "path": attempt + "/integration-transaction.json",
                "sha256": transaction_file_sha,
            },
            "wheel": {
                "path": attempt + "/dist/candidate.whl",
                "sha256": wheel_sha,
            },
            "gate_result": {
                "path": attempt + "/gate/result.json",
                "sha256": _sha256(gate_result),
            },
            "gate_log": {
                "path": attempt + "/logs/gate.log",
                "sha256": _sha256(gate_log),
            },
            "build_log": {
                "path": attempt + "/logs/build.log",
                "sha256": _sha256(build_log),
            },
            "candidate_test_log": {
                "path": attempt + "/logs/candidate-tests.log",
                "sha256": _sha256(candidate_log),
            },
            "source_tree_sha256": transaction["source"]["tree_sha256"],
            "source_file_count": transaction["source"]["file_count"],
            "source_revision": transaction["source"]["revision"],
            "source_overlay_sha256": dict(source_overlay),
            "runtime_core_sha256": native_sha,
        },
        "gap_spec": {
            "gap_id": gap_id,
            "request": {
                "path": "gap-request.json",
                "sha256": gap_request_sha,
            },
            "source_sha256": source_sha,
        },
        "source": {
            "logical_path": "examples/demo.py",
            "bundle_path": "source/examples/demo.py",
            "sha256": source_sha,
        },
        "gates": {
            name: dict(gate_refs[name])
            for name in (
                "artifact_integrity",
                "build",
                "unit_test",
                "numerical_validation",
                "public_asnumpy_api",
                "npu_execution",
            )
        },
        "entries": [entry.to_dict() for entry in _entries()],
    }
    payload["qualification_id"] = compute_official_qualification_id(payload)
    overlay = OfficialCapabilityOverlay.from_dict(payload)

    return Fixture(
        root=root,
        base=base,
        identity=identity,
        overlay=overlay,
        transaction=transaction,
        transaction_file_sha256=transaction_file_sha,
        wheel_path=wheel_path,
        wheel_sha256=wheel_sha,
        wheel_member_sha256=member_sha256,
        source_path=source_path,
        source_sha256=source_sha,
        gap_request_path=gap_request_path,
        gap_request_sha256=gap_request_sha,
        candidate_test_path=candidate_log_path,
        candidate_test_sha256=_sha256(candidate_log),
        gate_refs=gate_refs,
        runtime_core_sha256=native_sha,
        source_overlay_sha256=source_overlay,
    )


def _apply(
    fixture: Fixture,
    *,
    base: CapabilityManifest | None = None,
    overlay: OfficialCapabilityOverlay | None = None,
    identity: _Identity | None = None,
    trusted: Tuple[str, ...] = (OFFICIAL_PRODUCER,),
) -> CapabilityManifest:
    return apply_official_overlay(
        fixture.base if base is None else base,
        fixture.overlay if overlay is None else overlay,
        evidence_root=str(fixture.root),
        trusted_producers=trusted,
        current_identity=fixture.identity if identity is None else identity,
    )


def _rebuild_with_gate(
    fixture: Fixture,
    gate: str,
    mutate,
) -> OfficialCapabilityOverlay:
    path = fixture.root / ("gates/%s.json" % gate)
    report = json.loads(path.read_bytes())
    mutate(report)
    payload = _json_bytes(report)
    _write_bytes(path, payload)
    data = fixture.overlay.to_dict()
    data["gates"][gate] = {
        "path": "gates/%s.json" % gate,
        "sha256": _sha256(payload),
    }
    data["qualification_id"] = compute_official_qualification_id(data)
    return OfficialCapabilityOverlay.from_dict(data)


def test_roundtrip_and_apply_promotes_unknown_to_native(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    overlay = OfficialCapabilityOverlay.from_dict(fixture.overlay.to_dict())

    assert overlay.to_dict() == fixture.overlay.to_dict()
    assert overlay.qualification_id == compute_official_qualification_id(
        overlay.to_dict()
    )
    assert overlay.entry_keys == ENTRY_KEYS

    manifest = _apply(fixture)
    suffix = overlay.qualification_id[len(OFFICIAL_QUALIFICATION_PREFIX):]
    assert manifest.audit_id.endswith("+official-qualification-" + suffix)
    assert manifest.audit_id.startswith(fixture.base.audit_id + "+")
    for key in ENTRY_KEYS:
        capability = manifest.operators[key]
        assert capability.available is True
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert capability.constraints == _ENTRY_CONSTRAINTS[key]
        assert capability.signature_variants == _ENTRY_SIGNATURES[key]
        assert capability.evidence == (
            "official ACLNN qualification %s" % overlay.qualification_id
        )
    assert manifest.backend == fixture.base.backend
    assert dict(manifest.environment) == dict(fixture.base.environment)

    again = _apply(fixture)
    assert again.to_dict() == manifest.to_dict()


def test_apply_rejects_runtime_identity_drift(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")

    with pytest.raises(QualificationError, match="runtime soc_model drift"):
        _apply(fixture, identity=replace(fixture.identity, soc_model="Ascend9999"))
    with pytest.raises(QualificationError, match="runtime identity drift"):
        _apply(
            fixture,
            identity=replace(fixture.identity, identity_sha256="9" * 64),
        )
    with pytest.raises(QualificationError, match="runtime wrapper drift"):
        _apply(
            fixture,
            identity=replace(
                fixture.identity,
                wrapper_sha256={
                    "array.py": fixture.identity.wrapper_sha256["array.py"],
                    "__init__.py": "8" * 64,
                },
            ),
        )


def test_apply_rejects_base_manifest_drift(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    drifted = CapabilityManifest(
        backend=fixture.base.backend,
        backend_version=fixture.base.backend_version,
        operators=dict(fixture.base.operators),
        environment={
            **dict(fixture.base.environment),
            "cann_version": "9.0.0",
        },
        audit_id=fixture.base.audit_id,
    )

    with pytest.raises(QualificationError, match="base manifest drift"):
        _apply(fixture, base=drifted)


def test_apply_rejects_untrusted_producer(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")

    with pytest.raises(QualificationError, match="producer is not trusted"):
        _apply(fixture, trusted=("some-other-producer",))


def test_apply_rejects_gate_claim_semantic_tamper(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")

    def mutate(report):
        for target in report["targets"].values():
            target["claims"]["soc_model"] = "Ascend0000"

    tampered = _rebuild_with_gate(fixture, "npu_execution", mutate)
    with pytest.raises(QualificationError, match="SoC model mismatch"):
        _apply(fixture, overlay=tampered)


def test_apply_rejects_failed_gate_report(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")

    def mutate(report):
        for target in report["targets"].values():
            target["status"] = "failed"
        report["status"] = "failed"

    tampered = _rebuild_with_gate(fixture, "npu_execution", mutate)
    with pytest.raises(QualificationError, match="did not pass"):
        _apply(fixture, overlay=tampered)


def test_apply_rejects_gate_evidence_tamper(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    evidence = fixture.root / "gates/numerical_validation.evidence.json"
    _write_bytes(evidence, b'{"oracle": "tampered"}')

    with pytest.raises(QualificationError, match="hash does not match"):
        _apply(fixture)


def test_apply_rejects_wheel_drift(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    _write_bytes(fixture.wheel_path, b"not the built wheel")

    with pytest.raises(QualificationError, match="wheel.*hash does not match"):
        _apply(fixture)


def test_overlay_qualification_id_is_content_bound(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    payload = fixture.overlay.to_dict()
    payload["revision"] = 2

    with pytest.raises(QualificationError, match="does not derive from its content"):
        OfficialCapabilityOverlay.from_dict(payload)


def test_publisher_revisions_and_compare_and_swap(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    target = tmp_path / "publish/overlay.json"

    first_sha = publish_official_overlay_atomic(str(target), fixture.overlay)
    assert first_sha == fixture.overlay.content_sha256
    loaded = load_official_overlay(str(target))
    assert loaded.to_dict() == fixture.overlay.to_dict()

    payload = fixture.overlay.to_dict()
    payload["revision"] = 2
    payload["parent_sha256"] = first_sha
    payload["qualification_id"] = compute_official_qualification_id(payload)
    second = OfficialCapabilityOverlay.from_dict(payload)
    second_sha = publish_official_overlay_atomic(
        str(target), second, expected_previous_sha256=first_sha
    )
    assert second_sha == second.content_sha256
    assert load_official_overlay(str(target)).revision == 2

    with pytest.raises(QualificationError, match="compare-and-swap"):
        publish_official_overlay_atomic(
            str(target), second, expected_previous_sha256="f" * 64
        )


def test_publisher_rejects_a_first_revision_above_one(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle", revision=2)
    target = tmp_path / "publish/overlay.json"

    with pytest.raises(QualificationError, match="revision 1"):
        publish_official_overlay_atomic(str(target), fixture.overlay)


def test_producer_build_overlay_matches_the_fixture(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "bundle")
    context = producer._Context(
        run_root=fixture.root,
        bundle=fixture.root,
        attempt="integration/attempt-04",
        attempt_dir=fixture.root / "integration/attempt-04",
        revision=1,
        parent_sha256=ZERO_SHA256,
        gates_dir=fixture.root / "gates",
        entries=ENTRY_KEYS,
        identity=fixture.identity,
        transaction=fixture.transaction,
        transaction_file_sha256=fixture.transaction_file_sha256,
        wheel_path=fixture.wheel_path,
        runtime_core_sha256=fixture.runtime_core_sha256,
        source_overlay_sha256=fixture.source_overlay_sha256,
        base_manifest_sha256=capability_manifest_sha256(fixture.base),
        source_file=fixture.source_path,
        source_logical_path="examples/demo.py",
        source_bundle_path="source/examples/demo.py",
        source_sha256=fixture.source_sha256,
        gap_request=fixture.gap_request_path,
        gap_request_sha256=fixture.gap_request_sha256,
        candidate_test_file=fixture.candidate_test_path,
        candidate_test_sha256=fixture.candidate_test_sha256,
    )

    built = producer._build_overlay(context, fixture.gate_refs)
    assert built.to_dict() == fixture.overlay.to_dict()
    assert built.qualification_id == fixture.overlay.qualification_id


def test_producer_entry_names_are_canonical_and_deduplicated() -> None:
    assert producer._validate_entry_names(("zeros", "ones")) == ("ones", "zeros")
    assert producer._validate_entry_names(("concatenate", "")) == ("concatenate",)
    with pytest.raises(producer.OfficialProducerError, match="unknown"):
        producer._validate_entry_names(("matmul",))
    with pytest.raises(producer.OfficialProducerError, match="duplicate"):
        producer._validate_entry_names(("ones", "ones"))
    with pytest.raises(producer.OfficialProducerError, match="at least one"):
        producer._validate_entry_names((" ",))


def test_producer_relative_paths_are_bounded() -> None:
    assert producer._relative_path("a/b", "path") == "a/b"
    for value in ("", ".", "/etc/passwd", "a/../b"):
        with pytest.raises(producer.OfficialProducerError):
            producer._relative_path(value, "path")


def test_producer_evidence_refs_stay_under_gates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    evidence = bundle / "gates/evidence.json"
    _write_bytes(evidence, b"{}\n")

    assert producer._evidence_ref(evidence, bundle) == {
        "path": "gates/evidence.json",
        "sha256": _sha256(b"{}\n"),
    }
    outside = bundle / "source/evidence.json"
    _write_bytes(outside, b"{}\n")
    with pytest.raises(producer.OfficialProducerError, match="gates/"):
        producer._evidence_ref(outside, bundle)
    with pytest.raises(producer.OfficialProducerError, match="escaped"):
        producer._evidence_ref(tmp_path / "elsewhere.json", bundle)


def test_producer_replaces_only_private_single_link_files(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    producer._replace_private_file(target, b"one")
    producer._replace_private_file(target, b"two")
    assert target.read_bytes() == b"two"
    assert target.stat().st_mode & 0o777 == 0o600

    victim = tmp_path / "victim.json"
    _write_bytes(victim, b"victim")
    symlink = tmp_path / "symlink.json"
    os.symlink(victim, symlink)
    with pytest.raises(producer.OfficialProducerError, match="non-private"):
        producer._replace_private_file(symlink, b"x")
    assert victim.read_bytes() == b"victim"

    hardlink = tmp_path / "hardlink.json"
    _write_bytes(hardlink, b"linked")
    os.link(hardlink, tmp_path / "other.json")
    with pytest.raises(producer.OfficialProducerError, match="non-private"):
        producer._replace_private_file(hardlink, b"x")

    writable = tmp_path / "writable.json"
    _write_bytes(writable, b"writable", mode=0o620)
    with pytest.raises(producer.OfficialProducerError, match="non-private"):
        producer._replace_private_file(writable, b"x")


def test_producer_parses_junit_counts(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_bytes(
        junit,
        b'<?xml version="1.0"?><testsuites><testsuite>'
        b'<testcase name="test_concatenate_contract"/>'
        b'<testcase name="test_ones_contract"/>'
        b'<testcase name="test_skipped"><skipped/></testcase>'
        b"</testsuite></testsuites>",
    )

    stats = producer._parse_junit(junit)
    assert stats == {
        "tests": 3,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
        "passed": 2,
        "test_names": [
            "test_concatenate_contract",
            "test_ones_contract",
            "test_skipped",
        ],
    }

    broken = tmp_path / "broken.xml"
    _write_bytes(broken, b"<not-xml")
    with pytest.raises(producer.OfficialProducerError, match="JUnit"):
        producer._parse_junit(broken)


def test_producer_loads_only_a_passed_transaction(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    transaction = {
        "record_type": "numpilot_integration_transaction",
        "schema_version": 1,
        "status": "passed",
        "failure": None,
        "transaction_id": "tx-test",
        "gap_id": "gap-test",
    }
    transaction["transaction_sha256"] = _transaction_sha256(transaction)
    path = attempt / "integration-transaction.json"
    _write_json(path, transaction)

    loaded, file_sha = producer._load_transaction(attempt)
    assert loaded == transaction
    assert file_sha == _sha256(path.read_bytes())

    failed = dict(transaction)
    failed["status"] = "failed"
    failed["failure"] = "synthetic failure"
    failed["transaction_sha256"] = _transaction_sha256(failed)
    _write_json(path, failed)
    with pytest.raises(producer.OfficialProducerError, match="did not pass"):
        producer._load_transaction(attempt)


def test_producer_entry_specs_are_valid_scoped_profiles() -> None:
    assert producer._ENTRY_ORDER[: len(ENTRY_KEYS)] == ENTRY_KEYS
    for key in ENTRY_KEYS:
        spec = producer._ENTRY_SPECS[key]
        parsed = parse_profile_constraints(spec.constraints)
        assert parsed["dtype"] == "float32"
        assert spec.signatures
        assert len(set(spec.signatures)) == len(spec.signatures)
        assert spec.test_marker == key
    assert producer._API_LANE == {
        "invert",
        "argsort",
        "searchsorted",
        "median",
    }
    for key in sorted(producer._API_LANE):
        assert key in producer._ENTRY_ORDER
        spec = producer._ENTRY_SPECS[key]
        parsed = parse_profile_constraints(spec.constraints)
        assert parsed["dtype"] in {"float32", "bool"}
        assert parsed["contiguous"] is True
        assert spec.signatures
        assert len(set(spec.signatures)) == len(spec.signatures)
        assert spec.test_marker is None
        assert producer._TEST_FILES[key].startswith("tests/asnumpy_tests/")
        assert producer._API_WRAPPER[key].endswith(".py")
        assert producer._API_NATIVE[key][1].startswith("_")
    assert parse_profile_constraints(
        producer._ENTRY_SPECS["invert"].constraints
    )["dtype"] == "bool"
