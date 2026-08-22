"""Fail-closed tests for offline gap qualification and overlay admission."""

from copy import deepcopy
from dataclasses import dataclass, replace
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    OperatorSignature,
    validate_manifest,
)
from numpilot.gaps.models import GapCallsite, GapSpec
from numpilot.gaps import qualification
from numpilot.gaps.qualification import (
    CapabilityOverlay,
    QualificationError,
    QualificationEvidence,
    QualificationRef,
    ZERO_SHA256,
    apply_overlay,
    capability_manifest_sha256,
    load_overlay,
    load_qualification,
    publish_overlay_atomic,
    validate_qualification,
)


PRODUCER = "numpilot-qualification-harness-v1"
BACKEND_VERSION = "asnumpy-0.3.1+private"
BACKEND_AUDIT_ID = "asnumpy-audit-20260812"
CANN_VERSION = "9.0.0"
SOC_VERSION = "Ascend910B"
PYTHON_VERSION = "3.10.20"
SIGNATURE_COVERAGE_GATES = (
    "unit_test",
    "numerical_validation",
    "public_asnumpy_api",
    "npu_execution",
)
DEFAULT_SIGNATURE_VARIANTS = (OperatorSignature(positional_args=1),)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(data) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _artifact_id(gap_id, files, generation) -> str:
    """Compute the provider contract without calling qualification internals."""

    identity = json.dumps(
        {"gap_id": gap_id, "files": files, "generation": generation},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "draft-" + _sha256(identity)[:20]


def _write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(mode)


def _write_json(path: Path, data, *, mode: int = 0o600) -> None:
    _write_bytes(path, _json_bytes(data), mode=mode)


def _gate_claims(
    name: str,
    *,
    asnumpy_api: str,
    file_count: int,
    signature_variants,
):
    claims = {
        "artifact_integrity": {
            "verified_files": file_count,
            "mismatches": 0,
        },
        "build": {"exit_code": 0, "output_artifacts": 1},
        "unit_test": {"passed": 4, "failed": 0},
        "numerical_validation": {
            "oracle": "numpy",
            "cases": 12,
            "failed": 0,
            "cpu_fallback_detected": False,
        },
        "public_asnumpy_api": {
            "api": asnumpy_api,
            "callable": True,
            "cpu_fallback_detected": False,
        },
        "npu_execution": {
            "backend": "npu",
            "device_count": 1,
            "kernel_launches": 3,
            "cpu_fallback_detected": False,
        },
    }[name]
    if name in SIGNATURE_COVERAGE_GATES:
        claims.update({
            "covered_signature_variants": len(signature_variants),
            "signature_variants_sha256": (
                qualification.qualification_signatures_sha256(
                    signature_variants
                )
            ),
        })
    return claims


@dataclass
class EvidenceBundle:
    root: Path
    prefix: str
    evidence_data: dict
    evidence_path: Path
    source_path: Path
    spec_path: Path
    manifest_path: Path
    artifact_path: Path
    runtime_module_path: Path
    gate_paths: dict
    current_manifest: CapabilityManifest

    def qualification(self) -> QualificationEvidence:
        return QualificationEvidence.from_dict(deepcopy(self.evidence_data))

    def write_evidence(self) -> QualificationEvidence:
        self.evidence_data["qualification_id"] = (
            qualification.compute_qualification_id(self.evidence_data)
        )
        _write_json(self.evidence_path, self.evidence_data)
        return self.qualification()

    def update_gate_report(self, name: str, report: dict) -> QualificationEvidence:
        _write_json(self.gate_paths[name], report)
        self.evidence_data["gates"][name]["report_sha256"] = _sha256(
            self.gate_paths[name].read_bytes()
        )
        return self.write_evidence()

    def rebind_manifest(self, manifest: dict) -> QualificationEvidence:
        """Update every legitimate outer hash while preserving changed claims."""

        _write_json(self.manifest_path, manifest)
        self.evidence_data["artifact_id"] = manifest["artifact_id"]
        manifest_sha256 = _sha256(self.manifest_path.read_bytes())
        self.evidence_data["artifact_manifest_sha256"] = manifest_sha256
        for name, path in self.gate_paths.items():
            report = json.loads(path.read_text(encoding="utf-8"))
            report["artifact_id"] = manifest["artifact_id"]
            report["artifact_manifest_sha256"] = manifest_sha256
            _write_json(path, report)
            self.evidence_data["gates"][name]["report_sha256"] = _sha256(
                path.read_bytes()
            )
        return self.write_evidence()

    def rebind_spec(self, spec: dict) -> QualificationEvidence:
        """Update the spec and every report that is explicitly bound to it."""

        _write_json(self.spec_path, spec)
        spec_sha256 = _sha256(self.spec_path.read_bytes())
        self.evidence_data["gap_spec_sha256"] = spec_sha256
        for name, path in self.gate_paths.items():
            report = json.loads(path.read_text(encoding="utf-8"))
            report["gap_spec_sha256"] = spec_sha256
            _write_json(path, report)
            self.evidence_data["gates"][name]["report_sha256"] = _sha256(
                path.read_bytes()
            )
        return self.write_evidence()

    def rebind_signature_variants(
        self, signatures
    ) -> QualificationEvidence:
        """Bind a new exact signature set through the spec and every runtime gate."""

        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        spec["signature_variants"] = [
            signature.to_dict() for signature in signatures
        ]
        self.rebind_spec(spec)
        signature_digest = qualification.qualification_signatures_sha256(
            signatures
        )
        for name in SIGNATURE_COVERAGE_GATES:
            path = self.gate_paths[name]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["claims"].update({
                "covered_signature_variants": len(signatures),
                "signature_variants_sha256": signature_digest,
            })
            self.update_gate_report(name, report)
        return self.qualification()

    def reference(self) -> QualificationRef:
        return QualificationRef(
            registry_key=self.evidence_data["registry_key"],
            evidence_path=self.evidence_path.relative_to(self.root).as_posix(),
            evidence_sha256=_sha256(self.evidence_path.read_bytes()),
        )


def _build_evidence(root: Path, *, api: str = "cbrt") -> EvidenceBundle:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    prefix = api
    qualified_api = "numpy." + api
    asnumpy_api = "asnumpy." + api
    gap_id = api + "-gap-20260812"

    source_path = root / prefix / "source" / (api + "_case.py")
    source_payload = (
        "import numpy as np\n"
        "\n"
        "def forward(x):\n"
        f"    return np.{api}(x)\n"
    ).encode("utf-8")
    _write_bytes(source_path, source_payload)

    spec = GapSpec(
        gap_id=gap_id,
        api_name=api,
        qualified_api=qualified_api,
        asnumpy_api=asnumpy_api,
        source_path=source_path.relative_to(root).as_posix(),
        source_sha256=_sha256(source_payload),
        callsites=(
            GapCallsite(
                lineno=4,
                col_offset=11,
                qualified_api=qualified_api,
                call_expr=qualified_api + "(<arg0>)",
                reason="missing_from_audited_asnumpy_coverage",
            ),
        ),
        signature_variants=tuple(
            signature.to_dict()
            for signature in DEFAULT_SIGNATURE_VARIANTS
        ),
        target_soc=SOC_VERSION,
        cann_version=CANN_VERSION,
        shape_rules=("rank>=1", "contiguous"),
        dtype_rules=("float32",),
        semantic_source="numpy",
        reference_cases=({"dtype": "float32", "shape": [8]},),
    )
    spec_path = root / prefix / "spec" / "gap-spec.json"
    _write_json(spec_path, spec.to_dict())
    spec_sha256 = _sha256(spec_path.read_bytes())

    marker = root / prefix / "SHOULD_NOT_EXIST"
    draft_dir = root / prefix / "draft"
    artifact_payloads = {
        "build.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"touch {marker}\n"
        ).encode("utf-8"),
        "op_kernel/cbrt.cpp": b"// offline fixture: generated kernel source\n",
    }
    artifact_files = {}
    artifact_path = draft_dir / "op_kernel" / "cbrt.cpp"
    for relative, payload in artifact_payloads.items():
        target = draft_dir.joinpath(*relative.split("/"))
        _write_bytes(target, payload, mode=0o700 if relative == "build.sh" else 0o600)
        artifact_files[relative] = _sha256(payload)

    generation = {
        "mode": "injected_test_double",
        "provider": "pytest-offline-fixture",
        "model": "deterministic-fixture-v1",
        "attempt_count": 1,
        "response_sha256": [_sha256(b"offline-model-response")],
        "local_validation": "passed",
        "fallback": "disabled",
    }
    artifact_id = _artifact_id(gap_id, artifact_files, generation)
    manifest = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "gap_id": gap_id,
        "workspace": str(draft_dir),
        "summary": "Offline unverified test draft.",
        "status": "generated_unverified",
        "files": artifact_files,
        "generation": generation,
        "gates": {
            "spec": "prepared",
            "generation": "passed",
            "build": "pending",
            "unit_test": "pending",
            "numerical_validation": "pending",
            "activation": "pending",
        },
    }
    manifest_path = draft_dir / "operator-artifact.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path.read_bytes())

    runtime_dir = root / "runtime"
    runtime_module_path = runtime_dir / "libasnumpy_private.so"
    runtime_module_payload = b"offline-fixture-asnumpy-extension\x00"
    _write_bytes(runtime_module_path, runtime_module_payload)
    opp_manifest_path = runtime_dir / "opp-manifest.json"
    _write_json(
        opp_manifest_path,
        {
            "schema_version": 1,
            "registry": "offline-private-opp-fixture",
            "soc": SOC_VERSION,
            "cann": CANN_VERSION,
        },
    )
    runtime_data = {
        "backend": "npu",
        "backend_version": BACKEND_VERSION,
        "backend_audit_id": BACKEND_AUDIT_ID,
        "backend_module_sha256": _sha256(runtime_module_payload),
        "cann_version": CANN_VERSION,
        "soc_version": SOC_VERSION,
        "python_version": PYTHON_VERSION,
        "opp_manifest_path": opp_manifest_path.relative_to(root).as_posix(),
        "opp_manifest_sha256": _sha256(opp_manifest_path.read_bytes()),
    }
    runtime_artifacts = {
        runtime_module_path.relative_to(root).as_posix(): _sha256(
            runtime_module_payload
        )
    }

    gate_paths = {}
    gates = {}
    for name in sorted(qualification.REQUIRED_QUALIFICATION_GATES):
        report_path = root / prefix / "reports" / (name + ".json")
        report = {
            "schema_version": qualification.GATE_REPORT_SCHEMA_VERSION,
            "gate": name,
            "status": "passed",
            "producer": PRODUCER,
            "artifact_id": artifact_id,
            "gap_spec_sha256": spec_sha256,
            "artifact_manifest_sha256": manifest_sha256,
            "constraints_sha256": qualification.qualification_constraints_sha256(
                ("dtype=float32", "rank>=1", "contiguous=true")
            ),
            "target": runtime_data,
            "claims": _gate_claims(
                name,
                asnumpy_api=asnumpy_api,
                file_count=len(artifact_files),
                signature_variants=DEFAULT_SIGNATURE_VARIANTS,
            ),
        }
        _write_json(report_path, report)
        gate_paths[name] = report_path
        gates[name] = {
            "status": "passed",
            "report_path": report_path.relative_to(root).as_posix(),
            "report_sha256": _sha256(report_path.read_bytes()),
            "producer": PRODUCER,
            "format_version": qualification.GATE_REPORT_SCHEMA_VERSION,
        }

    evidence_data = {
        "schema_version": 1,
        "qualification_id": "qual-" + "0" * 20,
        "gap_id": gap_id,
        "registry_key": api,
        "qualified_api": qualified_api,
        "asnumpy_api": asnumpy_api,
        "gap_spec_path": spec_path.relative_to(root).as_posix(),
        "gap_spec_sha256": spec_sha256,
        "source_path": source_path.relative_to(root).as_posix(),
        "source_sha256": _sha256(source_payload),
        "artifact_manifest_path": manifest_path.relative_to(root).as_posix(),
        "artifact_manifest_sha256": manifest_sha256,
        "artifact_id": artifact_id,
        "artifact_files": artifact_files,
        "runtime_artifacts": runtime_artifacts,
        "runtime": runtime_data,
        "constraints": ["dtype=float32", "rank>=1", "contiguous=true"],
        "gates": gates,
    }
    evidence_path = root / prefix / "qualification.json"
    current_manifest = CapabilityManifest(
        backend="npu",
        backend_version=BACKEND_VERSION,
        audit_id=BACKEND_AUDIT_ID,
        environment={
            "module_path": str(runtime_module_path),
            "cann_home": "/opt/ascend/cann",
            "cann_version": CANN_VERSION,
            "soc_version": SOC_VERSION,
            "python_version": PYTHON_VERSION,
        },
        operators={
            api: OperatorCapability(
                name=api,
                available=True,
                provenance=ExecutionProvenance.UNKNOWN,
                evidence="runtime callable; provenance not yet admitted",
            )
        },
    )
    bundle = EvidenceBundle(
        root=root,
        prefix=prefix,
        evidence_data=evidence_data,
        evidence_path=evidence_path,
        source_path=source_path,
        spec_path=spec_path,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        runtime_module_path=runtime_module_path,
        gate_paths=gate_paths,
        current_manifest=current_manifest,
    )
    bundle.write_evidence()
    return bundle


@pytest.fixture
def evidence(tmp_path) -> EvidenceBundle:
    return _build_evidence(tmp_path)


def _validate(bundle: EvidenceBundle, *, manifest=None):
    return validate_qualification(
        bundle.qualification(),
        evidence_root=str(bundle.root),
        current_manifest=manifest or bundle.current_manifest,
    )


def _manifest_with_environment(bundle: EvidenceBundle, **changes):
    environment = dict(bundle.current_manifest.environment)
    environment.update(changes)
    return replace(bundle.current_manifest, environment=environment)


def test_complete_offline_evidence_promotes_only_to_custom_npu(evidence):
    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == _artifact_id(
        manifest["gap_id"], manifest["files"], manifest["generation"]
    )
    assert manifest["workspace"] == str(evidence.manifest_path.parent)

    capability = _validate(evidence)

    assert capability.name == "cbrt"
    assert capability.available
    assert capability.provenance is ExecutionProvenance.CUSTOM_NPU
    assert not capability.strict_npu
    assert capability.evidence == (
        "qualification:" + evidence.evidence_data["qualification_id"]
    )
    assert capability.constraints == (
        "dtype=float32",
        "rank>=1",
        "contiguous=true",
    )
    assert capability.signature_variants == (
        OperatorSignature(positional_args=1),
    )


def test_canonical_signature_digest_is_order_independent():
    signatures = (
        OperatorSignature(positional_args=1),
        OperatorSignature(positional_args=1, keyword_args=("where",)),
    )

    assert qualification.qualification_signatures_sha256(
        signatures
    ) == qualification.qualification_signatures_sha256(
        tuple(reversed(signatures))
    )


def test_every_signature_variant_is_bound_through_every_runtime_gate(evidence):
    signatures = (
        OperatorSignature(positional_args=1),
        OperatorSignature(positional_args=1, keyword_args=("where",)),
    )
    evidence.rebind_signature_variants(signatures)

    capability = _validate(evidence)

    assert capability.signature_variants == signatures
    expected_digest = qualification.qualification_signatures_sha256(
        signatures
    )
    for name in SIGNATURE_COVERAGE_GATES:
        report = json.loads(
            evidence.gate_paths[name].read_text(encoding="utf-8")
        )
        assert report["claims"]["covered_signature_variants"] == 2
        assert report["claims"]["signature_variants_sha256"] == expected_digest


def test_expanding_signature_scope_without_new_gate_evidence_fails_closed(
    evidence,
):
    spec = json.loads(evidence.spec_path.read_text(encoding="utf-8"))
    spec["signature_variants"].append(
        OperatorSignature(
            positional_args=1,
            keyword_args=("where",),
        ).to_dict()
    )
    evidence.rebind_spec(spec)

    with pytest.raises(
        QualificationError,
        match="signature coverage count mismatch",
    ):
        _validate(evidence)


@pytest.mark.parametrize("gate_name", SIGNATURE_COVERAGE_GATES)
def test_each_runtime_gate_rejects_incomplete_signature_coverage_count(
    evidence,
    gate_name,
):
    path = evidence.gate_paths[gate_name]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["claims"]["covered_signature_variants"] = 0
    evidence.update_gate_report(gate_name, report)

    with pytest.raises(
        QualificationError,
        match="gate %s signature coverage count mismatch" % gate_name,
    ):
        _validate(evidence)


@pytest.mark.parametrize("gate_name", SIGNATURE_COVERAGE_GATES)
def test_each_runtime_gate_rejects_signature_coverage_digest_drift(
    evidence,
    gate_name,
):
    path = evidence.gate_paths[gate_name]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["claims"]["signature_variants_sha256"] = "f" * 64
    evidence.update_gate_report(gate_name, report)

    with pytest.raises(
        QualificationError,
        match="gate %s signature coverage digest mismatch" % gate_name,
    ):
        _validate(evidence)


def test_gate_report_v1_metadata_is_rejected_after_signature_binding_upgrade(
    evidence,
):
    data = deepcopy(evidence.evidence_data)
    data["gates"]["unit_test"]["format_version"] = 1

    with pytest.raises(QualificationError, match="format version is unsupported"):
        QualificationEvidence.from_dict(data)


def test_name_only_validation_cannot_broaden_scoped_custom_qualification(evidence):
    capability = _validate(evidence)
    admitted = replace(
        evidence.current_manifest,
        operators={"cbrt": capability},
    )

    validation = validate_manifest(admitted, ["cbrt"], strict_npu=True)

    assert not validation.ok
    assert validation.non_strict == ("cbrt",)


def test_qualification_requires_every_gate(evidence):
    data = deepcopy(evidence.evidence_data)
    data["gates"].pop("npu_execution")

    with pytest.raises(QualificationError, match="gate set is incomplete"):
        QualificationEvidence.from_dict(data)


def test_qualification_rejects_non_passed_gate_metadata(evidence):
    data = deepcopy(evidence.evidence_data)
    data["gates"]["build"]["status"] = "failed"

    with pytest.raises(QualificationError, match="gate build did not pass"):
        QualificationEvidence.from_dict(data)


def test_qualification_rejects_non_passed_gate_report(evidence):
    path = evidence.gate_paths["build"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    qualified = evidence.update_gate_report("build", report)

    with pytest.raises(QualificationError, match="identity/status mismatch"):
        validate_qualification(
            qualified,
            evidence_root=str(evidence.root),
            current_manifest=evidence.current_manifest,
        )


@pytest.mark.parametrize(
    "gate_name",
    sorted(qualification.REQUIRED_QUALIFICATION_GATES),
)
def test_each_gate_claim_schema_rejects_unknown_fields(evidence, gate_name):
    path = evidence.gate_paths[gate_name]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["claims"]["unexpected_claim"] = "must fail closed"
    qualified = evidence.update_gate_report(gate_name, report)

    with pytest.raises(QualificationError):
        validate_qualification(
            qualified,
            evidence_root=str(evidence.root),
            current_manifest=evidence.current_manifest,
        )


def test_build_gate_cannot_hide_a_cpu_fallback_claim(evidence):
    path = evidence.gate_paths["build"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["claims"]["cpu_fallback_detected"] = True
    qualified = evidence.update_gate_report("build", report)

    with pytest.raises(QualificationError):
        validate_qualification(
            qualified,
            evidence_root=str(evidence.root),
            current_manifest=evidence.current_manifest,
        )


@pytest.mark.parametrize(
    "gate_name",
    ["numerical_validation", "public_asnumpy_api", "npu_execution"],
)
def test_cpu_fallback_claim_fails_closed(evidence, gate_name):
    path = evidence.gate_paths[gate_name]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["claims"]["cpu_fallback_detected"] = True
    qualified = evidence.update_gate_report(gate_name, report)

    with pytest.raises(QualificationError, match="CPU fallback|cpu fallback"):
        validate_qualification(
            qualified,
            evidence_root=str(evidence.root),
            current_manifest=evidence.current_manifest,
        )


def test_runtime_cpu_fallback_capability_cannot_be_promoted(evidence):
    manifest = replace(
        evidence.current_manifest,
        operators={
            "cbrt": OperatorCapability(
                name="cbrt",
                available=True,
                provenance=ExecutionProvenance.CPU_FALLBACK,
                evidence="audited CPU fallback",
            )
        },
    )

    with pytest.raises(QualificationError, match="replace audited provenance"):
        _validate(evidence, manifest=manifest)


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("spec", "gap spec hash mismatch"),
        ("manifest", "artifact manifest hash mismatch"),
        ("source", "source.*hash"),
        ("artifact", "artifact file hash mismatch"),
        ("runtime", "runtime.*(hash mismatch|hash drift)"),
    ],
)
def test_bound_file_tampering_fails_closed(evidence, target, match):
    path = {
        "spec": evidence.spec_path,
        "manifest": evidence.manifest_path,
        "source": evidence.source_path,
        "artifact": evidence.artifact_path,
        "runtime": evidence.runtime_module_path,
    }[target]
    path.write_bytes(path.read_bytes() + b"\nTAMPERED\n")

    with pytest.raises(QualificationError, match=match):
        _validate(evidence)


def test_manifest_artifact_id_must_be_derived_from_bound_content(evidence):
    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    derived = _artifact_id(
        manifest["gap_id"], manifest["files"], manifest["generation"]
    )
    assert manifest["artifact_id"] == derived
    manifest["artifact_id"] = "draft-" + "f" * 20
    assert manifest["artifact_id"] != derived
    evidence.rebind_manifest(manifest)

    with pytest.raises(QualificationError, match="artifact_id"):
        _validate(evidence)


def test_manifest_workspace_must_equal_its_real_draft_directory(evidence):
    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workspace"] == str(evidence.manifest_path.parent)
    manifest["workspace"] = str(evidence.root / "some-other-draft")
    evidence.rebind_manifest(manifest)

    with pytest.raises(QualificationError, match="workspace"):
        _validate(evidence)


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "provider",
        "attempt_count",
        "response_sha256",
        "local_validation",
    ],
)
def test_artifact_generation_evidence_requires_every_audit_field(evidence, field):
    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    manifest["generation"].pop(field)
    manifest["artifact_id"] = _artifact_id(
        manifest["gap_id"], manifest["files"], manifest["generation"]
    )
    evidence.rebind_manifest(manifest)

    with pytest.raises(QualificationError):
        _validate(evidence)


def test_gate_reports_bind_the_qualification_constraints(evidence):
    evidence.evidence_data["constraints"] = [
        "dtype=float32|float64",
        "rank>=1",
        "contiguous=true",
    ]
    evidence.write_evidence()

    with pytest.raises(QualificationError, match="constraints mismatch"):
        _validate(evidence)


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("backend_version", "backend version drift"),
        ("cann_version", "CANN version drift"),
        ("soc_version", "SoC version drift"),
        ("python_version", "Python version drift"),
        ("module_hash", "module hash drift"),
    ],
)
def test_current_runtime_identity_drift_fails_closed(evidence, kind, match):
    if kind == "backend_version":
        current = replace(evidence.current_manifest, backend_version="0.0-drifted")
    elif kind == "module_hash":
        evidence.runtime_module_path.write_bytes(b"different-runtime-module")
        current = evidence.current_manifest
    else:
        values = {
            "cann_version": "9.1.0",
            "soc_version": "Ascend310P",
            "python_version": "3.11.0",
        }
        current = _manifest_with_environment(evidence, **{kind: values[kind]})

    with pytest.raises(QualificationError, match=match):
        _validate(evidence, manifest=current)


@pytest.mark.parametrize(
    "field",
    ["backend_version", "cann_version", "soc_version", "python_version"],
)
def test_evidence_rejects_unknown_runtime_identity(evidence, field):
    data = deepcopy(evidence.evidence_data)
    data["runtime"][field] = "unknown"

    with pytest.raises(QualificationError, match="concrete and bounded"):
        QualificationEvidence.from_dict(data)


@pytest.mark.parametrize(
    "field",
    ["module_path", "cann_version", "soc_version", "python_version"],
)
def test_current_runtime_unknown_identity_fails_closed(evidence, field):
    current = _manifest_with_environment(evidence, **{field: "unknown"})

    with pytest.raises(QualificationError, match="concrete and bounded"):
        _validate(evidence, manifest=current)


@pytest.mark.parametrize("unsafe", ["/etc/passwd", "../outside.json"])
@pytest.mark.parametrize("field", ["gap_spec_path", "source_path"])
def test_evidence_rejects_absolute_and_parent_paths(evidence, unsafe, field):
    data = deepcopy(evidence.evidence_data)
    data[field] = unsafe

    with pytest.raises(QualificationError, match="unsafe"):
        QualificationEvidence.from_dict(data)


def test_qualification_source_hash_must_match_gap_spec(evidence):
    replacement = b"different but internally bound source\n"
    _write_bytes(evidence.source_path, replacement)
    evidence.evidence_data["source_sha256"] = _sha256(replacement)
    evidence.write_evidence()

    with pytest.raises(QualificationError, match="GapSpec source hashes differ"):
        _validate(evidence)


def test_qualification_source_path_must_match_gap_spec(evidence):
    alternate = evidence.source_path.with_name("same-content-copy.py")
    _write_bytes(alternate, evidence.source_path.read_bytes())
    evidence.evidence_data["source_path"] = alternate.relative_to(
        evidence.root
    ).as_posix()
    evidence.write_evidence()

    with pytest.raises(QualificationError, match="source paths differ"):
        _validate(evidence)


def test_gap_spec_schema_rejects_unknown_top_level_fields(evidence):
    spec = json.loads(evidence.spec_path.read_text(encoding="utf-8"))
    spec["untrusted_extension"] = {"accepted": True}
    evidence.rebind_spec(spec)

    with pytest.raises(QualificationError, match="gap spec fields are invalid"):
        _validate(evidence)


def test_gap_spec_callsite_integer_fields_reject_bool(evidence):
    spec = json.loads(evidence.spec_path.read_text(encoding="utf-8"))
    spec["callsites"][0]["lineno"] = True
    evidence.rebind_spec(spec)

    with pytest.raises(QualificationError, match="callsite.*integer"):
        _validate(evidence)


def test_gap_spec_callsite_text_fields_do_not_coerce_numbers(evidence):
    spec = json.loads(evidence.spec_path.read_text(encoding="utf-8"))
    spec["callsites"][0]["reason"] = 123
    evidence.rebind_spec(spec)

    with pytest.raises(QualificationError, match="callsite.*reason.*text"):
        _validate(evidence)


def test_qualification_constraints_reject_normalized_duplicates(evidence):
    data = deepcopy(evidence.evidence_data)
    data["constraints"] = ["dtype=float32", " dtype=float32 "]

    with pytest.raises(QualificationError, match="constraints contain duplicates"):
        QualificationEvidence.from_dict(data)


def test_qualification_registry_key_does_not_coerce_bool(evidence):
    data = deepcopy(evidence.evidence_data)
    data["registry_key"] = True
    data["qualified_api"] = "numpy.True"
    data["asnumpy_api"] = "asnumpy.True"

    with pytest.raises(QualificationError, match="registry_key must be text"):
        QualificationEvidence.from_dict(data)


def test_gap_spec_signature_integer_fields_reject_bool(evidence):
    spec = json.loads(evidence.spec_path.read_text(encoding="utf-8"))
    spec["signature_variants"][0]["positional_args"] = True
    evidence.rebind_spec(spec)

    with pytest.raises(QualificationError, match="positional_args.*integer"):
        _validate(evidence)


def test_bound_evidence_rejects_symlink(evidence):
    path = evidence.gate_paths["build"]
    real_path = path.with_name("build-real.json")
    path.rename(real_path)
    path.symlink_to(real_path.name)

    with pytest.raises(QualificationError, match="symlink"):
        _validate(evidence)


def test_qualification_json_rejects_duplicate_keys(evidence):
    duplicate = evidence.root / "duplicate.json"
    _write_bytes(duplicate, b'{"schema_version":1,"schema_version":1}')

    with pytest.raises(QualificationError, match="duplicate key"):
        load_qualification(
            duplicate.relative_to(evidence.root).as_posix(),
            evidence_root=str(evidence.root),
        )


def test_qualification_json_rejects_nonstandard_nan(evidence):
    invalid = evidence.root / "nonstandard-number.json"
    _write_bytes(invalid, b'{"value":NaN}')

    with pytest.raises(QualificationError, match="non-standard constant"):
        load_qualification(
            invalid.relative_to(evidence.root).as_posix(),
            evidence_root=str(evidence.root),
        )


def test_world_writable_bound_file_is_rejected(evidence):
    evidence.gate_paths["unit_test"].chmod(0o666)

    with pytest.raises(QualificationError, match="group/world writable"):
        _validate(evidence)


def test_world_writable_evidence_root_is_rejected(evidence):
    evidence.root.chmod(0o777)

    with pytest.raises(QualificationError, match="group/world writable"):
        _validate(evidence)


def test_world_writable_intermediate_evidence_directory_is_rejected(evidence):
    evidence.gate_paths["build"].parent.chmod(0o777)

    with pytest.raises(QualificationError, match="writable|permission|trusted"):
        _validate(evidence)


def test_world_writable_runtime_backend_module_is_rejected(evidence):
    opp_manifest = evidence.root / evidence.evidence_data["runtime"][
        "opp_manifest_path"
    ]
    evidence.evidence_data["runtime_artifacts"] = {
        opp_manifest.relative_to(evidence.root).as_posix(): _sha256(
            opp_manifest.read_bytes()
        )
    }
    evidence.write_evidence()
    evidence.runtime_module_path.chmod(0o666)

    with pytest.raises(QualificationError, match="writable|permission|module"):
        _validate(evidence)


def test_world_writable_artifact_subdirectory_is_rejected(evidence):
    (evidence.manifest_path.parent / "op_kernel").chmod(0o777)

    with pytest.raises(QualificationError, match="artifact directory.*writable"):
        _validate(evidence)


def test_runtime_artifacts_must_include_backend_module_content(evidence):
    opp_manifest = evidence.root / evidence.evidence_data["runtime"][
        "opp_manifest_path"
    ]
    evidence.evidence_data["runtime_artifacts"] = {
        opp_manifest.relative_to(evidence.root).as_posix(): _sha256(
            opp_manifest.read_bytes()
        )
    }
    evidence.write_evidence()

    with pytest.raises(QualificationError, match="bind the backend module"):
        _validate(evidence)


def test_runtime_artifact_reads_have_a_total_bound(evidence, monkeypatch):
    opp_manifest = evidence.root / evidence.evidence_data["runtime"][
        "opp_manifest_path"
    ]
    module_relative = evidence.runtime_module_path.relative_to(
        evidence.root
    ).as_posix()
    evidence.evidence_data["runtime_artifacts"] = {
        module_relative: _sha256(evidence.runtime_module_path.read_bytes()),
    }
    evidence.write_evidence()
    monkeypatch.setattr(
        qualification,
        "_MAX_RUNTIME_TOTAL_BYTES",
        len(opp_manifest.read_bytes())
        + len(evidence.runtime_module_path.read_bytes())
        - 1,
    )

    with pytest.raises(QualificationError, match="size limit"):
        _validate(evidence)


def test_admission_never_executes_generated_code(evidence, monkeypatch):
    marker = evidence.root / evidence.prefix / "SHOULD_NOT_EXIST"
    assert not marker.exists()
    qualified = evidence.qualification()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("qualification admission attempted code execution")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_call", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)

    capability = validate_qualification(
        qualified,
        evidence_root=str(evidence.root),
        current_manifest=evidence.current_manifest,
    )

    assert capability.provenance is ExecutionProvenance.CUSTOM_NPU
    assert not marker.exists()


def _combined_manifest(*bundles: EvidenceBundle) -> CapabilityManifest:
    first = bundles[0].current_manifest
    operators = {}
    for bundle in bundles:
        operators.update(bundle.current_manifest.operators)
    return replace(first, operators=operators)


def test_apply_overlay_is_all_or_nothing_when_last_entry_fails(tmp_path):
    first = _build_evidence(tmp_path, api="cbrt")
    second = _build_evidence(tmp_path, api="cube_root_alt")
    base = _combined_manifest(first, second)
    before = base.to_dict()
    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256=capability_manifest_sha256(base),
        entries=(first.reference(), second.reference()),
    )
    assert _validate(first, manifest=base).provenance is ExecutionProvenance.CUSTOM_NPU
    second.gate_paths["npu_execution"].write_bytes(b"tampered last entry")

    with pytest.raises(QualificationError):
        apply_overlay(base, overlay, evidence_root=str(tmp_path))

    assert base.to_dict() == before
    assert base.operators["cbrt"].provenance is ExecutionProvenance.UNKNOWN
    assert base.operators["cube_root_alt"].provenance is ExecutionProvenance.UNKNOWN


def _overlay(bundle: EvidenceBundle, *, revision: int, parent: str):
    return CapabilityOverlay(
        revision=revision,
        parent_sha256=parent,
        base_manifest_sha256=capability_manifest_sha256(bundle.current_manifest),
        entries=(bundle.reference(),),
    )


def test_directly_constructed_invalid_overlay_is_never_published(evidence):
    path = evidence.root / "registry" / "invalid-overlay.json"
    error = None

    try:
        invalid = CapabilityOverlay(
            revision=1,
            parent_sha256=ZERO_SHA256,
            base_manifest_sha256="not-a-sha256",
            entries=(),
        )
        publish_overlay_atomic(str(path), invalid)
    except QualificationError as caught:
        error = caught

    assert error is not None
    assert not path.exists()


@pytest.mark.parametrize("field", ["schema_version", "revision"])
def test_overlay_schema_and_revision_reject_bool(evidence, field):
    data = _overlay(evidence, revision=1, parent=ZERO_SHA256).to_dict()
    data[field] = True

    with pytest.raises(QualificationError, match="version|revision|integer"):
        CapabilityOverlay.from_dict(data)


def test_qualification_schema_rejects_bool(evidence):
    data = deepcopy(evidence.evidence_data)
    data["schema_version"] = True

    with pytest.raises(QualificationError, match="schema"):
        QualificationEvidence.from_dict(data)


def test_direct_qualification_dataclass_cannot_bypass_schema_validation(evidence):
    direct = replace(evidence.qualification(), schema_version=True)

    with pytest.raises(QualificationError, match="schema.*integer"):
        validate_qualification(
            direct,
            evidence_root=str(evidence.root),
            current_manifest=evidence.current_manifest,
        )


def test_gate_metadata_format_version_rejects_bool(evidence):
    data = deepcopy(evidence.evidence_data)
    data["gates"]["build"]["format_version"] = True

    with pytest.raises(QualificationError, match="format version"):
        QualificationEvidence.from_dict(data)


def test_gate_report_schema_version_rejects_bool(evidence):
    path = evidence.gate_paths["build"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema_version"] = True
    qualified = evidence.update_gate_report("build", report)

    with pytest.raises(QualificationError, match="schema"):
        validate_qualification(
            qualified,
            evidence_root=str(evidence.root),
            current_manifest=evidence.current_manifest,
        )


def test_publish_overlay_enforces_cas_parent_hash_and_revision(evidence):
    path = evidence.root / "registry" / "capability-overlay.json"
    first = _overlay(evidence, revision=1, parent=ZERO_SHA256)
    first_sha256 = publish_overlay_atomic(str(path), first)
    assert first_sha256 == _sha256(path.read_bytes())
    assert load_overlay(str(path)).to_dict() == first.to_dict()
    original = path.read_bytes()

    valid_second = _overlay(evidence, revision=2, parent=first_sha256)
    with pytest.raises(QualificationError, match="compare-and-swap"):
        publish_overlay_atomic(
            str(path), valid_second, expected_previous_sha256=ZERO_SHA256
        )
    assert path.read_bytes() == original

    wrong_parent = _overlay(evidence, revision=2, parent=ZERO_SHA256)
    with pytest.raises(QualificationError, match="parent SHA-256"):
        publish_overlay_atomic(
            str(path), wrong_parent, expected_previous_sha256=first_sha256
        )
    assert path.read_bytes() == original

    wrong_revision = _overlay(evidence, revision=3, parent=first_sha256)
    with pytest.raises(QualificationError, match="revision is not monotonic"):
        publish_overlay_atomic(
            str(path), wrong_revision, expected_previous_sha256=first_sha256
        )
    assert path.read_bytes() == original

    second_sha256 = publish_overlay_atomic(
        str(path), valid_second, expected_previous_sha256=first_sha256
    )
    assert second_sha256 == _sha256(path.read_bytes())
    assert load_overlay(str(path)).revision == 2


def test_publish_rejects_world_writable_overlay_parent(evidence):
    parent = evidence.root / "public-registry"
    parent.mkdir(mode=0o700)
    parent.chmod(0o777)
    path = parent / "capability-overlay.json"

    with pytest.raises(QualificationError, match="writable|permission|trusted"):
        publish_overlay_atomic(
            str(path),
            _overlay(evidence, revision=1, parent=ZERO_SHA256),
        )

    assert not path.exists()


def test_publish_rejects_world_writable_existing_overlay(evidence):
    path = evidence.root / "registry" / "capability-overlay.json"
    first_sha256 = publish_overlay_atomic(
        str(path), _overlay(evidence, revision=1, parent=ZERO_SHA256)
    )
    original = path.read_bytes()
    path.chmod(0o666)

    with pytest.raises(QualificationError, match="existing overlay.*writable"):
        publish_overlay_atomic(
            str(path),
            _overlay(evidence, revision=2, parent=first_sha256),
            expected_previous_sha256=first_sha256,
        )

    assert path.read_bytes() == original


def test_publish_rejects_special_lock_file(evidence):
    path = evidence.root / "registry" / "capability-overlay.json"
    path.parent.mkdir(mode=0o700)
    lock_path = path.parent / ("." + path.name + ".lock")
    os.mkfifo(lock_path, mode=0o600)

    with pytest.raises(QualificationError, match="overlay lock.*regular file"):
        publish_overlay_atomic(
            str(path), _overlay(evidence, revision=1, parent=ZERO_SHA256)
        )

    assert not path.exists()


def test_publish_write_failure_preserves_previous_overlay(evidence, monkeypatch):
    path = evidence.root / "registry" / "capability-overlay.json"
    first = _overlay(evidence, revision=1, parent=ZERO_SHA256)
    first_sha256 = publish_overlay_atomic(str(path), first)
    original = path.read_bytes()
    second = _overlay(evidence, revision=2, parent=first_sha256)

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(qualification.os, "replace", fail_replace)

    with pytest.raises(QualificationError, match="atomically publish"):
        publish_overlay_atomic(
            str(path), second, expected_previous_sha256=first_sha256
        )

    assert path.read_bytes() == original
    assert load_overlay(str(path)).revision == 1
    assert not tuple(path.parent.glob(".capability-overlay.json-*.tmp"))


def test_post_replace_fsync_failure_reports_visible_but_uncertain_state(
    evidence, monkeypatch
):
    path = evidence.root / "registry" / "capability-overlay.json"
    first = _overlay(evidence, revision=1, parent=ZERO_SHA256)
    first_sha256 = publish_overlay_atomic(str(path), first)
    second = _overlay(evidence, revision=2, parent=first_sha256)
    expected_payload = second.to_bytes()
    real_fsync = qualification.os.fsync
    calls = 0

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(qualification.os, "fsync", fail_directory_fsync)

    with pytest.raises(qualification.OverlayDurabilityError) as caught:
        publish_overlay_atomic(
            str(path), second, expected_previous_sha256=first_sha256
        )

    assert caught.value.published_sha256 == _sha256(expected_payload)
    assert path.read_bytes() == expected_payload
    assert load_overlay(str(path)).revision == 2


def test_directory_close_failure_does_not_hide_durability_failure(
    evidence, monkeypatch
):
    path = evidence.root / "registry" / "capability-overlay.json"
    first_sha256 = publish_overlay_atomic(
        str(path), _overlay(evidence, revision=1, parent=ZERO_SHA256)
    )
    second = _overlay(evidence, revision=2, parent=first_sha256)
    expected_payload = second.to_bytes()
    real_open = qualification.os.open
    real_close = qualification.os.close
    real_fsync = qualification.os.fsync
    directory_fd = None
    fsync_calls = 0
    close_injected = False

    def capture_directory_open(target, flags, mode=0o777):
        nonlocal directory_fd
        fd = real_open(target, flags, mode)
        if os.fspath(target) == os.fspath(path.parent):
            directory_fd = fd
        return fd

    def fail_directory_fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync(fd)

    def close_directory_then_fail(fd):
        nonlocal close_injected
        if fd == directory_fd and not close_injected:
            close_injected = True
            real_close(fd)
            raise OSError("injected directory close failure")
        return real_close(fd)

    monkeypatch.setattr(qualification.os, "open", capture_directory_open)
    monkeypatch.setattr(qualification.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(qualification.os, "close", close_directory_then_fail)

    with pytest.raises(qualification.OverlayDurabilityError) as caught:
        publish_overlay_atomic(
            str(path), second, expected_previous_sha256=first_sha256
        )

    assert type(caught.value) is qualification.OverlayDurabilityError
    assert close_injected
    assert caught.value.published_sha256 == _sha256(expected_payload)
    assert path.read_bytes() == expected_payload


def test_post_replace_directory_close_failure_reports_published_state(
    evidence, monkeypatch
):
    path = evidence.root / "registry" / "capability-overlay.json"
    first_sha256 = publish_overlay_atomic(
        str(path), _overlay(evidence, revision=1, parent=ZERO_SHA256)
    )
    second = _overlay(evidence, revision=2, parent=first_sha256)
    expected_payload = second.to_bytes()
    real_open = qualification.os.open
    real_close = qualification.os.close
    directory_fd = None
    injected = False

    def capture_directory_open(target, flags, mode=0o777):
        nonlocal directory_fd
        fd = real_open(target, flags, mode)
        if os.fspath(target) == os.fspath(path.parent):
            directory_fd = fd
        return fd

    def close_directory_then_fail(fd):
        nonlocal injected
        if fd == directory_fd and not injected:
            injected = True
            real_close(fd)
            raise OSError("injected post-replace directory close failure")
        return real_close(fd)

    monkeypatch.setattr(qualification.os, "open", capture_directory_open)
    monkeypatch.setattr(qualification.os, "close", close_directory_then_fail)

    caught = None
    try:
        publish_overlay_atomic(
            str(path), second, expected_previous_sha256=first_sha256
        )
    except Exception as error:  # exact type is asserted below
        caught = error

    assert injected
    assert directory_fd is not None
    with pytest.raises(OSError):
        os.fstat(directory_fd)
    assert path.read_bytes() == expected_payload
    assert isinstance(caught, qualification.OverlayDurabilityError)
    assert caught.published_sha256 == _sha256(expected_payload)


@pytest.mark.parametrize("phase", ["unlock", "lock_close"])
def test_post_replace_lock_cleanup_failure_reports_published_state(
    evidence, monkeypatch, phase
):
    path = evidence.root / "registry" / "capability-overlay.json"
    first_sha256 = publish_overlay_atomic(
        str(path), _overlay(evidence, revision=1, parent=ZERO_SHA256)
    )
    second = _overlay(evidence, revision=2, parent=first_sha256)
    expected_payload = second.to_bytes()
    lock_path = path.parent / ("." + path.name + ".lock")
    real_open = qualification.os.open
    real_close = qualification.os.close
    real_flock = qualification.fcntl.flock
    lock_fd = None
    injected = False

    def capture_lock_open(target, flags, mode=0o777):
        nonlocal lock_fd
        fd = real_open(target, flags, mode)
        if os.fspath(target) == os.fspath(lock_path):
            lock_fd = fd
        return fd

    def unlock_then_fail(fd, operation):
        nonlocal injected
        result = real_flock(fd, operation)
        if operation == qualification.fcntl.LOCK_UN and not injected:
            injected = True
            raise OSError("injected post-replace unlock failure")
        return result

    def close_lock_then_fail(fd):
        nonlocal injected
        if fd == lock_fd and not injected:
            injected = True
            real_close(fd)
            raise OSError("injected post-replace lock close failure")
        return real_close(fd)

    monkeypatch.setattr(qualification.os, "open", capture_lock_open)
    if phase == "unlock":
        monkeypatch.setattr(qualification.fcntl, "flock", unlock_then_fail)
    else:
        monkeypatch.setattr(qualification.os, "close", close_lock_then_fail)

    caught = None
    try:
        publish_overlay_atomic(
            str(path), second, expected_previous_sha256=first_sha256
        )
    except Exception as error:  # exact type is asserted below
        caught = error

    assert injected
    assert lock_fd is not None
    with pytest.raises(OSError):
        os.fstat(lock_fd)
    assert path.read_bytes() == expected_payload
    assert isinstance(caught, qualification.OverlayDurabilityError)
    assert caught.published_sha256 == _sha256(expected_payload)
