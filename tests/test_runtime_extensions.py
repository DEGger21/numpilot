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

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpilot.runtime_extensions as runtime_extensions_module
from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    RuntimeIdentity,
    RuntimeProfileGuard,
)
from numpilot.runtime_extensions import (
    CONSTITUENTS,
    CONSTRAINTS,
    EXTENSION_MODULE_NAME,
    INTERFACE,
    PROVENANCE,
    SCOPE,
    RunnerObservation,
    RuntimeExtensionError,
    activate_qualified_add_one_extension,
    load_materialized_add_one_extension,
    materialize_add_one_extension,
    qualify_add_one_extension,
)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _verified_runtime_attestation():
    identity = RuntimeIdentity(
        asnumpy_repository="https://example.invalid/asnumpy.git",
        asnumpy_commit="1" * 40,
        asnumpy_tree="2" * 40,
        asnumpy_version="test-runtime-1",
        wrapper_sha256={"__init__.py": "3" * 64},
        module_path="/opt/asnumpy/asnumpy/__init__.py",
        module_sha256="3" * 64,
        native_module_path="/opt/asnumpy/asnumpy/_core.so",
        native_module_sha256="4" * 64,
        cann_home="/opt/ascend/cann",
        cann_version="9.0.0",
        soc_model="Ascend910B4",
        python_executable="/opt/numpilot/bin/python",
        python_version="3.11.0",
        bootstrap_receipt_path="/opt/numpilot/bootstrap-receipt.json",
        bootstrap_receipt_sha256="5" * 64,
        audit_id="test-runtime-audit",
        status="matched",
        issues=(),
    )
    operators = {
        name: OperatorCapability(
            name=name,
            available=True,
            provenance=ExecutionProvenance.NATIVE_NPU,
            evidence="hash-bound test runtime source audit",
        )
        for name in ("add", "ones_like")
    }
    manifest = CapabilityManifest(
        backend="npu",
        backend_version=identity.asnumpy_version,
        audit_id=identity.audit_id,
        operators=operators,
        environment={
            "module_path": identity.module_path,
            "cann_home": identity.cann_home,
            "cann_version": identity.cann_version,
            "soc_version": identity.soc_model,
            "python_version": identity.python_version,
        },
    )
    receipt = RuntimeProfileGuard(manifest, operators).to_dict()
    return runtime_extensions_module._runtime_attestation_record(
        identity.to_dict(), receipt, ()
    )


def _cannbot_draft(tmp_path, *, legacy=False):
    root = tmp_path / "cannbot"
    root.mkdir(parents=True)
    draft = root / "operator-draft.cpp"
    if legacy:
        payload = (
            "// UNVERIFIED_DRAFT\n"
            "float add_one(float x) {\n"
            "  return x + 1.0f;\n"
            "}\n"
        ).encode()
    else:
        payload = (
            "// UNVERIFIED_DRAFT\n"
            "__aicore__ inline void add_one(\n"
            "    AscendC::LocalTensor<float> dst,\n"
            "    const AscendC::LocalTensor<float> src,\n"
            "    uint32_t count) {\n"
            "  AscendC::Adds(dst, src, 1.0f, count);\n"
            "}\n"
        ).encode()
    draft.write_bytes(payload)
    report = root / "operator-generation-report.json"
    report_value = {
        "schema_version": 1,
        "record_type": "numpilot_demo_operator_generation",
        "status": "generated_unverified",
        "candidate_status": "draft",
        "generation_route": "cannbot_registry_invoke_mini_gap",
        "cannbot_route": True,
        "actual_remote_call": True,
        "generated_file": draft.name,
        "generated_file_sha256": _sha256(payload),
        "cannbot_skill_gate": {
            "status": "pass",
            "content_embedded_for_remote_model": True,
        },
        "claims": {
            "model_generated": True,
            "compiled": False,
            "unit_tested": False,
            "numerically_verified": False,
            "activated": False,
            "strict_npu_performance_verified": False,
        },
    }
    if not legacy:
        report_value.update(
            {
                "draft_kind": "ascendc_kernel_fragment",
                "operator_language": "ascend_c",
                "standalone_compilation_unit": False,
            }
        )
    report.write_text(
        json.dumps(report_value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, draft


def _materialized(tmp_path):
    report, draft = _cannbot_draft(tmp_path)
    return materialize_add_one_extension(
        report, draft, (tmp_path / "extension").absolute()
    )


def test_materialize_binds_unverified_cannbot_inputs_and_composite_source(tmp_path):
    extension = _materialized(tmp_path)
    restored = load_materialized_add_one_extension(extension.manifest_path)
    assert restored == extension

    manifest = json.loads(Path(extension.manifest_path).read_text())
    assert manifest["status"] == "materialized_unqualified"
    assert manifest["interface"] == INTERFACE
    assert manifest["provenance"] == PROVENANCE
    assert manifest["scope"] == SCOPE
    assert manifest["constituents"] == list(CONSTITUENTS)
    assert manifest["interface_available"] is False
    assert manifest["upstream_modified"] is False
    assert manifest["cannbot_draft_activated"] is False
    assert manifest["cannbot"]["draft_status"] == "generated_unverified"
    assert manifest["cannbot"]["draft_kind"] == "ascendc_kernel_fragment"
    assert manifest["cannbot"]["operator_language"] == "ascend_c"

    source = Path(extension.module_path).read_text()
    assert "asnumpy_module.ones_like(x, dtype=x.dtype)" in source
    assert "asnumpy_module.add(x, one, dtype=x.dtype)" in source
    assert "CANNBOT_DRAFT_ACTIVATED = False" in source
    assert Path(extension.module_path).stat().st_mode & 0o077 == 0
    assert Path(extension.manifest_path).stat().st_mode & 0o077 == 0


def test_materialized_module_activates_only_on_supplied_module_object(tmp_path):
    extension = _materialized(tmp_path)
    calls = []

    def ones_like(x, *, dtype):
        calls.append(("ones_like", x, dtype))
        return "one"

    def add(x, one, *, dtype):
        calls.append(("add", x, one, dtype))
        return "result"

    fake = SimpleNamespace(ones_like=ones_like, add=add)
    spec = importlib.util.spec_from_file_location(
        EXTENSION_MODULE_NAME, extension.module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interface = module.activate(fake)
    value = SimpleNamespace(dtype="float32")
    assert interface(value) == "result"
    assert fake.add_one is interface
    assert [item[0] for item in calls] == ["ones_like", "add"]


def test_materialize_rejects_draft_hash_drift_without_creating_output(tmp_path):
    report, draft = _cannbot_draft(tmp_path)
    draft.write_text("// UNVERIFIED_DRAFT\nfloat add_one(float x){return x;}\n")
    out = (tmp_path / "extension").absolute()
    with pytest.raises(RuntimeExtensionError, match="do not match") as caught:
        materialize_add_one_extension(report, draft, out)
    assert caught.value.code == "cannbot_binding_mismatch"
    assert not out.exists()


def test_materialize_rejects_cannbot_report_that_claims_activation(tmp_path):
    report, draft = _cannbot_draft(tmp_path)
    value = json.loads(report.read_text())
    value["claims"]["activated"] = True
    report.write_text(json.dumps(value) + "\n")
    with pytest.raises(RuntimeExtensionError) as caught:
        materialize_add_one_extension(
            report, draft, (tmp_path / "extension").absolute()
        )
    assert caught.value.code == "invalid_cannbot_report"


def test_materialize_rejects_scalar_cpp_labeled_as_ascendc_fragment(tmp_path):
    report, draft = _cannbot_draft(tmp_path)
    payload = b"// UNVERIFIED_DRAFT\nfloat add_one(float x){return x + 1.0f;}\n"
    draft.write_bytes(payload)
    value = json.loads(report.read_text())
    value["generated_file_sha256"] = _sha256(payload)
    report.write_text(json.dumps(value) + "\n")

    with pytest.raises(RuntimeExtensionError) as caught:
        materialize_add_one_extension(
            report, draft, (tmp_path / "extension").absolute()
        )
    assert caught.value.code == "invalid_cannbot_draft"


def test_materialize_accepts_legacy_scalar_receipt_without_relabeling_it(tmp_path):
    report, draft = _cannbot_draft(tmp_path, legacy=True)
    extension = materialize_add_one_extension(
        report, draft, (tmp_path / "extension").absolute()
    )
    manifest = json.loads(Path(extension.manifest_path).read_text())
    assert manifest["cannbot"]["draft_kind"] == "legacy_scalar_cpp_function"
    assert manifest["cannbot"]["operator_language"] == "cpp"
    assert manifest["cannbot_draft_activated"] is False


def test_injected_runner_can_validate_contract_but_never_qualifies(tmp_path):
    extension = _materialized(tmp_path)
    attestation = _verified_runtime_attestation()

    def runner(invocation):
        evidence = {
            "schema_version": 1,
            "record_type": "numpilot_runtime_extension_worker_evidence",
            "status": "passed",
            "contract_sha256": invocation.contract_sha256,
            "manifest_sha256": extension.manifest_sha256,
            "extension_module_sha256": extension.module_sha256,
            "interface": INTERFACE,
            "interface_available": True,
            "provenance": PROVENANCE,
            "scope": SCOPE,
            "constituent_dispatches": {name: 3 for name in CONSTITUENTS},
            "constraints": list(CONSTRAINTS),
            "backend": "npu",
            "npu_execution_performed": True,
            "no_fallback_provenance_verified": True,
            "runtime_identity_stable": True,
            "runtime_attestation_before": attestation,
            "runtime_attestation_after": attestation,
            "result_type": "asnumpy.ndarray",
            "numerical_cases": 3,
            "max_abs_error": 0.0,
            "upstream_modified": False,
            "cannbot_draft_activated": False,
        }
        return RunnerObservation(returncode=0, evidence=evidence)

    result = qualify_add_one_extension(
        extension,
        (tmp_path / "qualification").absolute(),
        runner=runner,
    )
    assert result.status == "test_evidence_only"
    assert result.interface_available is False
    assert result.numerical_cases == 0
    persisted = json.loads(Path(result.report_path).read_text())
    assert persisted["worker_evidence_valid"] is True
    assert persisted["runner"] == "injected_test_runner"
    assert persisted["claims"]["numerically_verified"] is False
    assert persisted["claims"]["npu_execution_verified"] is False
    with pytest.raises(RuntimeExtensionError, match="not qualified"):
        activate_qualified_add_one_extension(result.report_path, SimpleNamespace())


def test_callable_cpu_shim_cannot_satisfy_runtime_attestation():
    fake = SimpleNamespace(
        __version__="cpu-shim-1",
        __file__="/tmp/cpu-shim/asnumpy/__init__.py",
        ones_like=lambda value, **kwargs: value,
        add=lambda left, right, **kwargs: left,
    )

    attestation, guard = runtime_extensions_module._prepare_runtime_attestation(fake)

    assert guard is None
    assert attestation["status"] == "audit_required"
    assert attestation["verified"] is False
    assert "runtime_identity_not_matched" in attestation["issues"]


def test_builtin_runner_truthfully_downgrades_unmatched_runtime(
    tmp_path, monkeypatch
):
    extension = _materialized(tmp_path)

    def audit_required_runner(invocation):
        attestation = runtime_extensions_module._runtime_attestation_record(
            None, None, ("runtime_identity_unavailable",)
        )
        evidence = runtime_extensions_module._runtime_audit_required_worker_evidence(
            contract_payload=Path(invocation.contract_path).read_bytes(),
            materialized=extension,
            attestation=attestation,
        )
        Path(invocation.evidence_path).write_text(
            json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
        )
        return RunnerObservation(returncode=0, evidence=evidence)

    monkeypatch.setattr(
        runtime_extensions_module, "_default_runner", audit_required_runner
    )
    result = qualify_add_one_extension(
        extension, (tmp_path / "qualification-audit-required").absolute()
    )

    assert result.status == "runtime_audit_required"
    assert result.interface_available is False
    report = json.loads(Path(result.report_path).read_text())
    assert report["runtime_audit_evidence_valid"] is True
    assert report["runtime_identity_status"] == "audit_required"
    assert report["npu_execution_performed"] is False
    assert report["no_fallback_provenance_verified"] is False
    assert report["claims"]["npu_execution_verified"] is False


def test_timeout_and_runner_failure_fail_closed(tmp_path):
    extension = _materialized(tmp_path)

    def timed_out(_invocation):
        return RunnerObservation(returncode=-1, timed_out=True, error_type="TimeoutExpired")

    result = qualify_add_one_extension(
        extension,
        (tmp_path / "timeout").absolute(),
        timeout_seconds=0.05,
        runner=timed_out,
    )
    assert result.status == "failed"
    assert result.interface_available is False
    report = json.loads(Path(result.report_path).read_text())
    assert report["timed_out"] is True
    assert report["cannbot_draft_activated"] is False

    def broken(_invocation):
        raise RuntimeError("offline test failure")

    failed = qualify_add_one_extension(
        extension,
        (tmp_path / "runner-failed").absolute(),
        runner=broken,
    )
    assert failed.status == "failed"
    assert failed.interface_available is False


def test_invalid_timeout_fails_before_output_creation(tmp_path):
    extension = _materialized(tmp_path)
    out = (tmp_path / "qualification").absolute()
    with pytest.raises(RuntimeExtensionError) as caught:
        qualify_add_one_extension(extension, out, timeout_seconds=0)
    assert caught.value.code == "invalid_timeout"
    assert not out.exists()
