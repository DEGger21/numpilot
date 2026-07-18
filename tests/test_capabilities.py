import json
import platform
import types

import pytest

from numpilot.backends.npu import AUDITED_ASNUMPY_VERSION, NpuBackend
from numpilot.capabilities import (
    BackendCapabilityError,
    CapabilityManifest,
    ExecutionProvenance,
)
from numpilot.contracts import OffloadPlan


def _fake_asnumpy(version=AUDITED_ASNUMPY_VERSION, **apis):
    module = types.ModuleType("asnumpy")
    module.__version__ = version
    for name, api in apis.items():
        setattr(module, name, api)
    return module


def _callable(*args, **kwargs):
    return None


def test_capability_manifest_is_versioned_and_json_round_trips():
    backend = NpuBackend(
        _fake_asnumpy(matmul=_callable, add=_callable, not_callable=42)
    )

    manifest = backend.capability_manifest(["matmul", "add", "not_callable"])
    payload = manifest.to_json()
    restored = CapabilityManifest.from_json(payload)

    assert restored.to_dict() == manifest.to_dict()
    assert json.loads(payload)["schema_version"] == "1.0"
    assert restored.backend_version == AUDITED_ASNUMPY_VERSION
    assert set(restored.environment) == {
        "module_path",
        "cann_home",
        "cann_version",
        "python_version",
        "soc_version",
    }
    assert restored.operators["matmul"].provenance is ExecutionProvenance.CPU_FALLBACK
    assert restored.operators["add"].strict_npu
    assert not restored.operators["not_callable"].available


def test_audited_matmul_is_available_but_rejected_as_strict_npu():
    backend = NpuBackend(_fake_asnumpy(matmul=_callable))

    validation = backend.validate_required_ops(
        ["matmul"], raise_on_error=False
    )

    assert not validation.ok
    assert validation.missing == ()
    assert validation.non_strict == ("matmul",)
    assert validation.capabilities["matmul"].available
    assert not validation.capabilities["matmul"].strict_npu
    with pytest.raises(BackendCapabilityError, match="non_strict_npu=matmul"):
        backend.validate_required_ops(["matmul"])


def test_matmul_fallback_profile_survives_runtime_lookup_failure():
    backend = NpuBackend(_fake_asnumpy())

    capability = backend.capability_manifest(["matmul"]).operators["matmul"]

    assert not capability.available
    assert capability.provenance is ExecutionProvenance.CPU_FALLBACK
    assert "runtime lookup failed" in capability.evidence


def test_availability_only_mode_keeps_legacy_execution_compatible():
    backend = NpuBackend(_fake_asnumpy(matmul=_callable))

    validation = backend.validate_required_ops(
        ["matmul"], strict_npu=False
    )

    assert validation.ok
    assert validation.non_strict == ("matmul",)


def test_runtime_missing_api_fails_even_without_strict_provenance():
    backend = NpuBackend(_fake_asnumpy(add=_callable))

    validation = backend.validate_required_ops(
        ["add", "missing"], strict_npu=False, raise_on_error=False
    )

    assert not validation.ok
    assert validation.missing == ("missing",)
    assert validation.capabilities["add"].strict_npu
    assert not validation.capabilities["missing"].available


def test_plan_validation_checks_only_npu_placements():
    backend = NpuBackend(_fake_asnumpy(add=_callable, matmul=_callable))
    plan = OffloadPlan(
        placement={"add": "NPU", "missing_cpu_op": "CPU"},
        threshold_elements=1,
    )

    validation = backend.validate_plan(plan)

    assert validation.ok
    assert validation.required_ops == ("add",)
    assert "missing_cpu_op" not in validation.capabilities


def test_plan_validation_rejects_matmul_cpu_fallback():
    backend = NpuBackend(_fake_asnumpy(matmul=_callable))
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1)

    validation = backend.validate_plan(plan, raise_on_error=False)

    assert not validation.ok
    assert validation.missing == ()
    assert validation.non_strict == ("matmul",)


def test_unaudited_asnumpy_version_defaults_to_unknown_and_non_strict():
    backend = NpuBackend(_fake_asnumpy(version="0.3.2", add=_callable))

    validation = backend.validate_required_ops(["add"], raise_on_error=False)

    capability = validation.capabilities["add"]
    assert capability.available
    assert capability.provenance is ExecutionProvenance.UNKNOWN
    assert not validation.ok
    assert validation.non_strict == ("add",)


def test_callable_outside_audited_coverage_is_not_promoted_to_npu():
    backend = NpuBackend(_fake_asnumpy(mystery=_callable))

    validation = backend.validate_required_ops(["mystery"], raise_on_error=False)

    assert not validation.ok
    assert validation.capabilities["mystery"].provenance is ExecutionProvenance.UNKNOWN


def test_static_demo_coverage_alone_does_not_prove_angle_provenance():
    backend = NpuBackend(_fake_asnumpy(angle=_callable))

    validation = backend.validate_required_ops(["angle"], raise_on_error=False)

    assert not validation.ok
    assert validation.capabilities["angle"].available
    assert validation.capabilities["angle"].provenance is ExecutionProvenance.UNKNOWN


def test_manifest_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="unsupported capability schema"):
        CapabilityManifest.from_dict(
            {
                "schema_version": "2.0",
                "backend": "npu",
                "backend_version": AUDITED_ASNUMPY_VERSION,
                "operators": {},
            }
        )


def test_fake_module_environment_probe_degrades_to_unknown(monkeypatch):
    for name in (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
        "CANN_HOME",
        "CANN_VERSION",
        "ASCEND_SOC_VERSION",
        "NPU_SOC_VERSION",
        "SOC_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    backend = NpuBackend(_fake_asnumpy(add=_callable))

    environment = backend.capability_manifest(["add"]).environment

    assert environment == {
        "module_path": "unknown",
        "cann_home": "unknown",
        "cann_version": "unknown",
        "soc_version": "unknown",
        "python_version": platform.python_version(),
    }


def test_runtime_environment_can_be_bound_without_initialising_acl(monkeypatch):
    module = _fake_asnumpy(add=_callable)
    module.__file__ = "/opt/asnumpy/asnumpy/__init__.py"
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/ascend/cann")
    monkeypatch.setenv("CANN_VERSION", "9.1.0")
    monkeypatch.setenv("ASCEND_SOC_VERSION", "Ascend910B4")

    environment = NpuBackend(module).capability_manifest(["add"]).environment

    assert environment == {
        "module_path": "/opt/asnumpy/asnumpy/__init__.py",
        "cann_home": "/opt/ascend/cann",
        "cann_version": "9.1.0",
        "soc_version": "Ascend910B4",
        "python_version": platform.python_version(),
    }
