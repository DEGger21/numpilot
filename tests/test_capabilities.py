import json
import hashlib
from pathlib import Path
import platform
import types

import numpy as np
import pytest

from numpilot.backends import npu as npu_module
from numpilot.backends.npu import (
    AUDITED_ASNUMPY_VERSION,
    CBRT_ASNUMPY_AUDIT_ID,
    CBRT_ASNUMPY_VERSION,
    NpuBackend,
    OverlayRuntimeConfig,
    P0_ASNUMPY_AUDIT_ID,
    P0_ASNUMPY_VERSION,
    P0_LINALG_ASNUMPY_AUDIT_ID,
    P0_MOVE_ASNUMPY_AUDIT_ID,
    P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
    P0_MOVE_ASNUMPY_VERSION,
)
from numpilot.capabilities import (
    AnalysisCallsiteProfile,
    AnalysisProfileSet,
    BackendCapabilityError,
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCallProfile,
    OperatorSignature,
    OperatorCapability,
    RuntimeProfileError,
    RuntimeProfileGuard,
    TensorProfile,
    load_analysis_profile_set,
    match_capability_profile,
    profile_operator_call,
    strict_receipt_issues,
    tensor_profile_from_value,
)
from numpilot.contracts import OffloadPlan


_REAL_AUDITED_RUNTIME_IDENTITY_MATCHES = (
    npu_module._audited_runtime_identity_matches
)


@pytest.fixture(autouse=True)
def _trust_generic_fake_runtime_identity(monkeypatch):
    """Unit fakes have no wheel tree; identity matching has dedicated tests."""

    monkeypatch.setattr(
        npu_module, "_audited_runtime_identity_matches", lambda _module: True
    )


def _fake_asnumpy(version=AUDITED_ASNUMPY_VERSION, **apis):
    module = types.ModuleType("asnumpy")
    module.__version__ = version
    for name, api in apis.items():
        setattr(module, name, api)
    return module


def _callable(*args, **kwargs):
    return None


def test_p0_audit_ids_bind_wrapper_dependencies():
    assert "init8761f7f9" in npu_module.ASNUMPY_AUDIT_ID
    assert "coreab71b004" in npu_module.ASNUMPY_AUDIT_ID
    assert npu_module.ASNUMPY_AUDIT_ID.endswith("-v2")
    assert "utilsfd94c025" in P0_ASNUMPY_AUDIT_ID
    assert P0_ASNUMPY_AUDIT_ID.endswith("-v2")
    assert "array16c901f5" in P0_LINALG_ASNUMPY_AUDIT_ID
    assert "utilsfd94c025" in P0_LINALG_ASNUMPY_AUDIT_ID
    assert P0_LINALG_ASNUMPY_AUDIT_ID.endswith("-v3")
    assert P0_LINALG_ASNUMPY_AUDIT_ID != (
        "asnumpy-0.1.dev106-g21368abf9-poisson-linalg-"
        "init4f4e8083-math62e0ba09-direct4479c588-cored3ccb39f-v1"
    )
    assert "array16c901f5" in P0_MOVE_ASNUMPY_AUDIT_ID
    assert P0_MOVE_ASNUMPY_AUDIT_ID.endswith("-v2")
    assert P0_MOVE_ASNUMPY_AUDIT_ID != (
        "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
        "init4f4e8083-math62e0ba09-direct4479c588-"
        "utils1b298145-core4465ce6e-v1"
    )
    assert "utils1b298145" in CBRT_ASNUMPY_AUDIT_ID
    assert CBRT_ASNUMPY_AUDIT_ID.endswith("-v2")


def test_p0_core_identity_rejects_utils_drift(tmp_path, monkeypatch):
    from numpilot.backends import npu

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "math.py": b"math",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_P0_RUNTIME_FILE_SHA256",
    )

    assert npu._p0_runtime_identity_matches(module)
    (root / "utils.py").write_bytes(b"drift")
    assert not npu._p0_runtime_identity_matches(module)


def _write_runtime_fixture(tmp_path, relative_payloads, monkeypatch, digest_name):
    from numpilot.backends import npu

    root = tmp_path / "asnumpy"
    for relative, payload in relative_payloads.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    digests = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in relative_payloads.items()
    }
    core = digests.pop("_core.so")
    digests["_core"] = core
    monkeypatch.setattr(npu, digest_name, digests)
    module = types.ModuleType("asnumpy")
    module.__file__ = str(root / "__init__.py")
    return module, root


def test_generic_audited_identity_rejects_wrapper_drift(tmp_path, monkeypatch):
    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "_types.py": b"types",
            "_version.py": b"version",
            "array.py": b"array",
            "linalg/__init__.py": b"linalg-init",
            "linalg/direct.py": b"direct",
            "logic.py": b"logic",
            "math.py": b"math",
            "nn.py": b"nn",
            "sorting.py": b"sorting",
            "statistics.py": b"statistics",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_AUDITED_RUNTIME_FILE_SHA256",
    )

    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    (root / "logic.py").write_bytes(b"drift")
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


def test_generic_version_without_exact_runtime_hashes_is_unaudited(monkeypatch):
    monkeypatch.setattr(
        npu_module, "_audited_runtime_identity_matches", lambda _module: False
    )
    manifest = NpuBackend(
        _fake_asnumpy(add=_callable)
    ).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert not manifest.operators["add"].strict_npu


def test_p0_linalg_identity_rejects_utils_drift(tmp_path, monkeypatch):
    from numpilot.backends import npu

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "array.py": b"array",
            "math.py": b"math",
            "linalg/direct.py": b"direct",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_P0_LINALG_RUNTIME_FILE_SHA256",
    )

    assert npu._p0_linalg_runtime_identity_matches(module)
    (root / "utils.py").write_bytes(b"drift")
    assert not npu._p0_linalg_runtime_identity_matches(module)


def test_cbrt_identity_rejects_utils_drift(tmp_path, monkeypatch):
    from numpilot.backends import npu

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "math.py": b"math",
            "custom_ops.py": b"custom",
            "_custom_ops_registry.py": b"registry",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_CBRT_ASNUMPY_RUNTIME_FILE_SHA256",
    )

    assert npu._cbrt_runtime_identity_matches(module)
    (root / "utils.py").write_bytes(b"drift")
    assert not npu._cbrt_runtime_identity_matches(module)


def test_capability_manifest_is_versioned_and_json_round_trips():
    backend = NpuBackend(
        _fake_asnumpy(matmul=_callable, add=_callable, not_callable=42)
    )

    manifest = backend.capability_manifest(["matmul", "add", "not_callable"])
    payload = manifest.to_json()
    restored = CapabilityManifest.from_json(payload)

    assert restored.to_dict() == manifest.to_dict()
    assert json.loads(payload)["schema_version"] == "1.1"
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


def test_p0_wheel_audit_is_hash_bound_and_limited_to_poisson_core(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_p0_runtime_identity_matches", lambda module: True)
    backend = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        add=_callable,
        multiply=_callable,
        clip=_callable,
        subtract=_callable,
        apply_gradient_descent=_callable,
    ))

    manifest = backend.capability_manifest(
        ["add", "multiply", "clip", "subtract", "apply_gradient_descent"]
    )

    assert manifest.audit_id == P0_ASNUMPY_AUDIT_ID
    assert all(
        manifest.operators[name].strict_npu
        for name in ("add", "multiply", "clip")
    )
    assert manifest.operators["subtract"].provenance is ExecutionProvenance.UNKNOWN
    assert (
        manifest.operators["apply_gradient_descent"].provenance
        is ExecutionProvenance.UNKNOWN
    )


def test_p0_version_without_exact_runtime_hashes_remains_unaudited(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_p0_runtime_identity_matches", lambda module: False)
    manifest = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        add=_callable,
    )).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


def test_cbrt_wheel_audit_admits_only_inherited_base_ops(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_cbrt_runtime_identity_matches", lambda module: True)
    manifest = NpuBackend(_fake_asnumpy(
        version=CBRT_ASNUMPY_VERSION,
        add=_callable,
        multiply=_callable,
        cbrt=_callable,
    )).capability_manifest(["add", "multiply", "cbrt"])

    assert manifest.audit_id == CBRT_ASNUMPY_AUDIT_ID
    assert manifest.operators["add"].strict_npu
    assert manifest.operators["multiply"].strict_npu
    assert manifest.operators["cbrt"].provenance is ExecutionProvenance.UNKNOWN
    assert not manifest.operators["cbrt"].strict_npu


def test_cbrt_wheel_without_exact_runtime_hashes_remains_unaudited(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_cbrt_runtime_identity_matches", lambda module: False)
    manifest = NpuBackend(_fake_asnumpy(
        version=CBRT_ASNUMPY_VERSION,
        add=_callable,
    )).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


def test_p0_linalg_wheel_is_hash_bound_and_profile_scoped(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_linalg_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        add=_callable,
        outer=_callable,
        matmul=_callable,
        subtract=_callable,
    ))

    manifest = backend.capability_manifest(["add", "outer", "matmul", "subtract"])

    assert manifest.audit_id == P0_LINALG_ASNUMPY_AUDIT_ID
    assert manifest.operators["add"].strict_npu
    for name in ("outer", "matmul"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert not capability.strict_npu
        assert capability.signature_variants == (OperatorSignature(positional_args=2),)
        assert "rank<=2" in capability.constraints
        assert "numel>=1" in capability.constraints
    subtract = manifest.operators["subtract"]
    assert subtract.provenance is ExecutionProvenance.NATIVE_NPU
    assert not subtract.strict_npu
    assert subtract.signature_variants == (OperatorSignature(positional_args=2),)
    assert "numel>=1" in subtract.constraints


def test_p0_linalg_wheel_admits_hash_bound_inherited_demo_apis(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_linalg_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        absolute=_callable,
        amax=_callable,
        cos=_callable,
        linspace=_callable,
        sin=_callable,
        sum=_callable,
        angle=_callable,
    ))

    manifest = backend.capability_manifest(
        ["absolute", "amax", "cos", "linspace", "sin", "sum", "angle"]
    )

    for name in ("absolute", "amax", "cos", "linspace", "sin", "sum"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert capability.strict_npu
    angle = manifest.operators["angle"]
    assert angle.provenance is ExecutionProvenance.UNKNOWN
    assert not angle.strict_npu


def test_p0_move_wheel_is_hash_bound_and_keeps_linalg_scope(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        add=_callable,
        outer=_callable,
        matmul=_callable,
        subtract=_callable,
    ))

    manifest = backend.capability_manifest(["add", "outer", "matmul", "subtract"])

    assert manifest.audit_id == P0_MOVE_ASNUMPY_AUDIT_ID
    assert manifest.operators["add"].strict_npu
    for name in ("outer", "matmul", "subtract"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert not capability.strict_npu
        assert capability.signature_variants == (OperatorSignature(positional_args=2),)


def test_p0_move_wheel_admits_hash_bound_inherited_demo_apis(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        absolute=_callable,
        amax=_callable,
        cos=_callable,
        linspace=_callable,
        sin=_callable,
        sum=_callable,
        angle=_callable,
    ))

    manifest = backend.capability_manifest(
        ["absolute", "amax", "cos", "linspace", "sin", "sum", "angle"]
    )

    for name in ("absolute", "amax", "cos", "linspace", "sin", "sum"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert capability.strict_npu
    angle = manifest.operators["angle"]
    assert angle.provenance is ExecutionProvenance.UNKNOWN
    assert not angle.strict_npu


def test_p0_move_version_without_exact_runtime_hashes_remains_unaudited(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: False
    )
    manifest = NpuBackend(_fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        add=_callable,
    )).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


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


@pytest.mark.parametrize(
    "provenance",
    [
        ExecutionProvenance.NATIVE_NPU,
        ExecutionProvenance.CUSTOM_NPU,
        ExecutionProvenance.COMPOSITE_NPU,
    ],
)
def test_name_only_strict_npu_rejects_all_scoped_provenance(provenance):
    capability = OperatorCapability(
        name="scoped_op",
        available=True,
        provenance=provenance,
        evidence="profile-scoped evidence",
        constraints=("dtype=float32",),
    )

    assert not capability.strict_npu


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
                "audit_id": "runtime",
                "backend": "npu",
                "backend_version": AUDITED_ASNUMPY_VERSION,
                "environment": {},
                "operators": {},
            }
        )


def test_manifest_detaches_caller_owned_mappings():
    capability = _scoped_cbrt_capability()
    operators = {"cbrt": capability}
    environment = {"module_path": "/private/asnumpy"}
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators=operators,
        environment=environment,
    )

    operators["cbrt"] = OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.NATIVE_NPU,
    )
    environment["module_path"] = "/poisoned"

    assert manifest.operators["cbrt"] is capability
    assert manifest.environment["module_path"] == "/private/asnumpy"
    with pytest.raises(TypeError):
        manifest.operators["other"] = capability


def test_capability_json_rejects_duplicate_keys_and_inconsistent_derived_value():
    manifest = NpuBackend(_fake_asnumpy(add=_callable)).capability_manifest(["add"])
    data = manifest.to_dict()
    data["operators"]["add"]["strict_npu"] = False

    with pytest.raises(ValueError, match="strict_npu is inconsistent"):
        CapabilityManifest.from_dict(data)
    with pytest.raises(ValueError, match="duplicate key"):
        CapabilityManifest.from_json(
            '{"schema_version":"1.1","schema_version":"1.1"}'
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


def _scoped_cbrt_capability(*, constraints=None, signatures=None):
    return OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="qualification:qual-0123456789abcdefabcd",
        constraints=tuple(constraints or (
            "dtype=float32",
            "rank>=1",
            "contiguous=true",
        )),
        signature_variants=tuple(signatures or (
            OperatorSignature(positional_args=1),
        )),
    )


def test_scoped_capability_round_trip_preserves_qualified_signatures():
    capability = _scoped_cbrt_capability()
    restored = OperatorCapability.from_dict(capability.to_dict())

    assert restored == capability
    assert not restored.strict_npu
    assert restored.signature_variants == (
        OperatorSignature(positional_args=1),
    )


def test_analysis_profile_set_round_trip_binds_source_callsite_and_signature():
    profiles = AnalysisProfileSet(
        source_sha256="a" * 64,
        callsites=(AnalysisCallsiteProfile(
            qualified_name="numpy.cbrt",
            lineno=3,
            col_offset=11,
            profile=profile_operator_call(
                (np.ones((4,), dtype=np.float32),),
                {},
            ),
        ),),
    )

    restored = AnalysisProfileSet.from_json(json.dumps(profiles.to_dict()))

    assert restored == profiles
    assert restored.find("numpy.cbrt", 3, 11, 1, ()) == profiles.callsites[0].profile
    assert restored.find("numpy.cbrt", 4, 11, 1, ()) is None


def test_analysis_profile_set_rejects_duplicate_callsite_identity():
    callsite = AnalysisCallsiteProfile(
        qualified_name="numpy.cbrt",
        lineno=3,
        col_offset=11,
        profile=profile_operator_call(
            (np.ones((4,), dtype=np.float32),),
            {},
        ),
    )

    with pytest.raises(ValueError, match="duplicate callsites"):
        AnalysisProfileSet(
            source_sha256="a" * 64,
            callsites=(callsite, callsite),
        )


def test_analysis_profile_loader_rejects_symlink_and_reads_private_file(tmp_path):
    profiles = AnalysisProfileSet(
        source_sha256="a" * 64,
        callsites=(AnalysisCallsiteProfile(
            qualified_name="numpy.cbrt",
            lineno=3,
            col_offset=11,
            profile=profile_operator_call(
                (np.ones((4,), dtype=np.float32),),
                {},
            ),
        ),),
    )
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profiles.to_dict()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)

    assert load_analysis_profile_set(str(path)) == profiles
    with pytest.raises(ValueError, match="cannot open"):
        load_analysis_profile_set(str(linked))


@pytest.mark.parametrize(
    "profile",
    [
        lambda: TensorProfile("float32", (True,), True),
        lambda: TensorProfile("float32", tuple(1 for _ in range(65)), True),
        lambda: OperatorCallProfile(
            inputs=(TensorProfile("float32", (1,), True),),
            positional_args=2,
            keyword_args=(),
        ),
    ],
)
def test_profile_contracts_reject_malformed_or_inconsistent_values(profile):
    with pytest.raises(ValueError):
        profile()


def test_tensor_profile_uses_shape_dtype_and_real_c_contiguity():
    contiguous = np.ones((2, 3), dtype=np.float32)
    transposed = contiguous.T

    assert tensor_profile_from_value(contiguous).to_dict() == {
        "dtype": "float32",
        "shape": [2, 3],
        "rank": 2,
        "c_contiguous": True,
    }
    assert not tensor_profile_from_value(transposed).c_contiguous


def test_generic_tensor_profile_rejects_forged_duck_metadata():
    class ForgedTensor:
        shape = (4,)
        dtype = np.dtype("float32")

        class Flags:
            c_contiguous = True

        flags = Flags()

    assert tensor_profile_from_value(ForgedTensor()) is None


def test_generic_tensor_profile_rejects_subclasses_and_non_native_byte_order():
    class ForgedArray(np.ndarray):
        @property
        def dtype(self):
            return np.dtype("float32")

    backing = np.ones((4,), dtype=np.float64)
    forged = backing.view(ForgedArray)
    big_endian = np.ones((4,), dtype=">f4")

    assert tensor_profile_from_value(forged) is None
    assert tensor_profile_from_value(big_endian) is None


def test_profile_match_accepts_only_the_exact_qualified_cbrt_scope():
    capability = _scoped_cbrt_capability()
    valid = profile_operator_call(
        (np.ones((4,), dtype=np.float32),),
        {},
    )

    match = match_capability_profile(capability, valid)

    assert match.matched
    assert match.reasons == ()


@pytest.mark.parametrize(
    "args,kwargs,reason",
    [
        ((np.ones((4,), dtype=np.float64),), {}, "dtype=float64"),
        ((np.array(1.0, dtype=np.float32),), {}, "rank=0"),
        ((np.ones((2, 3), dtype=np.float32).T,), {}, "contiguous=false"),
        ((1.0,), {}, "unprofiled operands"),
        (
            (np.ones((4,), dtype=np.float32), np.ones((4,), dtype=np.float32)),
            {},
            "call signature",
        ),
        ((np.ones((4,), dtype=np.float32),), {"out": None}, "call signature"),
    ],
)
def test_profile_match_fails_closed_outside_scope(args, kwargs, reason):
    match = match_capability_profile(
        _scoped_cbrt_capability(),
        profile_operator_call(args, kwargs),
    )

    assert not match.matched
    assert reason in "; ".join(match.reasons)


@pytest.mark.parametrize(
    "constraints,reason",
    [
        (("dtype=float16", "rank>=1", "contiguous=true"), "unsupported dtype"),
        (("dtype=float32", "layout=ND"), "unsupported profile"),
    ],
)
def test_profile_constraint_grammar_rejects_unknown_or_duplicate_rules(
    constraints, reason
):
    match = match_capability_profile(
        _scoped_cbrt_capability(constraints=constraints),
        profile_operator_call((np.ones((2,), dtype=np.float32),), {}),
    )

    assert not match.matched
    assert reason in "; ".join(match.reasons)


@pytest.mark.parametrize(
    "value,reason",
    [
        (np.ones((1, 2, 3), dtype=np.float32), "rank<=2"),
        (np.ones((0, 3), dtype=np.float32), "numel>=1"),
    ],
)
def test_profile_constraint_upper_rank_and_nonempty_bounds(value, reason):
    capability = _scoped_cbrt_capability(constraints=(
        "dtype=float32",
        "rank>=1",
        "rank<=2",
        "contiguous=true",
        "numel>=1",
    ))

    match = match_capability_profile(
        capability,
        profile_operator_call((value,), {}),
    )

    assert not match.matched
    assert reason in "; ".join(match.reasons)


def test_matcher_rejects_signature_only_half_scoped_capability():
    capability = OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="incomplete qualification",
        signature_variants=(OperatorSignature(positional_args=1),),
    )

    match = match_capability_profile(
        capability,
        profile_operator_call((object(),), {}),
    )

    assert not match.matched
    assert "no profile constraints" in "; ".join(match.reasons)


def test_profile_capability_rejects_duplicate_constraints_at_construction():
    with pytest.raises(ValueError, match="constraints are invalid"):
        _scoped_cbrt_capability(
            constraints=("dtype=float32", "dtype=float32")
        )


def test_runtime_guard_rejects_before_target_dispatch():
    calls = []
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )
    guard = RuntimeProfileGuard(manifest, ["cbrt"])

    with pytest.raises(RuntimeProfileError, match="runtime profile rejected"):
        guard.validate_call(
            "cbrt",
            (np.ones((2,), dtype=np.float64),),
            {},
        )
        calls.append("dispatched")

    assert calls == []
    assert not guard.runtime_profile_ok
    assert not guard.checks["cbrt"][0]["matched"]
    assert guard.to_dict()["profile_status"] == "rejected"
    assert guard.to_dict()["ok"] is False
    assert guard.to_dict()["strict_npu"] is False


def test_runtime_guard_records_a_valid_scoped_call():
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )
    guard = RuntimeProfileGuard(manifest, ["cbrt"])

    guard.validate_call("cbrt", (np.ones((2,), dtype=np.float32),), {})

    assert guard.runtime_profile_ok
    assert guard.to_dict()["profile_scoped"] == ["cbrt"]
    assert guard.to_dict()["profile_status"] == "matched"
    assert guard.to_dict()["ok"] is True


def test_runtime_guard_reports_profile_pending_before_real_dispatch():
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )

    data = RuntimeProfileGuard(manifest, ["cbrt"]).to_dict()

    assert data["preflight_ok"] is True
    assert data["profile_status"] == "pending"
    assert data["ok"] is False
    assert data["strict_npu"] is False


def test_runtime_guard_receipt_is_a_detached_snapshot():
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )
    guard = RuntimeProfileGuard(manifest, ["cbrt"])
    receipt = guard.to_dict()

    receipt["runtime_profile_checks"]["cbrt"].append({"matched": True})

    assert guard.profile_status == "pending"
    assert guard.ok is False
    assert guard.checks == {"cbrt": []}


def test_native_npu_guard_does_not_require_a_runtime_profile():
    backend = NpuBackend(_fake_asnumpy(add=_callable))

    guard = backend.prepare_profile_guard(["add"])

    assert guard.profile_scoped == ()
    assert guard.runtime_profile_ok


def test_runtime_guard_rejects_callable_identity_drift_before_dispatch():
    module = _fake_asnumpy(add=_callable)
    backend = NpuBackend(module)
    guard = backend.prepare_profile_guard(["add"])
    module.add = lambda *args: "host replacement"

    with pytest.raises(RuntimeError, match="callable identity drifted"):
        guard.validate_call("add", (object(), object()), {})

    receipt = guard.to_dict()
    assert receipt["profile_status"] == "rejected"
    assert receipt["runtime_target_integrity_ok"] is False
    assert receipt["ok"] is False


def test_profile_guard_binds_target_from_same_capability_lookup():
    first = lambda *args: "first"
    second = lambda *args: "second"

    class RotatingModule(types.ModuleType):
        def __init__(self):
            super().__init__("asnumpy")
            self.__version__ = AUDITED_ASNUMPY_VERSION
            self.lookups = 0

        def __getattribute__(self, name):
            if name == "add":
                lookups = object.__getattribute__(self, "lookups")
                object.__setattr__(self, "lookups", lookups + 1)
                return first if lookups == 0 else second
            return super().__getattribute__(name)

    module = RotatingModule()
    guard = NpuBackend(module).prepare_profile_guard(["add"])

    assert module.lookups == 1
    assert guard.runtime_targets["add"] is first
    with pytest.raises(RuntimeError, match="callable identity drifted"):
        guard.validate_call("add", (object(), object()), {})


def test_strict_receipt_rejects_scoped_capability_without_real_guard():
    capability = _scoped_cbrt_capability()
    receipt = {
        "ok": True,
        "strict_npu": True,
        "required_ops": ["cbrt"],
        "missing": [],
        "non_strict": [],
        "profile_status": "matched",
        "capabilities": {"cbrt": capability.to_dict()},
    }

    issues = strict_receipt_issues(receipt, ["cbrt"])

    assert any("concrete runtime profile guard" in issue for issue in issues)


def test_strict_receipt_rejects_malformed_capability_schema():
    receipt = {
        "ok": True,
        "strict_npu": True,
        "required_ops": ["add"],
        "missing": [],
        "non_strict": [],
        "capabilities": {
            "add": {
                "name": "add",
                "available": True,
                "provenance": "native_npu",
                "strict_npu": True,
                "evidence": "fake",
                "constraints": ["layout=anything"],
                "signature_variants": [],
            }
        },
    }

    issues = strict_receipt_issues(receipt, ["add"])

    assert any("schema rejected" in issue for issue in issues)


def test_overlay_configuration_is_explicit_and_fail_closed(monkeypatch):
    from numpilot.gaps import qualification
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationRef,
        ZERO_SHA256,
    )

    module = _fake_asnumpy(cbrt=_callable)
    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256="1" * 64,
        entries=(QualificationRef("cbrt", "cbrt/qualification.json", "2" * 64),),
    )
    seen = {}

    monkeypatch.setattr(qualification, "load_overlay", lambda path: overlay)

    def fake_apply(base, supplied, **kwargs):
        seen["names"] = tuple(base.operators)
        seen["root"] = kwargs["evidence_root"]
        return CapabilityManifest(
            backend=base.backend,
            backend_version=base.backend_version,
            environment=base.environment,
            audit_id=base.audit_id + "+qualification-test",
            operators={"cbrt": _scoped_cbrt_capability()},
        )

    monkeypatch.setattr(qualification, "apply_overlay", fake_apply)
    backend = NpuBackend(
        module,
        overlay_config=OverlayRuntimeConfig(
            "/private/overlay.json",
            "/private/evidence",
        ),
    )

    manifest = backend.capability_manifest(["cbrt"])

    assert seen == {"names": ("cbrt",), "root": "/private/evidence"}
    assert manifest.operators["cbrt"].provenance is ExecutionProvenance.CUSTOM_NPU


def test_overlay_failure_never_falls_back_to_unqualified_base(monkeypatch):
    from numpilot.gaps import qualification

    monkeypatch.setattr(
        qualification,
        "load_overlay",
        lambda path: (_ for _ in ()).throw(RuntimeError("tampered overlay")),
    )
    backend = NpuBackend(
        _fake_asnumpy(cbrt=_callable),
        overlay_config=OverlayRuntimeConfig("/private/overlay.json", "/private/evidence"),
    )

    with pytest.raises(RuntimeError, match="tampered overlay"):
        backend.capability_manifest(["cbrt"])


def test_stricter_move_identity_can_activate_exact_legacy_overlay_base(
    monkeypatch,
):
    from numpilot.backends import npu
    from numpilot.gaps import qualification
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationRef,
        ZERO_SHA256,
        capability_manifest_sha256,
    )

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: True
    )
    module = _fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        apply_gradient_descent=_callable,
    )
    plain = NpuBackend(module)._raw_capability_manifest(
        ("apply_gradient_descent",)
    )
    assert plain.audit_id == P0_MOVE_ASNUMPY_AUDIT_ID
    legacy = npu._overlay_base_with_audit_id(
        plain,
        ("apply_gradient_descent",),
        P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
    )
    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256=capability_manifest_sha256(legacy),
        entries=(QualificationRef(
            "apply_gradient_descent",
            "apply_gradient_descent/qualification.json",
            "2" * 64,
        ),),
    )
    monkeypatch.setattr(qualification, "load_overlay", lambda _path: overlay)
    seen = {}

    def fake_apply(base, _overlay, **_kwargs):
        seen["audit_id"] = base.audit_id
        return CapabilityManifest(
            backend=base.backend,
            backend_version=base.backend_version,
            environment=base.environment,
            audit_id=base.audit_id + "+qualification-test",
            operators={
                "apply_gradient_descent": OperatorCapability(
                    name="apply_gradient_descent",
                    available=True,
                    provenance=ExecutionProvenance.CUSTOM_NPU,
                    evidence="qualification:qual-" + "1" * 20,
                )
            },
        )

    monkeypatch.setattr(qualification, "apply_overlay", fake_apply)
    manifest = NpuBackend(
        module,
        overlay_config=OverlayRuntimeConfig(
            "/private/overlay.json", "/private/evidence"
        ),
    ).capability_manifest(("apply_gradient_descent",))

    assert seen["audit_id"] == P0_MOVE_LEGACY_OVERLAY_AUDIT_ID
    assert manifest.audit_id.endswith("+qualification-test")


def test_overlay_runtime_uses_one_atomic_raw_snapshot(monkeypatch):
    from numpilot.gaps import qualification
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationRef,
        ZERO_SHA256,
    )

    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256="1" * 64,
        entries=(QualificationRef("cbrt", "cbrt/qualification.json", "2" * 64),),
    )
    monkeypatch.setattr(qualification, "load_overlay", lambda path: overlay)

    module = _fake_asnumpy(cbrt=_callable, add=_callable)
    backend = NpuBackend(
        module,
        overlay_config=OverlayRuntimeConfig("/private/overlay.json", "/private/evidence"),
    )
    calls = []
    original = backend._raw_capability_manifest

    def counted(required):
        calls.append(tuple(sorted(required)))
        return original(required)

    backend._raw_capability_manifest = counted

    def fake_apply(base, supplied, **kwargs):
        assert tuple(base.operators) == ("cbrt",)
        return CapabilityManifest(
            backend=base.backend,
            backend_version=base.backend_version,
            environment=base.environment,
            audit_id=base.audit_id + "+qualification-test",
            operators={"cbrt": _scoped_cbrt_capability()},
        )

    monkeypatch.setattr(qualification, "apply_overlay", fake_apply)

    manifest = backend.capability_manifest(["add"])

    assert calls == [("add", "cbrt")]
    assert tuple(manifest.operators) == ("add",)
    assert manifest.operators["add"].strict_npu
