import types

import pytest

from numpilot.backends import npu as npu_module
from numpilot.backends.npu import AUDITED_ASNUMPY_VERSION, NpuBackend
from numpilot.capabilities import (
    ExecutionProvenance,
    OperatorCallProfile,
)
from numpilot.capability_catalog import ASNUMPY_CAPABILITY_CATALOG


def _callable(*args, **kwargs):
    return None


def _namespace(names):
    return types.SimpleNamespace(**{name: _callable for name in names})


def _generic_module():
    module = types.ModuleType("asnumpy")
    module.__version__ = AUDITED_ASNUMPY_VERSION
    module.add = _callable
    module.linalg = _namespace(
        name.split(".", 1)[1]
        for name in ASNUMPY_CAPABILITY_CATALOG.keys_for_namespace("linalg")
    )
    module.random = _namespace(
        name.split(".", 1)[1]
        for name in ASNUMPY_CAPABILITY_CATALOG.keys_for_namespace("random")
    )
    return module


@pytest.fixture(autouse=True)
def _trust_local_generic_fake(monkeypatch):
    monkeypatch.setattr(
        npu_module,
        "_audited_runtime_identity_matches",
        lambda _module: True,
    )


def test_nested_manifest_uses_canonical_keys_and_exact_linalg_routes():
    names = ASNUMPY_CAPABILITY_CATALOG.keys_for_namespace("linalg")

    manifest = NpuBackend(_generic_module()).capability_manifest(names)

    assert set(manifest.operators) == set(names)
    assert "det" not in manifest.operators
    assert manifest.operators["linalg.det"].provenance is (
        ExecutionProvenance.COMPOSITE_NPU
    )
    assert manifest.operators["linalg.inv"].provenance is (
        ExecutionProvenance.NATIVE_NPU
    )
    assert manifest.operators["linalg.qr"].provenance is (
        ExecutionProvenance.NATIVE_NPU
    )
    for name in ("linalg.matrix_power", "linalg.norm"):
        assert manifest.operators[name].provenance is (
            ExecutionProvenance.CPU_FALLBACK
        )
    assert manifest.operators["linalg.slogdet"].provenance is (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK
    )


def test_nested_manifest_has_exact_random_routes_without_leaf_aliases():
    names = ASNUMPY_CAPABILITY_CATALOG.keys_for_namespace("random")

    manifest = NpuBackend(_generic_module()).capability_manifest(names)

    assert set(manifest.operators) == set(names)
    assert "normal" not in manifest.operators
    native = {
        "random.normal",
        "random.standard_normal",
        "random.uniform",
    }
    for name in names:
        expected = (
            ExecutionProvenance.NATIVE_NPU
            if name in native
            else ExecutionProvenance.COMPOSITE_NPU
        )
        assert manifest.operators[name].provenance is expected


def test_nested_lookup_does_not_fall_back_to_same_named_top_level_leaf():
    module = _generic_module()
    module.det = _callable
    del module.linalg.det

    capability = NpuBackend(module).capability_manifest(
        ["linalg.det"]
    ).operators["linalg.det"]

    assert not capability.available
    assert "runtime lookup failed" in capability.evidence


@pytest.mark.parametrize(
    "unsafe",
    [
        "numpy.linalg.det",
        "linalg..det",
        "linalg._private",
        "linalg.__dict__",
        "linalg.det-name",
        "../linalg.det",
    ],
)
def test_runtime_manifest_rejects_noncanonical_or_private_paths(unsafe):
    with pytest.raises(ValueError, match="canonical capability"):
        NpuBackend(_generic_module()).capability_manifest([unsafe])


def test_analysis_profile_resolver_accepts_exact_nested_numpy_apis():
    backend = NpuBackend(_generic_module())
    resolver = backend.prepare_analysis_profile_resolver(
        ("numpy.linalg.det", "numpy.random.normal")
    )
    profile = OperatorCallProfile(
        inputs=(), positional_args=0, keyword_args=()
    )

    det = resolver.resolve_profile_capability("numpy.linalg.det", profile)
    normal = resolver.resolve_profile_capability(
        "numpy.random.normal", profile
    )

    assert det.matched
    assert normal.matched
    assert set(resolver.manifest.operators) == {
        "linalg.det",
        "random.normal",
    }


def test_nested_profile_guard_captures_targets_and_detects_leaf_drift():
    module = _generic_module()
    original_det = module.linalg.det
    original_normal = module.random.normal
    guard = NpuBackend(module).prepare_profile_guard(
        ("linalg.det", "random.normal")
    )

    assert guard.runtime_targets == {
        "linalg.det": original_det,
        "random.normal": original_normal,
    }
    guard.validate_call("random.normal", (), {})

    module.linalg.det = lambda *args, **kwargs: "replacement"
    with pytest.raises(RuntimeError, match="callable identity drifted"):
        guard.validate_call("linalg.det", (), {})


def test_nested_support_preserves_top_level_target_capture():
    module = _generic_module()
    guard = NpuBackend(module).prepare_profile_guard(("add",))

    assert guard.runtime_targets == {"add": module.add}
    guard.validate_call("add", (), {})
