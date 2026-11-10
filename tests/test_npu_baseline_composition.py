"""No-device tests for preserving independent official and baseline trust roots."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from numpilot.backends.npu import BaselineApiRuntimeConfig, NpuBackend, OverlayRuntimeConfig
from numpilot.capabilities import (
    BackendCapabilityError, CapabilityManifest, ExecutionProvenance, OperatorCapability,
    OperatorSignature, RuntimeProfileError, RuntimeProfileGuard, profile_api_call,
)


def manifest(names, audit="global-unaudited"):
    return CapabilityManifest(backend="npu", backend_version="candidate-version",
                              operators={name: OperatorCapability(name=name, available=True,
                                         provenance=ExecutionProvenance.UNKNOWN, evidence="not qualified")
                                         for name in names}, audit_id=audit)


@pytest.fixture
def composed(monkeypatch):
    import sys
    from numpilot.gaps import qualification, official_qualification

    events = []
    state = {"fail": False}
    cert = SimpleNamespace(entries=(SimpleNamespace(registry_key="add", qualified_api="numpy.add"),),
                           qualification_id="baseline-api-test")
    overlay = SimpleNamespace(entries=(SimpleNamespace(registry_key="concatenate"),))
    identity = SimpleNamespace(status="audit_required", identity_sha256="a" * 64)
    backend = NpuBackend(module=SimpleNamespace(),
                         overlay_config=OverlayRuntimeConfig("official.json", "official-evidence"),
                         baseline_api_config=BaselineApiRuntimeConfig("baseline.json", "baseline-evidence"))
    monkeypatch.setattr(backend, "runtime_identity", lambda: identity)

    def raw(names, **kwargs):
        events.append(("raw", set(names)))
        if kwargs.get("runtime_targets") is not None:
            kwargs["runtime_targets"].update({name: object() for name in names})
        return manifest(names)

    def official(base, candidate, **kwargs):
        events.append(("official", base, kwargs))
        assert set(base.operators) == {"concatenate"}
        assert base.audit_id == "global-unaudited"
        assert kwargs["current_identity"] is identity
        return replace(base, audit_id=base.audit_id + "+official", operators={
            "concatenate": replace(base.operators["concatenate"], provenance=ExecutionProvenance.NATIVE_NPU,
                                   evidence="official evidence",
                                   constraints=("dtype=float32", "rank>=1", "rank<=3", "contiguous=true"),
                                   signature_variants=(OperatorSignature(1, ("axis",)),)),
        })

    def baseline(base, certificate, **kwargs):
        events.append(("baseline", base, kwargs))
        assert base.operators["concatenate"].evidence == "official evidence"
        assert base.operators["add"].provenance is ExecutionProvenance.UNKNOWN
        assert kwargs["current_identity"] is identity and kwargs["evidence_root"] == "baseline-evidence"
        if state["fail"]:
            raise ValueError("baseline runtime/source evidence drifted")
        operators = dict(base.operators)
        operators["add"] = replace(operators["add"], provenance=ExecutionProvenance.NATIVE_NPU,
                                   evidence="independent baseline evidence")
        return replace(base, operators=operators, audit_id=base.audit_id + "+baseline")

    monkeypatch.setattr(backend, "_raw_capability_manifest", raw)
    monkeypatch.setattr(qualification, "load_overlay", lambda path: overlay)
    monkeypatch.setattr(qualification, "apply_overlay", official)
    monkeypatch.setattr(official_qualification, "is_official_overlay", lambda value: True)
    module = SimpleNamespace(load_baseline_api_certificate=lambda path: cert,
                             apply_baseline_api_certificate=baseline)
    monkeypatch.setitem(sys.modules, "numpilot.baseline_api_qualification", module)
    return backend, events, identity, cert, state


def test_baseline_composes_after_unchanged_official_base_and_preserves_global_identity(composed):
    backend, events, identity, _, _ = composed
    targets = {}
    result = backend._capability_manifest(("add", "concatenate", "multiply"), runtime_targets=targets)
    assert [event[0] for event in events] == ["raw", "official", "baseline"]
    assert result.operators["add"].provenance is ExecutionProvenance.NATIVE_NPU
    assert result.operators["concatenate"].provenance is ExecutionProvenance.NATIVE_NPU
    assert result.operators["multiply"].provenance is ExecutionProvenance.UNKNOWN
    assert set(targets) == {"add", "concatenate", "multiply"}
    assert identity.status == "audit_required" and identity.identity_sha256 == "a" * 64
    assert result.audit_id == "global-unaudited+official+baseline"


def test_unrequested_baseline_entries_validate_but_do_not_expand_returned_manifest(composed):
    backend, events, _, _, _ = composed
    result = backend.capability_manifest(("concatenate",))
    assert set(result.operators) == {"concatenate"}
    assert events[0][1] == {"add", "concatenate"}


def test_baseline_failure_cannot_fall_back_to_unverified_base_api(composed):
    backend, _, _, _, state = composed
    state["fail"] = True
    with pytest.raises(ValueError, match="drifted"):
        backend.capability_manifest(("add",))


def test_baseline_cannot_replace_official_operator_ownership(composed):
    backend, events, _, cert, _ = composed
    cert.entries = (SimpleNamespace(registry_key="concatenate", qualified_api="numpy.concatenate"),)
    with pytest.raises(ValueError, match="ownership overlaps"):
        backend.capability_manifest(("concatenate",))
    assert events == []


def test_baseline_configuration_is_explicit_and_requires_official_composition():
    with pytest.raises(ValueError, match="certificate_path"):
        BaselineApiRuntimeConfig("", "evidence")
    with pytest.raises(ValueError, match="evidence_root"):
        BaselineApiRuntimeConfig("certificate", "")
    with pytest.raises(TypeError, match="BaselineApiRuntimeConfig"):
        NpuBackend(baseline_api_config={"certificate_path": "certificate"})
    with pytest.raises(ValueError, match="explicit official overlay"):
        NpuBackend(baseline_api_config=BaselineApiRuntimeConfig("certificate", "evidence"))


def test_custom_operator_overlay_cannot_authorize_baseline_api(composed, monkeypatch):
    from numpilot.gaps import official_qualification
    backend, events, _, _, _ = composed
    monkeypatch.setattr(official_qualification, "is_official_overlay", lambda value: False)
    with pytest.raises(ValueError, match="official qualification overlay"):
        backend.capability_manifest(("add",))
    assert events == []


@pytest.fixture
def guarded_composition(composed, monkeypatch):
    """Run the production raw-snapshot and guard path with CPU-only callables."""
    import sys

    backend, events, identity, _, state = composed
    module = backend.get_module()
    module.add = lambda a, b: a + b
    module.concatenate = lambda arrays, axis=0: np.concatenate(arrays, axis=axis)
    module.multiply = lambda a, b: a * b

    def raw(names, **kwargs):
        events.append(("raw", set(names)))
        result = NpuBackend._raw_capability_manifest(backend, names, **kwargs)
        return replace(result, audit_id="global-unaudited")

    monkeypatch.setattr(backend, "_raw_capability_manifest", raw)
    baseline_module = sys.modules["numpilot.baseline_api_qualification"]
    apply = baseline_module.apply_baseline_api_certificate
    inputs = (np.ones((4, 8), dtype=np.float32),) * 2
    profile = profile_api_call("add", inputs, {})

    def scoped_apply(*args, **kwargs):
        result = apply(*args, **kwargs)
        operators = dict(result.operators)
        operators["add"] = replace(
            operators["add"],
            constraints=("dtype=float32", "rank>=1", "rank<=3", "contiguous=true"),
            signature_variants=(OperatorSignature(2, ()),),
            qualified_profiles=(profile,),
        )
        return replace(result, operators=operators)

    monkeypatch.setattr(baseline_module, "apply_baseline_api_certificate", scoped_apply)
    return backend, module, events, identity, state, inputs, profile


@pytest.mark.parametrize("required", [("add",), ("concatenate",), ("add", "concatenate")])
def test_real_guard_captures_only_requested_targets_after_validating_union(guarded_composition, required):
    backend, module, events, identity, _, _, _ = guarded_composition
    guard = backend.prepare_profile_guard(required)
    assert events[0] == ("raw", {"add", "concatenate"})
    assert [event[0] for event in events] == ["raw", "official", "baseline"]
    assert set(guard.runtime_targets) == set(required) == set(guard.manifest.operators)
    assert all(guard.runtime_targets[name] is getattr(module, name) for name in required)
    assert identity.status == "audit_required"


def test_exact_profile_and_identity_guards_survive_subset_target_capture(guarded_composition):
    backend, module, _, _, _, inputs, profile = guarded_composition
    guard = backend.prepare_profile_guard(("add",))
    assert guard.profile_status == "pending" and not guard.ok
    guard.validate_call("add", inputs, {})
    assert guard.ok
    receipt = json.loads(json.dumps(guard.to_dict()))
    assert receipt["capabilities"]["add"]["qualified_profiles"] == [profile.to_dict()]
    assert OperatorCapability.from_dict(receipt["capabilities"]["add"]) == guard.manifest.operators["add"]
    assert CapabilityManifest.from_dict(json.loads(json.dumps(guard.manifest.to_dict()))) == guard.manifest
    with pytest.raises(RuntimeProfileError):
        guard.validate_call("add", (np.ones((4, 9), np.float32),) * 2, {})
    assert not guard.ok
    fresh = backend.prepare_profile_guard(("add",))
    module.add = lambda a, b: a + b
    with pytest.raises(RuntimeError, match="callable identity drifted"):
        fresh.validate_call("add", inputs, {})
    assert not fresh.ok


def test_invalid_target_and_unqualified_api_still_fail_closed(guarded_composition):
    backend, module, _, _, _, _, _ = guarded_composition
    guard = backend.prepare_profile_guard(("add",))
    with pytest.raises(BackendCapabilityError):
        backend.prepare_profile_guard(("add", "multiply"))
    for targets in ({}, {"add": object()}, {"add": module.add, "concatenate": module.concatenate}):
        with pytest.raises(ValueError, match="runtime target snapshot is invalid"):
            RuntimeProfileGuard(guard.manifest, ("add",), runtime_module=module,
                                runtime_targets=targets)


def test_failed_certificate_validation_publishes_no_runtime_targets(guarded_composition):
    backend, _, _, _, state, _, _ = guarded_composition
    state["fail"] = True
    targets = {}
    with pytest.raises(ValueError, match="drifted"):
        backend._capability_manifest(("add",), runtime_targets=targets)
    assert targets == {}
