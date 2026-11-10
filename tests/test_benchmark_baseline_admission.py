"""Baseline API collection/composition boundaries without an NPU."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from numpilot.benchmark_overlay import BenchmarkOverlay, build_analysis_profiles, load_benchmark_overlay
from numpilot import benchmark_suite as suite


def descriptor(operators=("full", "mean", "add")):
    return BenchmarkOverlay(
        qualification_id="official-aclnn-" + "1" * 20, overlay_sha256="a" * 64,
        runtime_identity_sha256="b" * 64, operators=tuple(operators),
        qualified_names=tuple("numpy." + name for name in operators), resolver=object(),
    )


def test_collector_profiles_creation_and_reduction_without_inventing_input_tensors():
    source = '''
import numpy as np
def build_inputs():
    return {"a": np.arange(12, dtype=np.float32).reshape(3, 4)}
def forward(a):
    filled = np.full(a.shape, .25, dtype=a.dtype)
    summed = np.add(a, filled)
    return {"out": np.mean(summed, axis=1)}
'''
    result = build_analysis_profiles(source, descriptor())
    assert result.matched_callsites == result.attempted_callsites == 3
    profiles = {site.qualified_name: site.profile for site in result.profiles.callsites}
    creation = profiles["numpy.full"]
    assert creation.inputs == () and creation.creation_output.shape == (3, 4)
    assert creation.creation_output.dtype == "float32"
    reduction = profiles["numpy.mean"]
    assert len(reduction.inputs) == 1 and reduction.reduction_axis == 1
    assert reduction.reduction_keepdims is False


def test_cpu_input_builder_is_outside_migrated_forward_qualification_scope():
    source = '''
import numpy as np
def build_inputs():
    return {"a": np.full((3, 4), 1., dtype=np.float32)}
def forward(a):
    return {"out": np.mean(a, axis=1)}
'''
    result = build_analysis_profiles(source, descriptor())
    assert result.matched_callsites == result.attempted_callsites == 1
    assert [site.qualified_name for site in result.profiles.callsites] == ["numpy.mean"]


def test_mean_without_explicit_axis_is_not_admitted_as_device_array_reduction():
    source = '''
import numpy as np
def build_inputs():
    return {"a": np.ones((3, 4), dtype=np.float32)}
def forward(a):
    return {"out": np.mean(a)}
'''
    result = build_analysis_profiles(source, descriptor())
    assert result.attempted_callsites == 1 and result.matched_callsites == 0
    assert result.profiles is None


def test_loader_preserves_two_independent_certificate_descriptors(monkeypatch):
    import numpilot.backends.npu as module

    captured = []
    baseline = {"qualification_id": "baseline-api-" + "2" * 20,
                "certificate_sha256": "c" * 64, "runtime_identity_sha256": "b" * 64,
                "operators": ["add"], "qualified_names": ["numpy.add"]}

    class Resolver:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def runtime_identity(self):
            return object()

        def qualified_overlay_admission(self, identity):
            return SimpleNamespace(qualification_id="official-aclnn-" + "1" * 20,
                overlay_sha256="a" * 64, runtime_identity_sha256="b" * 64,
                entries={"concatenate": SimpleNamespace(source_api="numpy.concatenate")})

        def qualified_baseline_api_admission(self, identity):
            return baseline

    monkeypatch.setattr(module, "NpuBackend", Resolver)
    overlay = load_benchmark_overlay("original-official.json", "official-root",
                                    baseline_api_certificate="baseline.json",
                                    baseline_api_evidence_root="baseline-root")
    assert overlay.operators == ("add", "concatenate")
    assert overlay.qualified_names == ("numpy.add", "numpy.concatenate")
    assert overlay.to_dict()["baseline_api"] == baseline
    assert overlay.overlay_path == "original-official.json"
    assert overlay.baseline_certificate_path == "baseline.json"
    assert captured[0]["baseline_api_config"].evidence_root == "baseline-root"
    assert overlay.qualification_id.startswith("official-aclnn-")


def test_loader_rejects_half_configured_baseline():
    with pytest.raises(ValueError, match="supplied together"):
        load_benchmark_overlay("official.json", "official-root", baseline_api_certificate="baseline.json")


@pytest.mark.parametrize("extra", [
    ["--baseline-api-certificate", "baseline.json"],
    ["--baseline-api-certificate", "baseline.json", "--baseline-api-evidence-root", "baseline-root"],
])
def test_cli_rejects_baseline_without_explicit_npu_official_composition(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        suite.main(["--out", str(tmp_path), *extra])
    assert error.value.code == 2


def test_cli_passes_baseline_locations_into_validated_overlay_loader(tmp_path, monkeypatch, capsys):
    from numpilot import benchmark_overlay

    received = []
    backend = SimpleNamespace(name="npu")
    admitted = SimpleNamespace(resolver=backend)

    def load(*args, **kwargs):
        received.append((args, kwargs))
        return admitted

    monkeypatch.setattr(benchmark_overlay, "load_benchmark_overlay", load)
    monkeypatch.setattr(suite, "run_suite", lambda *args, **kwargs: {"status": "pass", "total": 0, "unexpected": 0})
    suite.main(["--out", str(tmp_path), "--backend", "npu", "--capability-overlay", "official.json",
                "--qualification-evidence-root", "official-root", "--baseline-api-certificate", "baseline.json",
                "--baseline-api-evidence-root", "baseline-root", "--json"])
    assert received == [(("official.json", "official-root"), {
        "baseline_api_certificate": "baseline.json", "baseline_api_evidence_root": "baseline-root"})]
