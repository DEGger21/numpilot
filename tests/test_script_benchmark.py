from pathlib import Path
from types import SimpleNamespace

import pytest

import numpilot.script_benchmark as benchmark
from numpilot.script_benchmark import (
    ScriptBenchmarkError,
    _WorkerResult,
    _bootstrap_speedup_ci,
    _percentile,
    _segment_binding,
    _strict_report_gates,
    _warm_runtime_component_sum_ns,
    _worker_gates,
)


MARKER = "# NUMPILOT_DEVICE_INPUTS_END"


def _inputs(source_suffix="result = np.exp(x)\n", migrated_suffix=None):
    if migrated_suffix is None:
        migrated_suffix = source_suffix.replace("np.", "ap.")
    source = (
        "import numpy as np\n"
        "x = np.linspace(0.0, 1.0, 16, dtype=np.float32)\n"
        + MARKER
        + "\n"
        + source_suffix
    )
    migrated = (
        "import asnumpy as ap\n"
        "x = ap.linspace(0.0, 1.0, 16, dtype=ap.float32)\n"
        + MARKER
        + "\n"
        + migrated_suffix
    )
    return SimpleNamespace(
        source_payload=source.encode(),
        migrated_payload=migrated.encode(),
        source_path=Path("/source.py"),
        migrated_path=Path("/migrated.py"),
    )


def test_segment_binding_accepts_namespace_only_migration():
    binding = _segment_binding(_inputs(), MARKER)

    assert binding["source_migrated_prefix_equivalent"] is True
    assert binding["source_migrated_suffix_equivalent"] is True
    assert binding["suffix_host_escape_free"] is True
    assert len(binding["prefix_ast_sha256"]) == 64
    assert len(binding["suffix_ast_sha256"]) == 64


def test_segment_binding_rejects_semantic_change():
    with pytest.raises(ScriptBenchmarkError) as caught:
        _segment_binding(
            _inputs(migrated_suffix="result = ap.log(x)\n"), MARKER
        )

    assert caught.value.code == "segment_mismatch"


@pytest.mark.parametrize(
    "suffix",
    [
        "result = np.asarray(x)\n",
        "result = helper(x)\n",
        "result = x.to_numpy()\n",
        "result = np.exp(x, out=x)\n",
        "if True:\n    result = np.exp(x)\n",
        "for _ in range(2):\n    result = np.exp(x)\n",
        "np.exp(x)\n",
    ],
)
def test_segment_binding_rejects_host_escape_or_control_flow(suffix):
    with pytest.raises(ScriptBenchmarkError) as caught:
        _segment_binding(_inputs(source_suffix=suffix), MARKER)

    assert caught.value.code == "unsafe_timed_region"


def _strict_report():
    return {
        "schema_version": 2,
        "mode": "script",
        "status": "migrated_strict_npu",
        "strict_npu": True,
        "performance_valid": False,
        "ast_audit": "passed",
        "actual_remote_call": True,
        "qualified_analysis": {
            "decision": "DIRECT",
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": "a" * 64,
            "calls": [
                {
                    "decision": "DIRECT",
                    "performance_eligible": True,
                }
            ],
        },
        "llm_receipt": {
            "http_status": 200,
            "tls_verification": "enabled",
            "response_model_identity": "exact",
        },
    }


def test_strict_report_gates_require_real_model_and_performance_eligibility():
    assert all(_strict_report_gates(_strict_report()).values())

    report = _strict_report()
    report["actual_remote_call"] = False
    report["qualified_analysis"]["calls"][0]["performance_eligible"] = False
    gates = _strict_report_gates(report)

    assert gates["model_remote_call"] is False
    assert gates["all_calls_performance_eligible"] is False


def _manifest(role):
    value = {
        "backend": role,
        "pid": 10 if role == "cpu" else 11,
        "affinity": {
            "before": [0, 1, 2, 3],
            "after_binding": [3],
            "after_execution": [3],
        },
        "thread_environment": {
            name: "1" for name in benchmark._THREAD_ENVIRONMENT
        },
        "threadpools": [{"user_api": "blas", "num_threads": 1}],
        "timing": {"warmup_ns": [1, 2], "samples_ns": [10] * 7},
    }
    if role == "npu":
        value.update({
            "migration_report": {"gates": {"strict": True}},
            "current_qualification": {
                "decision": "DIRECT",
                "all_calls_performance_eligible": True,
            },
            "runtime_identity_before": {
                "status": "matched",
                "identity_sha256": "a" * 64,
            },
            "runtime_identity_after": {
                "status": "matched",
                "identity_sha256": "a" * 64,
            },
            "profile_guard": {
                "strict_npu": True,
                "runtime_profile_ok": True,
                "runtime_target_integrity_ok": True,
            },
            "runtime_call_counts": {
                "observed": {"exp": 9},
                "expected": {"exp": 9},
            },
            "synchronization": {
                "prefix_completion_calls": 1,
                "before_and_after_each_resident_execution": True,
                "observed_calls": 19,
                "expected_calls": 19,
            },
        })
    return value


def _result(role):
    return _WorkerResult(
        manifest=_manifest(role), arrays={}, wall_ns=1, removed_secret_variables=0
    )


def test_worker_gates_require_exact_sync_affinity_threads_and_runtime_calls():
    gates = _worker_gates(
        _result("cpu"), _result("npu"), core=3, warmup=2, samples=7
    )

    assert all(gates.values())

    npu = _result("npu")
    npu.manifest["synchronization"]["observed_calls"] = 18
    npu.manifest["runtime_call_counts"]["observed"] = {"exp": 8}
    gates = _worker_gates(
        _result("cpu"), npu, core=3, warmup=2, samples=7
    )
    assert gates["sync_calls_exact"] is False
    assert gates["runtime_calls_exact"] is False


def test_statistics_are_deterministic_and_ordered():
    assert _percentile([10, 20, 30], 0.5) == 20
    first = _bootstrap_speedup_ci([20] * 7, [10] * 7, "a" * 64)
    second = _bootstrap_speedup_ci([20] * 7, [10] * 7, "a" * 64)

    assert first == second == (2.0, 2.0)


def test_component_sum_is_explicitly_prefix_plus_resident_plus_final_copy():
    manifest = {
        "timing": {
            "prefix_input_preparation_ns": 20,
            "final_output_transfer": {"total_ns": 5},
        }
    }

    assert _warm_runtime_component_sum_ns(manifest, 10.0) == 35.0
