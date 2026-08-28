import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import pytest

from tools import case06_scale_performance_evidence as evidence


def test_frozen_plan_has_exact_scales_samples_scopes_and_precommitted_hash():
    assert evidence._FROZEN_PLAN["scales"] == [1_000_000, 4_000_000, 9_000_000]
    assert evidence._FROZEN_PLAN["dtype"] == "float32"
    assert evidence._FROZEN_PLAN["chain"] == [
        "power(base, exponent)",
        "log(power)",
        "exp(log)",
        "equal(power, exp)",
    ]
    assert evidence._FROZEN_PLAN["timing"]["warmup_samples"] == 2
    assert evidence._FROZEN_PLAN["timing"]["measurement_samples"] == 7
    assert set(evidence._FROZEN_PLAN["scopes"]) == {
        "device_resident_compute", "core_direct_compute", "allocation", "h2d", "d2h",
    }
    assert evidence._plan_sha256() == evidence._FROZEN_PLAN_SHA256
    assert evidence._FROZEN_PLAN_SHA256 == (
        "35c47fdfb8e391e872ff1d8d5d1bcd71f133771f85d8c4cbdf7f31d86e95a5db"
    )


def test_chain_preserves_case06_operation_order_and_outputs():
    calls = []

    class Api:
        @staticmethod
        def power(left, right):
            calls.append("power")
            return np.power(left, right)

        @staticmethod
        def log(value):
            calls.append("log")
            return np.log(value)

        @staticmethod
        def exp(value):
            calls.append("exp")
            return np.exp(value)

        @staticmethod
        def equal(left, right):
            calls.append("equal")
            return np.equal(left, right)

    base = np.full((4,), 2.0, dtype=np.float32)
    exponent = np.full((4,), 3.0, dtype=np.float32)
    output = evidence._chain(Api, base, exponent)

    assert calls == ["power", "log", "exp", "equal"]
    assert [item.dtype for item in output] == [
        np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("bool"),
    ]
    np.testing.assert_array_equal(output[0], np.full((4,), 8.0, dtype=np.float32))


def test_measure_scope_keeps_ordered_samples_and_has_two_npu_syncs_per_sample(
    monkeypatch,
):
    ticks = iter(value for index in range(9) for value in (index * 1000, index * 1000 + 10))
    monkeypatch.setattr(evidence.time, "perf_counter_ns", lambda: next(ticks))

    class Synchronizer:
        call_count = 0

        def __call__(self):
            self.call_count += 1

    synchronizer = Synchronizer()
    operations = []
    timing = evidence._measure_scope(
        lambda: operations.append(len(operations)),
        scope="h2d",
        semantic_work="test",
        synchronize=synchronizer,
    )

    assert timing["warmup_samples_ns"] == [10, 10]
    assert timing["raw_samples_ns"] == [10] * 7
    assert timing["median_ns"] == 10.0
    assert timing["p10_ns"] == 10.0
    assert timing["p90_ns"] == 10.0
    assert operations == list(range(9))
    assert synchronizer.call_count == 18
    assert timing["synchronization"] == {
        "api": "aclrtSynchronizeDevice",
        "before_each_sample": True,
        "after_each_sample": True,
        "observed_calls": 18,
        "expected_calls": 18,
        "gate_passed": True,
    }


def test_percentile_uses_declared_linear_interpolation():
    values = [70, 10, 40, 20, 60, 30, 50]
    assert evidence._percentile(values, 0.1) == pytest.approx(16.0)
    assert evidence._percentile(values, 0.5) == 40.0
    assert evidence._percentile(values, 0.9) == pytest.approx(64.0)


def test_worker_commands_pin_both_backends_to_same_core(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(evidence.sys, "executable", "/private/python")
    monkeypatch.setattr(evidence, "__file__", "/repo/tools/runner.py")

    commands = [
        evidence._worker_command(
            worker,
            tmp_path / worker,
            cpu_core=5,
            harness_sha256="h" * 64,
            plan_sha256="p" * 64,
        )
        for worker in ("cpu", "npu")
    ]

    for worker, command in zip(("cpu", "npu"), commands):
        assert command[:5] == [
            "/usr/bin/taskset", "--cpu-list", "5", "/private/python",
            "/repo/tools/runner.py",
        ]
        assert command[5:7] == ["--worker", worker]
    assert commands[0][:5] == commands[1][:5]


def test_write_once_json_is_private_and_never_overwrites(tmp_path):
    path = tmp_path / "new-private-parent" / "evidence.json"
    evidence._write_once_json(path, {"status": "first"})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.read_text()) == {"status": "first"}
    with pytest.raises(FileExistsError):
        evidence._write_once_json(path, {"status": "second"})
    assert json.loads(path.read_text()) == {"status": "first"}


def _timing(value, *, npu):
    synchronization = {
        "api": "aclrtSynchronizeDevice" if npu else None,
        "before_each_sample": npu,
        "after_each_sample": npu,
        "observed_calls": 18 if npu else 0,
        "expected_calls": 18 if npu else 0,
        "gate_passed": True,
    }
    return {
        "warmup_samples_ns": [value, value],
        "raw_samples_ns": [value] * 7,
        "sample_count": 7,
        "median_ns": float(value),
        "p10_ns": float(value),
        "p90_ns": float(value),
        "synchronization": synchronization,
        "process_output": {
            "stdout_stderr_redirected_to_devnull": npu,
        },
    }


def _process(pid):
    return {
        "pid": pid,
        "observed_logical_cpus": [3],
        "affinity_gate_passed": True,
        "thread_environment_gate_passed": True,
        "thread_limit_environment": {
            name: "1" for name in evidence._THREAD_ENVIRONMENT
        },
        "threadpools": [{"user_api": "blas", "num_threads": 1}],
    }


def _identities(size):
    inputs = [
        {"shape": [size], "dtype": "float32", "nbytes": 4 * size, "sha256": "a" * 64},
        {"shape": [size], "dtype": "float32", "nbytes": 4 * size, "sha256": "b" * 64},
    ]
    outputs = [
        {"shape": [size], "dtype": dtype, "nbytes": width * size, "sha256": digest * 64}
        for dtype, width, digest in (
            ("float32", 4, "c"), ("float32", 4, "d"),
            ("float32", 4, "e"), ("bool", 1, "f"),
        )
    ]
    return inputs, outputs


def _memory(size, *, device):
    host_peak = 96 * size + 256 * 1024 * 1024
    device_peak = 64 * size + 256 * 1024 * 1024
    host = {
        "conservative_peak_bytes": host_peak,
        "available_bytes": 10 * host_peak,
        "limit_bytes": 6 * host_peak,
        "limit_fraction": 0.60,
        "gate_passed": True,
    }
    device_value = None
    if device:
        device_value = {
            "conservative_peak_bytes": device_peak,
            "available_bytes": 10 * device_peak,
            "limit_bytes": 6 * device_peak,
            "limit_fraction": 0.60,
            "gate_passed": True,
        }
    return {"host": host, "device": device_value, "gate_passed": True}


def _receipts():
    cpu_scales = []
    npu_scales = []
    for size in evidence._SCALES:
        inputs, outputs = _identities(size)
        cpu_scales.append({
            "size": size,
            "memory": _memory(size, device=False),
            "input_identity": inputs,
            "reference_output_identity": outputs,
            "scopes": {
                scope: _timing(20, npu=False) for scope in evidence._TIMED_SCOPES
            },
        })
        npu_scales.append({
            "size": size,
            "memory": _memory(size, device=True),
            "input_identity": inputs,
            "reference_output_identity": outputs,
            "comparison": {"consistent": True, "max_abs_diff": 0.0},
            "core_direct_comparison": {"consistent": True, "max_abs_diff": 0.0},
            "core_direct_runtime_targets": {"identity_gate_passed": True},
            "capability": {
                "ok": True,
                "preflight_ok": True,
                "preflight_strict_npu": True,
                "strict_npu": True,
                "runtime_profile_ok": True,
                "runtime_target_integrity_ok": True,
                "required_ops": list(evidence._REQUIRED_NPU_APIS),
                "profile_status": "not_required",
            },
            "runtime_target_identity_gate_passed": True,
            "scopes": {
                scope: _timing(10, npu=True) for scope in evidence._TIMED_SCOPES
            },
        })
    cpu = {
        "status": "pass",
        "backend": "numpy",
        "process_capture": {
            "returncode": 0,
            "timeout_seconds": evidence._WORKER_TIMEOUT_SECONDS,
            "credential_environment_scrubbed": True,
        },
        "harness_sha256": "h" * 64,
        "plan_sha256": evidence._FROZEN_PLAN_SHA256,
        "runtime": {"identity_sha256": "u" * 64},
        "process": _process(101),
        "scales": cpu_scales,
    }
    runtime = {"status": "matched", "identity_sha256": "r" * 64}
    npu = {
        "status": "pass",
        "backend": "asnumpy",
        "process_capture": {
            "returncode": 0,
            "timeout_seconds": evidence._WORKER_TIMEOUT_SECONDS,
            "credential_environment_scrubbed": True,
        },
        "harness_sha256": "h" * 64,
        "plan_sha256": evidence._FROZEN_PLAN_SHA256,
        "runtime_before": runtime,
        "runtime_after": dict(runtime),
        "runtime_stable": True,
        "asnumpy_logging": {"applied": True},
        "runtime_initialization": {"gate_passed": True},
        "synchronizer": {
            "symbol": "aclrtSynchronizeDevice",
            "resolved_library": "/runtime/libascendcl.so",
            "library_sha256": "l" * 64,
        },
        "process": _process(202),
        "scales": npu_scales,
    }
    return cpu, npu


def test_combine_publishes_only_equivalent_ratios_when_all_gates_pass():
    cpu, npu = _receipts()
    result = evidence._combine(cpu, npu, cpu_core=3, harness_sha="h" * 64)

    assert result["status"] == "pass"
    assert result["performance_valid"] is True
    assert result["performance_issues"] == []
    assert result["plan_frozen_before_execution"] is True
    for scale in result["comparisons"].values():
        assert scale["performance_valid"] is True
        assert scale["scopes"]["device_resident_compute"]["cpu_time_over_npu_time"] == 2.0
        assert scale["scopes"]["allocation"]["cpu_time_over_npu_time"] == 2.0
        assert scale["scopes"]["core_direct_compute"]["cpu_time_over_npu_time"] is None
        assert scale["scopes"]["h2d"]["cpu_time_over_npu_time"] is None
        assert scale["scopes"]["d2h"]["cpu_time_over_npu_time"] is None
        assert scale["npu_public_wrapper_time_over_core_direct_time"] == 1.0


def test_combine_withholds_every_ratio_after_a_numerical_gate_failure():
    cpu, npu = _receipts()
    npu["scales"][1]["comparison"]["consistent"] = False

    result = evidence._combine(cpu, npu, cpu_core=3, harness_sha="h" * 64)

    assert result["status"] == "pass"
    assert result["performance_valid"] is False
    assert "scale 4000000: CPU/NPU numerical comparison failed" in result[
        "performance_issues"
    ]
    for scale in result["comparisons"].values():
        for scope in scale["scopes"].values():
            assert scope["cpu_time_over_npu_time"] is None


def test_array_identity_binds_shape_dtype_bytes_and_content():
    array = np.array([1.0, 2.0], dtype=np.float32)
    identity = evidence._array_identity(array)

    assert identity == {
        "shape": [2],
        "dtype": "float32",
        "nbytes": 8,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def test_memory_budget_uses_conservative_formula_and_sixty_percent_gate(monkeypatch):
    monkeypatch.setattr(
        evidence,
        "_host_memory_info",
        lambda: {"effective_available_bytes": 2_000_000_000},
    )
    device = {
        "free_bytes": 1_500_000_000,
        "total_bytes": 2_000_000_000,
    }

    budget = evidence._memory_budget(9_000_000, device_info=device)

    assert budget["host"]["conservative_peak_bytes"] == (
        96 * 9_000_000 + 256 * 1024 * 1024
    )
    assert budget["device"]["conservative_peak_bytes"] == (
        64 * 9_000_000 + 256 * 1024 * 1024
    )
    assert budget["host"]["limit_bytes"] == 1_200_000_000
    assert budget["device"]["limit_bytes"] == 900_000_000
    assert budget["gate_passed"] is True


def test_worker_environment_removes_credentials_and_proxies_without_values():
    cleaned, removed = evidence._scrubbed_worker_environment({
        "PATH": "/bin",
        "ASCEND_HOME_PATH": "/runtime",
        "OPENAI_API_KEY": "not-inspected",
        "SERVICE_ACCESS_TOKEN": "not-inspected",
        "HTTPS_PROXY": "http://proxy",
        "NUMPILOT_LLM_CONFIG": "/private/config.json",
    })

    assert cleaned == {"PATH": "/bin", "ASCEND_HOME_PATH": "/runtime"}
    assert removed == [
        "HTTPS_PROXY",
        "NUMPILOT_LLM_CONFIG",
        "OPENAI_API_KEY",
        "SERVICE_ACCESS_TOKEN",
    ]
