import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from tools.demo_script_performance_evidence import (
    _CASE_OUTPUTS,
    _THREAD_ENVIRONMENT,
    _asnumpy_call_counts,
    _combine_case,
    _guarded_module,
    _save_outputs,
    _sha256,
    _silence_process_output,
    _timing_samples,
    _validate_case_entry,
    _worker_command,
    _worker_environment,
    _write_json_exclusive,
)


def test_call_audit_counts_nested_ops_but_not_output_boundaries():
    source = """
import asnumpy as ap
x = ap.full((2, 2), 1.0, dtype=ap.float32)
y = ap.linalg.det(x)
print(y.to_numpy())
"""

    assert _asnumpy_call_counts(source, "migrated.py") == {
        "full": 1,
        "linalg.det": 1,
    }


def test_guarded_module_traces_top_level_and_nested_runtime_calls():
    module = types.ModuleType("asnumpy")
    module.float32 = "dtype"
    module.full = lambda value: value
    module.linalg = SimpleNamespace(det=lambda value: value + 1)
    guarded = []

    class Guard:
        runtime_targets = {
            "full": module.full,
            "linalg.det": module.linalg.det,
        }

        def validate_call(self, name, args, kwargs):
            guarded.append((name, args, kwargs))

    proxy, counts = _guarded_module(
        module, Guard(), ("full", "linalg.det")
    )

    assert proxy.float32 == "dtype"
    assert proxy.full(2) == 2
    assert proxy.linalg.det(2) == 3
    assert counts == {"full": 1, "linalg.det": 1}
    assert [item[0] for item in guarded] == ["full", "linalg.det"]


def test_timing_keeps_ordered_warmups_and_synchronizes_every_sample():
    executions = []
    synchronizations = []

    def run_once():
        executions.append(True)
        return {"value": len(executions)}

    timing = _timing_samples(
        run_once,
        warmup=2,
        repeats=7,
        synchronize=lambda: synchronizations.append(True),
    )

    assert len(timing["warmup_samples_ns"]) == 2
    assert len(timing["samples_ns"]) == 7
    assert len(timing["samples_ms"]) == 7
    assert len(executions) == len(synchronizations) == 9
    assert timing["p10_ms"] <= timing["median_ms"] <= timing["p90_ms"]
    assert timing["clock"] == "time.perf_counter_ns"


def test_native_file_descriptor_output_is_silenced_and_restored(capfd):
    os.write(1, b"before-native\n")
    with _silence_process_output():
        os.write(1, b"hidden-stdout\n")
        os.write(2, b"hidden-stderr\n")
    os.write(2, b"after-native\n")

    captured = capfd.readouterr()
    assert "before-native" in captured.out
    assert "after-native" in captured.err
    assert "hidden" not in captured.out + captured.err


def _case_entry(tmp_path: Path):
    source_root = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_root.mkdir()
    run_dir.mkdir()
    source_path = source_root / "03_numpy.py"
    migrated_path = run_dir / "migrated.py"
    source_path.write_text(
        "import numpy as np\n"
        "diff = np.ones(1, dtype=np.float32)\n"
        "clipped = np.clip(diff, 0, 1)\n"
        "result = np.sqrt(clipped)\n",
        encoding="utf-8",
    )
    migrated_path.write_text(
        "import asnumpy as ap\n"
        "diff = ap.ones(1, dtype=ap.float32)\n"
        "clipped = ap.clip(diff, 0, 1)\n"
        "result = ap.sqrt(clipped)\n",
        encoding="utf-8",
    )
    report_path = run_dir / "script-migration-report.json"
    report_path.write_text(json.dumps({
        "status": "migrated_strict_npu",
        "source_sha256": _sha256(source_path),
        "migrated_sha256": _sha256(migrated_path),
        "qualified_analysis": {
            "decision": "DIRECT",
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": "a" * 64,
            "calls": [
                {"canonical_key": name, "performance_eligible": True}
                for name in ("ones", "clip", "sqrt")
            ],
        },
    }), encoding="utf-8")
    entry = {
        "case": "03",
        "decision": "DIRECT",
        "status": "migrated_strict_npu",
        "run_dir": str(run_dir.resolve()),
        "source_sha256": _sha256(source_path),
        "migrated_sha256": _sha256(migrated_path),
        "report_sha256": _sha256(report_path),
    }
    return source_root, source_path, entry


def test_case_entry_binds_source_artifact_report_and_output_contract(tmp_path):
    source_root, source_path, entry = _case_entry(tmp_path)

    spec = _validate_case_entry(entry, source_root)

    assert spec["case"] == "03"
    assert spec["output_names"] == list(_CASE_OUTPUTS["03"])
    assert spec["asnumpy_call_counts"] == {"clip": 1, "ones": 1, "sqrt": 1}
    assert spec["report_qualification"]["performance_eligible"] is True

    source_path.write_text(source_path.read_text() + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256 does not match"):
        _validate_case_entry(entry, source_root)


def _worker_receipt(backend: str, pid: int):
    return {
        "backend": backend,
        "pid": pid,
        "affinity": [4],
        "affinity_before": [4],
        "affinity_after": [4],
        "thread_limit_environment": {name: "1" for name in _THREAD_ENVIRONMENT},
        "threadpools": [{"user_api": "blas", "num_threads": 1}],
    }


def _case_receipts():
    cpu_case = {
        "timing": {
            "warmup_samples_ns": [1, 1],
            "samples_ns": [2] * 7,
            "median_ms": 2.0,
        },
    }
    npu_case = {
        "timing": {
            "warmup_samples_ns": [1, 1],
            "samples_ns": [1] * 7,
            "median_ms": 1.0,
        },
        "synchronize_count": 9,
        "current_qualification": {
            "decision": "DIRECT",
            "all_calls_performance_eligible": True,
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": "a" * 64,
        },
        "profile_guard": {
            "strict_npu": True,
            "runtime_profile_ok": True,
            "runtime_target_integrity_ok": True,
        },
    }
    return cpu_case, npu_case


def _spec(status="migrated_strict_npu"):
    return {
        "case": "03",
        "migration_status": status,
        "decision": "DIRECT",
        "source_sha256": "s" * 64,
        "migrated_sha256": "m" * 64,
        "report_sha256": "r" * 64,
        "output_names": ["result"],
        "report_qualification": {"performance_eligible": True},
    }


def test_combine_publishes_only_fully_gated_strict_speedup():
    cpu = _worker_receipt("cpu", 100)
    npu = _worker_receipt("npu", 200)
    npu["index_binding"] = {"runtime_identity_sha256": "a" * 64}
    for phase in ("before", "after"):
        npu[f"runtime_identity_{phase}"] = {
            "status": "matched", "identity_sha256": "a" * 64,
        }
    cpu_case, npu_case = _case_receipts()
    output = {"result": np.array([1.0], dtype=np.float32)}

    result = _combine_case(
        _spec(), cpu_case, npu_case, cpu, npu, output, output
    )

    assert result["comparison"]["consistent"] is True
    assert result["performance"]["performance_valid"] is True
    assert result["performance"]["speedup"] == 2.0
    assert result["performance"]["winner"] == "npu"

    scoped = _combine_case(
        _spec("migrated_scoped_npu"),
        cpu_case, npu_case, cpu, npu, output, output,
    )
    assert scoped["performance"]["performance_valid"] is False
    assert scoped["performance"]["speedup"] is None
    assert scoped["performance"]["diagnostic_cpu_time_over_npu_time"] == 2.0

    fallback_npu = {
        **npu_case,
        "timing": None,
        "timing_rejected": {"reason_code": "FALLBACK_NEGATIVE_CONTROL"},
        "synchronize_count": 0,
    }
    fallback = _combine_case(
        _spec("migrated_with_fallback"),
        cpu_case, fallback_npu, cpu, npu, output, output,
    )
    assert fallback["npu"]["timing"] is None
    assert fallback["performance"]["npu_median_ms"] is None
    assert fallback["performance"]["diagnostic_cpu_time_over_npu_time"] is None
    assert fallback["performance"]["performance_valid"] is False
    assert fallback["performance"]["speedup"] is None


def test_dtype_mismatch_is_numerical_failure_and_suppresses_speedup():
    cpu = _worker_receipt("cpu", 100)
    npu = _worker_receipt("npu", 200)
    npu["index_binding"] = {"runtime_identity_sha256": "a" * 64}
    for phase in ("before", "after"):
        npu[f"runtime_identity_{phase}"] = {
            "status": "matched", "identity_sha256": "a" * 64,
        }
    cpu_case, npu_case = _case_receipts()

    result = _combine_case(
        _spec(), cpu_case, npu_case, cpu, npu,
        {"result": np.array(np.nan, dtype=np.float32)},
        {"result": np.array(np.nan, dtype=np.float64)},
    )

    assert result["comparison"]["consistent"] is False
    assert "dtype" in result["comparison"]["reason"]
    assert result["performance"]["performance_valid"] is False
    assert result["performance"]["speedup"] is None


def test_worker_command_applies_taskset_before_python_and_env_is_fixed(tmp_path):
    args = SimpleNamespace(
        core=7,
        index=str(tmp_path / "index.json"),
        warmup=2,
        repeats=7,
    )
    command = _worker_command(args, "cpu", tmp_path / "cpu", "/usr/bin/taskset")

    assert command[:4] == ["/usr/bin/taskset", "--cpu-list", "7", sys.executable]
    environment = _worker_environment()
    assert all(environment[name] == "1" for name in _THREAD_ENVIRONMENT)


def test_outputs_and_json_are_private_and_write_once(tmp_path):
    output = tmp_path / "output.npz"
    _save_outputs(output, {"value": np.array([1], dtype=np.int64)})
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _save_outputs(output, {"value": np.array([2], dtype=np.int64)})

    receipt = tmp_path / "evidence.json"
    _write_json_exclusive(receipt, {"status": "pass"})
    assert receipt.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_json_exclusive(receipt, {"status": "changed"})
