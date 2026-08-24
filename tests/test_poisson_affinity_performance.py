from pathlib import Path

import pytest

from tools import poisson_affinity_performance_evidence as affinity


def test_extract_cpu_core_removes_wrapper_option(monkeypatch):
    monkeypatch.setattr(affinity, "_available_cpus", lambda: {2, 3})

    core, forwarded = affinity._extract_cpu_core([
        "--source", "input.py", "--cpu-core", "3", "--out", "result",
    ])

    assert core == 3
    assert forwarded == ["--source", "input.py", "--out", "result"]


def test_extract_cpu_core_rejects_unavailable_core(monkeypatch):
    monkeypatch.setattr(affinity, "_available_cpus", lambda: {4})

    with pytest.raises(ValueError, match="outside the allowed affinity mask"):
        affinity._extract_cpu_core(["--cpu-core=3"])


def test_cpu_worker_records_verified_single_core(monkeypatch):
    monkeypatch.setenv(affinity._CPU_CORE_ENV, "7")
    monkeypatch.setattr(affinity, "_available_cpus", lambda: {7})
    monkeypatch.setattr(
        affinity,
        "_ORIGINAL_CPU_WORKER",
        lambda args: {"status": "pass", "worker_out": args},
    )

    receipt = affinity._cpu_worker_with_affinity("output")

    assert receipt["status"] == "pass"
    assert receipt["cpu_affinity"] == {
        "requested_logical_cpu": 7,
        "observed_logical_cpus": [7],
        "logical_cpu_count": 1,
        "pinned": True,
        "launch_method": "taskset --cpu-list before Python startup",
        "verification_api": "os.sched_getaffinity(0)",
    }


def test_cpu_worker_fails_closed_if_affinity_is_broader(monkeypatch):
    monkeypatch.setenv(affinity._CPU_CORE_ENV, "7")
    monkeypatch.setattr(affinity, "_available_cpus", lambda: {7, 8})

    with pytest.raises(RuntimeError, match="expected exactly"):
        affinity._cpu_worker_with_affinity(None)


def test_only_cpu_worker_command_is_wrapped_in_taskset(monkeypatch):
    monkeypatch.setenv(affinity._CPU_CORE_ENV, "5")
    monkeypatch.setattr(affinity.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(
        affinity,
        "_ORIGINAL_WORKER_COMMAND",
        lambda args, worker, worker_out: ["python", worker, str(worker_out)],
    )

    cpu = affinity._worker_command_with_affinity(None, "cpu", Path("cpu"))
    npu = affinity._worker_command_with_affinity(None, "npu", Path("npu"))

    assert cpu == [
        "/usr/bin/taskset", "--cpu-list", "5", "python", "cpu", "cpu",
    ]
    assert npu == ["python", "npu", "npu"]
