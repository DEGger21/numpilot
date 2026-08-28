import json
from pathlib import Path
import stat
import subprocess
import threading

import pytest

from tools import demo_cold_start_evidence as evidence


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _synthetic_inputs(tmp_path: Path):
    source_root = (tmp_path / "demo").resolve()
    source_root.mkdir()
    statuses = {
        "02": "migrated_scoped_npu",
        "03": "migrated_strict_npu",
        "04": "migrated_scoped_npu",
        "06": "migrated_strict_npu",
        "10": "migrated_strict_npu",
    }
    entries = []
    steady_cases = []
    for case in evidence._CANDIDATE_CASES:
        source = source_root / f"{case}_numpy.py"
        run_dir = (tmp_path / f"run-{case}").resolve()
        migrated = run_dir / "migrated.py"
        report = run_dir / "script-migration-report.json"
        _write(source, "import numpy as np\nvalue = np.zeros((1,))\n")
        _write(migrated, "import asnumpy as ap\nvalue = ap.zeros((1,))\n")
        source_sha = evidence._sha256_path(source)
        migrated_sha = evidence._sha256_path(migrated)
        _write(report, json.dumps({
            "status": statuses[case],
            "source_sha256": source_sha,
            "migrated_sha256": migrated_sha,
        }))
        report_sha = evidence._sha256_path(report)
        entries.append({
            "case": case,
            "decision": "DIRECT",
            "status": statuses[case],
            "run_dir": str(run_dir),
            "source_sha256": source_sha,
            "migrated_sha256": migrated_sha,
            "report_sha256": report_sha,
        })
        consistent = case != "02"
        performance_valid = statuses[case] == "migrated_strict_npu" and consistent
        steady_cases.append({
            "case": case,
            "migration_status": statuses[case],
            "binding": {
                "source_sha256": source_sha,
                "migrated_sha256": migrated_sha,
                "report_sha256": report_sha,
            },
            "comparison": {
                "consistent": consistent,
                "reason": None if consistent else "dtype mismatch",
            },
            "performance": {
                "performance_valid": performance_valid,
                "issues": [] if performance_valid else ["diagnostic only"],
            },
        })

    index = tmp_path / "index.json"
    _write(index, json.dumps({
        "source_root": str(source_root),
        "runtime_identity_sha256": "a" * 64,
        "cases": entries,
    }))
    runner = (tmp_path / "steady-runner.py").resolve()
    _write(runner, "# immutable steady harness\n")
    steady = tmp_path / "steady.json"
    _write(steady, json.dumps({
        "status": "completed",
        "producer": "numpilot-demo-script-performance-evidence-v1",
        "runner_path": str(runner),
        "runner_sha256": evidence._sha256_path(runner),
        "index_binding": {
            "index_path": str(index.resolve()),
            "index_sha256": evidence._sha256_path(index),
            "runtime_identity_sha256": "a" * 64,
            "source_root": str(source_root),
        },
        "cases": steady_cases,
    }))
    return index, steady, source_root


def _timing(backend: str, value: int):
    return evidence._timing_summary([
        {
            "pid": 1000 + index,
            "elapsed_ns": value,
            "elapsed_ms": value / 1e6,
            "returncode": 0,
            "timed_out": False,
            "process_group_killed": False,
            "ordinal": index + 1,
        }
        for index in range(7)
    ], backend)


def _spec(status: str, *, consistent=True, steady_gate=True):
    value = {
        "case": "03",
        "decision": "DIRECT",
        "migration_status": status,
        "source_sha256": "a" * 64,
        "migrated_sha256": "b" * 64,
        "report_sha256": "c" * 64,
    }
    steady_case = {
        "comparison": {"consistent": consistent, "reason": None},
        "performance": {"performance_valid": steady_gate, "issues": []},
    }
    value["steady_policy"] = evidence._case_policy(value, steady_case)
    return value


def test_frozen_plan_is_precommitted_for_five_cases_and_seven_processes():
    assert evidence._FROZEN_PLAN["candidate_cases"] == ["02", "03", "04", "06", "10"]
    assert evidence._FROZEN_PLAN["measurement_samples"] == 7
    assert evidence._FROZEN_PLAN["process"]["cpu_timeout_seconds"] == 60
    assert evidence._FROZEN_PLAN["process"]["npu_timeout_seconds"] == 120
    assert evidence._FROZEN_PLAN["process"]["taskset_before_python"] is True
    assert "blocking process.wait()" in evidence._FROZEN_PLAN["process"][
        "exit_observation"
    ]
    assert evidence._plan_sha256() == evidence._FROZEN_PLAN_SHA256


def test_inputs_bind_index_artifacts_report_steady_and_apply_case_policy(tmp_path):
    index, steady, source_root = _synthetic_inputs(tmp_path)

    index_binding, steady_binding, specs = evidence._load_inputs(index, steady)

    assert index_binding["index_sha256"] == evidence._sha256_path(index)
    assert steady_binding["steady_sha256"] == evidence._sha256_path(steady)
    policies = {item["case"]: item["steady_policy"] for item in specs}
    assert policies["02"]["timing_admitted"] is False
    assert policies["02"]["timing_rejection_codes"] == [
        "STEADY_NUMERICAL_INCONSISTENCY"
    ]
    assert policies["03"]["cold_speedup_publication_eligible_before_cold_run"] is True
    assert policies["04"]["timing_admitted"] is True
    assert policies["04"]["cold_speedup_publication_eligible_before_cold_run"] is False

    source = source_root / "03_numpy.py"
    source.write_text(source.read_text() + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="case 03 source SHA-256 differs"):
        evidence._load_inputs(index, steady)


def test_script_policy_forbids_llm_file_network_and_dynamic_import_access():
    evidence._validate_script_access(
        "import math\nimport asnumpy as ap\nx = ap.zeros((1,))\n",
        "migrated.py",
        "npu",
    )
    with pytest.raises(ValueError, match="unexpected imports"):
        evidence._validate_script_access(
            "import asnumpy as ap\nimport requests\n", "bad.py", "npu"
        )
    with pytest.raises(ValueError, match="LLM configuration"):
        evidence._validate_script_access(
            "import numpy as np\nx = '.numpilot-config/llm.json'\n",
            "bad.py",
            "cpu",
        )
    with pytest.raises(ValueError, match="dynamic/file access open"):
        evidence._validate_script_access(
            "import numpy as np\nx = open('anything')\n", "bad.py", "cpu"
        )


def test_environment_removes_tokens_proxies_ssh_and_llm_config_without_values():
    source = {
        "PATH": "/bin",
        "ASCEND_HOME_PATH": "/opt/ascend",
        "HTTPS_PROXY": "credential@proxy",
        "demo_api_key": "secret",
        "HF_TOKEN": "secret",
        "SSH_AUTH_SOCK": "/secret/socket",
        "NUMPILOT_LLM_CONFIG": "/secret/llm.json",
        "OPENBLAS_NUM_THREADS": "32",
    }

    environment, removed = evidence._scrubbed_environment(source)

    assert environment["PATH"] == "/bin"
    assert environment["ASCEND_HOME_PATH"] == "/opt/ascend"
    assert all(environment[name] == "1" for name in evidence._THREAD_ENVIRONMENT)
    assert environment["PYTHONHASHSEED"] == "0"
    assert set(removed) == {
        "HTTPS_PROXY", "demo_api_key", "HF_TOKEN", "SSH_AUTH_SOCK",
        "NUMPILOT_LLM_CONFIG",
    }
    assert not any(value == "secret" for value in environment.values())


def test_command_places_taskset_before_python_on_same_requested_core(monkeypatch):
    monkeypatch.setattr(evidence.sys, "executable", "/private/python")
    spec = {"source_path": "/demo/03_numpy.py", "migrated_path": "/run/migrated.py"}

    cpu = evidence._cold_command(
        spec, "cpu", cpu_core=5, taskset="/usr/bin/taskset"
    )
    npu = evidence._cold_command(
        spec, "npu", cpu_core=5, taskset="/usr/bin/taskset"
    )

    assert cpu == [
        "/usr/bin/taskset", "--cpu-list", "5", "/private/python",
        "/demo/03_numpy.py",
    ]
    assert npu[:4] == cpu[:4]
    assert npu[-1] == "/run/migrated.py"


def test_sample_uses_parent_wall_clock_devnull_timeout_and_new_session(monkeypatch, tmp_path):
    observed = {}

    class Process:
        pid = 4321
        returncode = None

        def wait(self):
            observed["blocking_wait"] = True
            self.returncode = 0
            return 0

    def popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Process()

    ticks = iter((10_000, 25_000))
    monkeypatch.setattr(evidence.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(evidence.subprocess, "Popen", popen)

    sample = evidence._run_cold_sample(
        ["taskset", "python", "script.py"],
        cwd=tmp_path,
        environment={"PATH": "/bin"},
        timeout_seconds=120,
    )

    assert sample["elapsed_ns"] == 15_000
    assert sample["returncode"] == 0
    assert observed["blocking_wait"] is True
    assert observed["kwargs"]["stdin"] == subprocess.DEVNULL
    assert observed["kwargs"]["stdout"] == subprocess.DEVNULL
    assert observed["kwargs"]["stderr"] == subprocess.DEVNULL
    assert observed["kwargs"]["start_new_session"] is True
    assert observed["kwargs"]["close_fds"] is True


def test_sample_timeout_is_enforced_by_watchdog_without_timed_wait(
    monkeypatch, tmp_path
):
    killed = threading.Event()

    class Process:
        pid = 4321
        returncode = None

        def wait(self):
            assert killed.wait(1)
            self.returncode = -9
            return self.returncode

    process = Process()
    monkeypatch.setattr(evidence.subprocess, "Popen", lambda *args, **kwargs: process)

    def terminate(observed_process):
        assert observed_process is process
        killed.set()
        return True

    monkeypatch.setattr(evidence, "_terminate_process_group", terminate)

    sample = evidence._run_cold_sample(
        ["taskset", "python", "script.py"],
        cwd=tmp_path,
        environment={"PATH": "/bin"},
        timeout_seconds=0.001,
    )

    assert sample["returncode"] == -9
    assert sample["timed_out"] is True
    assert sample["process_group_killed"] is True


def test_ordered_raw_samples_and_linear_percentiles_are_retained():
    samples = []
    values = [70, 10, 40, 20, 60, 30, 50]
    for ordinal, value in enumerate(values, 1):
        samples.append({
            "pid": ordinal,
            "elapsed_ns": value,
            "elapsed_ms": value / 1e6,
            "returncode": 0,
            "timed_out": False,
            "process_group_killed": False,
            "ordinal": ordinal,
        })

    timing = evidence._timing_summary(samples, "cpu")

    assert timing["raw_samples_ns"] == values
    assert timing["median_ns"] == 40
    assert timing["p10_ns"] == pytest.approx(16)
    assert timing["p90_ns"] == pytest.approx(64)
    assert timing["all_samples_successful"] is True


def test_strict_can_publish_scoped_is_diagnostic_and_fallback_never_times_npu():
    strict = evidence._combine_case(
        _spec("migrated_strict_npu"),
        _timing("cpu", 20),
        _timing("npu", 10),
        steady_sha256="d" * 64,
        harness_sha256="e" * 64,
    )
    assert strict["performance"]["performance_valid"] is True
    assert strict["performance"]["speedup"] == 2.0

    scoped = evidence._combine_case(
        _spec("migrated_scoped_npu", steady_gate=False),
        _timing("cpu", 20),
        _timing("npu", 10),
        steady_sha256="d" * 64,
        harness_sha256="e" * 64,
    )
    assert scoped["performance"]["performance_valid"] is False
    assert scoped["performance"]["speedup"] is None
    assert scoped["performance"]["diagnostic_cpu_time_over_npu_time"] == 2.0

    fallback_spec = _spec("migrated_with_fallback", steady_gate=False)
    reasons = fallback_spec["steady_policy"]["timing_rejection_codes"]
    fallback = evidence._combine_case(
        fallback_spec,
        evidence._not_performed("cpu", reasons),
        evidence._not_performed("npu", reasons),
        steady_sha256="d" * 64,
        harness_sha256="e" * 64,
    )
    assert fallback["cpu_timing_performed"] is False
    assert fallback["npu_timing_performed"] is False
    assert fallback["npu"]["raw_samples_ns"] == []


def test_numerically_failed_case_is_rejected_before_collecting_any_backend(monkeypatch):
    rejected = _spec("migrated_scoped_npu", consistent=False, steady_gate=False)
    admitted = _spec("migrated_strict_npu")
    admitted["case"] = "06"
    calls = []

    def collect(spec, backend, **kwargs):
        calls.append((spec["case"], backend))
        return _timing(backend, 10)

    monkeypatch.setattr(evidence, "_collect_backend", collect)
    monkeypatch.setattr(evidence, "_assert_spec_unchanged", lambda spec: None)
    results = evidence._run_cases(
        [rejected, admitted],
        cpu_core=0,
        taskset="taskset",
        cwd=Path("/repo"),
        environment={},
        steady_sha256="d" * 64,
        harness_sha256="e" * 64,
    )

    assert calls == [("06", "cpu"), ("06", "npu")]
    assert results[0]["cpu_timing_performed"] is False
    assert results[0]["npu_timing_performed"] is False


def test_private_json_is_write_once(tmp_path):
    path = tmp_path / "private" / "cold.json"
    evidence._write_once_json(path, {"status": "first"})

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"status": "first"}
    with pytest.raises(FileExistsError):
        evidence._write_once_json(path, {"status": "second"})
    assert json.loads(path.read_text()) == {"status": "first"}


def test_summary_field_counts_valid_comparisons_not_claimed_speedups():
    source = Path(evidence.__file__).read_text(encoding="utf-8")

    assert '"valid_strict_cold_comparisons"' in source
    assert "published_strict_cold_speedups" not in source
