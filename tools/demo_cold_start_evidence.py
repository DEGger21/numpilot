#!/usr/bin/env python3
"""Produce source-bound cold-process evidence for migrated demo scripts.

This harness consumes the immutable Day-3 migration index and a required
steady-state evidence receipt.  It never imports or calls the migration/LLM
path.  For each admitted case it starts seven independent CPU processes and
seven independent NPU processes through ``taskset``.  The parent measures wall
time from immediately before process creation through process termination.

The five-case admission set is fixed.  A numerically inconsistent case or a
fallback artifact is rejected before either script is launched.  Scoped cases
may retain a diagnostic CPU/NPU ratio, but only strict artifacts whose complete
steady-state gate passed can publish a cold-process speedup.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping


_SCHEMA_VERSION = "1.1"
_PRODUCER = "numpilot-demo-cold-start-evidence-v2"
_DEFAULT_INDEX = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/day3-demo-migration-results-20260826.json"
)
_CANDIDATE_CASES = ("02", "03", "04", "06", "10")
_SAMPLES = 7
_TIMEOUT_SECONDS = {"cpu": 60, "npu": 120}
_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_ALLOWED_SCRIPT_IMPORTS = {"asnumpy", "math", "numpy"}
_FORBIDDEN_DYNAMIC_CALLS = {"__import__", "compile", "eval", "exec", "open"}
_SHA256_LENGTH = 64

_FROZEN_PLAN = {
    "schema_version": _SCHEMA_VERSION,
    "candidate_cases": list(_CANDIDATE_CASES),
    "admission": "steady numerical consistent and migration status is not fallback",
    "measurement_samples": _SAMPLES,
    "warmup_samples": 0,
    "clock": "parent time.perf_counter_ns",
    "scope": (
        "wall from immediately before subprocess creation through process exit; "
        "includes taskset, Python startup, imports, runtime initialization, exact "
        "script execution, output formatting to /dev/null, and process teardown"
    ),
    "process": {
        "one_fresh_process_per_sample": True,
        "taskset_before_python": True,
        "same_single_cpu_core": True,
        "stdin": "/dev/null",
        "stdout": "/dev/null",
        "stderr": "/dev/null",
        "start_new_session": True,
        "exit_observation": (
            "blocking process.wait() with an independent timeout watchdog; "
            "no timeout polling in the measured path"
        ),
        "cpu_timeout_seconds": _TIMEOUT_SECONDS["cpu"],
        "npu_timeout_seconds": _TIMEOUT_SECONDS["npu"],
        "thread_environment": {name: "1" for name in _THREAD_ENVIRONMENT},
        "credential_proxy_ssh_environment": "removed",
    },
    "publication": {
        "strict_only": True,
        "requires_steady_numerical_consistency": True,
        "requires_complete_steady_performance_gate": True,
        "requires_all_cold_processes_successful": True,
        "scoped_policy": "diagnostic ratio only",
        "fallback_policy": "no CPU or NPU timing; npu_timing_performed=false",
    },
    "cold_definition": (
        "fresh Python/runtime process; does not reset the device, driver, filesystem "
        "cache, or other machine-global caches between samples"
    ),
}

# Filled from the canonical JSON form of _FROZEN_PLAN before any real run.
_FROZEN_PLAN_SHA256 = "21d57f14ed61637f3e864b7edfb6ba83d4c67d405c6e62a88682890dcab3a803"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_sha256() -> str:
    return _sha256_bytes(_canonical_json_bytes(_FROZEN_PLAN))


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256_path(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: expected={expected} actual={actual}")


def _validate_script_access(source: str, filename: str, backend: str) -> None:
    """Fail closed if a demo can reach configuration, network or dynamic code."""

    tree = ast.parse(source, filename=filename)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(item.name.split(".", 1)[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError(f"relative import is forbidden in cold script: {filename}")
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            normalized = node.value.lower()
            if ".numpilot-config" in normalized or "llm.json" in normalized:
                raise ValueError(f"LLM configuration reference is forbidden: {filename}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_DYNAMIC_CALLS:
                raise ValueError(
                    f"dynamic/file access {node.func.id} is forbidden: {filename}"
                )

    unexpected = imported_roots - _ALLOWED_SCRIPT_IMPORTS
    if unexpected:
        raise ValueError(
            f"unexpected imports in cold script {filename}: {sorted(unexpected)}"
        )
    required = "numpy" if backend == "cpu" else "asnumpy"
    if required not in imported_roots:
        raise ValueError(f"cold {backend} script does not import {required}: {filename}")


def _validate_index(index_path: Path) -> tuple[dict, dict[str, dict]]:
    index_path = index_path.resolve()
    index = _read_json(index_path)
    source_root_value = index.get("source_root")
    if not isinstance(source_root_value, str) or not Path(source_root_value).is_absolute():
        raise ValueError("Day-3 source_root must be an absolute path")
    source_root = Path(source_root_value).resolve()
    runtime_sha = _require_sha256(
        index.get("runtime_identity_sha256"), "Day-3 runtime identity"
    )
    entries = index.get("cases")
    if not isinstance(entries, list):
        raise ValueError("Day-3 cases must be a list")
    by_case: dict[str, Mapping] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Day-3 case entry is not an object")
        case = str(entry.get("case", ""))
        if case in by_case:
            raise ValueError(f"duplicate Day-3 case entry: {case}")
        by_case[case] = entry
    if any(case not in by_case for case in _CANDIDATE_CASES):
        missing = [case for case in _CANDIDATE_CASES if case not in by_case]
        raise ValueError(f"Day-3 index is missing cold candidates: {missing}")

    specs = {}
    for case in _CANDIDATE_CASES:
        entry = by_case[case]
        run_dir_value = entry.get("run_dir")
        if not isinstance(run_dir_value, str) or not Path(run_dir_value).is_absolute():
            raise ValueError(f"case {case} run_dir must be absolute")
        run_dir = Path(run_dir_value).resolve()
        source_path = (source_root / f"{case}_numpy.py").resolve()
        migrated_path = (run_dir / "migrated.py").resolve()
        report_path = (run_dir / "script-migration-report.json").resolve()
        paths = {
            "source": source_path,
            "migrated": migrated_path,
            "report": report_path,
        }
        hashes = {}
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"case {case} {label} file is missing: {path}")
            field = f"{label}_sha256"
            expected = _require_sha256(entry.get(field), f"case {case} index {field}")
            _assert_hash(path, expected, f"case {case} {label}")
            hashes[field] = expected

        report = _read_json(report_path)
        if report.get("status") != entry.get("status"):
            raise ValueError(f"case {case} migration status differs from its report")
        for field in ("source_sha256", "migrated_sha256"):
            if report.get(field) != hashes[field]:
                raise ValueError(f"case {case} report {field} binding differs")

        source_text = source_path.read_text(encoding="utf-8")
        migrated_text = migrated_path.read_text(encoding="utf-8")
        _validate_script_access(source_text, str(source_path), "cpu")
        _validate_script_access(migrated_text, str(migrated_path), "npu")
        specs[case] = {
            "case": case,
            "decision": entry.get("decision"),
            "migration_status": entry.get("status"),
            "source_path": str(source_path),
            "migrated_path": str(migrated_path),
            "report_path": str(report_path),
            **hashes,
        }

    binding = {
        "index_path": str(index_path),
        "index_sha256": _sha256_path(index_path),
        "runtime_identity_sha256": runtime_sha,
        "source_root": str(source_root),
    }
    return binding, specs


def _case_policy(spec: Mapping, steady_case: Mapping) -> dict:
    comparison = steady_case.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"case {spec['case']} steady comparison is missing")
    performance = steady_case.get("performance")
    if not isinstance(performance, dict):
        raise ValueError(f"case {spec['case']} steady performance receipt is missing")

    numerically_consistent = comparison.get("consistent") is True
    fallback = spec.get("migration_status") == "migrated_with_fallback"
    timing_admitted = numerically_consistent and not fallback
    strict = spec.get("migration_status") == "migrated_strict_npu"
    steady_gate_passed = performance.get("performance_valid") is True
    publication_eligible = timing_admitted and strict and steady_gate_passed
    rejection_codes = []
    if not numerically_consistent:
        rejection_codes.append("STEADY_NUMERICAL_INCONSISTENCY")
    if fallback:
        rejection_codes.append("FALLBACK_TIMING_FORBIDDEN")
    publication_issues = []
    if not strict:
        publication_issues.append(
            f"migration status {spec.get('migration_status')} is diagnostic-only"
        )
    if not steady_gate_passed:
        publication_issues.append("complete steady-state performance gate did not pass")
    return {
        "numerically_consistent": numerically_consistent,
        "steady_comparison_reason": comparison.get("reason"),
        "fallback": fallback,
        "timing_admitted": timing_admitted,
        "timing_rejection_codes": rejection_codes,
        "strict_migration": strict,
        "steady_performance_gate_passed": steady_gate_passed,
        "steady_performance_issues": list(performance.get("issues") or []),
        "cold_speedup_publication_eligible_before_cold_run": publication_eligible,
        "publication_issues_before_cold_run": publication_issues,
    }


def _validate_steady(
    steady_path: Path,
    index_binding: Mapping,
    specs: Mapping[str, dict],
) -> tuple[dict, dict[str, dict]]:
    steady_path = steady_path.resolve()
    steady = _read_json(steady_path)
    if steady.get("status") != "completed":
        raise ValueError("steady evidence is not completed")
    if steady.get("producer") != "numpilot-demo-script-performance-evidence-v1":
        raise ValueError("unexpected steady evidence producer")
    steady_index = steady.get("index_binding")
    if not isinstance(steady_index, dict):
        raise ValueError("steady evidence has no Day-3 index binding")
    for field in ("index_path", "index_sha256", "runtime_identity_sha256", "source_root"):
        if steady_index.get(field) != index_binding.get(field):
            raise ValueError(f"steady Day-3 index binding differs: {field}")

    runner_path_value = steady.get("runner_path")
    runner_sha = _require_sha256(steady.get("runner_sha256"), "steady runner")
    if not isinstance(runner_path_value, str) or not Path(runner_path_value).is_absolute():
        raise ValueError("steady runner_path must be absolute")
    runner_path = Path(runner_path_value).resolve()
    if not runner_path.is_file():
        raise FileNotFoundError(f"steady runner is missing: {runner_path}")
    _assert_hash(runner_path, runner_sha, "steady runner")

    cases = steady.get("cases")
    if not isinstance(cases, list):
        raise ValueError("steady evidence cases must be a list")
    steady_cases = {}
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("steady case receipt is not an object")
        case = str(item.get("case", ""))
        if case in steady_cases:
            raise ValueError(f"duplicate steady case receipt: {case}")
        steady_cases[case] = item

    selected = {}
    for case in _CANDIDATE_CASES:
        if case not in steady_cases:
            raise ValueError(f"steady evidence is missing cold candidate {case}")
        item = steady_cases[case]
        spec = specs[case]
        if item.get("migration_status") != spec["migration_status"]:
            raise ValueError(f"case {case} steady migration status differs")
        binding = item.get("binding")
        if not isinstance(binding, dict):
            raise ValueError(f"case {case} steady artifact binding is missing")
        for field in ("source_sha256", "migrated_sha256", "report_sha256"):
            if binding.get(field) != spec[field]:
                raise ValueError(f"case {case} steady {field} binding differs")
        selected[case] = item

    binding = {
        "steady_path": str(steady_path),
        "steady_sha256": _sha256_path(steady_path),
        "producer": steady.get("producer"),
        "steady_runner_path": str(runner_path),
        "steady_runner_sha256": runner_sha,
    }
    return binding, selected


def _load_inputs(index_path: Path, steady_path: Path) -> tuple[dict, dict, list[dict]]:
    index_binding, specs = _validate_index(index_path)
    steady_binding, steady_cases = _validate_steady(
        steady_path, index_binding, specs
    )
    combined = []
    for case in _CANDIDATE_CASES:
        spec = dict(specs[case])
        spec["steady_policy"] = _case_policy(spec, steady_cases[case])
        spec["steady_case"] = steady_cases[case]
        combined.append(spec)
    return index_binding, steady_binding, combined


def _percentile(values: list[int], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0 <= q <= 1:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(int(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _cold_command(
    spec: Mapping,
    backend: str,
    *,
    cpu_core: int,
    taskset: str,
) -> list[str]:
    if backend not in _TIMEOUT_SECONDS:
        raise ValueError("backend must be cpu or npu")
    script = spec["source_path"] if backend == "cpu" else spec["migrated_path"]
    return [taskset, "--cpu-list", str(cpu_core), sys.executable, script]


def _is_sensitive_environment_name(name: str) -> bool:
    upper = name.upper()
    if upper.endswith("_PROXY") or upper.startswith("SSH_"):
        return True
    if upper in {
        "GIT_ASKPASS",
        "NUMPILOT_LLM_CONFIG",
        "NUMPILOT_LLM_CONFIG_PATH",
    }:
        return True
    if upper.startswith(("ANTHROPIC_", "GLM_", "OPENAI_", "ZHIPU_")):
        return True
    return upper.endswith((
        "API_KEY",
        "ACCESS_KEY",
        "ACCESS_TOKEN",
        "AUTH_TOKEN",
        "PASSWORD",
        "PASSWD",
        "PRIVATE_KEY",
        "SECRET",
        "CREDENTIAL",
        "CREDENTIALS",
    )) or upper == "TOKEN" or upper.endswith("_TOKEN")


def _scrubbed_environment(source: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    environment = {}
    removed = []
    for name, value in source.items():
        if _is_sensitive_environment_name(name):
            removed.append(name)
        else:
            environment[name] = value
    for name in _THREAD_ENVIRONMENT:
        environment[name] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment, sorted(removed)


def _terminate_process_group(process) -> bool:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


def _run_cold_sample(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> dict:
    start = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    completed = threading.Event()
    timeout_state = {"timed_out": False, "process_group_killed": False}

    def timeout_watchdog() -> None:
        if completed.wait(timeout_seconds):
            return
        timeout_state["timed_out"] = True
        timeout_state["process_group_killed"] = _terminate_process_group(process)

    watchdog = threading.Thread(
        target=timeout_watchdog,
        name=f"cold-timeout-{process.pid}",
        daemon=True,
    )
    watchdog.start()
    try:
        # A timeout passed to Popen.wait() makes CPython poll with an increasing
        # sleep on POSIX, adding up to about 50 ms to short-process timings.
        # Keep the measured path on the kernel's blocking wait and enforce the
        # timeout independently so process-exit observation is not quantized.
        returncode = process.wait()
        elapsed_ns = time.perf_counter_ns() - start
    finally:
        completed.set()
        watchdog.join()
    return {
        "pid": process.pid,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": elapsed_ns / 1e6,
        "returncode": returncode,
        "timed_out": timeout_state["timed_out"],
        "process_group_killed": timeout_state["process_group_killed"],
    }


def _timing_summary(samples: list[dict], backend: str) -> dict:
    raw_ns = [int(item["elapsed_ns"]) for item in samples]
    successful = (
        len(samples) == _SAMPLES
        and all(
            item.get("returncode") == 0
            and item.get("timed_out") is False
            and item.get("elapsed_ns", 0) > 0
            for item in samples
        )
    )
    median_ns = float(statistics.median(raw_ns)) if successful else None
    p10_ns = _percentile(raw_ns, 0.10) if successful else None
    p90_ns = _percentile(raw_ns, 0.90) if successful else None
    return {
        "timing_performed": True,
        "backend": backend,
        "clock": "parent time.perf_counter_ns",
        "scope": _FROZEN_PLAN["scope"],
        "sample_count": len(samples),
        "expected_sample_count": _SAMPLES,
        "one_fresh_process_per_sample": True,
        "samples": samples,
        "raw_samples_ns": raw_ns,
        "raw_samples_ms": [value / 1e6 for value in raw_ns],
        "all_samples_successful": successful,
        "median_ns": median_ns,
        "p10_ns": p10_ns,
        "p90_ns": p90_ns,
        "median_ms": None if median_ns is None else median_ns / 1e6,
        "p10_ms": None if p10_ns is None else p10_ns / 1e6,
        "p90_ms": None if p90_ns is None else p90_ns / 1e6,
        "timeout_seconds": _TIMEOUT_SECONDS[backend],
        "stdin_stdout_stderr": "/dev/null",
        "start_new_session": True,
    }


def _not_performed(backend: str, reason_codes: list[str]) -> dict:
    return {
        "timing_performed": False,
        "backend": backend,
        "sample_count": 0,
        "expected_sample_count": _SAMPLES,
        "samples": [],
        "raw_samples_ns": [],
        "raw_samples_ms": [],
        "all_samples_successful": False,
        "median_ns": None,
        "p10_ns": None,
        "p90_ns": None,
        "median_ms": None,
        "p10_ms": None,
        "p90_ms": None,
        "reason_codes": list(reason_codes),
    }


def _assert_spec_unchanged(spec: Mapping) -> None:
    for label in ("source", "migrated", "report"):
        _assert_hash(
            Path(spec[f"{label}_path"]),
            spec[f"{label}_sha256"],
            f"case {spec['case']} {label}",
        )


def _collect_backend(
    spec: Mapping,
    backend: str,
    *,
    cpu_core: int,
    taskset: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> dict:
    command = _cold_command(
        spec, backend, cpu_core=cpu_core, taskset=taskset
    )
    samples = []
    for ordinal in range(1, _SAMPLES + 1):
        _assert_spec_unchanged(spec)
        sample = _run_cold_sample(
            command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=_TIMEOUT_SECONDS[backend],
        )
        sample["ordinal"] = ordinal
        samples.append(sample)
    result = _timing_summary(samples, backend)
    result["launch"] = {
        "taskset_path": taskset,
        "taskset_before_python": True,
        "cpu_core": cpu_core,
        "python_executable": sys.executable,
        "script_path": command[-1],
        "script_sha256": (
            spec["source_sha256"] if backend == "cpu" else spec["migrated_sha256"]
        ),
    }
    return result


def _combine_case(
    spec: Mapping,
    cpu_timing: Mapping,
    npu_timing: Mapping,
    *,
    steady_sha256: str,
    harness_sha256: str,
) -> dict:
    policy = spec["steady_policy"]
    timing_admitted = policy["timing_admitted"]
    if not timing_admitted:
        if cpu_timing.get("timing_performed") or npu_timing.get("timing_performed"):
            raise ValueError(f"case {spec['case']} rejected timing was performed")
    if policy["fallback"] and npu_timing.get("timing_performed") is not False:
        raise ValueError(f"case {spec['case']} fallback NPU timing was performed")

    cold_gate = (
        timing_admitted
        and cpu_timing.get("all_samples_successful") is True
        and npu_timing.get("all_samples_successful") is True
    )
    ratio = None
    if cold_gate:
        cpu_median = cpu_timing.get("median_ns")
        npu_median = npu_timing.get("median_ns")
        if (
            isinstance(cpu_median, (int, float))
            and isinstance(npu_median, (int, float))
            and cpu_median > 0
            and npu_median > 0
        ):
            ratio = float(cpu_median) / float(npu_median)
        else:
            cold_gate = False

    publication_issues = list(policy["publication_issues_before_cold_run"])
    if timing_admitted and not cold_gate:
        publication_issues.append("one or more cold subprocess samples failed")
    publication_valid = (
        policy["cold_speedup_publication_eligible_before_cold_run"]
        and cold_gate
        and ratio is not None
    )
    return {
        "case": spec["case"],
        "migration_status": spec["migration_status"],
        "decision": spec["decision"],
        "binding": {
            "source_sha256": spec["source_sha256"],
            "migrated_sha256": spec["migrated_sha256"],
            "report_sha256": spec["report_sha256"],
            "steady_evidence_sha256": steady_sha256,
            "harness_sha256": harness_sha256,
        },
        "steady_policy": policy,
        "cpu_timing_performed": cpu_timing.get("timing_performed") is True,
        "npu_timing_performed": npu_timing.get("timing_performed") is True,
        "cpu": dict(cpu_timing),
        "npu": dict(npu_timing),
        "performance": {
            "cold_process_gate_passed": cold_gate,
            "performance_valid": publication_valid,
            "issues": publication_issues,
            "diagnostic_cpu_time_over_npu_time": ratio,
            "speedup": ratio if publication_valid else None,
            "winner": (
                "npu" if publication_valid and ratio > 1
                else "cpu" if publication_valid and ratio < 1
                else "tie" if publication_valid and ratio == 1
                else None
            ),
            "interpretation": (
                "greater than one means the NPU cold process completed faster; "
                "scoped/non-gated ratios are diagnostic and are not speedup claims"
            ),
        },
    }


def _run_cases(
    specs: list[dict],
    *,
    cpu_core: int,
    taskset: str,
    cwd: Path,
    environment: Mapping[str, str],
    steady_sha256: str,
    harness_sha256: str,
) -> list[dict]:
    results = []
    for spec in specs:
        policy = spec["steady_policy"]
        if policy["timing_admitted"]:
            cpu = _collect_backend(
                spec, "cpu", cpu_core=cpu_core, taskset=taskset,
                cwd=cwd, environment=environment,
            )
            npu = _collect_backend(
                spec, "npu", cpu_core=cpu_core, taskset=taskset,
                cwd=cwd, environment=environment,
            )
            _assert_spec_unchanged(spec)
        else:
            reasons = policy["timing_rejection_codes"]
            cpu = _not_performed("cpu", reasons)
            npu = _not_performed("npu", reasons)
        results.append(_combine_case(
            spec,
            cpu,
            npu,
            steady_sha256=steady_sha256,
            harness_sha256=harness_sha256,
        ))
    return results


def _resolve_cpu_core(value: int | None) -> int:
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("parent has no available logical CPU")
    cpu_core = available[0] if value is None else value
    if cpu_core not in available:
        raise ValueError(f"CPU core {cpu_core} is outside parent affinity {available}")
    return cpu_core


def _write_once_json(path: Path, value: Mapping) -> None:
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700)
    if not parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {parent}")
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _failed_receipt(error: Exception, *, harness_sha256: str | None) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer": _PRODUCER,
        "status": "fail",
        "performance_valid": False,
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": harness_sha256,
        "plan_sha256": _plan_sha256(),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def _run_parent(args) -> int:
    output = Path(args.out).resolve()
    harness_path = Path(__file__).resolve()
    harness_sha = None
    try:
        if output.exists():
            raise FileExistsError(f"--out is write-once and already exists: {output}")
        if _plan_sha256() != _FROZEN_PLAN_SHA256:
            raise RuntimeError("frozen cold-process plan literal hash requires review")
        index_path = Path(args.index).resolve()
        steady_path = Path(args.steady).resolve()
        if output in {index_path, steady_path, harness_path}:
            raise ValueError("--out must differ from every input and the harness")
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("taskset is required")
        cpu_core = _resolve_cpu_core(args.cpu_core)
        harness_sha = _sha256_path(harness_path)
        index_binding, steady_binding, specs = _load_inputs(
            index_path, steady_path
        )
        environment, removed_names = _scrubbed_environment(os.environ)
        cases = _run_cases(
            specs,
            cpu_core=cpu_core,
            taskset=taskset,
            cwd=harness_path.parents[1],
            environment=environment,
            steady_sha256=steady_binding["steady_sha256"],
            harness_sha256=harness_sha,
        )
        _assert_hash(index_path, index_binding["index_sha256"], "Day-3 index")
        _assert_hash(steady_path, steady_binding["steady_sha256"], "steady evidence")
        _assert_hash(harness_path, harness_sha, "cold harness")

        timed = [case for case in cases if case["cpu_timing_performed"]]
        failed_cold = [
            case for case in timed
            if not case["performance"]["cold_process_gate_passed"]
        ]
        status = "completed" if not failed_cold else "fail"
        valid_strict_comparisons = sum(
            case["performance"]["performance_valid"] for case in cases
        )
        result = {
            "schema_version": _SCHEMA_VERSION,
            "producer": _PRODUCER,
            "status": status,
            "performance_valid": not failed_cold,
            "harness_path": str(harness_path),
            "harness_sha256": harness_sha,
            "frozen_plan": _FROZEN_PLAN,
            "plan_sha256": _FROZEN_PLAN_SHA256,
            "plan_frozen_before_execution": True,
            "index_binding": index_binding,
            "steady_binding": steady_binding,
            "execution_environment": {
                "cpu_core": cpu_core,
                "taskset_path": taskset,
                "python_executable": sys.executable,
                "thread_limit_environment": {
                    name: environment[name] for name in _THREAD_ENVIRONMENT
                },
                "python_hash_seed": environment["PYTHONHASHSEED"],
                "credentials_proxies_ssh_scrubbed": True,
                "removed_environment_names": removed_names,
                "removed_environment_values_recorded": False,
                "llm_configuration_access": "forbidden by static script/import policy",
            },
            "summary": {
                "candidate_cases": len(cases),
                "timed_cases": len(timed),
                "rejected_before_launch": len(cases) - len(timed),
                "cold_process_failures": len(failed_cold),
                "valid_strict_cold_comparisons": valid_strict_comparisons,
                "diagnostic_only": len(cases) - valid_strict_comparisons,
            },
            "cases": cases,
        }
    except Exception as error:
        result = _failed_receipt(error, harness_sha256=harness_sha)

    _write_once_json(output, result)
    summary = result.get("summary") or {}
    print(
        f"status={result['status']} timed={summary.get('timed_cases', 0)} "
        f"valid_strict={summary.get('valid_strict_cold_comparisons', 0)} "
        f"out={output}"
    )
    return 0 if result["status"] == "completed" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady", help="required steady evidence JSON path")
    parser.add_argument("--out", help="write-once private cold evidence JSON path")
    parser.add_argument("--index", default=str(_DEFAULT_INDEX))
    parser.add_argument("--cpu-core", type=int)
    args = parser.parse_args(argv)
    if not args.steady:
        parser.error("--steady is required")
    if not args.out:
        parser.error("--out is required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
