#!/usr/bin/env python3
"""Independently audit a packaged NumPilot ordinary benchmark evidence case.

The auditor intentionally uses only the Python standard library.  It does not
execute user code, import NumPilot/NumPy/AsNumpy, access the network, or touch
the NPU.  It re-hashes the frozen artifacts and recomputes every published
statistic from the raw ordered timing samples.  Numerical arrays are not kept
in the compact public case, so their values cannot be independently replayed;
that boundary is reported explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class EvidenceAuditError(RuntimeError):
    """Raised when frozen evidence does not support its published claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceAuditError(message)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvidenceAuditError(f"cannot read {path}: {error}") from error


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"invalid JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "empty timing sample set")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _bootstrap_ci(
    cpu: Sequence[int], npu: Sequence[int], source_sha256: str
) -> tuple[float, float]:
    rng = random.Random(int(source_sha256[:16], 16))
    ratios: list[float] = []
    for _ in range(2000):
        cpu_draw = [rng.choice(cpu) for _ in cpu]
        npu_draw = [rng.choice(npu) for _ in npu]
        denominator = statistics.median(npu_draw)
        _require(denominator > 0, "non-positive NPU bootstrap median")
        ratios.append(statistics.median(cpu_draw) / denominator)
    return _percentile(ratios, 0.025), _percentile(ratios, 0.975)


def _same(actual: float, expected: Any, label: str) -> None:
    _require(isinstance(expected, (int, float)) and not isinstance(expected, bool),
             f"{label} is not numeric")
    _require(math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-9),
             f"{label} mismatch: recomputed={actual!r}, receipt={expected!r}")


def _all_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        result is True for result in value.values()
    )


def _verify_case_manifest(case: Path) -> tuple[int, int]:
    manifest = case / "MANIFEST.sha256"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceAuditError(f"cannot read case manifest: {error}") from error
    seen: set[str] = set()
    total_bytes = 0
    for number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise EvidenceAuditError(f"invalid manifest line {number}") from error
        _require(len(digest) == 64, f"invalid digest on manifest line {number}")
        _require(relative.startswith("./") and relative != "./MANIFEST.sha256",
                 f"unsafe manifest path on line {number}")
        _require(relative not in seen, f"duplicate manifest path: {relative}")
        target = (case / relative[2:]).resolve()
        try:
            target.relative_to(case)
        except ValueError as error:
            raise EvidenceAuditError(f"manifest path escapes case: {relative}") from error
        _require(target.is_file(), f"manifest target missing: {relative}")
        _require(_sha256(target) == digest, f"manifest digest mismatch: {relative}")
        seen.add(relative)
        total_bytes += target.stat().st_size
    expected = {
        "./" + str(path.relative_to(case))
        for path in case.rglob("*")
        if path.is_file() and path != manifest
    }
    _require(seen == expected, "case manifest is incomplete or contains stale entries")
    return len(seen), total_bytes


def _samples(worker: Mapping[str, Any], expected: int, role: str) -> list[int]:
    timing = worker.get("timing")
    _require(isinstance(timing, dict), f"{role} timing missing")
    samples = timing.get("samples_ns")
    _require(isinstance(samples, list) and len(samples) == expected,
             f"{role} sample count mismatch")
    _require(all(isinstance(value, int) and value > 0 for value in samples),
             f"{role} contains invalid timing samples")
    return samples


def audit_case(case_directory: str | Path) -> dict[str, Any]:
    case = Path(case_directory).resolve()
    _require(case.is_dir(), f"case directory does not exist: {case}")
    manifest_files, manifest_bytes = _verify_case_manifest(case)

    source = case / "source.py"
    migrated = case / "migration" / "migrated.py"
    migration_path = case / "migration" / "script-migration-report.json"
    qualified_path = case / "qualified-analysis.json"
    verification_path = case / "script-verification-report.json"
    benchmark_path = case / "script-benchmark-report.json"
    migration = _json(migration_path)
    qualified = _json(qualified_path)
    verification = _json(verification_path)
    receipt = _json(benchmark_path)

    source_hash = _sha256(source)
    migrated_hash = _sha256(migrated)
    migration_hash = _sha256(migration_path)
    binding = receipt.get("binding")
    _require(isinstance(binding, dict), "benchmark binding missing")
    _require(binding.get("source", {}).get("sha256") == source_hash,
             "benchmark does not bind the packaged source")
    _require(binding.get("migrated", {}).get("sha256") == migrated_hash,
             "benchmark does not bind the packaged migration")
    _require(binding.get("migration_report", {}).get("sha256") == migration_hash,
             "benchmark does not bind the packaged migration report")
    _require(migration.get("source_sha256") == source_hash,
             "migration report source hash mismatch")
    _require(migration.get("migrated_sha256") == migrated_hash,
             "migration report migrated hash mismatch")
    _require(qualified.get("source_sha256") == source_hash,
             "qualified analysis source hash mismatch")
    _require(verification.get("source", {}).get("sha256") == source_hash,
             "verification source hash mismatch")
    _require(verification.get("migration", {}).get("migrated_sha256") == migrated_hash,
             "verification migrated hash mismatch")
    _require(verification.get("migration", {}).get("report_sha256") == migration_hash,
             "verification migration report hash mismatch")

    _require(receipt.get("schema_version") == 1, "unsupported benchmark schema")
    _require(receipt.get("receipt_type") == "numpilot_ordinary_script_benchmark",
             "unexpected receipt type")
    _require(receipt.get("status") == "passed", "benchmark status is not passed")
    performance = receipt.get("performance")
    _require(isinstance(performance, dict), "performance block missing")
    _require(performance.get("performance_valid") is True,
             "performance claim was not admitted")
    _require(performance.get("issues") == [], "performance receipt contains issues")
    _require(_all_true(receipt.get("gates")), "not all benchmark gates passed")
    _require(binding.get("source_migrated_prefix_equivalent") is True,
             "prefix AST equivalence missing")
    _require(binding.get("source_migrated_suffix_equivalent") is True,
             "suffix AST equivalence missing")
    _require(binding.get("suffix_host_escape_free") is True,
             "host-escape audit did not pass")

    _require(migration.get("schema_version") == 2, "unsupported migration schema")
    _require(migration.get("status") == "migrated_strict_npu",
             "migration is not strict-NPU")
    _require(migration.get("actual_remote_call") is True,
             "migration lacks an actual configured-model call")
    llm = migration.get("llm_receipt")
    _require(isinstance(llm, dict), "LLM receipt missing")
    _require(llm.get("http_status") == 200, "LLM HTTP receipt did not pass")
    _require(llm.get("tls_verification") == "enabled", "LLM TLS was not verified")
    _require(llm.get("response_model_identity") == "exact",
             "LLM response model identity is not exact")
    _require(
        migration.get("configured_model")
        == migration.get("requested_model")
        == migration.get("response_model")
        == llm.get("response_model"),
        "configured/requested/response model identities diverge",
    )
    _require(qualified.get("decision") == "DIRECT", "analysis is not DIRECT")
    calls = qualified.get("calls")
    _require(isinstance(calls, list) and bool(calls), "qualified calls missing")
    _require(all(call.get("decision") == "DIRECT" and
                 call.get("performance_eligible") is True for call in calls),
             "a qualified call is not performance eligible")
    _require(verification.get("status") == "passed" and
             verification.get("numerically_consistent") is True,
             "standalone numerical verification did not pass")

    protocol = receipt.get("protocol")
    workers = receipt.get("workers")
    _require(isinstance(protocol, dict) and isinstance(workers, dict),
             "benchmark protocol/workers missing")
    sample_count = protocol.get("samples")
    _require(isinstance(sample_count, int) and sample_count >= 7,
             "benchmark has too few samples")
    cpu_worker = workers.get("cpu")
    npu_worker = workers.get("npu")
    _require(isinstance(cpu_worker, dict) and isinstance(npu_worker, dict),
             "CPU/NPU worker records missing")
    _require(cpu_worker.get("status") == npu_worker.get("status") == "pass",
             "a worker did not pass")
    _require(cpu_worker.get("pid") != npu_worker.get("pid"),
             "CPU/NPU workers did not use separate processes")
    cpu_samples = _samples(cpu_worker, sample_count, "CPU")
    npu_samples = _samples(npu_worker, sample_count, "NPU")
    cpu_median = float(statistics.median(cpu_samples))
    npu_median = float(statistics.median(npu_samples))
    _same(cpu_median / 1e6, performance.get("cpu_median_ms"), "CPU median")
    _same(npu_median / 1e6, performance.get("npu_median_ms"), "NPU median")
    speedup = cpu_median / npu_median
    conservative = _percentile(cpu_samples, 0.10) / _percentile(npu_samples, 0.90)
    ci = _bootstrap_ci(cpu_samples, npu_samples, source_hash)
    _same(speedup, performance.get("median_speedup"), "median speedup")
    _same(conservative, performance.get("conservative_cpu_p10_over_npu_p90"),
          "conservative speedup")
    stored_ci = performance.get("median_speedup_bootstrap_95ci")
    _require(isinstance(stored_ci, list) and len(stored_ci) == 2,
             "bootstrap interval missing")
    _same(ci[0], stored_ci[0], "bootstrap lower bound")
    _same(ci[1], stored_ci[1], "bootstrap upper bound")

    expected_core = [protocol.get("core")]
    for role, worker in (("CPU", cpu_worker), ("NPU", npu_worker)):
        affinity = worker.get("affinity")
        _require(isinstance(affinity, dict) and
                 affinity.get("after_binding") == expected_core and
                 affinity.get("after_execution") == expected_core,
                 f"{role} affinity does not match the protocol core")
        environment = worker.get("thread_environment")
        _require(isinstance(environment, dict) and environment and
                 all(environment.get(name) == "1" for name in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
                 f"{role} thread environment is not one")
    pools = cpu_worker.get("threadpools")
    _require(isinstance(pools, list) and bool(pools) and
             all(pool.get("num_threads") == 1 for pool in pools),
             "CPU BLAS runtime did not report one thread")
    _require(npu_worker.get("runtime_identity_before") ==
             npu_worker.get("runtime_identity_after"),
             "NPU runtime identity changed during execution")
    _require(npu_worker.get("runtime_call_counts", {}).get("expected") ==
             npu_worker.get("runtime_call_counts", {}).get("observed"),
             "NPU runtime call counts are not exact")
    sync = npu_worker.get("synchronization")
    expected_sync = 1 + 2 * (protocol.get("warmup") + sample_count)
    _require(isinstance(sync, dict) and sync.get("observed_calls") == expected_sync
             and sync.get("expected_calls") == expected_sync,
             "NPU synchronization count mismatch")

    component = performance.get("warm_runtime_single_pass_component_sum")
    _require(isinstance(component, dict) and
             component.get("directly_sampled_end_to_end") is False,
             "component sum is incorrectly labelled as sampled end-to-end")
    for role, worker, median, key in (
        ("CPU", cpu_worker, cpu_median, "cpu_ms"),
        ("NPU", npu_worker, npu_median, "npu_ms"),
    ):
        timing = worker["timing"]
        total = (timing["prefix_input_preparation_ns"] + median +
                 timing["final_output_transfer"]["total_ns"])
        _same(total / 1e6, component.get(key), f"{role} component sum")

    boundary = receipt.get("evidence_boundary")
    _require(isinstance(boundary, dict) and boundary.get("device_resident") is True
             and boundary.get("steady_state") is True
             and boundary.get("end_to_end") is False
             and boundary.get("kernel_only") is False,
             "benchmark evidence boundary is ambiguous")
    framework = receipt.get("framework_identity")
    _require(isinstance(framework, dict), "framework identity missing")
    bootstrap_path = case / "bootstrap-receipt.json"
    bootstrap_receipt_bound = False
    if bootstrap_path.exists():
        bootstrap = _json(bootstrap_path)
        runtime = npu_worker.get("runtime_identity_before")
        _require(isinstance(runtime, dict), "NPU runtime identity missing")
        _require(_sha256(bootstrap_path) == runtime.get("bootstrap_receipt_sha256"),
                 "packaged bootstrap receipt hash is not runtime-bound")
        _require(bootstrap.get("status") == "ready" and
                 bootstrap.get("ready") is True,
                 "packaged bootstrap receipt is not ready")
        _require(bootstrap.get("numpilot_version") ==
                 framework.get("numpilot_version"),
                 "bootstrap receipt NumPilot version mismatch")
        repository = bootstrap.get("repositories", {}).get("asnumpy", {})
        wheel = bootstrap.get("wheel", {})
        _require(repository.get("repository") == runtime.get("asnumpy_repository")
                 and repository.get("commit") == runtime.get("asnumpy_commit")
                 and repository.get("tree") == runtime.get("asnumpy_tree"),
                 "bootstrap source identity does not match NPU runtime")
        _require(wheel.get("version") == runtime.get("asnumpy_version"),
                 "bootstrap wheel version does not match NPU runtime")
        installed = wheel.get("installed_files", {})
        _require(installed.get("module", {}).get("sha256") ==
                 runtime.get("module_sha256"),
                 "bootstrap module hash does not match NPU runtime")
        native_hashes = {
            item.get("sha256")
            for item in installed.get("native_modules", [])
            if isinstance(item, dict)
        }
        _require(runtime.get("native_module_sha256") in native_hashes,
                 "bootstrap native-module hash does not match NPU runtime")
        bootstrap_receipt_bound = True
    return {
        "status": "passed",
        "case": str(case),
        "manifest": {"files": manifest_files, "bytes": manifest_bytes},
        "bindings": {
            "source_sha256": source_hash,
            "migrated_sha256": migrated_hash,
            "migration_report_sha256": migration_hash,
        },
        "configured_model": migration.get("configured_model"),
        "actual_remote_call": True,
        "bootstrap_receipt_bound": bootstrap_receipt_bound,
        "hardware": npu_worker["runtime_identity_before"].get("soc_model"),
        "cann_version": npu_worker["runtime_identity_before"].get("cann_version"),
        "samples_per_backend": sample_count,
        "all_benchmark_gates_passed": True,
        "resident": {
            "cpu_median_ms": cpu_median / 1e6,
            "npu_median_ms": npu_median / 1e6,
            "median_speedup": speedup,
            "conservative_speedup": conservative,
            "bootstrap_95ci": [ci[0], ci[1]],
        },
        "evidence_boundary": {
            "steady_device_resident": True,
            "direct_end_to_end_sample": False,
            "kernel_profiler_trace": False,
            "numerical_arrays_in_public_bundle": False,
            "numerical_values_independently_recomputed": False,
            "numeric_receipt_and_output_hashes_preserved": True,
            "development_tree_dirty_at_capture": framework.get("git_dirty"),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of a packaged ordinary benchmark case"
    )
    parser.add_argument("case", type=Path, help="evidence case directory")
    parser.add_argument("--json", action="store_true", help="emit audit JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_case(args.case)
    except EvidenceAuditError as error:
        print(f"evidence audit failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        resident = result["resident"]
        print("NumPilot ordinary evidence audit: passed")
        print(
            "resident CPU %.3f ms, NPU %.3f ms, median %.3fx, conservative %.3fx"
            % (resident["cpu_median_ms"], resident["npu_median_ms"],
               resident["median_speedup"], resident["conservative_speedup"])
        )
        print("boundary: steady device-resident; not direct end-to-end or kernel-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
