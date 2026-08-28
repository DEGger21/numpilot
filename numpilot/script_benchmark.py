"""Fail-closed resident benchmark for an ordinary NumPy migration.

The benchmark is deliberately narrower than :mod:`numpilot.script_verification`.
It accepts only a schema-v2, strict-NPU ordinary migration and a source marker
that separates one-time input preparation from a straight-line resident compute
region.  CPU and NPU execute immutable snapshots in separate, single-core
workers.  A speedup is published only after hash, model, AST-equivalence,
runtime-provenance, synchronization, thread-count, and numerical gates pass.

This is not a kernel-only profiler and it is not a cold-start benchmark.  The
receipt carries the measured input-preparation and final host-copy components,
but the headline scope remains the marked, steady device-resident region.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from numpilot import __version__
from numpilot.comparison import compare_outputs
from numpilot.script_verification import (
    ScriptVerificationError,
    _absolute_path,
    _assert_inputs_unchanged,
    _load_bound_inputs,
    _prepare_receipt_path,
    _python_executable,
    _read_regular,
    _sha256,
    _terminate_process_group,
    _utc_now,
    _validate_output_names,
    _worker_environment,
    _write_private,
)


DEFAULT_MARKER = "# NUMPILOT_DEVICE_INPUTS_END"
_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_SCRIPT_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
_MIN_SAMPLES = 7
_SHA256 = frozenset("0123456789abcdef")

_MESSAGES = {
    "invalid_argument": "benchmark arguments are invalid",
    "invalid_report": "migration report is not eligible for strict benchmarking",
    "invalid_marker": "source and migrated artifact need one valid timing marker",
    "segment_mismatch": "source and migrated timing segments are not AST-equivalent",
    "unsafe_timed_region": "timed region is outside the resident straight-line contract",
    "python_unavailable": "benchmark Python executable is unavailable",
    "worker_failed": "benchmark worker failed without admissible evidence",
    "worker_timeout": "benchmark worker exceeded its timeout",
    "invalid_worker_output": "benchmark worker evidence is invalid",
    "input_drift": "a hash-bound benchmark input changed during execution",
    "performance_gate_failed": "one or more strict performance gates failed",
    "receipt_exists": "benchmark receipt already exists",
    "receipt_publish_failed": "benchmark receipt could not be published",
    "internal_error": "benchmark failed closed due to an internal error",
}


class ScriptBenchmarkError(RuntimeError):
    """Stable failure raised by :func:`benchmark_script`."""

    def __init__(
        self,
        code: str,
        message: Optional[str] = None,
        *,
        phase: str = "preflight",
        exception_type: Optional[str] = None,
        receipt: Optional["ScriptBenchmarkReceipt"] = None,
    ) -> None:
        super().__init__(message or _MESSAGES.get(code, _MESSAGES["internal_error"]))
        self.code = code
        self.phase = phase
        self.exception_type = exception_type
        self.receipt = receipt


@dataclass(frozen=True)
class ScriptBenchmarkReceipt:
    """Published, content-addressed ordinary-script benchmark receipt."""

    path: str
    sha256: str
    payload: Mapping[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.payload, ensure_ascii=False))


@dataclass(frozen=True)
class _WorkerResult:
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    wall_ns: int
    removed_secret_variables: int


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256)
    )


def _validate_limits(
    warmup: int,
    samples: int,
    timeout_seconds: float,
    max_output_bytes: int,
) -> None:
    if (
        isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or not 0 <= warmup <= 100
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or not _MIN_SAMPLES <= samples <= 1000
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 1.0 <= float(timeout_seconds) <= 7200.0
        or isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or not 1024 <= max_output_bytes <= _MAX_OUTPUT_BYTES
    ):
        raise ScriptBenchmarkError("invalid_argument")


def _validated_output_names(output_names: Sequence[str]) -> Tuple[str, ...]:
    try:
        return _validate_output_names(output_names)
    except ScriptVerificationError as exc:
        raise ScriptBenchmarkError(
            "invalid_argument", phase=exc.phase, exception_type=exc.exception_type
        ) from exc


def _validated_python(value):
    try:
        return _python_executable(value)
    except ScriptVerificationError as exc:
        raise ScriptBenchmarkError(
            "python_unavailable", phase=exc.phase, exception_type=exc.exception_type
        ) from exc


def _assert_unchanged(inputs) -> None:
    try:
        _assert_inputs_unchanged(inputs)
    except ScriptVerificationError as exc:
        raise ScriptBenchmarkError(
            "input_drift", phase=exc.phase, exception_type=exc.exception_type
        ) from exc


def _validate_core(core: Optional[int]) -> int:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise ScriptBenchmarkError(
            "invalid_argument", phase="cpu_affinity", exception_type=type(exc).__name__
        ) from exc
    if not allowed:
        raise ScriptBenchmarkError("invalid_argument", phase="cpu_affinity")
    selected = allowed[0] if core is None else core
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in allowed
    ):
        raise ScriptBenchmarkError("invalid_argument", phase="cpu_affinity")
    return selected


def _strict_report_gates(report: Mapping[str, Any]) -> Dict[str, bool]:
    qualified = report.get("qualified_analysis")
    receipt = report.get("llm_receipt")
    calls = qualified.get("calls") if isinstance(qualified, dict) else None
    gates = {
        "schema_v2_ordinary": (
            report.get("schema_version") == 2 and report.get("mode") == "script"
        ),
        "strict_status": report.get("status") == "migrated_strict_npu",
        "strict_npu": report.get("strict_npu") is True,
        "no_prior_performance_claim": report.get("performance_valid") is False,
        "ast_audit": report.get("ast_audit") == "passed",
        "model_remote_call": report.get("actual_remote_call") is True,
        "qualified_direct": (
            isinstance(qualified, dict) and qualified.get("decision") == "DIRECT"
        ),
        "qualified_runtime_matched": (
            isinstance(qualified, dict)
            and qualified.get("runtime_identity_status") == "matched"
            and _is_sha256(qualified.get("runtime_identity_sha256"))
        ),
        "all_calls_performance_eligible": (
            isinstance(calls, list)
            and bool(calls)
            and all(
                isinstance(call, dict)
                and call.get("decision") == "DIRECT"
                and call.get("performance_eligible") is True
                for call in calls
            )
        ),
        "llm_http_200": isinstance(receipt, dict) and receipt.get("http_status") == 200,
        "llm_tls_verified": (
            isinstance(receipt, dict) and receipt.get("tls_verification") == "enabled"
        ),
        "llm_model_exact": (
            isinstance(receipt, dict)
            and receipt.get("response_model_identity") == "exact"
        ),
    }
    return gates


class _NamespaceNormalizer(ast.NodeTransformer):
    """Canonicalize only the NumPy/AsNumPy module namespace."""

    def __init__(self, expected_module: str) -> None:
        self.expected_module = expected_module
        self.aliases: set[str] = set()
        self.import_count = 0

    def visit_Import(self, node: ast.Import):  # noqa: N802 - ast API
        rewritten = []
        for item in node.names:
            if item.name != self.expected_module:
                raise ScriptBenchmarkError(
                    "unsafe_timed_region", phase="segment_import_audit"
                )
            self.import_count += 1
            alias = item.asname or item.name
            self.aliases.add(alias)
            rewritten.append(ast.alias(name="arraylib", asname="xp"))
        node.names = rewritten
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):  # noqa: N802 - ast API
        raise ScriptBenchmarkError(
            "unsafe_timed_region", phase="segment_import_audit"
        )

    def visit_Name(self, node: ast.Name):  # noqa: N802 - ast API
        if node.id in self.aliases:
            node.id = "xp"
        return node


def _parse_segments(
    payload: bytes,
    *,
    path: Path,
    marker: str,
    module_name: str,
) -> Tuple[ast.Module, ast.Module, str, str]:
    try:
        source = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ScriptBenchmarkError(
            "invalid_marker", exception_type=type(exc).__name__
        ) from exc
    if (
        not isinstance(marker, str)
        or not marker
        or "\n" in marker
        or "\r" in marker
        or len(marker) > 256
        or source.count(marker) != 1
    ):
        raise ScriptBenchmarkError("invalid_marker")
    prefix_text, suffix_text = source.split(marker, 1)
    try:
        prefix = ast.parse(prefix_text, filename=str(path) + ":prefix")
        suffix = ast.parse(suffix_text, filename=str(path) + ":suffix")
    except SyntaxError as exc:
        raise ScriptBenchmarkError(
            "invalid_marker", phase="segment_parse", exception_type=type(exc).__name__
        ) from exc
    normalizer = _NamespaceNormalizer(module_name)
    prefix = normalizer.visit(prefix)
    suffix = normalizer.visit(suffix)
    ast.fix_missing_locations(prefix)
    ast.fix_missing_locations(suffix)
    if normalizer.import_count != 1 or normalizer.aliases == set():
        raise ScriptBenchmarkError(
            "unsafe_timed_region", phase="segment_import_audit"
        )
    return prefix, suffix, prefix_text, suffix_text


def _attribute_path(node: ast.AST) -> Optional[Tuple[str, ...]]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def _audit_suffix(tree: ast.Module) -> None:
    if not tree.body:
        raise ScriptBenchmarkError("unsafe_timed_region", phase="empty_suffix")
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            raise ScriptBenchmarkError(
                "unsafe_timed_region", phase="suffix_statement_audit"
            )
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(not isinstance(target, ast.Name) for target in targets):
            raise ScriptBenchmarkError(
                "unsafe_timed_region", phase="suffix_assignment_audit"
            )
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Lambda,
        ast.AugAssign,
        ast.NamedExpr,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ScriptBenchmarkError(
                "unsafe_timed_region", phase="suffix_control_flow_audit"
            )
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise ScriptBenchmarkError(
                "unsafe_timed_region", phase="suffix_control_flow_audit"
            )
        if isinstance(node, ast.Call):
            path = _attribute_path(node.func)
            if (
                path is None
                or len(path) != 2
                or path[0] != "xp"
                or path[-1]
                in {"array", "asarray", "from_numpy", "to_numpy", "item", "copy"}
                or any(keyword.arg == "out" for keyword in node.keywords)
                or any(keyword.arg is None for keyword in node.keywords)
            ):
                raise ScriptBenchmarkError(
                    "unsafe_timed_region", phase="suffix_call_audit"
                )
            calls += 1
    if calls == 0:
        raise ScriptBenchmarkError("unsafe_timed_region", phase="suffix_call_audit")


def _ast_digest(tree: ast.AST) -> str:
    rendered = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _segment_binding(inputs, marker: str) -> Dict[str, Any]:
    source_prefix, source_suffix, _, _ = _parse_segments(
        inputs.source_payload,
        path=inputs.source_path,
        marker=marker,
        module_name="numpy",
    )
    migrated_prefix, migrated_suffix, _, _ = _parse_segments(
        inputs.migrated_payload,
        path=inputs.migrated_path,
        marker=marker,
        module_name="asnumpy",
    )
    _audit_suffix(source_suffix)
    _audit_suffix(migrated_suffix)
    source_prefix_sha = _ast_digest(source_prefix)
    source_suffix_sha = _ast_digest(source_suffix)
    migrated_prefix_sha = _ast_digest(migrated_prefix)
    migrated_suffix_sha = _ast_digest(migrated_suffix)
    if (
        source_prefix_sha != migrated_prefix_sha
        or source_suffix_sha != migrated_suffix_sha
    ):
        raise ScriptBenchmarkError("segment_mismatch", phase="segment_ast_comparison")
    return {
        "marker": marker,
        "prefix_ast_sha256": source_prefix_sha,
        "suffix_ast_sha256": source_suffix_sha,
        "source_migrated_prefix_equivalent": True,
        "source_migrated_suffix_equivalent": True,
        "suffix_host_escape_free": True,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ScriptBenchmarkError("invalid_worker_output", phase="statistics")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _bootstrap_speedup_ci(
    cpu_ns: Sequence[int], npu_ns: Sequence[int], seed_hex: str
) -> Tuple[float, float]:
    rng = random.Random(int(seed_hex[:16], 16))
    ratios = []
    for _ in range(2000):
        cpu = [rng.choice(cpu_ns) for _ in cpu_ns]
        npu = [rng.choice(npu_ns) for _ in npu_ns]
        denominator = statistics.median(npu)
        if denominator > 0:
            ratios.append(statistics.median(cpu) / denominator)
    if not ratios:
        raise ScriptBenchmarkError("invalid_worker_output", phase="statistics")
    return _percentile(ratios, 0.025), _percentile(ratios, 0.975)


def _read_json(path: Path, limit: int = _MAX_MANIFEST_BYTES) -> Mapping[str, Any]:
    try:
        raw = _read_regular(path, limit)
        value = json.loads(raw.decode("utf-8"))
    except (ScriptVerificationError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScriptBenchmarkError(
            "invalid_worker_output",
            phase="worker_evidence",
            exception_type=type(exc).__name__,
        ) from exc
    if not isinstance(value, dict):
        raise ScriptBenchmarkError("invalid_worker_output", phase="worker_evidence")
    return value


def _load_outputs(
    output_dir: Path,
    manifest: Mapping[str, Any],
    names: Tuple[str, ...],
    max_output_bytes: int,
) -> Mapping[str, np.ndarray]:
    entries = manifest.get("outputs")
    if not isinstance(entries, list) or len(entries) != len(names):
        raise ScriptBenchmarkError("invalid_worker_output", phase="worker_outputs")
    expected = {"manifest.json"}
    arrays: Dict[str, np.ndarray] = {}
    total = 0
    for index, (name, entry) in enumerate(zip(names, entries)):
        filename = "output-%04d.npy" % index
        if (
            not isinstance(entry, dict)
            or entry.get("name") != name
            or entry.get("filename") != filename
            or not _is_sha256(entry.get("sha256"))
            or not isinstance(entry.get("serialized_bytes"), int)
        ):
            raise ScriptBenchmarkError("invalid_worker_output", phase="worker_outputs")
        payload = _read_regular(output_dir / filename, max_output_bytes)
        if len(payload) != entry["serialized_bytes"] or _sha256(payload) != entry["sha256"]:
            raise ScriptBenchmarkError("invalid_worker_output", phase="worker_outputs")
        total += len(payload)
        if total > max_output_bytes:
            raise ScriptBenchmarkError("invalid_worker_output", phase="worker_outputs")
        try:
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise ScriptBenchmarkError(
                "invalid_worker_output",
                phase="worker_outputs",
                exception_type=type(exc).__name__,
            ) from exc
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ScriptBenchmarkError("invalid_worker_output", phase="worker_outputs")
        arrays[name] = array
        expected.add(filename)
    try:
        actual = {path.name for path in output_dir.iterdir()}
    except OSError as exc:
        raise ScriptBenchmarkError(
            "invalid_worker_output", phase="worker_outputs", exception_type=type(exc).__name__
        ) from exc
    if actual != expected:
        raise ScriptBenchmarkError("invalid_worker_output", phase="worker_outputs")
    return arrays


def _run_worker(
    *,
    root: Path,
    role: str,
    python_identity,
    script_path: Path,
    source_path: Path,
    report_path: Optional[Path],
    script_sha256: str,
    source_sha256: str,
    report_sha256: Optional[str],
    worker_sha256: str,
    marker: str,
    output_names: Tuple[str, ...],
    warmup: int,
    samples: int,
    core: int,
    timeout_seconds: float,
    max_output_bytes: int,
) -> _WorkerResult:
    side = root / role
    side.mkdir(mode=0o700)
    home = side / "home"
    home.mkdir(mode=0o700)
    output_dir = side / "outputs"
    output_dir.mkdir(mode=0o700)
    contract = {
        "schema_version": 1,
        "role": role,
        "backend": role,
        "script_sha256": script_sha256,
        "source_sha256": source_sha256,
        "migration_report_sha256": report_sha256,
        "migration_report": None if report_path is None else str(report_path),
        "reference_source": None if role == "cpu" else str(source_path),
        "marker": marker,
        "output_names": list(output_names),
        "warmup": warmup,
        "samples": samples,
        "cpu_threads": 1,
        "max_output_bytes": max_output_bytes,
        "worker_sha256": worker_sha256,
    }
    contract_path = side / "contract.json"
    contract_payload = (
        json.dumps(contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    _write_private(contract_path, contract_payload)
    environment, removed = _worker_environment(home)
    for name in _THREAD_ENVIRONMENT:
        environment[name] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    command = [
        python_identity.invocation_path,
        "-I",
        "-m",
        "numpilot._script_benchmark_worker",
        "--backend",
        role,
        "--script",
        str(script_path),
        "--contract",
        str(contract_path),
        "--out",
        str(output_dir),
        "--core",
        str(core),
    ]
    started = __import__("time").perf_counter_ns()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(side),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise ScriptBenchmarkError(
            "worker_failed", phase=role + ":process_start", exception_type=type(exc).__name__
        ) from exc
    try:
        try:
            process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            raise ScriptBenchmarkError(
                "worker_timeout", phase=role + ":execution"
            ) from exc
    finally:
        _terminate_process_group(process)
    wall_ns = __import__("time").perf_counter_ns() - started
    if process.returncode != 0:
        error = None
        try:
            error = _read_json(output_dir / "error.json", 32 * 1024)
        except ScriptBenchmarkError:
            pass
        code = error.get("code") if isinstance(error, dict) else None
        phase = error.get("phase") if isinstance(error, dict) else None
        raise ScriptBenchmarkError(
            "worker_failed",
            message=("benchmark worker rejected execution: %s" % code) if code else None,
            phase=(role + ":" + phase) if isinstance(phase, str) else role + ":execution",
        )
    manifest = _read_json(output_dir / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") not in {"completed", "pass"}
        or manifest.get("backend") != role
        or manifest.get("script_sha256") != script_sha256
        or manifest.get("worker_sha256") != worker_sha256
        or manifest.get("contract_sha256") != _sha256(contract_payload)
        or not isinstance(manifest.get("pid"), int)
        or isinstance(manifest.get("pid"), bool)
        or manifest.get("pid") <= 0
    ):
        raise ScriptBenchmarkError("invalid_worker_output", phase=role + ":manifest")
    arrays = _load_outputs(output_dir, manifest, output_names, max_output_bytes)
    return _WorkerResult(
        manifest=manifest,
        arrays=arrays,
        wall_ns=wall_ns,
        removed_secret_variables=removed,
    )


def _timing_samples(manifest: Mapping[str, Any]) -> Tuple[int, ...]:
    timing = manifest.get("timing")
    values = timing.get("samples_ns") if isinstance(timing, dict) else None
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        )
    ):
        raise ScriptBenchmarkError("invalid_worker_output", phase="timing")
    return tuple(values)


def _warm_runtime_component_sum_ns(
    manifest: Mapping[str, Any], resident_median_ns: float
) -> float:
    timing = manifest.get("timing")
    transfer = timing.get("final_output_transfer") if isinstance(timing, dict) else None
    prefix = timing.get("prefix_input_preparation_ns") if isinstance(timing, dict) else None
    final_copy = transfer.get("total_ns") if isinstance(transfer, dict) else None
    if (
        not isinstance(prefix, int)
        or isinstance(prefix, bool)
        or prefix < 0
        or not isinstance(final_copy, int)
        or isinstance(final_copy, bool)
        or final_copy < 0
    ):
        raise ScriptBenchmarkError(
            "invalid_worker_output", phase="component_timing"
        )
    return float(prefix) + float(resident_median_ns) + float(final_copy)


def _worker_gates(
    cpu: _WorkerResult,
    npu: _WorkerResult,
    *,
    core: int,
    warmup: int,
    samples: int,
) -> Dict[str, bool]:
    cpu_manifest = cpu.manifest
    npu_manifest = npu.manifest
    cpu_pools = [
        item
        for item in cpu_manifest.get("threadpools", [])
        if isinstance(item, dict) and item.get("user_api") == "blas"
    ]
    migration_binding = npu_manifest.get("migration_report")
    report_gates = (
        migration_binding.get("gates")
        if isinstance(migration_binding, dict)
        else None
    )
    qualification = npu_manifest.get("current_qualification")
    before = npu_manifest.get("runtime_identity_before")
    after = npu_manifest.get("runtime_identity_after")
    guard = npu_manifest.get("profile_guard")
    call_counts = npu_manifest.get("runtime_call_counts")
    observed_counts = (
        call_counts.get("observed") if isinstance(call_counts, dict) else None
    )
    expected_counts = (
        call_counts.get("expected") if isinstance(call_counts, dict) else None
    )
    synchronization = npu_manifest.get("synchronization")
    expected_sync = 1 + 2 * (warmup + samples)
    return {
        "separate_processes": cpu_manifest.get("pid") != npu_manifest.get("pid"),
        "same_single_core": all(
            (manifest.get("affinity") or {}).get(field) == [core]
            for manifest in (cpu_manifest, npu_manifest)
            for field in ("after_binding", "after_execution")
        ),
        "thread_environment_one": all(
            manifest.get("thread_environment", {}).get(name) == "1"
            for manifest in (cpu_manifest, npu_manifest)
            for name in _THREAD_ENVIRONMENT
        ),
        "cpu_blas_one": bool(cpu_pools)
        and all(item.get("num_threads") == 1 for item in cpu_pools),
        "worker_sample_counts": all(
            len((manifest.get("timing") or {}).get("samples_ns", [])) == samples
            and len((manifest.get("timing") or {}).get("warmup_ns", [])) == warmup
            for manifest in (cpu_manifest, npu_manifest)
        ),
        "worker_report_gates": isinstance(report_gates, dict)
        and bool(report_gates)
        and all(value is True for value in report_gates.values()),
        "current_qualification": isinstance(qualification, dict)
        and qualification.get("decision") == "DIRECT"
        and qualification.get("all_calls_performance_eligible") is True,
        "runtime_identity_stable": isinstance(before, dict)
        and isinstance(after, dict)
        and before.get("status") == after.get("status") == "matched"
        and before.get("identity_sha256") == after.get("identity_sha256"),
        "runtime_profile_guard": isinstance(guard, dict)
        and all(
            guard.get(name) is True
            for name in (
                "strict_npu",
                "runtime_profile_ok",
                "runtime_target_integrity_ok",
            )
        ),
        "runtime_calls_exact": isinstance(observed_counts, dict)
        and bool(observed_counts)
        and observed_counts == expected_counts,
        "sync_calls_exact": isinstance(synchronization, dict)
        and synchronization.get("prefix_completion_calls") == 1
        and synchronization.get("before_and_after_each_resident_execution") is True
        and synchronization.get("observed_calls") == expected_sync
        and synchronization.get("expected_calls") == expected_sync,
    }


def _git_identity() -> Mapping[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _publish(path: Path, payload: Mapping[str, Any]) -> ScriptBenchmarkReceipt:
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ScriptBenchmarkError("receipt_publish_failed")
    try:
        _write_private(path, raw)
    except FileExistsError as exc:
        raise ScriptBenchmarkError("receipt_exists") from exc
    except OSError as exc:
        raise ScriptBenchmarkError(
            "receipt_publish_failed", exception_type=type(exc).__name__
        ) from exc
    return ScriptBenchmarkReceipt(path=str(path), sha256=_sha256(raw), payload=dict(payload))


def benchmark_script(
    source_path: os.PathLike[str] | str,
    migration_report_path: os.PathLike[str] | str,
    output_names: Sequence[str],
    out_path: os.PathLike[str] | str,
    *,
    migrated_path: os.PathLike[str] | str | None = None,
    marker: str = DEFAULT_MARKER,
    core: Optional[int] = None,
    warmup: int = 3,
    samples: int = 20,
    timeout_seconds: float = 600.0,
    max_output_bytes: int = 64 * 1024 * 1024,
    source_python_executable: os.PathLike[str] | str | None = None,
    migrated_python_executable: os.PathLike[str] | str | None = None,
) -> ScriptBenchmarkReceipt:
    """Benchmark one hash-bound strict ordinary migration.

    A failure after preflight publishes a failed receipt when possible and sets
    ``ScriptBenchmarkError.receipt``.  Preflight/path errors do not publish.
    """

    names = _validated_output_names(output_names)
    _validate_limits(warmup, samples, timeout_seconds, max_output_bytes)
    selected_core = _validate_core(core)
    source = _absolute_path(source_path, code="invalid_argument")
    report_path = _absolute_path(migration_report_path, code="invalid_argument")
    receipt_path = _absolute_path(out_path, code="invalid_argument")
    try:
        _prepare_receipt_path(receipt_path, (source, report_path))
        inputs = _load_bound_inputs(source, report_path, migrated_path)
    except ScriptVerificationError as exc:
        code = (
            "receipt_exists"
            if exc.code == "receipt_exists"
            else "invalid_argument"
            if exc.code in {"invalid_output_path", "output_aliases_input"}
            else "invalid_report"
        )
        raise ScriptBenchmarkError(
            code, phase=exc.phase, exception_type=exc.exception_type
        ) from exc
    if os.path.realpath(receipt_path) == os.path.realpath(inputs.migrated_path):
        raise ScriptBenchmarkError("invalid_argument")
    report_gates = _strict_report_gates(inputs.report)
    if not all(report_gates.values()):
        raise ScriptBenchmarkError("invalid_report", phase="strict_report_gates")
    segment = _segment_binding(inputs, marker)
    source_python = _validated_python(source_python_executable)
    migrated_python = (
        source_python
        if migrated_python_executable is None
        else _validated_python(migrated_python_executable)
    )
    started_at = _utc_now()
    workers: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    try:
        _assert_unchanged(inputs)
        worker_source = Path(__file__).with_name("_script_benchmark_worker.py")
        worker_payload = _read_regular(worker_source, _MAX_SCRIPT_BYTES)
        worker_sha256 = _sha256(worker_payload)
        with tempfile.TemporaryDirectory(prefix="numpilot-script-benchmark-") as raw_root:
            root = Path(raw_root)
            os.chmod(root, 0o700)
            source_snapshot = root / "source.py"
            migrated_snapshot = root / "migrated.py"
            report_snapshot = root / "script-migration-report.json"
            _write_private(source_snapshot, inputs.source_payload)
            _write_private(migrated_snapshot, inputs.migrated_payload)
            _write_private(report_snapshot, inputs.report_payload)
            cpu = _run_worker(
                root=root,
                role="cpu",
                python_identity=source_python,
                script_path=source_snapshot,
                source_path=source_snapshot,
                report_path=None,
                script_sha256=inputs.source_sha256,
                source_sha256=inputs.source_sha256,
                report_sha256=None,
                worker_sha256=worker_sha256,
                marker=marker,
                output_names=names,
                warmup=warmup,
                samples=samples,
                core=selected_core,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
            workers["cpu"] = dict(cpu.manifest)
            workers["cpu"]["fresh_worker_wall_ns"] = cpu.wall_ns
            workers["cpu"]["removed_sensitive_environment_variables"] = (
                cpu.removed_secret_variables
            )
            _assert_unchanged(inputs)
            npu = _run_worker(
                root=root,
                role="npu",
                python_identity=migrated_python,
                script_path=migrated_snapshot,
                source_path=source_snapshot,
                report_path=report_snapshot,
                script_sha256=inputs.migrated_sha256,
                source_sha256=inputs.source_sha256,
                report_sha256=inputs.report_sha256,
                worker_sha256=worker_sha256,
                marker=marker,
                output_names=names,
                warmup=warmup,
                samples=samples,
                core=selected_core,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
            workers["npu"] = dict(npu.manifest)
            workers["npu"]["fresh_worker_wall_ns"] = npu.wall_ns
            workers["npu"]["removed_sensitive_environment_variables"] = (
                npu.removed_secret_variables
            )
            _assert_unchanged(inputs)
            for name in names:
                comparisons[name] = asdict(compare_outputs(cpu.arrays[name], npu.arrays[name]))
            numerical = all(item["consistent"] for item in comparisons.values())
            gates = {
                "source_report_migrated_hash_binding": True,
                "strict_migration_report": all(report_gates.values()),
                "segment_ast_equivalent": True,
                "suffix_host_escape_free": True,
                **_worker_gates(
                    cpu,
                    npu,
                    core=selected_core,
                    warmup=warmup,
                    samples=samples,
                ),
                "numerically_consistent": numerical,
            }
            cpu_ns = _timing_samples(cpu.manifest)
            npu_ns = _timing_samples(npu.manifest)
            cpu_median = float(statistics.median(cpu_ns))
            npu_median = float(statistics.median(npu_ns))
            cpu_component_sum = _warm_runtime_component_sum_ns(
                cpu.manifest, cpu_median
            )
            npu_component_sum = _warm_runtime_component_sum_ns(
                npu.manifest, npu_median
            )
            diagnostic_ratio = cpu_median / npu_median if npu_median > 0 else None
            valid = all(gates.values()) and diagnostic_ratio is not None
            issues = [name for name, passed in gates.items() if passed is not True]
            conservative = (
                _percentile(cpu_ns, 0.10) / _percentile(npu_ns, 0.90)
                if valid
                else None
            )
            ci = (
                _bootstrap_speedup_ci(cpu_ns, npu_ns, inputs.source_sha256)
                if valid
                else (None, None)
            )
            payload: Dict[str, Any] = {
                "schema_version": 1,
                "receipt_type": "numpilot_ordinary_script_benchmark",
                "producer": "numpilot-benchmark-script-v1",
                "status": "passed" if valid else "failed",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "framework_identity": {
                    "numpilot_version": __version__,
                    "runner_sha256": _sha256(_read_regular(Path(__file__), _MAX_SCRIPT_BYTES)),
                    "worker_sha256": worker_sha256,
                    **_git_identity(),
                },
                "binding": {
                    "source": {"path": str(source), "sha256": inputs.source_sha256},
                    "migration_report": {
                        "path": str(report_path),
                        "sha256": inputs.report_sha256,
                    },
                    "migrated": {
                        "path": str(inputs.migrated_path),
                        "sha256": inputs.migrated_sha256,
                    },
                    **segment,
                },
                "model_evidence": {
                    "actual_remote_call": inputs.report.get("actual_remote_call"),
                    "configured_model": inputs.report.get("configured_model"),
                    "requested_model": inputs.report.get("requested_model"),
                    "response_model": inputs.report.get("response_model"),
                    "llm_receipt": inputs.report.get("llm_receipt"),
                },
                "protocol": {
                    "scope": "steady_device_resident_marker_suffix",
                    "core": selected_core,
                    "cpu_threads_requested": 1,
                    "warmup": warmup,
                    "samples": samples,
                    "clock": "time.perf_counter_ns",
                    "npu_sync": "before_and_after_each_sample",
                    "headline_includes_input_preparation": False,
                    "headline_includes_final_d2h": False,
                    "cold_start_included": False,
                    "component_timings_are_reported": True,
                },
                "workers": workers,
                "comparisons": comparisons,
                "gates": gates,
                "performance": {
                    "performance_valid": valid,
                    "issues": issues,
                    "cpu_median_ms": cpu_median / 1e6,
                    "npu_median_ms": npu_median / 1e6,
                    "diagnostic_cpu_over_npu": diagnostic_ratio,
                    "median_speedup": diagnostic_ratio if valid else None,
                    "conservative_cpu_p10_over_npu_p90": conservative,
                    "median_speedup_bootstrap_95ci": (
                        [ci[0], ci[1]] if valid else None
                    ),
                    "warm_runtime_single_pass_component_sum": {
                        "directly_sampled_end_to_end": False,
                        "formula": (
                            "prefix_input_preparation + resident_median + "
                            "final_output_transfer"
                        ),
                        "cpu_ms": cpu_component_sum / 1e6,
                        "npu_ms": npu_component_sum / 1e6,
                        "diagnostic_cpu_over_npu": (
                            cpu_component_sum / npu_component_sum
                            if npu_component_sum > 0 else None
                        ),
                    },
                },
                "evidence_boundary": {
                    "steady_state": True,
                    "device_resident": True,
                    "kernel_only": False,
                    "end_to_end": False,
                    "os_sandbox": False,
                    "explicit_named_outputs": list(names),
                    "input_preparation_and_final_copy_available_in_worker_timings": True,
                },
                "failure": None if valid else {
                    "code": "performance_gate_failed",
                    "phase": "performance_gates",
                    "issues": issues,
                },
            }
            receipt = _publish(receipt_path, payload)
            if not valid:
                raise ScriptBenchmarkError(
                    "performance_gate_failed",
                    phase="performance_gates",
                    receipt=receipt,
                )
            return receipt
    except ScriptBenchmarkError as exc:
        if exc.receipt is not None or receipt_path.exists():
            raise
        failure = {
            "schema_version": 1,
            "receipt_type": "numpilot_ordinary_script_benchmark",
            "producer": "numpilot-benchmark-script-v1",
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "binding": {
                "source": {"path": str(source), "sha256": inputs.source_sha256},
                "migration_report": {
                    "path": str(report_path),
                    "sha256": inputs.report_sha256,
                },
                "migrated": {
                    "path": str(inputs.migrated_path),
                    "sha256": inputs.migrated_sha256,
                },
                **segment,
            },
            "workers": workers,
            "comparisons": comparisons,
            "gates": {
                "source_report_migrated_hash_binding": True,
                "strict_migration_report": True,
                "segment_ast_equivalent": True,
                "suffix_host_escape_free": True,
            },
            "performance": {
                "performance_valid": False,
                "issues": [exc.code],
                "median_speedup": None,
                "conservative_cpu_p10_over_npu_p90": None,
            },
            "evidence_boundary": {
                "steady_state": True,
                "device_resident": True,
                "kernel_only": False,
                "end_to_end": False,
                "os_sandbox": False,
            },
            "failure": {
                "code": exc.code,
                "phase": exc.phase,
                "exception_type": exc.exception_type,
            },
        }
        receipt = _publish(receipt_path, failure)
        exc.receipt = receipt
        raise
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        error = ScriptBenchmarkError(
            "internal_error",
            phase="benchmark_execution",
            exception_type=type(exc).__name__,
        )
        if not receipt_path.exists():
            failure = {
                "schema_version": 1,
                "receipt_type": "numpilot_ordinary_script_benchmark",
                "producer": "numpilot-benchmark-script-v1",
                "status": "failed",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "binding": {
                    "source": {"path": str(source), "sha256": inputs.source_sha256},
                    "migration_report": {
                        "path": str(report_path),
                        "sha256": inputs.report_sha256,
                    },
                    "migrated": {
                        "path": str(inputs.migrated_path),
                        "sha256": inputs.migrated_sha256,
                    },
                    **segment,
                },
                "workers": workers,
                "comparisons": comparisons,
                "performance": {"performance_valid": False, "median_speedup": None},
                "failure": {
                    "code": error.code,
                    "phase": error.phase,
                    "exception_type": error.exception_type,
                },
            }
            error.receipt = _publish(receipt_path, failure)
        raise error from exc
