#!/usr/bin/env python3
"""Generate fail-closed, repeated CPU/NPU evidence for the Poisson Ritz case.

The public command launches CPU and NPU workers in separate processes.  Both
workers start every timing sample from the same deterministic parameters.  NPU
inputs are uploaded once, timing is bounded by an explicit device synchronize,
and the final convergence state is downloaded only after timing.

A passing profiler audit for the same scale is required before this tool will
publish a CPU/NPU ratio.  Without it, timing samples are retained as diagnostic
data and ``performance.published`` remains false.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig
from numpilot.comparison import compare_outputs
from numpilot.engines.ast_rewrite import audit_forward_surface
from numpilot.module_injection import temporary_module, traced_module
from tools.poisson_iteration_evidence import (
    PARAMETER_SLICE,
    _comparison_receipt,
    _load_namespace,
    _normalise_inputs,
    _profile_summary,
    _sha256,
)


_OUTPUT_NAME = "convergence-output.npz"
_WORKER_RECEIPT = "worker.json"


def _build_scaled_inputs(interior_size: int, boundary_size: int, centre_size: int) -> list:
    """Build the iteration contract at a requested deterministic scale."""

    for name, value, lower, upper in (
        ("interior_size", interior_size, 2, 256),
        ("boundary_size", boundary_size, 2, 256),
        ("centre_size", centre_size, 2, 32),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")

    rng = np.random.default_rng(42)
    interior_axis = np.linspace(0.04, 0.96, interior_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(interior_axis, interior_axis)
    x_in = np.ascontiguousarray(grid_x.ravel(), dtype=np.float32)
    y_in = np.ascontiguousarray(grid_y.ravel(), dtype=np.float32)

    boundary_axis = np.linspace(0.0, 1.0, boundary_size, dtype=np.float32)
    x_bd = np.ascontiguousarray(np.concatenate((
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
        boundary_axis,
        boundary_axis,
    )), dtype=np.float32)
    y_bd = np.ascontiguousarray(np.concatenate((
        boundary_axis,
        boundary_axis,
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
    )), dtype=np.float32)

    centre_axis = np.linspace(0.15, 0.85, centre_size, dtype=np.float32)
    centre_x, centre_y = np.meshgrid(centre_axis, centre_axis)
    cx = np.ascontiguousarray(centre_x.ravel(), dtype=np.float32)
    cy = np.ascontiguousarray(centre_y.ravel(), dtype=np.float32)
    count_rbf = cx.size
    width = np.full(count_rbf, np.float32(0.18), dtype=np.float32)
    weight = rng.normal(0.0, 0.05, count_rbf).astype(np.float32)

    source = np.multiply(
        np.float32(2.0 * np.pi * np.pi),
        np.multiply(
            np.sin(np.multiply(np.float32(np.pi), x_in)),
            np.sin(np.multiply(np.float32(np.pi), y_in)),
        ),
    ).astype(np.float32)
    boundary_target = np.zeros_like(x_bd)
    source[[min(5, source.size - 1), min(37, source.size - 1)]] += np.array(
        [10.0, -10.0], dtype=np.float32
    )
    boundary_target[
        [min(3, boundary_target.size - 1), min(22, boundary_target.size - 1)]
    ] += np.array([-10.0, 10.0], dtype=np.float32)

    count_in = x_in.size
    count_bd = x_bd.size
    ones_rbf = np.ones(count_rbf, dtype=np.float32)
    ones_in = np.ones(count_in, dtype=np.float32)
    ones_bd = np.ones(count_bd, dtype=np.float32)
    return _normalise_inputs((
        x_in,
        y_in,
        np.ascontiguousarray(source, dtype=np.float32),
        x_bd,
        y_bd,
        np.ascontiguousarray(boundary_target, dtype=np.float32),
        weight,
        cx,
        cy,
        width,
        ones_rbf,
        ones_in,
        ones_bd,
        np.full(count_in, np.float32(1.0 / count_in), dtype=np.float32),
        np.full(count_bd, np.float32(1.0 / count_bd), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(-0.5), dtype=np.float32),
        np.full((count_bd, count_rbf), np.float32(-0.5), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(2.0), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(-4.0), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(4.0), dtype=np.float32),
        np.full(count_in, np.float32(-1.0), dtype=np.float32),
        np.full(count_in, np.float32(1.0), dtype=np.float32),
        np.full(count_bd, np.float32(-1.0), dtype=np.float32),
        np.full(count_bd, np.float32(1.0), dtype=np.float32),
        np.array([np.float32(5.0)], dtype=np.float32),
        np.array([np.float32(3e-3)], dtype=np.float32),
        np.array([np.float32(0.0)], dtype=np.float32),
        np.array([np.float32(0.0)], dtype=np.float32),
        np.full(count_rbf, np.float32(0.07), dtype=np.float32),
        np.full(count_rbf, np.float32(0.30), dtype=np.float32),
    ))


def _advance_capture(forward, canonical_inputs: list, iterations: int):
    """Run from the immutable initial state and retain first/final outputs."""

    inputs = list(canonical_inputs)
    return _advance_state(forward, inputs, iterations)


def _advance_state(forward, inputs: list, iterations: int):
    """Advance a prepared state; callers control whether reset is timed."""

    first = None
    last = None
    for _ in range(iterations):
        last = forward(*inputs)
        if not isinstance(last, tuple) or len(last) != 5:
            raise TypeError("forward() must return loss plus four parameter tensors")
        if first is None:
            first = last
        inputs[PARAMETER_SLICE] = last[1:5]
    return first, last


def _timing_samples(
    forward,
    canonical_inputs: list,
    *,
    iterations: int,
    warmup: int,
    repeats: int,
    synchronize=None,
    completion_scope: str = "completion",
) -> dict:
    """Keep ordered warmup and measured samples for an identical work unit."""

    def one_sample() -> float:
        inputs = list(canonical_inputs)
        start = time.perf_counter()
        _advance_state(forward, inputs, iterations)
        if synchronize is not None:
            synchronize()
        return (time.perf_counter() - start) * 1e3

    warmup_samples = [one_sample() for _ in range(warmup)]
    samples = [one_sample() for _ in range(repeats)]
    return {
        "clock": "time.perf_counter",
        "scope": (
            "resident inputs; forward iterations; " + completion_scope
            + "; excludes state reset and transfers"
        ),
        "iterations_per_sample": iterations,
        "warmup_samples_ms": warmup_samples,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "median_per_iteration_ms": statistics.median(samples) / iterations,
    }


class _AclDeviceSynchronizer:
    def __init__(self):
        self._library = ctypes.CDLL("libascendcl.so")
        self._function = self._library.aclrtSynchronizeDevice
        self._function.argtypes = []
        self._function.restype = ctypes.c_int

    @property
    def identity(self) -> str:
        return str(self._library._name)

    def __call__(self) -> None:
        result = int(self._function())
        if result != 0:
            raise RuntimeError(f"aclrtSynchronizeDevice failed with ACL error {result}")


def _save_output(path: Path, output) -> None:
    arrays = {f"output_{index}": np.asarray(item) for index, item in enumerate(output)}
    np.savez(path, **arrays)


def _load_output(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[f"output_{index}"] for index in range(5))


def _host_output(backend: NpuBackend, output, transfers: dict):
    values = []
    for item in output:
        if not hasattr(item, "to_numpy"):
            raise TypeError("strict-NPU output is not a device ndarray")
        host = backend.to_host(item)
        transfers["d2h_count"] += 1
        transfers["d2h_bytes"] += int(host.nbytes)
        values.append(host)
    return tuple(values)


def _host_loss(backend: NpuBackend, output, transfers: dict) -> float:
    item = output[0]
    if not hasattr(item, "to_numpy"):
        raise TypeError("strict-NPU loss is not a device ndarray")
    host = backend.to_host(item)
    transfers["d2h_count"] += 1
    transfers["d2h_bytes"] += int(host.nbytes)
    return _loss((host,))


def _loss(output) -> float:
    return float(np.asarray(output[0]).reshape(-1)[0])


def _threadpool_info() -> list:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        observed = []
    else:
        observed = [
            {
                "source": "threadpoolctl",
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "filepath": item.get("filepath"),
                "num_threads": item.get("num_threads"),
                "version": item.get("version"),
            }
            for item in threadpool_info()
        ]
    if observed:
        return observed

    # The isolated P0 venv intentionally has no optional threadpoolctl wheel.
    # OpenBLAS exposes an equivalent read-only runtime query.  Discover the
    # already-loaded library through procfs and record the exact queried file.
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    libraries = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and "openblas" in fields[5].lower():
            libraries.add(fields[5])
    for library_path in sorted(libraries):
        try:
            library = ctypes.CDLL(library_path)
        except OSError:
            continue
        getter = None
        for symbol in (
            "openblas_get_num_threads64_",
            "openblas_get_num_threads_64_",
            "openblas_get_num_threads",
        ):
            try:
                getter = getattr(library, symbol)
            except AttributeError:
                continue
            break
        if getter is None:
            continue
        getter.argtypes = []
        getter.restype = ctypes.c_int
        config = None
        for symbol in ("openblas_get_config64_", "openblas_get_config"):
            try:
                config_getter = getattr(library, symbol)
            except AttributeError:
                continue
            config_getter.argtypes = []
            config_getter.restype = ctypes.c_char_p
            raw_config = config_getter()
            config = None if raw_config is None else raw_config.decode("utf-8", "replace")
            break
        observed.append({
            "source": "openblas_runtime_api",
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "openblas",
            "filepath": library_path,
            "num_threads": int(getter()),
            "version": config,
        })
    return observed


def _worker_common(args) -> tuple[Path, str, dict, list]:
    source_path = Path(args.source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source is missing: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    namespace = _load_namespace(source, str(source_path))
    canonical_inputs = _build_scaled_inputs(
        args.interior_size, args.boundary_size, args.centre_size
    )
    return source_path, source, namespace, canonical_inputs


def _cpu_worker(args) -> dict:
    source_path, _, namespace, canonical_inputs = _worker_common(args)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limits = nullcontext()
    else:
        limits = threadpool_limits(limits=args.cpu_threads)

    with limits:
        timing = _timing_samples(
            namespace["forward"],
            canonical_inputs,
            iterations=args.sample_iterations,
            warmup=args.warmup,
            repeats=args.repeats,
            completion_scope="synchronous NumPy completion",
        )
        first, final = _advance_capture(
            namespace["forward"], canonical_inputs, args.convergence_iterations
        )
        pool_info = _threadpool_info()

    output_path = Path(args.worker_out) / _OUTPUT_NAME
    _save_output(output_path, final)
    return {
        "schema_version": "1.0",
        "producer": "numpilot-poisson-performance-cpu-worker-v1",
        "status": "pass",
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": _sha256(source_path),
        "scale": _scale_receipt(args),
        "cpu_threads_requested": args.cpu_threads,
        "thread_limit_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "threadpools": pool_info,
        "timing": timing,
        "convergence": {
            "iterations": args.convergence_iterations,
            "initial_loss": _loss(first),
            "final_loss": _loss(final),
            "finite": bool(math.isfinite(_loss(first)) and math.isfinite(_loss(final))),
            "decreased_from_initial": bool(_loss(final) <= _loss(first)),
        },
        "output_sha256": _sha256(output_path),
    }


def _validated_npu_surface(args, source_path: Path, source: str):
    migrated_path = Path(args.migrated).resolve()
    report_path = Path(args.migration_report).resolve()
    if not migrated_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("migrated source or migration report is missing")
    migrated = migrated_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError("migration report is not passing")
    analysis = AnalysisAgent().analyze(source, str(source_path))
    if not analysis.migratable:
        raise RuntimeError("source is no longer migratable")
    expected = MigrationAgent().migrate(source, DecisionAgent().plan(analysis))
    if expected.migrated_source != migrated:
        raise RuntimeError("migrated artifact drifted from deterministic AST output")
    surface = audit_forward_surface(migrated)
    if surface.violations:
        raise RuntimeError("strict migrated forward rejected: " + "; ".join(surface.violations))
    if dict(report.get("rewrites", {})) != surface.ap_calls:
        raise RuntimeError("migration report callsites do not match migrated source")
    return migrated_path, report_path, migrated, surface


def _npu_worker(args) -> dict:
    source_path, source, _, canonical_inputs = _worker_common(args)
    migrated_path, report_path, migrated, surface = _validated_npu_surface(
        args, source_path, source
    )
    overlay_path = Path(args.capability_overlay).resolve()
    evidence_root = Path(args.qualification_evidence_root).resolve()
    if not overlay_path.is_file() or not evidence_root.is_dir():
        raise FileNotFoundError("capability overlay or qualification evidence is missing")

    backend = NpuBackend(overlay_config=OverlayRuntimeConfig(
        overlay_path=str(overlay_path), evidence_root=str(evidence_root)
    ))
    guard = backend.prepare_profile_guard(surface.ap_calls)
    execution_module, runtime_counts = traced_module(
        backend.get_module(),
        surface.ap_calls,
        before_call=guard.validate_call,
        targets=guard.runtime_targets,
        enforce_target_identity=True,
    )
    transfers = {
        "h2d_count": 0,
        "h2d_bytes": 0,
        "d2h_count": 0,
        "d2h_bytes": 0,
        "h2d_phase": "before_timing",
        "d2h_phase": "after_timing_and_convergence",
    }
    device_inputs = []
    for item in canonical_inputs:
        device_inputs.append(backend.to_device(item.copy()))
        transfers["h2d_count"] += 1
        transfers["h2d_bytes"] += int(item.nbytes)

    synchronizer = _AclDeviceSynchronizer()
    with temporary_module("asnumpy", execution_module):
        namespace = _load_namespace(migrated, str(migrated_path))
        timing = _timing_samples(
            namespace["forward"],
            device_inputs,
            iterations=args.sample_iterations,
            warmup=args.warmup,
            repeats=args.repeats,
            synchronize=synchronizer,
            completion_scope="terminal aclrtSynchronizeDevice",
        )
        first, final = _advance_capture(
            namespace["forward"], device_inputs, args.convergence_iterations
        )
        synchronizer()
    initial_loss = _host_loss(backend, first, transfers)
    host_final = _host_output(backend, final, transfers)

    total_iterations = (
        args.sample_iterations * (args.warmup + args.repeats)
        + args.convergence_iterations
    )
    expected_counts = {
        name: count * total_iterations for name, count in surface.ap_calls.items()
    }
    if runtime_counts != expected_counts:
        raise RuntimeError(
            f"runtime call counts differ: expected={expected_counts}, observed={runtime_counts}"
        )
    if not guard.ok or not guard.runtime_profile_ok:
        raise RuntimeError("one or more runtime calls escaped the admitted profile")

    output_path = Path(args.worker_out) / _OUTPUT_NAME
    _save_output(output_path, host_final)
    provenance = guard.to_dict()
    final_loss = _loss(host_final)
    return {
        "schema_version": "1.0",
        "producer": "numpilot-poisson-performance-npu-worker-v1",
        "status": "pass",
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": _sha256(source_path),
        "migrated_sha256": _sha256(migrated_path),
        "migration_report_sha256": _sha256(report_path),
        "overlay_sha256": _sha256(overlay_path),
        "scale": _scale_receipt(args),
        "timing": timing,
        "synchronize_api": "aclrtSynchronizeDevice",
        "synchronize_library": synchronizer.identity,
        "convergence": {
            "iterations": args.convergence_iterations,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "finite": bool(math.isfinite(initial_loss) and math.isfinite(final_loss)),
            "decreased_from_initial": bool(final_loss <= initial_loss),
        },
        "capability_audit_id": provenance["audit_id"],
        "strict_npu": provenance["strict_npu"],
        "runtime_profile_ok": provenance["runtime_profile_ok"],
        "runtime_target_integrity_ok": provenance["runtime_target_integrity_ok"],
        "per_iteration_calls": dict(sorted(surface.ap_calls.items())),
        "runtime_call_counts": dict(sorted(runtime_counts.items())),
        "total_runtime_calls": sum(runtime_counts.values()),
        "profile_checks": _profile_summary(guard),
        "transfers": transfers,
        "output_sha256": _sha256(output_path),
    }


def _scale_receipt(args) -> dict:
    return {
        "interior_axis": args.interior_size,
        "interior_points": args.interior_size ** 2,
        "boundary_axis": args.boundary_size,
        "boundary_points": args.boundary_size * 4,
        "centre_axis": args.centre_size,
        "rbf_count": args.centre_size ** 2,
    }


def _profile_identity(receipt: dict) -> dict:
    return {
        "source_sha256": receipt["source_sha256"],
        "migrated_sha256": receipt["migrated_sha256"],
        "overlay_sha256": receipt["overlay_sha256"],
        "capability_audit_id": receipt["capability_audit_id"],
        "runner_sha256": receipt["runner_sha256"],
        "scale": receipt["scale"],
    }


def _load_profiler_audit(path: str | None, npu: dict) -> tuple[dict | None, list[str]]:
    if not path:
        return None, ["missing passing profiler audit for this scale"]
    audit_path = Path(path).resolve()
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, [f"profiler audit cannot be read: {type(error).__name__}: {error}"]
    issues = []
    if audit.get("status") != "pass":
        issues.append("profiler audit status is not pass")
    if audit.get("profile_identity") != _profile_identity(npu):
        issues.append("profiler audit identity or scale does not match benchmark")
    return audit, issues


def _combine(args, cpu_dir: Path, npu_dir: Path) -> dict:
    cpu = json.loads((cpu_dir / _WORKER_RECEIPT).read_text(encoding="utf-8"))
    npu = json.loads((npu_dir / _WORKER_RECEIPT).read_text(encoding="utf-8"))
    comparison = compare_outputs(
        _load_output(cpu_dir / _OUTPUT_NAME), _load_output(npu_dir / _OUTPUT_NAME)
    )
    profiler, issues = _load_profiler_audit(args.profiler_audit, npu)
    if cpu.get("status") != "pass" or npu.get("status") != "pass":
        issues.append("one or more benchmark workers failed")
    if cpu.get("pid") == npu.get("pid"):
        issues.append("CPU and NPU workers did not run in separate processes")
    if cpu.get("source_sha256") != npu.get("source_sha256"):
        issues.append("CPU/NPU source identity mismatch")
    if cpu.get("scale") != npu.get("scale"):
        issues.append("CPU/NPU scale mismatch")
    blas_pools = [
        pool for pool in cpu.get("threadpools", [])
        if pool.get("user_api") == "blas"
    ]
    if not blas_pools:
        issues.append("CPU BLAS thread count was not observed at runtime")
    elif any(pool.get("num_threads") != args.cpu_threads for pool in blas_pools):
        issues.append("CPU BLAS runtime thread count differs from the requested limit")
    if not comparison.consistent:
        issues.append("CPU/NPU convergence output comparison failed")
    if not cpu["convergence"]["finite"] or not npu["convergence"]["finite"]:
        issues.append("CPU or NPU convergence loss is non-finite")
    if (
        not cpu["convergence"]["decreased_from_initial"]
        or not npu["convergence"]["decreased_from_initial"]
    ):
        issues.append("CPU or NPU loss did not decrease")
    for key in ("strict_npu", "runtime_profile_ok", "runtime_target_integrity_ok"):
        if npu.get(key) is not True:
            issues.append(f"NPU gate failed: {key}")
    if npu["transfers"].get("h2d_phase") != "before_timing":
        issues.append("H2D was not isolated before timing")
    if npu["transfers"].get("d2h_phase") != "after_timing_and_convergence":
        issues.append("D2H was not isolated after timing")

    published = not issues
    cpu_median = float(cpu["timing"]["median_per_iteration_ms"])
    npu_median = float(npu["timing"]["median_per_iteration_ms"])
    ratio = cpu_median / npu_median if published and npu_median > 0 else None
    return {
        "schema_version": "1.0",
        "producer": "numpilot-poisson-performance-evidence-v1",
        "status": "pass" if cpu.get("status") == npu.get("status") == "pass" else "fail",
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "scale": npu["scale"],
        "workload": {
            "warmup_samples": args.warmup,
            "measurement_samples": args.repeats,
            "iterations_per_timing_sample": args.sample_iterations,
            "convergence_iterations": args.convergence_iterations,
        },
        "cpu": cpu,
        "npu": npu,
        "comparison": _comparison_receipt(comparison),
        "profiler_audit": profiler,
        "performance": {
            "measured": True,
            "published": published,
            "issues": issues,
            "timing_scope": npu["timing"]["scope"],
            "cpu_median_per_iteration_ms": cpu_median,
            "npu_median_per_iteration_ms": npu_median,
            "cpu_time_over_npu_time": ratio,
            "winner": (
                "npu" if ratio is not None and ratio > 1
                else "cpu" if ratio is not None and ratio < 1
                else "tie" if ratio == 1
                else None
            ),
        },
    }


def _render(receipt: dict) -> str:
    performance = receipt.get("performance", {})
    lines = [
        "# Poisson Ritz 放大规模重复性能证据",
        "",
        f"状态：`{receipt['status']}`",
        f"性能发布：`{str(performance.get('published', False)).lower()}`",
        "",
    ]
    if receipt.get("status") != "pass":
        lines.extend([
            f"失败：`{receipt.get('exception_type')}: {receipt.get('exception_message')}`",
            "",
        ])
        return "\n".join(lines)
    scale = receipt["scale"]
    lines.extend([
        f"- 规模：{scale['interior_axis']}×{scale['interior_axis']} 内点，"
        f"{scale['boundary_points']} 边界点，{scale['rbf_count']} 个 RBF",
        f"- CPU 中位每轮：`{performance['cpu_median_per_iteration_ms']:.6f} ms`",
        f"- NPU 中位每轮：`{performance['npu_median_per_iteration_ms']:.6f} ms`",
        f"- CPU/NPU 输出最大绝对误差：`{receipt['comparison']['max_abs_diff']}`",
        f"- 收敛轮数：`{receipt['workload']['convergence_iterations']}`",
    ])
    if performance.get("published"):
        ratio = performance["cpu_time_over_npu_time"]
        if ratio > 1:
            lines.append(f"- 合格结论：NPU 为 CPU 的 `{ratio:.6f}×`")
        elif ratio < 1:
            lines.append(f"- 合格结论：CPU 约快 `{1.0 / ratio:.6f}×`")
        else:
            lines.append("- 合格结论：CPU 与 NPU 中位时延相同")
    else:
        lines.extend(["", "## 尚未通过的发布门", ""])
        lines.extend(f"- {issue}" for issue in performance.get("issues", []))
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_worker(args) -> int:
    output = Path(args.worker_out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = _cpu_worker(args) if args.worker == "cpu" else _npu_worker(args)
    except Exception as error:
        receipt = {
            "schema_version": "1.0",
            "producer": f"numpilot-poisson-performance-{args.worker}-worker-v1",
            "status": "fail",
            "pid": os.getpid(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
    _write_json(output / _WORKER_RECEIPT, receipt)
    return 0 if receipt["status"] == "pass" else 2


def _worker_command(args, worker: str, worker_out: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker", worker,
        "--worker-out", str(worker_out),
        "--source", args.source,
        "--migrated", args.migrated,
        "--migration-report", args.migration_report,
        "--capability-overlay", args.capability_overlay,
        "--qualification-evidence-root", args.qualification_evidence_root,
        "--interior-size", str(args.interior_size),
        "--boundary-size", str(args.boundary_size),
        "--centre-size", str(args.centre_size),
        "--sample-iterations", str(args.sample_iterations),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--convergence-iterations", str(args.convergence_iterations),
        "--cpu-threads", str(args.cpu_threads),
    ]
    return command


def _run_parent(args) -> int:
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError(f"--out must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".poisson-performance-", dir=output.parent))
    try:
        worker_results = {}
        for worker in ("cpu", "npu"):
            worker_out = staging / worker
            environment = os.environ.copy()
            if worker == "cpu":
                for name in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    environment[name] = str(args.cpu_threads)
            completed = subprocess.run(
                _worker_command(args, worker, worker_out),
                cwd=str(Path.cwd()),
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (staging / f"{worker}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (staging / f"{worker}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            worker_results[worker] = completed.returncode
            if not (worker_out / _WORKER_RECEIPT).is_file():
                raise RuntimeError(f"{worker} worker did not produce a receipt")

        if any(code != 0 for code in worker_results.values()):
            failures = {
                worker: json.loads(
                    (staging / worker / _WORKER_RECEIPT).read_text(encoding="utf-8")
                )
                for worker in worker_results
            }
            raise RuntimeError("worker failure: " + json.dumps(failures, ensure_ascii=False))
        receipt = _combine(args, staging / "cpu", staging / "npu")
    except Exception as error:
        receipt = {
            "schema_version": "1.0",
            "producer": "numpilot-poisson-performance-evidence-v1",
            "status": "fail",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "performance": {"measured": False, "published": False},
        }
    _write_json(staging / "evidence.json", receipt)
    (staging / "README.md").write_text(_render(receipt), encoding="utf-8")
    staging.rename(output)
    print(
        f"status={receipt['status']} "
        f"performance_published={str(receipt['performance']['published']).lower()} "
        f"out={output}"
    )
    return 0 if receipt["status"] == "pass" else 2


def _positive(parser, name: str, value: int, *, minimum: int = 1) -> None:
    if value < minimum:
        parser.error(f"--{name} must be >= {minimum}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--migrated", required=True)
    parser.add_argument("--migration-report", required=True)
    parser.add_argument("--capability-overlay", required=True)
    parser.add_argument("--qualification-evidence-root", required=True)
    parser.add_argument("--interior-size", type=int, default=48)
    parser.add_argument("--boundary-size", type=int, default=48)
    parser.add_argument("--centre-size", type=int, default=7)
    parser.add_argument("--sample-iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--convergence-iterations", type=int, default=250)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--profiler-audit")
    parser.add_argument("--out")
    parser.add_argument("--worker", choices=("cpu", "npu"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name, value, minimum in (
        ("sample-iterations", args.sample_iterations, 1),
        ("warmup", args.warmup, 0),
        ("repeats", args.repeats, 3),
        ("convergence-iterations", args.convergence_iterations, 2),
        ("cpu-threads", args.cpu_threads, 1),
    ):
        _positive(parser, name, value, minimum=minimum)
    try:
        _build_scaled_inputs(args.interior_size, args.boundary_size, args.centre_size)
    except ValueError as error:
        parser.error(str(error))
    if args.worker:
        if not args.worker_out:
            parser.error("--worker-out is required for an internal worker")
        return _run_worker(args)
    if not args.out:
        parser.error("--out is required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
