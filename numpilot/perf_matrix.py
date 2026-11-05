"""Performance-matrix primitives for the Phase 9 acceptance run.

The matrix compares a single logical CPU core with one Ascend NPU over the same
float32 workload.  Three timing scopes are kept separate:

``resident``
    inputs uploaded once; repeated compute followed by a device synchronize,
    with no per-iteration device-to-host copy;
``warm end-to-end``
    a warm process where every sample performs host-to-device, compute and
    device-to-host;
``cold end-to-end``
    a fresh worker process per sample, including interpreter startup, runtime
    import, input construction, transfers and output download.

Measurements are always retained, but ``performance_valid`` is true only when
every executed API carries ``NATIVE_NPU`` provenance and the correctness gate
passes.  An unaudited runtime therefore produces ``profile_required`` evidence
instead of a publishable speedup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from numpilot.comparison import compare_outputs

MATRIX_SCHEMA = "numpilot.performance-matrix"
WORKER_SCHEMA = "numpilot.performance-worker"
MATRIX_PRODUCER = "numpilot-performance-matrix-v1"
WORKER_PRODUCER = "numpilot-performance-worker-v1"

PROFILES = ("matmul", "sin_matmul")
DEFAULT_SIZES = (64, 128, 256, 512, 1024)
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 20
DEFAULT_NPU_WARMUP = 3
DEFAULT_NPU_REPEATS = 15
DEFAULT_COLD_SAMPLES = 7
MIN_BOOTSTRAP_SAMPLES = 7
BOOTSTRAP_ITERATIONS = 2000

POLICY = {
    "resident_median_ratio": 1.5,
    "warm_end_to_end_median_ratio": 1.2,
    "conservative_p10_over_p90": 1.1,
    "cold_end_to_end_median_ratio": 1.0,
    "publish_requires_strict_provenance": True,
}

# Elementwise float32 tolerances are appropriate for scalar/elementwise APIs but
# not for two independent GEMM accumulations: near-zero outputs amplify the
# rounding difference.  GEMM profiles therefore use a float64 oracle, a
# relative-Frobenius bound, a size-scaled absolute bound, and an additional
# requirement that the NPU error is not worse than the CPU float32 baseline.
GEMM_POLICY = {
    "oracle": "float64",
    "relative_frobenius_max": 1e-5,
    "max_abs_atol_floor": 1e-5,
    "max_abs_error_factor": 2.0,
    "npu_vs_cpu_error_factor": 2.0,
    "npu_vs_cpu_error_floor": 1e-6,
}

_FLOAT32_EPS = float(np.finfo(np.float32).eps)

_RATIO_THRESHOLDS = {
    "resident": POLICY["resident_median_ratio"],
    "warm_end_to_end": POLICY["warm_end_to_end_median_ratio"],
    "cold_end_to_end": POLICY["cold_end_to_end_median_ratio"],
}

_PROFILE_OPS = {
    "matmul": ("matmul",),
    "sin_matmul": ("sin", "matmul"),
}

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_TIMING_KEYS = (
    "warmup_count",
    "warmup_samples_ms",
    "repeats",
    "samples_ms",
    "median_ms",
    "p10_ms",
    "p90_ms",
    "mean_ms",
    "stdev_ms",
    "min_ms",
    "max_ms",
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_seed(profile: str, size: int) -> int:
    digest = hashlib.sha256(f"{profile}:{size}".encode("ascii")).hexdigest()
    return int(digest[:8], 16)


def validate_profile(profile: str) -> str:
    if profile not in _PROFILE_OPS:
        raise ValueError(f"unknown profile: {profile!r}")
    return profile


def validate_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if not 2 <= size <= 4096:
        raise ValueError("size must be between 2 and 4096")
    return size


def build_inputs(profile: str, size: int, seed: Optional[int] = None) -> dict:
    """Build the deterministic host inputs shared by every worker."""

    validate_profile(profile)
    validate_size(size)
    base = default_seed(profile, size) if seed is None else int(seed)
    rng = np.random.default_rng(base)
    return {
        "a": np.ascontiguousarray(
            rng.standard_normal((size, size), dtype=np.float32)
        ),
        "b": np.ascontiguousarray(
            rng.standard_normal((size, size), dtype=np.float32)
        ),
    }


def numpy_forward(profile: str, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    a = np.asarray(inputs["a"])
    b = np.asarray(inputs["b"])
    if profile == "matmul":
        return np.matmul(a, b)
    if profile == "sin_matmul":
        return np.matmul(np.sin(a), b)
    raise ValueError(f"unknown profile: {profile!r}")


def float64_forward(profile: str, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the float64 oracle used by the GEMM correctness gate."""

    a = np.asarray(inputs["a"], dtype=np.float64)
    b = np.asarray(inputs["b"], dtype=np.float64)
    if profile == "matmul":
        return np.matmul(a, b)
    if profile == "sin_matmul":
        return np.matmul(np.sin(a), b)
    raise ValueError(f"unknown profile: {profile!r}")


def gemm_max_abs_limit(profile: str, inputs: Mapping[str, np.ndarray]) -> float:
    """Size-scaled absolute error bound for a float32 GEMM accumulation."""

    size = int(inputs["a"].shape[0])
    a = np.asarray(inputs["a"], dtype=np.float64)
    if profile == "sin_matmul":
        a = np.sin(a)
    b = np.asarray(inputs["b"], dtype=np.float64)
    scale = float(np.max(np.abs(a))) * float(np.max(np.abs(b)))
    return max(
        GEMM_POLICY["max_abs_atol_floor"],
        GEMM_POLICY["max_abs_error_factor"] * _FLOAT32_EPS * size * scale,
    )


def gemm_error(reference64: np.ndarray, actual: np.ndarray) -> dict:
    reference = np.asarray(reference64, dtype=np.float64)
    actual = np.asarray(actual)
    result = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "reference_shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "finite": bool(
            np.isfinite(actual).all() and np.isfinite(reference).all()
        ),
        "relative_frobenius": None,
        "max_abs_diff": None,
        "reference_norm": None,
    }
    if actual.shape != reference.shape or actual.dtype != np.float32:
        result["reason"] = "shape or dtype mismatch"
        return result
    difference = actual.astype(np.float64) - reference
    reference_norm = float(np.linalg.norm(reference))
    difference_norm = float(np.linalg.norm(difference))
    result["reference_norm"] = reference_norm
    result["relative_frobenius"] = difference_norm / max(
        reference_norm, np.finfo(np.float64).tiny
    )
    result["max_abs_diff"] = (
        float(np.max(np.abs(difference))) if difference.size else 0.0
    )
    return result


def gemm_check(
    reference64: np.ndarray,
    path: Path,
    *,
    max_abs_limit: float,
    cpu_relative_frobenius: Optional[float] = None,
) -> dict:
    actual = np.load(path, allow_pickle=False)
    error = gemm_error(reference64, actual)
    comparative_limit = None
    consistent = bool(error["finite"])
    if error["relative_frobenius"] is None:
        consistent = False
    else:
        consistent = consistent and (
            error["relative_frobenius"] <= GEMM_POLICY["relative_frobenius_max"]
        )
        consistent = consistent and (
            error["max_abs_diff"] <= max_abs_limit
        )
        if cpu_relative_frobenius is not None:
            comparative_limit = max(
                GEMM_POLICY["npu_vs_cpu_error_factor"]
                * cpu_relative_frobenius,
                GEMM_POLICY["npu_vs_cpu_error_floor"],
            )
            consistent = consistent and (
                error["relative_frobenius"] <= comparative_limit
            )
    reason = None
    if not consistent:
        reason = error.get("reason") or "GEMM output exceeds the numerical gate"
    return {
        "path": path.name,
        **error,
        "relative_limit": GEMM_POLICY["relative_frobenius_max"],
        "max_abs_limit": max_abs_limit,
        "cpu_relative_frobenius": cpu_relative_frobenius,
        "comparative_limit": comparative_limit,
        "consistent": consistent,
        "reason": reason,
    }


def module_forward(module, profile: str, a, b):
    if profile == "matmul":
        return module.matmul(a, b)
    if profile == "sin_matmul":
        return module.matmul(module.sin(a), b)
    raise ValueError(f"unknown profile: {profile!r}")


def expected_calls(profile: str) -> dict:
    if profile == "matmul":
        return {"matmul": 1}
    if profile == "sin_matmul":
        return {"sin": 1, "matmul": 1}
    raise ValueError(f"unknown profile: {profile!r}")


def required_ops(profile: str) -> tuple:
    return _PROFILE_OPS[validate_profile(profile)]


def summarize_samples(
    samples_ms: Sequence[float],
    *,
    warmup_count: int = 0,
    warmup_samples_ms: Sequence[float] = (),
) -> dict:
    """Return median, deciles, dispersion and the ordered raw samples."""

    samples = [float(value) for value in samples_ms]
    if not samples:
        raise ValueError("samples_ms must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("samples must be finite and non-negative")
    ordered = sorted(samples)
    p10 = _percentile(ordered, 0.10)
    p90 = _percentile(ordered, 0.90)
    return {
        "clock": "time.perf_counter",
        "unit": "ms",
        "warmup_count": int(warmup_count),
        "warmup_samples_ms": [float(value) for value in warmup_samples_ms],
        "repeats": len(samples),
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": p10,
        "p90_ms": p90,
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("ordered samples must not be empty")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_ratio_ci(
    cpu_samples_ms: Sequence[float],
    npu_samples_ms: Sequence[float],
    *,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> Optional[list]:
    """Deterministic bootstrap 95% interval for the median CPU/NPU ratio."""

    cpu = [float(value) for value in cpu_samples_ms]
    npu = [float(value) for value in npu_samples_ms]
    if len(cpu) < MIN_BOOTSTRAP_SAMPLES or len(npu) < MIN_BOOTSTRAP_SAMPLES:
        return None
    rng = random.Random(int(seed))
    ratios = []
    for _ in range(iterations):
        cpu_sample = [rng.choice(cpu) for _ in cpu]
        npu_sample = [rng.choice(npu) for _ in npu]
        denominator = statistics.median(npu_sample)
        if denominator > 0:
            ratios.append(statistics.median(cpu_sample) / denominator)
    if not ratios:
        return None
    ratios.sort()
    return [_percentile(ratios, 0.025), _percentile(ratios, 0.975)]


def ratio_summary(
    cpu: Mapping[str, Any],
    npu: Mapping[str, Any],
    *,
    seed: int,
    basis: str,
    publishable: bool = False,
) -> dict:
    median_ratio = cpu["median_ms"] / npu["median_ms"] if npu["median_ms"] > 0 else None
    conservative = (
        cpu["p10_ms"] / npu["p90_ms"] if npu["p90_ms"] > 0 else None
    )
    return {
        "basis": basis,
        "median_ratio": median_ratio,
        "conservative_p10_over_p90": conservative,
        "bootstrap_95ci": bootstrap_ratio_ci(
            cpu["samples_ms"], npu["samples_ms"], seed=seed
        ),
        "publishable": bool(publishable),
    }


def ratio_publishable(ratio: Mapping[str, Any], threshold: float) -> bool:
    """Apply the conservative public-claim gates from the theory document."""

    median = ratio.get("median_ratio")
    conservative = ratio.get("conservative_p10_over_p90")
    interval = ratio.get("bootstrap_95ci")
    if median is None or conservative is None or interval is None:
        return False
    return (
        median >= threshold
        and conservative >= POLICY["conservative_p10_over_p90"]
        and interval[0] > 1.0
    )


def provenance_summary(manifest, profile: str) -> dict:
    """Extract only the provenance facts needed by the matrix receipt."""

    required = required_ops(profile)
    operators = {}
    for name in required:
        capability = manifest.operators.get(name)
        if capability is None:
            operators[name] = {
                "available": False,
                "provenance": "missing",
                "evidence": "operator absent from capability manifest",
            }
            continue
        provenance = getattr(capability.provenance, "value", str(capability.provenance))
        operators[name] = {
            "available": bool(capability.available),
            "provenance": provenance,
            "constraints": list(capability.constraints),
            "evidence": str(capability.evidence),
        }
    native = all(
        item["available"] and item["provenance"] == "native_npu"
        for item in operators.values()
    )
    return {
        "required_operators": list(required),
        "operators": operators,
        "native_npu": native,
        "audit_id": getattr(manifest, "audit_id", None),
    }


def sanitize_identity(identity) -> dict:
    """Drop private absolute paths from a RuntimeIdentity dataclass."""

    fields = (
        "asnumpy_version",
        "module_sha256",
        "native_module_sha256",
        "cann_home",
        "cann_version",
        "soc_model",
        "python_version",
        "audit_id",
        "status",
        "issues",
        "wrapper_sha256",
    )
    result = {}
    for name in fields:
        if hasattr(identity, name):
            value = getattr(identity, name)
            if name == "cann_home":
                # The CANN root is an installation path, not a workspace path;
                # only its basename is retained in public receipts.
                value = os.path.basename(str(value)) if value else value
            result[name] = _json_value(value)
    return result


def _json_value(value):
    """Convert dataclass internals such as mappingproxy into JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return str(value)


def classify_row(
    *,
    correctness_ok: bool,
    provenance_ok: bool,
    ratios: Mapping[str, Optional[Mapping[str, Any]]],
) -> str:
    if not correctness_ok:
        return "FAILED"
    if provenance_ok:
        return "STRICT_NPU_VALIDATED"
    warm = ratios.get("warm_end_to_end") or {}
    resident = ratios.get("resident") or {}
    warm_ratio = warm.get("median_ratio")
    resident_ratio = resident.get("median_ratio")
    if (
        resident_ratio is not None
        and resident_ratio >= POLICY["resident_median_ratio"]
        and (warm_ratio is None or warm_ratio < POLICY["warm_end_to_end_median_ratio"])
    ):
        return "RESIDENT_GAIN"
    if warm_ratio is not None and warm_ratio < 1.0:
        return "CPU_PREFERRED"
    return "DIAGNOSTIC"


def crossover(
    rows: Sequence[Mapping[str, Any]],
    ratio_key: str,
    *,
    threshold: float = 1.0,
) -> dict:
    ordered = sorted(rows, key=lambda row: row["size"])
    first = None
    cpu_preferred = []
    npu_preferred = []
    for row in ordered:
        ratio = (row.get("ratios") or {}).get(ratio_key) or {}
        value = ratio.get("median_ratio")
        if value is None:
            continue
        if value >= threshold:
            npu_preferred.append(row["size"])
            if first is None:
                first = row["size"]
        else:
            cpu_preferred.append(row["size"])
    return {
        "ratio": ratio_key,
        "threshold": threshold,
        "first_size_at_or_above": first,
        "cpu_preferred_sizes": cpu_preferred,
        "npu_preferred_sizes": npu_preferred,
    }


def matrix_status(*, correctness_ok: bool, performance_valid: bool) -> str:
    if not correctness_ok:
        return "failed"
    return "pass" if performance_valid else "profile_required"


def assemble_row(
    *,
    profile: str,
    size: int,
    measurements: Mapping[str, Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    seed: int,
) -> dict:
    """Build one profile/size row from already-collected worker evidence."""

    correctness_ok = bool(checks) and all(item.get("consistent") for item in checks)
    strict_provenance = bool(provenance.get("native_npu"))
    ratios = {}
    if correctness_ok:
        pairs = (
            ("resident", "cpu_warm", "npu_resident"),
            ("warm_end_to_end", "cpu_warm", "npu_warm_e2e"),
            ("cold_end_to_end", "cpu_cold", "npu_cold_e2e"),
        )
        for key, cpu_mode, npu_mode in pairs:
            cpu = measurements.get(cpu_mode)
            npu = measurements.get(npu_mode)
            if cpu and npu:
                ratio = ratio_summary(
                    cpu,
                    npu,
                    seed=seed,
                    basis=f"{cpu_mode}/{npu_mode}",
                )
                ratio["publishable"] = strict_provenance and ratio_publishable(
                    ratio, _RATIO_THRESHOLDS[key]
                )
                ratios[key] = ratio
    label = classify_row(
        correctness_ok=correctness_ok,
        provenance_ok=strict_provenance,
        ratios=ratios,
    )
    return {
        "profile": profile,
        "size": size,
        "shape": [size, size],
        "dtype": "float32",
        "measurements": _select_timing(measurements),
        "correctness": {
            "ok": correctness_ok,
            "checks": list(checks),
        },
        "ratios": ratios,
        "label": label,
    }


def _select_timing(measurements: Mapping[str, Mapping[str, Any]]) -> dict:
    selected = {}
    for mode, summary in measurements.items():
        selected[mode] = {
            key: summary.get(key) for key in _TIMING_KEYS if key in summary
        }
    return selected


def host_environment() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cpu_model": cpu_model(),
        "thread_environment": {
            key: os.environ.get(key) for key in _THREAD_ENV_KEYS
        },
    }


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def write_output(path: Path, array: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)
    np.save(path, array)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def compare_saved_output(reference: np.ndarray, path: Path) -> dict:
    actual = np.load(path, allow_pickle=False)
    result = compare_outputs(reference, actual)
    return {
        "path": path.name,
        "consistent": result.consistent,
        "max_abs_diff": result.max_abs_diff,
        "max_rel_diff": result.max_rel_diff,
        "atol": result.atol,
        "rtol": result.rtol,
        "n_elements": result.n_elements,
        "reason": result.reason,
    }


def timed_call(function, *, warmup: int, repeats: int, synchronize=None):
    """Run warmup and measured calls; return summaries and the last value."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    warmup_samples = []
    for _ in range(warmup):
        start = time.perf_counter()
        value = function()
        if synchronize is not None:
            synchronize()
        warmup_samples.append((time.perf_counter() - start) * 1e3)
    samples = []
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        last = function()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return (
        summarize_samples(
            samples,
            warmup_count=warmup,
            warmup_samples_ms=warmup_samples,
        ),
        last,
    )
