#!/usr/bin/env python3
"""Generate frozen-scale CPU/AsNumpy performance evidence for demo Case 06.

This runner deliberately does not consume a migrated script.  It is a small,
trusted adapter for the exact Case 06 operation chain::

    power -> log -> exp -> equal

The plan is immutable at the command line: three float32 sizes, two warmups,
and seven ordered samples.  CPU and NPU workers are separate Python processes
started through ``taskset`` on the same logical CPU.  NPU timing is enclosed by
``aclrtSynchronizeDevice`` before and after every sample.

Only device-resident compute and the two-input ``full`` allocation scope have
equivalent CPU/NPU work and can receive a ratio.  H2D and D2H are reported as
transfer costs; CPU host-copy samples are retained only as explicitly
non-equivalent diagnostics.  Numerical comparison and capability checks happen
outside every timed scope.  Any failed identity, affinity, thread, numerical,
capability, or synchronization gate leaves ``performance_valid`` false.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np


_SCHEMA_VERSION = "1.0"
_PRODUCER = "numpilot-case06-scale-performance-evidence-v1"
_WORKER_RECEIPT = "worker.json"
_SCALES = (1_000_000, 4_000_000, 9_000_000)
_WARMUP = 2
_SAMPLES = 7
_CPU_THREADS = 1
_WORKER_TIMEOUT_SECONDS = 1_200
_MEMORY_LIMIT_FRACTION = 0.60
_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_NPU_APIS = ("equal", "exp", "full", "log", "power")
_TIMED_SCOPES = (
    "device_resident_compute",
    "core_direct_compute",
    "allocation",
    "h2d",
    "d2h",
)

# This object is the benchmark protocol.  There are intentionally no CLI knobs
# for any field below.  Its literal digest is checked before either worker is
# launched, so changing the protocol requires an explicit review and hash bump.
_FROZEN_PLAN = {
    "schema_version": "1.0",
    "case": "06_log_exp_power",
    "adapter": "trusted fixed-scale adapter; does not execute demo source",
    "dtype": "float32",
    "scales": list(_SCALES),
    "inputs": {
        "base": "full(shape=(scale,), fill_value=2.0, dtype=float32)",
        "exponent": "full(shape=(scale,), fill_value=3.0, dtype=float32)",
    },
    "chain": ["power(base, exponent)", "log(power)", "exp(log)", "equal(power, exp)"],
    "timing": {
        "clock": "time.perf_counter_ns",
        "warmup_samples": _WARMUP,
        "measurement_samples": _SAMPLES,
        "ordered_raw_samples": True,
        "percentiles": "linear interpolation at (n - 1) * q",
        "worker_order": ["cpu", "npu"],
        "python_gc": "disabled only while collecting each scope",
        "asnumpy_logging": "loguru package prefix 'asnumpy' disabled before timing",
        "native_output": (
            "NPU stdout/stderr file descriptors redirected to /dev/null once per "
            "scope outside measured intervals"
        ),
    },
    "scopes": {
        "device_resident_compute": {
            "work": "four public operations; resident base/exponent; outputs retained through completion",
            "includes": ["operator dispatch", "kernel work", "operator output allocation"],
            "excludes": ["input allocation", "H2D", "D2H", "output release"],
            "cpu_npu_ratio_eligible": True,
        },
        "core_direct_compute": {
            "work": "diagnostic private native calls matching the four-op chain",
            "purpose": "isolate public ndarray-wrapper construction/copy tax",
            "cpu_reference": "same NumPy public chain; diagnostic and non-equivalent",
            "correctness": "independently compared outside timing",
            "cpu_npu_ratio_eligible": False,
        },
        "allocation": {
            "work": "two full(shape, value, dtype=float32) calls",
            "includes": ["allocation", "fill", "public wrapper"],
            "excludes": ["release"],
            "cpu_npu_ratio_eligible": True,
        },
        "h2d": {
            "work": "upload two existing contiguous float32 host inputs",
            "cpu_reference": "two host copies; diagnostic and non-equivalent",
            "cpu_npu_ratio_eligible": False,
        },
        "d2h": {
            "work": "download three float32 and one bool resident chain outputs",
            "cpu_reference": "four host copies; diagnostic and non-equivalent",
            "cpu_npu_ratio_eligible": False,
        },
    },
    "process": {
        "cpu_and_npu_workers": "independent taskset processes on the same one-CPU mask",
        "requested_library_threads": _CPU_THREADS,
        "thread_environment": list(_THREAD_ENVIRONMENT),
        "python_hash_seed": "0",
        "worker_timeout_seconds": _WORKER_TIMEOUT_SECONDS,
        "environment": "credential-like and network-proxy variables removed",
    },
    "memory": {
        "gate": "conservative peak <= 60% of available bytes before each scale",
        "limit_fraction": _MEMORY_LIMIT_FRACTION,
        "host_peak_formula": "96 * scale + 256 MiB",
        "device_peak_formula": "64 * scale + 256 MiB",
        "host_source": "/proc/meminfo MemAvailable bounded by cgroup v2 when finite",
        "device_source": "aclrtGetMemInfo(ACL_HBM_MEM)",
    },
    "npu_completion": {
        "api": "aclrtSynchronizeDevice",
        "before_every_timed_sample": True,
        "after_every_timed_sample": True,
    },
    "correctness": {
        "phase": "outside timing",
        "shape_and_dtype_exact": True,
        "floating_tolerance": "numpilot.comparison.compare_outputs dtype policy",
        "nan_and_inf": "compare_outputs equal_nan and finite-pattern policy",
        "all_scales_required": True,
    },
    "publication": {
        "policy": "fail closed",
        "requires": [
            "both workers pass",
            "distinct worker PIDs",
            "identical frozen plan and harness hashes",
            "same verified one-CPU affinity",
            "requested and observed BLAS threads equal one",
            "matched and stable AsNumpy runtime identity",
            "strict native capability/profile/target gates",
            "all numerical comparisons pass",
            "all five scopes contain exactly two warmups and seven measurements",
            "every NPU timed sample has pre/post device synchronization",
        ],
    },
}

# Filled from the canonical plan before any benchmark result was observed.
_FROZEN_PLAN_SHA256 = "35c47fdfb8e391e872ff1d8d5d1bcd71f133771f85d8c4cbdf7f31d86e95a5db"


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


def _plan_sha256() -> str:
    return _sha256_bytes(_canonical_json_bytes(_FROZEN_PLAN))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(memoryview(array).cast("B")),
    }


def _output_identity(output) -> list[dict]:
    return [_array_identity(np.asarray(item)) for item in output]


def _percentile(values: list[int], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


@contextmanager
def _silence_stdout_stderr_fds():
    """Redirect process fds 1/2 without putting dup/open work in a sample."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def _measure_scope(
    operation: Callable[[], Any],
    *,
    scope: str,
    semantic_work: str,
    synchronize: Callable[[], None] | None = None,
    suppress_process_output: bool = False,
) -> dict:
    if scope not in _TIMED_SCOPES:
        raise ValueError(f"unknown timing scope: {scope}")
    sync_before = getattr(synchronize, "call_count", None)

    def one_sample() -> int:
        if synchronize is not None:
            synchronize()
        start = time.perf_counter_ns()
        result = operation()
        if synchronize is not None:
            synchronize()
        elapsed = time.perf_counter_ns() - start
        # Reference-counted output release is deliberately outside the timed
        # interval.  The next pre-sample synchronization establishes a clean
        # completion boundary for any asynchronous release implementation.
        del result
        if elapsed < 0:
            raise RuntimeError("monotonic clock moved backwards")
        return elapsed

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        output_context = (
            _silence_stdout_stderr_fds()
            if suppress_process_output
            else nullcontext()
        )
        with output_context:
            warmup = [one_sample() for _ in range(_WARMUP)]
            measured = [one_sample() for _ in range(_SAMPLES)]
    finally:
        if was_enabled:
            gc.enable()

    sync_after = getattr(synchronize, "call_count", None)
    synchronization = {
        "api": None,
        "before_each_sample": False,
        "after_each_sample": False,
        "observed_calls": 0,
        "expected_calls": 0,
        "gate_passed": True,
    }
    if synchronize is not None:
        observed = (
            sync_after - sync_before
            if isinstance(sync_before, int) and isinstance(sync_after, int)
            else None
        )
        expected = 2 * (_WARMUP + _SAMPLES)
        synchronization = {
            "api": "aclrtSynchronizeDevice",
            "before_each_sample": True,
            "after_each_sample": True,
            "observed_calls": observed,
            "expected_calls": expected,
            "gate_passed": observed == expected,
        }
    return {
        "scope": scope,
        "semantic_work": semantic_work,
        "clock": "time.perf_counter_ns",
        "warmup_samples_ns": warmup,
        "raw_samples_ns": measured,
        "sample_count": len(measured),
        "median_ns": float(_percentile(measured, 0.5)),
        "p10_ns": _percentile(measured, 0.1),
        "p90_ns": _percentile(measured, 0.9),
        "percentile_method": "linear interpolation at (n - 1) * q",
        "process_output": {
            "stdout_stderr_redirected_to_devnull": suppress_process_output,
            "redirection_phase": (
                "once per scope outside measured intervals"
                if suppress_process_output
                else "not applied"
            ),
        },
        "synchronization": synchronization,
    }


class _AclDeviceSynchronizer:
    def __init__(self):
        self._library = ctypes.CDLL("libascendcl.so")
        self._function = self._library.aclrtSynchronizeDevice
        self._function.argtypes = []
        self._function.restype = ctypes.c_int
        self._memory_info = self._library.aclrtGetMemInfo
        self._memory_info.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._memory_info.restype = ctypes.c_int
        self.call_count = 0

    def __call__(self) -> None:
        result = int(self._function())
        self.call_count += 1
        if result != 0:
            raise RuntimeError(
                f"aclrtSynchronizeDevice failed with ACL error {result}"
            )

    def hbm_memory_info(self) -> dict:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        # aclrtMemAttr is defined as ACL_DDR_MEM=0, ACL_HBM_MEM=1 in the
        # audited CANN acl_rt.h.  The CANN/runtime identity gate binds that ABI.
        result = int(self._memory_info(1, ctypes.byref(free), ctypes.byref(total)))
        if result != 0:
            raise RuntimeError(f"aclrtGetMemInfo failed with ACL error {result}")
        if free.value <= 0 or total.value <= 0 or free.value > total.value:
            raise RuntimeError("aclrtGetMemInfo returned invalid HBM byte counts")
        return {
            "api": "aclrtGetMemInfo",
            "attribute": "ACL_HBM_MEM",
            "attribute_value": 1,
            "free_bytes": int(free.value),
            "total_bytes": int(total.value),
        }

    def identity(self) -> dict:
        requested = str(self._library._name)
        resolved = None
        maps = Path("/proc/self/maps")
        if maps.is_file():
            candidates = set()
            for line in maps.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6:
                    loaded_name = Path(fields[5]).name
                    requested_name = Path(requested).name
                    if (
                        loaded_name == requested_name
                        or loaded_name.startswith(requested_name + ".")
                    ):
                        candidates.add(fields[5])
            if len(candidates) == 1:
                candidate = Path(candidates.pop()).resolve()
                if candidate.is_file():
                    resolved = candidate
        return {
            "requested_library": requested,
            "resolved_library": None if resolved is None else str(resolved),
            "library_sha256": None if resolved is None else _sha256_path(resolved),
            "symbol": "aclrtSynchronizeDevice",
            "calls": self.call_count,
        }


def _threadpool_info() -> list[dict]:
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
            "scipy_openblas_get_num_threads64_",
            "scipy_openblas_get_num_threads_64_",
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
        for symbol in (
            "scipy_openblas_get_config64_",
            "scipy_openblas_get_config",
            "openblas_get_config64_",
            "openblas_get_config",
        ):
            try:
                config_getter = getattr(library, symbol)
            except AttributeError:
                continue
            config_getter.argtypes = []
            config_getter.restype = ctypes.c_char_p
            raw = config_getter()
            config = None if raw is None else raw.decode("utf-8", "replace")
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


def _process_thread_count() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Threads:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _host_memory_info() -> dict:
    meminfo = Path("/proc/meminfo")
    memavailable = None
    if meminfo.is_file():
        for line in meminfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) >= 2:
                    memavailable = int(fields[1]) * 1024
                break
    if memavailable is None or memavailable <= 0:
        raise RuntimeError("host MemAvailable bytes are unavailable")
    available = memavailable

    cgroup_max_path = Path("/sys/fs/cgroup/memory.max")
    cgroup_current_path = Path("/sys/fs/cgroup/memory.current")
    cgroup_limit = None
    cgroup_current = None
    cgroup_available = None
    if cgroup_max_path.is_file() and cgroup_current_path.is_file():
        raw_max = cgroup_max_path.read_text(encoding="ascii").strip()
        raw_current = cgroup_current_path.read_text(encoding="ascii").strip()
        if raw_max != "max":
            cgroup_limit = int(raw_max)
            cgroup_current = int(raw_current)
            cgroup_available = max(cgroup_limit - cgroup_current, 0)
            available = min(available, cgroup_available)
    if available <= 0:
        raise RuntimeError("effective host available memory is zero")
    return {
        "api": "/proc/meminfo plus cgroup v2",
        "memavailable_bytes": int(memavailable),
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup_current,
        "cgroup_available_bytes": cgroup_available,
        "effective_available_bytes": int(available),
    }


def _memory_budget(size: int, *, device_info: dict | None = None) -> dict:
    if size not in _SCALES:
        raise ValueError("size is outside the frozen memory plan")
    fixed_workspace = 256 * 1024 * 1024
    host_peak = 96 * size + fixed_workspace
    device_peak = 64 * size + fixed_workspace
    host_info = _host_memory_info()
    host_available = host_info["effective_available_bytes"]
    host_limit = int(host_available * _MEMORY_LIMIT_FRACTION)
    host_gate = host_peak <= host_limit

    device = None
    device_gate = True
    if device_info is not None:
        device_available = int(device_info["free_bytes"])
        device_limit = int(device_available * _MEMORY_LIMIT_FRACTION)
        device_gate = device_peak <= device_limit
        device = {
            "conservative_peak_bytes": device_peak,
            "available_bytes": device_available,
            "limit_fraction": _MEMORY_LIMIT_FRACTION,
            "limit_bytes": device_limit,
            "gate_passed": device_gate,
            "estimate": "64 * scale + 256 MiB",
            "observation": device_info,
        }
    return {
        "host": {
            "conservative_peak_bytes": host_peak,
            "available_bytes": host_available,
            "limit_fraction": _MEMORY_LIMIT_FRACTION,
            "limit_bytes": host_limit,
            "gate_passed": host_gate,
            "estimate": "96 * scale + 256 MiB",
            "observation": host_info,
        },
        "device": device,
        "gate_passed": host_gate and device_gate,
    }


def _process_observation(expected_core: int) -> dict:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("os.sched_getaffinity is required")
    affinity = sorted(os.sched_getaffinity(0))
    environment = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    return {
        "pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requested_logical_cpu": expected_core,
        "observed_logical_cpus": affinity,
        "affinity_gate_passed": affinity == [expected_core],
        "launch_method": "taskset --cpu-list before Python startup",
        "thread_limit_environment": environment,
        "thread_environment_gate_passed": all(
            value == str(_CPU_THREADS) for value in environment.values()
        ),
        "process_thread_count_observed": _process_thread_count(),
        "threadpools": _threadpool_info(),
    }


def _module_file_identity(module, *, native_module=None) -> dict:
    module_path = Path(module.__file__).resolve()
    native_path = None
    if native_module is not None:
        raw = getattr(native_module, "__file__", None)
        if raw:
            native_path = Path(raw).resolve()
    value = {
        "module": module.__name__,
        "version": str(getattr(module, "__version__", "unknown")),
        "module_path": str(module_path),
        "module_sha256": _sha256_path(module_path),
        "native_module_path": None if native_path is None else str(native_path),
        "native_module_sha256": (
            None if native_path is None else _sha256_path(native_path)
        ),
    }
    value["identity_sha256"] = _sha256_bytes(_canonical_json_bytes(value))
    return value


def _numpy_runtime_identity() -> dict:
    try:
        from numpy._core import _multiarray_umath as native
    except ImportError:  # NumPy 1.x public package layout
        from numpy.core import _multiarray_umath as native
    return _module_file_identity(np, native_module=native)


def _host_inputs(size: int) -> tuple[np.ndarray, np.ndarray]:
    if size not in _SCALES:
        raise ValueError("size is outside the frozen Case 06 plan")
    return (
        np.full((size,), 2.0, dtype=np.float32),
        np.full((size,), 3.0, dtype=np.float32),
    )


def _chain(api, base, exponent):
    powered = api.power(base, exponent)
    logged = api.log(powered)
    exponentiated = api.exp(logged)
    equal = api.equal(powered, exponentiated)
    return powered, logged, exponentiated, equal


class _ApiSnapshot:
    def __init__(self, targets: Mapping[str, Callable]):
        self._targets = dict(targets)

    def __getattr__(self, name: str):
        try:
            return self._targets[name]
        except KeyError as error:
            raise AttributeError(name) from error


class _CoreDirectApi:
    """Snapshot the private native callables used by the four public wrappers.

    This is diagnostic evidence only.  It is never accepted as a supported
    NumPilot API and never receives a CPU/NPU speedup claim.
    """

    def __init__(self):
        import importlib

        math_module = importlib.import_module("asnumpy.math")
        logic_module = importlib.import_module("asnumpy.logic")
        self._modules = {"math": math_module, "logic": logic_module}
        self._targets = {
            "power": getattr(math_module, "_power"),
            "log": getattr(math_module, "_log"),
            "exp": getattr(math_module, "_exp"),
            "equal": getattr(logic_module, "_equal"),
        }
        if any(not callable(target) for target in self._targets.values()):
            raise RuntimeError("private native Case 06 target is not callable")

    def power(self, left, right):
        return self._targets["power"](left, right, None)

    def log(self, value):
        return self._targets["log"](value)

    def exp(self, value):
        return self._targets["exp"](value)

    def equal(self, left, right):
        return self._targets["equal"](left, right, None)

    def identity_receipt(self) -> dict:
        locations = {
            "power": ("math", "_power"),
            "log": ("math", "_log"),
            "exp": ("math", "_exp"),
            "equal": ("logic", "_equal"),
        }
        checks = {
            name: getattr(self._modules[module_name], attribute) is self._targets[name]
            for name, (module_name, attribute) in locations.items()
        }
        return {
            "diagnostic_only": True,
            "not_a_supported_public_api": True,
            "targets": {
                name: {
                    "module": getattr(target, "__module__", None),
                    "name": getattr(target, "__name__", None),
                    "public_wrapper": name,
                }
                for name, target in self._targets.items()
            },
            "identity_checks": checks,
            "identity_gate_passed": all(checks.values()),
        }


def _comparison_receipt(comparison) -> dict:
    return {
        "consistent": comparison.consistent,
        "max_abs_diff": comparison.max_abs_diff,
        "max_rel_diff": comparison.max_rel_diff,
        "atol": comparison.atol,
        "rtol": comparison.rtol,
        "failure_path": comparison.failure_path,
        "failure_index": comparison.failure_index,
        "reason": comparison.reason,
        "n_elements": comparison.n_elements,
    }


def _cpu_scale(size: int) -> dict:
    memory = _memory_budget(size)
    if not memory["gate_passed"]:
        raise RuntimeError(f"scale {size} exceeds the frozen 60% host memory gate")
    base, exponent = _host_inputs(size)
    reference = _chain(np, base, exponent)
    scopes = {
        "device_resident_compute": _measure_scope(
            lambda: _chain(np, base, exponent),
            scope="device_resident_compute",
            semantic_work="resident NumPy inputs; power -> log -> exp -> equal",
        ),
        "core_direct_compute": _measure_scope(
            lambda: _chain(np, base, exponent),
            scope="core_direct_compute",
            semantic_work=(
                "CPU diagnostic only: repeated NumPy public chain; not equivalent "
                "to private AsNumpy native calls"
            ),
        ),
        "allocation": _measure_scope(
            lambda: (
                np.full((size,), 2.0, dtype=np.float32),
                np.full((size,), 3.0, dtype=np.float32),
            ),
            scope="allocation",
            semantic_work="two NumPy full float32 allocations and fills",
        ),
        "h2d": _measure_scope(
            lambda: (base.copy(), exponent.copy()),
            scope="h2d",
            semantic_work=(
                "CPU diagnostic only: two host copies; not equivalent to NPU H2D"
            ),
        ),
        "d2h": _measure_scope(
            lambda: tuple(item.copy() for item in reference),
            scope="d2h",
            semantic_work=(
                "CPU diagnostic only: four host copies; not equivalent to NPU D2H"
            ),
        ),
    }
    value = {
        "size": size,
        "dtype": "float32",
        "memory": memory,
        "input_identity": [_array_identity(base), _array_identity(exponent)],
        "reference_output_identity": _output_identity(reference),
        "scopes": scopes,
    }
    del reference, base, exponent
    gc.collect()
    return value


def _cpu_worker(expected_core: int) -> dict:
    runtime = _numpy_runtime_identity()
    scales = [_cpu_scale(size) for size in _SCALES]
    process = _process_observation(expected_core)
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer": _PRODUCER + "-cpu-worker",
        "status": "pass",
        "backend": "numpy",
        "harness_sha256": _sha256_path(Path(__file__).resolve()),
        "plan_sha256": _plan_sha256(),
        "runtime": runtime,
        "process": process,
        "scales": scales,
    }


def _disable_asnumpy_logging() -> dict:
    try:
        from loguru import logger
    except ImportError:
        return {"requested": True, "applied": False, "reason": "loguru unavailable"}
    logger.disable("asnumpy")
    return {"requested": True, "applied": True, "package_prefix": "asnumpy"}


def _validate_guard_call(guard, name: str, args: tuple, kwargs: dict) -> None:
    guard.validate_call(name, args, kwargs)


def _npu_scale(size: int, backend, module, synchronizer, core_api, memory) -> dict:
    from numpilot.comparison import compare_outputs

    base, exponent = _host_inputs(size)
    reference = _chain(np, base, exponent)
    guard = backend.prepare_profile_guard(_REQUIRED_NPU_APIS)
    targets = guard.runtime_targets
    if targets is None or set(targets) != set(_REQUIRED_NPU_APIS):
        raise RuntimeError("NPU runtime target snapshot is incomplete")
    api = _ApiSnapshot(targets)

    dtype = module.float32
    full_kwargs = {"dtype": dtype}
    _validate_guard_call(guard, "full", ((size,), 2.0), full_kwargs)
    first_allocation = targets["full"]((size,), 2.0, dtype=dtype)
    _validate_guard_call(guard, "full", ((size,), 3.0), full_kwargs)
    second_allocation = targets["full"]((size,), 3.0, dtype=dtype)
    synchronizer()
    del first_allocation, second_allocation
    gc.collect()
    synchronizer()

    device_base = backend.to_device(base)
    device_exponent = backend.to_device(exponent)
    synchronizer()

    _validate_guard_call(guard, "power", (device_base, device_exponent), {})
    powered = targets["power"](device_base, device_exponent)
    _validate_guard_call(guard, "log", (powered,), {})
    logged = targets["log"](powered)
    _validate_guard_call(guard, "exp", (logged,), {})
    exponentiated = targets["exp"](logged)
    _validate_guard_call(guard, "equal", (powered, exponentiated), {})
    equal = targets["equal"](powered, exponentiated)
    device_output = (powered, logged, exponentiated, equal)
    del powered, logged, exponentiated, equal
    synchronizer()
    host_output = tuple(backend.to_host(item) for item in device_output)
    comparison = compare_outputs(reference, host_output)
    reference_output_identity = _output_identity(reference)
    npu_output_identity = _output_identity(host_output)
    del host_output

    core_device_output = _chain(core_api, device_base, device_exponent)
    synchronizer()
    core_host_output = tuple(backend.to_host(item) for item in core_device_output)
    core_comparison = compare_outputs(reference, core_host_output)
    core_output_identity = _output_identity(core_host_output)
    del core_host_output, core_device_output, reference
    gc.collect()
    synchronizer()

    scopes = {
        "device_resident_compute": _measure_scope(
            lambda: _chain(api, device_base, device_exponent),
            scope="device_resident_compute",
            semantic_work=(
                "resident AsNumpy inputs; power -> log -> exp -> equal; terminal device sync"
            ),
            synchronize=synchronizer,
            suppress_process_output=True,
        ),
        "core_direct_compute": _measure_scope(
            lambda: _chain(core_api, device_base, device_exponent),
            scope="core_direct_compute",
            semantic_work=(
                "diagnostic private native power -> log -> exp -> equal; "
                "no public ndarray wrapper construction; terminal device sync"
            ),
            synchronize=synchronizer,
            suppress_process_output=True,
        ),
        "allocation": _measure_scope(
            lambda: (
                targets["full"]((size,), 2.0, dtype=dtype),
                targets["full"]((size,), 3.0, dtype=dtype),
            ),
            scope="allocation",
            semantic_work=(
                "two AsNumpy full float32 allocations and fills; terminal device sync"
            ),
            synchronize=synchronizer,
            suppress_process_output=True,
        ),
        "h2d": _measure_scope(
            lambda: (backend.to_device(base), backend.to_device(exponent)),
            scope="h2d",
            semantic_work=(
                "upload two existing contiguous float32 host inputs; terminal device sync"
            ),
            synchronize=synchronizer,
            suppress_process_output=True,
        ),
        "d2h": _measure_scope(
            lambda: tuple(backend.to_host(item) for item in device_output),
            scope="d2h",
            semantic_work=(
                "download three float32 and one bool resident outputs; terminal device sync"
            ),
            synchronize=synchronizer,
            suppress_process_output=True,
        ),
    }

    target_integrity = {
        name: getattr(module, name) is target for name, target in targets.items()
    }
    core_identity = core_api.identity_receipt()
    guard_receipt = guard.to_dict()
    value = {
        "size": size,
        "dtype": "float32",
        "memory": memory,
        "input_identity": [_array_identity(base), _array_identity(exponent)],
        "reference_output_identity": reference_output_identity,
        "npu_output_identity": npu_output_identity,
        "comparison": _comparison_receipt(comparison),
        "core_direct_output_identity": core_output_identity,
        "core_direct_comparison": _comparison_receipt(core_comparison),
        "core_direct_runtime_targets": core_identity,
        "capability": guard_receipt,
        "runtime_target_identity_after_timing": target_integrity,
        "runtime_target_identity_gate_passed": all(target_integrity.values()),
        "scopes": scopes,
    }
    del (
        device_output,
        device_base,
        device_exponent,
        base,
        exponent,
    )
    gc.collect()
    synchronizer()
    return value


def _npu_worker(expected_core: int) -> dict:
    from numpilot.backends.npu import NpuBackend

    backend = NpuBackend()
    runtime_before = backend.runtime_identity().to_dict()
    module = backend.get_module()
    logging = _disable_asnumpy_logging()
    synchronizer = _AclDeviceSynchronizer()
    core_api = _CoreDirectApi()
    initialization_value = backend.to_device(np.array([2.0], dtype=np.float32))
    synchronizer()
    del initialization_value
    gc.collect()
    synchronizer()
    runtime_initialization = {
        "phase": "before memory observation and timing",
        "method": "one-element float32 H2D followed by device synchronization",
        "timed": False,
        "gate_passed": True,
    }
    scales = []
    for size in _SCALES:
        memory = _memory_budget(
            size, device_info=synchronizer.hbm_memory_info()
        )
        if not memory["gate_passed"]:
            raise RuntimeError(
                f"scale {size} exceeds the frozen 60% host/device memory gate"
            )
        scales.append(
            _npu_scale(
                size, backend, module, synchronizer, core_api, memory
            )
        )
    runtime_after = backend.runtime_identity().to_dict()
    process = _process_observation(expected_core)
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer": _PRODUCER + "-npu-worker",
        "status": "pass",
        "backend": "asnumpy",
        "harness_sha256": _sha256_path(Path(__file__).resolve()),
        "plan_sha256": _plan_sha256(),
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "runtime_stable": (
            runtime_before.get("identity_sha256")
            == runtime_after.get("identity_sha256")
        ),
        "asnumpy_logging": logging,
        "runtime_initialization": runtime_initialization,
        "synchronizer": synchronizer.identity(),
        "process": process,
        "scales": scales,
    }


def _write_once_json(path: Path, value: dict) -> None:
    path = path.resolve()
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        path.parent.chmod(0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _read_receipt(path: Path) -> dict:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024 * 1024:
        raise ValueError("worker receipt is not a bounded regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("worker receipt must be a JSON object")
    return value


def _run_worker(args) -> int:
    output = Path(args.worker_out).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    try:
        harness_sha = _sha256_path(Path(__file__).resolve())
        if args.expected_harness_sha256 != harness_sha:
            raise RuntimeError("worker harness hash differs from parent launch hash")
        if args.expected_plan_sha256 != _plan_sha256():
            raise RuntimeError("worker plan hash differs from parent launch hash")
        if _plan_sha256() != _FROZEN_PLAN_SHA256:
            raise RuntimeError("frozen plan literal hash requires review")
        receipt = (
            _cpu_worker(args.expected_cpu_core)
            if args.worker == "cpu"
            else _npu_worker(args.expected_cpu_core)
        )
    except Exception as error:
        receipt = {
            "schema_version": _SCHEMA_VERSION,
            "producer": _PRODUCER + f"-{args.worker}-worker",
            "status": "fail",
            "backend": args.worker,
            "pid": os.getpid(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
    _write_once_json(output / _WORKER_RECEIPT, receipt)
    return 0 if receipt["status"] == "pass" else 2


def _available_cpus() -> set[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("os.sched_getaffinity is required")
    available = set(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("current process has an empty CPU affinity mask")
    return available


def _resolve_cpu_core(raw: int | None) -> int:
    available = _available_cpus()
    core = min(available) if raw is None else raw
    if isinstance(core, bool) or core not in available:
        raise ValueError(f"logical CPU {core!r} is outside the allowed affinity mask")
    return core


def _worker_command(
    worker: str,
    worker_out: Path,
    *,
    cpu_core: int,
    harness_sha256: str,
    plan_sha256: str,
) -> list[str]:
    if worker not in {"cpu", "npu"}:
        raise ValueError("worker must be cpu or npu")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required")
    return [
        taskset,
        "--cpu-list",
        str(cpu_core),
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--worker-out",
        str(worker_out),
        "--expected-cpu-core",
        str(cpu_core),
        "--expected-harness-sha256",
        harness_sha256,
        "--expected-plan-sha256",
        plan_sha256,
    ]


def _scrubbed_worker_environment(source: Mapping[str, str]) -> tuple[dict, list[str]]:
    environment = dict(source)
    explicit = {
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "NUMPILOT_LLM_CONFIG", "NUMPILOT_LLM_CONFIG_PATH",
    }
    sensitive_suffixes = (
        "API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "PASSWORD", "SECRET",
        "CREDENTIAL", "CREDENTIALS",
    )
    removed = []
    for name in list(environment):
        upper = name.upper()
        if name in explicit or upper.endswith(sensitive_suffixes):
            removed.append(name)
            environment.pop(name, None)
    return environment, sorted(removed)


def _blas_thread_gate(process: Mapping[str, Any]) -> tuple[bool, str | None]:
    pools = [
        item
        for item in process.get("threadpools", [])
        if item.get("user_api") == "blas"
    ]
    if not pools:
        return False, "BLAS runtime thread count was not observed"
    if any(item.get("num_threads") != _CPU_THREADS for item in pools):
        return False, "BLAS runtime thread count differs from one"
    return True, None


def _timing_gate(timing: Mapping[str, Any], *, npu: bool) -> list[str]:
    issues = []
    if len(timing.get("warmup_samples_ns", [])) != _WARMUP:
        issues.append("warmup count differs from frozen plan")
    if len(timing.get("raw_samples_ns", [])) != _SAMPLES:
        issues.append("measurement count differs from frozen plan")
    if timing.get("sample_count") != _SAMPLES:
        issues.append("recorded sample_count differs from frozen plan")
    samples = timing.get("raw_samples_ns", [])
    if any(type(value) is not int or value < 0 for value in samples):
        issues.append("timing samples are not non-negative integer nanoseconds")
    sync = timing.get("synchronization", {})
    output = timing.get("process_output", {})
    if npu:
        if not (
            sync.get("api") == "aclrtSynchronizeDevice"
            and sync.get("before_each_sample") is True
            and sync.get("after_each_sample") is True
            and sync.get("gate_passed") is True
            and sync.get("observed_calls") == 2 * (_WARMUP + _SAMPLES)
        ):
            issues.append("NPU synchronization boundary gate failed")
        if output.get("stdout_stderr_redirected_to_devnull") is not True:
            issues.append("NPU native output was not excluded at file-descriptor level")
    elif sync.get("api") is not None:
        issues.append("CPU timing unexpectedly declares a device synchronizer")
    elif output.get("stdout_stderr_redirected_to_devnull") is not False:
        issues.append("CPU output-redirection receipt is invalid")
    return issues


def _scale_map(receipt: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result = {}
    for item in receipt.get("scales", []):
        size = item.get("size") if isinstance(item, Mapping) else None
        if type(size) is int and size not in result:
            result[size] = item
    return result


def _combine(cpu: dict, npu: dict, *, cpu_core: int, harness_sha: str) -> dict:
    issues: list[str] = []
    if cpu.get("status") != "pass" or npu.get("status") != "pass":
        issues.append("one or more workers failed")
    for name, receipt, expected_backend in (
        ("CPU", cpu, "numpy"),
        ("NPU", npu, "asnumpy"),
    ):
        if receipt.get("backend") != expected_backend:
            issues.append(f"{name} backend identity mismatch")
        capture = receipt.get("process_capture", {})
        if capture.get("returncode") != 0:
            issues.append(f"{name} worker process did not exit successfully")
        if (
            capture.get("timeout_seconds") != _WORKER_TIMEOUT_SECONDS
            or capture.get("credential_environment_scrubbed") is not True
        ):
            issues.append(f"{name} worker timeout/environment policy gate failed")
        if receipt.get("harness_sha256") != harness_sha:
            issues.append(f"{name} harness hash mismatch")
        if receipt.get("plan_sha256") != _FROZEN_PLAN_SHA256:
            issues.append(f"{name} plan hash mismatch")
        process = receipt.get("process", {})
        if process.get("observed_logical_cpus") != [cpu_core]:
            issues.append(f"{name} affinity is not the requested single CPU")
        if process.get("affinity_gate_passed") is not True:
            issues.append(f"{name} affinity gate failed")
        if process.get("thread_environment_gate_passed") is not True:
            issues.append(f"{name} thread environment gate failed")
        environment = process.get("thread_limit_environment", {})
        if (
            set(environment) != set(_THREAD_ENVIRONMENT)
            or any(environment.get(key) != str(_CPU_THREADS) for key in _THREAD_ENVIRONMENT)
        ):
            issues.append(f"{name} recorded thread environment differs from one")
        thread_ok, thread_issue = _blas_thread_gate(process)
        if not thread_ok:
            issues.append(f"{name} {thread_issue}")

    cpu_pid = cpu.get("process", {}).get("pid")
    npu_pid = npu.get("process", {}).get("pid")
    if type(cpu_pid) is not int or type(npu_pid) is not int or cpu_pid == npu_pid:
        issues.append("workers do not have distinct valid PIDs")

    runtime_before = npu.get("runtime_before", {})
    runtime_after = npu.get("runtime_after", {})
    if runtime_before.get("status") != "matched":
        issues.append("AsNumpy runtime identity is not matched before timing")
    if runtime_after.get("status") != "matched":
        issues.append("AsNumpy runtime identity is not matched after timing")
    if (
        not runtime_before.get("identity_sha256")
        or runtime_before.get("identity_sha256")
        != runtime_after.get("identity_sha256")
        or npu.get("runtime_stable") is not True
    ):
        issues.append("AsNumpy runtime identity changed during the worker")
    if not cpu.get("runtime", {}).get("identity_sha256"):
        issues.append("NumPy runtime identity hash is missing")
    synchronizer = npu.get("synchronizer", {})
    if (
        synchronizer.get("symbol") != "aclrtSynchronizeDevice"
        or not synchronizer.get("resolved_library")
        or not synchronizer.get("library_sha256")
    ):
        issues.append("ACL synchronization library identity is incomplete")
    if npu.get("asnumpy_logging", {}).get("applied") is not True:
        issues.append("AsNumpy logging exclusion was not applied")
    if npu.get("runtime_initialization", {}).get("gate_passed") is not True:
        issues.append("NPU runtime initialization gate failed")

    cpu_scales = _scale_map(cpu)
    npu_scales = _scale_map(npu)
    if tuple(sorted(cpu_scales)) != _SCALES:
        issues.append("CPU scales differ from frozen plan")
    if tuple(sorted(npu_scales)) != _SCALES:
        issues.append("NPU scales differ from frozen plan")

    per_scale_issues: dict[int, list[str]] = {size: [] for size in _SCALES}
    for size in _SCALES:
        left = cpu_scales.get(size, {})
        right = npu_scales.get(size, {})
        local = per_scale_issues[size]
        expected_host_peak = 96 * size + 256 * 1024 * 1024
        expected_device_peak = 64 * size + 256 * 1024 * 1024
        for backend_name, memory, require_device in (
            ("CPU", left.get("memory", {}), False),
            ("NPU", right.get("memory", {}), True),
        ):
            host = memory.get("host", {})
            if (
                memory.get("gate_passed") is not True
                or host.get("gate_passed") is not True
                or host.get("conservative_peak_bytes") != expected_host_peak
                or type(host.get("available_bytes")) is not int
                or host.get("available_bytes", 0) <= 0
                or host.get("limit_fraction") != _MEMORY_LIMIT_FRACTION
                or host.get("limit_bytes")
                != int(host.get("available_bytes", 0) * _MEMORY_LIMIT_FRACTION)
                or host.get("conservative_peak_bytes", expected_host_peak + 1)
                > host.get("limit_bytes", 0)
            ):
                local.append(f"{backend_name} host 60% memory gate failed")
            device = memory.get("device")
            if require_device:
                if (
                    not isinstance(device, Mapping)
                    or device.get("gate_passed") is not True
                    or device.get("conservative_peak_bytes") != expected_device_peak
                    or type(device.get("available_bytes")) is not int
                    or device.get("available_bytes", 0) <= 0
                    or device.get("limit_fraction") != _MEMORY_LIMIT_FRACTION
                    or device.get("limit_bytes")
                    != int(
                        device.get("available_bytes", 0)
                        * _MEMORY_LIMIT_FRACTION
                    )
                    or device.get("conservative_peak_bytes", expected_device_peak + 1)
                    > device.get("limit_bytes", 0)
                ):
                    local.append(f"{backend_name} device 60% memory gate failed")
            elif device is not None:
                local.append("CPU memory receipt unexpectedly contains device memory")
        if left.get("input_identity") != right.get("input_identity"):
            local.append("CPU/NPU input identity mismatch")
        if (
            left.get("reference_output_identity")
            != right.get("reference_output_identity")
        ):
            local.append("NPU worker CPU reference differs from CPU worker")
        if right.get("comparison", {}).get("consistent") is not True:
            local.append("CPU/NPU numerical comparison failed")
        if right.get("core_direct_comparison", {}).get("consistent") is not True:
            local.append("private native diagnostic numerical comparison failed")
        if (
            right.get("core_direct_runtime_targets", {}).get(
                "identity_gate_passed"
            )
            is not True
        ):
            local.append("private native diagnostic target identity changed")
        capability = right.get("capability", {})
        for key in (
            "ok",
            "preflight_ok",
            "preflight_strict_npu",
            "strict_npu",
            "runtime_profile_ok",
            "runtime_target_integrity_ok",
        ):
            if capability.get(key) is not True:
                local.append(f"capability gate failed: {key}")
        if set(capability.get("required_ops", [])) != set(_REQUIRED_NPU_APIS):
            local.append("capability required API set differs from plan")
        if capability.get("profile_status") not in {"not_required", "matched"}:
            local.append("capability profile status is not admitted")
        if right.get("runtime_target_identity_gate_passed") is not True:
            local.append("runtime target identity changed after timing")
        for backend_name, scale, is_npu in (
            ("CPU", left, False),
            ("NPU", right, True),
        ):
            scopes = scale.get("scopes", {})
            if set(scopes) != set(_TIMED_SCOPES):
                local.append(f"{backend_name} timed scope set differs from plan")
                continue
            for scope in _TIMED_SCOPES:
                for issue in _timing_gate(scopes[scope], npu=is_npu):
                    local.append(f"{backend_name} {scope}: {issue}")
        issues.extend(f"scale {size}: {issue}" for issue in local)

    performance_valid = not issues
    comparisons = {}
    for size in _SCALES:
        cpu_scale = cpu_scales.get(size, {})
        npu_scale = npu_scales.get(size, {})
        scopes = {}
        for scope in _TIMED_SCOPES:
            cpu_timing = cpu_scale.get("scopes", {}).get(scope)
            npu_timing = npu_scale.get("scopes", {}).get(scope)
            ratio_eligible = scope in {"device_resident_compute", "allocation"}
            ratio = None
            if performance_valid and ratio_eligible and cpu_timing and npu_timing:
                npu_median = float(npu_timing["median_ns"])
                if npu_median > 0:
                    ratio = float(cpu_timing["median_ns"]) / npu_median
            scopes[scope] = {
                "cpu_timing_path": (
                    f"$.cpu.scales[size={size}].scopes.{scope}"
                ),
                "npu_timing_path": (
                    f"$.npu.scales[size={size}].scopes.{scope}"
                ),
                "cpu_median_ns": (
                    None if cpu_timing is None else cpu_timing.get("median_ns")
                ),
                "npu_median_ns": (
                    None if npu_timing is None else npu_timing.get("median_ns")
                ),
                "cpu_npu_ratio_eligible": ratio_eligible,
                "cpu_time_over_npu_time": ratio,
                "interpretation": (
                    "greater than one means NPU is faster"
                    if ratio_eligible
                    else "no speedup claim; CPU value is a non-equivalent host-copy reference"
                ),
            }
        wrapper_core_ratio = None
        if performance_valid:
            public_median = float(
                npu_scale["scopes"]["device_resident_compute"]["median_ns"]
            )
            core_median = float(
                npu_scale["scopes"]["core_direct_compute"]["median_ns"]
            )
            if core_median > 0:
                wrapper_core_ratio = public_median / core_median
        comparisons[str(size)] = {
            "performance_valid": performance_valid and not per_scale_issues[size],
            "issues": per_scale_issues[size],
            "numerical_path": f"$.npu.scales[size={size}].comparison",
            "core_direct_numerical_path": (
                f"$.npu.scales[size={size}].core_direct_comparison"
            ),
            "npu_public_wrapper_time_over_core_direct_time": wrapper_core_ratio,
            "wrapper_tax_interpretation": (
                "diagnostic only; greater than one means the public-wrapper chain "
                "was slower than the private native chain"
            ),
            "scopes": scopes,
        }

    status = "pass" if cpu.get("status") == npu.get("status") == "pass" else "fail"
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer": _PRODUCER,
        "status": status,
        "performance_valid": performance_valid,
        "performance_issues": issues,
        "frozen_plan": _FROZEN_PLAN,
        "plan_sha256": _FROZEN_PLAN_SHA256,
        "plan_frozen_before_execution": True,
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": harness_sha,
        "cpu_core": cpu_core,
        "cpu": cpu,
        "npu": npu,
        "comparisons": comparisons,
    }


def _failed_parent_receipt(error: Exception, *, harness_sha: str | None) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer": _PRODUCER,
        "status": "fail",
        "performance_valid": False,
        "performance_issues": [f"{type(error).__name__}: {error}"],
        "frozen_plan": _FROZEN_PLAN,
        "plan_sha256": _plan_sha256(),
        "plan_frozen_before_execution": True,
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": harness_sha,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def _run_parent(args) -> int:
    output = Path(args.out).resolve()
    harness_sha = None
    try:
        if output.exists():
            raise FileExistsError(f"--out is write-once and already exists: {output}")
        if _plan_sha256() != _FROZEN_PLAN_SHA256:
            raise RuntimeError("frozen plan literal hash requires review")
        cpu_core = _resolve_cpu_core(args.cpu_core)
        harness_sha = _sha256_path(Path(__file__).resolve())
        with tempfile.TemporaryDirectory(prefix="numpilot-case06-performance-") as raw:
            staging = Path(raw)
            staging.chmod(0o700)
            receipts = {}
            for worker in ("cpu", "npu"):
                worker_out = staging / worker
                environment, removed_environment = _scrubbed_worker_environment(
                    os.environ
                )
                for name in _THREAD_ENVIRONMENT:
                    environment[name] = str(_CPU_THREADS)
                environment["PYTHONHASHSEED"] = "0"
                completed = subprocess.run(
                    _worker_command(
                        worker,
                        worker_out,
                        cpu_core=cpu_core,
                        harness_sha256=harness_sha,
                        plan_sha256=_FROZEN_PLAN_SHA256,
                    ),
                    cwd=str(Path(__file__).resolve().parents[1]),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=_WORKER_TIMEOUT_SECONDS,
                )
                receipt_path = worker_out / _WORKER_RECEIPT
                if not receipt_path.is_file():
                    raise RuntimeError(f"{worker} worker did not produce a receipt")
                receipt = _read_receipt(receipt_path)
                receipt["process_capture"] = {
                    "returncode": completed.returncode,
                    "stdout_bytes": len(completed.stdout),
                    "stdout_sha256": _sha256_bytes(completed.stdout),
                    "stderr_bytes": len(completed.stderr),
                    "stderr_sha256": _sha256_bytes(completed.stderr),
                    "timeout_seconds": _WORKER_TIMEOUT_SECONDS,
                    "credential_environment_scrubbed": True,
                    "removed_environment_names": removed_environment,
                }
                receipts[worker] = receipt
            result = _combine(
                receipts["cpu"], receipts["npu"],
                cpu_core=cpu_core, harness_sha=harness_sha,
            )
    except Exception as error:
        result = _failed_parent_receipt(error, harness_sha=harness_sha)

    # Even failure evidence is committed once.  A caller must choose a fresh
    # path to retry; stale or partial evidence is never silently overwritten.
    _write_once_json(output, result)
    print(
        f"status={result['status']} "
        f"performance_valid={str(result['performance_valid']).lower()} "
        f"out={output}"
    )
    return 0 if result["status"] == "pass" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write-once private JSON evidence path")
    parser.add_argument("--cpu-core", type=int)
    parser.add_argument("--worker", choices=("cpu", "npu"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    parser.add_argument("--expected-cpu-core", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--expected-harness-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        required = {
            "--worker-out": args.worker_out,
            "--expected-cpu-core": args.expected_cpu_core,
            "--expected-harness-sha256": args.expected_harness_sha256,
            "--expected-plan-sha256": args.expected_plan_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("internal worker is missing " + ", ".join(missing))
        return _run_worker(args)
    if not args.out:
        parser.error("--out is required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
