#!/usr/bin/env python3
"""Produce source-bound numerical and steady whole-script demo evidence.

The parent validates the immutable Day-3 migration index, then starts one CPU
worker and one NPU worker through ``taskset`` on the same host core.  Workers do
not parse program output: demo stdout/stderr are redirected to ``/dev/null`` and
the explicitly declared module globals are serialized as non-object NPZ arrays.

Timing is intentionally the exact, tiny whole-script workload.  It includes
cached imports, allocations, operators, formatting and migrated output-boundary
copies.  Compilation, numerical capture and runtime initialization happen
before two warmups and seven ordered measurements.  Every NPU timing sample is
closed with ``aclrtSynchronizeDevice``.  Scoped/fallback artifacts remain useful
diagnostics, but can never publish a speedup.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import types
from typing import Mapping

import numpy as np

from numpilot.backends.npu import NpuBackend
from numpilot.comparison import compare_outputs
from numpilot.module_injection import temporary_module
from numpilot.qualified_analysis import analyze_qualified_script


_DEFAULT_INDEX = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/day3-demo-migration-results-20260826.json"
)
_CASE_OUTPUTS = {
    "01": ("logits", "probs", "row_sums"),
    "02": ("signal", "abs_signal", "max_val"),
    "03": ("diff", "clipped", "result"),
    "04": ("I", "tanh_D", "det_val"),
    "06": ("powed", "logged", "expped", "eq_result"),
    "08": ("d_np", "f_np", "c_np", "fn_np", "cn_np"),
    "10": ("mod_result", "rem_result", "eq_check", "all_eq"),
}
_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_WORKER_RECEIPT = "worker.json"
_SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _require_sha(value, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} is not a SHA-256 digest")
    return value


def _module_assignments(source: str, filename: str) -> Counter:
    tree = ast.parse(source, filename=filename)
    assigned = Counter()
    for statement in tree.body:
        targets = []
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[target.id] += 1
            elif isinstance(target, (ast.Tuple, ast.List)):
                assigned.update(
                    item.id for item in target.elts if isinstance(item, ast.Name)
                )
    return assigned


def _attribute_path(node) -> tuple[str, ...] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def _asnumpy_call_counts(source: str, filename: str) -> dict[str, int]:
    tree = ast.parse(source, filename=filename)
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "asnumpy":
                    aliases.add(item.asname or "asnumpy")
    if not aliases:
        raise ValueError("migrated artifact has no direct asnumpy import")
    counts = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if path and path[0] in aliases and len(path) > 1:
            canonical = ".".join(path[1:])
            if canonical not in {"ndarray.from_numpy"}:
                counts[canonical] += 1
    return dict(sorted(counts.items()))


def _validate_case_entry(
    entry: Mapping,
    source_root: Path,
    *,
    runtime_identity_sha256: str | None = None,
) -> dict:
    case = str(entry.get("case", ""))
    if case not in _CASE_OUTPUTS:
        raise ValueError(f"unexpected migrated case: {case!r}")
    run_dir_value = entry.get("run_dir")
    if not isinstance(run_dir_value, str) or not Path(run_dir_value).is_absolute():
        raise ValueError(f"case {case} run_dir must be absolute")
    run_dir = Path(run_dir_value).resolve()
    source_path = (source_root / f"{case}_numpy.py").resolve()
    migrated_path = (run_dir / "migrated.py").resolve()
    report_path = (run_dir / "script-migration-report.json").resolve()
    for name, path in (
        ("source", source_path),
        ("migrated artifact", migrated_path),
        ("migration report", report_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"case {case} {name} is missing: {path}")

    hashes = {
        "source_sha256": _sha256(source_path),
        "migrated_sha256": _sha256(migrated_path),
        "report_sha256": _sha256(report_path),
    }
    for field, actual in hashes.items():
        expected = _require_sha(entry.get(field), f"case {case} index {field}")
        if actual != expected:
            raise ValueError(f"case {case} {field} does not match the Day-3 index")

    report = _json(report_path)
    if report.get("status") != entry.get("status"):
        raise ValueError(f"case {case} status differs between index and report")
    for field in ("source_sha256", "migrated_sha256"):
        if report.get(field) != hashes[field]:
            raise ValueError(f"case {case} {field} differs between artifact and report")

    source = source_path.read_text(encoding="utf-8")
    migrated = migrated_path.read_text(encoding="utf-8")
    outputs = _CASE_OUTPUTS[case]
    for label, program, path in (
        ("source", source, source_path),
        ("migrated", migrated, migrated_path),
    ):
        assignments = _module_assignments(program, str(path))
        missing = [name for name in outputs if assignments[name] != 1]
        if missing:
            raise ValueError(
                f"case {case} {label} output contract is not assigned exactly once: "
                + ", ".join(missing)
            )

    qualified = report.get("qualified_analysis")
    if not isinstance(qualified, dict):
        raise ValueError(f"case {case} report has no qualified analysis")
    calls = qualified.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"case {case} report has no qualified callsites")
    if (
        runtime_identity_sha256 is not None
        and qualified.get("runtime_identity_sha256") != runtime_identity_sha256
    ):
        raise ValueError(
            f"case {case} qualified runtime identity differs from the Day-3 index"
        )
    report_eligible = (
        qualified.get("decision") == "DIRECT"
        and qualified.get("runtime_identity_status") == "matched"
        and all(call.get("performance_eligible") is True for call in calls)
    )
    return {
        "case": case,
        "decision": entry.get("decision"),
        "migration_status": entry.get("status"),
        "source_path": str(source_path),
        "migrated_path": str(migrated_path),
        "report_path": str(report_path),
        **hashes,
        "output_names": list(outputs),
        "asnumpy_call_counts": _asnumpy_call_counts(migrated, str(migrated_path)),
        "report_qualification": {
            "decision": qualified.get("decision"),
            "runtime_identity_status": qualified.get("runtime_identity_status"),
            "runtime_identity_sha256": qualified.get("runtime_identity_sha256"),
            "all_calls_performance_eligible": all(
                call.get("performance_eligible") is True for call in calls
            ),
            "performance_eligible": report_eligible,
        },
    }


def _load_case_specs(index_path: Path) -> tuple[dict, list[dict]]:
    index_path = index_path.resolve()
    index = _json(index_path)
    source_root_value = index.get("source_root")
    if not isinstance(source_root_value, str) or not Path(source_root_value).is_absolute():
        raise ValueError("Day-3 source_root must be absolute")
    runtime_sha = _require_sha(
        index.get("runtime_identity_sha256"), "Day-3 runtime identity"
    )
    entries = index.get("cases")
    if not isinstance(entries, list):
        raise ValueError("Day-3 index cases must be a list")
    migrated = [entry for entry in entries if entry.get("migrated_sha256") is not None]
    observed = {str(entry.get("case", "")) for entry in migrated}
    expected = set(_CASE_OUTPUTS)
    if observed != expected or len(migrated) != len(expected):
        raise ValueError(
            "Day-3 migrated case set differs: missing=%r extra=%r"
            % (sorted(expected - observed), sorted(observed - expected))
        )
    specs = [
        _validate_case_entry(
            entry,
            Path(source_root_value).resolve(),
            runtime_identity_sha256=runtime_sha,
        )
        for entry in migrated
    ]
    specs.sort(key=lambda item: item["case"])
    binding = {
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "runtime_identity_sha256": runtime_sha,
        "source_root": str(Path(source_root_value).resolve()),
    }
    return binding, specs


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing_samples(run_once, *, warmup: int, repeats: int, synchronize=None) -> dict:
    def measure() -> int:
        start = time.perf_counter_ns()
        namespace = run_once()
        if synchronize is not None:
            synchronize()
        elapsed = time.perf_counter_ns() - start
        del namespace
        return elapsed

    warmup_ns = [measure() for _ in range(warmup)]
    samples_ns = [measure() for _ in range(repeats)]
    samples_ms = [value / 1e6 for value in samples_ns]
    return {
        "clock": "time.perf_counter_ns",
        "scope": (
            "steady whole-script exec; cached imports; includes allocations, operators, "
            "formatting/output-boundary copies; excludes compile, runtime initialization, "
            "numerical capture and object cleanup"
        ),
        "warmup_samples_ns": warmup_ns,
        "samples_ns": samples_ns,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p10_ms": _percentile(samples_ms, 0.10),
        "p90_ms": _percentile(samples_ms, 0.90),
    }


class _AclDeviceSynchronizer:
    def __init__(self):
        self._library = ctypes.CDLL("libascendcl.so")
        self._function = self._library.aclrtSynchronizeDevice
        self._function.argtypes = []
        self._function.restype = ctypes.c_int
        self.calls = 0

    @property
    def identity(self) -> str:
        return str(self._library._name)

    def __call__(self) -> None:
        result = int(self._function())
        if result != 0:
            raise RuntimeError(f"aclrtSynchronizeDevice failed with ACL error {result}")
        self.calls += 1


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


def _host_array(value) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object output is forbidden by the evidence format")
    return array


def _capture_outputs(namespace: Mapping, names: list[str]) -> dict[str, np.ndarray]:
    missing = [name for name in names if name not in namespace]
    if missing:
        raise RuntimeError("output globals are missing: " + ", ".join(missing))
    return {name: _host_array(namespace[name]) for name in names}


def _save_outputs(path: Path, outputs: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **outputs)
    path.chmod(0o600)


def _load_outputs(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(names):
            raise ValueError("NPZ output names do not match the declared contract")
        return {name: archive[name] for name in names}


def _output_metadata(outputs: Mapping[str, np.ndarray]) -> dict:
    result = {}
    for name, value in outputs.items():
        array = np.asarray(value)
        if array.dtype.kind in "fc":
            nan = int(np.count_nonzero(np.isnan(array)))
            positive_inf = int(np.count_nonzero(np.isposinf(array)))
            negative_inf = int(np.count_nonzero(np.isneginf(array)))
        else:
            nan = positive_inf = negative_inf = 0
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "elements": int(array.size),
            "nan": nan,
            "positive_inf": positive_inf,
            "negative_inf": negative_inf,
        }
    return result


def _execute(compiled, filename: str, sink) -> dict:
    namespace = {"__name__": "__main__", "__file__": filename}
    with redirect_stdout(sink), redirect_stderr(sink):
        exec(compiled, namespace, namespace)
    return namespace


def _flush_c_stdio() -> None:
    """Best-effort flush of native stdio while its descriptors are redirected."""

    try:
        library = ctypes.CDLL(None)
        function = library.fflush
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
        function(None)
    except (AttributeError, OSError):
        pass


@contextmanager
def _silence_process_output():
    """Redirect Python and native fd 1/2 output to ``/dev/null``."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    _flush_c_stdio()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        _flush_c_stdio()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(null_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


class _NestedRuntimeProxy:
    def __init__(self, root_module, prefix, targets, guard, counts, cache):
        self._root = root_module
        self._prefix = prefix
        self._targets = targets
        self._guard = guard
        self._counts = counts
        self._cache = cache

    def _resolve_original(self, path):
        value = self._root
        for part in path.split("."):
            value = getattr(value, part)
        return value

    def __getattr__(self, name):
        canonical = f"{self._prefix}.{name}" if self._prefix else name
        if canonical in self._targets:
            if canonical not in self._cache:
                target = self._targets[canonical]

                def traced(*args, **kwargs):
                    self._guard.validate_call(canonical, args, kwargs)
                    self._counts[canonical] += 1
                    return target(*args, **kwargs)

                traced.__name__ = canonical
                self._cache[canonical] = traced
            return self._cache[canonical]
        if any(key.startswith(canonical + ".") for key in self._targets):
            if canonical not in self._cache:
                self._cache[canonical] = _NestedRuntimeProxy(
                    self._root, canonical, self._targets,
                    self._guard, self._counts, self._cache,
                )
            return self._cache[canonical]
        return self._resolve_original(canonical)


def _guarded_module(module, guard, operation_names):
    targets = dict(guard.runtime_targets or {})
    if set(targets) != set(operation_names):
        raise RuntimeError("runtime guard target set differs from artifact operations")
    counts = {name: 0 for name in sorted(operation_names)}
    cache = {}

    class _RootProxy(types.ModuleType):
        def __getattr__(self, name):
            return _NestedRuntimeProxy(
                module, "", targets, guard, counts, cache
            ).__getattr__(name)

    proxy = _RootProxy(getattr(module, "__name__", "asnumpy"))
    for attr in ("__file__", "__package__", "__path__", "__spec__", "__version__"):
        if hasattr(module, attr):
            setattr(proxy, attr, getattr(module, attr))
    return proxy, counts


def _runtime_identity_summary(identity) -> dict:
    return {
        "status": identity.status,
        "identity_sha256": identity.identity_sha256,
        "audit_id": identity.audit_id,
        "asnumpy_version": identity.asnumpy_version,
        "issues": list(identity.issues),
    }


def _current_qualification(source: str, source_path: str, identity) -> dict:
    analysis = analyze_qualified_script(
        source, source_path, runtime_identity=identity
    ).to_dict()
    calls = analysis.get("calls", [])
    return {
        "decision": analysis.get("decision"),
        "runtime_identity_status": analysis.get("runtime_identity_status"),
        "runtime_identity_sha256": analysis.get("runtime_identity_sha256"),
        "all_calls_performance_eligible": bool(calls) and all(
            call.get("performance_eligible") is True for call in calls
        ),
        "calls": [
            {
                "canonical_key": call.get("canonical_key"),
                "decision": call.get("decision"),
                "effective_route": call.get("effective_route"),
                "performance_eligible": call.get("performance_eligible"),
            }
            for call in calls
        ],
        "issues": analysis.get("issues", []),
    }


def _worker_case(spec: dict, backend_name: str, sink, args, backend=None, sync=None):
    path = Path(spec["source_path"] if backend_name == "cpu" else spec["migrated_path"])
    program = path.read_text(encoding="utf-8")
    compiled = compile(program, str(path), "exec")
    guard_receipt = None
    current_qualification = None

    if backend_name == "npu":
        identity = backend.runtime_identity()
        source = Path(spec["source_path"]).read_text(encoding="utf-8")
        current_qualification = _current_qualification(
            source, spec["source_path"], identity
        )
        operation_names = tuple(spec["asnumpy_call_counts"])
        if (
            spec["report_qualification"]["all_calls_performance_eligible"]
            and current_qualification["all_calls_performance_eligible"]
        ):
            guard = backend.prepare_profile_guard(operation_names)
            proxy, runtime_counts = _guarded_module(
                backend.get_module(), guard, operation_names
            )
            with temporary_module("asnumpy", proxy):
                numerical_namespace = _execute(compiled, str(path), sink)
            expected_counts = dict(spec["asnumpy_call_counts"])
            if runtime_counts != expected_counts:
                raise RuntimeError(
                    f"case {spec['case']} guarded calls differ: "
                    f"expected={expected_counts}, observed={runtime_counts}"
                )
            guard_receipt = guard.to_dict()
            guard_receipt["runtime_call_counts"] = runtime_counts
        else:
            numerical_namespace = _execute(compiled, str(path), sink)
            guard_receipt = {
                "strict_npu": False,
                "runtime_profile_ok": False,
                "runtime_target_integrity_ok": False,
                "status": "not_applicable_to_noneligible_artifact",
                "runtime_call_counts": None,
            }
    else:
        numerical_namespace = _execute(compiled, str(path), sink)

    outputs = _capture_outputs(numerical_namespace, spec["output_names"])
    del numerical_namespace
    output_path = Path(args.worker_out) / f"case{spec['case']}-output.npz"
    _save_outputs(output_path, outputs)

    timing_rejected = None
    if backend_name == "npu" and spec["migration_status"] == "migrated_with_fallback":
        timing = None
        sync_count = 0
        timing_rejected = {
            "reason_code": "FALLBACK_NEGATIVE_CONTROL",
            "reason": (
                "NPU performance timing is forbidden for migrated_with_fallback; "
                "only the one-shot numerical verification was executed"
            ),
        }
    else:
        def run_once():
            return _execute(compiled, str(path), sink)

        sync_before = sync.calls if sync is not None else 0
        timing = _timing_samples(
            run_once,
            warmup=args.warmup,
            repeats=args.repeats,
            synchronize=sync,
        )
        sync_count = (sync.calls - sync_before) if sync is not None else 0
    result = {
        "case": spec["case"],
        "source_sha256": spec["source_sha256"],
        "migrated_sha256": spec["migrated_sha256"],
        "report_sha256": spec["report_sha256"],
        "output_names": list(spec["output_names"]),
        "output_sha256": _sha256(output_path),
        "output_metadata": _output_metadata(outputs),
        "timing": timing,
        "timing_rejected": timing_rejected,
        "synchronize_count": sync_count,
    }
    if backend_name == "npu":
        result["current_qualification"] = current_qualification
        result["profile_guard"] = guard_receipt
    return result


def _worker(args) -> dict:
    binding, specs = _load_case_specs(Path(args.index))
    affinity_before = sorted(os.sched_getaffinity(0))
    if affinity_before != [args.core]:
        raise RuntimeError(
            "worker affinity differs from requested core: "
            f"{affinity_before} != {[args.core]}"
        )
    environment = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    if any(value != "1" for value in environment.values()):
        raise RuntimeError("worker thread environment is not fixed to one")

    backend = sync = identity_before = None
    if args.worker == "npu":
        backend = NpuBackend()
        identity_before = backend.runtime_identity()
        sync = _AclDeviceSynchronizer()

    cases = []
    with _silence_process_output():
        with open(os.devnull, "w", encoding="utf-8") as sink:
            for spec in specs:
                cases.append(
                    _worker_case(
                        spec, args.worker, sink, args, backend=backend, sync=sync
                    )
                )
    affinity_after = sorted(os.sched_getaffinity(0))
    pools = _threadpool_info()
    receipt = {
        "schema_version": 1,
        "producer": f"numpilot-demo-script-{args.worker}-worker-v1",
        "status": "pass",
        "backend": args.worker,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "index_binding": binding,
        "affinity": affinity_after,
        "affinity_before": affinity_before,
        "affinity_after": affinity_after,
        "thread_limit_environment": environment,
        "threadpools": pools,
        "cases": cases,
    }
    if args.worker == "npu":
        identity_after = backend.runtime_identity()
        receipt["runtime_identity_before"] = _runtime_identity_summary(identity_before)
        receipt["runtime_identity_after"] = _runtime_identity_summary(identity_after)
        receipt["synchronize_api"] = "aclrtSynchronizeDevice"
        receipt["synchronize_library"] = sync.identity
    return receipt


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
        "contract": (
            "exact container/name, shape and dtype; dtype-derived atol+rtol; "
            "exact non-floating values and NaN/+Inf/-Inf pattern"
        ),
    }


def _blas_gate(receipt: dict) -> tuple[bool, str | None]:
    pools = [
        pool for pool in receipt.get("threadpools", [])
        if pool.get("user_api") == "blas"
    ]
    if not pools:
        return False, f"{receipt.get('backend')} BLAS thread count was not observed"
    if any(pool.get("num_threads") != 1 for pool in pools):
        return False, f"{receipt.get('backend')} BLAS runtime thread count is not one"
    return True, None


def _combine_case(spec: dict, cpu_case: dict, npu_case: dict, cpu: dict, npu: dict,
                  cpu_output: Mapping[str, np.ndarray],
                  npu_output: Mapping[str, np.ndarray]) -> dict:
    comparison = compare_outputs(cpu_output, npu_output)
    issues = []
    if cpu.get("pid") == npu.get("pid"):
        issues.append("CPU and NPU did not run in separate processes")
    affinities = [
        receipt.get(field)
        for receipt in (cpu, npu)
        for field in ("affinity_before", "affinity_after")
    ]
    if not affinities or any(value != affinities[0] for value in affinities) or len(
        affinities[0] or []
    ) != 1:
        issues.append("CPU/NPU affinity does not match one identical host core")
    for receipt in (cpu, npu):
        if any(
            receipt.get("thread_limit_environment", {}).get(name) != "1"
            for name in _THREAD_ENVIRONMENT
        ):
            issues.append(f"{receipt.get('backend')} thread environment gate failed")
        _, blas_issue = _blas_gate(receipt)
        if blas_issue:
            issues.append(blas_issue)
    if not comparison.consistent:
        issues.append("CPU/NPU numerical output comparison failed")

    strict_status = spec["migration_status"] == "migrated_strict_npu"
    if not strict_status:
        issues.append(
            f"migration status {spec['migration_status']} is diagnostic-only"
        )
    if spec["report_qualification"].get("performance_eligible") is not True:
        issues.append("Day-3 qualified report is not performance-eligible")

    qualification = npu_case.get("current_qualification", {})
    if (
        qualification.get("decision") != "DIRECT"
        or qualification.get("all_calls_performance_eligible") is not True
    ):
        issues.append("current qualified analysis is not performance-eligible")
    expected_runtime = npu["index_binding"]["runtime_identity_sha256"]
    if (
        qualification.get("runtime_identity_status") != "matched"
        or qualification.get("runtime_identity_sha256") != expected_runtime
    ):
        issues.append("current qualified analysis runtime identity does not match Day-3")
    for phase in ("before", "after"):
        identity = npu.get(f"runtime_identity_{phase}", {})
        if identity.get("status") != "matched" or identity.get("identity_sha256") != expected_runtime:
            issues.append(f"NPU runtime identity {phase} does not match Day-3")
    guard = npu_case.get("profile_guard", {})
    for field in ("strict_npu", "runtime_profile_ok", "runtime_target_integrity_ok"):
        if guard.get(field) is not True:
            issues.append(f"NPU runtime guard failed: {field}")
    npu_timing = npu_case.get("timing")
    fallback_negative_control = (
        spec["migration_status"] == "migrated_with_fallback"
    )
    if fallback_negative_control:
        if npu_timing is not None:
            issues.append("fallback negative control illegally contains NPU timing")
        rejection = npu_case.get("timing_rejected") or {}
        if rejection.get("reason_code") != "FALLBACK_NEGATIVE_CONTROL":
            issues.append("fallback negative control has no timing-rejection receipt")
    elif not isinstance(npu_timing, dict):
        issues.append("eligible/scoped NPU timing is missing")
    else:
        expected_sync = len(npu_timing["warmup_samples_ns"]) + len(
            npu_timing["samples_ns"]
        )
        if npu_case.get("synchronize_count") != expected_sync:
            issues.append("not every NPU timing sample ended with device synchronization")

    cpu_timing = cpu_case.get("timing")
    cpu_median = (
        float(cpu_timing["median_ms"])
        if isinstance(cpu_timing, dict)
        else None
    )
    npu_median = (
        float(npu_timing["median_ms"])
        if isinstance(npu_timing, dict)
        else None
    )
    if cpu_median is None or cpu_median <= 0:
        issues.append("CPU median timing is missing or not positive")
    if not fallback_negative_control and (npu_median is None or npu_median <= 0):
        issues.append("NPU median timing is missing or not positive")
    diagnostic_ratio = (
        cpu_median / npu_median
        if cpu_median is not None and cpu_median > 0
        and npu_median is not None and npu_median > 0
        else None
    )
    performance_valid = not issues
    return {
        "case": spec["case"],
        "migration_status": spec["migration_status"],
        "decision": spec["decision"],
        "binding": {
            key: spec[key]
            for key in ("source_sha256", "migrated_sha256", "report_sha256")
        },
        "output_names": list(spec["output_names"]),
        "comparison": _comparison_receipt(comparison),
        "cpu": cpu_case,
        "npu": npu_case,
        "performance": {
            "measured": True,
            "performance_valid": performance_valid,
            "issues": issues,
            "cpu_median_ms": cpu_median,
            "npu_median_ms": npu_median,
            "diagnostic_cpu_time_over_npu_time": diagnostic_ratio,
            "speedup": diagnostic_ratio if performance_valid else None,
            "winner": (
                "npu" if performance_valid and diagnostic_ratio is not None
                and diagnostic_ratio > 1
                else "cpu" if performance_valid and diagnostic_ratio is not None
                and diagnostic_ratio < 1
                else "tie" if performance_valid and diagnostic_ratio == 1
                else None
            ),
        },
    }


def _combine(index_path: Path, cpu_dir: Path, npu_dir: Path) -> dict:
    binding, specs = _load_case_specs(index_path)
    cpu = _json(cpu_dir / _WORKER_RECEIPT)
    npu = _json(npu_dir / _WORKER_RECEIPT)
    if cpu.get("status") != "pass" or npu.get("status") != "pass":
        raise RuntimeError("one or more workers failed")
    runner_sha = _sha256(Path(__file__).resolve())
    if cpu.get("runner_sha256") != runner_sha or npu.get("runner_sha256") != runner_sha:
        raise RuntimeError("worker runner identity differs from parent")
    if cpu.get("index_binding") != binding or npu.get("index_binding") != binding:
        raise RuntimeError("worker Day-3 index binding differs from parent")
    cpu_cases = {item["case"]: item for item in cpu.get("cases", [])}
    npu_cases = {item["case"]: item for item in npu.get("cases", [])}
    if set(cpu_cases) != set(_CASE_OUTPUTS) or set(npu_cases) != set(_CASE_OUTPUTS):
        raise RuntimeError("worker case receipts do not match the seven-case contract")

    cases = []
    for spec in specs:
        case = spec["case"]
        for worker_name, worker_case, worker_dir in (
            ("cpu", cpu_cases[case], cpu_dir),
            ("npu", npu_cases[case], npu_dir),
        ):
            for field in ("source_sha256", "migrated_sha256", "report_sha256"):
                if worker_case.get(field) != spec[field]:
                    raise RuntimeError(
                        f"case {case} {worker_name} {field} binding differs"
                    )
            output_path = worker_dir / f"case{case}-output.npz"
            if worker_case.get("output_sha256") != _sha256(output_path):
                raise RuntimeError(
                    f"case {case} {worker_name} output hash differs"
                )
        cpu_output = _load_outputs(
            cpu_dir / f"case{case}-output.npz", spec["output_names"]
        )
        npu_output = _load_outputs(
            npu_dir / f"case{case}-output.npz", spec["output_names"]
        )
        cases.append(_combine_case(
            spec, cpu_cases[case], npu_cases[case], cpu, npu,
            cpu_output, npu_output,
        ))
    valid = sum(item["performance"]["performance_valid"] for item in cases)
    consistent = sum(item["comparison"]["consistent"] for item in cases)
    return {
        "schema_version": 1,
        "producer": "numpilot-demo-script-performance-evidence-v1",
        "status": "completed",
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "index_binding": binding,
        "method": {
            "worker_isolation": "separate CPU and NPU processes",
            "affinity": "same single taskset core before Python startup",
            "threads": 1,
            "stdout_parsed": False,
            "warmup_samples": len(cases[0]["cpu"]["timing"]["warmup_samples_ns"]),
            "measurement_samples": len(cases[0]["cpu"]["timing"]["samples_ns"]),
            "timing_scope": cases[0]["cpu"]["timing"]["scope"],
            "npu_completion": "explicit aclrtSynchronizeDevice after every sample",
            "fallback_timing_policy": (
                "migrated_with_fallback runs one-shot NPU numerical verification only; "
                "NPU warmup/measurement is rejected before timing"
            ),
        },
        "summary": {
            "cases": len(cases),
            "numerically_consistent": consistent,
            "numerically_inconsistent": len(cases) - consistent,
            "performance_valid": valid,
            "diagnostic_only": len(cases) - valid,
        },
        "workers": {"cpu": cpu, "npu": npu},
        "cases": cases,
    }


def _write_json_exclusive(path: Path, value: dict) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)


def _run_worker(args) -> int:
    output = Path(args.worker_out).resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        receipt = _worker(args)
    except Exception as error:
        receipt = {
            "schema_version": 1,
            "producer": f"numpilot-demo-script-{args.worker}-worker-v1",
            "status": "fail",
            "backend": args.worker,
            "pid": os.getpid(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
    _write_json_exclusive(output / _WORKER_RECEIPT, receipt)
    return 0 if receipt["status"] == "pass" else 2


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _THREAD_ENVIRONMENT:
        environment[name] = "1"
    return environment


def _worker_command(args, worker: str, output: Path, taskset: str) -> list[str]:
    return [
        taskset, "--cpu-list", str(args.core),
        sys.executable, str(Path(__file__).resolve()),
        "--worker", worker,
        "--worker-out", str(output),
        "--index", str(Path(args.index).resolve()),
        "--core", str(args.core),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
    ]


def _write_log(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)


def _run_parent(args) -> int:
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError(f"--out must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("taskset is required for CPU/NPU affinity parity")
    available = sorted(os.sched_getaffinity(0))
    if args.core not in available:
        raise ValueError(f"--core {args.core} is outside parent affinity {available}")
    _load_case_specs(Path(args.index))

    staging = Path(tempfile.mkdtemp(prefix=".demo-performance-", dir=output.parent))
    try:
        failures = []
        for worker in ("cpu", "npu"):
            worker_out = staging / worker
            completed = subprocess.run(
                _worker_command(args, worker, worker_out, taskset),
                cwd=str(Path(__file__).resolve().parents[1]),
                env=_worker_environment(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            _write_log(staging / f"{worker}.stdout.log", completed.stdout)
            _write_log(staging / f"{worker}.stderr.log", completed.stderr)
            if completed.returncode != 0:
                failures.append(f"{worker} worker exited {completed.returncode}")
            if not (worker_out / _WORKER_RECEIPT).is_file():
                failures.append(f"{worker} worker produced no receipt")
        if failures:
            raise RuntimeError("; ".join(failures))
        receipt = _combine(Path(args.index), staging / "cpu", staging / "npu")
        exit_code = 0
    except Exception as error:
        receipt = {
            "schema_version": 1,
            "producer": "numpilot-demo-script-performance-evidence-v1",
            "status": "fail",
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "summary": None,
        }
        exit_code = 2
    _write_json_exclusive(staging / "evidence.json", receipt)
    staging.rename(output)
    summary = receipt.get("summary") or {}
    print(
        f"status={receipt['status']} cases={summary.get('cases', 0)} "
        f"performance_valid={summary.get('performance_valid', 0)} out={output}"
    )
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(_DEFAULT_INDEX))
    parser.add_argument("--out")
    parser.add_argument("--core", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--worker", choices=("cpu", "npu"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.core is None:
        affinity = sorted(os.sched_getaffinity(0))
        if not affinity:
            parser.error("current process has no available CPU core")
        args.core = affinity[0]
    if args.core < 0:
        parser.error("--core must be non-negative")
    if args.warmup < 2:
        parser.error("--warmup must be >= 2")
    if args.repeats < 7:
        parser.error("--repeats must be >= 7")
    if args.worker:
        if not args.worker_out:
            parser.error("--worker-out is required for an internal worker")
        return _run_worker(args)
    if not args.out:
        parser.error("--out is required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
