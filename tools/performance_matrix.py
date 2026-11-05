#!/usr/bin/env python3
"""Run the Phase 9 single-core CPU versus single-NPU performance matrix.

The parent process only orchestrates.  Every measurement happens in a separate
worker process so CPU affinity, thread environment and imports are controlled
before NumPy or AsNumpy is loaded.

Examples
--------
Run the full diagnostic matrix::

    python tools/performance_matrix.py \
        --out runs/phase9-performance-20261105/matrix-01

Run one profile and one size::

    python tools/performance_matrix.py --profiles matmul --sizes 512 \
        --out runs/phase9-performance-20261105/smoke
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODES = (
    "cpu_warm",
    "cpu_cold",
    "npu_resident",
    "npu_warm_e2e",
    "npu_cold_e2e",
)
_COLD_MODES = {"cpu_cold", "npu_cold_e2e"}
_SENSITIVE_ENV = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY", "PRIVATE")


class MatrixWorkerError(RuntimeError):
    """A worker process failed or produced an invalid receipt."""


def _pin_core(core: int) -> None:
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("sched_setaffinity is unavailable on this platform")
    os.sched_setaffinity(0, {core})


def _thread_environment() -> dict:
    return {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }


def _blas_threads():
    try:
        from threadpoolctl import threadpool_info

        return [
            {
                key: item.get(key)
                for key in ("internal_api", "num_threads", "prefix")
            }
            for item in threadpool_info()
        ]
    except Exception:  # pragma: no cover - optional diagnostic dependency
        return None


def _worker_environment(repo_root: Path) -> dict:
    """Return a sanitized environment with one-thread BLAS settings."""

    environment = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(marker in upper for marker in _SENSITIVE_ENV):
            continue
        environment[key] = value
    existing = environment.get("PYTHONPATH")
    paths = [str(repo_root)]
    if existing:
        paths.append(existing)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(paths),
        }
    )
    return environment


class _DeviceSynchronizer:
    """Minimal, auditable ACL device probe and synchronize wrapper."""

    def __init__(self):
        self._library = ctypes.CDLL("libascendcl.so")
        self._function = self._library.aclrtSynchronizeDevice
        self._function.argtypes = []
        self._function.restype = ctypes.c_int
        self._get_device = getattr(self._library, "aclrtGetDevice", None)
        if self._get_device is not None:
            self._get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
            self._get_device.restype = ctypes.c_int
        self._soc_name = getattr(self._library, "aclrtGetSocName", None)
        if self._soc_name is not None:
            self._soc_name.argtypes = []
            self._soc_name.restype = ctypes.c_char_p

    @property
    def identity(self) -> str:
        return Path(str(self._library._name)).name

    def device(self) -> dict:
        record = {"device_index": None, "soc_name": None}
        if self._get_device is not None:
            value = ctypes.c_int(-1)
            if int(self._get_device(ctypes.byref(value))) == 0:
                record["device_index"] = int(value.value)
        if self._soc_name is not None:
            try:
                raw = self._soc_name()
            except Exception:
                raw = None
            if raw:
                record["soc_name"] = raw.decode("utf-8", "replace")
        return record

    def __call__(self) -> None:
        result = int(self._function())
        if result != 0:
            raise RuntimeError(
                f"aclrtSynchronizeDevice failed with ACL error {result}"
            )


def _receipt_path(out_dir: Path) -> Path:
    return out_dir / "receipt.json"


def _write_receipt(out_dir: Path, payload: dict) -> None:
    from numpilot import perf_matrix as matrix

    matrix.write_json(_receipt_path(out_dir), payload)


def _worker_common(profile: str, size: int, mode: str) -> dict:
    from numpilot import perf_matrix as matrix

    return {
        "schema": matrix.WORKER_SCHEMA,
        "schema_version": 1,
        "producer": matrix.WORKER_PRODUCER,
        "status": "ok",
        "mode": mode,
        "profile": profile,
        "size": size,
        "shape": [size, size],
        "dtype": "float32",
        "pid": os.getpid(),
        "python_version": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "environment": {
            "cpu_model": matrix.cpu_model(),
            "logical_cpus": os.cpu_count(),
            "thread_environment": _thread_environment(),
        },
        "blas_threads": _blas_threads(),
    }


def _run_cpu_worker(args) -> int:
    _pin_core(args.core)
    import numpy as np

    from numpilot import perf_matrix as matrix

    inputs = matrix.build_inputs(args.profile, args.size)
    forward = lambda: matrix.numpy_forward(args.profile, inputs)

    timing, output = matrix.timed_call(
        forward,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    out_dir = Path(args.worker_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_receipt = matrix.write_output(out_dir / "output.npy", np.asarray(output))

    receipt = _worker_common(args.profile, args.size, args.mode)
    receipt["timing"] = timing
    receipt["output"] = output_receipt
    receipt["work"] = {
        "scope": "host_only",
        "host_to_device_count": 0,
        "device_to_host_count": 0,
        "synchronize_api": None,
    }
    _write_receipt(out_dir, receipt)
    return 0


def _check_trace(expected: dict, observed: dict) -> dict:
    names = sorted(set(expected) | set(observed))
    mismatched = {
        name: {"expected": expected.get(name, 0), "observed": observed.get(name, 0)}
        for name in names
        if expected.get(name, 0) != observed.get(name, 0)
    }
    return {
        "expected": expected,
        "observed": observed,
        "ok": not mismatched,
        "mismatched": mismatched,
    }


def _run_npu_worker(args) -> int:
    # Pin before importing NumPy/AsNumpy so the host side of the NPU path is
    # compared on the same single logical core as the CPU baseline.
    _pin_core(args.core)
    import numpy as np

    from numpilot import perf_matrix as matrix
    from numpilot.backends.npu import NpuBackend
    from numpilot.module_injection import traced_module

    backend = NpuBackend()
    module = backend.get_module()
    identity = matrix.sanitize_identity(backend.runtime_identity())
    manifest = backend.capability_manifest(list(matrix.required_ops(args.profile)))
    provenance = matrix.provenance_summary(manifest, args.profile)

    inputs = matrix.build_inputs(args.profile, args.size)
    host_a = inputs["a"]
    host_b = inputs["b"]
    out_dir = Path(args.worker_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = matrix.expected_calls(args.profile)
    transfers = {
        "host_to_device_count": 0,
        "host_to_device_bytes": 0,
        "device_to_host_count": 0,
        "device_to_host_bytes": 0,
    }

    def upload():
        da = backend.to_device(host_a)
        db = backend.to_device(host_b)
        transfers["host_to_device_count"] += 2
        transfers["host_to_device_bytes"] += int(host_a.nbytes + host_b.nbytes)
        return da, db

    def download(value):
        host = backend.to_host(value)
        transfers["device_to_host_count"] += 1
        transfers["device_to_host_bytes"] += int(np.asarray(host).nbytes)
        return np.asarray(host)

    def snapshot():
        return dict(transfers)

    def reset():
        for key in transfers:
            transfers[key] = 0

    synchronizer = _DeviceSynchronizer()
    trace = {
        "performed": False,
        "expected": expected,
        "observed": {},
        "ok": False,
    }
    timing = None

    setup_transfers = dict(transfers)
    verification_transfers = dict(transfers)
    if args.mode == "npu_resident":
        da, db = upload()
        setup_transfers = snapshot()
        reset()
        traced, counts = traced_module(module, expected)
        verification = matrix.module_forward(traced, args.profile, da, db)
        synchronizer()
        host_verification = download(verification)
        verification_transfers = snapshot()
        reset()
        trace = {
            "performed": True,
            **_check_trace(expected, dict(counts)),
            "verification_max_abs": float(
                np.max(np.abs(host_verification))
            ),
        }
        forward = lambda: matrix.module_forward(module, args.profile, da, db)
        timing, output = matrix.timed_call(
            forward,
            warmup=args.warmup,
            repeats=args.repeats,
            synchronize=synchronizer,
        )
        output = download(output)
    elif args.mode == "npu_warm_e2e":
        traced, counts = traced_module(module, expected)
        da, db = upload()
        verification = matrix.module_forward(traced, args.profile, da, db)
        synchronizer()
        host_verification = download(verification)
        verification_transfers = snapshot()
        reset()
        trace = {
            "performed": True,
            **_check_trace(expected, dict(counts)),
            "verification_max_abs": float(
                np.max(np.abs(host_verification))
            ),
        }

        def forward():
            da, db = upload()
            value = matrix.module_forward(module, args.profile, da, db)
            return download(value)

        timing, output = matrix.timed_call(
            forward,
            warmup=args.warmup,
            repeats=args.repeats,
            synchronize=synchronizer,
        )
    elif args.mode == "npu_cold_e2e":
        start = time.perf_counter()
        da, db = upload()
        value = matrix.module_forward(module, args.profile, da, db)
        synchronizer()
        output = download(value)
        internal_ms = (time.perf_counter() - start) * 1e3
        timing = matrix.summarize_samples([internal_ms])
        trace = {
            "performed": False,
            "expected": expected,
            "observed": {},
            "ok": False,
            "reason": "cold timing keeps the process free of tracing instrumentation",
        }
    else:
        raise MatrixWorkerError(f"unsupported worker mode: {args.mode}")

    measured_transfers = snapshot()
    device_record = synchronizer.device()
    output_receipt = matrix.write_output(out_dir / "output.npy", output)
    receipt = _worker_common(args.profile, args.size, args.mode)
    receipt["timing"] = timing
    receipt["output"] = output_receipt
    receipt["work"] = {
        "scope": args.mode,
        "setup_transfers": setup_transfers,
        "verification_transfers": verification_transfers,
        "measured_transfers": measured_transfers,
        "synchronize_api": "aclrtSynchronizeDevice",
        "synchronize_library": synchronizer.identity,
        "runtime_calls": trace,
    }
    receipt["runtime"] = {
        "identity": identity,
        "provenance": {**provenance, "identity": identity, "device": device_record},
        "asnumpy_version": identity.get("asnumpy_version"),
        "device": device_record,
        "synchronize_api": "aclrtSynchronizeDevice",
        "synchronize_library": synchronizer.identity,
        "kernel_evidence": {
            name: {
                "provenance": item["provenance"],
                "evidence": item["evidence"],
            }
            for name, item in provenance["operators"].items()
        },
    }
    _write_receipt(out_dir, receipt)
    return 0


def _worker_main(args) -> int:
    if args.mode in {"cpu_warm", "cpu_cold"}:
        return _run_cpu_worker(args)
    if args.mode in {"npu_resident", "npu_warm_e2e", "npu_cold_e2e"}:
        return _run_npu_worker(args)
    raise MatrixWorkerError(f"unsupported worker mode: {args.mode}")


_PRIVATE_HOME = re.compile(r"/home/[A-Za-z0-9._-]+")


def _sanitize_text(value: str) -> str:
    return _PRIVATE_HOME.sub("/home/<user>", value)


def _invoke_worker(args, *, profile, size, mode, out_dir, warmup, repeats) -> tuple:
    from numpilot import perf_matrix as matrix

    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--profile",
        profile,
        "--size",
        str(size),
        "--worker-out",
        str(out_dir),
        "--core",
        str(args.core),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(args.repo_root),
        env=_worker_environment(Path(args.repo_root)),
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    wall_ms = (time.perf_counter() - start) * 1e3
    if completed.returncode != 0:
        detail = _sanitize_text((completed.stderr or completed.stdout or "")[-2000:])
        raise MatrixWorkerError(
            f"{profile}/{size}/{mode} exited {completed.returncode}: {detail}"
        )
    receipt = json.loads(_receipt_path(out_dir).read_text(encoding="utf-8"))
    if receipt.get("status") != "ok":
        raise MatrixWorkerError(f"{profile}/{size}/{mode} receipt is not ok")
    return receipt, wall_ms


def _run_row(args, profile: str, size: int) -> dict:
    from numpilot import perf_matrix as matrix

    row_dir = Path(args.out) / "workers" / profile / str(size)
    reference_inputs = matrix.build_inputs(profile, size)
    reference64 = matrix.float64_forward(profile, reference_inputs)
    max_abs_limit = matrix.gemm_max_abs_limit(profile, reference_inputs)

    measurements = {}
    checks = []
    provenance = None
    cpu_relative = None

    for mode in MODES:
        if mode in _COLD_MODES:
            walls = []
            internal = []
            for index in range(args.cold_samples):
                sample_dir = row_dir / mode / f"{index:02d}"
                receipt, wall_ms = _invoke_worker(
                    args,
                    profile=profile,
                    size=size,
                    mode=mode,
                    out_dir=sample_dir,
                    warmup=0,
                    repeats=1,
                )
                walls.append(wall_ms)
                internal.append(receipt["timing"]["median_ms"])
                check = matrix.gemm_check(
                    reference64,
                    sample_dir / "output.npy",
                    max_abs_limit=max_abs_limit,
                    cpu_relative_frobenius=(
                        cpu_relative if mode.startswith("npu") else None
                    ),
                )
                checks.append({"mode": mode, "sample": index, **check})
            summary = matrix.summarize_samples(walls)
            summary["scope"] = "fresh_process_wall_clock"
            summary["internal_forward_samples_ms"] = internal
            measurements[mode] = summary
        else:
            sample_dir = row_dir / mode
            warmup, repeats = _mode_budget(args, mode)
            receipt, wall_ms = _invoke_worker(
                args,
                profile=profile,
                size=size,
                mode=mode,
                out_dir=sample_dir,
                warmup=warmup,
                repeats=repeats,
            )
            measurements[mode] = receipt["timing"]
            measurements[mode]["process_wall_ms"] = wall_ms
            check = matrix.gemm_check(
                reference64,
                sample_dir / "output.npy",
                max_abs_limit=max_abs_limit,
                cpu_relative_frobenius=(
                    cpu_relative if mode.startswith("npu") else None
                ),
            )
            checks.append({"mode": mode, "sample": None, **check})
            if mode == "cpu_warm":
                cpu_relative = check["relative_frobenius"]

        if mode.startswith("npu"):
            current = receipt["runtime"]["provenance"]
            if provenance is None:
                provenance = current
            elif current != provenance:
                raise MatrixWorkerError(
                    "NPU provenance changed between matrix rows"
                )

    if provenance is None:
        raise MatrixWorkerError("no NPU worker receipt was collected")

    row = matrix.assemble_row(
        profile=profile,
        size=size,
        measurements=measurements,
        checks=checks,
        provenance=provenance,
        seed=matrix.default_seed(profile, size),
    )
    row["provenance"] = provenance
    return row


def _mode_budget(args, mode: str) -> tuple:
    if mode.startswith("cpu"):
        return args.cpu_warmup, args.cpu_repeats
    return args.npu_warmup, args.npu_repeats


def _format_ms(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_ratio(value, publishable: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x" + ("*" if publishable else "")


def _render_markdown(matrix: dict) -> str:
    lines = [
        "# NumPilot Phase 9 performance matrix",
        "",
        f"- status: `{matrix['status']}`",
        f"- performance_valid: `{str(matrix['performance_valid']).lower()}`",
        f"- correctness_ok: `{str(matrix['correctness_ok']).lower()}`",
        f"- provenance_ok: `{str(matrix['provenance_ok']).lower()}`",
        f"- generated: `{matrix['generated_at']}`",
        f"- tool sha256: `{(matrix.get('tool') or {}).get('sha256')}`",
        f"- library sha256: "
        f"`{((matrix.get('tool') or {}).get('library') or {}).get('sha256')}`",
        f"- repo head: `{(matrix.get('tool') or {}).get('repo_head')}`",
        "",
        "## Environment",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    environment = matrix["environment"]
    for key in ("platform", "machine", "logical_cpus", "cpu_model"):
        lines.append(f"| {key} | {environment.get(key)} |")
    thread = environment.get("thread_environment") or {}
    lines.append(
        "| thread environment | "
        + ", ".join(f"{key}={thread.get(key)}" for key in sorted(thread))
        + " |"
    )
    runtime = environment.get("runtime") or {}
    device = environment.get("device") or {}
    lines.extend(
        [
            f"| asnumpy version | {runtime.get('asnumpy_version')} |",
            f"| audit id | {runtime.get('audit_id')} |",
            f"| runtime status | {runtime.get('status')} |",
            f"| CANN version | {runtime.get('cann_version')} |",
            f"| SoC | {runtime.get('soc_model')} |",
            f"| device index | {device.get('device_index')} |",
            f"| device SoC name | {device.get('soc_name')} |",
        ]
    )
    lines.extend(["", "## Measurements", ""])
    for profile in matrix["profiles"]:
        lines.append(f"### {profile['profile']}")
        lines.append("")
        lines.append(
            "| size | CPU warm | NPU resident | NPU warm E2E | CPU cold | "
            "NPU cold E2E | resident ratio | warm ratio | cold ratio | label |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in profile["rows"]:
            measurements = row["measurements"]
            ratios = row.get("ratios") or {}
            lines.append(
                "| {size} | {cpu} | {resident} | {warm} | {cpu_cold} | {cold} | "
                "{resident_ratio} | {warm_ratio} | {cold_ratio} | {label} |".format(
                    size=row["size"],
                    cpu=_format_ms(measurements["cpu_warm"]["median_ms"]),
                    resident=_format_ms(
                        measurements["npu_resident"]["median_ms"]
                    ),
                    warm=_format_ms(measurements["npu_warm_e2e"]["median_ms"]),
                    cpu_cold=_format_ms(measurements["cpu_cold"]["median_ms"]),
                    cold=_format_ms(
                        measurements["npu_cold_e2e"]["median_ms"]
                    ),
                    resident_ratio=_format_ratio(
                        (ratios.get("resident") or {}).get("median_ratio"),
                        (ratios.get("resident") or {}).get("publishable", False),
                    ),
                    warm_ratio=_format_ratio(
                        (ratios.get("warm_end_to_end") or {}).get("median_ratio"),
                        (ratios.get("warm_end_to_end") or {}).get("publishable", False),
                    ),
                    cold_ratio=_format_ratio(
                        (ratios.get("cold_end_to_end") or {}).get("median_ratio"),
                        (ratios.get("cold_end_to_end") or {}).get("publishable", False),
                    ),
                    label=row["label"],
                )
            )
        lines.append("")
        lines.append("Crossover (median ratio >= 1.0):")
        lines.append("")
        for key, value in profile["crossover"].items():
            lines.append(
                f"- `{key}`: first size {value['first_size_at_or_above']}, "
                f"CPU-preferred {value['cpu_preferred_sizes']}, "
                f"NPU-preferred {value['npu_preferred_sizes']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Method",
            "",
            "- CPU workers are pinned to one logical core before NumPy import; "
            "OMP/OpenBLAS/MKL/NumExpr are fixed to one thread.",
            "- `cpu_warm` repeats the host forward; `cpu_cold` uses a fresh "
            "process per sample and reports parent wall clock.",
            "- `npu_resident` uploads once, repeats compute and an explicit "
            "`aclrtSynchronizeDevice`, and downloads the final output once.",
            "- `npu_warm_e2e` performs H2D, compute and D2H in every warm sample.",
            "- `npu_cold_e2e` uses a fresh process per sample and reports parent "
            "wall clock, including interpreter startup and runtime import.",
            "- Every saved output is checked against the NumPy reference before "
            "any ratio is computed.",
            "- A `*` marks a ratio that passes the public-claim gates: median "
            "threshold, conservative CPU p10 / NPU p90 >= 1.1, bootstrap 95% "
            "lower bound > 1.0 and strict `native_npu` provenance.",
            "",
            "## Boundary",
            "",
            "- A speedup is publishable only when every executed API has "
            "`native_npu` provenance and the runtime identity is audited.",
            "- When provenance is unknown, the matrix is `profile_required`; "
            "ratios are diagnostic and must not be quoted as an NPU claim.",
            "- Numbers are valid only for this host, CANN, AsNumpy wheel, "
            "dtype, shape, thread policy and sample protocol.",
            "",
        ]
    )
    issues = matrix.get("issues") or []
    if issues:
        lines.extend(["## Issues", ""])
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")
    return "\n".join(lines)


def _run_matrix(args) -> int:
    from numpilot import perf_matrix as matrix

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = [matrix.validate_profile(profile) for profile in args.profiles]
    sizes = [matrix.validate_size(size) for size in args.sizes]
    if not profiles or not sizes:
        raise ValueError("at least one profile and one size are required")
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity control is unavailable on this platform")
    if args.core is None:
        args.core = min(os.sched_getaffinity(0))
    if args.core < 0:
        raise ValueError("core must be non-negative")
    if args.cpu_warmup < 0 or args.npu_warmup < 0:
        raise ValueError("warmup counts must be non-negative")
    if args.cpu_repeats < 1 or args.npu_repeats < 1:
        raise ValueError("repeats must be at least one")
    if args.cold_samples < 1:
        raise ValueError("cold samples must be at least one")
    if args.timeout < 1:
        raise ValueError("timeout must be at least one second")

    rows_by_profile = {}
    all_rows = []
    provenances = {}
    errors = []

    for profile in profiles:
        rows = []
        for size in sizes:
            try:
                row = _run_row(args, profile, size)
            except Exception as exc:  # fail closed, keep the receipt
                errors.append(
                    f"{profile}/{size}: {type(exc).__name__}: "
                    + _sanitize_text(str(exc))
                )
                continue
            rows.append(row)
            all_rows.append(row)
            current = row.get("provenance") or {}
            if profile not in provenances:
                provenances[profile] = current
            elif current != provenances[profile]:
                errors.append(
                    f"{profile}/{size}: provenance changed between rows"
                )
        rows_by_profile[profile] = rows

    correctness_ok = bool(all_rows) and all(
        row["correctness"]["ok"] for row in all_rows
    )
    provenance_ok = bool(provenances) and all(
        bool(value.get("native_npu")) for value in provenances.values()
    )
    expected_rows = len(profiles) * len(sizes)
    complete = len(all_rows) == expected_rows
    performance_valid = correctness_ok and provenance_ok and complete

    for profile, value in provenances.items():
        if value.get("native_npu"):
            continue
        unknown = [
            f"{name}={item['provenance']}"
            for name, item in value["operators"].items()
            if item["provenance"] != "native_npu"
        ]
        errors.append(
            f"{profile}: runtime provenance is not strict for: "
            + ", ".join(sorted(unknown))
        )
    for row in all_rows:
        if not row["correctness"]["ok"]:
            failed = [
                f"{check['mode']}"
                + (f"[{check['sample']}]" if check.get("sample") is not None else "")
                for check in row["correctness"]["checks"]
                if not check["consistent"]
            ]
            errors.append(
                f"{row['profile']}/{row['size']} correctness failed: "
                + ", ".join(failed)
            )

    profile_payload = []
    for profile in profiles:
        rows = rows_by_profile.get(profile, [])
        crossovers = {}
        if rows:
            for key in ("resident", "warm_end_to_end", "cold_end_to_end"):
                crossovers[key] = matrix.crossover(rows, key)
        profile_payload.append(
            {
                "profile": profile,
                "rows": rows,
                "crossover": crossovers,
            }
        )

    representative = next(iter(provenances.values()), {})
    environment = matrix.host_environment()
    environment["runtime"] = _runtime_environment(representative)
    environment["device"] = _device_environment(representative)

    matrix_payload = {
        "schema": matrix.MATRIX_SCHEMA,
        "schema_version": 2,
        "producer": matrix.MATRIX_PRODUCER,
        "generated_at": matrix._now_iso(),
        "tool": _tool_identity(),
        "status": matrix.matrix_status(
            correctness_ok=correctness_ok,
            performance_valid=performance_valid,
        ),
        "performance_valid": performance_valid,
        "correctness_ok": correctness_ok,
        "provenance_ok": provenance_ok,
        "policy": dict(matrix.POLICY),
        "environment": environment,
        "provenance": provenances,
        "profiles": profile_payload,
        "issues": errors,
        "sample_protocol": {
            "cpu_warmup": args.cpu_warmup,
            "cpu_repeats": args.cpu_repeats,
            "npu_warmup": args.npu_warmup,
            "npu_repeats": args.npu_repeats,
            "cold_samples": args.cold_samples,
        },
    }

    markdown_path = out_dir / "PERF-MATRIX.md"
    markdown_path.write_text(_render_markdown(matrix_payload), encoding="utf-8")
    matrix_payload["artifacts"] = {
        "markdown": markdown_path.name,
        "markdown_sha256": matrix.sha256_file(markdown_path),
    }
    json_path = out_dir / "perf-matrix.json"
    matrix.write_json(json_path, matrix_payload)

    sums = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{matrix.sha256_file(path)}  {path.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    print(
        f"[matrix] status={matrix_payload['status']} "
        f"correctness={correctness_ok} provenance={provenance_ok} "
        f"performance_valid={performance_valid}"
    )
    for profile in profile_payload:
        for row in profile["rows"]:
            ratios = row.get("ratios") or {}
            print(
                f"  {profile['profile']:11s} {row['size']:4d} "
                f"label={row['label']:18s} "
                f"resident={_format_ratio((ratios.get('resident') or {}).get('median_ratio'))} "
                f"warm={_format_ratio((ratios.get('warm_end_to_end') or {}).get('median_ratio'))} "
                f"cold={_format_ratio((ratios.get('cold_end_to_end') or {}).get('median_ratio'))}"
            )
    for issue in errors:
        print(f"[issue] {issue}")
    print(f"[evidence] {json_path}")
    print(f"[evidence] {markdown_path}")

    if not correctness_ok or not complete:
        return 1
    if args.require_performance_valid and not performance_valid:
        return 1
    return 0


def _tool_identity() -> dict:
    from numpilot import perf_matrix as matrix

    path = Path(__file__).resolve()
    head = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            head = completed.stdout.strip() or None
    except Exception:
        head = None
    library = REPO_ROOT / "numpilot" / "perf_matrix.py"
    return {
        "name": "tools/performance_matrix.py",
        "sha256": matrix.sha256_file(path),
        "library": {
            "name": "numpilot/perf_matrix.py",
            "sha256": matrix.sha256_file(library),
        },
        "repo_head": head,
    }


def _runtime_environment(provenance) -> dict:
    if not provenance:
        return {}
    identity = provenance.get("identity") or {}
    return {
        "asnumpy_version": identity.get("asnumpy_version"),
        "audit_id": identity.get("audit_id"),
        "status": identity.get("status"),
        "cann_version": identity.get("cann_version"),
        "soc_model": identity.get("soc_model"),
        "python_version": identity.get("python_version"),
        "module_sha256": identity.get("module_sha256"),
        "native_module_sha256": identity.get("native_module_sha256"),
    }


def _device_environment(provenance) -> dict:
    if not provenance:
        return {}
    return dict(provenance.get("device") or {})


def _parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 9 single-core CPU versus single-NPU matrix"
    )
    parser.add_argument(
        "--worker", dest="mode", choices=MODES, help=argparse.SUPPRESS
    )
    parser.add_argument("--profile", choices=("matmul", "sin_matmul"))
    parser.add_argument("--size", type=int)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    parser.add_argument("--out", help="matrix output directory")
    parser.add_argument(
        "--profiles",
        default="matmul,sin_matmul",
        help="comma-separated profiles",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in (64, 128, 256, 512, 1024)),
        help="comma-separated square sizes",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=None,
        help="logical core for CPU workers; defaults to the first allowed core",
    )
    parser.add_argument("--cpu-warmup", type=int, default=5)
    parser.add_argument("--cpu-repeats", type=int, default=20)
    parser.add_argument("--npu-warmup", type=int, default=3)
    parser.add_argument("--npu-repeats", type=int, default=15)
    parser.add_argument("--cold-samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--require-performance-valid",
        action="store_true",
        help="return 1 when strict provenance is missing",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode:
        if args.profile is None or args.size is None or args.worker_out is None:
            parser.error("worker mode requires --profile, --size and --worker-out")
        if args.core is None:
            parser.error("worker mode requires --core")
        if args.warmup < 0 or args.repeats < 1:
            parser.error("worker mode requires --warmup >= 0 and --repeats >= 1")
        try:
            return _worker_main(args)
        except Exception:
            traceback.print_exc()
            return 1
    if args.out is None:
        parser.error("--out is required")
    args.profiles = _parse_csv(args.profiles, str)
    args.sizes = _parse_csv(args.sizes, int)
    try:
        return _run_matrix(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
