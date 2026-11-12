#!/usr/bin/env python3
"""End-to-end timing for archived no-gap NumPy/AsNumpy program pairs.

Orchestrator mode runs one pinned single-thread CPU worker and one NPU worker
per case, then writes timings.json/timings.csv.  The internal worker modes are
invoked by the orchestrator and are not a public interface.

Timing scopes:
  cpu_end_to_end_ms      build_inputs() + forward() on one pinned CPU core
  npu_end_to_end_ms      build_inputs() + H2D + forward() + D2H sync
  cpu_compute_only_ms    forward() with prebuilt inputs on one pinned CPU core
  npu_device_resident_ms forward() on pre-resident device inputs + D2H sync
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_arrays(value):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_arrays(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_arrays(item)
    else:
        yield value


def _host(value):
    if isinstance(value, dict):
        return {key: _host(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
        return [_host(item) for item in value]
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return value


def _compare(expected, actual):
    import numpy as np

    expected = _host(expected)
    actual = _host(actual)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if sorted(expected) != sorted(actual):
            return {"consistent": False, "reason": "output name set differs"}
        results = []
        for key in sorted(expected):
            item = _compare(expected[key], actual[key])
            item["name"] = key
            results.append(item)
        return {
            "consistent": all(item["consistent"] for item in results),
            "items": results,
        }
    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "consistent": False,
            "reason": "shape/dtype mismatch",
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "expected_dtype": str(left.dtype),
            "actual_dtype": str(right.dtype),
        }
    if left.dtype.kind in "fc":
        equal = np.allclose(left, right, rtol=1e-4, atol=1e-5, equal_nan=True)
        with np.errstate(invalid="ignore", over="ignore"):
            diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
            denominator = np.maximum(np.abs(left.astype(np.float64)), 1e-300)
            rel = diff / denominator
        finite = diff[np.isfinite(diff)]
        finite_rel = rel[np.isfinite(rel)]
        return {
            "consistent": bool(equal),
            "max_abs_diff": float(finite.max()) if finite.size else 0.0,
            "max_rel_diff": float(finite_rel.max()) if finite_rel.size else 0.0,
        }
    return {"consistent": bool(np.array_equal(left, right))}


def _samples_ms(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    ordered = sorted(samples)
    return {
        "warmup_iterations": warmup,
        "measurement_iterations": iters,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "mean_ms": statistics.fmean(samples),
    }


def _pin(core: int):
    try:
        available = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    if core not in available:
        core = available[0]
    os.sched_setaffinity(0, {core})
    return core


def worker_cpu(program: Path, out: Path, warmup: int, iters: int):
    core = _pin(0)
    module = _load(program, "numpilot_cpu_program")
    inputs = module.build_inputs()
    expected = module.forward(**inputs)
    _compare(expected, expected)

    def end_to_end():
        return module.forward(**module.build_inputs())

    def compute_only():
        return module.forward(**inputs)

    result = {
        "mode": "cpu",
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "pinned_core": core,
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV},
        "end_to_end": _samples_ms(end_to_end, warmup, iters),
        "compute_only": _samples_ms(compute_only, warmup, iters),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def worker_npu(program: Path, migrated: Path, out: Path, warmup: int, iters: int):
    import numpy as np
    import asnumpy as ap

    core = _pin(1)
    source = _load(program, "numpilot_npu_oracle")
    module = _load(migrated, "numpilot_npu_migrated")
    expected = source.forward(**source.build_inputs())
    actual = module.forward(**module.build_inputs())
    comparison = _compare(expected, actual)

    def to_device(value):
        if isinstance(value, dict):
            return {key: to_device(value[key]) for key in value}
        if isinstance(value, np.ndarray):
            return ap.ndarray.from_numpy(np.ascontiguousarray(value))
        return value

    def end_to_end():
        result = module.forward(**module.build_inputs())
        return _host(result)

    cold_start = time.perf_counter()
    cold_result = end_to_end()
    cold_ms = (time.perf_counter() - cold_start) * 1e3
    cold_checksum = hashlib.sha256(
        json.dumps({k: np.asarray(v).tolist() for k, v in cold_result.items()},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    end_to_end_timing = _samples_ms(end_to_end, warmup, iters)

    device_inputs = to_device(module.build_inputs())

    def device_resident():
        result = module.forward(**device_inputs)
        return _host(result)

    device_timing = _samples_ms(device_resident, warmup, iters)
    result = {
        "mode": "npu",
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "migrated_sha256": hashlib.sha256(migrated.read_bytes()).hexdigest(),
        "asnumpy_version": getattr(ap, "__version__", None),
        "pinned_core": core,
        "comparison": comparison,
        "cold_end_to_end_ms": cold_ms,
        "cold_output_sha256": cold_checksum,
        "end_to_end": end_to_end_timing,
        "device_resident": device_timing,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_worker(args_list, env, log_path: Path):
    with log_path.open("wb") as log:
        completed = subprocess.run(
            args_list,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def orchestrate(archive: Path, out_dir: Path, warmup: int, iters: int, only=None):
    cases_root = archive / "cases"
    cases = sorted(path for path in cases_root.iterdir() if path.is_dir())
    if only:
        wanted = set(only)
        cases = [path for path in cases if path.name in wanted]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(THREAD_ENV)
    rows = []
    for index, case_dir in enumerate(cases, 1):
        case = case_dir.name
        program = case_dir / "program.py"
        migrated = case_dir / "migrated.py"
        cpu_json = out_dir / ("%s.cpu.json" % case)
        npu_json = out_dir / ("%s.npu.json" % case)
        cpu_log = log_dir / ("%s.cpu.log" % case)
        npu_log = log_dir / ("%s.npu.log" % case)
        if not cpu_json.is_file():
            code = _run_worker(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "cpu",
                 str(program), str(cpu_json), "--warmup", str(warmup), "--iters", str(iters)],
                env, cpu_log,
            )
            if code != 0:
                cpu_json.write_text(json.dumps({"mode": "cpu", "error": "worker exit %d" % code}) + "\n")
        if not npu_json.is_file():
            code = _run_worker(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "npu",
                 str(program), str(migrated), str(npu_json), "--warmup", str(warmup), "--iters", str(iters)],
                env, npu_log,
            )
            if code != 0:
                npu_json.write_text(json.dumps({"mode": "npu", "error": "worker exit %d" % code}) + "\n")
        cpu = json.loads(cpu_json.read_text(encoding="utf-8"))
        npu = json.loads(npu_json.read_text(encoding="utf-8"))
        npu_ops = {}
        if npu_log.is_file():
            for name in re.findall(r"aclnn[A-Za-z0-9_]+", npu_log.read_text(errors="ignore")):
                npu_ops[name] = npu_ops.get(name, 0) + 1
        row = {"case": case, "cpu": cpu, "npu": npu, "npu_ops": npu_ops}
        if "end_to_end" in cpu and "end_to_end" in npu:
            row["speedup_end_to_end"] = cpu["end_to_end"]["median_ms"] / npu["end_to_end"]["median_ms"]
            row["speedup_device_resident"] = cpu["compute_only"]["median_ms"] / npu["device_resident"]["median_ms"]
        rows.append(row)
        status = "ok" if "end_to_end" in cpu and "end_to_end" in npu else "error"
        print("[%d/%d] %s %s" % (index, len(cases), case, status), flush=True)
    (out_dir / "timings.json").write_text(
        json.dumps({"schema": "numpilot.no-gap-timings", "schema_version": 1,
                    "archive": archive.name, "warmup": warmup, "iters": iters,
                    "cpu_policy": "single core, BLAS/OMP=1", "rows": rows},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case", "cpu_end_to_end_ms", "npu_end_to_end_ms", "speedup_end_to_end",
            "cpu_compute_only_ms", "npu_device_resident_ms", "speedup_device_resident",
            "npu_cold_ms", "numeric_ok",
        ])
        for row in rows:
            cpu = row["cpu"]; npu = row["npu"]
            writer.writerow([
                row["case"],
                cpu.get("end_to_end", {}).get("median_ms"),
                npu.get("end_to_end", {}).get("median_ms"),
                row.get("speedup_end_to_end"),
                cpu.get("compute_only", {}).get("median_ms"),
                npu.get("device_resident", {}).get("median_ms"),
                row.get("speedup_device_resident"),
                npu.get("cold_end_to_end_ms"),
                npu.get("comparison", {}).get("consistent"),
            ])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="examples/no_gap_archive_20261112")
    parser.add_argument("--out", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--worker", choices=["cpu", "npu"])
    parser.add_argument("positional", nargs="*")
    args = parser.parse_args()
    if args.worker == "cpu":
        worker_cpu(Path(args.positional[0]), Path(args.positional[1]), args.warmup, args.iters)
        return 0
    if args.worker == "npu":
        worker_npu(Path(args.positional[0]), Path(args.positional[1]),
                   Path(args.positional[2]), args.warmup, args.iters)
        return 0
    archive = Path(args.archive)
    out = Path(args.out) if args.out else archive / "timings"
    orchestrate(archive, out, args.warmup, args.iters, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
