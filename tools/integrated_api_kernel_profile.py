#!/usr/bin/env python3
"""Collect real msprof kernel rows for one isolated API-lane adapter.

Profiling establishes observed device execution, never performance.  The
worker binds the reviewed wrapper alias to the native extension in the
integration runtime, replays deterministic NumPy-oracle cases twice, and the
collector recounts the device tasks from the raw profiler CSV.  Python call
counts are recorded separately and are never used as a kernel count.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig

# One entry per reviewed API: wrapper module, native alias, core function,
# OP Type names that the ACLNN kernel may report in op_summary, and the
# deterministic oracle cases (name, rank, call builder).
API_SPECS = {
    "numpy.invert": {
        "wrapper": "logic.py",
        "module": "logic",
        "alias": "_invert",
        "native": "invert",
        "op_types": ("LogicalNot",),
        "op_name_prefix": "aclnnBitwiseNot_",
        "task_types": ("AI_VECTOR_CORE",),
    },
    "numpy.argsort": {
        "wrapper": "sorting.py",
        "module": "sorting",
        "alias": "_argsort",
        "native": "argsort",
        "op_types": ("Sort",),
        "op_name_prefix": "aclnnArgsort_",
        "task_types": ("MIX_AIV", "AI_VECTOR_CORE"),
    },
    "numpy.searchsorted": {
        "wrapper": "sorting.py",
        "module": "sorting",
        "alias": "_searchsorted",
        "native": "searchsorted",
        "op_types": ("SearchSorted",),
        "op_name_prefix": "aclnnSearchSorted_",
        "task_types": ("AI_CPU",),
    },
    "numpy.median": {
        "wrapper": "statistics.py",
        "module": "statistics",
        "alias": "_median",
        "native": "median",
        "op_types": ("Median",),
        "op_name_prefix": "aclnnMedian_",
        "task_types": ("AI_VECTOR_CORE", "AI_CPU", "MIX_AIV"),
    },
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    Path(path).chmod(0o600)


def runtime_binding(runtime_site, transaction, spec):
    import asnumpy as ap

    core = importlib.import_module("asnumpy._core." + spec["module"])
    wrapper = importlib.import_module("asnumpy." + spec["module"])
    files = {}
    for key, module in (("module", ap), ("core", core), ("wrapper", wrapper)):
        path = Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to(runtime_site):
            raise RuntimeError("module escaped isolated runtime: " + key)
        files[key] = {"path": str(path), "sha256": digest(path)}
    overlay = transaction["source"]["overlay_sha256"]
    expected = {
        "core": transaction["runtime"]["core_sha256"],
        "module": overlay["src/asnumpy/__init__.py"],
        "wrapper": overlay["src/asnumpy/" + spec["wrapper"]],
    }
    for key, value in expected.items():
        if files[key]["sha256"] != value:
            raise RuntimeError("live runtime does not match the integrated transaction: " + key)
    return files, ap, core, wrapper


def api_cases(spec_name, ap):
    """Return (cases, expected) where each case is a zero-argument call."""

    import numpy as np

    name = spec_name.split(".", 1)[1]
    if name == "invert":
        first = np.array([True, False, True, True], dtype=np.bool_)
        second = np.array(
            [[True, False, True], [False, True, False]], dtype=np.bool_
        )
        pairs = [
            ("bool_rank_one", first, lambda value: np.invert(value)),
            ("bool_rank_two", second, lambda value: np.invert(value)),
        ]
        cases = []
        for label, values, oracle in pairs:
            device = ap.ndarray.from_numpy(values)
            cases.append(
                {
                    "name": label,
                    "rank": values.ndim,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "call": lambda d=device: ap.invert(d),
                    "oracle": lambda v=values: oracle(v),
                }
            )
        return cases
    if name == "argsort":
        first = np.array([9.0, 7.0, 5.0, 3.0, 1.0], dtype=np.float32)
        second = np.array(
            [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype=np.float32
        )
        cases = []
        for label, values, axis in (
            ("rank_one_default", first, -1),
            ("rank_two_default", second, -1),
            ("rank_two_axis_zero", second, 0),
        ):
            device = ap.ndarray.from_numpy(values)
            cases.append(
                {
                    "name": label,
                    "rank": values.ndim,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "call": lambda d=device, a=axis: ap.argsort(d, a),
                    "oracle": lambda v=values, a=axis: np.argsort(v, axis=a),
                }
            )
        return cases
    if name == "searchsorted":
        boundaries = np.linspace(0.0, 1.0, 11, dtype=np.float32)
        probes = np.array([-0.5, 0.25, 0.75, 1.5], dtype=np.float32)
        cases = []
        for label, right in (("left", False), ("right", True)):
            device_boundaries = ap.ndarray.from_numpy(boundaries)
            device_probes = ap.ndarray.from_numpy(probes)
            cases.append(
                {
                    "name": "ramp_" + label,
                    "rank": probes.ndim,
                    "shape": list(probes.shape),
                    "dtype": str(probes.dtype),
                    "call": lambda b=device_boundaries, v=device_probes, r=right: (
                        ap.searchsorted(b, v, right=r)
                    ),
                    "oracle": lambda b=boundaries, v=probes, r=right: np.searchsorted(
                        b, v, side="right" if r else "left"
                    ),
                }
            )
        return cases
    if name == "median":
        odd = np.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=np.float32)
        even = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        cases = []
        for label, values in (("odd_rank_one", odd), ("even_rank_two", even)):
            device = ap.ndarray.from_numpy(values)
            cases.append(
                {
                    "name": label,
                    "rank": values.ndim,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "call": lambda d=device: ap.median(d),
                    "oracle": lambda v=values: float(np.median(v)),
                }
            )
        return cases
    raise RuntimeError("unsupported API profile: " + spec_name)


def host_value(value):
    import numpy as np

    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy())
    return np.asarray(float(value), dtype=np.float64)


def worker(args):
    import numpy as np
    from loguru import logger

    logger.remove()
    site = Path(args.runtime_site).resolve(strict=True)
    transaction = json.loads(Path(args.transaction).read_text())
    spec = API_SPECS[args.api]
    before, ap, core, wrapper = runtime_binding(site, transaction, spec)
    native = getattr(core, spec["native"])
    if getattr(wrapper, spec["alias"]) is not native:
        raise RuntimeError("public wrapper does not select the reviewed native binding")
    observed = {"native_calls": 0, "legacy_calls": 0}

    def native_call(*call_args, **call_kwargs):
        observed["native_calls"] += 1
        return native(*call_args, **call_kwargs)

    setattr(wrapper, spec["alias"], native_call)
    library = ctypes.CDLL("libascendcl.so")
    synchronize = library.aclrtSynchronizeDevice
    synchronize.argtypes = []
    synchronize.restype = ctypes.c_int
    cases = []
    for case in api_cases(args.api, ap):
        expected = case["oracle"]()
        for repeat in range(2):
            actual = case["call"]()
            if synchronize() != 0:
                raise RuntimeError("device synchronization failed")
            actual_host = host_value(actual)
            expected_host = np.asarray(expected)
            if actual_host.shape == () and expected_host.shape == ():
                if not np.isclose(
                    float(actual_host), float(expected_host), rtol=1e-6, atol=1e-6
                ):
                    raise RuntimeError("median oracle mismatch")
            else:
                np.testing.assert_array_equal(actual_host, expected_host)
            cases.append(
                {
                    "name": case["name"],
                    "repeat": repeat,
                    "rank": case["rank"],
                    "shape": case["shape"],
                    "dtype": case["dtype"],
                    "exact_equal": True,
                }
            )
    after, *_ = runtime_binding(site, transaction, spec)
    expected_calls = len(cases)
    if before != after or observed != {
        "native_calls": expected_calls,
        "legacy_calls": 0,
    }:
        raise RuntimeError("runtime identity or dispatch drift")
    write_json(
        Path(args.out),
        {
            "schema": "numpilot.integrated-api-profile-worker.v1",
            "status": "passed",
            "api": args.api,
            "transaction_sha256": digest(args.transaction),
            "worker_sha256": digest(__file__),
            "runtime": before,
            "dispatch": observed,
            "cases": cases,
        },
    )


def summarize_trace(path, op_types, op_name_prefix):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"OP Type", "Op Name", "Task Type", "Task Duration(us)"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("op_summary CSV lacks required device task columns")
        rows = list(reader)
    counts = {}
    task_types = {}
    kernel_rows = 0
    for row in rows:
        kind = row["OP Type"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind not in op_types:
            continue
        if op_name_prefix and not row.get("Op Name", "").startswith(op_name_prefix):
            continue
        task_type = row.get("Task Type")
        task_types[task_type] = task_types.get(task_type, 0) + 1
        if row.get("Device_id") not in {None, "", "0"}:
            raise ValueError("kernel task ran on an unexpected device")
        duration = float(row["Task Duration(us)"])
        if not 0 < duration < float("inf"):
            raise ValueError("kernel task duration must be finite and positive")
        kernel_rows += 1
    return len(rows), counts, task_types, kernel_rows


def collect(args):
    output = Path(args.out).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    runtime_site = Path(args.runtime_site).resolve(strict=True)
    transaction = Path(args.transaction).resolve(strict=True)
    spec = API_SPECS[args.api]
    op_types = tuple(args.expect_op_type) or spec["op_types"]
    op_name_prefix = args.expect_name_prefix or spec.get("op_name_prefix", "")
    accepted_task_types = set(args.expect_task_type) or set(spec.get("task_types", ()))
    profiler = shutil.which("msprof")
    if not profiler:
        raise RuntimeError("msprof is unavailable")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(runtime_site), sysconfig.get_path("purelib"))
    )
    env["PYTHONNOUSERSITE"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    worker_script = output / "worker.py"
    shutil.copyfile(__file__, worker_script)
    app = [
        sys.executable,
        "-S",
        str(worker_script),
        "--worker",
        "--api",
        args.api,
        "--runtime-site",
        str(runtime_site),
        "--transaction",
        str(transaction),
        "--out",
        str(output / "worker.json"),
    ]
    launch = output / "launch.sh"
    launch.write_text("#!/usr/bin/env bash\nset -e\nexec " + shlex.join(app) + "\n")
    launch.chmod(0o700)
    command = [
        profiler,
        "--application=" + str(launch),
        "--output=" + str(output / "raw"),
        "--task-time=on",
        "--runtime-api=on",
        "--aic-metrics=PipeUtilization",
    ]
    with (output / "profiler.log").open("x") as stream:
        completed = subprocess.run(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError("msprof failed; inspect profiler.log")
    info = json.loads((output / "worker.json").read_text())
    if info.get("status") != "passed" or info["worker_sha256"] != digest(__file__):
        raise RuntimeError("profile worker did not pass")
    csvs = sorted((output / "raw").rglob("op_summary*.csv"))
    if not csvs:
        raise RuntimeError("msprof produced no op_summary CSV")
    files = []
    types = {}
    task_types = {}
    kernel_rows = 0
    for path in csvs:
        count, counts, observed_types, observed = summarize_trace(
            path, set(op_types), op_name_prefix
        )
        for key, value in counts.items():
            types[key] = types.get(key, 0) + value
        for key, value in observed_types.items():
            task_types[key] = task_types.get(key, 0) + value
        kernel_rows += observed
        files.append(
            {
                "path": str(path.relative_to(output)),
                "sha256": digest(path),
                "rows": count,
            }
        )
    unexpected = sorted(set(task_types) - accepted_task_types)
    if unexpected:
        raise RuntimeError(
            "measured device tasks ran on unexpected task types: "
            + ",".join(unexpected)
        )
    if not kernel_rows:
        raise RuntimeError(
            "msprof produced no measured device tasks for " + ",".join(op_types)
        )
    manifest = {
        "schema": "numpilot.integrated-api-kernel-profile.v1",
        "status": "passed",
        "api": args.api,
        "transaction_sha256": json.loads(transaction.read_text())["transaction_sha256"],
        "transaction_file_sha256": digest(transaction),
        "runtime_core_sha256": info["runtime"]["core"]["sha256"],
        "runtime_module_sha256": info["runtime"]["module"]["sha256"],
        "runtime_wrapper_sha256": info["runtime"]["wrapper"]["sha256"],
        "wrapper_module": spec["wrapper"],
        "worker": {"path": "worker.py", "sha256": digest(worker_script)},
        "execution": {"path": "worker.json", "sha256": digest(output / "worker.json")},
        "native_calls": info["dispatch"]["native_calls"],
        "kernel_op_types": list(op_types),
        "kernel_rows": kernel_rows,
        "device_task_types": task_types,
        "tested_ranks": sorted({case["rank"] for case in info["cases"]}),
        "traces": files,
        "op_type_counts": types,
        "profiler_command": command,
        "performance_claim": False,
        "boundary": (
            "Observed profiler rows from a separate deterministic execution; "
            "not inferred from Python calls and not a performance claim."
        ),
    }
    write_json(output / "kernel-profile.json", manifest)
    print(
        json.dumps(
            {
                "profile": str(output / "kernel-profile.json"),
                "kernel_rows": kernel_rows,
        "device_task_types": task_types,
                "op_type_counts": types,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-site", required=True)
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--api", required=True, choices=sorted(API_SPECS))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expect-op-type",
        action="append",
        default=[],
        help="override the OP Type names counted as measured kernels",
    )
    parser.add_argument(
        "--expect-task-type",
        action="append",
        default=[],
        help="override the device task types admitted as measured kernels",
    )
    parser.add_argument(
        "--expect-name-prefix",
        default="",
        help="override the OP Name prefix counted as measured kernels",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
