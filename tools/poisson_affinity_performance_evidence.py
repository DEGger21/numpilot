#!/usr/bin/env python3
"""Run the broadcast Poisson benchmark with the CPU worker pinned to one core.

This is an experimental evidence runner.  It deliberately has a separate file
identity from the published benchmark runners, so an old profiler audit cannot
accidentally qualify measurements made at a new scale.

The CPU child is launched through ``taskset``.  That applies the affinity mask
before Python, NumPy, or a BLAS runtime starts and makes every subsequently
created CPU thread inherit the same one-CPU mask.  The worker verifies the mask
again with ``os.sched_getaffinity`` and records the observation in its receipt.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

from tools import poisson_broadcast_performance_evidence as _broadcast
from tools import poisson_performance_evidence as _baseline


_CPU_CORE_ENV = "NUMPILOT_POISSON_CPU_CORE"
_ORIGINAL_CPU_WORKER = _baseline._cpu_worker
_ORIGINAL_WORKER_COMMAND = _baseline._worker_command


def _available_cpus() -> set[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity requires os.sched_getaffinity on Linux")
    available = set(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("the current process has an empty CPU affinity mask")
    return available


def _extract_cpu_core(argv: list[str]) -> tuple[int, list[str]]:
    """Remove the wrapper-only option and resolve one allowed logical CPU."""

    cleaned: list[str] = []
    values: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--cpu-core":
            if index + 1 >= len(argv):
                raise ValueError("--cpu-core requires a logical CPU number")
            values.append(argv[index + 1])
            index += 2
            continue
        if item.startswith("--cpu-core="):
            values.append(item.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(item)
        index += 1

    if len(values) > 1:
        raise ValueError("--cpu-core may be specified only once")
    raw = values[0] if values else os.environ.get(_CPU_CORE_ENV)
    available = _available_cpus()
    if raw is None:
        core = min(available)
    else:
        try:
            core = int(raw)
        except ValueError as error:
            raise ValueError("--cpu-core must be an integer") from error
    if core not in available:
        preview = ",".join(str(value) for value in sorted(available)[:16])
        suffix = ",..." if len(available) > 16 else ""
        raise ValueError(
            f"logical CPU {core} is outside the allowed affinity mask "
            f"[{preview}{suffix}]"
        )
    return core, cleaned


def _affinity_observation(core: int) -> dict:
    observed = sorted(_available_cpus())
    if observed != [core]:
        raise RuntimeError(
            f"CPU worker affinity is {observed}, expected exactly [{core}]"
        )
    return {
        "requested_logical_cpu": core,
        "observed_logical_cpus": observed,
        "logical_cpu_count": len(observed),
        "pinned": True,
        "launch_method": "taskset --cpu-list before Python startup",
        "verification_api": "os.sched_getaffinity(0)",
    }


def _cpu_worker_with_affinity(args) -> dict:
    raw = os.environ.get(_CPU_CORE_ENV)
    if raw is None:
        raise RuntimeError(f"CPU worker is missing {_CPU_CORE_ENV}")
    try:
        core = int(raw)
    except ValueError as error:
        raise RuntimeError(f"invalid {_CPU_CORE_ENV}: {raw!r}") from error
    affinity = _affinity_observation(core)
    receipt = _ORIGINAL_CPU_WORKER(args)
    receipt["cpu_affinity"] = affinity
    return receipt


def _worker_command_with_affinity(args, worker: str, worker_out: Path) -> list[str]:
    command = _ORIGINAL_WORKER_COMMAND(args, worker, worker_out)
    if worker != "cpu":
        return command
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required to pin the CPU worker before startup")
    raw = os.environ.get(_CPU_CORE_ENV)
    if raw is None:
        raise RuntimeError(f"parent process is missing {_CPU_CORE_ENV}")
    return [taskset, "--cpu-list", raw, *command]


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        core, forwarded = _extract_cpu_core(raw_argv)
    except ValueError as error:
        raise SystemExit(f"CPU affinity configuration error: {error}") from error
    os.environ[_CPU_CORE_ENV] = str(core)

    previous = {
        "build": _baseline._build_scaled_inputs,
        "file": _baseline.__file__,
        "cpu_worker": _baseline._cpu_worker,
        "worker_command": _baseline._worker_command,
    }
    _baseline._build_scaled_inputs = _broadcast._build_scaled_inputs
    _baseline.__file__ = str(Path(__file__).resolve())
    _baseline._cpu_worker = _cpu_worker_with_affinity
    _baseline._worker_command = _worker_command_with_affinity
    try:
        return _baseline.main(forwarded)
    finally:
        _baseline._build_scaled_inputs = previous["build"]
        _baseline.__file__ = previous["file"]
        _baseline._cpu_worker = previous["cpu_worker"]
        _baseline._worker_command = previous["worker_command"]


if __name__ == "__main__":
    raise SystemExit(main())
