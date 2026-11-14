#!/usr/bin/env python3
"""Measure one benchmark case: original NumPy forward vs migrated AsNumpy forward."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def load(path: Path):
    spec = importlib.util.spec_from_file_location("case_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_numpy(value):
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    import numpy as np

    return np.asarray(value)


def compare(original, migrated):
    import numpy as np

    if set(original) != set(migrated):
        return {"consistent": False, "reason": "output names differ"}
    comparisons = {}
    for name in original:
        left = np.asarray(original[name])
        right = as_numpy(migrated[name])
        comparisons[name] = {
            "shape_equal": left.shape == right.shape,
            "dtype_equal": left.dtype == right.dtype,
            "max_abs_diff": float(np.max(np.abs(left - right))) if left.size else 0.0,
            "consistent": bool(
                left.shape == right.shape
                and left.dtype == right.dtype
                and np.allclose(left, right, rtol=1e-4, atol=1e-5, equal_nan=True)
            ),
            "n_elements": int(left.size),
        }
    return {
        "consistent": all(item["consistent"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def timed(module, warmup: int, samples: int, *, npu: bool = False) -> dict:
    import numpy as np

    def inputs_for_run():
        current = module.build_inputs()
        if npu:
            import asnumpy as ap

            return {
                name: ap.ndarray.from_numpy(value)
                if isinstance(value, np.ndarray) else value
                for name, value in current.items()
            }
        return current

    inputs = inputs_for_run()
    outputs = module.forward(**inputs)
    for _ in range(warmup):
        module.forward(**inputs_for_run())
    values = []
    for _ in range(samples):
        current = inputs_for_run()
        start = time.perf_counter_ns()
        outputs = module.forward(**current)
        values.append(time.perf_counter_ns() - start)
    return {
        "samples_ns": values,
        "median_ms": statistics.median(values) / 1e6,
        "p10_ms": sorted(values)[max(0, int(len(values) * 0.1) - 1)] / 1e6,
        "p90_ms": sorted(values)[min(len(values) - 1, int(len(values) * 0.9))] / 1e6,
        "outputs": outputs,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--migrated", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--core", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        os.sched_setaffinity(0, {args.core})
    except (AttributeError, OSError):
        pass
    cpu = timed(load(Path(args.original)), args.warmup, args.samples)
    npu = timed(load(Path(args.migrated)), args.warmup, args.samples, npu=True)
    comparison = compare(cpu["outputs"], npu["outputs"])
    result = {
        "schema": "numpilot.benchmark-case-perf",
        "version": 1,
        "original": str(Path(args.original).resolve()),
        "migrated": str(Path(args.migrated).resolve()),
        "core": args.core,
        "warmup": args.warmup,
        "samples": args.samples,
        "cpu": {k: v for k, v in cpu.items() if k != "outputs"},
        "npu": {k: v for k, v in npu.items() if k != "outputs"},
        "comparison": comparison,
        "speedup": (
            cpu["median_ms"] / npu["median_ms"]
            if npu["median_ms"] > 0 else None
        ),
    }
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output already exists: %s" % out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if comparison["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
