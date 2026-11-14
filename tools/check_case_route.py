#!/usr/bin/env python3
"""Verify that one migrated benchmark case reaches the custom AsNumpy API."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_numpy(value):
    return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--migrated", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    original = load(Path(args.original), "route_original")
    migrated = load(Path(args.migrated), "route_migrated")
    inputs = original.build_inputs()
    expected = original.forward(**inputs)
    actual = migrated.forward(**inputs)
    import asnumpy as ap

    routed = getattr(ap, args.op)
    comparisons = {}
    for name in expected:
        left = np.asarray(expected[name])
        right = to_numpy(actual[name])
        comparisons[name] = {
            "shape": list(right.shape),
            "dtype": str(right.dtype),
            "max_abs_diff": float(np.nanmax(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            if left.size and np.isfinite(left).any() else 0.0,
            "consistent": bool(
                left.shape == right.shape
                and np.allclose(left, right, rtol=1e-4, atol=1e-5, equal_nan=True)
            ),
        }
    result = {
        "schema": "numpilot.generated-case-route.v1",
        "op": args.op,
        "public_module": getattr(routed, "__module__", ""),
        "public_name": getattr(routed, "__name__", ""),
        "is_numpy_alias": routed is getattr(np, args.op, None),
        "comparisons": comparisons,
        "ok": bool(all(item["consistent"] for item in comparisons.values())),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
