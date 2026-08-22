#!/usr/bin/env python3
"""Audit Poisson profiler copy elimination against strict-NPU evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


KERNEL_PREFIX_TO_API = {
    "Add": "add",
    "ApplyGradientDescent": "apply_gradient_descent",
    "ClipByValueV2": "clip",
    "Exp": "exp",
    "Ger": "outer",
    "MatMulV2": "matmul",
    "Mul": "multiply",
    "Neg": "negative",
    "NumPilotSmoothMeanLoss": "smooth_l1_loss",
    "RealDiv": "divide",
    "Sqrt": "sqrt",
    "Sub": "subtract",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memcpy_stats(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows, total_bytes, minimum, maximum = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(data_size), 0), "
            "MIN(data_size), MAX(data_size) FROM MemcpyInfo"
        ).fetchone()
        grouped = connection.execute(
            "SELECT data_size, COUNT(*) FROM MemcpyInfo "
            "GROUP BY data_size ORDER BY data_size"
        ).fetchall()
        directions = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT memcpy_direction FROM MemcpyInfo "
                "ORDER BY memcpy_direction"
            )
        ]
    return {
        "rows": int(rows),
        "bytes": int(total_bytes),
        "min_bytes": None if minimum is None else int(minimum),
        "max_bytes": None if maximum is None else int(maximum),
        "profiler_direction_codes": directions,
        "size_histogram": {str(size): count for size, count in grouped},
    }


def _kernel_counts(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT kernel_name FROM TaskTrack WHERE task_type LIKE 'KERNEL_%'"
        ).fetchall()
    counts = Counter()
    unknown = Counter()
    for (name,) in rows:
        prefix = str(name).split("_", 1)[0]
        api = KERNEL_PREFIX_TO_API.get(prefix)
        if api is None:
            unknown[prefix] += 1
        else:
            counts[api] += 1
    return {
        "total": len(rows),
        "by_api": dict(sorted(counts.items())),
        "unknown_prefixes": dict(sorted(unknown.items())),
    }


def audit(
    *,
    before_runtime_db: Path,
    after_runtime_db: Path,
    after_task_db: Path,
    iteration_evidence: Path,
    direction_probe_analysis: Path,
    before_utils: Path,
    after_utils: Path,
    wheel: Path,
) -> dict:
    evidence = json.loads(iteration_evidence.read_text(encoding="utf-8"))
    probe = json.loads(direction_probe_analysis.read_text(encoding="utf-8"))
    if evidence.get("status") != "pass" or not evidence.get("strict_npu"):
        raise RuntimeError("iteration evidence is not passing strict-NPU evidence")
    if (
        probe.get("status") != "pass"
        or not probe.get("all_probe_calls_found_exactly_once")
    ):
        raise RuntimeError("direction probe analysis is not a passing exact-match receipt")

    before_text = before_utils.read_text(encoding="utf-8")
    after_text = after_utils.read_text(encoding="utf-8")
    if "super().__init__(shape_or_array, _move=True)" in before_text:
        raise RuntimeError("before wrapper unexpectedly contains consuming move")
    if "super().__init__(shape_or_array, _move=True)" not in after_text:
        raise RuntimeError("after wrapper does not contain consuming move")

    before = _memcpy_stats(before_runtime_db)
    after = _memcpy_stats(after_runtime_db)
    kernels = _kernel_counts(after_task_db)
    transfers = evidence["transfers"]
    h2d = int(transfers["h2d"])
    d2h = int(transfers["d2h"])
    calls = int(evidence["total_runtime_calls"])
    expected_calls = {
        name: int(count)
        for name, count in evidence["runtime_call_counts"].items()
    }

    expected_before = calls + 2 * h2d + d2h
    expected_after = h2d + d2h
    expected_eliminated = calls + h2d
    checks = {
        "before_copy_accounting": before["rows"] == expected_before,
        "after_boundary_copy_accounting": after["rows"] == expected_after,
        "eliminated_copy_accounting": (
            before["rows"] - after["rows"] == expected_eliminated
        ),
        "kernel_total_matches_runtime_calls": kernels["total"] == calls,
        "kernel_api_counts_match_runtime_trace": kernels["by_api"] == expected_calls,
        "no_unknown_kernel_prefixes": not kernels["unknown_prefixes"],
        "runtime_profile_ok": bool(evidence["runtime_profile_ok"]),
        "runtime_target_integrity_ok": bool(evidence["runtime_target_integrity_ok"]),
        "numerical_comparison_ok": bool(evidence["comparison"]["consistent"]),
        "profiler_direction_field_nondiscriminating": (
            probe.get("direction_field_discriminates") is False
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    eliminated = before["rows"] - after["rows"]
    return {
        "schema_version": "1.0",
        "producer": "numpilot-poisson-profiler-copy-audit-v1",
        "status": status,
        "iterations": int(evidence["iterations"]),
        "runtime_calls": calls,
        "explicit_boundary_transfers": {"h2d": h2d, "d2h": d2h},
        "before": before,
        "after": after,
        "eliminated": {
            "copies": eliminated,
            "bytes": before["bytes"] - after["bytes"],
            "copy_reduction_ratio": eliminated / before["rows"],
        },
        "accounting": {
            "expected_before": f"runtime_calls + 2*h2d + d2h = {expected_before}",
            "expected_after": f"h2d + d2h = {expected_after}",
            "expected_eliminated": f"runtime_calls + h2d = {expected_eliminated}",
        },
        "kernel_tasks": kernels,
        "checks": checks,
        "numerical": evidence["comparison"],
        "capability_audit_id": evidence["capability_audit_id"],
        "direction_probe_conclusion": probe["conclusion"],
        "artifacts": {
            "before_runtime_db_sha256": _sha256(before_runtime_db),
            "after_runtime_db_sha256": _sha256(after_runtime_db),
            "after_task_db_sha256": _sha256(after_task_db),
            "iteration_evidence_sha256": _sha256(iteration_evidence),
            "direction_probe_analysis_sha256": _sha256(direction_probe_analysis),
            "before_utils_sha256": _sha256(before_utils),
            "after_utils_sha256": _sha256(after_utils),
            "wheel_sha256": _sha256(wheel),
        },
        "performance": {
            "published": False,
            "reason": "copy-elimination correctness audit; not a repeated performance benchmark",
        },
    }


def _render(receipt: dict) -> str:
    return "\n".join([
        "# Poisson profiler copy-elision audit",
        "",
        f"Status: `{receipt['status']}`",
        "",
        f"- Runtime calls/kernel tasks: `{receipt['runtime_calls']}`",
        f"- Memcpy before: `{receipt['before']['rows']}` / `{receipt['before']['bytes']}` bytes",
        f"- Memcpy after: `{receipt['after']['rows']}` / `{receipt['after']['bytes']}` bytes",
        f"- Eliminated: `{receipt['eliminated']['copies']}` copies",
        f"- Max absolute numerical difference: `{receipt['numerical']['max_abs_diff']}`",
        "- Performance published: `false`",
        "",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "before-runtime-db",
        "after-runtime-db",
        "after-task-db",
        "iteration-evidence",
        "direction-probe-analysis",
        "before-utils",
        "after-utils",
        "wheel",
        "out",
    ):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    output = Path(args.out).resolve()
    if output.exists():
        parser.error(f"--out must not already exist: {output}")
    receipt = audit(
        before_runtime_db=Path(args.before_runtime_db).resolve(),
        after_runtime_db=Path(args.after_runtime_db).resolve(),
        after_task_db=Path(args.after_task_db).resolve(),
        iteration_evidence=Path(args.iteration_evidence).resolve(),
        direction_probe_analysis=Path(args.direction_probe_analysis).resolve(),
        before_utils=Path(args.before_utils).resolve(),
        after_utils=Path(args.after_utils).resolve(),
        wheel=Path(args.wheel).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_render(receipt), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
