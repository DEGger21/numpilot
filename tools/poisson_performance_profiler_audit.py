#!/usr/bin/env python3
"""Audit a profiled Poisson performance run for kernels and boundary copies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.poisson_performance_evidence import _profile_identity, _sha256
from tools.poisson_profiler_copy_audit import _kernel_counts, _memcpy_stats


def audit(*, run_evidence: Path, runtime_db: Path, task_db: Path) -> dict:
    run = json.loads(run_evidence.read_text(encoding="utf-8"))
    if run.get("status") != "pass":
        raise RuntimeError("profiled run is not passing evidence")
    npu = run.get("npu")
    if not isinstance(npu, dict):
        raise RuntimeError("profiled run has no NPU worker receipt")

    memcpy = _memcpy_stats(runtime_db)
    kernels = _kernel_counts(task_db)
    transfers = npu["transfers"]
    expected_memcpy_count = int(transfers["h2d_count"]) + int(transfers["d2h_count"])
    expected_memcpy_bytes = int(transfers["h2d_bytes"]) + int(transfers["d2h_bytes"])
    expected_kernel_counts = {
        name: int(count) for name, count in npu["runtime_call_counts"].items()
    }
    checks = {
        "strict_npu": npu.get("strict_npu") is True,
        "runtime_profile_ok": npu.get("runtime_profile_ok") is True,
        "runtime_target_integrity_ok": npu.get("runtime_target_integrity_ok") is True,
        "kernel_total_matches_runtime_calls": (
            kernels["total"] == int(npu["total_runtime_calls"])
        ),
        "kernel_api_counts_match_runtime_trace": kernels["by_api"] == expected_kernel_counts,
        "no_unknown_kernel_prefixes": not kernels["unknown_prefixes"],
        "memcpy_count_matches_boundary_transfers": memcpy["rows"] == expected_memcpy_count,
        "memcpy_bytes_match_boundary_transfers": memcpy["bytes"] == expected_memcpy_bytes,
    }
    return {
        "schema_version": "1.0",
        "producer": "numpilot-poisson-performance-profiler-audit-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "profile_identity": _profile_identity(npu),
        "profiled_workload": run["workload"],
        "runtime_calls": int(npu["total_runtime_calls"]),
        "kernel_tasks": kernels,
        "memcpy": memcpy,
        "expected_boundary_transfers": {
            "count": expected_memcpy_count,
            "bytes": expected_memcpy_bytes,
            "h2d_count": int(transfers["h2d_count"]),
            "d2h_count": int(transfers["d2h_count"]),
        },
        "checks": checks,
        "artifacts": {
            "run_evidence_sha256": _sha256(run_evidence),
            "runtime_db_sha256": _sha256(runtime_db),
            "task_db_sha256": _sha256(task_db),
        },
        "performance": {
            "published": False,
            "reason": "profiler gate only; use a separate non-profiled repeated run",
        },
    }


def _render(receipt: dict) -> str:
    if receipt.get("status") != "pass" and "kernel_tasks" not in receipt:
        return "\n".join([
            "# Poisson 放大规模 profiler 门禁",
            "",
            "状态：`fail`",
            "",
            f"失败：`{receipt.get('exception_type')}: {receipt.get('exception_message')}`",
            "",
        ])
    return "\n".join([
        "# Poisson 放大规模 profiler 门禁",
        "",
        f"状态：`{receipt['status']}`",
        "",
        f"- kernel tasks：`{receipt['kernel_tasks']['total']}`",
        f"- aclrtMemcpy：`{receipt['memcpy']['rows']}` 次 / `{receipt['memcpy']['bytes']}` bytes",
        f"- 未知 kernel 前缀：`{receipt['kernel_tasks']['unknown_prefixes']}`",
        "- 本审计不发布性能结论",
        "",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-evidence", required=True)
    parser.add_argument("--runtime-db", required=True)
    parser.add_argument("--task-db", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    output = Path(args.out).resolve()
    if output.exists():
        parser.error(f"--out must not already exist: {output}")
    try:
        receipt = audit(
            run_evidence=Path(args.run_evidence).resolve(),
            runtime_db=Path(args.runtime_db).resolve(),
            task_db=Path(args.task_db).resolve(),
        )
    except Exception as error:
        receipt = {
            "schema_version": "1.0",
            "producer": "numpilot-poisson-performance-profiler-audit-v1",
            "status": "fail",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "performance": {"published": False},
        }
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
