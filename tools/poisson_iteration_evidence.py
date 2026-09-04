#!/usr/bin/env python3
# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""为 Poisson Ritz 迁移产物生成多轮驻留设备的 strict-NPU 证据。

这是 Poisson P0 的受信证据工具，不是通用 benchmark。它重新校验迁移源码、显式加载
qualification overlay，把全部输入只上传一次，在轮间仅替换设备上的四组参数，最后
一次性下载最终损失和参数。工具不测时、不计算或发布加速比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig
from numpilot.comparison import compare_outputs
from numpilot.engines.ast_rewrite import audit_forward_surface
from numpilot.module_injection import temporary_module, traced_module


PARAMETER_SLICE = slice(6, 10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_namespace(source: str, filename: str) -> dict:
    namespace = {"__name__": "<numpilot-poisson-evidence>", "__file__": filename}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def _normalise_inputs(value) -> list:
    if not isinstance(value, (tuple, list)):
        raise TypeError("build_inputs() must return a tuple or list")
    inputs = list(value)
    if len(inputs) != 30:
        raise ValueError(f"Poisson iteration contract requires 30 inputs, got {len(inputs)}")
    if any(not isinstance(item, np.ndarray) for item in inputs):
        raise TypeError("every Poisson iteration input must be a NumPy ndarray")
    return inputs


def _advance(forward, inputs: list, iterations: int):
    last_output = None
    for _ in range(iterations):
        last_output = forward(*inputs)
        if not isinstance(last_output, tuple) or len(last_output) != 5:
            raise TypeError("forward() must return loss plus four parameter tensors")
        inputs[PARAMETER_SLICE] = last_output[1:5]
    return last_output


def _to_host(backend: NpuBackend, value, transfers: dict):
    if isinstance(value, tuple):
        return tuple(_to_host(backend, item, transfers) for item in value)
    if isinstance(value, list):
        return [_to_host(backend, item, transfers) for item in value]
    if hasattr(value, "to_numpy"):
        transfers["d2h"] += 1
        return backend.to_host(value)
    return value


def _comparison_receipt(result) -> dict:
    return {
        "consistent": result.consistent,
        "max_abs_diff": result.max_abs_diff,
        "max_rel_diff": result.max_rel_diff,
        "atol": result.atol,
        "rtol": result.rtol,
        "failure_path": result.failure_path,
        "failure_index": result.failure_index,
        "reason": result.reason,
        "n_elements": result.n_elements,
    }


def _loss_value(output) -> float:
    return float(np.asarray(output[0]).reshape(-1)[0])


def _profile_summary(guard) -> dict:
    summary = {}
    for name, checks in guard.checks.items():
        matched = sum(bool(check.get("matched")) for check in checks)
        summary[name] = {
            "checks": len(checks),
            "matched": matched,
            "rejected": len(checks) - matched,
        }
    return summary


def _render_markdown(receipt: dict) -> str:
    lines = [
        "# Poisson Ritz 多轮 strict-NPU 驻留执行证据",
        "",
        f"状态：`{receipt['status']}`",
        "",
    ]
    if receipt["status"] != "pass":
        lines.extend([
            f"失败阶段：`{receipt.get('failure_phase')}`",
            "",
            f"异常：`{receipt.get('exception_type')}: {receipt.get('exception_message')}`",
            "",
        ])
        return "\n".join(lines)
    lines.extend([
        f"- 迭代轮数：{receipt['iterations']}",
        f"- CPU loss：`{receipt['cpu_loss']['initial']}` → `{receipt['cpu_loss']['final']}`",
        f"- NPU 最终 loss：`{receipt['npu_final_loss']}`",
        f"- CPU/NPU 最大绝对误差：`{receipt['comparison']['max_abs_diff']}`",
        f"- H2D：{receipt['transfers']['h2d']}（仅循环前）",
        f"- D2H：{receipt['transfers']['d2h']}（仅循环后）",
        f"- strict-NPU：`{str(receipt['strict_npu']).lower()}`",
        f"- runtime profile：`{str(receipt['runtime_profile_ok']).lower()}`",
        f"- runtime target identity：`{str(receipt['runtime_target_integrity_ok']).lower()}`",
        "- 性能：未测量、未发布",
        "",
        "## 每轮调用点与累计实际调用",
        "",
        "| API | 每轮 | 累计实际调用 |",
        "|---|---:|---:|",
    ])
    for name in sorted(receipt["per_iteration_calls"]):
        lines.append(
            f"| `{name}` | {receipt['per_iteration_calls'][name]} | "
            f"{receipt['runtime_call_counts'][name]} |"
        )
    lines.extend([
        "",
        "## 固定身份",
        "",
        f"- runner SHA-256：`{receipt['runner_sha256']}`",
        f"- source SHA-256：`{receipt['source_sha256']}`",
        f"- migrated SHA-256：`{receipt['migrated_sha256']}`",
        f"- overlay SHA-256：`{receipt['overlay_sha256']}`",
        f"- capability audit：`{receipt['capability_audit_id']}`",
        "",
    ])
    return "\n".join(lines)


def _write_receipt(output: Path, receipt: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _render_markdown(receipt),
        encoding="utf-8",
    )


def run(args) -> dict:
    runner_path = Path(__file__).resolve()
    source_path = Path(args.source).resolve()
    migrated_path = Path(args.migrated).resolve()
    report_path = Path(args.migration_report).resolve()
    overlay_path = Path(args.capability_overlay).resolve()
    evidence_root = Path(args.qualification_evidence_root).resolve()

    for label, path in (
        ("source", source_path),
        ("migrated", migrated_path),
        ("migration report", report_path),
        ("capability overlay", overlay_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"qualification evidence root is missing: {evidence_root}")

    source = source_path.read_text(encoding="utf-8")
    migrated = migrated_path.read_text(encoding="utf-8")
    migration_report = json.loads(report_path.read_text(encoding="utf-8"))
    if migration_report.get("status") != "pass":
        raise RuntimeError("migration report is not a passing strict-NPU report")

    analysis = AnalysisAgent().analyze(source, str(source_path))
    if not analysis.migratable:
        raise RuntimeError("source is no longer migratable: " + ", ".join(analysis.unsupported))
    expected = MigrationAgent().migrate(
        source,
        DecisionAgent().plan(analysis),
    )
    if expected.migrated_source != migrated:
        raise RuntimeError("migrated artifact drifted from deterministic AST output")

    surface = audit_forward_surface(migrated)
    if surface.violations:
        raise RuntimeError("strict migrated forward rejected: " + "; ".join(surface.violations))
    if dict(migration_report.get("rewrites", {})) != surface.ap_calls:
        raise RuntimeError("migration report callsites do not match migrated source")

    original_ns = _load_namespace(source, str(source_path))
    canonical_inputs = _normalise_inputs(original_ns["build_inputs"]())
    cpu_inputs = [item.copy() for item in canonical_inputs]
    first_cpu_output = original_ns["forward"](*[item.copy() for item in cpu_inputs])
    cpu_output = _advance(original_ns["forward"], cpu_inputs, args.iterations)

    backend = NpuBackend(overlay_config=OverlayRuntimeConfig(
        overlay_path=str(overlay_path),
        evidence_root=str(evidence_root),
    ))
    guard = backend.prepare_profile_guard(surface.ap_calls)
    execution_module, runtime_counts = traced_module(
        backend.get_module(),
        surface.ap_calls,
        before_call=guard.validate_call,
        targets=guard.runtime_targets,
        enforce_target_identity=True,
    )

    transfers = {"h2d": 0, "d2h": 0, "h2d_phase": "before_loop", "d2h_phase": "after_loop"}
    device_inputs = []
    for item in canonical_inputs:
        device_inputs.append(backend.to_device(item.copy()))
        transfers["h2d"] += 1

    with temporary_module("asnumpy", execution_module):
        migrated_ns = _load_namespace(migrated, str(migrated_path))
        npu_output = _advance(
            migrated_ns["forward"],
            device_inputs,
            args.iterations,
        )
    host_output = _to_host(backend, npu_output, transfers)

    expected_counts = {
        name: count * args.iterations
        for name, count in surface.ap_calls.items()
    }
    if runtime_counts != expected_counts:
        raise RuntimeError(
            f"runtime call counts differ: expected={expected_counts}, observed={runtime_counts}"
        )
    if not guard.runtime_profile_ok or not guard.ok:
        raise RuntimeError("one or more runtime calls escaped the admitted profile")

    comparison = compare_outputs(cpu_output, host_output)
    if not comparison.consistent:
        raise RuntimeError(
            "multi-iteration CPU/NPU comparison failed: "
            + json.dumps(_comparison_receipt(comparison), ensure_ascii=False)
        )

    provenance = guard.to_dict()
    return {
        "schema_version": "1.0",
        "producer": "numpilot-poisson-iteration-evidence-v1",
        "status": "pass",
        "iterations": args.iterations,
        "runner_path": str(runner_path),
        "runner_sha256": _sha256(runner_path),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "migrated_path": str(migrated_path),
        "migrated_sha256": _sha256(migrated_path),
        "migration_report_path": str(report_path),
        "migration_report_sha256": _sha256(report_path),
        "overlay_path": str(overlay_path),
        "overlay_sha256": _sha256(overlay_path),
        "qualification_evidence_root": str(evidence_root),
        "capability_audit_id": provenance["audit_id"],
        "strict_npu": provenance["strict_npu"],
        "runtime_profile_ok": provenance["runtime_profile_ok"],
        "runtime_target_integrity_ok": provenance["runtime_target_integrity_ok"],
        "per_iteration_calls": dict(sorted(surface.ap_calls.items())),
        "runtime_call_counts": dict(sorted(runtime_counts.items())),
        "total_runtime_calls": sum(runtime_counts.values()),
        "profile_checks": _profile_summary(guard),
        "transfers": transfers,
        "cpu_loss": {
            "initial": _loss_value(first_cpu_output),
            "final": _loss_value(cpu_output),
        },
        "npu_final_loss": _loss_value(host_output),
        "comparison": _comparison_receipt(comparison),
        "performance": {
            "measured": False,
            "published": False,
            "reason": "correctness/residency evidence only",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--migrated", required=True)
    parser.add_argument("--migration-report", required=True)
    parser.add_argument("--capability-overlay", required=True)
    parser.add_argument("--qualification-evidence-root", required=True)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.iterations < 2 or args.iterations > 10000:
        parser.error("--iterations must be between 2 and 10000")

    output = Path(args.out).resolve()
    if output.exists():
        parser.error(f"--out must not already exist: {output}")
    try:
        receipt = run(args)
    except Exception as error:
        receipt = {
            "schema_version": "1.0",
            "producer": "numpilot-poisson-iteration-evidence-v1",
            "status": "fail",
            "failure_phase": "strict_npu_iteration",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "performance": {"measured": False, "published": False},
        }
        _write_receipt(output, receipt)
        print(
            "strict_npu_iteration=fail verified_strict_npu_evidence=no "
            f"reason={type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    _write_receipt(output, receipt)
    print(
        "strict_npu_iteration=pass actual_npu_execution=yes "
        f"iterations={receipt['iterations']} calls={receipt['total_runtime_calls']} "
        f"max_abs={receipt['comparison']['max_abs_diff']:.9g} "
        "performance_published=no"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
