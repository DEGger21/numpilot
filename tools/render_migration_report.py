#!/usr/bin/env python3
"""Render a human-readable report from one NumPilot ``run`` directory.

The workflow writes machine-readable receipts for each stage.  This helper is
deliberately read-only with respect to those receipts: it only reads the
directory and renders a Markdown hand-off containing the migration decision,
numeric gates, resident benchmark scope and the runtime component identity.

Usage::

    python tools/render_migration_report.py /tmp/numpilot-run
    python tools/render_migration_report.py /tmp/numpilot-run --output report.md
    python tools/render_migration_report.py /tmp/numpilot-run --stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class ReportError(RuntimeError):
    """Raised when a run directory cannot be interpreted safely."""


def _load_receipt(run_dir: Path, name: str) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the first deterministic receipt named *name* below *run_dir*."""

    candidates = []
    direct = run_dir / name
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(sorted(p for p in run_dir.rglob(name) if p != direct and p.is_file()))
    if not candidates:
        return None, None
    path = candidates[0]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"receipt {path} must contain a JSON object")
    return value, path


def _load_timing(run_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Load optional user timing metadata from the run or its parent.

    ``timing.json`` is intentionally separate from workflow receipts: the
    former is a stopwatch measurement of the terminal migration chain,
    while the latter contains framework stage wall-clock timings.
    """
    for path in (run_dir / "timing.json", run_dir.parent / "timing.json"):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReportError(f"cannot read {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ReportError(f"timing file {path} must contain a JSON object")
        return value, path
    return None, None


def _iso_elapsed(start: Any, finish: Any) -> float | None:
    try:
        left = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        right = datetime.fromisoformat(str(finish).replace("Z", "+00:00"))
        delta = (right - left).total_seconds()
        return delta if delta >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _text(value: Any, default: str = "未记录") -> str:
    if value is None or value == "":
        return default
    # Receipt values are data, not Markdown markup.  Keep a path/model name
    # from closing the inline-code span or injecting a second line.
    return str(value).replace("`", "'").replace("\r", " ").replace("\n", " ")


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms(value: Any) -> str:
    number = _num(value)
    return f"{number:.3f} ms" if number is not None else "未记录"


def _elements(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return _text(value)


def _comparison_lines(lines: list[str], label: str, value: Mapping[str, Any] | None) -> None:
    if not isinstance(value, Mapping):
        lines.append(f"- {label}: 未执行")
        return
    mark = "通过 ✓" if value.get("consistent") else "失败 ✗"
    lines.append(
        f"- {label}: **{mark}**；max_abs={_text(value.get('max_abs_diff'), '0')}, "
        f"max_rel={_text(value.get('max_rel_diff'), '0')}, "
        f"atol={_text(value.get('atol'), '0')}, rtol={_text(value.get('rtol'), '0')}"
    )


def build_run_report(run_dir: str | Path) -> str:
    """Return Markdown for receipts in *run_dir*.

    Missing optional stages are reported as such.  No performance claim is
    inferred unless the benchmark receipt explicitly marks
    ``performance_valid`` true.
    """

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ReportError(f"run directory does not exist: {root}")

    analysis, analysis_path = _load_receipt(root, "qualified-analysis.json")
    migration, migration_path = _load_receipt(root, "script-migration-report.json")
    verification, verification_path = _load_receipt(root, "script-verification-report.json")
    benchmark, benchmark_path = _load_receipt(root, "script-benchmark-report.json")
    workflow, workflow_path = _load_receipt(root, "workflow-report.json")
    timing, timing_path = _load_timing(root)

    source = (analysis or {}).get("source_path")
    source_hash = (analysis or {}).get("source_sha256")
    if isinstance((migration or {}).get("qualified_analysis"), Mapping):
        qa = migration["qualified_analysis"]
        source = source or qa.get("source_path")
        source_hash = source_hash or qa.get("source_sha256")
    binding = (benchmark or {}).get("binding") or {}
    bound_source = binding.get("source") if isinstance(binding, Mapping) else {}
    source = source or (bound_source or {}).get("path")
    source_hash = source_hash or (bound_source or {}).get("sha256")
    migrated_hash = (migration or {}).get("migrated_sha256")
    migrated_hash = migrated_hash or ((binding.get("migrated") or {}).get("sha256") if isinstance(binding, Mapping) else None)

    status = (workflow or {}).get("status")
    status = status or (benchmark or {}).get("status") or (verification or {}).get("status") or (migration or {}).get("status") or "未记录"
    model = (migration or {}).get("configured_model") or (migration or {}).get("model")
    model = model or ((benchmark or {}).get("model_evidence") or {}).get("configured_model")

    lines = ["# NumPilot 程序迁移报告", "", f"- 运行目录: `{root}`", f"- 总体状态: **{_text(status)}**"]
    lines += [f"- 源程序: `{_text(source)}`", f"- 源程序 SHA-256: `{_text(source_hash)}`", f"- 迁移产物 SHA-256: `{_text(migrated_hash)}`"]
    lines += [f"- 配置模型: `{_text(model)}`"]

    user_elapsed = _num((timing or {}).get("elapsed_seconds")) if timing else None
    if user_elapsed is None and timing:
        elapsed_ms = _num((timing or {}).get("elapsed_ms"))
        if elapsed_ms is not None and elapsed_ms >= 0:
            user_elapsed = elapsed_ms / 1000.0
    if user_elapsed is None and timing:
        user_elapsed = _iso_elapsed((timing or {}).get("started_at_utc"), (timing or {}).get("finished_at_utc"))
    if user_elapsed is not None:
        lines.append(
            f"- 用户记录的终端主链墙钟耗时: **{user_elapsed:.3f} s**"
            "（含启动、模型等待、人工确认及终端操作；不含报告后处理）"
        )
    elif timing:
        lines.append("- 用户记录的终端主链墙钟耗时: timing.json 存在但耗时字段/时间戳无效")

    lines += ["", "## 阶段回执", "", "| 阶段 | 状态 | 回执 | 耗时 |", "|---|---|---|---|"]
    stages = (workflow or {}).get("stages") if isinstance(workflow, Mapping) else None
    if isinstance(stages, list) and stages:
        elapsed_values = [_num(stage.get("elapsed_seconds")) for stage in stages if isinstance(stage, Mapping)]
        elapsed_values = [value for value in elapsed_values if value is not None]
        if elapsed_values:
            stage_total = sum(elapsed_values)
            lines.insert(4, f"- 框架阶段耗时之和: **{stage_total:.3f} s**（不等同于用户终端主链墙钟时间）")
            if user_elapsed is not None:
                lines.insert(5, "- 说明: 阶段和可能小于端到端耗时；两者分别用于框架性能与用户实际体验记录。")
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            name = _text(stage.get("name"))
            lines.append(f"| {name} | {_text(stage.get('status'))} | workflow | {_text(stage.get('elapsed_seconds'), '未记录')} s |")
    else:
        for label, receipt, path in (("静态分析", analysis, analysis_path), ("迁移", migration, migration_path), ("数值验证", verification, verification_path), ("性能评测", benchmark, benchmark_path)):
            fallback_status = (receipt or {}).get("decision") if label == "静态分析" else None
            stage_status = (receipt or {}).get("status") if receipt else None
            lines.append(f"| {label} | {_text(stage_status or fallback_status, '未执行')} | `{path.name if path else '—'}` | — |")

    lines += ["", "## 迁移结论"]
    if analysis:
        calls = analysis.get("calls") or []
        lines.append(f"- 资格决策: **{_text(analysis.get('decision'))}**；调用点 {len(calls)} 处")
        decision_counts = Counter(
            str(call.get("decision"))
            for call in calls
            if isinstance(call, Mapping) and call.get("decision")
        )
        if decision_counts:
            summary = ", ".join(
                f"{name}={count}" for name, count in sorted(decision_counts.items())
            )
            lines.append(f"- 五态决策计数: `{_text(summary)}`")
        lines.append(f"- 已识别 API: `{', '.join(map(str, analysis.get('qualified_apis') or [])) or '—'}`")
        lines.append(f"- AsNumpy 运行时身份: {_text(analysis.get('runtime_identity_status'))}；审计 ID `{_text(analysis.get('runtime_audit_id'))}`")
    else:
        lines.append("- 未找到 qualified-analysis.json。")
    if migration:
        scan = migration.get("analysis") or {}
        counts = scan.get("counts") if isinstance(scan, Mapping) else {}
        rewrites = (scan.get("scan") or {}).get("rewrite_count") if isinstance(scan, Mapping) else None
        lines.append(f"- 迁移报告状态: **{_text(migration.get('status'))}**；重写调用 {rewrites if rewrites is not None else '未记录'} 处")
        lines.append(f"- strict-NPU 调用点: {_text(counts.get('strict_npu') if isinstance(counts, Mapping) else None, '未记录')}；fallback 已接受: {bool(migration.get('fallback_accepted'))}")
    else:
        lines.append("- 未找到 script-migration-report.json。")

    lines += ["", "## 数值验证"]
    comparisons = (verification or {}).get("comparisons") if verification else None
    if isinstance(comparisons, Mapping):
        for name, child in comparisons.items():
            _comparison_lines(lines, str(name), child if isinstance(child, Mapping) else None)
    else:
        lines.append(f"- 验证状态: **{_text((verification or {}).get('status') if verification else None, '未执行')}**")
    if verification:
        lines.append(f"- strict-NPU 已验证: {bool(verification.get('strict_npu_verified'))}；数值一致: {bool(verification.get('numerically_consistent'))}")

    lines += ["", "## 数据规模与性能优势"]
    perf = (benchmark or {}).get("performance") if benchmark else None
    perf = perf if isinstance(perf, Mapping) else {}
    workers = (benchmark or {}).get("workers") if benchmark else {}
    npu_worker = workers.get("npu") if isinstance(workers, Mapping) else {}
    cpu_worker = workers.get("cpu") if isinstance(workers, Mapping) else {}
    output_worker = npu_worker if isinstance(npu_worker, Mapping) and npu_worker.get("outputs") else cpu_worker
    output = (output_worker.get("outputs") or [{}])[0] if isinstance(output_worker, Mapping) else {}
    elements = output.get("elements")
    shape = output.get("shape")
    dtype = output.get("dtype")
    bytes_ = output.get("data_bytes")
    if elements is not None:
        lines.append(
            f"- 评测输出规模: **{_elements(elements)} 个元素**；shape=`{_text(shape)}`；"
            f"dtype=`{_text(dtype)}`；数据量 `{_text(bytes_)}` bytes"
        )
    elif benchmark:
        lines.append("- 评测输出规模: 回执未记录元素数。")
    else:
        lines.append("- 未执行 benchmark，无法判断规模优势。")
    valid = perf.get("performance_valid") is True
    speedup = _num(perf.get("median_speedup"))
    npu_timing = npu_worker.get("timing") if isinstance(npu_worker, Mapping) else None
    npu_timing = npu_timing if isinstance(npu_timing, Mapping) else {}
    warmups = npu_timing.get("warmup_ns")
    samples = npu_timing.get("samples_ns")
    affinity = npu_worker.get("affinity") if isinstance(npu_worker, Mapping) else None
    requested_core = affinity.get("requested_core") if isinstance(affinity, Mapping) else None
    if benchmark:
        lines.append(
            f"- 评测参数: warmup={len(warmups) if isinstance(warmups, list) else '未记录'}；"
            f"samples={len(samples) if isinstance(samples, list) else '未记录'}；"
            f"core={_text(requested_core)}"
        )
    if valid:
        lines.append(f"- CPU resident 中位数: **{_ms(perf.get('cpu_median_ms'))}**")
        lines.append(f"- NPU resident 中位数: **{_ms(perf.get('npu_median_ms'))}**")
        if speedup is not None and speedup > 1:
            lines.append(f"- resident 加速比: **{_text(perf.get('median_speedup'))}×**")
            lines.append(f"- 保守加速比（CPU P10 / NPU P90）: **{_text(perf.get('conservative_cpu_p10_over_npu_p90'))}×**")
            if elements is not None:
                lines.append(
                    f"- 优势结论: 迁移后的 AsNumpy 程序在上述 **{_elements(elements)} 个元素** 的 resident 计算区间观察到 NPU 优势；"
                    "该结论不外推到其他规模或端到端耗时。"
                )
            else:
                lines.append("- 优势结论: 迁移后的 AsNumpy 程序观察到 resident NPU 优势，但回执未记录元素规模，不能形成规模声明。")
        else:
            lines.append("- 优势结论: 性能门未达到 median_speedup > 1 的优势阈值，不发布 NPU 加速声明。")
        lines.append("- 口径: steady-state device-resident；不包含冷启动、输入准备和最终 D2H，因此不是端到端总耗时。")
    elif benchmark:
        lines.append("- **未发布 NPU 加速比**：benchmark provenance 或数值门禁未通过。")
        for issue in perf.get("issues") or []:
            lines.append(f"  - {_text(issue)}")

    lines += ["", "## 运行时组件与限制"]
    identity = npu_worker.get("runtime_identity_after") if isinstance(npu_worker, Mapping) else None
    if isinstance(identity, Mapping):
        lines.append(f"- AsNumpy: `{_text(identity.get('asnumpy_version'))}`；commit `{_text(identity.get('asnumpy_commit'))}`")
        lines.append(f"- CANN: `{_text(identity.get('cann_version'))}`；SoC: `{_text(identity.get('soc_model'))}`；身份状态: `{_text(identity.get('status'))}`")
    else:
        lines.append("- 未找到 NPU runtime identity。")
    lines.append("- 组件为平台/ABI 专用；目标机仍需预装匹配的 CANN、驱动、固件和 OPP。")
    lines.append("- 本报告只总结回执，不把 shim、诊断时延或未经 provenance 审计的结果宣传为 NPU 加速证据。")

    lines += ["", "## 回执文件"]
    for path in (analysis_path, migration_path, verification_path, benchmark_path, workflow_path):
        if path:
            lines.append(f"- `{path.relative_to(root)}`")
    if timing_path:
        try:
            label = timing_path.relative_to(root)
        except ValueError:
            label = Path("../") / timing_path.name
        lines.append(f"- `{label}`（用户计时）")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="NumPilot run 目录")
    parser.add_argument("--output", type=Path, help="输出 Markdown 路径（默认 RUN_DIR/migration-report.md）")
    parser.add_argument("--stdout", action="store_true", help="只输出到标准输出，不写文件")
    args = parser.parse_args(argv)
    try:
        payload = build_run_report(args.run_dir)
    except ReportError as exc:
        parser.error(str(exc))
    if args.stdout:
        sys.stdout.write(payload)
        return 0
    output = args.output or (args.run_dir / "migration-report.md")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    except OSError as exc:
        parser.error(f"cannot write {output}: {exc}")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
