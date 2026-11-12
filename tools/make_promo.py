#!/usr/bin/env python3
"""Regenerate PROMO_SUMMARY.md from the finalized 71-case benchmark manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_ms(value):
    return "—" if value is None else "%.2f" % value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    contract = set(summary["contract_cases"])

    selected = manifest["selected_case_per_base"]
    rows = [
        entry
        for entry in manifest["cases"]
        if selected.get(entry["base_case"]) == entry["case"]
    ]
    measured = [
        entry for entry in rows
        if entry["case"] not in contract
        and entry["timing"]["speedup_device_resident"] is not None
        and entry["timing"]["speedup_device_resident"] >= 1.0
    ]
    measured.sort(key=lambda e: -e["timing"]["speedup_device_resident"])
    top = measured[:10]

    times = [entry["migration_time_s"] for entry in rows]
    min_t, max_t = min(times), max(times)

    lines = []
    lines.append("# NumPilot 无缺口迁移：可引用口径（2026-11-12）")
    lines.append("")
    lines.append("> 本页只列已验证数字。引用时必须带口径与边界，不得省略。")
    lines.append("")
    lines.append("## 1. 自动化迁移")
    lines.append("")
    lines.append(
        "- 71/71 个无缺口 NumPy 程序由 NumPilot 自动迁移为 AsNumpy 程序；"
        "全部通过框架表面与数值门禁（shim 后端）；70/71 零人工修复。"
    )
    lines.append(
        "- 端到端迁移总耗时 6.05 分钟（71 例，真实 deepseek-flash 模型，"
        "含执行与数值验证）。"
    )
    lines.append(
        "- 参数化人工估算：29.8–155.6 小时（3.7–19.4 人日），即框架快约 "
        "295x–1542x。该区间是估算，不是实测。"
    )
    lines.append("")
    lines.append("## 2. 真实 NPU 性能（71 例规模化覆盖）")
    lines.append("")
    lines.append(
        "基线：单核 NumPy（BLAS/OMP=1）；设备：Ascend 910B3 + CANN 9.2.0-beta.2 "
        "+ AsNumpy `0.4.1.dev12+gc9115e8ce.d20261031`；warmup 5，iters 20，"
        "取中位数。"
    )
    lines.append("")
    lines.append(
        "稳态加速比 = 单核 CPU `forward()` 中位耗时 ÷ NPU 设备常驻 `forward()` "
        "中位耗时（输入预置，NPU 含结果回传，不含 H2D 与输入构造）。"
    )
    lines.append("")
    lines.append(
        "- 71 个无缺口样例全部规模化，共 %d 个计时用例；%d 个计算样例给出加速比，"
        "范围 %.2fx–%.2fx，中位 %.2fx。"
        % (
            summary["n_timed_variants"],
            summary["n_measured_speedups"],
            summary["speedup_min"],
            summary["speedup_max"],
            summary["speedup_median"],
        )
    )
    lines.append(
        "- %d 个契约/语义样例（0 维、空轴、dtype 提升、float64、非连续布局等）"
        "以 “—” 标注，衡量正确性而非加速比。"
        % summary["n_contract_rows"]
    )
    lines.append(
        "- 严格数值阈值 `rtol=1e-4, atol=1e-5` 下 %d/71 一致；其余为 float32 "
        "大规模归约/累加顺序（含 round 边界放大）或已登记的 IEEE 边界语义差异。"
        % summary["n_numeric_ok"]
    )
    lines.append("")
    lines.append("### 加速比前 10 名")
    lines.append("")
    lines.append("| 用例 | 规模 | 稳态加速 | CPU / NPU (ms) |")
    lines.append("| --- | --- | ---: | ---: |")
    for entry in top:
        timing = entry["timing"]
        lines.append(
            "| %s | %s | **%.2fx** | %s / %s |"
            % (
                entry["base_case"],
                entry["note"].split(": ", 1)[-1].rstrip("."),
                timing["speedup_device_resident"],
                _fmt_ms(timing["cpu_compute_only_ms"]),
                _fmt_ms(timing["npu_device_resident_ms"]),
            )
        )
    lines.append("")
    lines.append("- 完整 71 行清单（源程序、迁移程序、替换 API、迁移耗时、加速比）见")
    lines.append("  `examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md`。")
    lines.append("- 含输入构造与 H2D 的端到端原始计时见")
    lines.append("  `examples/scaled_benchmark_20261112/timings/timings.csv`。")
    lines.append("")
    lines.append("## 3. 不能说的话")
    lines.append("")
    lines.append(
        "- 不能说“71 个正确性用例平均加速 X 倍”：它们只有 1–528 个元素，"
        "计时由 NPU 启动与 H2D/D2H 开销主导，不是产品性能；对外统一引用"
        "规模化清单。"
    )
    lines.append(
        "- 不能说规模化 71 例“全部逐位一致”或“全部通过严格 allclose”："
        "严格阈值下 %d/71 一致，其余是 float32 归约顺序差异，均已逐例记录。"
        % summary["n_numeric_ok"]
    )
    lines.append(
        "- 不能把 shim 后端的 71/71 说成“真实 NPU 71/71 数值一致”；真实 NPU "
        "上 B011（fmax NaN）、B018（copysign -0.0）是已登记的契约差异。"
    )
    lines.append(
        "- 不能把标注为 “—” 的契约/语义样例计入平均加速比。"
    )
    lines.append("")
    lines.append("## 4. 建议一句话")
    lines.append("")
    lines.append(
        "“NumPilot 用真实大模型在 6.05 分钟内自动迁移 71 个 NumPy 程序；在昇腾 "
        "910B3 上对全部 71 个样例完成大规模改写，%d 个计算样例相对单核 NumPy "
        "取得 %.2fx–%.2fx 的稳态计算加速（中位 %.2fx），单例迁移耗时 %.2f–%.2f 秒。”"
        % (
            summary["n_measured_speedups"],
            summary["speedup_min"],
            summary["speedup_max"],
            summary["speedup_median"],
            min_t,
            max_t,
        )
    )
    lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("wrote %s (%d measured rows, top %d)" % (args.out, len(measured), len(top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
