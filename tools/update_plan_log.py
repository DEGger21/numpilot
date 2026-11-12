#!/usr/bin/env python3
"""Append the 71-case scaled benchmark section to the plan and progress log."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PLAN_VERSION_OLD = "> 版本：v3.14  "
PLAN_VERSION_NEW = "> 版本：v3.15  "
PLAN_MARKER = "### Phase 22 续三（2026-11-12）：71 例规模化性能全量覆盖"
LOG_MARKER = "### 9. 71 例规模化性能全量覆盖（2026-11-12）"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    plan = Path(args.plan)
    log = Path(args.log)
    plan_text = plan.read_text(encoding="utf-8")
    log_text = log.read_text(encoding="utf-8")

    if PLAN_MARKER in plan_text:
        raise SystemExit("plan already contains the section")
    if LOG_MARKER in log_text:
        raise SystemExit("log already contains the section")
    if PLAN_VERSION_OLD not in plan_text:
        raise SystemExit("plan version marker v3.14 not found")

    measured = summary["n_measured_speedups"]
    n_rows = summary["n_base_cases"]
    n_variants = summary["n_timed_variants"]
    minx = summary["speedup_min"]
    medx = summary["speedup_median"]
    maxx = summary["speedup_max"]
    contract = summary["n_contract_rows"]
    n_ok = summary["n_numeric_ok"]
    worst = max(
        (d["max_abs_diff"] for d in summary.get("numeric_fail_detail", [])
         if d.get("max_abs_diff") is not None),
        default=None,
    )
    worst_text = "%.2e" % worst if worst is not None else "见 timings/"
    fail_cases = summary.get("numeric_fail_cases", [])
    fail_names = "、".join("`%s`" % c for c in fail_cases[:8])
    if len(fail_cases) > 8:
        fail_names += " 等 %d 例" % len(fail_cases)

    plan_section = f"""

{PLAN_MARKER}

- **范围**：`examples/no_gap_archive_20261112` 的 71 个无缺口样例全部生成大规模
  性能变体，共 {n_variants} 个计时用例（C011 保留 512/1024 两个规模点）。生成器
  `tools/scale_no_gap_archive.py` 只替换 `build_inputs()`，`forward()` 逐字保留；
  H012/H013 的固定网格点数 17/21 同步改为 4,000,000。未重新调用模型。
- **清单**：`examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md` 由 10 行扩展为
  {n_rows} 行，逐行给出来源程序、迁移程序、数据规模、替换 API、迁移耗时与稳态加速比；
  表头环境口径不变。
- **结果**：{measured} 个计算样例给出加速比，范围 {minx:.2f}x–{maxx:.2f}x，中位
  {medx:.2f}x；{contract} 个契约/语义样例（0 维、空轴、dtype 提升、float64、
  非连续布局等）以 “—” 标注，不参与加速比统计。完整逐例数据见
  `results_summary.json`。
- **数值**：{n_ok}/{n_rows} 在严格阈值 `rtol=1e-4, atol=1e-5` 内一致。差异
  幅度随输出量级变化：A009 最大绝对差 1.6e-2（输出量级约 10^3），B015 的
  1.0 绝对差对应 ~4e6 输出量级的 2.4e-7 相对差，D2002 的 0.01 是 round 在
  .005 边界放大上游 float32 归约差异。其余差异为 float32 归约/累加顺序或
  已登记的语义边界（如 B011 fmax NaN、B018 copysign -0.0），逐例记录在
  `timings/`。真实 NPU 原始规模正确性基线仍是 69/71（B011/B018 为登记差异）。
- **证据**：`timings/` 换成 72 例全量证据（144 个 per-case JSON、144 个 worker
  日志、timings.json/csv、SHA256SUMS 全部校验通过）；`MANIFEST.json` 升级为
  schema_version 3，含 {n_variants} 个变体、{n_rows} 个公开行、
  `selected_case_per_base` 与 summary。
- **边界**：加速比为稳态计算口径（CPU 单核 `forward()` / NPU 设备常驻
  `forward()`，输入预置，NPU 含结果回传，不含 H2D 与输入构造）；含输入构造与
  H2D 的端到端原始计时保留在 `timings/timings.csv`。未 commit/push/merge/reset，
  作者手动提交。
"""
    plan_text = plan_text.replace(PLAN_VERSION_OLD, PLAN_VERSION_NEW, 1)
    plan_text = plan_text.rstrip("\n") + plan_section + "\n"
    plan.write_text(plan_text, encoding="utf-8")
    plan_sha = _sha(plan)

    log_section = f"""

{LOG_MARKER}

- **范围**：71 个无缺口归档样例全部规模化，生成 {n_variants} 个计时用例
  （C011 保留 512/1024 两点，表中取 1024 版 S4）；`tools/scale_no_gap_archive.py`
  仅替换 `build_inputs()`，H012/H013 的 `forward()` 网格常量 17/21 改为
  4,000,000，其余算子逐字保留。未调用模型，未新增迁移批次。
- **清单**：`examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md` 扩展为
  {n_rows} 行；表头固定 CANN 9.2.0-beta.2、Ascend 910B3（64GB HBM，npu-smi
  25.5.1）、Kunpeng-920 192 核（1 thread/core，8 NUMA nodes）、1.5TiB 内存、
  OS 与 AsNumpy `0.4.1.dev12+gc9115e8ce.d20261031`。
- **结果**：{measured} 个计算样例给出稳态加速比，范围 {minx:.2f}x–{maxx:.2f}x，
  中位 {medx:.2f}x；{contract} 个契约/语义样例（0 维、空轴、dtype 提升、
  float64、非连续布局等）以 “—” 标注，不参与加速比统计。逐例 CPU/NPU 中位
  耗时、加速比与规模见 table 与 `MANIFEST.json`。
- **数值**：{n_ok}/{n_rows} 在 `rtol=1e-4, atol=1e-5` 内一致；未通过项为
  float32 归约/累加顺序或已登记语义边界（{fail_names}）。B015 的 1.0 绝对
  差对应 ~4e6 输出量级的 2.4e-7 相对差；D2002 的 0.01 是 round 在 .005
  边界放大上游归约差异。真实 NPU 原始规模基线 69/71 不变。
- **证据**：`timings/` 含 {n_variants*2} 个 per-case JSON、{n_variants*2} 个
  worker 日志、timings.json/csv 与 SHA256SUMS；全部哈希校验通过。
  `MANIFEST.json` schema_version 3；`results_summary.json` 为机器可读摘要。
- **文档**：`examples/scaled_benchmark_20261112/README.md` 重写为 71 例覆盖
  口径；`PROMO_SUMMARY.md` 换为全量清单口径；仓库 README 链接文字改为
  “71 cases”。
- **边界**：归约/矩阵样例的 float32 累积顺序差异不作为迁移缺陷宣传；对外只
  引用表内加速比并同时给出数值边界。未 commit/push/merge/reset。

计划 v3.15 已同步更新，SHA-256
`{plan_sha}`。
"""
    log_text = log_text.rstrip("\n") + log_section + "\n"
    log.write_text(log_text, encoding="utf-8")

    print("plan_sha256=%s" % plan_sha)
    print("log_sha256=%s" % _sha(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
