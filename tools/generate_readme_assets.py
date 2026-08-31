#!/usr/bin/env python3
"""Generate the bilingual README architecture diagrams.

The SVGs are deliberately text-and-shape based so they remain sharp, searchable,
and reviewable without a proprietary slide renderer.  They are README-optimized
vector companions to the original-style PPT decks; both formats share the same
audited capability boundaries, but neither is a rendering source for the other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable


WIDTH = 1600
HEIGHT = 1067


@dataclass(frozen=True)
class Copy:
    title: str
    subtitle: str
    current: str
    planned: str
    optional: str
    control_title: str
    control_lines: tuple[str, ...]
    ordinary_title: str
    input_title: str
    input_lines: tuple[str, ...]
    analysis_title: str
    analysis_lines: tuple[str, ...]
    decision_title: str
    decision_note: str
    model_title: str
    model_lines: tuple[str, ...]
    gates_title: str
    gates_lines: tuple[str, ...]
    receipt_title: str
    receipt_lines: tuple[str, ...]
    gap_title: str
    gap_note: str
    aclnn_title: str
    aclnn_lines: tuple[str, ...]
    ops_title: str
    ops_lines: tuple[str, ...]
    cannbot_title: str
    cannbot_lines: tuple[str, ...]
    limit_title: str
    limit_lines: tuple[str, ...]
    no_match: str
    planned_title: str
    planned_steps: tuple[str, ...]
    runtime_title: str
    runtime_note: str
    runtime_steps: tuple[tuple[str, str], ...]
    footer: str


EN = Copy(
    title="NumPilot architecture",
    subtitle="Evidence-gated NumPy → Ascend NPU migration",
    current="SOLID · IMPLEMENTED",
    planned="DASHED · PLANNED",
    optional="OUTLINED · OPTIONAL",
    control_title="Optional collaboration plane",
    control_lines=(
        "CodeArts roles or another external orchestrator may coordinate tasks and approvals.",
        "Capability facts still come from local deterministic evidence; this plane is not a runtime dependency.",
    ),
    ordinary_title="CURRENT ORDINARY SCRIPT PATH — IMPLEMENTED",
    input_title="NumPy script",
    input_lines=("import numpy as np", "No entry-point", "Trusted source only"),
    analysis_title="analyze",
    analysis_lines=("Whole-file AST", "Bounded data flow", "Call-site + signature", "Runtime/hash identity", "No execution or model"),
    decision_title="Five-state decision",
    decision_note="Only DIRECT / accepted fallback proceeds →",
    model_title="Configured model",
    model_lines=("OpenAI-compatible", "Authorized routes only", "Source-bound response", "No capability guessing"),
    gates_title="Deterministic gates",
    gates_lines=("Model receipt + identity", "Exact AST + file hash", "verify-script: numerical", "benchmark-script: scoped", "optional strict-NPU perf"),
    receipt_title="Evidence receipts",
    receipt_lines=("Write-once artifacts", "Source/runtime hashes", "Truthful status + scope", "No credentials or raw", "model response body"),
    gap_title="OFFICIAL-FIRST GAP RESOLUTION — IMPLEMENTED, WITH ROUTE-SPECIFIC OUTPUTS",
    gap_note="Entered only for GAP_REQUIRED; no single generic chain reaches an adapter.",
    aclnn_title="1 · Installed ACLNN",
    aclnn_lines=("generic official_binary", "→ CANDIDATE_READY", "REVIEWED_MATMUL_ONLY", "→ ADAPTER_CANDIDATE"),
    ops_title="2 · Pinned local ops-nn",
    ops_lines=("After ACLNN clean miss", "SOURCE_EVIDENCE_ONLY", "materialization not wired"),
    cannbot_title="3 · CANNBot Skills + model",
    cannbot_lines=("After two clean misses", "UNVERIFIED_OPERATOR_DRAFT", "not an adapter by itself"),
    limit_title="ROUTE-SPECIFIC CURRENT LIMITS",
    limit_lines=("No automatic integration, qualification, activation, or rescan",),
    no_match="clean no-match",
    planned_title="TARGET RECOVERY LOOP — NOT AUTO-WIRED",
    planned_steps=("Integrate", "Six-gate qualify", "Activate", "Rescan", "Final migrate + verify"),
    runtime_title="PROFILE-BOUND DATA PLANE",
    runtime_note="Unsupported version, SoC, ABI, wrapper, native library, or receipt drift → CAPABILITY_AUDIT_REQUIRED",
    runtime_steps=(
        ("AsNumpy", "pinned source + installed hashes"),
        ("CANN / ACLNN", "exact toolkit identity"),
        ("Ascend NPU", "exact SoC + runtime profile"),
        ("Scoped claim", "numeric / provenance / performance receipt"),
    ),
    footer="The model proposes a candidate. Deterministic evidence decides whether it may be adopted.",
)


ZH = Copy(
    title="NumPilot 架构",
    subtitle="以证据门禁驱动 NumPy → Ascend NPU 迁移",
    current="实线 · 已实现",
    planned="虚线 · 规划中",
    optional="描边 · 可选 / 依赖环境",
    control_title="可选协作控制面",
    control_lines=(
        "CodeArts 角色或其他外部编排器可协调任务与审批。",
        "能力事实仍由本地确定性证据给出；该控制面不是运行时依赖。",
    ),
    ordinary_title="当前 ORDINARY 普通脚本主链 — 已实现",
    input_title="NumPy 脚本",
    input_lines=("import numpy as np", "无需入口约定", "仅运行可信源码"),
    analysis_title="analyze",
    analysis_lines=("全文件 AST", "有界数据流", "调用点 + 签名", "运行时 / 哈希身份", "不执行源码、不调用模型"),
    decision_title="五态能力判定",
    decision_note="仅 DIRECT / 已接受回退继续 →",
    model_title="用户配置模型",
    model_lines=("OpenAI-compatible", "仅处理已授权路径", "响应与源码绑定", "不负责猜测能力"),
    gates_title="确定性门禁",
    gates_lines=("模型回执 + 身份", "精确 AST + 产物哈希", "verify-script：数值", "benchmark-script：范围绑定", "可选 strict-NPU 性能门"),
    receipt_title="证据回执",
    receipt_lines=("不可覆盖的运行产物", "源码 / 运行时哈希", "如实记录状态与范围", "不保存凭据或原始", "模型响应正文"),
    gap_title="官方优先 GAP 解析 — 已实现，且不同路由有各自边界",
    gap_note="仅由 GAP_REQUIRED 进入；不存在统一自动通向适配器的链。",
    aclnn_title="1 · 已安装 ACLNN",
    aclnn_lines=("通用 official_binary", "→ CANDIDATE_READY", "REVIEWED_MATMUL_ONLY", "→ ADAPTER_CANDIDATE"),
    ops_title="2 · 固定本地 ops-nn",
    ops_lines=("ACLNN 明确无匹配后", "SOURCE_EVIDENCE_ONLY", "尚未接线物化"),
    cannbot_title="3 · CANNBot Skills + 配置模型",
    cannbot_lines=("两层明确无匹配后", "UNVERIFIED_OPERATOR_DRAFT", "不会自行成为适配器"),
    limit_title="当前上限因路由而异",
    limit_lines=("尚无自动集成、资格化、激活或重扫",),
    no_match="明确无匹配",
    planned_title="目标恢复闭环 — 尚未自动接线",
    planned_steps=("集成", "六门资格化", "激活", "重扫", "最终迁移 + 验证"),
    runtime_title="运行时画像绑定的数据面",
    runtime_note="版本、SoC、ABI、wrapper、原生库或回执发生未审计漂移 → CAPABILITY_AUDIT_REQUIRED",
    runtime_steps=(
        ("AsNumpy", "固定源码 + 安装文件哈希"),
        ("CANN / ACLNN", "精确工具链身份"),
        ("Ascend NPU", "精确 SoC + 运行时画像"),
        ("范围绑定声明", "数值 / 来源 / 性能回执"),
    ),
    footer="模型负责提出候选；确定性证据决定候选能否采用。",
)


def _text(
    x: int,
    y: int,
    lines: Iterable[str],
    *,
    size: int = 18,
    weight: int = 500,
    fill: str = "#334155",
    anchor: str = "start",
    line_height: int | None = None,
    css_class: str = "",
) -> str:
    line_height = line_height or int(size * 1.38)
    escaped = [escape(line) for line in lines]
    tspans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else line_height}">{line}</tspan>'
        for index, line in enumerate(escaped)
    )
    class_attr = f' class="{css_class}"' if css_class else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}"{class_attr}>{tspans}</text>'
    )


def _card(
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
    title_size: int = 21,
    body_size: int = 16,
    dashed: bool = False,
    fill: str = "#ffffff",
) -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    body_y = y + 77
    return "".join(
        (
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
            f'fill="{fill}" stroke="{accent}" stroke-width="3"{dash}/>',
            f'<rect x="{x}" y="{y}" width="{width}" height="10" rx="5" fill="{accent}"/>',
            _text(x + 18, y + 43, (title,), size=title_size, weight=750, fill="#0f172a"),
            _text(x + 18, body_y, lines, size=body_size, weight=500, fill="#475569", line_height=26),
        )
    )


def _arrow(x1: int, y1: int, x2: int, y2: int, *, dashed: bool = False, color: str = "#64748b") -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    marker = "url(#arrow-planned)" if dashed else "url(#arrow-current)"
    return (
        f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" '
        f'stroke-width="4" stroke-linecap="round" marker-end="{marker}"{dash}/>'
    )


def _pill(x: int, y: int, width: int, text_value: str, color: str, *, text_size: int = 14) -> str:
    return "".join(
        (
            f'<rect x="{x}" y="{y}" width="{width}" height="31" rx="15.5" fill="{color}" fill-opacity="0.12"/>',
            _text(x + width // 2, y + 21, (text_value,), size=text_size, weight=750, fill=color, anchor="middle"),
        )
    )


def build_svg(copy: Copy, *, language: str) -> str:
    title_id = f"title-{language}"
    desc_id = f"desc-{language}"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{escape(copy.title)}</title>',
        f'<desc id="{desc_id}">{escape(copy.footer)}</desc>',
        """<defs>
  <linearGradient id="header" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#5b4ee8"/><stop offset="1" stop-color="#008aa3"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="5" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.10"/>
  </filter>
  <marker id="arrow-current" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
  </marker>
  <marker id="arrow-planned" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#a855f7"/>
  </marker>
  <style>
    text { font-family: Inter, "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", Arial, sans-serif; }
    .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
  </style>
</defs>""",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        '<rect x="0" y="0" width="1600" height="12" fill="url(#header)"/>',
        _text(52, 58, (copy.title,), size=34, weight=800, fill="#0f172a"),
        _text(52, 86, (copy.subtitle,), size=18, weight=500, fill="#64748b"),
        _pill(930, 42, 180, copy.current, "#059669", text_size=13),
        _pill(1120, 42, 180, copy.planned, "#9333ea", text_size=13),
        _pill(1310, 42, 240, copy.optional, "#2563eb", text_size=11),
        '<g filter="url(#shadow)">',
        '<rect x="50" y="112" width="1500" height="92" rx="18" fill="#eff6ff" stroke="#3b82f6" stroke-width="2"/>',
        _text(76, 150, (copy.control_title,), size=22, weight=750, fill="#1d4ed8"),
        _text(355, 145, copy.control_lines, size=16, weight=500, fill="#334155", line_height=26),
        '</g>',
        _text(50, 242, (copy.ordinary_title,), size=17, weight=800, fill="#047857"),
        '<g filter="url(#shadow)">',
        _card(50, 260, 170, 250, copy.input_title, copy.input_lines, accent="#0ea5e9", body_size=16),
        _card(250, 260, 220, 250, copy.analysis_title, copy.analysis_lines, accent="#14b8a6", body_size=15),
        _card(500, 260, 290, 250, copy.decision_title, (), accent="#22c55e", body_size=15),
        _card(820, 260, 210, 250, copy.model_title, copy.model_lines, accent="#8b5cf6", body_size=15),
        _card(1060, 260, 220, 250, copy.gates_title, copy.gates_lines, accent="#f59e0b", body_size=14),
        _card(1310, 260, 240, 250, copy.receipt_title, copy.receipt_lines, accent="#ec4899", body_size=14),
        '</g>',
        _arrow(220, 385, 246, 385),
        _arrow(470, 385, 496, 385),
        _arrow(772, 345, 816, 345),
        _arrow(1030, 385, 1056, 385),
        _arrow(1280, 385, 1306, 385),
        _pill(518, 330, 96, "DIRECT", "#059669", text_size=12),
        _pill(624, 330, 148, "FALLBACK_CONSENT", "#d97706", text_size=9),
        _pill(518, 374, 122, "GAP_REQUIRED", "#be123c", text_size=10),
        _pill(650, 374, 122, "BLOCKED", "#64748b", text_size=11),
        _pill(518, 418, 254, "CAPABILITY_AUDIT_REQUIRED", "#2563eb", text_size=10),
        _text(645, 478, (copy.decision_note,), size=12, weight=600, fill="#64748b", anchor="middle"),
        _text(50, 553, (copy.gap_title,), size=16, weight=800, fill="#9f1239"),
        _text(50, 578, (copy.gap_note,), size=13, weight=500, fill="#64748b"),
        _arrow(579, 510, 579, 596, color="#be123c"),
        '<g filter="url(#shadow)">',
        _card(50, 600, 450, 170, copy.aclnn_title, copy.aclnn_lines, accent="#dc2626", title_size=18, body_size=13),
        _card(550, 600, 450, 170, copy.ops_title, copy.ops_lines, accent="#dc2626", title_size=18, body_size=13),
        _card(1050, 600, 500, 170, copy.cannbot_title, copy.cannbot_lines, accent="#dc2626", title_size=18, body_size=13),
        '</g>',
        _arrow(500, 685, 546, 685, color="#be123c"),
        _arrow(1000, 685, 1046, 685, color="#be123c"),
        _text(523, 662, (copy.no_match,), size=11, weight=600, fill="#9f1239", anchor="middle"),
        _text(1023, 662, (copy.no_match,), size=11, weight=600, fill="#9f1239", anchor="middle"),
        '<rect x="50" y="797" width="1500" height="84" rx="18" fill="#faf5ff" stroke="#a855f7" stroke-width="3" stroke-dasharray="10 8"/>',
        _text(75, 825, (copy.planned_title,), size=15, weight=800, fill="#7e22ce"),
        _text(75, 852, (copy.limit_lines[0],), size=12, weight=600, fill="#64748b"),
        _arrow(800, 770, 800, 793, dashed=True, color="#a855f7"),
    ]

    planned_widths = [78, 120, 78, 78, 165]
    planned_x = 940
    for index, (step, width) in enumerate(zip(copy.planned_steps, planned_widths)):
        parts.append(_pill(planned_x, 823, width, step, "#9333ea", text_size=10))
        if index < len(copy.planned_steps) - 1:
            parts.append(_arrow(planned_x + width + 3, 838, planned_x + width + 16, 838, dashed=True, color="#a855f7"))
        planned_x += width + 14

    parts.extend(
        (
            '<rect x="50" y="900" width="1500" height="108" rx="20" fill="#ecfeff" stroke="#0891b2" stroke-width="2"/>',
            _text(75, 928, (copy.runtime_title,), size=16, weight=800, fill="#0e7490"),
            _text(350, 928, (copy.runtime_note,), size=12, weight=550, fill="#475569"),
        )
    )
    runtime_x = [75, 430, 785, 1140]
    for index, ((title, note), x) in enumerate(zip(copy.runtime_steps, runtime_x)):
        parts.extend(
            (
                f'<rect x="{x}" y="944" width="285" height="48" rx="13" fill="#ffffff" stroke="#06b6d4" stroke-width="2"/>',
                _text(x + 14, 965, (title,), size=15, weight=750, fill="#0f172a"),
                _text(x + 14, 984, (note,), size=10, weight=500, fill="#64748b"),
            )
        )
        if index < len(runtime_x) - 1:
            parts.append(_arrow(x + 285, 968, runtime_x[index + 1] - 4, 968, color="#0891b2"))

    parts.extend(
        (
            _text(800, 1040, (copy.footer,), size=15, weight=700, fill="#334155", anchor="middle"),
            '</svg>',
        )
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "assets" / "architecture",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        args.output_dir / "numpilot-architecture-en.svg": build_svg(EN, language="en"),
        args.output_dir / "numpilot-architecture-zh.svg": build_svg(ZH, language="zh"),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
