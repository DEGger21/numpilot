#!/usr/bin/env python3
"""Align the bilingual NumPilot architecture deck with the implemented code.

The source decks are user-provided design masters.  This updater preserves the
existing slide geometry, palette, typography, and embedded brand assets while
changing only status-sensitive copy and line styles.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lxml import etree


PML = "http://schemas.openxmlformats.org/presentationml/2006/main"
DML = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"p": PML, "a": DML}


ENGLISH: dict[int, tuple[str, ...]] = {
    25: ("                  ", " + ", "AsNumpy", "        | Trustworthy Migration"),
    26: ("STATUS", " · solid = implemented · dashed = planned · CodeArts plane = optional"),
    29: ("①", "  NumPy Workload"),
    30: (
        "01 import numpy as np",
        "02 x = ",
        "np.linspace(...)",
        "03 y = ",
        "np.sin(x)",
        "",
        "04 z = np.matmul(a, b)",
        "05 out = np.add(y, z)",
    ),
    31: ("Unclassified input",),
    32: ("②", "  Whole-File Analysis"),
    33: ("AST + Bounded Data Flow",),
    34: ("Call-Surface Inventory",),
    35: ("callsites", " → ", "signatures", "capability unknown"),
    38: ("③", "  Capability Check"),
    46: ("AsNumpy API match",),
    47: ("④ API Classification",),
    50: ("In AsNumpy",),
    53: ("Accepted",),
    56: ("API missing",),
    59: ("Profile drift",),
    62: ("Unsafe / dynamic",),
    67: ("⑤", "  Whole-File Gate"),
    68: (
        "ELIGIBLE",
        " → ",
        "migrate",
        "whole-file",
        " + AST gate",
        "",
        "ANY GAP",
        " → ",
        "hold",
        "complete",
        " → ",
        "rescan",
    ),
    69: ("ELIGIBLE", " → ", "migrate", "GAP → ", "complete ↓"),
    70: ("⑥", "  Verify + Evidence"),
    74: ("Numeric + Scoped Perf",),
    75: ("Receipt + Decision",),
    79: ("strict-NPU gate", "explicit CPU route", "perf receipt · rollback planned"),
    81: ("AsNumpy", " source + binaries", "CANN", " · SoC · ABI"),
    82: ("Evidence + Target Feedback",),
    87: (
        "NOW · binary · evidence · draft",
        "MatMul · ADAPTER_CANDIDATE",
        "TARGET · qualify · activate · rescan",
    ),
    88: ("Classified GAP",),
    93: ("GAP · np.matmul",),
    94: ("np.matmul(a, b)", "signature · dtype · shape"),
    95: ("ACLNN / ops-nn evidence",),
    96: ("2× no-match → route B",),
    97: ("TARGET", " Completion Routes"),
    98: ("A · Official op", " reuse"),
    99: ("B · CANNBot Skill + model",),
    100: ("A/B → AsNumpy integration",),
    101: ("qualify → activate → rescan",),
    102: ("Six-Gate Qualification",),
    103: ("1  Artifact integrity",),
    104: ("2–3  Build + unit tests",),
    105: ("4  NumPy numerical",),
    106: ("5  Public AsNumpy API",),
    107: ("6  Real NPU · then activate/rescan",),
    108: ("Gap → A/B → AsNumpy → qualify → activate → rescan",),
    109: ("                ", "          | ", "NumPilot", " Evidence-Gated Migration · Current + Target"),
    118: (
        "Profile-bound runtime · NumPy API → ",
        "AsNumpy",
        " → ",
        "CANN",
        "/ACLNN",
        " → ",
        "Ascend NPU",
    ),
    120: ("Bound Runtime",),
    125: ("Profile NPU",),
    126: ("device memory", "ACLNN workspace", "operator launch + sync", "runtime identity", "typed diagnostic"),
    138: ("per-op trace (planned)",),
    139: ("Current Runtime · Planned Trace",),
    141: ("Gap API Routes",),
    142: ("A · ACLNN", "ops-nn op →"),
    143: ("Reuse", "operator"),
    144: ("B · CANNBot", "Skill+model →"),
    146: ("Custom op", "draft"),
    147: ("A/B → AsNumpy",),
    148: ("Controlled CPU NumPy Baseline  ↔  H2D / D2H  ↔  Profile-Qualified NPU",),
    149: ("AsNumpy", " | Pinned NumPy-Compatible Runtime"),
    150: ("hash-bound capability · typed diagnostics · scoped performance gates",),
    153: (
        "OPTIONAL ENGINEERING CONTROL PLANE · ",
        "CodeArts",
        " IDE / ",
        "Agent",
        " · environment-dependent",
    ),
    155: ("Configured", " Model"),
    164: ("Task coordination → review → approval → evidence audit · environment-dependent",),
    166: ("TARGET RETURN · A/B → AsNumpy → qualify → activate → rescan → migrate → verify",),
}


CHINESE: dict[int, tuple[str, ...]] = {
    25: ("                  ", " + ", "AsNumpy", "        | ", "可信迁移"),
    26: ("状态", " · 实线＝已实现 · 虚线＝规划中 · CodeArts 控制面＝可选"),
    29: ("①", "  ", "NumPy 工作负载"),
    30: (
        "01 import numpy as np",
        "02 x = ",
        "np.linspace(...)",
        "03 y = ",
        "np.sin(x)",
        "",
        "",
        "04 z = np.matmul(a, b)",
        "05 out = np.add(y, z)",
    ),
    31: ("入口尚未分类",),
    32: ("②", "  ", "全文件静态分析"),
    33: ("全文件 AST + 有界数据流",),
    34: ("调用表面", " ", "清单"),
    35: ("调用点", " → ", "签名", "能力尚未判定"),
    38: ("③", "  ", "能力检查"),
    46: ("AsNumpy API 匹配",),
    47: ("④ ", "逐 API 分类"),
    50: ("AsNumpy 已有",),
    53: ("已接受",),
    56: ("API 缺失",),
    59: ("身份漂移",),
    62: ("不安全 / 动态",),
    67: ("⑤", "  ", "全文件门禁"),
    68: (
        "全部合格",
        " → ",
        "迁移",
        "全文件",
        " + AST 门禁",
        "",
        "存在缺口",
        " → ",
        "暂停",
        "补全",
        " → ",
        "重扫",
    ),
    69: ("全部合格", " ", "", " → ", "迁移", "缺口 → ", "补全 ↓"),
    70: ("⑥", "  ", "验证 + 证据"),
    74: ("数值 + 范围绑定性能",),
    75: ("回执 + 决策",),
    79: ("严格 NPU 门", "显式 CPU 路由", "性能回执 · 回滚规划"),
    81: ("AsNumpy", " ", "源码 + 二进制哈希", "CANN", " · SoC · ", "ABI"),
    82: ("证据回执 + 目标反馈",),
    87: ("当前 · 二进制 · 证据 · 草稿", "MatMul · ADAPTER_CANDIDATE", "目标 · 资格化 · 激活 · 重扫"),
    88: ("已识别缺口",),
    93: ("GAP · np.matmul",),
    94: ("np.matmul(a, b)", "签名", " · dtype · shape"),
    95: ("ACLNN / ops-nn 证据",),
    96: ("两层无匹配 → 路线 B",),
    97: ("目标", " ", "补全路线"),
    98: ("A · 官方算子", "复用"),
    99: ("B · CANNBot Skill + 模型",),
    100: ("A/B → 接入 AsNumpy",),
    101: ("资格化 → 激活 → 重扫",),
    102: ("六门资格化",),
    103: ("1  ", "制品完整性"),
    104: ("2–3  ", "构建 + 单元测试"),
    105: ("4  ", "NumPy 数值"),
    106: ("5  ", "AsNumpy 公共 API"),
    107: ("6  ", "真实 NPU · 再激活/重扫"),
    108: ("缺口 → A/B → AsNumpy → 资格化 → 激活 → 重扫",),
    109: ("                ", "          | ", "NumPilot", " 证据门禁迁移 · 当前态 + 目标态"),
    118: ("画像绑定运行时", " · NumPy API → ", "AsNumpy", " → ", "CANN", "/ACLNN", " → ", "Ascend NPU"),
    120: ("画像绑定运行时",),
    125: ("画像 NPU",),
    126: ("设备内存", "ACLNN 工作区", "算子启动 + 同步", "运行时身份", "类型化诊断"),
    138: ("逐算子追踪（规划）",),
    139: ("当前运行时 · 规划追踪",),
    141: ("缺口 API 补全路线",),
    142: ("A · ACLNN", "ops-nn 算子 →"),
    143: ("复用已有", "算子"),
    144: ("B · CANNBot", "Skill+模型 →"),
    146: ("自定义算子", "草稿"),
    147: ("A/B → AsNumpy",),
    148: ("受控 CPU NumPy 基线", "  ↔  H2D / D2H  ↔  ", "画像合格 NPU"),
    149: ("AsNumpy", " | ", "固定 NumPy 兼容运行时"),
    150: ("能力画像绑定", " · ", "类型化诊断", " · ", "范围绑定", " · ", "性能门"),
    153: ("可选研发协作控制面 · ", " ", "CodeArts", " IDE / ", "Agent", " · 依赖具体环境"),
    155: ("用户配置", "模型"),
    164: ("任务协调 → 评审 → 审批 → 证据审计 · 依赖具体环境",),
    166: ("目标回流 · A/B → AsNumpy → 资格化 → 激活 → 重扫 → 迁移 → 验证",),
}


PLANNED_SHAPES = {10, 11, 12, 97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 143, 147, 165}


NOTES = {
    "en": """[Architecture status and sources]
- Audited identities: NumPilot main 06b0acd55ef59af0e99a47cdaee25e83b4bbafa3; pinned AsNumpy 1929a3ba44eb532a9f544f774a249cdf1c4119be.
- The workload is unclassified at intake. Static analysis inventories each callsite; runtime-bound evidence classifies calls only afterward. If any call remains gap/audit/blocked, whole-file migration stops—there is no partial migration.
- Gap outputs are route-specific: installed official-binary reaches CANDIDATE_READY; only the reviewed installed-ACLNN MatMul slice reaches ADAPTER_CANDIDATE; pinned ops-nn currently yields SOURCE_EVIDENCE_ONLY; only after two clean misses do pinned CANNBot Skills provide workflow context for the configured model to produce UNVERIFIED_OPERATOR_DRAFT.
- Target route A audits/private-builds a reusable Huawei official operator and adds an AsNumpy adapter. Target route B integrates the custom operator draft through the same AsNumpy boundary. Both must pass six gates, activate, and rescan the original workload before final migration and verification.
- Solid elements are current primitives; dashed elements are target automation. Current producers require route-homogeneous runs; mixed-route integration, qualification, activation, rescan, final migration/verification, feedback, tracing, and rollback are not generically auto-wired.
- The model endpoint is user-configured and OpenAI-compatible. CodeArts is an optional environment-dependent collaboration plane, not a NumPilot runtime dependency.
- NPU/performance claims require the exact runtime profile and receipts. Supplied brand images are shown only for technical identification; no endorsement is implied.
""",
    "zh": """[架构状态与来源]
- 已审计身份：NumPilot main 06b0acd55ef59af0e99a47cdaee25e83b4bbafa3；固定 AsNumpy 1929a3ba44eb532a9f544f774a249cdf1c4119be。
- 工作负载在入口尚未分类。静态分析先枚举每个调用点，随后才依据运行时绑定证据进行能力分类；只要任一调用仍为缺口、需审计或阻断，全文件迁移即停止，不做部分迁移。
- 缺口输出因路由而异：已安装官方二进制通用链到 CANDIDATE_READY；仅已审计的 installed-ACLNN MatMul 定向切片到 ADAPTER_CANDIDATE；固定 ops-nn 当前只到 SOURCE_EVIDENCE_ONLY；只有两层明确无匹配后，固定 CANNBot Skills 才为用户配置模型提供工程上下文并产生 UNVERIFIED_OPERATOR_DRAFT。
- 目标路线 A 对可复用华为官方算子进行审计/私有构建并增加 AsNumpy adapter；目标路线 B 将自定义算子草稿接入同一 AsNumpy 边界。两条路线都必须通过六门、激活并重扫原始工作负载，之后才能进行最终迁移与验证。
- 实线表示当前原语，虚线表示目标自动化。当前 producer 要求路由同质；mixed-route 集成、资格化、激活、重扫、最终迁移/验证、反馈、追踪和回滚尚未通用自动接线。
- 模型端点由用户配置并使用 OpenAI-compatible 协议。CodeArts 是依赖具体环境的可选协作控制面，不是 NumPilot 运行时依赖。
- NPU/性能声明须绑定精确运行时画像和回执；随源 PPT 提供的品牌图片仅用于技术识别，不表示官方背书。
""",
}


def _shape_by_id(root: etree._Element, shape_id: int) -> etree._Element:
    matches = root.xpath(
        f'.//p:sp[p:nvSpPr/p:cNvPr[@id="{shape_id}"]]',
        namespaces=NS,
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one shape id={shape_id}, found {len(matches)}")
    return matches[0]


def _replace_runs(root: etree._Element, replacements: dict[int, tuple[str, ...]]) -> None:
    for shape_id, values in replacements.items():
        shape = _shape_by_id(root, shape_id)
        text_nodes = shape.xpath('./p:txBody//a:t', namespaces=NS)
        if len(text_nodes) != len(values):
            raise ValueError(
                f"shape id={shape_id} run mismatch: source={len(text_nodes)} replacement={len(values)}"
            )
        for node, value in zip(text_nodes, values):
            node.text = value


def _normalize_unclassified_workload(root: etree._Element) -> None:
    """Remove inherited red/green emphasis that implied an intake decision."""

    shape = _shape_by_id(root, 30)
    properties = shape.xpath(
        './p:txBody//a:rPr | ./p:txBody//a:defRPr | ./p:txBody//a:endParaRPr',
        namespaces=NS,
    )
    for run_properties in properties:
        run_properties.set("b", "0")
        for child_name in ("solidFill", "gradFill", "noFill", "blipFill", "pattFill"):
            for child in run_properties.findall(f'./a:{child_name}', NS):
                run_properties.remove(child)
        solid = etree.Element(f"{{{DML}}}solidFill")
        colour = etree.SubElement(solid, f"{{{DML}}}srgbClr")
        colour.set("val", "374151")
        run_properties.insert(0, solid)
        for typeface_name in ("latin", "ea", "cs"):
            typeface = run_properties.find(f'./a:{typeface_name}', NS)
            if typeface is None:
                typeface = etree.SubElement(
                    run_properties, f"{{{DML}}}{typeface_name}"
                )
            typeface.set("typeface", "Consolas")


def _set_dashed(root: etree._Element, shape_id: int) -> None:
    shape = _shape_by_id(root, shape_id)
    line = shape.find('./p:spPr/a:ln', NS)
    if line is None:
        raise ValueError(f"shape id={shape_id} has no explicit line style")
    dash = line.find('./a:prstDash', NS)
    if dash is None:
        dash = etree.SubElement(line, f"{{{DML}}}prstDash")
    dash.set("val", "dash")


def _set_solid(root: etree._Element, shape_id: int) -> None:
    shape = _shape_by_id(root, shape_id)
    line = shape.find('./p:spPr/a:ln', NS)
    if line is None:
        raise ValueError(f"shape id={shape_id} has no explicit line style")
    dash = line.find('./a:prstDash', NS)
    if dash is None:
        dash = etree.SubElement(line, f"{{{DML}}}prstDash")
    dash.set("val", "solid")


def _set_shape_width(root: etree._Element, shape_id: int, width_emu: int) -> None:
    shape = _shape_by_id(root, shape_id)
    extent = shape.find('./p:spPr/a:xfrm/a:ext', NS)
    if extent is None:
        raise ValueError(f"shape id={shape_id} has no explicit extent")
    extent.set("cx", str(width_emu))


def _make_route_merge_visible(root: etree._Element) -> None:
    """Expose the planned A/B-to-AsNumpy boundary hidden in the source master."""

    shape = _shape_by_id(root, 147)
    line = shape.find('./p:spPr/a:ln', NS)
    if line is None:
        raise ValueError("shape id=147 has no explicit line style")
    line.set("w", "9525")
    for child_name in ("solidFill", "gradFill", "noFill", "pattFill"):
        for child in line.findall(f'./a:{child_name}', NS):
            line.remove(child)
    solid = etree.Element(f"{{{DML}}}solidFill")
    colour = etree.SubElement(solid, f"{{{DML}}}srgbClr")
    colour.set("val", "5C4BDB")
    line.insert(0, solid)
    dash = line.find('./a:prstDash', NS)
    if dash is None:
        dash = etree.SubElement(line, f"{{{DML}}}prstDash")
    dash.set("val", "dash")


def _remove_model_codearts_icon(root: etree._Element) -> None:
    pictures = root.xpath(
        './/p:pic[p:nvPicPr/p:cNvPr[@name="CodeArts logo - model access"]]',
        namespaces=NS,
    )
    if len(pictures) != 1:
        raise ValueError(f"expected one model-access CodeArts icon, found {len(pictures)}")
    pictures[0].getparent().remove(pictures[0])


def _replace_notes(payload: bytes, note: str) -> bytes:
    root = etree.fromstring(payload)
    nodes = root.xpath('.//a:t', namespaces=NS)
    if not nodes:
        raise ValueError("notes slide contains no text node")
    lines = note.rstrip("\n").splitlines()
    if len(lines) > len(nodes):
        raise ValueError(
            f"notes slide has {len(nodes)} text nodes for {len(lines)} lines"
        )
    for index, node in enumerate(nodes):
        node.text = lines[index] if index < len(lines) else ""
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _sanitize_core_properties(payload: bytes) -> bytes:
    root = etree.fromstring(payload)
    namespaces = {
        "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    for expression in ("./dc:creator", "./cp:lastModifiedBy"):
        for node in root.xpath(expression, namespaces=namespaces):
            node.text = "NumPilot Contributors"
    for node in root.xpath("./dc:title", namespaces=namespaces):
        node.text = "NumPilot Architecture"
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _normalize_app_properties(payload: bytes) -> bytes:
    root = etree.fromstring(payload)
    namespace = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    for local_name, value in (
        ("Application", "NumPilot PPT asset updater"),
        ("Slides", "1"),
        ("Notes", "1"),
    ):
        node = root.find(f"{{{namespace}}}{local_name}")
        if node is not None:
            node.text = value
    for node in root.iter():
        if node.text == "ChatGPT":
            node.text = "NumPilot Architecture"
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _rename_internal_objects(root: etree._Element) -> None:
    replacements = {
        "Closed-loop typed feedback rail": "Target feedback rail (planned)",
        "Closed-loop typed feedback label": "Target feedback label (planned)",
    }
    for node in root.xpath('.//p:cNvPr', namespaces=NS):
        if node.get("name") in replacements:
            node.set("name", replacements[node.get("name")])


def update(source: Path, destination: Path, *, language: str) -> None:
    replacements = ENGLISH if language == "en" else CHINESE
    with ZipFile(source, "r") as archive:
        entries: list[tuple[ZipInfo, bytes]] = []
        for info in archive.infolist():
            payload = archive.read(info.filename)
            if info.filename == "ppt/slides/slide1.xml":
                root = etree.fromstring(payload)
                _replace_runs(root, replacements)
                _normalize_unclassified_workload(root)
                for shape_id in PLANNED_SHAPES:
                    _set_dashed(root, shape_id)
                for shape_id in (69, 141):
                    _set_solid(root, shape_id)
                for shape_id in (142, 144):
                    _set_shape_width(root, shape_id, 722376)  # 0.79 in
                _make_route_merge_visible(root)
                _rename_internal_objects(root)
                _remove_model_codearts_icon(root)
                payload = etree.tostring(
                    root,
                    xml_declaration=True,
                    encoding="UTF-8",
                    standalone=True,
                )
            elif info.filename == "ppt/notesSlides/notesSlide1.xml":
                payload = _replace_notes(payload, NOTES[language])
            elif info.filename == "docProps/core.xml":
                payload = _sanitize_core_properties(payload)
            elif info.filename == "docProps/app.xml":
                payload = _normalize_app_properties(payload)
            entries.append((deepcopy(info), payload))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w") as output:
        for info, payload in entries:
            info.compress_type = ZIP_DEFLATED
            output.writestr(info, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-en", type=Path, required=True)
    parser.add_argument("--source-zh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = (
        (args.source_en, args.output_dir / "numpilot-en.pptx", "en"),
        (args.source_zh, args.output_dir / "numpilot-zh.pptx", "zh"),
    )
    for source, destination, language in targets:
        update(source, destination, language=language)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
