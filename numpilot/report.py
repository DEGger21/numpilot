"""迁移报告落盘:把一次迁移的结果渲染成带时间戳的 Markdown, 存到默认 reports/ 目录。

wizard 与 migrate 复用同一渲染, 保证报告口径一致。时间戳用系统时间(datetime.now)。
"""
import os
from datetime import datetime

DEFAULT_REPORTS_DIR = "reports"


def build_report_md(info: dict) -> str:
    """从规范化的 info dict 渲染迁移报告 Markdown。缺失字段自动跳过。"""
    ts = info.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    net = info.get("net", "?")
    lines = [f"# NumPilot 迁移报告 · {net}", ""]
    lines.append(f"- 生成时间:{ts}")
    lines.append(f"- 源程序:`{info.get('source_path', '?')}`")
    lines.append(f"- 迁移引擎:{info.get('engine', '?')}")
    lines.append(f"- 执行后端:{info.get('backend', '?')}")
    lines.append(f"- 状态:**{info.get('status', '?')}**")

    # 多智能体阵容(每个智能体用什么模型/引擎)
    agents = info.get("agents")
    if agents:
        lines += ["", "## 多智能体阵容", "", "| 智能体 | 职责 | 模型 / 引擎 |", "|---|---|---|"]
        for name, role, model in agents:
            lines.append(f"| {name} | {role} | {model} |")

    if info.get("unsupported"):
        lines += ["", "## 缺口算子(无法迁移)",
                  "- " + ", ".join(info["unsupported"]),
                  "", "> 这些算子不在 AsNumpy 覆盖集,本 demo 不做缺口算子生成。"]
        return "\n".join(lines) + "\n"

    # 算子分析
    lines += ["", "## 算子分析"]
    lines.append(f"- numpy 调用点:{info.get('n_calls', '?')} 处")
    lines.append(f"- 去重算子数:{info.get('n_ops', '?')} 个")
    lines.append("- 覆盖情况:全部在 AsNumpy 覆盖集 ✓")

    # 算子替换
    rewrites = info.get("rewrites") or {}
    if rewrites:
        lines += ["", f"## 算子替换(np.* → ap.*,共 {info.get('total_rewrites', 0)} 处)"]
        for op, n in sorted(rewrites.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `ap.{op}` ×{n}")

    # 自定义 NPU 算子
    custom = info.get("custom") or {}
    lines += ["", "## 接入的自定义 NPU 算子"]
    if custom:
        lines.append(f"- 共 **{len(custom)}** 种 / {sum(custom.values())} 处"
                     "(relu/softmax/gelu/angle 等 numpy 没有、由 NPU 提供的算子):")
        for op, n in sorted(custom.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - `ap.{op}` ×{n}")
    else:
        lines.append("- 无(本网络未用到 relu/softmax/gelu/angle)")

    # 卸载方案
    if info.get("npu_ops") is not None or info.get("cpu_ops") is not None:
        lines += ["", "## 卸载方案"]
        lines.append(f"- NPU:{', '.join(info.get('npu_ops') or []) or '—'}")
        lines.append(f"- CPU:{', '.join(info.get('cpu_ops') or []) or '—'}")

    # 执行证据
    lines += ["", "## 执行证据"]
    if info.get("backend_version"):
        lines.append(f"- 真机 asnumpy 版本:`{info['backend_version']}`")
        lines.append(f"- 模块路径:`{info.get('backend_module', '?')}`")
    else:
        lines.append("- shim 仿真(numpy 承载 asnumpy 语义,非真机)")

    # 数值校验
    if info.get("max_rel_diff") is not None:
        mark = "一致 ✓" if info.get("numerically_consistent") else "不一致 ✗"
        lines += ["", "## 数值校验",
                  f"- {mark}  max_rel_diff={info['max_rel_diff']:.2e}  (tol={info.get('tol', 0):.0e})"]

    # 性能对比
    lines += ["", "## 整网性能对比"]
    t_cpu, t_npu = info.get("t_cpu_ms"), info.get("t_npu_ms")
    if t_cpu and t_npu:
        speedup = info.get("speedup") or (t_cpu / t_npu)
        verdict = "asnumpy@NPU 更快" if speedup > 1 else "numpy 更快"
        if info.get("n_elements"):
            lines.append(f"- 输出规模:{info['n_elements']:,} 元素")
        lines.append(f"- numpy@CPU(1 核):**{t_cpu:.3f} ms**")
        lines.append(f"- asnumpy@NPU(单卡):**{t_npu:.3f} ms**")
        lines.append(f"- 加速比:**{speedup:.4f}×**({verdict})")
        lines.append("- 口径:CPU 限 1 核 vs NPU 单卡(公平对比;本机 numpy 默认吃满全核会虚高)")
    else:
        lines.append("- 未做真机性能对比。可跑:"
                     "`numpilot bench-cpu <file> --out D` + `numpilot bench-npu <file> --out D`")

    if info.get("migrated_path"):
        lines += ["", f"- 迁移代码:`{info['migrated_path']}`"]
    return "\n".join(lines) + "\n"


def save_migration_report(info: dict, reports_dir=DEFAULT_REPORTS_DIR) -> str:
    """渲染并落盘带时间戳的报告, 返回绝对路径。文件名 migration_<net>_<时间戳>.md。"""
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now()
    info = dict(info)
    info.setdefault("timestamp", stamp.strftime("%Y-%m-%d %H:%M:%S"))
    net = info.get("net", "task")
    fname = f"migration_{net}_{stamp.strftime('%Y%m%d_%H%M%S')}.md"
    path = os.path.abspath(os.path.join(reports_dir, fname))
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report_md(info))
    return path
