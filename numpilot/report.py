"""迁移报告落盘:把一次迁移的结果渲染成带时间戳的 Markdown。

wizard 与 migrate 复用同一渲染, 保证报告口径一致。默认目录位于仓库外的
用户状态目录，可用 ``NUMPILOT_REPORTS_DIR`` 或 CLI 参数覆盖。
"""
import os
from datetime import datetime
import re

from numpilot.paths import reports_dir as default_reports_dir


def _scientific(value) -> str:
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def _append_comparison(lines, label: str, comparison: dict) -> None:
    """追加一个可复核的语义比较子门禁。"""
    mark = "通过 ✓" if comparison.get("consistent") else "失败 ✗"
    lines.append(f"- {label}:{mark}")
    lines.append(
        "  - "
        f"max_abs={_scientific(comparison.get('max_abs_diff', 0))}, "
        f"max_norm={_scientific(comparison.get('max_rel_diff', 0))}, "
        f"atol={_scientific(comparison.get('atol', 0))}, "
        f"rtol={_scientific(comparison.get('rtol', 0))}"
    )
    if not comparison.get("consistent"):
        index = comparison.get("failure_index") or ""
        lines.append(
            f"  - 首个失败:{comparison.get('failure_path') or '?'}{index} · "
            f"{comparison.get('reason') or comparison.get('failure_reason') or '?'}"
        )


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

    model_identity = info.get("model_identity") or {}
    model_receipt = info.get("model_receipt") or {}
    if model_identity:
        lines += ["", "## Configured model 身份"]
        lines.append(
            "- provider / protocol:"
            f"`{model_identity.get('provider', '?')}` / "
            f"`{model_identity.get('protocol', '?')}`"
        )
        lines.append(
            "- configured / response model:"
            f"`{model_identity.get('configured_model', '?')}` / "
            f"`{model_identity.get('response_model', '?')}`"
        )
        lines.append(
            "- response identity:"
            f"`{model_identity.get('response_model_identity', '?')}`"
        )
        lines.append(
            "- response content SHA-256:"
            f"`{model_receipt.get('content_sha256', '?')}`"
        )
        lines.append(
            "- migration artifact SHA-256:"
            f"`{model_identity.get('artifact_sha256', '?')}`"
        )

    # 多智能体阵容(每个智能体用什么模型/引擎)
    agents = info.get("agents")
    if agents:
        lines += ["", "## 多智能体阵容", "", "| 智能体 | 职责 | 模型 / 引擎 |", "|---|---|---|"]
        for name, role, model in agents:
            lines.append(f"| {name} | {role} | {model} |")

    if info.get("unsupported"):
        lines += ["", "## 缺口算子（无法迁移）",
                  "- " + ", ".join(info["unsupported"])]
        for item in info.get("unsupported_details") or []:
            lines.append(
                f"- `{item.get('name', '?')}`@L{item.get('lineno', '?')}: "
                f"{item.get('reason') or '未通过静态 capability/signature 门禁'}"
            )
        lines += ["", "> 当前实现只生成 GapSpec/阻断报告，不会把未经验证的 API 冒充可迁移算子。"]
        return "\n".join(lines) + "\n"

    # 算子分析
    lines += ["", "## 算子分析"]
    lines.append(f"- NumPy 调用点:{info.get('n_numpy_calls', info.get('n_calls', '?'))} 处")
    if info.get("n_operator_calls"):
        lines.append(f"- OperatorSpec 调用点:{info['n_operator_calls']} 处")
    lines.append(f"- 去重算子数:{info.get('n_ops', '?')} 个")
    lines.append("- 静态门禁:API 名称与已建模签名均通过（运行时仍须 capability/provenance 审计）")

    # 算子替换
    rewrites = info.get("rewrites") or {}
    if rewrites:
        mapping = (
            "np.*/OperatorSpec helper → ap.*"
            if info.get("n_operator_calls") else "np.* → ap.*"
        )
        lines += ["", f"## 算子替换({mapping},共 {info.get('total_rewrites', 0)} 处)"]
        for op, n in sorted(rewrites.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `ap.{op}` ×{n}")

    # 这些名称来自已有 AsNumpy 扩展 API；不能表述为本次新生成的自定义算子。
    extension_apis = info.get("extension_apis") or info.get("custom") or {}
    lines += ["", "## 命中的已有 AsNumpy 扩展 API"]
    if extension_apis:
        lines.append(
            f"- 共 **{len(extension_apis)}** 种 / {sum(extension_apis.values())} 处；"
            "表示复用已有 API，不表示本次生成了新的 Ascend C 算子。"
        )
        for op, n in sorted(extension_apis.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - `ap.{op}` ×{n}")
        if info.get("operator_specs"):
            lines.append(
                "- OperatorSpec 只提供显式映射；真机执行仍须通过 qualification overlay。"
            )
    else:
        lines.append("- 无；本次新生成自定义算子数量为 **0**。")

    # 卸载方案
    if info.get("npu_ops") is not None or info.get("cpu_ops") is not None:
        lines += ["", "## 卸载方案"]
        lines.append(f"- 策略:{info.get('strategy') or '未记录'}")
        lines.append(f"- NPU:{', '.join(info.get('npu_ops') or []) or '—'}")
        lines.append(f"- CPU:{', '.join(info.get('cpu_ops') or []) or '—'}")

    # 执行证据
    lines += ["", "## 执行证据"]
    if info.get("backend_version"):
        lines.append(f"- 实际加载 AsNumpy 版本:`{info['backend_version']}`")
        lines.append(f"- 实际模块路径:`{info.get('backend_module', '?')}`")
    else:
        lines.append("- shim 仿真(numpy 承载 asnumpy 语义,非真机)")
    provenance = info.get("backend_provenance") or {}
    if provenance:
        strict = provenance.get("strict_npu")
        ok = provenance.get("ok")
        verdict = "通过 ✓" if strict and ok else "未通过/未执行 ✗"
        lines.append(f"- strict-NPU capability 门禁:{verdict}")
        lines.append(f"- Audit ID:`{provenance.get('audit_id') or '未记录'}`")
        environment = provenance.get("environment") or {}
        if environment:
            lines.append(
                f"- 环境:CANN={environment.get('cann_version') or '?'}, "
                f"SoC={environment.get('soc_version') or '?'}"
            )
        for name in provenance.get("missing") or []:
            lines.append(f"- 缺失 API:`{name}`")
        for name in provenance.get("non_strict") or []:
            lines.append(f"- 非严格 NPU API:`{name}`")
    elif info.get("backend") == "npu":
        lines.append("- strict-NPU capability 门禁:无证据，按失败处理 ✗")

    # 数值校验
    if info.get("exception_type") or info.get("max_rel_diff") is not None:
        lines += ["", "## 三层语义校验"]
        if info.get("exception_type"):
            lines.append(
                f"- 执行异常 ✗ `{info.get('exception_phase') or '?'}` / "
                f"`{info['exception_type']}: {info.get('exception_message') or ''}`"
            )
        comparisons = info.get("comparisons") or {}
        labels = {
            "inputs": "输入生成一致性",
            "kernel": "同一规范输入上的 forward",
            "artifact": "迁移产物 build_inputs + forward",
        }
        if comparisons:
            for key in ("inputs", "kernel", "artifact"):
                if comparisons.get(key):
                    _append_comparison(lines, labels[key], comparisons[key])
        elif info.get("max_rel_diff") is not None:
            mark = "一致 ✓" if info.get("numerically_consistent") else "不一致 ✗"
            lines.append(
                f"- 旧版汇总结果:{mark}，max_abs={_scientific(info.get('max_abs_diff', 0))}, "
                f"max_norm={_scientific(info['max_rel_diff'])}, "
                f"atol={_scientific(info.get('atol', 0))}, rtol={_scientific(info.get('rtol', 0))}"
            )
        lines.append(
            "- 迁移产物自身 build_inputs 已执行:"
            + ("是" if info.get("artifact_inputs_executed") else "否")
        )

    # 性能对比。只有调用方显式给出 performance_valid=True 时才发布加速比；
    # 旧报告或未做 provenance audit 的流程不再被默认当成 NPU 性能证据。
    lines += ["", "## 整网性能对比"]
    t_cpu, t_npu = info.get("t_cpu_ms"), info.get("t_npu_ms")
    performance_valid = info.get("performance_valid") is True
    if t_cpu and t_npu and performance_valid:
        speedup = info.get("speedup") or (t_cpu / t_npu)
        verdict = "asnumpy@NPU 更快" if speedup > 1 else "numpy 更快"
        if info.get("n_elements"):
            lines.append(f"- 输出规模:{info['n_elements']:,} 元素")
        threads = info.get("cpu_threads")
        thread_label = str(threads) if threads is not None else "未记录"
        lines.append(f"- numpy@CPU(BLAS 线程={thread_label}):**{t_cpu:.3f} ms**")
        lines.append(f"- asnumpy@NPU(单卡):**{t_npu:.3f} ms**")
        lines.append(f"- 合格 NPU 加速比:**{speedup:.4f}×**({verdict})")
        lines.append("- 资格:数值门禁通过，且 capability audit 未发现 CPU fallback 或未知来源。")
    elif t_cpu and t_npu:
        lines.append(f"- numpy@CPU 诊断时延:{t_cpu:.3f} ms")
        lines.append(f"- 迁移侧诊断时延:{t_npu:.3f} ms")
        lines.append("- **未发布 NPU 加速比**:缺少严格 NPU provenance，或数值/基线门禁未通过。")
        for issue in info.get("performance_issues") or []:
            lines.append(f"  - {issue}")
    else:
        lines.append("- 未做真机性能对比。可跑:"
                     "`numpilot bench-cpu <file> --out D` + `numpilot bench-npu <file> --out D`")

    if info.get("migrated_path"):
        lines += ["", f"- 迁移代码:`{info['migrated_path']}`"]
    return "\n".join(lines) + "\n"


def build_benchmark_report_md(result: dict) -> str:
    """渲染 ``bench-npu`` 的可审计报告，不合格结果绝不展示加速比。"""
    lines = [f"# NumPilot 性能评测报告 · {result.get('net', '?')}", ""]
    lines.append(f"- 源程序:`{result.get('file', '?')}`")
    lines.append(f"- 源码 SHA256:`{result.get('source_sha256') or '未记录'}`")
    lines.append(f"- 迁移产物 SHA256:`{result.get('migrated_sha256') or '未记录'}`")
    lines.append(f"- 状态:**{result.get('status', '?')}**")
    lines.append(f"- AsNumpy 版本:`{result.get('backend_version') or '?'}`")
    lines.append(f"- AsNumpy 路径:`{result.get('backend_module') or '?'}`")
    lines.append(f"- 允许 fallback 逃生口:{bool(result.get('allow_cpu_fallback'))}")
    lines.append(f"- 实际使用逃生口:{bool(result.get('fallback_override_used'))}")
    lines.append(f"- 迁移产物自身 build_inputs 已执行:{bool(result.get('artifact_inputs_executed'))}")

    provenance = result.get("provenance") or {}
    lines += ["", "## 执行来源门禁"]
    lines.append(f"- Audit 接口:{provenance.get('audit_source') or '不可用'}")
    lines.append(f"- Audit ID:{provenance.get('audit_id') or '未记录'}")
    lines.append(f"- 结论:{'严格 NPU ✓' if provenance.get('ok') else '不合格 ✗'}")
    required = provenance.get("required_ops") or []
    lines.append(f"- 迁移后算子:{', '.join(required) or '—'}")
    for issue in provenance.get("issues") or []:
        lines.append(f"- 问题:{issue}")
    surface = provenance.get("static_forward_surface") or {}
    if surface:
        lines.append(f"- strict forward 静态数据流:{'通过 ✓' if surface.get('ok') else '失败 ✗'}")
        lines.append(
            "- 静态 ap.* callsite:"
            + ", ".join(
                f"{name}×{count}"
                for name, count in sorted((surface.get("ap_calls") or {}).items())
            )
        )
    runtime_counts = provenance.get("runtime_call_counts") or {}
    if runtime_counts:
        lines.append(
            f"- 运行时 callsite 门禁:"
            f"{'通过 ✓' if provenance.get('runtime_trace_ok') else '失败 ✗'}"
        )
        for path, counts in runtime_counts.items():
            lines.append(
                f"  - {path}: "
                + ", ".join(
                    f"{name}×{count}" for name, count in sorted(counts.items())
                )
            )
    environment = provenance.get("environment") or {}
    if environment:
        lines.append(f"- 模块路径:`{environment.get('module_path') or '?'}`")
        lines.append(f"- CANN:{environment.get('cann_version') or '?'}")
        lines.append(f"- SoC:{environment.get('soc_version') or '?'}")

    comparison = result.get("comparison")
    lines += ["", "## 数值门禁"]
    comparisons = {
        "输入生成一致性": result.get("input_comparison"),
        "同一规范输入上的 forward": result.get("kernel_comparison") or comparison,
        "迁移产物 build_inputs + forward": result.get("artifact_comparison"),
    }
    if any(comparisons.values()):
        for label, child in comparisons.items():
            if child:
                _append_comparison(lines, label, child)
    else:
        lines.append("- 未执行（被 provenance 门禁提前拒绝）。")
    if result.get("runtime_error"):
        lines.append(f"- 迁移侧运行异常:`{result['runtime_error']}`")

    cpu_blas = (result.get("cpu_baseline") or {}).get("blas") or {}
    npu_blas = result.get("npu_process_blas") or {}
    lines += ["", "## 线程与计时证据"]
    lines.append(
        f"- CPU 基线 BLAS 线程:{cpu_blas.get('effective_threads', '未记录')} "
        f"(来源={cpu_blas.get('source', '未记录')})")
    lines.append(
        f"- NPU 评测进程 BLAS 线程:{npu_blas.get('effective_threads', '未记录')} "
        f"(来源={npu_blas.get('source', '未记录')})")
    if result.get("t_cpu_ms") is not None:
        lines.append(f"- CPU 时延:{result['t_cpu_ms']:.3f} ms")
    if result.get("t_npu_ms") is not None:
        label = "NPU 时延" if result.get("performance_valid") else "迁移侧诊断时延"
        lines.append(f"- {label}:{result['t_npu_ms']:.3f} ms")
    cpu_timing = (result.get("cpu_baseline") or {}).get("timing") or {}
    npu_timing = result.get("npu_timing") or {}
    if cpu_timing:
        lines.append(
            "- CPU 原始样本:"
            f"warmup={cpu_timing.get('warmup_iterations', '?')}, "
            f"repeats={cpu_timing.get('measurement_iterations', '?')}, "
            f"samples_ms={cpu_timing.get('samples_ms', [])}"
        )
    if npu_timing:
        lines.append(
            "- NPU 原始样本:"
            f"warmup={npu_timing.get('warmup_iterations', '?')}, "
            f"repeats={npu_timing.get('measurement_iterations', '?')}, "
            f"samples_ms={npu_timing.get('samples_ms', [])}"
        )
    timing_scope = result.get("timing_scope") or {}
    if timing_scope:
        lines.append(f"- NPU 计时口径:{timing_scope.get('description', '?')}")
        lines.append(
            "- 口径分类:"
            f"kernel-only={bool(timing_scope.get('kernel_only'))}, "
            f"end-to-end={bool(timing_scope.get('end_to_end'))}, "
            f"含 H2D={bool(timing_scope.get('includes_h2d'))}, "
            f"每轮含 D2H 同步={bool(timing_scope.get('includes_d2h_sync_each_iteration'))}"
        )

    lines += ["", "## 性能结论"]
    if result.get("performance_valid") and result.get("speedup") is not None:
        speedup = result["speedup"]
        if speedup >= 1:
            verdict = f"NPU 快 {speedup:.2f}×"
        elif speedup > 0:
            verdict = f"CPU 快 {1.0 / speedup:.2f}×"
        else:
            verdict = "NPU 时延无效"
        lines.append(f"- **合格口径的 NPU/CPU 比值:{speedup:.4f}×（{verdict}）**")
        lines.append("- 数值、provenance 与 CPU 基线一致性门禁均已通过。")
    else:
        lines.append("- **未发布 NPU 加速比。**")
        lines.append("- 该结果不得作为 NPU 加速证据；诊断时延仅用于排查。")
        for issue in result.get("performance_issues") or []:
            lines.append(f"  - {issue}")
    return "\n".join(lines) + "\n"


def save_migration_report(info: dict, reports_dir=None) -> str:
    """渲染并以 no-clobber 语义落盘报告，返回绝对路径。

    微秒时间戳降低顺序调用的文件名碰撞概率；独占创建与有界后缀重试
    则保证同一任务在同一时刻并发写报告时也不会覆盖已有证据。
    """
    reports_dir = reports_dir or default_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now()
    info = dict(info)
    info.setdefault("timestamp", stamp.strftime("%Y-%m-%d %H:%M:%S"))
    raw_net = str(info.get("net", "task"))
    net = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_net).strip("._")[:100]
    net = net or "task"
    filename_stem = (
        f"migration_{net}_{stamp.strftime('%Y%m%d_%H%M%S_%f')}"
    )
    payload = build_report_md(info)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    for collision in range(10_000):
        suffix = "" if collision == 0 else f"_{collision}"
        fname = f"{filename_stem}{suffix}.md"
        path = os.path.abspath(os.path.join(reports_dir, fname))
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as report_file:
                descriptor = -1
                report_file.write(payload)
                report_file.flush()
                os.fsync(report_file.fileno())
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return path

    raise FileExistsError(
        "cannot allocate a unique migration report filename after 10000 attempts"
    )
