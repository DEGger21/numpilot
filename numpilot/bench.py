"""可审计的 CPU/NPU 性能评测。

``bench-cpu`` 在独立进程中生成 CPU 基线；``bench-npu`` 只有在迁移后
算子的执行来源均可证明为严格 NPU 时才发布加速比。已知 CPU fallback、
计划中的 CPU placement 或 provenance 未知都会默认拒绝。诊断时可显式传入
``--allow-cpu-fallback`` 继续执行，但结果会标记为不具备 NPU 性能证明效力，
且不会计算或展示 NPU 加速比。
"""
import hashlib
import json
import math
import os
import time

import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.backends.base import select_backend
from numpilot.comparison import compare_outputs
from numpilot.engines.ast_rewrite import audit_forward_surface
from numpilot.module_injection import temporary_module, traced_module
from numpilot.report import build_benchmark_report_md

_BASELINE_JSON = "cpu_baseline.json"
_BASELINE_NPY = "cpu_baseline.npy"
_BENCHMARK_JSON = "npu_benchmark.json"
_BENCHMARK_MD = "benchmark_report.md"
_REJECTED = 3


def _net_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _source_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _speedup_verdict(speedup):
    if speedup >= 1:
        return f"NPU 快 {speedup:.2f}x"
    if speedup > 0:
        return f"CPU 快 {1.0 / speedup:.2f}x"
    return "比值无效"


def _load_namespace(source):
    ns = {}
    exec(compile(source, "<bench>", "exec"), ns)
    return ns


def _to_numpy(x):
    if isinstance(x, dict):
        return {key: _to_numpy(value) for key, value in x.items()}
    if isinstance(x, tuple):
        return tuple(_to_numpy(value) for value in x)
    if isinstance(x, list):
        return [_to_numpy(value) for value in x]
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)


def _to_host(backend, value):
    """递归同步并回传嵌套输出，供计时与严格比较共同使用。"""
    if isinstance(value, dict):
        return {key: _to_host(backend, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_host(backend, item) for item in value)
    if isinstance(value, list):
        return [_to_host(backend, item) for item in value]
    # CPU fallback、标量或混合输出可能已经是 host value；此时不能假定存在
    # ``to_numpy`` 并再次调用 NPU backend.to_host。
    return backend.to_host(value) if hasattr(value, "to_numpy") else value


def _clone_tree(value):
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _to_device(backend, value):
    if isinstance(value, dict):
        return {key: _to_device(backend, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(backend, item) for item in value)
    if isinstance(value, list):
        return [_to_device(backend, item) for item in value]
    if hasattr(value, "to_numpy"):
        return value
    if isinstance(value, np.ndarray):
        return backend.to_device(value)
    return value


def _median_ms(fn, args, sync=None, warmup=3, iters=10):
    """``fn(*args)`` 的中位耗时；sync 用于等待异步设备完成。"""
    for _ in range(warmup):
        result = fn(*args)
        if sync is not None:
            sync(result)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn(*args)
        if sync is not None:
            sync(result)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def _env_thread_limit():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value and value.isdigit() and int(value) > 0:
            return int(value), key
    return None, None


def _blas_runtime_info():
    """返回运行时可观测的 BLAS 线程池；threadpoolctl 不可用时保守降级。

    环境变量只是请求值，不能冒充运行库的实际状态。因此仅当
    ``threadpoolctl`` 返回线程池时将 source 标为 ``threadpoolctl``；降级信息会
    明确标为 ``environment`` 或 ``unavailable``。
    """
    try:
        from threadpoolctl import threadpool_info

        raw_pools = threadpool_info()
    except Exception:  # optional observer must never make the benchmark unusable
        raw_pools = []

    pools = []
    for item in raw_pools:
        if item.get("user_api") != "blas":
            continue
        pools.append({
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "threading_layer": item.get("threading_layer"),
            "architecture": item.get("architecture"),
        })
    observed = [
        item["num_threads"] for item in pools
        if isinstance(item.get("num_threads"), int) and item["num_threads"] > 0
    ]
    if observed:
        return {
            "source": "threadpoolctl",
            "effective_threads": max(observed),
            "pools": pools,
        }

    requested, key = _env_thread_limit()
    if requested is not None:
        return {
            "source": "environment",
            "effective_threads": requested,
            "environment_key": key,
            "pools": [],
        }
    return {"source": "unavailable", "effective_threads": None, "pools": []}


def _blas_threads_now():
    """兼容旧调用者：返回最佳可观测线程数，而非推测的 CPU 总核数。"""
    return _blas_runtime_info()["effective_threads"]


def _plain_validation(validation):
    if validation is None:
        return {}
    if isinstance(validation, dict):
        return dict(validation)
    to_dict = getattr(validation, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    fields = (
        "ok", "missing", "non_strict", "required_ops", "backend_version",
        "capabilities", "audit_id", "strict_npu", "environment",
    )
    return {name: getattr(validation, name) for name in fields if hasattr(validation, name)}


def _names(items):
    names = []
    for item in items or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            names.append(str(item.get("name") or item.get("op") or item))
        else:
            names.append(str(getattr(item, "name", item)))
    return sorted(set(names))


def _audit_execution_sources(backend, plan, required_ops):
    """兼容消费 NPU backend 的 capability audit；缺接口时 fail closed。"""
    validation = None
    audit_source = None
    error = None

    validate_ops = getattr(backend, "validate_required_ops", None)
    validate_plan = getattr(backend, "validate_plan", None)
    try:
        if callable(validate_ops):
            audit_source = f"{type(backend).__name__}.validate_required_ops"
            validation = validate_ops(
                required_ops, strict_npu=True, raise_on_error=False)
        elif callable(validate_plan):
            audit_source = f"{type(backend).__name__}.validate_plan"
            validation = validate_plan(plan, strict_npu=True, raise_on_error=False)
    except Exception as exc:  # capability 检查失败必须保守拒绝，但保留诊断。
        error = f"{type(exc).__name__}: {exc}"

    data = _plain_validation(validation)
    missing = _names(data.get("missing"))
    non_strict = _names(data.get("non_strict"))
    capabilities = data.get("capabilities") or {}

    cpu_fallback = []
    unknown_non_strict = []
    provenance_by_op = {}
    if isinstance(capabilities, dict):
        for name, capability in capabilities.items():
            cap = capability
            if hasattr(cap, "to_dict"):
                cap = cap.to_dict()
            if not isinstance(cap, dict):
                continue
            provenance = str(
                cap.get("provenance") or cap.get("implementation") or cap.get("backend") or ""
            ).lower()
            provenance_by_op[str(name)] = provenance or "unknown"
            if str(name) not in non_strict:
                continue
            if not provenance or provenance == "unknown":
                unknown_non_strict.append(str(name))
            else:
                # host_assisted/conditional fallback 同样不能作为纯 NPU 性能证据。
                cpu_fallback.append(str(name))
    # 旧 manifest 若只返回 non_strict 名单却没有 provenance，按来源未知处理。
    unresolved_non_strict = set(non_strict) - set(provenance_by_op)
    unknown_non_strict.extend(unresolved_non_strict)
    cpu_fallback = sorted(set(cpu_fallback))
    missing = sorted(set(missing) | set(unknown_non_strict))

    if validation is None:
        missing = sorted(set(missing) | set(required_ops))
    elif data.get("ok") is False and not missing and not non_strict:
        # 无法解释的 audit 失败仍按 provenance unknown 处理。
        missing = sorted(set(required_ops))

    planned_cpu = sorted(
        name for name, placement in plan.placement.items() if placement != "NPU"
    )
    issues = []
    if planned_cpu:
        issues.append("计划包含 CPU placement: " + ", ".join(planned_cpu))
    if cpu_fallback:
        issues.append("已知非严格 NPU/CPU fallback: " + ", ".join(cpu_fallback))
    if missing:
        issues.append("执行来源未知: " + ", ".join(missing))
    if error:
        issues.append("capability audit 失败: " + error)
    if validation is not None and data.get("strict_npu") is not True:
        issues.append("capability audit 未明确声明 strict_npu=true")

    return {
        "ok": not issues,
        "audit_source": audit_source,
        "audit_error": error,
        "required_ops": sorted(set(required_ops)),
        "planned_cpu_ops": planned_cpu,
        "cpu_fallback_ops": cpu_fallback,
        "unknown_ops": missing,
        "backend_version": data.get("backend_version"),
        "audit_id": data.get("audit_id"),
        "environment": data.get("environment") or {},
        "strict_npu": data.get("strict_npu") is True,
        "provenance_by_op": provenance_by_op,
        "capabilities": capabilities,
        "issues": issues,
    }


def _comparison_data(comparison):
    def json_number(value):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value

    return {
        "consistent": comparison.consistent,
        "max_abs_diff": json_number(comparison.max_abs_diff),
        "max_rel_diff": json_number(comparison.max_rel_diff),
        "atol": json_number(comparison.atol),
        "rtol": json_number(comparison.rtol),
        "failure_path": comparison.failure_path,
        "failure_index": comparison.failure_index,
        "reason": comparison.reason,
        "n_elements": comparison.n_elements,
    }


def _write_benchmark_artifacts(out_dir, result):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, _BENCHMARK_JSON)
    md_path = os.path.join(out_dir, _BENCHMARK_MD)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with open(md_path, "w", encoding="utf-8") as stream:
        stream.write(build_benchmark_report_md(result))
    return json_path, md_path


# ───────────────────────── ① CPU 基线 ─────────────────────────
def run_cpu_baseline(file, out_dir, cpu_threads):
    with open(file, encoding="utf-8") as stream:
        source = stream.read()
    namespace = _load_namespace(source)
    inputs = namespace["build_inputs"]()
    out = _to_numpy(namespace["forward"](*inputs))
    if not isinstance(out, np.ndarray):
        raise TypeError("bench-cpu 当前要求 forward 返回单个 ndarray")
    t_cpu = _median_ms(namespace["forward"], inputs)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, _BASELINE_NPY), out)
    blas = _blas_runtime_info()
    threads = blas["effective_threads"]
    meta = {
        "file": file,
        "source_sha256": _source_hash(source),
        "net": _net_name(file),
        "t_cpu_ms": t_cpu,
        "n_elements": int(out.size),
        "output_shape": list(out.shape),
        "cpu_threads": threads,
        "cpu_threads_requested": cpu_threads,
        "blas": blas,
        "dtype": str(out.dtype),
    }
    with open(os.path.join(out_dir, _BASELINE_JSON), "w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)

    thread_label = str(threads) if threads is not None else "未知"
    print(f"── NumPilot CPU 基线 · {_net_name(file)} " + "─" * max(4, 30 - len(_net_name(file))))
    print(
        f"[CPU] numpy@CPU · BLAS 线程={thread_label} "
        f"(观测来源={blas['source']}, 逻辑 CPU={os.cpu_count()})"
    )
    print(f"[性能] {out.size:,} 元素   numpy forward = {t_cpu:.3f} ms")
    print(f"[基线] 已存 {os.path.join(out_dir, _BASELINE_NPY)}(供 ② bench-npu 对比)")
    return 0


# ───────────────────────── ② NPU 对比 ─────────────────────────
def run_npu_compare(
        file, out_dir, engine=None, threshold=1_000_000, allow_cpu_fallback=False):
    with open(file, encoding="utf-8") as stream:
        source = stream.read()

    analysis = AnalysisAgent().analyze(source, file)
    net = _net_name(file)
    print(f"── NumPilot NPU 对比 · {net} " + "─" * max(4, 30 - len(net)))
    if not analysis.migratable:
        print(f"[分析] 命中 {len(analysis.unsupported)} 个覆盖集缺口 ✗")
        print("[缺口] " + ", ".join(analysis.unsupported) + "  (无法迁移)")
        return 1

    plan = DecisionAgent(threshold).plan(analysis)
    migration = MigrationAgent(engine).migrate(source, plan)
    surface = audit_forward_surface(migration.migrated_source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "migrated.py"), "w", encoding="utf-8") as stream:
        stream.write(migration.migrated_source)
    total = migration.total_rewrites
    items = "  ".join(
        f"{op}×{count}" for op, count in
        sorted(migration.rewrites.items(), key=lambda item: (-item[1], item[0])))
    print(f"[迁移] 引擎={migration.engine} · 替换 {total} 处 np.* → ap.*   {items}")

    backend = select_backend("npu")
    ap_module = backend.get_module()
    planned_npu_ops = {
        name for name, placement in plan.placement.items() if placement == "NPU"
    }
    required_ops = (
        set(migration.rewrites)
        | set(migration.expected_rewrites)
        | planned_npu_ops
    )
    provenance = _audit_execution_sources(backend, plan, required_ops)
    provenance["static_forward_surface"] = {
        "ok": surface.ok,
        "ap_calls": dict(surface.ap_calls),
        "violations": list(surface.violations),
    }
    if surface.violations:
        provenance["ok"] = False
        provenance["issues"].extend(
            "迁移后 forward 执行表面不合格: " + violation
            for violation in surface.violations
        )
    if not migration.rewrite_integrity:
        provenance["ok"] = False
        provenance["issues"].append(
            "迁移算子/callsite 与原始 forward + plan 推导结果不一致: "
            f"expected={migration.expected_rewrites}, actual={migration.rewrites}"
        )
    if migration.total_rewrites == 0:
        provenance["ok"] = False
        provenance["issues"].append("迁移产物没有任何 ap.* 调用，无法证明执行发生在 NPU")
    override_used = bool(provenance["issues"] and allow_cpu_fallback)
    result = {
        "schema_version": 1,
        "status": "rejected" if provenance["issues"] else "pending",
        "file": file,
        "source_sha256": _source_hash(source),
        "migrated_sha256": _source_hash(migration.migrated_source),
        "net": net,
        "engine": migration.engine,
        "backend": "npu",
        "backend_version": getattr(ap_module, "__version__", None),
        "backend_module": getattr(ap_module, "__file__", None),
        "rewrites": migration.rewrites,
        "expected_rewrites": migration.expected_rewrites,
        "rewrite_integrity": migration.rewrite_integrity,
        "allow_cpu_fallback": bool(allow_cpu_fallback),
        "fallback_override_used": override_used,
        "provenance": provenance,
        "performance_valid": False,
        "performance_issues": list(provenance["issues"]),
        "speedup": None,
        "comparison": None,
        "t_cpu_ms": None,
        "t_npu_ms": None,
        "cpu_baseline": None,
        "npu_process_blas": _blas_runtime_info(),
        "artifact_inputs_executed": False,
        "timing_scope": {
            "classification": "device_resident_inputs_plus_d2h_sync",
            "inputs_resident_on_device": True,
            "includes_h2d": False,
            "includes_d2h_sync_each_iteration": True,
            "kernel_only": False,
            "end_to_end": False,
            "description": "forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步",
        },
    }

    if provenance["issues"] and not allow_cpu_fallback:
        print("[门禁] 拒绝发布 NPU 性能结果 ✗")
        for issue in provenance["issues"]:
            print(f"       - {issue}")
        print("       如仅需诊断，可显式添加 --allow-cpu-fallback；该模式不会计算 NPU 加速比。")
        json_path, md_path = _write_benchmark_artifacts(out_dir, result)
        print(f"[证据] {json_path} · {md_path}")
        return _REJECTED

    if override_used:
        print("[门禁] 已显式允许 fallback/未知来源，仅运行诊断；结果不得作为 NPU 加速证据 ⚠")
        for issue in provenance["issues"]:
            print(f"       - {issue}")

    # NumPy 参考只用于正确性，不参与本进程计时。
    orig_ns = _load_namespace(source)
    inputs = orig_ns["build_inputs"]()
    reference = _to_numpy(orig_ns["forward"](*_clone_tree(inputs)))

    try:
        execution_module, runtime_call_counts = traced_module(
            ap_module, surface.ap_calls
        )
        with temporary_module("asnumpy", execution_module):
            migrated_ns = _load_namespace(migration.migrated_source)
            migrated_inputs = migrated_ns["build_inputs"]()
            result["artifact_inputs_executed"] = True
            forward = migrated_ns["forward"]
            input_comparison = compare_outputs(inputs, _to_numpy(migrated_inputs))
            for name in runtime_call_counts:
                runtime_call_counts[name] = 0

            # 同一规范输入验证迁移计算主体，防止通过篡改 build_inputs 补偿错误 forward。
            common_dev_inputs = _to_device(backend, _clone_tree(inputs))
            common_output = _to_host(backend, forward(*common_dev_inputs))
            kernel_comparison = compare_outputs(reference, _to_numpy(common_output))
            kernel_runtime_counts = dict(runtime_call_counts)

            # 独立执行产物自己的 build_inputs + forward，作为 artifact 完整性门禁。
            dev_inputs = _to_device(backend, _clone_tree(migrated_inputs))
            out_npu = _to_host(backend, forward(*dev_inputs))
            artifact_comparison = compare_outputs(reference, _to_numpy(out_npu))
            artifact_runtime_counts = {
                name: runtime_call_counts[name] - kernel_runtime_counts[name]
                for name in runtime_call_counts
            }
            trace_missing = {}
            for path, observed in (
                ("kernel", kernel_runtime_counts),
                ("artifact", artifact_runtime_counts),
            ):
                missing = {
                    name: {"expected": count, "observed": observed.get(name, 0)}
                    for name, count in surface.ap_calls.items()
                    if observed.get(name, 0) < count
                }
                if missing:
                    trace_missing[path] = missing
            provenance["runtime_call_counts"] = {
                "kernel": kernel_runtime_counts,
                "artifact": artifact_runtime_counts,
            }
            provenance["runtime_trace_ok"] = not trace_missing
            if trace_missing:
                raise RuntimeError(
                    "strict-NPU 每条验证路径都必须执行全部静态 ap.* callsite: "
                    + str(trace_missing)
                )
            t_npu = _median_ms(
                forward, dev_inputs, sync=lambda value: _to_host(backend, value))
    except Exception as exc:
        result["status"] = "runtime_error"
        result["runtime_error"] = f"{type(exc).__name__}: {exc}"
        result["performance_issues"].append("迁移侧运行失败: " + result["runtime_error"])
        print(f"[执行] 迁移侧运行失败 ✗  {result['runtime_error']}")
        json_path, md_path = _write_benchmark_artifacts(out_dir, result)
        print(f"[证据] {json_path} · {md_path}")
        return 4

    comparison = kernel_comparison
    result["input_comparison"] = _comparison_data(input_comparison)
    result["kernel_comparison"] = _comparison_data(kernel_comparison)
    result["artifact_comparison"] = _comparison_data(artifact_comparison)
    result["comparison"] = _comparison_data(comparison)
    result["t_npu_ms"] = t_npu
    result["status"] = (
        "pass_unqualified" if (
            input_comparison.consistent and kernel_comparison.consistent
            and artifact_comparison.consistent and provenance["issues"]
        )
        else "pass" if (
            input_comparison.consistent and kernel_comparison.consistent
            and artifact_comparison.consistent
        )
        else "fail"
    )
    result["performance_valid"] = bool(
        input_comparison.consistent
        and kernel_comparison.consistent
        and artifact_comparison.consistent
        and not provenance["issues"]
    )
    if not kernel_comparison.consistent:
        result["performance_issues"].append("同一规范输入上的迁移计算主体比较失败")
    if not input_comparison.consistent:
        result["performance_issues"].append("迁移产物 build_inputs 与原始输入不一致")
    if not artifact_comparison.consistent:
        result["performance_issues"].append("迁移产物端到端输出比较失败")

    print(
        f"[执行] 后端=npu · asnumpy {getattr(ap_module, '__version__', '?')} · "
        f"provenance={'strict-npu' if not provenance['issues'] else 'unqualified'}"
    )
    print(f"       {getattr(ap_module, '__file__', '?')}")
    all_consistent = (
        input_comparison.consistent
        and kernel_comparison.consistent
        and artifact_comparison.consistent
    )
    mark = "✓" if all_consistent else "✗"
    print(
        f"[校验] 数值一致 {mark}  max_abs_diff={comparison.max_abs_diff:.2e} "
        f"max_rel_diff={comparison.max_rel_diff:.2e} "
        f"(atol={comparison.atol:.0e}, rtol={comparison.rtol:.0e})"
    )
    if not kernel_comparison.consistent:
        print(
            f"       失败位置={comparison.failure_path or '?'}"
            f"{comparison.failure_index or ''} 原因={comparison.reason or '?'}"
        )
    timing_label = (
        "asnumpy@npu(输入驻留 + 每轮 D2H 同步)"
        if result["performance_valid"]
        else "迁移侧诊断时延(输入驻留 + D2H，同样非 NPU 性能证据)"
    )
    print(f"[性能] {comparison.n_elements:,} 元素   {timing_label} = {t_npu:.3f} ms")

    base_path = os.path.join(out_dir, _BASELINE_JSON)
    if os.path.exists(base_path):
        with open(base_path, encoding="utf-8") as stream:
            baseline = json.load(stream)
        result["cpu_baseline"] = baseline
        result["t_cpu_ms"] = baseline.get("t_cpu_ms")
        baseline_output_comparison = None
        baseline_output_path = os.path.join(out_dir, _BASELINE_NPY)
        try:
            baseline_output = np.load(baseline_output_path, allow_pickle=False)
            baseline_output_comparison = compare_outputs(reference, baseline_output)
            result["baseline_output_comparison"] = _comparison_data(
                baseline_output_comparison)
        except (OSError, ValueError, TypeError) as exc:
            result["baseline_output_error"] = f"{type(exc).__name__}: {exc}"
        baseline_ok = (
            baseline.get("source_sha256") == result["source_sha256"]
            and baseline.get("n_elements") == comparison.n_elements
            and baseline_output_comparison is not None
            and baseline_output_comparison.consistent
        )
        result["baseline_compatible"] = baseline_ok
        if not baseline_ok:
            result["performance_valid"] = False
            result["performance_issues"].append("CPU 基线与当前源码或参考输出不一致")
            if all_consistent:
                result["status"] = "pass_unqualified"
            print("[对比] CPU 基线与当前源码/输出规模不一致，不计算加速比 ✗")
        elif result["performance_valid"] and t_npu > 0:
            speedup = baseline["t_cpu_ms"] / t_npu
            result["speedup"] = speedup
            threads = baseline.get("cpu_threads")
            thread_label = str(threads) if threads is not None else "未知"
            print(
                f"[对比] numpy@CPU(BLAS线程={thread_label}) {baseline['t_cpu_ms']:.3f} ms "
                f"vs strict-asnumpy@npu {t_npu:.3f} ms"
            )
            print(
                f"       → 合格口径 NPU/CPU 比值 {speedup:.2f}x "
                f"({_speedup_verdict(speedup)}) [基线来自 ① bench-cpu]"
            )
        else:
            print("[对比] provenance 或数值门禁未通过，不计算 NPU 加速比")
    else:
        result["baseline_compatible"] = None
        result["performance_valid"] = False
        result["performance_issues"].append("缺少与当前源码绑定的 CPU 基线")
        if all_consistent:
            result["status"] = "pass_unqualified"
        print(f"[对比] 未找到 CPU 基线 {base_path}")
        print(f"       请先跑:python -m numpilot bench-cpu {file} --out {out_dir}")

    json_path, md_path = _write_benchmark_artifacts(out_dir, result)
    print(f"[证据] {json_path} · {md_path}")
    return 0 if all_consistent else 2
