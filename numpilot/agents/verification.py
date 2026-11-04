"""验证智能体：独立执行原始/迁移产物并做严格语义比较。"""
import inspect
import os
import random
import time
import traceback as traceback_module
from dataclasses import replace

import numpy as np

from numpilot.backends.base import ExecutionBackend
from numpilot.backends.shim import ShimBackend
from numpilot.comparison import ComparisonResult, compare_outputs
from numpilot.capabilities import strict_receipt_issues
from numpilot.contracts import VerificationReport
from numpilot.engines.ast_rewrite import audit_forward_surface
from numpilot.module_injection import temporary_module, traced_module


def _blas_threads():
    """尽量返回 NumPy 当前真实线程池大小，缺依赖时退回环境变量。"""
    try:
        from threadpoolctl import threadpool_info

        pools = [
            item.get("num_threads")
            for item in threadpool_info()
            if item.get("user_api") in {"blas", "openmp"} and item.get("num_threads")
        ]
        if pools:
            return max(int(value) for value in pools)
    except (ImportError, RuntimeError):
        pass
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return None


def _load_namespace(source: str, filename: str) -> dict:
    namespace = {"__name__": "<numpilot-artifact>", "__file__": filename}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def _normalise_inputs(value):
    """Accept the tuple/list contract and the benchmark's named-dict contract.

    ``build_inputs()`` may return a tuple/list (legacy contract, bound
    positionally) or a ``dict`` (benchmark contract, bound by parameter name).
    The dict is copied so neither side can mutate the caller's object.
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("build_inputs() 的 dict 键必须是字符串")
        return dict(value)
    raise TypeError("build_inputs() 必须返回 tuple、list 或 dict")


def _validate_dict_inputs(forward, inputs):
    """Fail closed when the named inputs do not bind to ``forward``."""
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "forward 签名不可检查，无法按名字绑定 build_inputs() 的 dict"
        ) from exc
    try:
        signature.bind(**inputs)
    except TypeError as exc:
        raise TypeError(
            "build_inputs() 的 dict 键与 forward 参数不匹配: %s" % exc
        ) from exc


def _invoke_forward(forward, inputs):
    """Invoke ``forward`` with either positional or named inputs."""
    if isinstance(inputs, dict):
        _validate_dict_inputs(forward, inputs)
        return forward(**inputs)
    return forward(*inputs)


def _to_host_tree(backend, value):
    if isinstance(value, tuple):
        return tuple(_to_host_tree(backend, item) for item in value)
    if isinstance(value, list):
        return [_to_host_tree(backend, item) for item in value]
    if isinstance(value, dict):
        return {key: _to_host_tree(backend, item) for key, item in value.items()}
    if hasattr(value, "to_numpy"):
        return backend.to_host(value)
    return value


def _to_device_tree(backend, value):
    if isinstance(value, tuple):
        return tuple(_to_device_tree(backend, item) for item in value)
    if isinstance(value, list):
        return [_to_device_tree(backend, item) for item in value]
    if isinstance(value, dict):
        return {key: _to_device_tree(backend, item) for key, item in value.items()}
    if hasattr(value, "to_numpy"):
        return value
    if isinstance(value, np.ndarray):
        return backend.to_device(value)
    return value


def _array_span(array):
    """返回数组覆盖的字节区间 ``[start, end)``。

    负 stride 会把起点前移，因此不能只看 data 指针；空维度不产生访问，
    不参与区间扩展。
    """
    start = array.__array_interface__["data"][0]
    end = start + array.dtype.itemsize
    for length, stride in zip(array.shape, array.strides):
        if length <= 1:
            continue
        extent = (length - 1) * stride
        if extent >= 0:
            end += extent
        else:
            start += extent
    return start, end


def _clone_array_group(arrays):
    """在同一个缓冲区里重建一组共享存储的数组。

    只对数值 dtype 使用字节缓冲；object dtype 的引用计数语义无法用
    裸字节安全重建，退化为彼此独立的 ``copy(order="K")``（此时本就不
    存在可迁移的数值别名语义）。
    """
    if any(array.dtype.hasobject for array in arrays):
        return [array.copy(order="K") for array in arrays]

    spans = [_array_span(array) for array in arrays]
    base_address = min(span[0] for span in spans)
    total = max(span[1] for span in spans) - base_address
    buffer = np.empty(max(total, 1), dtype=np.uint8)
    clones = []
    for array in arrays:
        # ``offset`` 是首元素地址；负 stride 的首元素位于区间高端，
        # 不能用 _array_span 的低端起点。
        offset = array.__array_interface__["data"][0] - base_address
        clone = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=buffer,
            offset=offset,
            strides=array.strides,
        )
        # 重叠视图的内容彼此一致，写入顺序不影响结果。
        clone[...] = array
        if not array.flags.writeable:
            clone.flags.writeable = False
        clones.append(clone)
    return clones


def _collect_arrays(value, found):
    if isinstance(value, dict):
        for item in value.values():
            _collect_arrays(item, found)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_arrays(item, found)
    elif isinstance(value, np.ndarray):
        found.append(value)


def _replace_arrays(value, replacements):
    if isinstance(value, tuple):
        return tuple(_replace_arrays(item, replacements) for item in value)
    if isinstance(value, list):
        return [_replace_arrays(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_arrays(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        clone = replacements.get(id(value))
        if clone is None:
            return value.copy(order="K")
        return clone
    return value


def _clone_tree(value):
    """克隆双端规范输入，同时保留共享存储、偏移、strides 与只读标记。

    直接对每个 ndarray 调用 ``copy()`` 会把重叠别名拆成独立内存，并默认
    转成 C 顺序，从而让 B014 这类原地别名陷阱与 Fortran/负 stride 输入
    在原始 oracle 里失去意义。这里按真实 ``shares_memory`` 分组，在同一个
    字节缓冲区上按原偏移和 strides 重建视图，再递归替换回输入树。
    """
    arrays = []
    _collect_arrays(value, arrays)
    unique = []
    seen = set()
    for array in arrays:
        if id(array) in seen:
            continue
        seen.add(id(array))
        unique.append(array)

    components = []
    for array in unique:
        merged = [array]
        remaining = []
        for component in components:
            if any(np.shares_memory(array, member) for member in component):
                merged.extend(component)
            else:
                remaining.append(component)
        remaining.append(merged)
        components = remaining

    replacements = {}
    for component in components:
        for original, clone in zip(component, _clone_array_group(component)):
            replacements[id(original)] = clone
    return _replace_arrays(value, replacements)


def _set_rng_state(state):
    np.random.set_state(state[0])
    random.setstate(state[1])


def _comparison_dict(result):
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


def _overall_comparison(input_result, kernel_result, artifact_result):
    """计算主体优先，其次输入完整性和产物端到端；同时保留全部子报告。"""
    for label, result in (
        ("kernel", kernel_result),
        ("inputs", input_result),
        ("artifact", artifact_result),
    ):
        if not result.consistent:
            path = result.failure_path or "$"
            return replace(result, failure_path=f"{label}:{path}")
    return ComparisonResult(
        consistent=True,
        max_abs_diff=max(kernel_result.max_abs_diff, artifact_result.max_abs_diff),
        max_rel_diff=max(kernel_result.max_rel_diff, artifact_result.max_rel_diff),
        atol=max(kernel_result.atol, artifact_result.atol),
        rtol=max(kernel_result.rtol, artifact_result.rtol),
        n_elements=kernel_result.n_elements,
    )


def _time_call(fn, inputs, sync=None, warmup=3, iters=10):
    """测 ``fn(*inputs)`` / ``fn(**inputs)`` 的中位耗时；sync 用于强制异步后端完成。"""
    for _ in range(warmup):
        result = _invoke_forward(fn, inputs)
        if sync is not None:
            sync(result)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        result = _invoke_forward(fn, inputs)
        if sync is not None:
            sync(result)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


class VerificationAgent:
    def __init__(self, backend: ExecutionBackend = None):
        self.backend = backend or ShimBackend()

    def verify(self, orig_source: str, migrated_source: str,
               dtype=np.float32, measure: bool = True,
               required_ops=None) -> VerificationReport:
        """运行双端产物。

        ``dtype`` 仅为旧调用兼容参数；容差始终取自每个实际输出叶子的 dtype。
        迁移侧会调用迁移源码自己的 ``build_inputs``，随后直接调用其 ``forward``；
        测速时才另建一份驻留设备的输入，用于 kernel-only 口径。
        """
        del dtype
        phase = "original.load"
        backend_module = None
        backend_version = None
        provenance = {}
        profile_guard = None
        artifact_inputs_executed = False
        rng_initial = (np.random.get_state(), random.getstate())
        try:
            phase = "migrated.backend"
            ap_module = self.backend.get_module()
            backend_module = getattr(ap_module, "__file__", None)
            backend_version = getattr(ap_module, "__version__", None)
            phase = "migrated.provenance"
            surface = audit_forward_surface(migrated_source)
            strict_surface = self.backend.name == "npu" or required_ops is not None
            if strict_surface and surface.violations:
                phase = "migrated.surface"
                raise RuntimeError(
                    "迁移后 forward 不满足 strict 执行表面: "
                    + "; ".join(surface.violations)
                )
            discovered_counts = dict(surface.ap_calls)
            discovered_ops = set(discovered_counts)
            if required_ops is not None:
                if hasattr(required_ops, "items"):
                    expected_counts = {
                        str(name): int(count) for name, count in required_ops.items()
                    }
                    if expected_counts != discovered_counts:
                        phase = "migrated.surface"
                        raise RuntimeError(
                            "迁移产物直接 ap.* callsite 与 MigrationResult 不一致: "
                            f"expected={expected_counts}, discovered={discovered_counts}"
                        )
                    expected_ops = set(expected_counts)
                else:
                    expected_ops = {str(name) for name in required_ops}
                if not expected_ops:
                    raise RuntimeError("迁移产物没有记录任何直接 ap.* 重写")
                unresolved = expected_ops - discovered_ops
                if unresolved:
                    raise RuntimeError(
                        "迁移产物缺少可静态追踪的直接 ap.* 调用: "
                        + ", ".join(sorted(unresolved))
                    )
            else:
                expected_ops = set()
            if self.backend.name == "npu" and not discovered_ops:
                raise RuntimeError("NPU 后端无法从迁移产物解析任何直接 ap.* 调用，已 fail closed")
            audited_ops = tuple(sorted(discovered_ops | expected_ops))
            guard_factory = getattr(self.backend, "prepare_profile_guard", None)
            validator = getattr(self.backend, "validate_required_ops", None)
            if (
                self.backend.name == "npu"
                and callable(guard_factory)
                and audited_ops
            ):
                phase = "migrated.capabilities"
                try:
                    profile_guard = guard_factory(audited_ops)
                except Exception as error:
                    failed_validation = getattr(error, "validation", None)
                    if failed_validation is not None:
                        provenance = failed_validation.to_dict()
                    raise
                provenance = profile_guard.to_dict()
            elif callable(validator) and audited_ops:
                phase = "migrated.capabilities"
                try:
                    validation = validator(
                        audited_ops,
                        strict_npu=self.backend.name == "npu",
                        raise_on_error=True,
                    )
                except Exception as error:
                    failed_validation = getattr(error, "validation", None)
                    if failed_validation is not None:
                        provenance = failed_validation.to_dict()
                    raise
                provenance = validation.to_dict()
                if self.backend.name == "npu":
                    receipt_issues = strict_receipt_issues(
                        provenance,
                        audited_ops,
                    )
                    if receipt_issues:
                        raise RuntimeError(
                            "strict capability receipt rejected: "
                            + "; ".join(receipt_issues)
                        )

            runtime_targets = (
                profile_guard.runtime_targets
                if profile_guard is not None else None
            )
            execution_module, runtime_call_counts = traced_module(
                ap_module,
                audited_ops,
                before_call=(
                    profile_guard.validate_call
                    if profile_guard is not None else None
                ),
                targets=runtime_targets,
                enforce_target_identity=self.backend.name == "npu",
            )

            # Bind strict backend targets before executing either user-authored
            # source.  Any subsequent monkeypatch is rejected before dispatch.
            phase = "original.load"
            orig_ns = _load_namespace(orig_source, "<original>")
            phase = "original.build_inputs"
            _set_rng_state(rng_initial)
            canonical_inputs = _normalise_inputs(orig_ns["build_inputs"]())
            original_forward_rng = (np.random.get_state(), random.getstate())
            phase = "original.forward"
            _set_rng_state(original_forward_rng)
            out_orig = _invoke_forward(
                orig_ns["forward"], _clone_tree(canonical_inputs)
            )
            t_orig = (
                _time_call(orig_ns["forward"], _clone_tree(canonical_inputs))
                if measure else None
            )

            with temporary_module("asnumpy", execution_module):
                phase = "migrated.load"
                migrated_ns = _load_namespace(migrated_source, "<migrated-artifact>")
                phase = "migrated.build_inputs"
                _set_rng_state(rng_initial)
                migrated_inputs = _normalise_inputs(migrated_ns["build_inputs"]())
                migrated_forward_rng = (np.random.get_state(), random.getstate())
                artifact_inputs_executed = True
                migrated_inputs_host = _to_host_tree(self.backend, migrated_inputs)
                input_comparison = compare_outputs(canonical_inputs, migrated_inputs_host)

                # build_inputs 不得为 forward 的 runtime trace 充值。
                for name in runtime_call_counts:
                    runtime_call_counts[name] = 0

                phase = "migrated.forward"
                forward = migrated_ns["forward"]
                # 第一层：两侧使用同一份规范输入，只验证迁移计算主体。
                _set_rng_state(original_forward_rng)
                out_common = _to_host_tree(
                    self.backend,
                    _invoke_forward(forward, _clone_tree(canonical_inputs)),
                )
                kernel_comparison = compare_outputs(out_orig, out_common)
                kernel_runtime_counts = dict(runtime_call_counts)

                # 第二层：直接运行产物自己的 build_inputs + forward，验证交付完整性。
                _set_rng_state(migrated_forward_rng)
                out_artifact = _to_host_tree(
                    self.backend,
                    _invoke_forward(forward, _clone_tree(migrated_inputs)),
                )
                artifact_comparison = compare_outputs(out_orig, out_artifact)
                artifact_runtime_counts = {
                    name: runtime_call_counts[name] - kernel_runtime_counts[name]
                    for name in runtime_call_counts
                }

                phase = "migrated.runtime_trace"
                if self.backend.name == "npu":
                    missing_paths = []
                    for path, observed in (
                        ("kernel", kernel_runtime_counts),
                        ("artifact", artifact_runtime_counts),
                    ):
                        missing = {
                            name: {"expected": count, "observed": observed.get(name, 0)}
                            for name, count in discovered_counts.items()
                            if observed.get(name, 0) < count
                        }
                        if missing:
                            missing_paths.append(f"{path}={missing}")
                    if missing_paths:
                        raise RuntimeError(
                            "strict-NPU 每条验证路径都必须执行全部静态 ap.* callsite: "
                            + "; ".join(missing_paths)
                        )
                provenance = dict(provenance)
                provenance["runtime_trace_ok"] = bool(
                    self.backend.name != "npu" or not missing_paths
                )
                if profile_guard is not None:
                    provenance.update(profile_guard.to_dict())
                    if (
                        profile_guard.profile_scoped
                        and not profile_guard.runtime_profile_ok
                    ):
                        raise RuntimeError(
                            "strict-NPU scoped capability was not validated "
                            "against a real runtime call"
                        )
                provenance["runtime_execution_ok"] = True
                provenance["runtime_call_counts"] = {
                    "kernel": kernel_runtime_counts,
                    "artifact": artifact_runtime_counts,
                }

                if measure:
                    # kernel-only：输入先驻留设备，计时末尾递归 D2H 强制同步。
                    device_inputs = _to_device_tree(self.backend, migrated_inputs)
                    t_migrated = _time_call(
                        forward,
                        device_inputs,
                        sync=lambda value: _to_host_tree(self.backend, value),
                    )
                else:
                    t_migrated = None
            phase = "compare"
            comparison = _overall_comparison(
                input_comparison, kernel_comparison, artifact_comparison
            )
            return self._report(
                comparison,
                # 双端验证进程不具备独立 benchmark 的线程/provenance 口径，
                # 因此永不在这里发布 speedup。
                speedup=None,
                t_orig=t_orig,
                t_migrated=t_migrated,
                backend_module=backend_module,
                backend_version=backend_version,
                provenance=provenance,
                artifact_inputs_executed=artifact_inputs_executed,
                input_comparison=input_comparison,
                kernel_comparison=kernel_comparison,
                artifact_comparison=artifact_comparison,
            )
        except Exception as error:  # 验证失败是结构化结果，不让编排进程直接崩溃。
            if profile_guard is not None:
                provenance = dict(provenance)
                provenance.update(profile_guard.to_dict())
            if self.backend.name == "npu":
                provenance["runtime_execution_ok"] = False
                provenance["ok"] = False
                provenance["strict_npu"] = False
            report = self._exception_report(
                phase,
                error,
                backend_module=backend_module,
                backend_version=backend_version,
                provenance=provenance,
                artifact_inputs_executed=artifact_inputs_executed,
            )
            _set_rng_state(rng_initial)
            return report
        finally:
            _set_rng_state(rng_initial)

    def _report(self, comparison: ComparisonResult, **kwargs):
        t_orig = kwargs.pop("t_orig")
        t_migrated = kwargs.pop("t_migrated")
        input_comparison = kwargs.pop("input_comparison")
        kernel_comparison = kwargs.pop("kernel_comparison")
        artifact_comparison = kwargs.pop("artifact_comparison")
        return VerificationReport(
            numerically_consistent=comparison.consistent,
            max_rel_diff=comparison.max_rel_diff,
            max_abs_diff=comparison.max_abs_diff,
            tol=comparison.tol,
            atol=comparison.atol,
            rtol=comparison.rtol,
            speedup=kwargs.pop("speedup"),
            per_op=[],
            status="pass" if comparison.consistent else "fail",
            backend=self.backend.name,
            t_orig_ms=t_orig * 1e3 if t_orig is not None else None,
            t_mig_ms=t_migrated * 1e3 if t_migrated is not None else None,
            n_elements=comparison.n_elements,
            failure_path=comparison.failure_path,
            failure_index=comparison.failure_index,
            failure_reason=comparison.reason,
            cpu_threads=_blas_threads() if t_orig is not None else None,
            backend_provenance=kwargs.pop("provenance"),
            input_consistent=input_comparison.consistent,
            kernel_consistent=kernel_comparison.consistent,
            artifact_consistent=artifact_comparison.consistent,
            comparisons={
                "inputs": _comparison_dict(input_comparison),
                "kernel": _comparison_dict(kernel_comparison),
                "artifact": _comparison_dict(artifact_comparison),
            },
            **kwargs,
        )

    def _exception_report(self, phase, error, **kwargs):
        return VerificationReport(
            numerically_consistent=False,
            max_rel_diff=float(np.finfo(np.float64).max),
            max_abs_diff=float(np.finfo(np.float64).max),
            tol=0.0,
            atol=0.0,
            rtol=0.0,
            speedup=None,
            status="fail",
            backend=self.backend.name,
            exception_phase=phase,
            exception_type=type(error).__name__,
            exception_message=str(error),
            traceback="".join(
                traceback_module.format_exception(type(error), error, error.__traceback__)
            ),
            backend_provenance=kwargs.pop("provenance"),
            **kwargs,
        )
