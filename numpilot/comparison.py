"""NumPy/AsNumpy 输出的严格语义比较。

比较器刻意不使用“全局最大值作分母”的简化公式。它先验证容器结构、
shape 与 dtype，再按实际浮点 dtype 使用 ``atol + rtol * abs(reference)``；
整数、布尔及其他非浮点类型要求完全相等。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    consistent: bool
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    atol: float = 0.0
    rtol: float = 0.0
    failure_path: Optional[str] = None
    failure_index: Optional[str] = None
    reason: Optional[str] = None
    n_elements: int = 0

    @property
    def tol(self) -> float:
        """兼容旧报告中的单一 tolerance 字段。"""
        return max(self.atol, self.rtol)


def tolerance_for(dtype) -> tuple[float, float]:
    """返回 ``(atol, rtol)``，精度由真实输出 dtype 决定。"""
    dt = np.dtype(dtype)
    if dt.kind not in "fc":
        return 0.0, 0.0
    # complex 的 itemsize 是同精度实数的两倍。
    real_itemsize = dt.itemsize // 2 if dt.kind == "c" else dt.itemsize
    if real_itemsize <= 2:
        return 1e-2, 1e-2
    if real_itemsize <= 4:
        return 1e-5, 1e-4
    return 1e-9, 1e-7


def compare_outputs(reference, actual) -> ComparisonResult:
    """递归比较 tuple/list/dict 及数组/标量叶子。"""
    leaves = []
    failures = []
    _compare_node(reference, actual, "$", leaves, failures)
    return _aggregate(leaves, failures)


def _compare_node(reference, actual, path, leaves, failures):
    if isinstance(reference, dict):
        if not isinstance(actual, dict):
            failures.append(_failure(path, f"容器类型不同: dict != {type(actual).__name__}"))
            return
        reference_keys = set(reference)
        actual_keys = set(actual)
        if reference_keys != actual_keys:
            failures.append(_failure(
                path,
                "字典键不同: missing=%r, extra=%r"
                % (sorted(reference_keys - actual_keys, key=repr),
                   sorted(actual_keys - reference_keys, key=repr)),
            ))
        # dict 的语义与插入顺序无关；按 reference 顺序遍历公共键以稳定报告。
        for key in reference:
            if key in actual:
                _compare_node(
                    reference[key], actual[key], f"{path}[{key!r}]", leaves, failures
                )
        return

    if isinstance(reference, (tuple, list)):
        if type(actual) is not type(reference):
            failures.append(_failure(
                path,
                f"容器类型不同: {type(reference).__name__} != {type(actual).__name__}",
            ))
            return
        if len(reference) != len(actual):
            failures.append(
                _failure(path, f"容器长度不同: {len(reference)} != {len(actual)}")
            )
        for index, (ref_item, actual_item) in enumerate(zip(reference, actual)):
            _compare_node(ref_item, actual_item, f"{path}[{index}]", leaves, failures)
        return

    # 迁移侧叶子在进入此函数前应已回传主机；这里仍兼容公开 to_numpy 协议。
    if hasattr(reference, "to_numpy"):
        reference = reference.to_numpy()
    if hasattr(actual, "to_numpy"):
        actual = actual.to_numpy()
    ref = np.asarray(reference)
    got = np.asarray(actual)

    if ref.shape != got.shape:
        failures.append(_failure(path, f"shape 不同: {ref.shape} != {got.shape}"))
        return
    if ref.dtype != got.dtype:
        failures.append(_failure(path, f"dtype 不同: {ref.dtype} != {got.dtype}"))
        return

    atol, rtol = tolerance_for(ref.dtype)
    leaf = _leaf_metrics(ref, got, atol, rtol, path)
    leaves.append(leaf)
    if not leaf.consistent:
        failures.append(leaf)


def _leaf_metrics(ref, got, atol, rtol, path):
    n_elements = int(ref.size)
    if ref.dtype.kind not in "fc":
        equal = np.equal(ref, got)
        ok = bool(np.all(equal))
        index = None if ok else _first_bad_index(~equal)
        if ok:
            max_abs = 0.0
            max_rel = 0.0
        elif ref.dtype.kind in "biu":
            difference = np.abs(
                ref.astype(np.float64, copy=False) - got.astype(np.float64, copy=False)
            )
            max_abs = float(np.max(difference)) if difference.size else 0.0
            max_rel = max_abs
        else:
            max_abs = float(np.finfo(np.float64).max)
            max_rel = max_abs
        return ComparisonResult(
            consistent=ok,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            failure_path=None if ok else path,
            failure_index=index,
            reason=None if ok else "非浮点输出不完全相等",
            n_elements=n_elements,
        )

    close = np.isclose(ref, got, atol=atol, rtol=rtol, equal_nan=True)
    ok = bool(np.all(close))

    # NaN/Inf 由 isclose 处理；只对两侧有限的位置计算差值，避免报告出现 nan。
    finite = np.isfinite(ref) & np.isfinite(got)
    if ref.size and np.any(finite):
        abs_diff = np.abs(ref[finite] - got[finite])
        max_abs = float(np.max(abs_diff))
        ref_abs = np.abs(ref[finite])
        # 归一化误差使用 max(abs(reference), atol)，避免零参考产生 Infinity。
        # 是否通过仍严格由 np.isclose(atol + rtol*abs(reference)) 决定。
        floor = atol if atol > 0 else np.finfo(np.float64).tiny
        rel = abs_diff / np.maximum(ref_abs, floor)
        max_rel = float(np.max(rel))
    else:
        max_abs = 0.0
        max_rel = 0.0

    # NaN/Inf 模式不一致时不存在有意义的有限差值，明确报告为无穷大。
    if np.any((~close) & (~finite)):
        max_abs = float(np.finfo(np.float64).max)
        max_rel = max_abs

    index = None if ok else _first_bad_index(~close)
    return ComparisonResult(
        consistent=ok,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        atol=atol,
        rtol=rtol,
        failure_path=None if ok else path,
        failure_index=index,
        reason=None if ok else "浮点输出超出 atol + rtol * abs(reference)",
        n_elements=n_elements,
    )


def _first_bad_index(mask):
    if np.ndim(mask) == 0:
        return "()" if bool(mask) else None
    positions = np.argwhere(mask)
    if positions.size == 0:
        return None
    pos = tuple(int(v) for v in positions[0])
    return repr(pos)


def _failure(path, reason):
    return ComparisonResult(False, failure_path=path, reason=reason)


def _aggregate(leaves, failures):
    all_results = list(leaves) + [item for item in failures if item not in leaves]
    if not all_results:
        return ComparisonResult(not failures)
    first_failure = failures[0] if failures else None
    return ComparisonResult(
        consistent=not failures,
        max_abs_diff=max(item.max_abs_diff for item in all_results),
        max_rel_diff=max(item.max_rel_diff for item in all_results),
        atol=max(item.atol for item in all_results),
        rtol=max(item.rtol for item in all_results),
        failure_path=getattr(first_failure, "failure_path", None),
        failure_index=getattr(first_failure, "failure_index", None),
        reason=getattr(first_failure, "reason", None),
        n_elements=sum(item.n_elements for item in leaves),
    )
