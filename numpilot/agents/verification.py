# numpilot/agents/verification.py
"""验证智能体:双端执行 + 数值对比 + 测速, 产出 VerificationReport。

原始(np)源码与迁移后(ap)源码各自 exec, 用原始 build_inputs() 造 numpy 输入。
迁移侧把执行后端模块临时挂到 sys.modules["asnumpy"], 满足迁移源码的
`import asnumpy as ap`。

真机(NPU)对齐两点(参考 RUN_LLM_TEMP Part 3):
1. **输入先 to_device**:asnumpy 混合类型(NPUArray + numpy)算子会报错,且把 H2D
   拷贝算进计时不公平。故计时前把全部输入搬到设备,forward 全程在设备上算。
2. **计时内同步**:NPU 执行异步,`perf_counter` 裸测测不到真实计算。每次迭代末尾
   调用 backend.to_host() 强制 completion,才是"numpy@CPU vs asnumpy@NPU"的真时间。
"""
import sys
import time

import numpy as np

from numpilot.backends.base import ExecutionBackend
from numpilot.backends.shim import ShimBackend
from numpilot.contracts import VerificationReport

_TOL = {"float32": 1e-4, "float16": 1e-2}


def _tol_for(dtype) -> float:
    return _TOL.get(np.dtype(dtype).name, 0.0)


def _load_namespace(source: str) -> dict:
    ns = {}
    exec(compile(source, "<migrated>", "exec"), ns)
    return ns


def _to_numpy(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)


def _time_call(fn, args, sync=None, warmup=3, iters=10):
    """测 fn(*args) 的中位耗时(秒)。sync(result) 用于强制异步后端同步完成。"""
    for _ in range(warmup):
        r = fn(*args)
        if sync is not None:
            sync(r)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn(*args)
        if sync is not None:
            sync(r)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


class VerificationAgent:
    def __init__(self, backend: ExecutionBackend = None):
        self.backend = backend or ShimBackend()

    def verify(self, orig_source: str, migrated_source: str,
               dtype=np.float32) -> VerificationReport:
        tol = _tol_for(dtype)

        # ── 原始 numpy@CPU 侧 ───────────────────────────────
        orig_ns = _load_namespace(orig_source)
        inputs = orig_ns["build_inputs"]()
        out_orig = _to_numpy(orig_ns["forward"](*inputs))
        t_orig = _time_call(orig_ns["forward"], inputs)   # numpy 本身同步

        # ── 迁移 asnumpy@后端 侧 ────────────────────────────
        ap_module = self.backend.get_module()
        saved = sys.modules.get("asnumpy")
        sys.modules["asnumpy"] = ap_module
        try:
            mig_ns = _load_namespace(migrated_source)
            # 输入搬到设备(H2D 移出计时;避免 NPUArray+numpy 混合报错)
            dev_inputs = tuple(self.backend.to_device(_to_numpy(x)) for x in inputs)
            forward = mig_ns["forward"]
            out_mig = self.backend.to_host(forward(*dev_inputs))
            # 计时:每次迭代 to_host 强制设备完成(真时间)
            t_mig = _time_call(forward, dev_inputs, sync=self.backend.to_host)
        finally:
            if saved is not None:
                sys.modules["asnumpy"] = saved
            else:
                sys.modules.pop("asnumpy", None)

        out_mig = _to_numpy(out_mig)
        denom = np.max(np.abs(out_orig))
        max_rel = float(np.max(np.abs(out_orig - out_mig)) / denom) if denom > 0 else 0.0
        consistent = max_rel <= tol
        speedup = float(t_orig / t_mig) if t_mig > 0 else None

        return VerificationReport(
            numerically_consistent=consistent,
            max_rel_diff=max_rel,
            tol=tol,
            speedup=speedup,
            per_op=[],
            status="pass" if consistent else "fail",
            backend=self.backend.name,
            t_orig_ms=t_orig * 1e3,
            t_mig_ms=t_mig * 1e3,
            n_elements=int(out_orig.size),
        )
