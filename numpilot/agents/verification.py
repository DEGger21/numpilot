# numpilot/agents/verification.py
"""验证智能体:双端执行 + 数值对比 + 测速, 产出 VerificationReport。

原始(np)源码与迁移后(ap)源码各自 exec, 用原始 build_inputs() 造 numpy 输入,
分别调用两侧 forward(*inputs) 比对数值。迁移侧把执行后端模块临时挂到
sys.modules["asnumpy"], 满足迁移源码的 `import asnumpy as ap`。
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


def _time_call(fn, *args, warmup=3, iters=10, **kw):
    for _ in range(warmup):
        fn(*args, **kw)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


class VerificationAgent:
    def __init__(self, backend: ExecutionBackend = None):
        self.backend = backend or ShimBackend()

    def verify(self, orig_source: str, migrated_source: str,
               dtype=np.float32) -> VerificationReport:
        tol = _tol_for(dtype)

        # 原始 numpy 侧
        orig_ns = _load_namespace(orig_source)
        inputs = orig_ns["build_inputs"]()
        out_orig = _to_numpy(orig_ns["forward"](*inputs))
        t_orig = _time_call(orig_ns["forward"], *inputs)

        # 迁移侧:把后端模块挂成 asnumpy 供迁移源码 import
        ap_module = self.backend.get_module()
        saved = sys.modules.get("asnumpy")
        sys.modules["asnumpy"] = ap_module
        try:
            mig_ns = _load_namespace(migrated_source)
            out_mig = _to_numpy(mig_ns["forward"](*inputs))
            t_mig = _time_call(mig_ns["forward"], *inputs)
        finally:
            if saved is not None:
                sys.modules["asnumpy"] = saved
            else:
                sys.modules.pop("asnumpy", None)

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
        )
