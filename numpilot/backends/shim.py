"""numpy 支撑的假 asnumpy 后端。让迁移后的 `import asnumpy as ap` 代码
在任意机器离线跑通, 数值与 numpy 恒等, 便于复现与 CI。"""
import types

import numpy as np

from numpilot.backends.base import ExecutionBackend


class _NDArray:
    """对标 asnumpy.NPUArray 的最小包装, 内部用 numpy 承载。"""
    def __init__(self, data):
        self._data = np.asarray(data)

    @staticmethod
    def from_numpy(arr):
        return _NDArray(arr)

    def to_numpy(self):
        return np.asarray(self._data)


def _unwrap(x):
    return x._data if isinstance(x, _NDArray) else x


def _wrap(x):
    return _NDArray(x)


def _make_module():
    m = types.ModuleType("asnumpy")

    def _u(name, npfunc):
        setattr(m, name, lambda a, *args, **kw: _wrap(npfunc(_unwrap(a), *args, **kw)))

    def _b(name, npfunc):
        setattr(m, name, lambda a, b, *args, **kw: _wrap(npfunc(_unwrap(a), _unwrap(b), *args, **kw)))

    # 二元 / 通用
    for name in ["add", "subtract", "multiply", "divide", "matmul", "dot",
                 "maximum", "minimum", "power"]:
        _b(name, getattr(np, name))
    # 一元
    for name in ["exp", "sqrt", "square", "tanh", "sin", "cos", "log", "abs"]:
        _u(name, getattr(np, name))
    # 归约(带 axis 透传)
    for name in ["sum", "mean", "max", "min"]:
        _u(name, getattr(np, name))

    # 非 numpy 原生算子
    m.relu = lambda a: _wrap(np.maximum(_unwrap(a), 0))

    def _softmax(a, axis=-1):
        x = _unwrap(a)
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return _wrap(e / np.sum(e, axis=axis, keepdims=True))
    m.softmax = _softmax

    # 创建
    m.zeros = lambda *a, **k: _wrap(np.zeros(*a, **k))
    m.ones = lambda *a, **k: _wrap(np.ones(*a, **k))

    m.ndarray = _NDArray
    return m


class ShimBackend(ExecutionBackend):
    name = "shim"

    def __init__(self):
        self._module = _make_module()

    def get_module(self):
        return self._module

    def to_device(self, np_array):
        return _NDArray.from_numpy(np_array)

    def to_host(self, device_array):
        return device_array.to_numpy()
