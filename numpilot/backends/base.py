"""执行后端端口:让验证阶段在不同后端(shim / 真实 NPU)运行迁移代码。"""
import os
from abc import ABC, abstractmethod


class ExecutionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def get_module(self):
        """返回一个可作为 `asnumpy` 使用的模块对象。"""

    @abstractmethod
    def to_device(self, np_array):
        ...

    @abstractmethod
    def to_host(self, device_array):
        ...


def select_backend(name=None):
    name = name or os.getenv("NUMPILOT_BACKEND", "shim")
    if name == "shim":
        from numpilot.backends.shim import ShimBackend
        return ShimBackend()
    if name == "npu":
        from numpilot.backends.npu import NpuBackend
        return NpuBackend()
    raise ValueError(f"未知后端: {name!r}(可选 shim | npu)")
