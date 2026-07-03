"""真实 asnumpy(NPU)后端适配器。需已 build/install asnumpy。"""
from numpilot.backends.base import ExecutionBackend


class NpuBackend(ExecutionBackend):
    name = "npu"

    def get_module(self):
        try:
            import asnumpy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "未找到 asnumpy。请先在 CANN 环境下 build/install asnumpy, "
                "或改用默认 shim 后端(NUMPILOT_BACKEND=shim)。"
            ) from e
        return asnumpy

    def to_device(self, np_array):
        return self.get_module().ndarray.from_numpy(np_array)

    def to_host(self, device_array):
        return device_array.to_numpy()
