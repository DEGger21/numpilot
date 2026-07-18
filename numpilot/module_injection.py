"""进程内模块注入的串行化保护。

``sys.modules`` 是进程全局状态。NumPilot 尚未把用户产物迁入 OS 级隔离进程前，
所有 shim/NPU 注入必须持有同一把可重入锁，避免并行智能体互相污染 backend。
"""
from contextlib import contextmanager
from functools import wraps
import sys
import threading
import types


_MODULE_INJECTION_LOCK = threading.RLock()
_MISSING = object()


@contextmanager
def temporary_module(name, module):
    with _MODULE_INJECTION_LOCK:
        saved = sys.modules.get(name, _MISSING)
        sys.modules[name] = module
        try:
            yield
        finally:
            if saved is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def traced_module(module, operation_names):
    """返回一个透明模块代理，并统计实际执行的公开算子调用。

    计数发生在 Python API 包装器真正被调用时，而不是看到 AST 中出现了
    ``ap.<op>`` 就乐观推断。代理只用于 NumPilot 生成的 ``import asnumpy as ap``
    产物；不修改原模块，因此能和 ``temporary_module`` 的全局锁一起使用。
    """
    names = tuple(sorted({str(name) for name in operation_names}))
    targets = {name: getattr(module, name) for name in names}
    counts = {name: 0 for name in names}
    cache = {}

    class _TracingModule(types.ModuleType):
        def __getattr__(self, name):
            if name in targets:
                if name not in cache:
                    target = targets[name]

                    @wraps(target)
                    def traced(*args, __name=name, __target=target, **kwargs):
                        counts[__name] += 1
                        return __target(*args, **kwargs)

                    cache[name] = traced
                return cache[name]
            return getattr(module, name)

    proxy = _TracingModule(getattr(module, "__name__", "asnumpy"))
    for attr in ("__file__", "__package__", "__path__", "__spec__", "__version__"):
        if hasattr(module, attr):
            setattr(proxy, attr, getattr(module, attr))
    return proxy, counts
