"""进程内模块注入的串行化保护。

``sys.modules`` 是进程全局状态。NumPilot 尚未把用户产物迁入 OS 级隔离进程前，
所有 shim/NPU 注入必须持有同一把可重入锁，避免并行智能体互相污染 backend。
"""
from contextlib import contextmanager
import hashlib
import sys
import threading
import types


_MODULE_INJECTION_LOCK = threading.RLock()
_MISSING = object()


def _state_fingerprint(value, seen=None, *, mutable_state=True):
    """Return a bounded structural fingerprint without invoking user code."""

    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return ("cycle", type(value).__module__, type(value).__qualname__, identity)
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return (type(value).__name__, value)
    if isinstance(value, types.CodeType):
        return (
            "code",
            hashlib.sha256(value.co_code).hexdigest(),
            _state_fingerprint(
                value.co_consts,
                seen | {identity},
                mutable_state=False,
            ),
            tuple(value.co_names),
            tuple(value.co_freevars),
        )
    if isinstance(value, types.FunctionType):
        next_seen = seen | {identity}
        globals_map = value.__globals__
        referenced = tuple(
            sorted(
                (
                    name,
                    _state_fingerprint(
                        globals_map[name],
                        next_seen,
                        mutable_state=isinstance(
                            globals_map[name], types.FunctionType
                        ),
                    ),
                )
                for name in value.__code__.co_names
                if name in globals_map
            )
        )
        closure = []
        for cell in value.__closure__ or ():
            try:
                item = cell.cell_contents
            except ValueError:
                item = _MISSING
            closure.append(_state_fingerprint(
                item,
                next_seen,
                mutable_state=isinstance(item, types.FunctionType),
            ))
        return (
            "function",
            _state_fingerprint(value.__code__, next_seen),
            _state_fingerprint(value.__defaults__, next_seen),
            _state_fingerprint(value.__kwdefaults__, next_seen),
            tuple(closure),
            referenced,
        )
    if type(value) in {tuple, list}:
        if not mutable_state:
            return ("opaque", type(value).__module__, type(value).__qualname__, identity)
        return (
            type(value).__name__,
            tuple(
                _state_fingerprint(item, seen | {identity})
                for item in value
            ),
        )
    if type(value) in {dict}:
        if not mutable_state:
            return ("opaque", type(value).__module__, type(value).__qualname__, identity)
        return (
            "dict",
            tuple(sorted(
                (
                    _state_fingerprint(key, seen | {identity}),
                    _state_fingerprint(item, seen | {identity}),
                )
                for key, item in value.items()
            )),
        )
    if type(value) in {set, frozenset}:
        if not mutable_state:
            return ("opaque", type(value).__module__, type(value).__qualname__, identity)
        return (
            type(value).__name__,
            tuple(sorted(
                _state_fingerprint(item, seen | {identity})
                for item in value
            )),
        )
    return ("opaque", type(value).__module__, type(value).__qualname__, identity)


def _callable_fingerprint(target):
    return _state_fingerprint(target)


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


def traced_module(
    module,
    operation_names,
    *,
    before_call=None,
    targets=None,
    enforce_target_identity=False,
):
    """返回一个透明模块代理，并统计实际执行的公开算子调用。

    计数发生在 Python API 包装器真正被调用时，而不是看到 AST 中出现了
    ``ap.<op>`` 就乐观推断。代理只用于 NumPilot 生成的 ``import asnumpy as ap``
    产物；不修改原模块，因此能和 ``temporary_module`` 的全局锁一起使用。
    """
    names = tuple(sorted({str(name) for name in operation_names}))
    if targets is None:
        targets = {name: getattr(module, name) for name in names}
    else:
        targets = dict(targets)
        if set(targets) != set(names) or any(
            not callable(target) for target in targets.values()
        ):
            raise ValueError("traced target snapshot is invalid")
    target_fingerprints = {
        name: _callable_fingerprint(target)
        for name, target in targets.items()
    }
    counts = {name: 0 for name in names}
    cache = {}
    target_registry = types.MappingProxyType(dict(targets))
    fingerprint_registry = types.MappingProxyType(dict(target_fingerprints))

    class _TracingModule(types.ModuleType):
        def __getattr__(self, name):
            if name in targets:
                if name not in cache:
                    def traced(*args, **kwargs):
                        # Avoid the conventional ``__wrapped__``/default-arg
                        # exposure.  This remains same-process drift detection,
                        # not a sandbox against hostile Python introspection.
                        __name = traced.__name__
                        if before_call is not None:
                            before_call(__name, args, kwargs)
                        __target = target_registry[__name]
                        if enforce_target_identity:
                            try:
                                current = getattr(module, __name)
                            except Exception as exc:
                                raise RuntimeError(
                                    "runtime target integrity rejected for %s: "
                                    "%s: %s"
                                    % (__name, type(exc).__name__, exc)
                                ) from exc
                            if current is not __target:
                                raise RuntimeError(
                                    "runtime target integrity rejected for %s: "
                                    "callable identity drifted after audit"
                                    % __name
                                )
                            if (
                                _callable_fingerprint(__target)
                                != fingerprint_registry[__name]
                            ):
                                raise RuntimeError(
                                    "runtime target integrity rejected for %s: "
                                    "callable implementation drifted after audit"
                                    % __name
                                )
                        counts[__name] += 1
                        return __target(*args, **kwargs)

                    traced.__name__ = name
                    cache[name] = traced
                return cache[name]
            return getattr(module, name)

    proxy = _TracingModule(getattr(module, "__name__", "asnumpy"))
    for attr in ("__file__", "__package__", "__path__", "__spec__", "__version__"):
        if hasattr(module, attr):
            setattr(proxy, attr, getattr(module, attr))
    return proxy, counts
