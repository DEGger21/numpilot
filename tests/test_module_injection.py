import sys
import threading
import types

from numpilot.module_injection import temporary_module, traced_module


def test_temporary_module_is_serialized_and_restored():
    original = sys.modules.get("asnumpy")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    observations = []

    def first():
        module = types.ModuleType("backend_a")
        with temporary_module("asnumpy", module):
            observations.append(sys.modules["asnumpy"].__name__)
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second():
        assert first_entered.wait(timeout=2)
        module = types.ModuleType("backend_b")
        with temporary_module("asnumpy", module):
            observations.append(sys.modules["asnumpy"].__name__)
            second_entered.set()

    thread_a = threading.Thread(target=first)
    thread_b = threading.Thread(target=second)
    thread_a.start()
    thread_b.start()
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    thread_a.join(timeout=2)
    thread_b.join(timeout=2)
    assert observations == ["backend_a", "backend_b"]
    assert sys.modules.get("asnumpy") is original


def test_traced_module_counts_real_calls_only():
    module = types.ModuleType("asnumpy")
    module.sqrt = lambda value: value
    module.ndarray = object()
    proxy, counts = traced_module(module, ["sqrt"])

    assert counts == {"sqrt": 0}
    assert proxy.ndarray is module.ndarray
    assert proxy.sqrt(3) == 3
    assert proxy.sqrt(4) == 4
    assert counts == {"sqrt": 2}
