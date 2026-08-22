import sys
import threading
import types

import pytest

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


def test_traced_module_runs_profile_guard_before_target_dispatch():
    module = types.ModuleType("asnumpy")
    target_calls = []
    guarded_calls = []

    def target(value):
        target_calls.append(value)
        return value

    def reject(name, args, kwargs):
        guarded_calls.append((name, args, kwargs))
        raise RuntimeError("profile rejected")

    module.cbrt = target
    proxy, counts = traced_module(
        module,
        ["cbrt"],
        before_call=reject,
    )

    try:
        proxy.cbrt(8)
    except RuntimeError as error:
        assert str(error) == "profile rejected"
    else:
        raise AssertionError("profile guard did not reject the call")

    assert guarded_calls == [("cbrt", (8,), {})]
    assert target_calls == []
    assert counts == {"cbrt": 0}


def test_traced_module_rejects_target_identity_drift_and_hides_wrapped_target():
    module = types.ModuleType("asnumpy")
    target = lambda value: value
    module.cbrt = target
    proxy, counts = traced_module(
        module,
        ["cbrt"],
        enforce_target_identity=True,
    )
    traced = proxy.cbrt

    assert not hasattr(traced, "__wrapped__")
    module.cbrt = lambda value: value

    with pytest.raises(RuntimeError, match="callable identity drifted"):
        traced(8)
    assert counts == {"cbrt": 0}


def test_traced_module_rejects_python_callable_implementation_drift():
    module = types.ModuleType("asnumpy")

    def target(value):
        return value

    def replacement(value):
        return -value

    module.cbrt = target
    proxy, counts = traced_module(
        module,
        ["cbrt"],
        enforce_target_identity=True,
    )
    target.__code__ = replacement.__code__

    with pytest.raises(RuntimeError, match="implementation drifted"):
        proxy.cbrt(8)
    assert counts == {"cbrt": 0}


def test_traced_module_rejects_referenced_callable_code_drift():
    module = types.ModuleType("asnumpy")

    def dependency(value):
        return value

    def target(value):
        return dependency(value)

    def replacement(value):
        return -value

    module.cbrt = target
    proxy, counts = traced_module(
        module,
        ["cbrt"],
        enforce_target_identity=True,
    )
    dependency.__code__ = replacement.__code__

    with pytest.raises(RuntimeError, match="implementation drifted"):
        proxy.cbrt(8)
    assert counts == {"cbrt": 0}


def test_traced_module_rejects_mutable_default_state_drift():
    module = types.ModuleType("asnumpy")

    def target(value, dispatch={"mode": "npu"}):
        return (dispatch["mode"], value)

    module.cbrt = target
    proxy, counts = traced_module(
        module,
        ["cbrt"],
        enforce_target_identity=True,
    )
    target.__defaults__[0]["mode"] = "cpu"

    with pytest.raises(RuntimeError, match="implementation drifted"):
        proxy.cbrt(8)
    assert counts == {"cbrt": 0}


def test_traced_wrapper_defaults_do_not_expose_qualified_target():
    module = types.ModuleType("asnumpy")
    target = lambda value: value
    module.cbrt = target
    proxy, _ = traced_module(module, ["cbrt"])

    wrapper = proxy.cbrt

    assert wrapper.__defaults__ is None
    assert wrapper.__kwdefaults__ is None


def test_traced_module_allows_unrelated_mutation_of_captured_observer_list():
    module = types.ModuleType("asnumpy")
    calls = []

    def target(value):
        calls.append(value)
        return value

    module.cbrt = target
    proxy, _ = traced_module(
        module,
        ["cbrt"],
        enforce_target_identity=True,
    )

    assert proxy.cbrt(8) == 8
    assert proxy.cbrt(27) == 27
    assert calls == [8, 27]
