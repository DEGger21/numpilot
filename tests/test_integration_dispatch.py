from types import SimpleNamespace

import pytest

from numpilot.gaps.integration_gate import _install_dispatch_probe


def test_probe_rejects_existing_copy_binding_even_if_numerically_equivalent():
    old = lambda inputs, axis: inputs
    official = lambda inputs, axis: inputs
    with pytest.raises(RuntimeError, match='reviewed ACLNN binding'):
        _install_dispatch_probe(SimpleNamespace(_concatenate=old),
                                SimpleNamespace(concatenate_aclnn=official))


def test_probe_distinguishes_public_wrapper_dispatch_from_legacy_copy():
    official = lambda inputs, axis: ('native', inputs, axis)
    array_module = SimpleNamespace(_concatenate=official, _concatenate_legacy=lambda *a: a)
    counts = _install_dispatch_probe(array_module, SimpleNamespace(concatenate_aclnn=official))
    assert array_module._concatenate([1], axis=0) == ('native', [1], 0)
    assert counts == {'native_calls': 1, 'legacy_calls': 0}
    with pytest.raises(RuntimeError, match='legacy copy route'):
        array_module._concatenate_legacy([1], 0)
    assert counts == {'native_calls': 1, 'legacy_calls': 1}
