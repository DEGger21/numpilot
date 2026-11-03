from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'asnumpy_contract_probe', ROOT / 'tools/asnumpy_contract_probe.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_probe_normalises_zero_dimensional_values():
    assert MODULE._normalise(np.array(3.0, dtype=np.float32)) == 3.0
    assert MODULE._normalise(np.array([-0.0], dtype=np.float32)) == ['-0.0']


def test_contract_probe_treats_signed_zero_as_semantic_data():
    expected = np.array([-0.0], dtype=np.float32)
    observed = np.array([0.0], dtype=np.float32)
    assert not MODULE._semantic_equal(expected, observed)
    assert MODULE._semantic_equal(expected, expected.copy())


def test_contract_probe_records_match_and_mismatch_without_runtime_import():
    match = MODULE._run_case(
        'match', 'numpy.add', {'dtype': 'float32'},
        lambda: np.array([3.0], dtype=np.float32),
        lambda: np.array([3.0], dtype=np.float32),
    )
    mismatch = MODULE._run_case(
        'mismatch', 'numpy.fmax', {'dtype': 'float32'},
        lambda: np.array([2.0], dtype=np.float32),
        lambda: np.array([np.nan], dtype=np.float32),
    )
    assert match['status'] == 'verified_match'
    assert mismatch['status'] == 'semantic_mismatch'


def test_contract_probe_resolves_cann_version_from_toolkit_home(monkeypatch):
    monkeypatch.delenv('CANN_VERSION', raising=False)
    monkeypatch.setenv(
        'ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/cann-9.2.0-beta.2'
    )
    assert MODULE._cann_version() == '9.2.0-beta.2'


def test_contract_probe_honours_explicit_runtime_identity(monkeypatch):
    monkeypatch.setenv('CANN_VERSION', 'explicit-cann')
    monkeypatch.setenv('ASCEND_SOC_VERSION', 'Ascend910B3')
    assert MODULE._cann_version() == 'explicit-cann'
    assert MODULE._soc_name() == 'Ascend910B3'


def test_contract_probe_soc_is_unknown_when_acl_is_unavailable(monkeypatch):
    monkeypatch.delenv('ASCEND_SOC_VERSION', raising=False)
    original_import = __import__

    def _blocked_acl(name, *args, **kwargs):
        if name == 'acl':
            raise ImportError('acl is not installed')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', _blocked_acl)
    assert MODULE._soc_name() == 'unknown'
