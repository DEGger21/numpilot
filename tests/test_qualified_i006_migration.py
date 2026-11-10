"""CPU-only acceptance-contract tests; these never claim actual NPU execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from numpilot.backends.shim import ShimBackend
from numpilot.benchmark_lowering import lower_benchmark_program
from numpilot.benchmark_overlay import build_analysis_profiles
from numpilot.capabilities import match_capability_profile
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.orchestrator import Orchestrator
from tests.test_benchmark_overlay import _overlay
from tests.test_concatenate_call_profiles import _capability
from tools import qualified_i006_migration as tool


SOURCE = Path(__file__).parents[1] / tool.SOURCE_RELATIVE


def _profiles():
    original = SOURCE.read_text()
    lowered = lower_benchmark_program(original)
    collected = build_analysis_profiles(lowered.source, _overlay())
    return original, lowered, collected, {
        'matched_callsites': collected.matched_callsites,
        'attempted_callsites': collected.attempted_callsites,
        'profiles': collected.profiles.to_dict(),
    }


def _contract_fixture():
    """Synthetic receipt shape for validator rejection tests only."""
    source, lowered, collected, profiles = _profiles()
    profile = collected.profiles.callsites[0].profile.to_dict()
    generated = 'import asnumpy as ap\n# contract-fixture; not real NPU evidence\n'
    guard = {
        'backend': 'npu', 'strict_npu': True, 'ok': True,
        'preflight_ok': True, 'preflight_strict_npu': True,
        'runtime_profile_ok': True, 'runtime_target_integrity_ok': True,
        'runtime_execution_ok': True, 'runtime_trace_ok': True,
        'profile_status': 'matched', 'required_ops': ['concatenate'],
        'profile_scoped': ['concatenate'], 'missing': [], 'non_strict': [],
        'runtime_target_integrity_errors': [],
        'runtime_call_counts': {'kernel': {'concatenate': 1}, 'artifact': {'concatenate': 1}},
        'runtime_profile_checks': {'concatenate': [
            {'matched': True, 'profile': profile}, {'matched': True, 'profile': profile},
        ]},
    }
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    analysis_hash = hashlib.sha256(lowered.source.encode()).hexdigest()
    evidence = {
        'status': 'pass', 'backend': 'npu', 'numerically_consistent': True,
        'input_consistent': True, 'kernel_consistent': True,
        'artifact_consistent': True, 'artifact_inputs_executed': True,
        'source_sha256': source_hash,
        'migrated_sha256': hashlib.sha256(generated.encode()).hexdigest(),
        'comparisons': {name: {'consistent': True} for name in ('inputs', 'kernel', 'artifact')},
        'backend_provenance': guard,
    }
    result = {
        'status': 'pass', 'engine': 'ast', 'rewrite_integrity': True,
        'rewrites': {'concatenate': 1}, 'expected_rewrites': {'concatenate': 1},
        'source_sha256': source_hash, 'analysis_source_sha256': analysis_hash,
        'migrated_source': generated, 'validation_evidence': evidence,
    }
    identity = {'identity_sha256': 'test-contract-identity', 'status': 'audit_required'}
    inputs = {
        'profiles': profiles,
        'overlay': {'operators': ['concatenate'], 'runtime_identity_sha256': identity['identity_sha256']},
        'runtime_before': identity, 'runtime_after': deepcopy(identity),
        'source_sha256': source_hash, 'analysis_sha256': analysis_hash,
        'source_unchanged': True,
    }
    return result, inputs


def test_complete_contract_keeps_global_audit_unpromoted():
    result, inputs = _contract_fixture()
    assert tool.assess_evidence(result, **inputs) == []
    assert inputs['runtime_before']['status'] == inputs['runtime_after']['status'] == 'audit_required'


@pytest.mark.parametrize('failure', [
    'shim', 'guard_shim', 'missing_trace', 'missing_profile', 'failed_profile',
    'wrong_shape', 'missing_artifact_call', 'extra_call', 'extra_operator',
    'oracle_hash', 'generated_hash', 'source_changed', 'runtime_drift',
    'overlay_identity', 'profile_source', 'non_scoped', 'target_drift',
    'bad_comparison', 'unexecuted_artifact_inputs', 'extra_rewrite',
])
def test_evidence_gaps_cannot_be_promoted(failure):
    result, inputs = _contract_fixture()
    evidence = result['validation_evidence']
    guard = evidence['backend_provenance']
    if failure == 'shim': evidence['backend'] = 'shim'
    elif failure == 'guard_shim': guard['backend'] = 'shim'
    elif failure == 'missing_trace': guard.pop('runtime_trace_ok')
    elif failure == 'missing_profile': guard['runtime_profile_checks']['concatenate'].pop()
    elif failure == 'failed_profile': guard['runtime_profile_checks']['concatenate'][0]['matched'] = False
    elif failure == 'wrong_shape':
        guard['runtime_profile_checks']['concatenate'][0]['profile'] = deepcopy(guard['runtime_profile_checks']['concatenate'][0]['profile'])
        guard['runtime_profile_checks']['concatenate'][0]['profile']['inputs'][0]['shape'][0] = 2
    elif failure == 'missing_artifact_call': guard['runtime_call_counts']['artifact'] = {'concatenate': 0}
    elif failure == 'extra_call': guard['runtime_call_counts']['kernel']['concatenate'] = 2
    elif failure == 'extra_operator': guard['required_ops'].append('sin')
    elif failure == 'oracle_hash': evidence['source_sha256'] = 'wrong'
    elif failure == 'generated_hash': evidence['migrated_sha256'] = 'wrong'
    elif failure == 'source_changed': inputs['source_unchanged'] = False
    elif failure == 'runtime_drift': inputs['runtime_after']['status'] = 'matched'
    elif failure == 'overlay_identity': inputs['overlay']['runtime_identity_sha256'] = 'other'
    elif failure == 'profile_source': inputs['profiles']['profiles']['source_sha256'] = 'other'
    elif failure == 'non_scoped': guard['profile_scoped'] = []
    elif failure == 'target_drift': guard['runtime_target_integrity_errors'] = ['target changed']
    elif failure == 'bad_comparison': evidence['comparisons']['kernel']['consistent'] = False
    elif failure == 'unexecuted_artifact_inputs': evidence['artifact_inputs_executed'] = False
    elif failure == 'extra_rewrite': result['rewrites']['sin'] = 1
    assert tool.assess_evidence(result, **inputs)


class _ProfiledCpuBackend(ShimBackend):
    """Allow analysis of the scoped call, while execution remains explicitly CPU."""
    def prepare_analysis_profile_resolver(self, qualified_names):
        assert tuple(qualified_names) == ('numpy.concatenate',)
        return self

    def resolve_profile_capability(self, qualified_name, profile):
        assert qualified_name == 'numpy.concatenate'
        return match_capability_profile(_capability(), profile)


def test_original_i006_uses_real_ast_mainline_but_cpu_cannot_issue_npu_receipt(tmp_path):
    before = SOURCE.read_bytes()
    original, lowered, collected, profiles = _profiles()
    result = Orchestrator(
        backend=_ProfiledCpuBackend(extra_ops=('concatenate',)),
        engine=AstRewriteEngine(), analysis_profiles=collected.profiles,
    ).run_source(lowered.source, 'program.py', out_dir=str(tmp_path / 'mainline'),
                 verification_source=original, strict_input_types=True)
    assert result['status'] == 'pass'
    assert result['rewrites'] == result['expected_rewrites'] == {'concatenate': 1}
    assert 'ap.concatenate' in result['migrated_source']
    assert result['validation_evidence']['backend'] == 'shim'
    assert all(result['validation_evidence'][name] is True for name in (
        'input_consistent', 'kernel_consistent', 'artifact_consistent', 'artifact_inputs_executed'))
    _, inputs = _contract_fixture()
    inputs['profiles'] = profiles
    problems = tool.assess_evidence(result, **inputs)
    assert 'verification is not a passing NPU execution' in problems
    assert SOURCE.read_bytes() == before
    assert tool._sha(before) == tool.SOURCE_SHA256
    assert (tmp_path / 'mainline/migrated.py').read_text() == result['migrated_source']


def test_runtime_site_rejects_already_loaded_package_outside_requested_site(tmp_path, monkeypatch):
    old = tmp_path / 'old.py'
    old.touch()
    site = tmp_path / 'site'
    site.mkdir()
    monkeypatch.setitem(sys.modules, 'asnumpy', SimpleNamespace(__file__=str(old)))
    with pytest.raises(ValueError, match='already imported outside'):
        tool.select_runtime_site(site)


def test_invalid_runtime_persists_failure_and_output_is_write_once(tmp_path):
    args = SimpleNamespace(out=str(tmp_path / 'run'), runtime_site=str(tmp_path / 'missing'),
                           capability_overlay='/missing', evidence_root='/missing')
    receipt = tool.run(args)
    assert receipt['status'] == 'blocked' and receipt['scoped_qualified'] is False
    assert receipt['global_runtime_audit_promoted'] is False
    assert receipt['benchmark_suite_updated'] is False
    assert receipt['performance_valid'] is False
    assert receipt['model_called'] is False
    target = tmp_path / 'run/receipt.json'
    before = target.read_bytes()
    assert json.loads(before) == receipt
    with pytest.raises(FileExistsError):
        tool.run(args)
    assert target.read_bytes() == before
