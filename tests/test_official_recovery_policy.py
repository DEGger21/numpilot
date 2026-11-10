"""New official recovery runs freeze policy; existing histories cannot escalate."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from numpilot.gaps import official_candidate_recovery as bridge
from numpilot.gaps.recovery import (
    EvidenceRef, LicenseBinding, OperatorCandidateOrigin, RecoveryPhase,
    RecoveryStore, create_operator_candidate, recovery_policy_for_profile,
)


def _inputs(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('import numpy as np\n', encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    paths = {name: tmp_path / (name + '.json') for name in ('handoff', 'preflight', 'resolution', 'gate')}
    for p in paths.values():
        p.write_text(json.dumps({'fixture': p.stem}) + '\n', encoding='utf-8')
    ref = lambda name: EvidenceRef(name, str(source), digest)
    candidate = create_operator_candidate(
        origin_kind=OperatorCandidateOrigin.OFFICIAL_BINARY,
        gap_id='concatenate-test', qualified_api='numpy.concatenate',
        gap_spec_ref=ref('gap_spec'), resolution_decision_ref=ref('resolution_decision'),
        target_soc='Ascend910B3', cann_version='9.2.0-beta.2',
        license=LicenseBinding('LicenseRef-External-Installed-CANN-Terms', False, ref('license')),
        provenance_refs=(ref('provenance'),), candidate_refs=(ref('candidate'),),
    )
    gate = {
        'schema': 'numpilot.official-adapter-context-gate.v1',
        'status': 'ready_for_official_adapter_model', 'source': str(source),
        'reference_root': str(tmp_path / 'asnumpy'), 'qualified_api': 'numpy.concatenate',
        'candidate_id': candidate.candidate_id, 'context_plan_sha256': digest,
        'adapter_contract_sha256': digest,
        'next_action': 'invoke_configured_model_for_official_adapter',
    }
    gate.update({k: False for k in (
        'model_invocation_performed', 'network_access_performed', 'system_write_performed',
        'asnumpy_import_performed', 'build_performed', 'execution_performed', 'activation_performed',
    )})
    paths['gate'].write_text(json.dumps(gate), encoding='utf-8')
    spec = SimpleNamespace(gap_id=candidate.gap_id, qualified_api=candidate.qualified_api)
    decision = SimpleNamespace(gap_id=candidate.gap_id, qualified_api=candidate.qualified_api, route=bridge.INSTALLED_ADAPTER_ROUTE)
    report = SimpleNamespace(target_soc=candidate.target_soc, cann_version=candidate.cann_version,
                             decisions=(decision,), validate_preparation=lambda preparation: None)
    monkeypatch.setattr(bridge, 'load_official_candidate_handoff', lambda *a: SimpleNamespace(payload={'source_sha256': digest}, handoff_id='test-handoff'))
    monkeypatch.setattr(bridge, 'load_official_resolution_report', lambda *a: report)
    monkeypatch.setattr(bridge, 'prepare_gap_completion', lambda *a, **kw: SimpleNamespace(specs=(spec,)))
    monkeypatch.setattr(bridge, 'prepare_official_adapter_input_from_preflight', lambda *a: SimpleNamespace(operator_manifest={'candidate_id': candidate.candidate_id, 'qualified_api': candidate.qualified_api}))
    monkeypatch.setattr(bridge, '_candidate_refs', lambda *a: candidate)
    args = tuple(str(paths[name]) if name != 'source' else str(source)
                 for name in ('handoff', 'preflight', 'source', 'resolution', 'gate'))
    return args, {'runtime_backend': 'ascend-npu', 'runtime_audit_id': 'unit-policy', 'runtime_identity_sha256': digest}


@pytest.mark.parametrize('profile', ['receipt-only', 'candidate-generation', 'full-local-recovery'])
def test_official_new_run_freezes_requested_policy_without_execution(tmp_path, monkeypatch, profile):
    args, kwargs = _inputs(tmp_path, monkeypatch)
    result = bridge.materialize_official_candidate_recovery(*args, str(tmp_path / 'state'), profile=profile, **kwargs)
    assert result.recovery.state.phase is RecoveryPhase.CANDIDATE_READY
    assert result.recovery.state.policy == recovery_policy_for_profile(profile)
    assert result.appended_revisions == 4
    assert not any(result.side_effects.values())
    assert result.candidates[0].activation_allowed is False
    again = bridge.materialize_official_candidate_recovery(*args, str(tmp_path / 'state'), profile=profile, **kwargs)
    assert again.appended_revisions == 0
    assert again.recovery == result.recovery


def test_official_existing_receipt_run_cannot_upgrade_policy(tmp_path, monkeypatch):
    args, kwargs = _inputs(tmp_path, monkeypatch)
    root = tmp_path / 'state'
    bridge.materialize_official_candidate_recovery(*args, str(root), **kwargs)
    before = RecoveryStore(root).status()
    with pytest.raises(bridge.OfficialCandidateRecoveryError) as error:
        bridge.materialize_official_candidate_recovery(*args, str(root), profile='full-local-recovery', **kwargs)
    assert error.value.code == 'state_binding_mismatch'
    assert RecoveryStore(root).status() == before


def test_full_local_still_requires_valid_context_gate(tmp_path):
    for name in ('handoff', 'preflight', 'resolution', 'gate'):
        (tmp_path / (name + '.json')).write_text('{}\n', encoding='utf-8')
    (tmp_path / 'source.py').write_text('import numpy as np\n', encoding='utf-8')
    args = tuple(str(tmp_path / (name + ('.py' if name == 'source' else '.json')))
                 for name in ('handoff', 'preflight', 'source', 'resolution', 'gate'))
    with pytest.raises(bridge.OfficialCandidateRecoveryError) as error:
        bridge.materialize_official_candidate_recovery(*args, str(tmp_path / 'state'), profile='full-local-recovery', runtime_backend='ascend-npu', runtime_audit_id='unit-policy', runtime_identity_sha256='a' * 64)
    assert error.value.code == 'unsafe_gate'
    assert not (tmp_path / 'state').exists()
