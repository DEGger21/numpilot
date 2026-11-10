import hashlib
import json

import pytest

from numpilot.gaps.adapter_handoff import (
    HANDOFF_SCHEMA,
    PROTOCOL,
    load_adapter_handoff,
    write_adapter_handoff,
)


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def test_write_and_load_adapter_handoff(tmp_path):
    path = 'src/asnumpy/linalg/direct.py'
    content = 'def matmul(a, b):\n    return a @ b\n'
    plan = {
        'schema_version': 3,
        'architecture': 'official_aclnn_existing_module',
        'mode': 'existing_module',
        'expected_overlay_paths': [path],
        'context_sha256': {path: _sha(content)},
        'plan_sha256': 'a' * 64,
        'total_bytes': len(content.encode()),
    }
    contract = '{"strategy":"official_aclnn_existing_module"}\n'
    root = tmp_path / 'handoff'
    metadata = write_adapter_handoff(
        root,
        operator_artifact={'artifact_id': 'candidate-abc'},
        adapter_contract=contract,
        asnumpy_context={path: content},
        adapter_context_plan=plan,
        skill_files=('references/official.md',),
        skill_bundle_sha256='b' * 64,
    )
    assert metadata['schema'] == HANDOFF_SCHEMA
    assert metadata['protocol'] == PROTOCOL
    loaded = load_adapter_handoff(root)
    assert loaded['expected_overlay_paths'] == (path,)
    assert loaded['context'][path] == content
    assert (root / 'prompts' / (path + '.md')).is_file()
    assert (root / 'windows' / (path + '.json')).is_file()
    assert (root / 'responses').is_dir()


def test_load_adapter_handoff_rejects_context_drift(tmp_path):
    path = 'src/asnumpy/linalg/direct.py'
    content = 'def matmul(a, b):\n    return a @ b\n'
    plan = {
        'expected_overlay_paths': [path],
        'context_sha256': {path: _sha(content)},
    }
    root = tmp_path / 'handoff'
    write_adapter_handoff(
        root,
        operator_artifact={'artifact_id': 'candidate-abc'},
        adapter_contract='{}',
        asnumpy_context={path: content},
        adapter_context_plan=plan,
        skill_files=('references/official.md',),
        skill_bundle_sha256='b' * 64,
    )
    target = root / 'context' / path
    target.write_text('drift\n', encoding='utf-8')
    with pytest.raises(ValueError, match='context hash mismatch'):
        load_adapter_handoff(root)



def test_validate_handoff_missing_response_writes_failure_receipt(tmp_path, monkeypatch):
    from argparse import Namespace

    import numpilot.__main__ as main
    from numpilot.asnumpy_skill import load_asnumpy_integration_skill
    from numpilot.gaps.provider import DraftRepairError, LlmDraftProvider

    path = 'src/asnumpy/linalg/direct.py'
    content = 'def matmul(a, b):\n    return a @ b\n'
    plan = {
        'schema_version': 1,
        'architecture': 'legacy_manual',
        'mode': 'manual',
        'reference_root': str(tmp_path / 'asnumpy'),
        'context_sha256': {path: _sha(content)},
        'expected_overlay_paths': [path],
        'total_bytes': len(content.encode()),
        'plan_sha256': '',
    }
    plan['plan_sha256'] = hashlib.sha256(
        json.dumps(
            {
                'schema_version': plan['schema_version'],
                'architecture': plan['architecture'],
                'mode': plan['mode'],
                'context_sha256': plan['context_sha256'],
                'expected_overlay_paths': plan['expected_overlay_paths'],
            },
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    bundle = load_asnumpy_integration_skill()
    handoff_root = tmp_path / 'handoff'
    write_adapter_handoff(
        handoff_root,
        operator_artifact={'artifact_id': 'candidate-abc'},
        adapter_contract='{}',
        asnumpy_context={path: content},
        adapter_context_plan=plan,
        skill_files=tuple(bundle.files),
        skill_bundle_sha256=bundle.bundle_sha256,
    )

    def fail_without_receipt(self, *args, **kwargs):
        raise DraftRepairError(
            'missing Codex response',
            receipt_count=0,
            request_count=1,
            failures=('missing Codex response: %s' % path,),
            actual_remote_call=False,
        )

    monkeypatch.setattr(
        LlmDraftProvider,
        'generate_asnumpy_adapter_candidate',
        fail_without_receipt,
    )
    out_root = tmp_path / 'out'
    args = Namespace(
        handoff_dir=str(handoff_root),
        response_dir=str(handoff_root / 'responses'),
        out=str(out_root),
        max_attempts=1,
        asnumpy_skill_root=None,
        offline_skills=True,
    )
    assert main._cmd_validate_asnumpy_adapter_handoff(args) == 2
    receipt = json.loads(
        (out_root / 'asnumpy-adapter-validation-failure.json').read_text(
            encoding='utf-8'
        )
    )
    assert receipt['actual_remote_call'] is False
    assert receipt['receipt_count'] == 0
    assert receipt['missing_response_paths'] == [path + '.response']
    assert receipt['expected_response_paths'] == [path + '.response']
