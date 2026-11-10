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
