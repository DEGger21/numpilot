import hashlib
import json
from argparse import Namespace

import numpilot.__main__ as main
from numpilot.gaps.adapter_handoff import write_adapter_handoff


def _make_handoff(tmp_path):
    path = 'src/asnumpy/linalg/direct.py'
    content = 'def matmul(a, b):\n    return a @ b\n'
    plan = {
        'schema_version': 1,
        'architecture': 'official_aclnn_existing_module',
        'mode': 'existing_module',
        'reference_root': str(tmp_path / 'asnumpy'),
        'context_sha256': {path: hashlib.sha256(content.encode()).hexdigest()},
        'expected_overlay_paths': [path],
        'total_bytes': len(content.encode()),
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
        ).encode()
    ).hexdigest()
    root = tmp_path / 'handoff'
    write_adapter_handoff(
        root,
        operator_artifact={'artifact_id': 'candidate-status'},
        adapter_contract='{}',
        asnumpy_context={path: content},
        adapter_context_plan=plan,
        skill_files=(),
        skill_bundle_sha256='0' * 64,
    )
    return root, path


def _args(root):
    return Namespace(
        handoff_dir=str(root),
        response_dir=str(root / 'responses'),
        json=True,
    )


def test_status_waiting_when_responses_are_missing(tmp_path, capsys):
    root, path = _make_handoff(tmp_path)
    assert main._cmd_inspect_asnumpy_adapter_handoff(_args(root)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['status'] == 'waiting_for_agent'
    assert payload['next_action'] == 'write_responses'
    assert payload['counts'] == {'expected': 1, 'ready': 0, 'missing': 1, 'invalid': 0}
    assert payload['missing_response_paths'] == [path + '.response']
    assert payload['actual_remote_call'] is False


def test_status_ready_for_validation_when_response_is_present(tmp_path, capsys):
    root, path = _make_handoff(tmp_path)
    (root / 'responses' / (path + '.response')).parent.mkdir(parents=True, exist_ok=True)
    (root / 'responses' / (path + '.response')).write_text('NUMPILOT_WINDOW_REPLACE_V1\n', encoding='utf-8')
    assert main._cmd_inspect_asnumpy_adapter_handoff(_args(root)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['status'] == 'ready_for_validation'
    assert payload['next_action'] == 'validate-asnumpy-adapter-handoff'
    assert payload['counts'] == {'expected': 1, 'ready': 1, 'missing': 0, 'invalid': 0}


def test_status_marks_symlink_response_invalid(tmp_path, capsys):
    root, path = _make_handoff(tmp_path)
    response = root / 'responses' / (path + '.response')
    response.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / 'response.txt'
    target.write_text('payload\n', encoding='utf-8')
    response.symlink_to(target)
    assert main._cmd_inspect_asnumpy_adapter_handoff(_args(root)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['status'] == 'waiting_for_agent'
    assert payload['counts'] == {'expected': 1, 'ready': 0, 'missing': 0, 'invalid': 1}
    assert payload['invalid_responses'] == [{'path': path + '.response', 'reason': 'symlink'}]
