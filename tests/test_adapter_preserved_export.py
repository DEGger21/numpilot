"""An existing lazy export can be preserved without weakening response binding."""

import hashlib

import pytest

from numpilot.gaps import provider
from numpilot.gaps.provider import DraftGenerationError


_PATH = 'src/asnumpy/__init__.py'
_BASE = (
    'from typing import TYPE_CHECKING\n'
    'if TYPE_CHECKING:\n'
    '    from .array import (\n'
    '        concatenate,\n'
    '    )\n'
    '    from .io import load, save, savez, savez_compressed\n'
    '    from .math import (\n'
    '        absolute,\n'
    '    )\n'
    '    from .nn import softmax\n'
    '\n_LAZY_MAPPING = {\n'
    '    # .array\n'
    '    "concatenate": ".array",\n'
    '    # .linalg\n'
    '    "matmul": ".linalg.direct",\n'
    '    # .math\n'
    '    "absolute": ".math",\n'
    '    # .random\n'
    '    "random": ".random",\n'
    '}\n'
)


def _response():
    _, intervals = provider._render_official_exact_edit_target_windows(target_path=_PATH, base_content=_BASE)
    specs = provider._OFFICIAL_EXACT_EDIT_TARGET_WINDOWS[_PATH]
    payload = (
        'NUMPILOT_WINDOW_REPLACE_V1\n'
        f'PATH: {_PATH}\n'
        f'BASE_SHA256: {hashlib.sha256(_BASE.encode()).hexdigest()}\n'
        f'WINDOW_COUNT: {len(intervals)}\n'
    )
    for index, ((start, end), spec) in enumerate(zip(intervals, specs), 1):
        payload += f'BEGIN_WINDOW {index}\nLABEL: {spec[0]}\nCONTENT:\n'
        payload += _BASE[start:end]
        payload += f'\nEND_WINDOW {index}\n'
    return payload + 'END_WINDOW_REPLACE\n'


def test_preserved_export_noop_remains_rejected_by_default():
    with pytest.raises(DraftGenerationError, match='no-op'):
        provider._apply_official_window_replace_response(_response(), expected_path=_PATH, base_content=_BASE)


def test_preserved_export_explicitly_admitted_is_byte_identical():
    result = provider._apply_official_window_replace_response(
        _response(), expected_path=_PATH, base_content=_BASE, allow_unchanged=True,
    )
    assert result == _BASE


@pytest.mark.parametrize('tamper,reason', [
    ('hash', 'base hash mismatch'),
    ('path', 'path mismatch'),
    ('label', 'label 1 is invalid'),
    ('count', 'count is invalid'),
])
def test_preserved_export_still_requires_exact_protocol_binding(tamper, reason):
    payload = _response()
    if tamper == 'hash':
        payload = payload.replace(hashlib.sha256(_BASE.encode()).hexdigest(), '0' * 64)
    elif tamper == 'path':
        payload = payload.replace(f'PATH: {_PATH}', 'PATH: src/asnumpy/array.py')
    elif tamper == 'label':
        payload = payload.replace('LABEL: type_checking_array_imports', 'LABEL: unrelated_imports')
    else:
        payload = payload.replace('WINDOW_COUNT: 4', 'WINDOW_COUNT: 1')
    with pytest.raises(DraftGenerationError, match=reason):
        provider._apply_official_window_replace_response(
            payload, expected_path=_PATH, base_content=_BASE, allow_unchanged=True,
        )
