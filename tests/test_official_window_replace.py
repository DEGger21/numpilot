import hashlib

import pytest

from numpilot.gaps import provider as provider_module
from numpilot.gaps.provider import (
    DraftGenerationError,
    _apply_official_window_replace_response,
    _render_official_exact_edit_target_windows,
)


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _response(path, base, windows, *, labels=None):
    specs = provider_module._OFFICIAL_EXACT_EDIT_TARGET_WINDOWS[path]
    labels = labels or tuple(item[0] for item in specs)
    parts = [
        'NUMPILOT_WINDOW_REPLACE_V1\n',
        'PATH: %s\n' % path,
        'BASE_SHA256: %s\n' % _sha(base),
        'WINDOW_COUNT: %d\n' % len(windows),
    ]
    for index, (label, content) in enumerate(zip(labels, windows), 1):
        parts.extend((
            'BEGIN_WINDOW %d\nLABEL: %s\nCONTENT:\n' % (index, label),
            content,
            '\nEND_WINDOW %d\n' % index,
        ))
    parts.append('END_WINDOW_REPLACE')
    return ''.join(parts)


def test_window_replace_rebuilds_only_reviewed_window():
    path = 'csrc/linalg/product.cpp'
    base = (
        '// immutable preamble\n'
        'NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {\n'
        '  return 0;\n'
        '}\n'
        '\nNPUArray Einsum(const char* spec) { return 1; }\n'
    )
    context, intervals = _render_official_exact_edit_target_windows(
        target_path=path, base_content=base
    )
    del context
    start, end = intervals[0]
    replacement = base[start:end].replace('return 0;', 'return aclnnMatmul();')
    response = _response(path, base, [replacement])
    rebuilt = _apply_official_window_replace_response(
        response,
        expected_path=path,
        base_content=base,
        allowed_intervals=intervals,
    )
    assert rebuilt.startswith('// immutable preamble\n')
    assert 'return aclnnMatmul();' in rebuilt
    assert rebuilt.endswith('NPUArray Einsum(const char* spec) { return 1; }\n')


def test_window_replace_rejects_label_drift():
    path = 'src/asnumpy/linalg/direct.py'
    base = 'def matmul(a, b):\n    return a @ b\n'
    _context, intervals = _render_official_exact_edit_target_windows(
        target_path=path, base_content=base
    )
    response = _response(path, base, [base.replace('@', '+')], labels=('wrong',))
    with pytest.raises(DraftGenerationError, match='label 1 is invalid'):
        _apply_official_window_replace_response(
            response,
            expected_path=path,
            base_content=base,
            allowed_intervals=intervals,
        )


def test_window_replace_rejects_changes_outside_custom_window():
    path = 'csrc/linalg/product.cpp'
    base = (
        '// immutable preamble\n'
        'NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {\n'
        '  return 0;\n'
        '}\n'
        '\nNPUArray Einsum(const char* spec) { return 1; }\n'
    )
    _context, intervals = _render_official_exact_edit_target_windows(
        target_path=path, base_content=base
    )
    start, end = intervals[0]
    replacement = base[start:end].replace('return 0;', 'return aclnnMatmul();')
    narrowed = ((start + 1, end),)
    response = _response(path, base, [replacement])
    with pytest.raises(DraftGenerationError, match='outside target windows'):
        _apply_official_window_replace_response(
            response,
            expected_path=path,
            base_content=base,
            allowed_intervals=narrowed,
        )


def test_window_replace_rejects_noop():
    path = 'src/asnumpy/linalg/direct.py'
    base = 'def matmul(a, b):\n    return a @ b\n'
    _context, intervals = _render_official_exact_edit_target_windows(
        target_path=path, base_content=base
    )
    response = _response(path, base, [base])
    with pytest.raises(DraftGenerationError, match='no-op'):
        _apply_official_window_replace_response(
            response,
            expected_path=path,
            base_content=base,
            allowed_intervals=intervals,
        )
