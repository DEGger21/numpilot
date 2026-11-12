import ast
import hashlib
import json
from pathlib import Path
import re

import pytest

from numpilot.asnumpy_skill import (
    ASNUMPY_INTEGRATION_SKILL_FILES,
    ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES,
)
from numpilot.engines.llm import LlmEngine, LlmHttpError
from numpilot.gaps import provider as provider_module
from numpilot.gaps.provider import (
    DraftGenerationError,
    DraftRepairError,
    LlmDraftProvider,
    _apply_official_exact_edit_response,
    _load_asnumpy_adapter_report,
)
from numpilot.llm_config import LlmConfigStatus


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_response(path, base, edits, *, base_sha256=None):
    pieces = [
        "NUMPILOT_EXACT_EDIT_V1\n",
        "PATH: %s\n" % path,
        "BASE_SHA256: %s\n" % (base_sha256 or _sha256_text(base)),
        "EDIT_COUNT: %d\n" % len(edits),
    ]
    for index, (old, new) in enumerate(edits, start=1):
        pieces.extend((
            "BEGIN_EDIT %d\nOLD:\n" % index,
            old,
            "\nEND_OLD %d\nNEW:\n" % index,
            new,
            "\nEND_NEW %d\nEND_EDIT %d\n" % (index, index),
        ))
    pieces.append("END_EXACT_EDIT")
    return "".join(pieces)



def _window_response(path, base, replacements):
    pieces = [
        "NUMPILOT_WINDOW_REPLACE_V1\n",
        "PATH: %s\n" % path,
        "BASE_SHA256: %s\n" % _sha256_text(base),
    ]
    specifications = provider_module._OFFICIAL_EXACT_EDIT_TARGET_WINDOWS[path]
    pieces.append("WINDOW_COUNT: %d\n" % len(specifications))
    for index, (label, start_marker, end_marker) in enumerate(specifications, 1):
        start = 0 if start_marker is None else base.find(start_marker)
        end = len(base) if end_marker is None else base.find(end_marker)
        content = base[start:end]
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new, 1)
        pieces.extend((
            "BEGIN_WINDOW %d\nLABEL: %s\nCONTENT:\n" % (index, label),
            content,
            "\nEND_WINDOW %d\n" % index,
        ))
    pieces.append("END_WINDOW_REPLACE")
    return "".join(pieces)


def _context_plan(
    tmp_path,
    architecture,
    mode,
    context,
    paths,
    *,
    prompt_dependencies=None,
    official_library_binding=None,
):
    hashes = {
        path: _sha256_text(content)
        for path, content in sorted(context.items())
    }
    schema_version = (
        3
        if official_library_binding is not None
        else 2
        if prompt_dependencies is not None
        else 1
    )
    identity = {
        "schema_version": schema_version,
        "architecture": architecture,
        "mode": mode,
        "context_sha256": hashes,
        "expected_overlay_paths": list(paths),
    }
    if prompt_dependencies is not None:
        identity["prompt_dependencies"] = prompt_dependencies
    if official_library_binding is not None:
        identity["official_library_binding"] = official_library_binding
    return {
        **identity,
        "reference_root": str((tmp_path / "asnumpy-reference").absolute()),
        "total_bytes": sum(
            len(content.encode("utf-8")) for content in context.values()
        ),
        "plan_sha256": _sha256_text(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        )),
    }


def _configured_status(tmp_path):
    return LlmConfigStatus(
        path=tmp_path / "llm.json",
        configured=True,
        missing=(),
        source="user-config",
        model="glm-audited",
        provider="glm",
        token_present=True,
    )


def _remote_receipt(content):
    return {
        "schema_version": 1,
        "utc_start": "2026-08-28T00:00:00Z",
        "utc_end": "2026-08-28T00:00:01Z",
        "http_status": 200,
        "requested_model": "glm-audited",
        "response_model": "glm-audited",
        "response_id": "response-exact-edit",
        "response_created": 1787875200,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": _sha256_text(content),
        "request_id": "request-exact-edit",
        "usage": {"total_tokens": 101},
        "tls_verification": "enabled",
    }


class _FakeSkillBundle:
    bundle_sha256 = "b" * 64
    files = {
        path: "route skill fixture for %s" % path
        for path in ASNUMPY_INTEGRATION_SKILL_FILES
    }

    def prompt_context(self, selected_files=None):
        selected = tuple(selected_files or ASNUMPY_INTEGRATION_SKILL_FILES)
        return "\n".join(self.files[path] for path in selected)


def test_official_exact_edit_repair_prompt_documents_tensor_list_ownership():
    prompt = provider_module._ASNUMPY_OFFICIAL_EXACT_EDIT_REPAIR_SYSTEM_PROMPT
    assert "`aclDestroyTensorList` destroys every tensor" in prompt
    assert "`NPUArray` owns its `tensorPtr`" in prompt
    assert "input.aclDtype" in prompt
    assert "`input.device_address()`" in prompt
    assert "never `input.tensorPtr`" in prompt
    assert "double-destroys" in prompt


def test_official_exact_edit_repair_prompt_documents_python_validation():
    prompt = provider_module._ASNUMPY_OFFICIAL_EXACT_EDIT_REPAIR_SYSTEM_PROMPT
    assert "PYTHON VALIDATION INVARIANT" in prompt
    assert "before it calls native code" in prompt
    assert "`_concatenate`" in prompt
    assert "C-contiguous" in prompt
    assert "operator.index" in prompt


def test_official_exact_edit_repair_prompt_preserves_reviewed_binding_route():
    prompt = provider_module._ASNUMPY_OFFICIAL_EXACT_EDIT_REPAIR_SYSTEM_PROMPT
    assert "ROUTE PRESERVATION INVARIANT" in prompt
    assert "keep that binding call in the repaired wrapper" in prompt
    assert "Do not\nreplace `_<op>(...)`" in prompt
    assert "`Copysign` implementation `Absolute(x1) * Sign(x2)`" in prompt
    assert "negative zero" in prompt


def test_official_exact_edit_repair_prompt_documents_pinned_test_api():
    prompt = provider_module._ASNUMPY_OFFICIAL_EXACT_EDIT_REPAIR_SYSTEM_PROMPT
    assert "TEST FILE INVARIANT" in prompt
    assert "`ndarray` subclass that overrides `strides`" in prompt
    assert "does not provide `transpose`" in prompt
    assert "do not use `.T`" in prompt


def test_exact_edit_mechanically_rebuilds_original_position_order():
    path = "csrc/linalg/product.cpp"
    base = "alpha one\nmiddle two\nomega three\n"
    response = _exact_response(
        path,
        base,
        (("omega three", "omega 3"), ("alpha one", "alpha 1")),
    )

    rebuilt = _apply_official_exact_edit_response(
        response,
        expected_path=path,
        base_content=base,
    )

    assert rebuilt == "alpha 1\nmiddle two\nomega 3\n"


def test_exact_edit_rejects_old_that_is_not_unique_in_base():
    path = "src/asnumpy/linalg/direct.py"
    base = "token + token\n"
    response = _exact_response(path, base, (("token", "replacement"),))

    with pytest.raises(DraftGenerationError, match="occur exactly once"):
        _apply_official_exact_edit_response(
            response,
            expected_path=path,
            base_content=base,
        )


def test_exact_edit_rejects_overlapping_original_intervals():
    path = "src/asnumpy/linalg/direct.py"
    base = "abcdef\n"
    response = _exact_response(
        path,
        base,
        (("abcd", "first"), ("cdef", "second")),
    )

    with pytest.raises(DraftGenerationError, match="intervals overlap"):
        _apply_official_exact_edit_response(
            response,
            expected_path=path,
            base_content=base,
        )


def test_exact_edit_rejects_base_hash_drift():
    path = "src/asnumpy/linalg/direct.py"
    base = "original\n"
    response = _exact_response(
        path,
        base,
        (("original", "replacement"),),
        base_sha256="0" * 64,
    )

    with pytest.raises(DraftGenerationError, match="base hash mismatch"):
        _apply_official_exact_edit_response(
            response,
            expected_path=path,
            base_content=base,
        )


@pytest.mark.parametrize(
    ("response_path", "message"),
    (
        ("include/asnumpy/linalg/product.hpp", "path mismatch"),
        ("../outside.py", "unsafe AsNumpy adapter path"),
    ),
)
def test_exact_edit_rejects_path_drift(response_path, message):
    expected = "src/asnumpy/linalg/direct.py"
    base = "original\n"
    response = _exact_response(
        response_path,
        base,
        (("original", "replacement"),),
    )

    with pytest.raises(DraftGenerationError, match=message):
        _apply_official_exact_edit_response(
            response,
            expected_path=expected,
            base_content=base,
        )


def test_exact_edit_rejects_rebuilt_file_over_size_limit(monkeypatch):
    path = "src/asnumpy/linalg/direct.py"
    base = "original\n"
    response = _exact_response(
        path,
        base,
        (("original", "replacement-is-too-large"),),
    )
    monkeypatch.setattr(provider_module, "_MAX_ADAPTER_FILE_BYTES", 16)

    with pytest.raises(DraftGenerationError, match="rebuilt file exceeds"):
        _apply_official_exact_edit_response(
            response,
            expected_path=path,
            base_content=base,
        )


def test_exact_edit_rejects_empty_rebuilt_file():
    path = "src/asnumpy/linalg/direct.py"
    base = "entire-file"
    response = _exact_response(path, base, ((base, ""),))

    with pytest.raises(DraftGenerationError, match="rebuilt file is empty"):
        _apply_official_exact_edit_response(
            response,
            expected_path=path,
            base_content=base,
        )


@pytest.mark.parametrize(
    ("response", "message"),
    (
        ("", "response is empty"),
        ("bad\r\n", "contains a carriage return"),
        ("bad\x00", "contains a NUL byte"),
    ),
)
def test_exact_edit_reports_safe_response_shape_failures(response, message):
    with pytest.raises(DraftGenerationError, match=message):
        _apply_official_exact_edit_response(
            response,
            expected_path="csrc/linalg/product.cpp",
            base_content="base\n",
        )


def test_exact_edit_rejects_unique_old_text_outside_reviewed_target_window():
    path = "csrc/linalg/product.cpp"
    base = (
        "UNRELATED_SENTINEL\n"
        "NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {\n"
        "    return 0;\n"
        "}\n"
        "\nNPUArray Einsum(const char* spec) { return 1; }\n"
    )
    _rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    response = _exact_response(
        path,
        base,
        (("UNRELATED_SENTINEL", "CHANGED_OUTSIDE_WINDOW"),),
    )

    with pytest.raises(DraftGenerationError, match="leaves its target edit windows"):
        _apply_official_exact_edit_response(
            response,
            expected_path=path,
            base_content=base,
            allowed_intervals=intervals,
        )


def test_concatenate_exact_edit_windows_cover_reviewed_targets():
    bases = {
        "include/asnumpy/array/basic.hpp": (
            "namespace asnumpy {\n}\n"
        ),
        "csrc/array/basic.cpp": (
            "#include <asnumpy/array/basic.hpp>\n"
            "#include <vector>\n\n"
            "namespace asnumpy {\n\n"
            "NPUArray Linspace(const py::object& start, const py::object& end, "
            "const py::object& steps, const py::object& dtype) {\n"
            "    return {};\n"
            "}\n\n"
            "} // namespace asnumpy\n"
        ),
        "bindings/python/bind_array.cpp": "void bind_array() {}\n",
        "src/asnumpy/array.py": "def zeros(): pass\n",
        "src/asnumpy/__init__.py": (
            "if TYPE_CHECKING:\n"
            "    from .array import (\n"
            "        zeros,\n"
            "    )\n"
            "    from .io import load, save, savez, savez_compressed\n"
            "    from .math import (\n"
            "        absolute,\n"
            "    )\n"
            "    from .nn import softmax\n\n"
            "_LAZY_MAPPING = {\n"
            "    # .array\n"
            '    "zeros": ".array",\n'
            "    # .linalg\n"
            '    "linalg": ".linalg",\n'
            "    # .math\n"
            '    "absolute": ".math",\n'
            "    # .random\n"
            '    "random": ".random",\n'
            "}\n"
        ),
        "tests/asnumpy_tests/creation_tests/test_basic.py": (
            "def test_zeros(): pass\n"
        ),
    }
    for path, base in bases.items():
        rendered, intervals = (
            provider_module._render_official_exact_edit_target_windows(
                target_path=path,
                base_content=base,
            )
        )
        assert rendered
        assert intervals

    native_path = "csrc/array/basic.cpp"
    native_base = bases[native_path]
    native_response = _exact_response(
        native_path,
        native_base,
        ((
            "#include <asnumpy/array/basic.hpp>",
            "#include <asnumpy/array/basic.hpp>\n"
            "#include <aclnnop/aclnn_cat.h>",
        ),),
    )
    _rendered, native_intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=native_path,
            base_content=native_base,
        )
    )
    rebuilt = _apply_official_exact_edit_response(
        native_response,
        expected_path=native_path,
        base_content=native_base,
        allowed_intervals=native_intervals,
    )
    assert "#include <aclnnop/aclnn_cat.h>" in rebuilt

    export_path = "src/asnumpy/__init__.py"
    export_base = bases[export_path]
    export_response = _exact_response(
        export_path,
        export_base,
        ((
            '    "zeros": ".array",\n',
            '    "zeros": ".array",\n'
            '    "concatenate": ".array",\n',
        ),),
    )
    _rendered, export_intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=export_path,
            base_content=export_base,
        )
    )
    rebuilt = _apply_official_exact_edit_response(
        export_response,
        expected_path=export_path,
        base_content=export_base,
        allowed_intervals=export_intervals,
    )
    assert '"concatenate": ".array"' in rebuilt


def test_rebuilt_python_syntax_gate_rejects_unparseable_overlay():
    from numpilot.gaps.provider import _validate_rebuilt_python_syntax

    broken = (
        "if TYPE_CHECKING:\n"
        "    from .array import (\n"
        "        zeros,\n"
        "    )    from .io import load\n"
    )
    with pytest.raises(DraftGenerationError) as excinfo:
        _validate_rebuilt_python_syntax("src/asnumpy/__init__.py", broken)
    message = str(excinfo.value)
    assert "src/asnumpy/__init__.py" in message
    assert "line 4" in message
    assert "glued" in message
    assert "newline" in message

    valid = (
        "if TYPE_CHECKING:\n"
        "    from .array import (\n"
        "        zeros,\n"
        "    )\n"
        "    from .io import load\n"
    )
    _validate_rebuilt_python_syntax("src/asnumpy/__init__.py", valid)

    # The deterministic syntax gate only governs Python overlays; C++ files
    # are qualified by the build gate, not by a Python parser.
    _validate_rebuilt_python_syntax("csrc/math/exponents_and_logarithms.cpp", "not python {")


def _malformed_cbrt_init_base():
    return (
        "if TYPE_CHECKING:\n"
        "    from .array import (\n"
        "        zeros,\n"
        "    )    from .io import load, save, savez, savez_compressed\n"
        "    from .math import (\n"
        "        exp,\n"
        "    )\n"
        "    from .nn import softmax\n\n"
        "_LAZY_MAPPING = {\n"
        "    # .array\n"
        '    "zeros": ".array",\n'
        "    # .linalg\n"
        '    "linalg": ".linalg",\n'
        "    # .math\n"
        '    "exp": ".math",\n'
        "    # .random\n"
        '    "random": ".random",\n'
        "}\n"
    )


def test_target_windows_expose_read_only_suffix_for_mid_line_window():
    path = "src/asnumpy/__init__.py"
    base = _malformed_cbrt_init_base()
    rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    array_start = base.find("    from .array import (")
    io_start = base.find("    from .io import")
    assert intervals[0] == (array_start, io_start)
    assert base[intervals[0][0]:intervals[0][1]].endswith("    )")
    assert "WINDOW_SUFFIX (read-only context, not editable" in rendered
    assert (
        "    from .io import load, save, savez, savez_compressed" in rendered
    )
    assert "WINDOW_BOUNDARY" in rendered
    assert "DETERMINISTIC MID-LINE IMPORT BOUNDARY AUDIT" in rendered
    assert "`type_checking_array_imports` at line 4" in rendered
    math_start = rendered.find(
        "TARGET EDIT WINDOW `type_checking_math_imports`"
    )
    assert math_start >= 0
    assert "WINDOW_SUFFIX" not in rendered[math_start:]


def test_window_replace_accepts_newline_fix_for_mid_line_window():
    path = "src/asnumpy/__init__.py"
    base = _malformed_cbrt_init_base()
    _rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    response = _window_response(
        path,
        base,
        (("        zeros,\n    )", "        zeros,\n    )\n"),),
    )
    rebuilt = provider_module._apply_official_window_replace_response(
        response,
        expected_path=path,
        base_content=base,
        allowed_intervals=intervals,
    )
    assert "    )\n    from .io import load" in rebuilt
    ast.parse(rebuilt)


def test_window_replace_rejects_mid_line_replacement_without_newline():
    path = "src/asnumpy/__init__.py"
    base = _malformed_cbrt_init_base()
    _rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    response = _window_response(
        path,
        base,
        (("        zeros,\n    )", "        zeros,\n    ) "),),
    )
    with pytest.raises(
        DraftGenerationError, match="before an import statement"
    ):
        provider_module._apply_official_window_replace_response(
            response,
            expected_path=path,
            base_content=base,
            allowed_intervals=intervals,
        )


def test_window_replace_allows_mid_line_comment_suffix_without_newline():
    path = "src/asnumpy/__init__.py"
    base = (
        "if TYPE_CHECKING:\n"
        "    from .array import (\n"
        "        zeros,\n"
        "    )\n"
        "    from .io import load, save, savez, savez_compressed\n"
        "    from .math import (\n"
        "        exp,\n"
        "    )\n"
        "    from .nn import softmax\n\n"
        "_LAZY_MAPPING = {\n"
        "    # .array\n"
        '    "zeros": ".array",    # .linalg\n'
        '    "linalg": ".linalg",\n'
        "    # .math\n"
        '    "exp": ".math",\n'
        "    # .random\n"
        '    "random": ".random",\n'
        "}\n"
    )
    _rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    response = _window_response(
        path,
        base,
        (('    "zeros": ".array",', '    "zeros": ".arrayX",'),),
    )
    rebuilt = provider_module._apply_official_window_replace_response(
        response,
        expected_path=path,
        base_content=base,
        allowed_intervals=intervals,
    )
    ast.parse(rebuilt)



def test_window_replace_requires_trailing_newline_at_line_boundary():
    path = "src/asnumpy/__init__.py"
    base = (
        "if TYPE_CHECKING:\n"
        "    from .array import (\n"
        "        zeros,\n"
        "    )\n"
        "    from .io import load, save, savez, savez_compressed\n"
        "    from .math import (\n"
        "        exp,\n"
        "    )\n"
        "    from .nn import softmax\n\n"
        "_LAZY_MAPPING = {\n"
        "    # .array\n"
        '    "zeros": ".array",\n'
        "    # .linalg\n"
        '    "linalg": ".linalg",\n'
        "    # .math\n"
        '    "exp": ".math",\n'
        "    # .random\n"
        '    "random": ".random",\n'
        "}\n"
    )
    _rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    response = _window_response(
        path,
        base,
        (("        zeros,\n    )\n", "        zeros,\n    )"),),
    )
    with pytest.raises(DraftGenerationError, match="line boundary"):
        provider_module._apply_official_window_replace_response(
            response,
            expected_path=path,
            base_content=base,
            allowed_intervals=intervals,
        )


def test_exact_edit_can_insert_missing_newline_inside_array_window():
    path = "src/asnumpy/__init__.py"
    base = _malformed_cbrt_init_base()
    _rendered, intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=path,
            base_content=base,
        )
    )
    response = _exact_response(
        path,
        base,
        (("        zeros,\n    )", "        zeros,\n    )\n"),),
    )
    rebuilt = _apply_official_exact_edit_response(
        response,
        expected_path=path,
        base_content=base,
        allowed_intervals=intervals,
    )
    assert "    )\n    from .io import load" in rebuilt
    ast.parse(rebuilt)



def test_cbrt_exact_edit_windows_cover_reviewed_math_targets():
    bases = {
        "include/asnumpy/math/exponents_and_logarithms.hpp": (
            "#pragma once\n"
            "namespace asnumpy {\n"
            "}\n"
        ),
        "csrc/math/exponents_and_logarithms.cpp": (
            "#include <asnumpy/math/exponents_and_logarithms.hpp>\n"
            "\n"
            "namespace asnumpy {\n"
            "\n"
            "NPUArray Exp(const NPUArray& x) {\n"
            "    return {};\n"
            "}\n"
            "\n"
            "} // namespace asnumpy\n"
        ),
        "bindings/python/bind_math.cpp": "void bind_math() {}\n",
        "src/asnumpy/math.py": "def exp(): pass\n",
        "src/asnumpy/__init__.py": (
            "if TYPE_CHECKING:\n"
            "    from .array import (\n"
            "        zeros,\n"
            "    )\n"
            "    from .io import load, save, savez, savez_compressed\n"
            "    from .math import (\n"
            "        exp,\n"
            "    )\n"
            "    from .nn import softmax\n\n"
            "_LAZY_MAPPING = {\n"
            "    # .array\n"
            '    "zeros": ".array",\n'
            "    # .linalg\n"
            '    "linalg": ".linalg",\n'
            "    # .math\n"
            '    "exp": ".math",\n'
            "    # .random\n"
            '    "random": ".random",\n'
            "}\n"
        ),
        "tests/asnumpy_tests/math_tests/test_exponents_and_logarithms.py": (
            "def test_exp(): pass\n"
        ),
    }
    expected_labels = {
        "include/asnumpy/math/exponents_and_logarithms.hpp": (
            "complete_math_exponents_header",
        ),
        "csrc/math/exponents_and_logarithms.cpp": (
            "includes",
            "cbrt_implementation",
        ),
        "bindings/python/bind_math.cpp": ("complete_math_binding_module",),
        "src/asnumpy/math.py": ("complete_math_dispatch_module",),
        "src/asnumpy/__init__.py": (
            "type_checking_array_imports",
            "type_checking_math_imports",
            "lazy_array_exports",
            "lazy_math_exports",
        ),
        "tests/asnumpy_tests/math_tests/test_exponents_and_logarithms.py": (
            "complete_math_exponents_test_module",
        ),
    }
    for path, base in bases.items():
        rendered, intervals = (
            provider_module._render_official_exact_edit_target_windows(
                target_path=path,
                base_content=base,
            )
        )
        assert rendered
        assert (
            "DETERMINISTIC MID-LINE IMPORT BOUNDARY AUDIT" not in rendered
        )
        assert len(intervals) == len(expected_labels[path])
        planned = tuple(
            spec[0]
            for spec in provider_module._OFFICIAL_EXACT_EDIT_TARGET_WINDOWS[path]
        )
        assert planned == expected_labels[path]

    init_path = "src/asnumpy/__init__.py"
    init_base = bases[init_path]
    init_response = _window_response(
        init_path,
        init_base,
        ((
            '    # .math\n    "exp": ".math",\n',
            '    # .math\n    "cbrt": ".math",\n    "exp": ".math",\n',
        ),),
    )
    _rendered, init_intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=init_path,
            base_content=init_base,
        )
    )
    rebuilt = provider_module._apply_official_window_replace_response(
        init_response,
        expected_path=init_path,
        base_content=init_base,
        allowed_intervals=init_intervals,
    )
    assert '"cbrt": ".math"' in rebuilt
    assert "    from .math import (\n        exp,\n    )" in rebuilt

    cpp_path = "csrc/math/exponents_and_logarithms.cpp"
    cpp_base = bases[cpp_path]
    cpp_response = _window_response(
        cpp_path,
        cpp_base,
        (
            (
                "#include <asnumpy/math/exponents_and_logarithms.hpp>\n",
                "#include <asnumpy/math/exponents_and_logarithms.hpp>\n"
                "#include <aclnnop/aclnn_pow.h>\n",
            ),
            (
                "} // namespace asnumpy",
                "NPUArray Cbrt(const NPUArray& x) {\n"
                "    return {};\n"
                "}\n\n"
                "} // namespace asnumpy",
            ),
        ),
    )
    _rendered, cpp_intervals = (
        provider_module._render_official_exact_edit_target_windows(
            target_path=cpp_path,
            base_content=cpp_base,
        )
    )
    rebuilt = provider_module._apply_official_window_replace_response(
        cpp_response,
        expected_path=cpp_path,
        base_content=cpp_base,
        allowed_intervals=cpp_intervals,
    )
    assert "#include <aclnnop/aclnn_pow.h>" in rebuilt
    assert "NPUArray Cbrt(const NPUArray& x)" in rebuilt


def test_official_exact_edit_generation_binds_full_rebuilt_files_and_receipts(
    tmp_path, monkeypatch
):
    paths = (
        "csrc/linalg/product.cpp",
        "src/asnumpy/linalg/direct.py",
        "tests/asnumpy_tests/linalg_tests/test_product.py",
    )
    planned_context = {
        paths[0]: (
            "// unrelated preamble\n"
            "NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {\n"
            "  return 0;\n"
            "}\n"
            "\nNPUArray Einsum(const char* spec) { return 1; }\n"
        ),
        paths[1]: "def matmul(a, b):\n    return 'host'\n",
        paths[2]: (
            "import asnumpy\n\n"
            "def helper():\n    return None\n"
            "\n# ==========================================================================\n"
            "# 1. dot 点积测试\n"
            "def test_dot():\n    assert True\n"
            "\n# ==========================================================================\n"
            "# 5. matmul 矩阵乘法测试\n"
            "def test_matmul():\n    assert False\n"
            "\n# ==========================================================================\n"
            "# 6. matrix_power 矩阵幂测试\n"
            "def test_matrix_power():\n    assert True\n"
        ),
        "include/asnumpy/linalg/product.hpp": "void matmul();\n",
        "bindings/python/bind_linalg.cpp": "linalg.def('matmul');\n",
        "src/asnumpy/utils.py": "class ndarray: ndim = dtype = shape = strides = None\n",
        "bindings/python/bind_utils.cpp": "// strides are exposed in bytes\n",
        "bindings/python/CMakeLists.txt": (
            "target_link_libraries(_core\n"
            "    PUBLIC\n"
            "    ascendcl\n"
            "    runtime\n"
            "    nnopbase\n"
            "    opapi\n"
            "    Python::NumPy\n"
            ")\n"
            "target_link_libraries(_core\n"
            "    PUBLIC\n"
            "    array\n"
            "    linalg\n"
            ")\n"
        ),
    }
    prompt_dependencies = {
        paths[0]: [
            {
                "source": "reference",
                "path": "include/asnumpy/linalg/product.hpp",
            },
            {
                "source": "signed_official_header",
                "path": "SIGNED_OFFICIAL_HEADER/aclnn_matmul.h",
            },
        ],
        paths[1]: [
            {
                "source": "reference",
                "path": "bindings/python/bind_linalg.cpp",
            },
        ],
        paths[2]: [
            {"source": "generated_overlay", "path": paths[1]},
        ],
    }
    library_binding = {
        "schema_version": 1,
        "proof_kind": "reviewed_exact_cmake_target_link_item",
        "cmake_path": "bindings/python/CMakeLists.txt",
        "cmake_sha256": _sha256_text(
            planned_context["bindings/python/CMakeLists.txt"]
        ),
        "target": "_core",
        "link_item": "opapi",
        "selected_library_relative_path": "lib64/libopapi.so",
        "selected_library_basename": "libopapi.so",
        "selected_library_sha256": "2" * 64,
    }
    plan = _context_plan(
        tmp_path,
        "official_aclnn_existing_module",
        "existing_module",
        planned_context,
        paths,
        prompt_dependencies=prompt_dependencies,
        official_library_binding=library_binding,
    )
    header_content = (
        "aclnnStatus aclnnMatmulGetWorkspaceSize();\n"
        "aclnnStatus aclnnMatmul();\n"
    )
    candidate_id = "candidate-official-matmul"
    candidate_sha256 = "c" * 64
    contract_value = {
        "schema_version": 1,
        "contract_type": "official_binary_asnumpy_adapter",
        "strategy": "official_aclnn_existing_module",
        "source_candidate": {
            "artifact_id": candidate_id,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha256,
            "origin_kind": "official_binary",
            "license": {
                "identifier": "LicenseRef-CANN-Toolkit",
                "accepted": False,
                "ref": {
                    "name": "license",
                    "locator": "/evidence/cann-license.json",
                    "sha256": "d" * 64,
                },
            },
        },
        "api": {
            "gap_id": "gap-matmul",
            "api_name": "matmul",
            "qualified_api": "numpy.matmul",
            "asnumpy_api": "asnumpy.matmul",
            "signature_variants": [],
            "shape_rules": ["rank-2 non-empty matrices"],
            "dtype_rules": ["float32"],
            "semantic_source": "numpy.matmul",
            "reference_cases": [],
        },
        "target": {
            "soc": "Ascend910B4",
            "cann_version": "9.0.0",
        },
        "evidence": {
            "gap_spec_sha256": "e" * 64,
            "resolution_decision_sha256": "f" * 64,
            "candidate_mapping_sha256": "1" * 64,
        },
        "official_binary": {
            "candidate_mapping": {
                "candidate_id": "aclnn-matmul-reviewed",
                "qualified_api": "numpy.matmul",
            },
            "candidate_mapping_sha256": "1" * 64,
            "selected_header": {
                "relative_path": "include/aclnnop/aclnn_matmul.h",
                "sha256": _sha256_text(header_content),
                "size": len(header_content.encode("utf-8")),
                "declared_symbols": [
                    "aclnnMatmulGetWorkspaceSize",
                    "aclnnMatmul",
                ],
            },
            "selected_library": {
                "relative_path": "lib64/libopapi.so",
                "sha256": "2" * 64,
                "size": 4096,
                "probe_tool": "nm -D",
                "aclnn_symbol_count": 2,
                "aclnn_symbols_sha256": "3" * 64,
            },
            "required_symbols": [
                "aclnnMatmulGetWorkspaceSize",
                "aclnnMatmul",
            ],
            "semantic_review": ["float32 rank-2 only"],
        },
        "permissions": {
            "reuse_installed_aclnn": True,
            "generate_new_kernel": False,
            "generate_custom_opp": False,
            "activation_allowed": False,
        },
        "activation_allowed": False,
    }
    contract = json.dumps(
        contract_value, sort_keys=True, separators=(",", ":")
    ) + "\n"
    artifact = {
        "schema_version": 1,
        "artifact_type": "official_binary_adapter_input",
        "artifact_id": candidate_id,
        "artifact_id_semantics": "operator_candidate_id_compatibility_alias",
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "strategy": "official_aclnn_existing_module",
        "origin_kind": "official_binary",
        "gap_id": "gap-matmul",
        "qualified_api": "numpy.matmul",
        "asnumpy_api": "asnumpy.matmul",
        "target_soc": "Ascend910B4",
        "cann_version": "9.0.0",
        "gap_spec_sha256": "e" * 64,
        "resolution_decision_sha256": "f" * 64,
        "model_invocation_allowed": True,
        "context_plan_status": "prepared",
        "required_next_gate": "invoke_configured_model_for_official_adapter",
        "new_kernel_generation_allowed": False,
        "custom_opp_generation_allowed": False,
        "activation_allowed": False,
        "adapter_contract_sha256": _sha256_text(contract),
        "asnumpy_adapter_context_plan": plan,
    }
    context = dict(planned_context)
    context["SIGNED_OFFICIAL_HEADER/aclnn_matmul.h"] = header_content
    replacements = {
        paths[0]: ("return 0;", "return aclnnMatmul();"),
        paths[1]: ("return 'host'", "return _C.matmul(a, b)"),
        paths[2]: ("assert False", "assert asnumpy.matmul is not None"),
    }
    prompts = []

    def remote_chat(system, user, **_kwargs):
        path = re.search(r"TARGET PATH: `([^`]+)`", user).group(1)
        response = (
            ""
            if not prompts
            else _window_response(
                path,
                planned_context[path],
                (replacements[path],),
            )
        )
        prompts.append((system, user, dict(_kwargs)))
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=2,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    output = tmp_path / "official-adapter"
    output.mkdir()

    report = provider.generate_asnumpy_adapter_candidate(
        output,
        operator_artifact=artifact,
        adapter_contract=contract,
        asnumpy_context=context,
        expected_paths=paths,
        adapter_context_plan=plan,
        skill_files=ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES,
    )

    assert report["schema_version"] == 4
    assert report["attempt_count"] == len(paths) + 1
    assert report["integration_skill_files"] == list(
        ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES
    )
    for path, (old, new) in replacements.items():
        expected = planned_context[path].replace(old, new)
        assert (output / "candidate" / path).read_text(
            encoding="utf-8"
        ) == expected
        assert report["files_sha256"][path] == _sha256_text(expected)
    reloaded, _payload, hashes = _load_asnumpy_adapter_report(output)
    assert reloaded == report
    assert hashes == report["files_sha256"]
    assert len(prompts) == len(paths) + 1
    assert [kwargs for _system, _user, kwargs in prompts] == [
        {"stream": True},
        {"stream": True},
        {"stream": True},
        {"stream": True},
    ]
    assert all("NUMPILOT_WINDOW_REPLACE_V1" in system for system, _, _ in prompts)
    assert all("complete replacement content for that reviewed window" in system for system, _, _ in prompts)
    assert header_content in prompts[0][1]
    assert header_content in prompts[1][1]
    assert header_content not in prompts[2][1]
    assert header_content not in prompts[3][1]
    assert all(contract.strip() in user for _system, user, _kwargs in prompts)
    assert all(
        "route skill fixture for references/official-aclnn-existing-module.md"
        in user
        for _system, user, _kwargs in prompts
    )
    assert all("ALREADY GENERATED" not in user for _system, user, _ in prompts)
    assert planned_context["include/asnumpy/linalg/product.hpp"] in prompts[0][1]
    assert planned_context["include/asnumpy/linalg/product.hpp"] in prompts[1][1]
    assert planned_context[paths[2]] not in prompts[0][1]
    assert planned_context["bindings/python/bind_linalg.cpp"] in prompts[2][1]
    assert planned_context["src/asnumpy/utils.py"] not in prompts[2][1]
    assert planned_context["bindings/python/bind_utils.cpp"] not in prompts[2][1]
    assert planned_context[paths[0]] not in prompts[2][1]
    rebuilt_direct = planned_context[paths[1]].replace(*replacements[paths[1]])
    assert rebuilt_direct in prompts[3][1]
    assert planned_context[paths[1]] not in prompts[3][1]
    assert planned_context[paths[0]] not in prompts[3][1]
    assert all('"projection_sha256"' in user for _, user, _ in prompts)
    assert '"source": "generated_overlay"' in prompts[3][1]
    assert '"request_body_sha256": "%s"' % ("1" * 64) in prompts[3][1]


def test_manifest_adapter_route_keeps_complete_file_bundle_protocol(
    tmp_path, monkeypatch
):
    path = "csrc/custom_ops/registry/cbrt/op.yaml"
    context = {"CMakeLists.txt": "project(asnumpy)\n"}
    plan = _context_plan(
        tmp_path,
        "manifest_generator",
        "manifest",
        context,
        (path,),
    )
    response = "\n".join((
        "NUMPILOT_BUNDLE_V1",
        "SUMMARY: manifest route fixture",
        "FILE: " + path,
        "CONTENT:",
        "name: cbrt",
        "END_FILE",
        "END_BUNDLE",
    ))
    observed_system = []

    def remote_chat(system, _user):
        observed_system.append(system)
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    output = tmp_path / "manifest-adapter"
    output.mkdir()

    report = provider.generate_asnumpy_adapter_candidate(
        output,
        operator_artifact={"artifact_id": "draft-custom-cbrt"},
        adapter_contract='{"api_name":"cbrt"}',
        asnumpy_context=context,
        expected_paths=(path,),
        adapter_context_plan=plan,
    )

    assert report["files_sha256"] == {path: _sha256_text("name: cbrt")}
    assert len(observed_system) == 1
    assert "NUMPILOT_BUNDLE_V1" in observed_system[0]
    assert "NUMPILOT_EXACT_EDIT_V1" not in observed_system[0]


def test_adapter_http_failure_records_request_without_receipt(
    tmp_path, monkeypatch
):
    path = "csrc/custom_ops/registry/cbrt/op.yaml"
    context = {"CMakeLists.txt": "project(asnumpy)\n"}
    plan = _context_plan(
        tmp_path,
        "manifest_generator",
        "manifest",
        context,
        (path,),
    )

    def remote_failure(_system, _user, **_kwargs):
        raise LlmHttpError(504)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_failure)
    )
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    output = tmp_path / "failed-adapter"
    output.mkdir()

    with pytest.raises(DraftRepairError, match="HTTP status 504") as caught:
        provider.generate_asnumpy_adapter_candidate(
            output,
            operator_artifact={"artifact_id": "draft-custom-cbrt"},
            adapter_contract='{"api_name":"cbrt"}',
            asnumpy_context=context,
            expected_paths=(path,),
            adapter_context_plan=plan,
        )

    assert caught.value.actual_remote_call is True
    assert caught.value.request_count == 1
    assert caught.value.receipt_count == 0
    assert not (output / "candidate").exists()
    assert not (output / "asnumpy-adapter-report.json").exists()
