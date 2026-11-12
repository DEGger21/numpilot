# tests/test_benchmark_lowering.py
"""契约测试：benchmark 专用的整程序内联与静态控制流展开。"""
import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.benchmark_lowering import lower_benchmark_program


def _run_forward(source):
    namespace = {}
    exec(compile(source, "<benchmark-lowering>", "exec"), namespace)
    inputs = namespace["build_inputs"]()
    return namespace["forward"](**inputs)


def _kinds(lowered):
    return [item.kind for item in lowered.rewrites]


def test_inlines_helper_with_static_control_flow_and_matches_original():
    original = """
import numpy as np

def _scale(x):
    return np.multiply(x, np.float32(2.0))

def build_inputs():
    return {'x': np.ones((3,), dtype=np.float32)}

def forward(x):
    y = _scale(x)
    for i in range(2):
        y = np.add(y, np.float32(1.0))
    if True:
        y = np.multiply(y, np.float32(3.0))
    else:
        y = np.subtract(y, np.float32(1.0))
    return {'out': y}
"""
    lowered = lower_benchmark_program(original)

    assert lowered.applied
    assert "inline_helper" in _kinds(lowered)
    assert "unroll_for" in _kinds(lowered)
    assert "static_if" in _kinds(lowered)
    forward_body = lowered.source.split("def forward", 1)[1]
    assert "_scale(" not in forward_body
    assert "for i in range" not in forward_body

    np.testing.assert_allclose(
        _run_forward(original)["out"],
        _run_forward(lowered.source)["out"],
    )
    np.testing.assert_allclose(_run_forward(lowered.source)["out"], [12.0] * 3)


def test_inlines_helper_with_tuple_return_and_destructuring():
    original = """
import numpy as np

def _pair(x):
    return np.add(x, np.float32(1.0)), np.subtract(x, np.float32(1.0))

def build_inputs():
    return {'x': np.arange(4, dtype=np.float32)}

def forward(x):
    plus, minus = _pair(x)
    return {'plus': plus, 'minus': minus}
"""
    lowered = lower_benchmark_program(original)

    assert lowered.applied
    assert _kinds(lowered) == ["inline_helper"]
    forward_body = lowered.source.split("def forward", 1)[1]
    assert "_pair(" not in forward_body

    expected = _run_forward(original)
    actual = _run_forward(lowered.source)
    np.testing.assert_allclose(actual["plus"], expected["plus"])
    np.testing.assert_allclose(actual["minus"], expected["minus"])


def test_unrolls_static_while_loop():
    original = """
import numpy as np

def build_inputs():
    return {'x': np.ones((2,), dtype=np.float32)}

def forward(x):
    i = 0
    y = x
    while i < 3:
        y = np.multiply(y, np.float32(2.0))
        i += 1
    return {'out': y}
"""
    lowered = lower_benchmark_program(original)

    assert lowered.applied
    assert "unroll_while" in _kinds(lowered)
    assert "while " not in lowered.source.split("def forward", 1)[1]

    np.testing.assert_allclose(_run_forward(lowered.source)["out"], [8.0, 8.0])


def test_fails_closed_on_array_valued_condition():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.ones((3,), dtype=np.float32)}

def forward(x):
    if np.sum(x) > 0:
        return {'out': np.add(x, np.float32(1.0))}
    return {'out': x}
"""
    lowered = lower_benchmark_program(source)

    assert not lowered.applied
    assert lowered.source == source
    assert lowered.notes
    assert "whole_program_lowering_skipped" in lowered.notes[0]


def test_fails_closed_on_non_static_for_iterable():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.ones((3,), dtype=np.float32)}

def forward(x):
    y = x
    for step in [1.0, 2.0]:
        y = np.add(y, step)
    return {'out': y}
"""
    lowered = lower_benchmark_program(source)

    assert not lowered.applied
    assert lowered.source == source


def test_build_inputs_is_never_inlined():
    source = """
import numpy as np

def _make():
    return np.arange(4, dtype=np.float32)

def build_inputs():
    return {'x': _make()}

def forward(x):
    return {'out': np.add(x, np.float32(1.0))}
"""
    lowered = lower_benchmark_program(source)

    assert "inline_helper" not in _kinds(lowered)
    assert "def _make" in lowered.source
    assert "_make()" in lowered.source

    namespace = {}
    exec(compile(lowered.source, "<lowered>", "exec"), namespace)
    inputs = namespace["build_inputs"]()
    np.testing.assert_allclose(inputs["x"], [0.0, 1.0, 2.0, 3.0])


def test_lowered_structural_program_passes_strict_analysis():
    source = """
import numpy as np

def _affine(x, scale, bias):
    return np.add(np.multiply(x, scale), bias)

def build_inputs():
    return {'x': np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(2, 3)}

def forward(x):
    y = x
    for i in range(2):
        y = _affine(y, 2.0, 0.5)
    return {'out': y}
"""
    lowered = lower_benchmark_program(source)
    assert lowered.applied

    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )

    assert analysis.migratable, analysis.unsupported


def test_lowers_isnan_to_not_equal_exactly():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([0.0, -0.0, np.nan, np.inf, -np.inf, -1.0], dtype=np.float32)}

def forward(x):
    return {'out': np.isnan(x)}
"""
    lowered = lower_benchmark_program(source)

    assert "not_equal" in lowered.source
    np.testing.assert_array_equal(
        _run_forward(lowered.source)["out"],
        _run_forward(source)["out"],
    )


def test_lowers_cbrt_preserving_sign_and_negative_zero():
    source = """
import numpy as np

def build_inputs():
    x = np.linspace(-27, 27, 257, dtype=np.float32)
    x[128] = -0.0
    return {'x': x}

def forward(x):
    return {'out': np.cbrt(x)}
"""
    lowered = lower_benchmark_program(source)

    assert "copysign" in lowered.source
    assert "full_like" in lowered.source
    expected = _run_forward(source)["out"]
    actual = _run_forward(lowered.source)["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    negative = expected < 0
    assert np.array_equal(np.signbit(actual[negative]), np.signbit(expected[negative]))
    assert np.signbit(actual[128]) == np.signbit(expected[128]) == True


def test_negative_boolean_axis_uses_single_element_tuple():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([[[True, False], [True, True]]])}

def forward(x):
    return {'out': np.all(x, axis=-1, keepdims=True)}
"""
    lowered = lower_benchmark_program(source)

    assert "(-1,)" in lowered.source
    np.testing.assert_array_equal(
        _run_forward(lowered.source)["out"],
        _run_forward(source)["out"],
    )


def test_lowers_logspace_to_power_of_linear_ramp():
    source = """
import numpy as np

def build_inputs():
    return {'start': np.array(-2.0), 'stop': np.array(2.0)}

def forward(start, stop):
    decades = np.logspace(start, stop, 17, base=10.0)
    octaves = np.logspace(start, stop, 17, base=2.0)
    return {'decades': decades, 'octaves': octaves}
"""
    lowered = lower_benchmark_program(source)

    assert "composition" in _kinds(lowered)
    assert "np.logspace" not in lowered.source
    expected = _run_forward(source)
    actual = _run_forward(lowered.source)
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-12, atol=0.0
        )

    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert analysis.migratable, analysis.unsupported


def test_lowers_geomspace_to_ratio_power():
    source = """
import numpy as np

def build_inputs():
    return {'start': np.array(0.125), 'stop': np.array(128.0)}

def forward(start, stop):
    ascending = np.geomspace(start, stop, 21)
    descending = np.geomspace(stop, start, 21)
    return {'ascending': ascending, 'descending': descending}
"""
    lowered = lower_benchmark_program(source)

    assert "composition" in _kinds(lowered)
    assert "np.geomspace" not in lowered.source
    expected = _run_forward(source)
    actual = _run_forward(lowered.source)
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-12, atol=0.0
        )

    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert analysis.migratable, analysis.unsupported


def test_array_augassign_fails_closed_instead_of_rebinding():
    source = """
import numpy as np

def build_inputs():
    base = np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)
    return {'x': base[1:], 'y': base[:-1]}

def forward(x, y):
    x += y
    return {'out': x, 'alias': y}
"""
    lowered = lower_benchmark_program(source)

    assert lowered.source == source
    assert any(
        "augmented assignment on array" in note
        for note in lowered.notes
    )
    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert not analysis.migratable
    assert "python_structure.AugAssign" in analysis.unsupported


def test_scalar_augassign_still_unrolls():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.ones((3,), dtype=np.float32)}

def forward(x):
    total = 0
    for i in range(4):
        total += i
    return {'out': np.multiply(x, total)}
"""
    lowered = lower_benchmark_program(source)

    assert lowered.applied
    assert "for i in range" not in lowered.source
    np.testing.assert_allclose(
        _run_forward(lowered.source)["out"],
        _run_forward(source)["out"],
    )

def test_drops_default_sort_kind_without_changing_values():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([[3, 1, 2, 1], [0, -1, 0, 2]], dtype=np.float32)}

def forward(x):
    out = np.sort(x, axis=1, kind="quicksort")
    return {'out': out}
"""
    lowered = lower_benchmark_program(source)

    assert "contract" in _kinds(lowered)
    assert "kind" not in lowered.source
    np.testing.assert_array_equal(
        _run_forward(lowered.source)["out"],
        _run_forward(source)["out"],
    )

    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert analysis.migratable, analysis.unsupported


def test_keeps_stable_sort_as_shared_stable_keyword():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float64)}

def forward(x):
    out = np.sort(x, kind="stable")
    return {'out': out}
"""
    lowered = lower_benchmark_program(source)

    assert "contract" in _kinds(lowered)
    assert "stable=True" in lowered.source
    assert "kind" not in lowered.source
    expected = _run_forward(source)["out"]
    actual = _run_forward(lowered.source)["out"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))

    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert analysis.migratable, analysis.unsupported


def test_sort_kind_is_not_dropped_when_it_is_not_a_literal():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([3, 1, 2], dtype=np.float32), 'kind': 'stable'}

def forward(x, kind):
    out = np.sort(x, kind=kind)
    return {'out': out}
"""
    lowered = lower_benchmark_program(source)

    assert "contract" not in _kinds(lowered)
    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert not analysis.migratable
    assert "sort" in analysis.unsupported


def test_sort_kind_drop_rejects_an_unverified_companion_keyword():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([3, 1, 2], dtype=np.float32)}

def forward(x):
    out = np.sort(x, kind="stable", order="field")
    return {'out': out}
"""
    lowered = lower_benchmark_program(source)

    assert "contract" not in _kinds(lowered)
    analysis = AnalysisAgent().analyze(
        lowered.source,
        "program.py",
        strict_input_types=True,
    )
    assert not analysis.migratable
    assert "sort" in analysis.unsupported

