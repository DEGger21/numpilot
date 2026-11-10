import hashlib
from dataclasses import replace

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.capabilities import (
    AnalysisCallsiteProfile,
    AnalysisProfileSet,
    ExecutionProvenance,
    OperatorCallProfile,
    OperatorCapability,
    OperatorSignature,
    TensorProfile,
    match_capability_profile,
)

SRC_OK = """
import numpy as np
def forward(x, w):
    h = np.matmul(x, w)
    return np.maximum(h, 0)
"""

SRC_GAP = """
import numpy as np
def forward(x):
    return np.fft.fft(x)
"""

def test_extracts_op_uses():
    r = AnalysisAgent().analyze(SRC_OK)
    names = {op.name for op in r.op_uses}
    assert "matmul" in names and "maximum" in names
    assert r.migratable and r.unsupported == []

def test_detects_unsupported():
    r = AnalysisAgent().analyze(SRC_GAP)
    assert "fft" in r.unsupported
    assert not r.migratable


def test_nonexistent_angle_is_not_part_of_the_static_asnumpy_contract():
    result = AnalysisAgent().analyze(
        "import numpy as np\ndef forward(x): return np.angle(x)\n"
    )

    assert result.unsupported == ["angle"]
    assert result.op_uses[0].qualified_name == "numpy.angle"
    assert result.op_uses[0].reason == "API 不在静态覆盖集"


def test_est_elements_from_zeros():
    src = "import numpy as np\ndef forward():\n    return np.zeros((100, 200))\n"
    r = AnalysisAgent().analyze(src)
    z = [op for op in r.op_uses if op.name == "zeros"][0]
    assert z.est_elements == 20000


def test_build_inputs_and_top_level_numpy_are_outside_npu_island():
    src = """
import numpy as np
x = np.fft.fft(np.ones(4))
def build_inputs():
    return (np.random.rand(4),)
def forward(x):
    return np.square(x)
"""
    result = AnalysisAgent().analyze(src)
    assert result.migratable
    assert [op.name for op in result.op_uses] == ["square"]


def test_requires_exactly_one_forward_function():
    no_forward = AnalysisAgent().analyze("import numpy as np\ndef f(x): return np.square(x)")
    two = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x): return np.square(x)\n"
        "def forward(x): return np.sqrt(x)\n"
    )
    assert not no_forward.migratable
    assert not two.migratable


def test_rejects_unverified_scalar_binary_signature_but_keeps_relu_pattern():
    scalar = AnalysisAgent().analyze(
        "import numpy as np\ndef forward(x): return np.add(x, 1.0)\n"
    )
    relu = AnalysisAgent().analyze(
        "import numpy as np\ndef forward(x): return np.maximum(x, 0)\n"
    )
    assert not scalar.migratable
    assert scalar.unsupported == ["add"]
    assert "标量" in scalar.op_uses[0].reason
    assert relu.migratable


def test_rejects_unverified_numpy_ufunc_keywords():
    result = AnalysisAgent().analyze(
        "import numpy as np\ndef forward(x): return np.sqrt(x, out=x)\n"
    )
    assert not result.migratable
    assert result.unsupported == ["sqrt"]
    assert "out" in result.op_uses[0].reason


@pytest.mark.parametrize(
    "call, marker",
    [
        ("np.linspace(0.0, 1.0, num=8)", "num"),
        ("np.linspace(start=0.0, stop=1.0)", "stop"),
        ("np.full(shape=(2,), fill_value=1.0)", "fill_value"),
        ("np.full_like(a=x, fill_value=1.0)", "a"),
        ("np.full((2,), 1.0, None, 'C')", "位置参数"),
        ("np.sum(x, initial=0.0)", "initial"),
        ("np.sum(x, None, np.float32)", "位置参数"),
        ("np.amax(x, None, None)", "位置参数"),
        ("np.add(x, x, None)", "位置参数"),
        ("np.sin(x, None)", "位置参数"),
        ("np.dot(x, x, None)", "位置参数"),
        ("np.einsum('i,i->', x, x, optimize=True)", "optimize"),
        ("np.all(a=x)", "a"),
        ("np.around(a=x)", "a"),
        ("np.real(val=x)", "val"),
        ("np.add(*x)", "*args"),
        ("np.sum(x, **x)", "**kwargs"),
        ("np.add()", "必需参数"),
        ("np.full(shape=(2,), dtype=np.float32)", "必需参数"),
        ("np.sum(x, keepdims=True)", "axis=None"),
        ("np.sum(x, dtype=np.float32)", "axis=None"),
        ("np.max(x, keepdims=True)", "axis=None"),
        ("np.all(x, keepdims=True)", "axis=None"),
        ("np.mean(x, 0, axis=1)", "重复绑定"),
        ("np.dot(x, x, a=x)", "重复绑定"),
        ("np.sort(x, stable=True)", "stable"),
        ("np.zeros((2,))", "必需参数"),
        ("np.zeros((2,), dtype=None)", "静态可验证"),
        ("np.eye(2, 3)", "位置参数"),
        ("np.full((2,), x, dtype=np.float32)", "实数标量"),
        ("np.linspace(x, 1.0)", "实数标量"),
        ("np.linspace(0.0, 1.0, 0)", "正整数"),
        ("np.all(x, axis=0)", "单元素"),
        ("np.all(x, axis=())", "单元素"),
        ("np.all(x, axis=(0, 1))", "单元素"),
        ("np.sum(x)", "Python scalar"),
        ("np.sum(x, axis=(0, 1))", "静态整数 axis"),
        ("np.cumsum(x)", "显式静态整数 axis"),
        ("np.cross(x, x)", "显式静态整数 axis"),
        ("np.sort(x, axis=None)", "静态整数 axis"),
        ("np.einsum(x, [0])", "静态字符串"),
        ("np.add(x, x, dtype=np.float32)", "dtype"),
        ("np.cosh(x, dtype=np.float32)", "dtype"),
        ("np.vdot(x, 2)", "静态证明为数组"),
        ("np.greater(1.5, x)", "第一个操作数"),
        ("np.power(1, 2)", "至少一个数组"),
    ],
)
def test_rejects_numpy_calls_whose_signature_changes_after_rewrite(call, marker):
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x):\n"
        f"    return {call}\n"
    )

    assert not result.migratable
    assert marker in result.op_uses[0].reason


@pytest.mark.parametrize(
    "call",
    [
        "np.dot(a=x, b=x)",
        "np.sum(a=x, axis=0, keepdims=True)",
        "np.sum(a=x, axis=0, dtype=np.float32)",
        "np.mean(a=x, axis=0, dtype=np.float32)",
        "np.clip(a=x, a_min=0.0, a_max=1.0)",
        "np.identity(n=4, dtype=np.float32)",
        "np.zeros((2,), dtype=np.float32)",
        "np.eye(2, dtype=np.float32)",
        "np.full((2,), 1.0, dtype=np.float32)",
        "np.linspace(0.0, 1.0, 8, dtype=np.float32)",
        "np.all(x, axis=(0,), keepdims=True)",
        "np.cumsum(a=x, axis=0, dtype=np.float32)",
        "np.cross(a=x, b=x, axis=0)",
        "np.sort(a=x)",
        "np.einsum('i,i->', x, x)",
        "np.greater(x, 1.5)",
        "np.equal(x, 1)",
        "np.power(x, 2)",
        "np.power(2, x)",
        "np.fix(x=x)",
        "np.sinc(x=x)",
        "np.isneginf(x=x)",
        "np.isposinf(x=x)",
    ],
)
def test_accepts_keywords_shared_by_numpy_and_asnumpy(call):
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x):\n"
        f"    return {call}\n"
    )

    assert result.migratable


def test_rejects_unrewritten_python_array_operators():
    result = AnalysisAgent().analyze(
        "import numpy as np\ndef forward(x): return np.sqrt(x + 1)\n"
    )
    assert not result.migratable
    assert "python_operator.Add" in result.unsupported


def test_inplace_augassign_is_reported_with_a_reason():
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x, y):\n"
        "    x += y\n"
        "    return np.sqrt(x)\n"
    )
    assert not result.migratable
    assert "python_structure.AugAssign" in result.unsupported
    reason = [
        use.reason for use in result.op_uses
        if use.name == "python_structure.AugAssign"
    ][0]
    assert "原地" in reason


def test_requires_a_real_numpy_import_and_respects_shadowing():
    no_import = AnalysisAgent().analyze(
        "def forward(x): return np.square(x)\n"
    )
    parameter_shadow = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(np): return np.square(1)\n"
    )
    local_shadow = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x):\n"
        "    np = x\n"
        "    return np.square(x)\n"
    )

    assert not no_import.migratable
    assert no_import.op_uses == []
    assert parameter_shadow.op_uses == []
    assert local_shadow.op_uses == []


@pytest.mark.parametrize("attribute", ["size", "T", "real", "flags", "__dict__"])
def test_rejects_array_attributes_outside_the_shared_read_only_surface(attribute):
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.add(x, x).%s\n" % attribute
    )

    assert not result.migratable
    assert "python_attribute." + attribute in result.unsupported


@pytest.mark.parametrize(
    "attribute", ["shape", "dtype", "ndim", "itemsize", "nbytes", "strides"]
)
def test_accepts_array_metadata_shared_with_asnumpy(attribute):
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.add(x, x).%s\n" % attribute
    )

    assert result.migratable


@pytest.mark.parametrize(
    "scalar_expression",
    ["2.0", "np.random.rand()", "(1.0, 2.0)", "None", "'bad'"],
)
def test_build_inputs_scalar_or_nonarray_cannot_mask_array_only_binding(
    scalar_expression,
):
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return (np.ones(2, dtype=np.float32), %s)\n"
        "def forward(x, scale):\n"
        "    return np.add(x, scale)\n" % scalar_expression,
        strict_input_types=True,
    )

    assert not result.migratable
    assert result.unsupported == ["add"]


def test_build_inputs_random_array_is_proven_for_array_only_binding():
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return (np.ones(2, dtype=np.float32), np.random.rand(2))\n"
        "def forward(x, scale):\n"
        "    return np.add(x, scale)\n",
        strict_input_types=True,
    )

    assert result.migratable


def test_missing_ufunc_array_result_can_feed_a_supported_array_only_op():
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return (np.ones(2, dtype=np.float32),)\n"
        "def forward(x):\n"
        "    y = np.cbrt(x)\n"
        "    return np.add(y, x)\n",
        strict_input_types=True,
    )

    assert result.unsupported == ["cbrt"]


def test_forward_destructuring_is_outside_the_strict_array_surface():
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return (np.ones(2, dtype=np.float32),)\n"
        "def forward(x):\n"
        "    a, b = x\n"
        "    return np.add(a, b)\n",
        strict_input_types=True,
    )

    assert not result.migratable
    assert "python_structure.DestructuringAssign" in result.unsupported


def test_from_numpy_import_is_canonical_and_sensitive_values_are_redacted():
    result = AnalysisAgent().analyze(
        "from numpy import cbrt as cube_root\n"
        "def forward(x):\n"
        "    return cube_root('do-not-send', token='secret-value')\n"
    )

    op = result.op_uses[0]
    assert op.qualified_name == "numpy.cbrt"
    assert op.call_expr == (
        "numpy.cbrt(<arg0>, token=<redacted>)"
    )
    assert "do-not-send" not in op.call_expr
    assert "secret-value" not in op.call_expr


def test_nested_namespace_leaf_does_not_inherit_top_level_coverage():
    result = AnalysisAgent().analyze(
        "import numpy as np\n"
        "def forward(x): return np.some_namespace.sum(x)\n"
    )

    assert result.op_uses[0].qualified_name == "numpy.some_namespace.sum"
    assert result.op_uses[0].supported is False


def test_forward_local_import_is_rejected_before_strict_surface_validation():
    result = AnalysisAgent().analyze(
        "def forward(x):\n"
        "    from numpy import sqrt\n"
        "    return sqrt(x)\n"
    )

    assert not result.migratable
    assert "python_structure.ImportFrom" in result.unsupported
    blocker = next(
        op for op in result.op_uses
        if op.name == "python_structure.ImportFrom"
    )
    assert "forward 内部 import" in blocker.reason


class _ProfileResolver:
    def __init__(self, capability=None, error=None):
        self.capability = capability or OperatorCapability(
            name="cbrt",
            available=True,
            provenance=ExecutionProvenance.CUSTOM_NPU,
            evidence="qualification:qual-test",
            constraints=(
                "dtype=float32",
                "rank>=1",
                "contiguous=true",
            ),
            signature_variants=(OperatorSignature(positional_args=1),),
        )
        self.error = error
        self.calls = []

    def resolve_profile_capability(self, qualified_name, profile):
        self.calls.append((qualified_name, profile))
        if self.error is not None:
            raise self.error
        return match_capability_profile(self.capability, profile)


class _SessionProfileResolver(_ProfileResolver):
    def __init__(self):
        super().__init__()
        self.prepared = []

    def prepare_analysis_profile_resolver(self, qualified_names):
        self.prepared.append(tuple(qualified_names))
        return self


def _profile_set(source, callsites, *, source_sha256=None):
    return AnalysisProfileSet(
        source_sha256=(
            source_sha256
            or hashlib.sha256(source.encode("utf-8")).hexdigest()
        ),
        callsites=tuple(callsites),
    )


def _callsite(op, *, dtype="float32", shape=(8,), contiguous=True):
    return AnalysisCallsiteProfile(
        qualified_name=op.qualified_name,
        lineno=op.lineno,
        col_offset=op.col_offset,
        profile=OperatorCallProfile(
            inputs=(TensorProfile(dtype, shape, contiguous),),
            positional_args=op.positional_args,
            keyword_args=op.keyword_args,
        ),
    )


def test_exact_external_profile_and_overlay_capability_promote_missing_callsite():
    source = "import numpy as np\ndef forward(x):\n    return np.cbrt(x)\n"
    op = AnalysisAgent().analyze(source).op_uses[0]
    resolver = _ProfileResolver()

    result = AnalysisAgent(resolver).analyze(
        source,
        profiles=_profile_set(source, (_callsite(op),)),
    )

    assert result.migratable
    assert result.unsupported == []
    assert result.op_uses[0].supported
    assert len(resolver.calls) == 1


@pytest.mark.parametrize(
    "profile_kwargs",
    [
        {"dtype": "float64"},
        {"shape": ()},
        {"contiguous": False},
    ],
)
def test_profile_mismatch_keeps_missing_api_unsupported(profile_kwargs):
    source = "import numpy as np\ndef forward(x):\n    return np.cbrt(x)\n"
    op = AnalysisAgent().analyze(source).op_uses[0]
    resolver = _ProfileResolver()

    result = AnalysisAgent(resolver).analyze(
        source,
        profiles=_profile_set(source, (_callsite(op, **profile_kwargs),)),
    )

    assert not result.migratable
    assert result.unsupported == ["cbrt"]
    assert not result.op_uses[0].supported


def test_stale_or_wrong_callsite_profile_is_never_used():
    source = "import numpy as np\ndef forward(x):\n    return np.cbrt(x)\n"
    op = AnalysisAgent().analyze(source).op_uses[0]

    stale_resolver = _ProfileResolver()
    stale = AnalysisAgent(stale_resolver).analyze(
        source,
        profiles=_profile_set(
            source,
            (_callsite(op),),
            source_sha256="0" * 64,
        ),
    )
    wrong_resolver = _ProfileResolver()
    wrong_callsite = AnalysisCallsiteProfile(
        qualified_name=op.qualified_name,
        lineno=op.lineno + 1,
        col_offset=op.col_offset,
        profile=_callsite(op).profile,
    )
    wrong = AnalysisAgent(wrong_resolver).analyze(
        source,
        profiles=_profile_set(source, (wrong_callsite,)),
    )

    assert not stale.migratable and not wrong.migratable
    assert stale_resolver.calls == []
    assert wrong_resolver.calls == []


def test_every_missing_callsite_requires_its_own_exact_profile():
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    a = np.cbrt(x)\n"
        "    return np.cbrt(a)\n"
    )
    ops = AnalysisAgent().analyze(source).op_uses
    resolver = _ProfileResolver()

    partial = AnalysisAgent(resolver).analyze(
        source,
        profiles=_profile_set(source, (_callsite(ops[0]),)),
    )

    assert not partial.migratable
    assert [op.supported for op in partial.op_uses] == [True, False]
    assert len(resolver.calls) == 1


def test_analysis_uses_one_resolver_snapshot_for_all_profiled_callsites():
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    a = np.cbrt(x)\n"
        "    return np.cbrt(a)\n"
    )
    ops = AnalysisAgent().analyze(source).op_uses
    resolver = _SessionProfileResolver()

    result = AnalysisAgent(resolver).analyze(
        source,
        profiles=_profile_set(
            source,
            tuple(_callsite(op) for op in ops),
        ),
    )

    assert result.migratable
    assert resolver.prepared == [("numpy.cbrt",)]
    assert len(resolver.calls) == 2


def test_profile_resolver_never_bypasses_existing_surface_rejections():
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    a = np.add(x, 1.0)\n"
        "    return np.sqrt(a, out=a)\n"
    )
    ops = AnalysisAgent().analyze(source).op_uses
    resolver = _ProfileResolver()
    unrelated = AnalysisCallsiteProfile(
        qualified_name="numpy.cbrt",
        lineno=ops[0].lineno,
        col_offset=ops[0].col_offset,
        profile=OperatorCallProfile(
            inputs=(TensorProfile("float32", (8,), True),),
            positional_args=1,
            keyword_args=(),
        ),
    )

    result = AnalysisAgent(resolver).analyze(
        source,
        profiles=_profile_set(source, (unrelated,)),
    )

    assert not result.migratable
    assert set(result.unsupported) == {"add", "sqrt"}
    assert resolver.calls == []


def test_profile_resolver_integrity_errors_propagate():
    source = "import numpy as np\ndef forward(x):\n    return np.cbrt(x)\n"
    op = AnalysisAgent().analyze(source).op_uses[0]
    resolver = _ProfileResolver(error=RuntimeError("overlay integrity failure"))

    with pytest.raises(RuntimeError, match="overlay integrity failure"):
        AnalysisAgent(resolver).analyze(
            source,
            profiles=_profile_set(source, (_callsite(op),)),
        )


def _concatenate_capability():
    return OperatorCapability(
        name="concatenate",
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="qualification:concatenate-test",
        constraints=(
            "dtype=float32",
            "rank>=2",
            "rank<=2",
            "contiguous=true",
        ),
        signature_variants=(
            OperatorSignature(
                positional_args=1, keyword_args=("axis",)
            ),
        ),
    )


def test_promoted_missing_api_produces_a_proven_array_downstream():
    source = (
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return {\n"
        "        'a': np.ones((4, 8), dtype=np.float32),\n"
        "        'b': np.ones((4, 8), dtype=np.float32),\n"
        "    }\n"
        "def forward(a, b):\n"
        "    joined = np.concatenate((a, b), axis=1)\n"
        "    return np.add(joined, joined)\n"
    )
    baseline = AnalysisAgent().analyze(source, strict_input_types=True)
    concat = next(op for op in baseline.op_uses if op.name == "concatenate")
    assert concat.supported is False
    assert "add" in baseline.unsupported

    resolver = _ProfileResolver(_concatenate_capability())
    callsite = AnalysisCallsiteProfile(
        qualified_name="numpy.concatenate",
        lineno=concat.lineno,
        col_offset=concat.col_offset,
        profile=OperatorCallProfile(
            inputs=(
                TensorProfile("float32", (4, 8), True),
                TensorProfile("float32", (4, 8), True),
            ),
            positional_args=1,
            keyword_args=("axis",),
            sequence_axis=1,
        ),
    )

    result = AnalysisAgent(resolver).analyze(
        source,
        profiles=_profile_set(source, (callsite,)),
        strict_input_types=True,
    )

    assert result.unsupported == []
    assert result.migratable
    promoted = next(op for op in result.op_uses if op.name == "concatenate")
    assert promoted.supported is True
    assert promoted.reason is None

    # The old generic fixture lacks sequence/axis evidence. It must neither
    # promote concatenate nor manufacture a proven array for downstream add.
    generic_callsite = replace(
        callsite, profile=replace(callsite.profile, sequence_axis=None)
    )
    rejected = AnalysisAgent(_ProfileResolver(_concatenate_capability())).analyze(
        source,
        profiles=_profile_set(source, (generic_callsite,)),
        strict_input_types=True,
    )
    assert set(rejected.unsupported) == {"add", "concatenate"}
    assert not rejected.migratable
