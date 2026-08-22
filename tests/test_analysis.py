import hashlib

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


def test_rejects_unrewritten_python_array_operators():
    result = AnalysisAgent().analyze(
        "import numpy as np\ndef forward(x): return np.sqrt(x + 1)\n"
    )
    assert not result.migratable
    assert "python_operator.Add" in result.unsupported


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

    assert no_import.op_uses == []
    assert parameter_shadow.op_uses == []
    assert local_shadow.op_uses == []


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
