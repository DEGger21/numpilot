import hashlib

import numpy as np
import pytest

from numpilot.backends.shim import ShimBackend
from numpilot.benchmark_overlay import BenchmarkOverlay, build_analysis_profiles


SOURCE = """
import numpy as np
def build_inputs():
    return {
        "a": np.ones((4, 8), dtype=np.float32),
        "b": np.ones((4, 8), dtype=np.float32),
    }
def forward(a, b):
    joined = np.concatenate((a, b), axis=1)
    return np.add(joined, joined)
"""


def _overlay(operators=("concatenate",), qualified_names=("numpy.concatenate",)):
    return BenchmarkOverlay(
        qualification_id="official-aclnn-test",
        overlay_sha256="a" * 64,
        runtime_identity_sha256="b" * 64,
        operators=tuple(operators),
        qualified_names=tuple(qualified_names),
        resolver=object(),
    )


def test_profiles_capture_exact_concatenate_callsite():
    result = build_analysis_profiles(SOURCE, _overlay())

    assert result.profiles is not None
    assert result.matched_callsites == 1
    assert result.attempted_callsites == 1
    callsite = result.profiles.callsites[0]
    assert callsite.qualified_name == "numpy.concatenate"
    assert callsite.profile.positional_args == 1
    assert callsite.profile.keyword_args == ("axis",)
    assert [profile.shape for profile in callsite.profile.inputs] == [
        (4, 8),
        (4, 8),
    ]
    assert result.profiles.source_sha256 == hashlib.sha256(
        SOURCE.encode("utf-8")
    ).hexdigest()


def test_overlay_api_without_callsites_yields_no_profiles():
    source = (
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return {}\n"
        "def forward():\n"
        "    return np.zeros((2, 2), dtype=np.float32)\n"
    )

    result = build_analysis_profiles(source, _overlay())

    assert result.profiles is None
    assert result.attempted_callsites == 0


def test_creation_calls_are_not_tensor_operand_profiles():
    source = (
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return {}\n"
        "def forward():\n"
        "    return np.zeros((2, 2), dtype=np.float32)\n"
    )

    result = build_analysis_profiles(
        source,
        _overlay(operators=("zeros",), qualified_names=("numpy.zeros",)),
    )

    assert result.profiles is None
    assert result.attempted_callsites == 1
    assert result.matched_callsites == 0


def test_shim_overlay_concatenate_is_opt_in_and_matches_numpy():
    assert not hasattr(ShimBackend().get_module(), "concatenate")

    backend = ShimBackend(extra_ops=("concatenate",))
    module = backend.get_module()
    left = np.arange(8, dtype=np.float32).reshape(2, 4)
    right = np.ones((2, 4), dtype=np.float32)

    output = module.concatenate(
        (backend.to_device(left), backend.to_device(right)), axis=1
    )

    np.testing.assert_array_equal(
        backend.to_host(output), np.concatenate((left, right), axis=1)
    )


def test_shim_rejects_unknown_overlay_operator():
    with pytest.raises(ValueError, match="no overlay stand-in"):
        ShimBackend(extra_ops=("take",))


def test_sequence_operand_without_axis_keyword_profiles_explicit_default_axis():
    source = SOURCE.replace(
        "np.concatenate((a, b), axis=1)",
        "np.concatenate((a, b))",
    )

    result = build_analysis_profiles(source, _overlay())

    assert result.profiles is not None
    assert result.attempted_callsites == 1
    assert result.matched_callsites == 1
    assert result.profiles.callsites[0].profile.sequence_axis == 0
