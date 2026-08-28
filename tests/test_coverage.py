from numpilot.capability_catalog import ASNUMPY_CAPABILITY_CATALOG
from numpilot.coverage import CUSTOM_OPS, SUPPORTED_OPS, is_custom, is_supported


def test_core_ops_supported():
    for name in ["add", "subtract", "multiply", "divide", "matmul",
                 "exp", "sqrt", "sum", "mean", "maximum", "relu", "softmax"]:
        assert is_supported(name), name


def test_unsupported_op():
    assert not is_supported("fft")          # 不在覆盖集
    assert not is_supported("convolve")
    assert not is_supported("angle")
    assert not is_custom("angle")


def test_supported_ops_is_frozenset():
    assert isinstance(SUPPORTED_OPS, frozenset)
    assert "matmul" in SUPPORTED_OPS
    assert SUPPORTED_OPS == ASNUMPY_CAPABILITY_CATALOG.top_level_keys
    assert len(SUPPORTED_OPS) == 121
    assert CUSTOM_OPS == frozenset({"gelu", "relu", "softmax"})
