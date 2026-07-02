from numpilot.coverage import SUPPORTED_OPS, is_supported

def test_core_ops_supported():
    for name in ["add", "subtract", "multiply", "divide", "matmul",
                 "exp", "sqrt", "sum", "mean", "maximum", "relu", "softmax"]:
        assert is_supported(name), name

def test_unsupported_op():
    assert not is_supported("fft")          # 不在覆盖集
    assert not is_supported("convolve")

def test_supported_ops_is_frozenset():
    assert isinstance(SUPPORTED_OPS, frozenset)
    assert "matmul" in SUPPORTED_OPS
