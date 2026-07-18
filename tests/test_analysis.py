from numpilot.agents.analysis import AnalysisAgent

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
