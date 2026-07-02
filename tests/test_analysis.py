from numpilot.agents.analysis import AnalysisAgent

SRC_OK = """
import numpy as np
def f(x, w):
    h = np.matmul(x, w)
    return np.maximum(h, 0)
"""

SRC_GAP = """
import numpy as np
def g(x):
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
    src = "import numpy as np\nx = np.zeros((100, 200))\n"
    r = AnalysisAgent().analyze(src)
    z = [op for op in r.op_uses if op.name == "zeros"][0]
    assert z.est_elements == 20000
