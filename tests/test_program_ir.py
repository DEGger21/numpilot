"""Program IR/DAG 解析的单元测试。"""
from numpilot.program_ir import build_program_ir

SOURCE = """\
import numpy as np


def build_inputs():
    return (np.ones((2, 2), dtype=np.float32),)


def forward(x):
    y = np.add(x, x)
    return np.sqrt(y)
"""


def test_ir_tracks_inputs_calls_and_return():
    ir = build_program_ir(SOURCE)
    labels = [node.label for node in ir.nodes]
    assert labels.count("x") == 1
    assert "np.add" in labels and "np.sqrt" in labels
    assert "return" in labels
    # x -> add, add -> sqrt, sqrt -> return
    assert ir.n_edges == 3


def test_ir_is_empty_without_forward():
    ir = build_program_ir("import numpy as np\ny = np.ones(2)\n")
    assert ir.n_nodes == 0
    assert ir.n_edges == 0
    assert ir.to_dict()["n_nodes"] == 0


def test_ir_serialises_to_plain_dict():
    payload = build_program_ir(SOURCE).to_dict()
    assert set(payload) == {"n_nodes", "n_edges", "nodes", "edges"}
    assert all(set(node) == {"id", "kind", "label", "lineno"} for node in payload["nodes"])
