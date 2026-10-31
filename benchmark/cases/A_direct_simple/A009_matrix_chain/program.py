"""A009_matrix_chain: Three matrix chain."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(9)
    return {
        "x": rng.standard_normal((5, 7), dtype=np.float32),
        "w": rng.standard_normal((7, 6), dtype=np.float32),
        "v": rng.standard_normal((6, 3), dtype=np.float32),
    }

def forward(x, w, v):
    hidden = np.matmul(x, w)
    return {"out": np.matmul(hidden, v)}
