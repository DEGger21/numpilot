"""A008_matrix_activation: Matrix product with bounded activation."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(8)
    return {
        "x": rng.standard_normal((6, 9), dtype=np.float32),
        "w": rng.standard_normal((9, 4), dtype=np.float32),
    }


def forward(x, w):
    z = np.matmul(x, w)
    return {"out": np.tanh(z)}
