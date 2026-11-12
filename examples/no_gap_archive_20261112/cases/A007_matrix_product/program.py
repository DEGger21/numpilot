"""A007_matrix_product: Rectangular matrix product."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(7)
    return {
        "x": rng.standard_normal((8, 12), dtype=np.float32),
        "w": rng.standard_normal((12, 5), dtype=np.float32),
    }


def forward(x, w):
    return {"out": np.matmul(x, w)}
