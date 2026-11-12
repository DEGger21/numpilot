"""A004_row_sum: Sum each row."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(4)
    return {
        "x": rng.standard_normal((8, 16), dtype=np.float32),
    }


def forward(x):
    return {"out": np.sum(x, axis=1)}
