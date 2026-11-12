"""A004_row_sum: Sum each row."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(4)
    return {
        "x": rng.standard_normal((4096, 4096), dtype=np.float32),
    }


def forward(x):
    return {"out": np.sum(x, axis=1)}
