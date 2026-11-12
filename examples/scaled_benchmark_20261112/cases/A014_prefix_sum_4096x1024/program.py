"""A014_prefix_sum: Cumulative row totals."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(14)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }


def forward(x):
    return {"out": np.cumsum(x, axis=1)}
