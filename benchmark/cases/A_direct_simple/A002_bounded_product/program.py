"""A002_bounded_product: Clip a pointwise product."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(2)
    return {
        "x": rng.standard_normal((64,), dtype=np.float32),
        "y": rng.standard_normal((64,), dtype=np.float32),
    }


def forward(x, y):
    product = np.multiply(x, y)
    return {"out": np.clip(product, -1.0, 1.0)}
