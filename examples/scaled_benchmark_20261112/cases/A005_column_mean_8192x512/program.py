"""A005_column_mean: Mean each feature column."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(5)
    return {
        "x": rng.standard_normal((8192, 512), dtype=np.float32),
    }


def forward(x):
    return {"out": np.mean(x, axis=0)}
