"""A006_row_max: Maximum per row."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(6)
    return {
        "x": rng.standard_normal((4096, 4096), dtype=np.float32),
    }


def forward(x):
    return {"out": np.max(x, axis=1)}
