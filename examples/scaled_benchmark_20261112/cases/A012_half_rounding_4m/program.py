"""A012_half_rounding: Round finite half precision values."""
import numpy as np


def build_inputs():
    pattern = np.array([-2.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float16)
    return {
        "x": np.tile(pattern, 800_000),
    }


def forward(x):
    return {"nearest": np.rint(x), "floor": np.floor(x)}
