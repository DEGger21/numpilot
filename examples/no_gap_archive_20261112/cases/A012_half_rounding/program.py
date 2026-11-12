"""A012_half_rounding: Round finite half precision values."""
import numpy as np


def build_inputs():
    return {"x": np.array([-2.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float16)}


def forward(x):
    return {"nearest": np.rint(x), "floor": np.floor(x)}
