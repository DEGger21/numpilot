"""A011_integer_modulo: Signed integer remainders."""
import numpy as np


def build_inputs():
    return {"x": np.array([-9, -4, 0, 7, 11], dtype=np.int32),
            "y": np.array([4, 3, 2, 5, 6], dtype=np.int32)}


def forward(x, y):
    return {"out": np.remainder(x, y)}
