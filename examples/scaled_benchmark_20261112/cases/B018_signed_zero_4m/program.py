"""B018_signed_zero: Signs of IEEE zero survive copy-sign.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: IEEE signed zero, not distinguishable by ordinary equality.
"""
import numpy as np


def build_inputs():
    return {
        "x": np.resize(np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32), (4_000_000,)),
        "y": np.ones(4_000_000, dtype=np.float32),
    }


def forward(x, y):
    return {
        'sign': np.signbit(x),
        'out': np.copysign(y, x),
    }
