"""B005_mixed_float_precision: Float64 input preserves small increments.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: float32 plus float64 promotion.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(205)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
        "y": rng.standard_normal((4_000_000,), dtype=np.float64) * 1e-10,
    }


def forward(x, y):
    sum_xy = np.add(x, y)
    return {
        'out': np.subtract(sum_xy, x),
    }
