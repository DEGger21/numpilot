"""B005_mixed_float_precision: Float64 input preserves small increments.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: float32 plus float64 promotion.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([1, 2, 3], dtype=np.float32),
        'y': np.array([1e-10, 2e-10, 3e-10], dtype=np.float64),
    }


def forward(x, y):
    sum_xy = np.add(x, y)
    return {
        'out': np.subtract(sum_xy, x),
    }
