"""I011_nextafter_promotion: Mixed precision adjacent representable numbers.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I3: float32/float64 promotion and IEEE boundaries.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    x = np.array([0, -0.0, 1, -1, np.inf, -np.inf, np.nan], dtype=np.float32)
    y = np.array([1, -1, 2, -2, 0, 0, 1], dtype=np.float64)
    return {
        'x': x,
        'y': y,
    }


def forward(x, y):
    return {
        'out': np.nextafter(x, y),
    }
