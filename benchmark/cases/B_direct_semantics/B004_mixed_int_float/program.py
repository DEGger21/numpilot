"""B004_mixed_int_float: Promotion of int32 and float32.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: dtype promotion; 16777217 detects premature float32 conversion.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([16777217, -3, 2], dtype=np.int32),
        'y': np.array([0.5, 0.25, -0.5], dtype=np.float32),
    }


def forward(x, y):
    return {
        'out': np.add(x, y),
    }
