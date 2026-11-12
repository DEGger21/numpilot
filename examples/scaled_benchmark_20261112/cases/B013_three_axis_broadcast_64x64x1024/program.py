"""B013_three_axis_broadcast: Independent batch and channel broadcasts.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: broadcasting (2,1,4) with (1,3,1).
"""
import numpy as np


def build_inputs():
    return {
        "x": np.arange(64 * 1024, dtype=np.float32).reshape(64, 1, 1024),
        "y": np.arange(64, dtype=np.float32).reshape(1, 64, 1),
    }


def forward(x, y):
    return {
        'out': np.subtract(x, y),
    }
