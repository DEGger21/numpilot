"""B013_three_axis_broadcast: Independent batch and channel broadcasts.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: broadcasting (2,1,4) with (1,3,1).
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(8, dtype=np.float32).reshape(2, 1, 4),
        'y': np.arange(3, dtype=np.float32).reshape(1, 3, 1),
    }


def forward(x, y):
    return {
        'out': np.subtract(x, y),
    }
