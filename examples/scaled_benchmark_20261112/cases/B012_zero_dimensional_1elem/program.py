"""B012_zero_dimensional: Zero-dimensional array inputs and output.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: scalar ufunc results must be wrapped as shape ().
"""
import numpy as np


def build_inputs():
    return {
        "x": np.array(3.5, dtype=np.float32),
        "y": np.array(-1.25, dtype=np.float32),
    }


def forward(x, y):
    value = np.add(x, y)
    return {
        'out': np.asarray(value),
    }
