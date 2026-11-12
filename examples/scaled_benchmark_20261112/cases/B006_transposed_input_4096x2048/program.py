"""B006_transposed_input: Noncontiguous transposed matrix.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: noncontiguous input; x is F-contiguous, not C-contiguous.
"""
import numpy as np


def build_inputs():
    base = np.arange(8_388_608, dtype=np.float32).reshape(4096, 2048)
    return {
        "x": base.T,
        "y": np.full((2048, 4096), 0.5, dtype=np.float32),
    }


def forward(x, y):
    return {
        'out': np.multiply(x, y),
    }
