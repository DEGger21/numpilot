"""B003_unsafe_cast: Explicitly permitted float to integer conversion.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: casting=unsafe. Pinned add has no casting/out parameters.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([1.75, -2.75, 3.25], dtype=np.float32),
        'y': np.array([0.5, 0.5, -0.5], dtype=np.float32),
        'buffer': np.zeros(3, dtype=np.int32),
    }


def forward(x, y, buffer):
    np.add(x, y, out=buffer, casting="unsafe")
    return {
        'out': buffer,
    }
