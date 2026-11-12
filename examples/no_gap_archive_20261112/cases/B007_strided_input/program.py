"""B007_strided_input: Step slices skip physical storage.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: noncontiguous input with a 12-byte stride.
"""
import numpy as np


def build_inputs():
    base = np.linspace(-3, 3, 64, dtype=np.float32)
    return {
        'x': base[1::3],
    }


def forward(x):
    return {
        'out': np.square(x),
    }
