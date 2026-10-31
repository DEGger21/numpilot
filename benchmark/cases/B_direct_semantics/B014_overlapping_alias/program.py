"""B014_overlapping_alias: In-place addition reads an overlapping reverse view.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: in-place aliasing; x and y share storage with shifted offsets.
"""
import numpy as np


def build_inputs():
    base = np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)
    return {
        'x': base[1:],
        'y': base[:-1],
    }


def forward(x, y):
    x += y
    return {
        'out': x,
        'alias': y,
    }
