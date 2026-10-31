"""I004_integer_invert: Bitwise inversion of signed integer patterns.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise int32 output.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    x = np.array([0, 1, -1, 255, 256, -2147483648, 2147483647], dtype=np.int32)
    return {
        'x': x,
    }


def forward(x):
    return {
        'out': np.invert(x),
    }
