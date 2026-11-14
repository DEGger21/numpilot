"""I005_population_count: Population count of unsigned words.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise uint8 output; NumPy >= 2.0.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    x = np.array([0, 1, 3, 7, 255, 256, 65535, 4294967295], dtype=np.uint32)
    return {
        'x': x,
    }


def forward(x):
    return {
        'out': np.bitwise_count(x),
    }
