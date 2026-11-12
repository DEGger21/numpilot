"""B016_reverse_strides: Negative strides through transcendental operations.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: negative stride.
"""
import numpy as np


def build_inputs():
    base = np.linspace(-1, 1, 4_000_000, dtype=np.float32)
    return {
        "x": base[::-1],
    }


def forward(x):
    return {
        'out': np.sin(x),
        'sign': np.signbit(x),
    }
