"""B010_empty_sum: Additive identity over an empty axis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: empty array with a nonempty output.
"""
import numpy as np


def build_inputs():
    return {
        "x": np.empty((16384, 0, 256), dtype=np.float32),
    }


def forward(x):
    return {
        'out': np.sum(x, axis=1),
        'product': np.prod(x, axis=1),
    }
