"""B017_empty_product: Multiplicative identity over an empty axis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Traps: empty axis and explicit reduction accumulator dtype.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.empty((0, 5), dtype=np.int32),
    }


def forward(x):
    return {
        'out': np.prod(x, axis=0, dtype=np.int64),
    }
