"""F001_add_out: add writes into the supplied buffer.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [5, 7, 9].
Pinned 1929a3ba wrapper add rejects out: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([1, 2, 3], dtype=np.float32),
        'y': np.array([4, 5, 6], dtype=np.float32),
        'buffer': np.full(3, -9, dtype=np.float32),
    }


def forward(x, y, buffer):
    np.add(x, y, out=buffer)
    return {
        'out': buffer,
        'norm': np.square(buffer),
    }
