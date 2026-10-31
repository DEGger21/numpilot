"""F003_subtract_casting: Unsafe casting is explicit.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [3, -3].
Pinned 1929a3ba wrapper subtract rejects casting/out: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([3.75, -2.75], dtype=np.float32),
        'buffer': np.zeros(2, dtype=np.int32),
    }


def forward(x, buffer):
    np.subtract(x, 0.5, out=buffer, casting="unsafe")
    return {
        'out': buffer,
        'twice': np.add(buffer, buffer),
    }
