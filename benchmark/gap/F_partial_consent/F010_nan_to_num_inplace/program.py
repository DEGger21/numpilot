"""F010_nan_to_num_inplace: NaN replacement mutates the input.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [3,4,-4,2].
Pinned 1929a3ba wrapper nan_to_num rejects copy: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([np.nan, np.inf, -np.inf, 2], dtype=np.float32),
    }


def forward(x):
    out = np.nan_to_num(x, copy=False, nan=3.0, posinf=4.0, neginf=-4.0)
    return {
        'out': out,
        'input_after': x,
        'squared': np.square(x),
    }
