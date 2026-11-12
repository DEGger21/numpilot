"""B007_strided_input: Step slices skip physical storage.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: noncontiguous input with a 12-byte stride.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    base = np.linspace(-3, 3, 4_000_000, dtype=np.float32)
    return {
        "x": base[1::3],
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.square(x)
    return {'out': __numpilot_v_1}