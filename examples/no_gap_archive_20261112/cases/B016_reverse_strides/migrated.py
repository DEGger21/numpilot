"""B016_reverse_strides: Negative strides through transcendental operations.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: negative stride.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    base = np.linspace(-1, 1, 19, dtype=np.float32)
    return {'x': base[::-1]}


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.sin(x)
    __numpilot_v_2 = ap.signbit(x)
    return {'out': __numpilot_v_1, 'sign': __numpilot_v_2}