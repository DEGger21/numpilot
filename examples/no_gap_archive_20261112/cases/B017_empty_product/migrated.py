"""B017_empty_product: Multiplicative identity over an empty axis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Traps: empty axis and explicit reduction accumulator dtype.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {'x': np.empty((0, 5), dtype=np.int32)}


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.prod(x, axis=0, dtype=np.int64)
    return {'out': __numpilot_v_1}