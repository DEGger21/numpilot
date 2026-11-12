"""B013_three_axis_broadcast: Independent batch and channel broadcasts.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: broadcasting (2,1,4) with (1,3,1).
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    return {'x': np.arange(8, dtype=np.float32).reshape(2, 1, 4), 'y': np.arange(3, dtype=np.float32).reshape(1, 3, 1)}

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.subtract(x, y)
    return {'out': __numpilot_v_1}