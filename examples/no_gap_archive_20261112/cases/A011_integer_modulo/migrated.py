"""A011_integer_modulo: Signed integer remainders."""
import numpy as np
import asnumpy as ap

def build_inputs():
    return {'x': np.array([-9, -4, 0, 7, 11], dtype=np.int32), 'y': np.array([4, 3, 2, 5, 6], dtype=np.int32)}

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.remainder(x, y)
    return {'out': __numpilot_v_1}