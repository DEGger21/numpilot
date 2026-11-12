"""B009_keepdims: Keep the singleton reduction dimension.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Traps: keepdims and broadcasting.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(209)
    return {'x': rng.standard_normal((5, 7), dtype=np.float32)}

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.mean(x, axis=1, keepdims=True)
    __numpilot_v_2 = ap.subtract(x, __numpilot_v_1)
    return {'mean': __numpilot_v_1, 'centered': __numpilot_v_2}