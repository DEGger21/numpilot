"""B020_nan_reductions: All-NaN rows have explicit additive identities.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: NaN/Inf and all-NaN slices.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    return {'x': np.array([[np.nan, np.nan], [1, np.nan], [np.inf, 2]], dtype=np.float32)}

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.nansum(x, axis=1)
    __numpilot_v_2 = ap.nancumsum(x, axis=1)
    return {'total': __numpilot_v_1, 'prefix': __numpilot_v_2}