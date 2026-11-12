"""A010_boolean_logic: Two boolean masks."""
import numpy as np
import asnumpy as ap

def build_inputs():
    return {'x': np.array([True, False, True, False]), 'y': np.array([False, True, True, False])}

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.logical_and(x, y)
    __numpilot_v_2 = ap.logical_or(x, y)
    return {'both': __numpilot_v_1, 'either': __numpilot_v_2}