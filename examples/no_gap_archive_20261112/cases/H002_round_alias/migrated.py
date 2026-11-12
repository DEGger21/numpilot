"""H002_round_alias: Round-to-even decimal alias.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.round(x, decimals=n) = np.around(x, decimals=n).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    y = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float64)
    return {'x': x, 'y': y}

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.around(x)
    __numpilot_v_2 = ap.around(y, decimals=2)
    return {'integer': __numpilot_v_1, 'hundredths': __numpilot_v_2}