"""H007_asinh_alias: Inverse hyperbolic sine signed compression.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.asinh(x) = np.arcsinh(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    x = np.linspace(-30, 30, 45, dtype=np.float32).reshape(5, 9)
    return {'x': x, 'gain': np.linspace(0.1, 1.0, 9, dtype=np.float32)}

def forward(x, gain):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(gain, np.ndarray):
        gain = ap.ndarray.from_numpy(gain)
    __numpilot_v_1 = ap.multiply(x, gain)
    __numpilot_v_2 = ap.arcsinh(__numpilot_v_1)
    __numpilot_v_3 = ap.negative(x)
    __numpilot_v_4 = ap.multiply(__numpilot_v_3, gain)
    __numpilot_v_5 = ap.arcsinh(__numpilot_v_4)
    __numpilot_v_6 = ap.add(__numpilot_v_5, __numpilot_v_2)
    return {'out': __numpilot_v_2, 'odd_check': __numpilot_v_6}