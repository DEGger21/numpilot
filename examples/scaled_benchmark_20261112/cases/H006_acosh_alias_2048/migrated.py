"""H006_acosh_alias: Inverse hyperbolic cosine on its real domain.

Inputs are constructed on the CPU; forward returns named device arrays.
Alias reference; NumPy >= 2.0.
np.acosh(x) = np.arccosh(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    distance = np.linspace(0, 8, 4_194_304, dtype=np.float32).reshape(2048, 2048)
    return {
        "x": 1 + distance,
        "y": 1 + distance * distance,
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.arccosh(x)
    __numpilot_v_2 = ap.arccosh(y)
    __numpilot_v_3 = ap.add(__numpilot_v_1, __numpilot_v_2)
    return {'out': __numpilot_v_3, 'first': __numpilot_v_1}