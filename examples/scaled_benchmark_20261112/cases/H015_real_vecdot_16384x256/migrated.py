"""H015_real_vecdot: Vector dot products over explicit axes.

Inputs are constructed on the CPU; forward returns named arrays.
Composition reference; NumPy >= 2.0.
For real float32: linalg.vecdot(x,y,axis=a) = sum(multiply(x,y),axis=a).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(815)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.sum(ap.multiply(x, y), axis=1)
    __numpilot_v_2 = ap.sum(ap.multiply(x, y), axis=0)
    return {'rows': __numpilot_v_1, 'columns': __numpilot_v_2}