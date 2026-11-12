import numpy as np
import asnumpy as ap

def build_inputs():
    return {'x': np.array([1, 2, 3], dtype=np.float32), 'y': np.array([1e-10, 2e-10, 3e-10], dtype=np.float64)}

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.add(x, y)
    __numpilot_v_2 = ap.subtract(__numpilot_v_1, x)
    return {'out': __numpilot_v_2}