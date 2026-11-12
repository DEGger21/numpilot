"""A009_matrix_chain: Three matrix chain."""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(9)
    return {'x': rng.standard_normal((5, 7), dtype=np.float32), 'w': rng.standard_normal((7, 6), dtype=np.float32), 'v': rng.standard_normal((6, 3), dtype=np.float32)}

def forward(x, w, v):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(w, np.ndarray):
        w = ap.ndarray.from_numpy(w)
    if isinstance(v, np.ndarray):
        v = ap.ndarray.from_numpy(v)
    __numpilot_v_1 = ap.matmul(x, w)
    __numpilot_v_2 = ap.matmul(__numpilot_v_1, v)
    return {'out': __numpilot_v_2}