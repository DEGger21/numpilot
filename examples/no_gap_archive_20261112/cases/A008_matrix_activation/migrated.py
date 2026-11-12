"""A008_matrix_activation: Matrix product with bounded activation."""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(8)
    return {'x': rng.standard_normal((6, 9), dtype=np.float32), 'w': rng.standard_normal((9, 4), dtype=np.float32)}

def forward(x, w):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(w, np.ndarray):
        w = ap.ndarray.from_numpy(w)
    __numpilot_v_1 = ap.matmul(x, w)
    __numpilot_v_2 = ap.tanh(__numpilot_v_1)
    return {'out': __numpilot_v_2}