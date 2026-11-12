"""A007_matrix_product: Rectangular matrix product."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(7)
    return {'x': rng.standard_normal((8, 12), dtype=np.float32), 'w': rng.standard_normal((12, 5), dtype=np.float32)}


def forward(x, w):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(w, np.ndarray):
        w = ap.ndarray.from_numpy(w)
    __numpilot_v_1 = ap.matmul(x, w)
    return {'out': __numpilot_v_1}