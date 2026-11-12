"""A002_bounded_product: Clip a pointwise product."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(2)
    return {'x': rng.standard_normal((64,), dtype=np.float32), 'y': rng.standard_normal((64,), dtype=np.float32)}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.multiply(x, y)
    __numpilot_v_2 = ap.clip(__numpilot_v_1, -1.0, 1.0)
    return {'out': __numpilot_v_2}