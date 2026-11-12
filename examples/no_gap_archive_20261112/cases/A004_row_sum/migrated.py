"""A004_row_sum: Sum each row."""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(4)
    return {'x': rng.standard_normal((8, 16), dtype=np.float32)}

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.sum(x, axis=1)
    return {'out': __numpilot_v_1}