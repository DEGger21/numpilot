"""A005_column_mean: Mean each feature column."""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(5)
    return {'x': rng.standard_normal((12, 6), dtype=np.float32)}

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.mean(x, axis=0)
    return {'out': __numpilot_v_1}