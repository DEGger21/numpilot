"""A015_row_product: Products of bounded positive entries."""
import numpy as np
import asnumpy as ap

def build_inputs():
    return {
        "x": np.linspace(0.8, 1.2, 4_194_304, dtype=np.float32).reshape(65536, 64),
    }

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.prod(x, axis=1)
    return {'out': __numpilot_v_1}