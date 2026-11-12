import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(205)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
        "y": rng.standard_normal((4_000_000,), dtype=np.float64) * 1e-10,
    }

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.add(x, y)
    __numpilot_v_2 = ap.subtract(__numpilot_v_1, x)
    return {'out': __numpilot_v_2}