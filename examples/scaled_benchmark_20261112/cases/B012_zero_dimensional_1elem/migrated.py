import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        "x": np.array(3.5, dtype=np.float32),
        "y": np.array(-1.25, dtype=np.float32),
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.add(x, y)
    return {'out': __numpilot_v_1}