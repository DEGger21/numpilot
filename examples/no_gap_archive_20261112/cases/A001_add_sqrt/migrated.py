"""A001_add_sqrt: Magnitude of a vector sum."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(1)
    return {'x': rng.standard_normal((128,), dtype=np.float32), 'y': rng.standard_normal((128,), dtype=np.float32)}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.add(x, y)
    __numpilot_v_2 = ap.square(__numpilot_v_1)
    __numpilot_v_3 = ap.sqrt(__numpilot_v_2)
    return {'out': __numpilot_v_3}