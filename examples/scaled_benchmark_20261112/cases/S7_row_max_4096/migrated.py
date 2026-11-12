"""A006_row_max: Maximum per row."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(6)
    return {'x': rng.standard_normal((4096, 4096), dtype=np.float32)}


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.max(x, axis=1)
    return {'out': __numpilot_v_1}