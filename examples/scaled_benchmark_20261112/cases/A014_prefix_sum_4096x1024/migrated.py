"""A014_prefix_sum: Cumulative row totals."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(14)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.cumsum(x, axis=1)
    return {'out': __numpilot_v_1}