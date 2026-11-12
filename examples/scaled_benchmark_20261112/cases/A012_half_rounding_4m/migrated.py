"""A012_half_rounding: Round finite half precision values."""
import numpy as np
import asnumpy as ap


def build_inputs():
    pattern = np.array([-2.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float16)
    return {
        "x": np.tile(pattern, 800_000),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.rint(x)
    __numpilot_v_2 = ap.floor(x)
    return {'nearest': __numpilot_v_1, 'floor': __numpilot_v_2}