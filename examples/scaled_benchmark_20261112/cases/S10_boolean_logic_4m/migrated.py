"""S10_boolean_logic_4m: Two boolean masks, 4M-element scaled variant."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(10)
    return {"x": rng.random(4_000_000) < 0.5, "y": rng.random(4_000_000) < 0.5}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.logical_and(x, y)
    __numpilot_v_2 = ap.logical_or(x, y)
    return {"both": __numpilot_v_1, "either": __numpilot_v_2}
