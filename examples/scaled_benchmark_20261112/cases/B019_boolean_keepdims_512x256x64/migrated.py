"""B019_boolean_keepdims: Boolean reductions keep axes.

Inputs are constructed on the CPU; forward returns named device arrays.
Traps: negative axis, keepdims, bool output.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(219)
    return {
        "x": rng.integers(0, 2, (512, 256, 64)).astype(bool),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.all(x, (-1,), keepdims=True)
    __numpilot_v_2 = ap.any(x, (1,), keepdims=True)
    return {'all': __numpilot_v_1, 'any': __numpilot_v_2}