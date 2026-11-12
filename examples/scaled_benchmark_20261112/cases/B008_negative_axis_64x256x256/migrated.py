"""B008_negative_axis: Reduction along the last tensor dimension.

Inputs are constructed on the CPU; forward returns named device arrays.
Trap: negative axis on a rank-three tensor.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(208)
    return {
        "x": rng.standard_normal((64, 256, 256), dtype=np.float32),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.sum(x, axis=-1)
    return {'out': __numpilot_v_1}