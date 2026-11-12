"""D2001_absolute_alias: Absolute-value alias inside a residual score.

Inputs are constructed on the CPU; forward returns named device arrays.
Equivalent: np.abs(residual) -> np.absolute(residual).
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(451)
    return {'x': rng.standard_normal((8, 13), dtype=np.float32), 'y': rng.standard_normal((8, 13), dtype=np.float32)}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.subtract(x, y)
    __numpilot_v_2 = ap.absolute(__numpilot_v_1)
    __numpilot_v_3 = ap.minimum(__numpilot_v_2, ap.full(__numpilot_v_2.shape, 1.0, dtype=__numpilot_v_2.dtype))
    __numpilot_v_4 = ap.square(__numpilot_v_3)
    __numpilot_v_5 = ap.multiply(__numpilot_v_4, ap.full(__numpilot_v_4.shape, 0.5, dtype=__numpilot_v_4.dtype))
    __numpilot_v_6 = ap.subtract(__numpilot_v_2, __numpilot_v_3)
    __numpilot_v_7 = ap.add(__numpilot_v_5, __numpilot_v_6)
    __numpilot_v_8 = ap.clip(__numpilot_v_1, -1.0, 1.0)
    __numpilot_v_9 = ap.mean(__numpilot_v_7, axis=1)
    return {'bounded_residual': __numpilot_v_8, 'loss': __numpilot_v_7, 'row_loss': __numpilot_v_9}