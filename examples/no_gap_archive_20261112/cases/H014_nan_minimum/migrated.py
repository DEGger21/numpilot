"""H014_nan_minimum: NaN-ignoring minima with a finite witness per slice.

Inputs are constructed on the CPU; forward returns named NPU arrays.
Migrated to AsNumpy (NPU) runtime.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    x = np.array([[np.nan, 3, -2], [4, np.nan, 1], [-1, 6, np.nan]], dtype=np.float32)
    return {'x': x}

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.nanmin(x, axis=1)
    __numpilot_v_2 = ap.nanmin(x, axis=0)
    return {'rows': __numpilot_v_1, 'columns': __numpilot_v_2}