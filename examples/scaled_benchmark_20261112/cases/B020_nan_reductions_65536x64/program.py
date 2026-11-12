"""B020_nan_reductions: All-NaN rows have explicit additive identities.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: NaN/Inf and all-NaN slices.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(220)
    x = rng.standard_normal((65536, 64), dtype=np.float32)
    x[:, 0] = np.nan
    x[::3, 1] = np.nan
    x[::5, 2] = np.inf
    x[::7, 3] = -np.inf
    x[::1000, :] = np.nan
    return {
        "x": x,
    }


def forward(x):
    return {
        'total': np.nansum(x, axis=1),
        'prefix': np.nancumsum(x, axis=1),
    }
