"""B020_nan_reductions: All-NaN rows have explicit additive identities.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: NaN/Inf and all-NaN slices.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([[np.nan, np.nan], [1, np.nan], [np.inf, 2]], dtype=np.float32),
    }


def forward(x):
    return {
        'total': np.nansum(x, axis=1),
        'prefix': np.nancumsum(x, axis=1),
    }
