"""B011_nan_inf_extrema: Propagating and ignoring NaN differ.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: NaN/Inf propagation is different for maximum and fmax.
"""
import numpy as np


def build_inputs():
    return {
        "x": np.resize(np.array([np.nan, np.inf, -np.inf, 1], dtype=np.float32), (4_000_000,)),
        "y": np.resize(np.array([2, 3, -4, np.nan], dtype=np.float32), (4_000_000,)),
    }


def forward(x, y):
    return {
        'maximum': np.maximum(x, y),
        'fmax': np.fmax(x, y),
    }
