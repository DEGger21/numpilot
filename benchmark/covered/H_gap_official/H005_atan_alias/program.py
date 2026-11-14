"""H005_atan_alias: Inverse tangent compression at three scales.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.atan(x) = np.arctan(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    x = np.array([-100, -10, -1, 0, 1, 10, 100], dtype=np.float32)
    return {
        'x': x,
        'scale': np.full(7, 0.25, dtype=np.float32),
    }


def forward(x, scale):
    fine = np.atan(x * scale)
    coarse = np.atan(x)
    return {
        'fine': fine,
        'coarse': coarse,
        'delta': coarse - fine,
    }
