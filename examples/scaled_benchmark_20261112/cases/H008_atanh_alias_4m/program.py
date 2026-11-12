"""H008_atanh_alias: Inverse hyperbolic tangent inside the open domain.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.atanh(x) = np.arctanh(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    x = np.linspace(-0.95, 0.95, 4_000_000, dtype=np.float64)
    return {
        "x": x,
        "shrink": np.full(4_000_000, 0.5, dtype=np.float64),
    }


def forward(x, shrink):
    full = np.atanh(x)
    half = np.atanh(x * shrink)
    return {
        'full': full,
        'half': half,
        'difference': full - half,
    }
