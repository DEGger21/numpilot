"""H004_asin_alias: Inverse sine for a symmetric angular grid.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.asin(x) = np.arcsin(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    x = np.linspace(-0.9, 0.9, 25, dtype=np.float64)
    return {
        'x': x,
        'offset': np.full(25, 0.025, dtype=np.float64),
    }


def forward(x, offset):
    base = np.asin(x)
    perturbed = np.asin(x + offset)
    return {
        'base': base,
        'sensitivity': (perturbed - base) / offset,
    }
