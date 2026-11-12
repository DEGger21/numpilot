"""H003_acos_alias: Inverse cosine with endpoint values.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.acos(x) = np.arccos(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    x = np.linspace(-1, 1, 4_000_000, dtype=np.float32).reshape(2000, 2000)
    return {
        "x": x,
        "y": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    }


def forward(x, y):
    angles = np.acos(x)
    endpoints = np.acos(y)
    return {
        'angles': angles,
        'endpoints': endpoints,
    }
