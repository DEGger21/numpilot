"""H007_asinh_alias: Inverse hyperbolic sine signed compression.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.asinh(x) = np.arcsinh(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    x = np.linspace(-30, 30, 4_194_304, dtype=np.float32).reshape(2048, 2048)
    return {
        "x": x,
        "gain": np.linspace(0.1, 1.0, 2048, dtype=np.float32),
    }


def forward(x, gain):
    out = np.asinh(x * gain)
    return {
        'out': out,
        'odd_check': np.asinh(-x * gain) + out,
    }
