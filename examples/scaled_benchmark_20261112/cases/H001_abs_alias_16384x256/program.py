"""H001_abs_alias: Absolute-value alias on signed samples.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.abs(x) = np.absolute(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(801)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0, 0] = -0.0
    return {
        "x": x,
        "y": x[:, ::-1].copy(),
    }


def forward(x, y):
    left = np.abs(x)
    right = np.abs(y)
    return {
        'left': left,
        'difference': left - right,
    }
