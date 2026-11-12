"""B008_negative_axis: Reduction along the last tensor dimension.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: negative axis on a rank-three tensor.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(208)
    return {
        "x": rng.standard_normal((64, 256, 256), dtype=np.float32),
    }


def forward(x):
    return {
        'out': np.sum(x, axis=-1),
    }
