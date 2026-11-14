"""B009_keepdims: Keep the singleton reduction dimension.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Traps: keepdims and broadcasting.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(209)
    return {
        'x': rng.standard_normal((5, 7), dtype=np.float32),
    }


def forward(x):
    mean = np.mean(x, axis=1, keepdims=True)
    return {
        'mean': mean,
        'centered': np.subtract(x, mean),
    }
