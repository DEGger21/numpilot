"""J009_vectorize_callback: Python callback supplied to vectorize.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([-2, -1, 0, 1, 2], dtype=np.float64),
    }


def piecewise(value):
    if value < 0:
        return -value * 0.5
    return value * value


def forward(x):
    callback = np.vectorize(piecewise, otypes=[np.float64])
    return {
        'out': callback(x),
    }
