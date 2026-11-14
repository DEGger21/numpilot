"""H009_variance: Population variance on two tensor axes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
var(x, axis=a, keepdims=True) = mean(square(x-mean(x,a,True)),a,True).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(809)
    return {
        'x': rng.standard_normal((3, 5, 7), dtype=np.float32),
    }


def forward(x):
    rows = np.var(x, axis=2, keepdims=True)
    channels = np.var(x, axis=1, keepdims=True)
    return {
        'rows': rows,
        'channels': channels,
    }
