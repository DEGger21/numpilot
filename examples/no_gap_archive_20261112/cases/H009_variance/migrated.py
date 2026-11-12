"""H009_variance: Population variance on two tensor axes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
var(x, axis=a, keepdims=True) = mean(square(x-mean(x,a,True)),a,True).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(809)
    return {'x': rng.standard_normal((3, 5, 7), dtype=np.float32)}


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    # rows: variance along axis 2
    mean_rows = ap.mean(x, axis=2, keepdims=True)
    centered_rows = ap.subtract(x, mean_rows)
    squared_rows = ap.square(centered_rows)
    rows = ap.mean(squared_rows, axis=2, keepdims=True)
    # channels: variance along axis 1
    mean_channels = ap.mean(x, axis=1, keepdims=True)
    centered_channels = ap.subtract(x, mean_channels)
    squared_channels = ap.square(centered_channels)
    channels = ap.mean(squared_channels, axis=1, keepdims=True)
    return {'rows': rows, 'channels': channels}