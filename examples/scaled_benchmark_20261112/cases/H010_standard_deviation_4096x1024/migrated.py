"""H010_standard_deviation: Population standard deviations with explicit axes.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Composition reference; NumPy >= 2.0.
std(x,axis=a) = sqrt(mean(square(x-mean(x,a,True)),axis=a)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(810)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    row_mean_keep = ap.mean(x, axis=1, keepdims=True)
    row_centered = ap.subtract(x, row_mean_keep)
    row_squared = ap.square(row_centered)
    row_var = ap.mean(row_squared, axis=1, keepdims=False)
    row_std = ap.sqrt(row_var)
    col_mean_keep = ap.mean(x, axis=0, keepdims=True)
    col_centered = ap.subtract(x, col_mean_keep)
    col_squared = ap.square(col_centered)
    col_var = ap.mean(col_squared, axis=0, keepdims=False)
    col_std = ap.sqrt(col_var)
    return {'row': row_std, 'column': col_std}