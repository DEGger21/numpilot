"""H014_nan_minimum: NaN-ignoring minima with a finite witness per slice.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
nanmin(x,axis=a) = negative(nanmax(negative(x),axis=a)); absent in pinned exports.
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(814)
    x = rng.standard_normal((4096, 1024), dtype=np.float32)
    x[::3, ::7] = np.nan
    x[::5, 1] = np.nan
    return {
        "x": x,
    }


def forward(x):
    rows = np.nanmin(x, axis=1)
    columns = np.nanmin(x, axis=0)
    return {
        'rows': rows,
        'columns': columns,
    }
