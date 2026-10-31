"""H014_nan_minimum: NaN-ignoring minima with a finite witness per slice.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
nanmin(x,axis=a) = negative(nanmax(negative(x),axis=a)); absent in pinned exports.
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    x = np.array([[np.nan, 3, -2], [4, np.nan, 1], [-1, 6, np.nan]], dtype=np.float32)
    return {
        'x': x,
    }


def forward(x):
    rows = np.nanmin(x, axis=1)
    columns = np.nanmin(x, axis=0)
    return {
        'rows': rows,
        'columns': columns,
    }
