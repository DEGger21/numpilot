"""H011_peak_to_peak: Peak-to-peak extent per row and column.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
ptp(x,axis=a) = subtract(max(x,axis=a),min(x,axis=a)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(811)
    return {
        'x': rng.standard_normal((6, 13), dtype=np.float32),
    }


def forward(x):
    row = np.ptp(x, axis=1)
    column = np.ptp(x, axis=0)
    return {
        'row': row,
        'column': column,
    }
