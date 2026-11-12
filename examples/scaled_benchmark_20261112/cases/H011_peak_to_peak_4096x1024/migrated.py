"""H011_peak_to_peak: Peak-to-peak extent per row and column.

Inputs are constructed on the CPU; forward returns named device arrays.
Composition reference; NumPy >= 2.0.
ptp(x,axis=a) = subtract(max(x,axis=a),min(x,axis=a)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(811)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    row_max = ap.max(x, axis=1, keepdims=False)
    row_min = ap.min(x, axis=1, keepdims=False)
    row = ap.subtract(row_max, row_min)
    col_max = ap.max(x, axis=0, keepdims=False)
    col_min = ap.min(x, axis=0, keepdims=False)
    column = ap.subtract(col_max, col_min)
    return {'row': row, 'column': column}