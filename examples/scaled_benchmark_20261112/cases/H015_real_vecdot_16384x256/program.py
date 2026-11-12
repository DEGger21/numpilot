"""H015_real_vecdot: Vector dot products over explicit axes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
For real float32: linalg.vecdot(x,y,axis=a) = sum(multiply(x,y),axis=a).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(815)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(x, y):
    rows = np.linalg.vecdot(x, y, axis=1)
    columns = np.linalg.vecdot(x, y, axis=0)
    return {
        'rows': rows,
        'columns': columns,
    }
